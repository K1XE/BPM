"""TRL GOLD/ULD GOLD ORACLE  --  standalone, CPU-only, self-contained reference.

Purpose
-------
This file is the *canonical numerical reference* for the GOLD/ULD distillation
losses used by the paper's baselines.  The paper compares BPM against GOLD/ULD
baselines implemented under ``slime_plugins/baselines/gold``.  The
authoritative implementation those baselines claim to reproduce lives in TRL:
``trl/experimental/gold/gold_trainer.py`` (+ ``gold_config.py``,
+ ``trl/experimental/utils.py``).

To make the reference runnable WITHOUT importing the full ``trl`` package (which
drags in accelerate / datasets / vllm / liger and refuses to import on a CPU box),
the load-bearing loss functions are COPIED VERBATIM here, each with a header
citing the exact source file:line it was lifted from.  The math is unchanged; only
dead plumbing (deepspeed/zero3 gather, OOM-retry telemetry, timing prints) is
dropped and noted.

The second half of the file (``IMPLEMENTATION CORE MIRROR``) copies the
pure-torch per-group math from ``gold_kernels.py`` so the two can be
cross-checked on identical synthetic inputs at TP=1/CP=1.

Run:  python3 trl_gold_oracle.py

===========================================================================
TRL PROVENANCE  (pinned 2026-07-03)   -- DO NOT edit the reference copy
===========================================================================
Source tree : https://github.com/huggingface/trl.git
  git rev-parse HEAD : d71a6ca90806bd6dc62cdea5cc26b69f4c376403
  git describe        : v1.7.0-3-gd71a6ca9
  HEAD commit         : "Remove RUNNING_NAME from KTO (#6179)"  (2026-06-26, upstream main)
  WORKING TREE IS MODIFIED (uncommitted).  git status --short:
       M trl/experimental/gold/gold_config.py
       M trl/experimental/gold/gold_trainer.py     (+1257 / -137 vs HEAD)
       M trl/generation/vllm_client.py
       M trl/generation/vllm_generation.py
       M trl/scripts/vllm_serve.py
      ?? scripts/gold_glm47_qwen35_dapo_aime.py
      ?? scripts/gold_glm47_qwen35_dapo_aime.sh
      ?? trl/experimental/gold/gold_trainer.py.bak.20260628_173805
  The load-bearing loss code copied below is lifted from the MODIFIED WORKING-TREE
  gold_trainer.py (NOT bare HEAD).  d71a6ca IS a real upstream commit, so the tree
  = upstream d71a6ca + the uncommitted +1257/-137 gold_trainer diff.

UPSTREAM COMPARISON  (huggingface/trl, fetched via proxy raw.githubusercontent.com)
  Compared the working tree against (a) gold_trainer.py @ the pinned base commit
  d71a6ca, and (b) gold_trainer.py @ current upstream main.

  Q: Does upstream contain the hybrid matched/unmatched path
     (uld_use_hybrid_loss / hybrid weights)?
  A: YES -- it is UPSTREAM, not a local invention.  In BOTH base d71a6ca AND main:
       - config flags  uld_use_hybrid_loss, uld_hybrid_matched_weight,
         uld_hybrid_unmatched_weight  are present (1 occurrence each).
       - _compute_hybrid_uld_loss (prob-space: generalized_jsd_loss over matched
         columns + sorted-L1 over unmatched columns, then weighted sum) is present.
     So the gold-matched / gold-unmatched arms exercise an UPSTREAM algorithm.

  LOCAL-FORK-ONLY additions (in the working tree; absent in BOTH base AND main):
       - _compute_extended_hybrid_uld_loss_streaming          (logits-streaming hybrid)
       - _compute_chunked_hybrid_loss_components
       - _compute_chunked_jsd_loss_for_matched_tokens
       - _compute_sorted_l1_unmatched_loss
       - _compute_chunked_unmatched_loss_with_oom_retry
       - *_from_hidden_states streaming variants, timing/debug helpers,
         think-open-prompt restore, passrate metrics, zero3 lm-head gather ctx
         (plumbing -- not loss math).
     When (use_extended_uld AND use_hybrid_loss), LOCAL _compute_distillation_loss
     routes to _compute_extended_hybrid_uld_loss_streaming INSTEAD of the base
     prob-space _compute_hybrid_uld_loss.  This streaming path is a memory-chunked
     RE-IMPLEMENTATION of the SAME objective (chunked-JSD over matched +
     sorted-L1 over unmatched).  THIS ORACLE COPIES THE STREAMING VARIANT VERBATIM,
     so the harness tests exactly what the fork runs.

  FUNCTIONAL DELTAS local-vs-upstream, per paper arm:
       * uld-trl        : NONE.  The non-hybrid branch of _compute_distillation_loss,
                          _align_by_byte_offsets, _merge_probabilities_with_alignment_groups,
                          and generalized_jsd_loss are BYTE-IDENTICAL between base
                          d71a6ca and the working tree (verified by textual diff;
                          only a comment line differs in the merge helper).
       * gold-matched   } streaming hybrid impl (LOCAL-fork-only, listed above)
       * gold-unmatched } replaces the prob-space hybrid; SAME math objective.

  UPSTREAM-MAIN-ONLY (post-base; in NEITHER base d71a6ca NOR our fork -> does NOT
  affect our arms): uld_token_merge_strategy in {observed, bayesian}.  main adds a
  "bayesian" merge (base distribution at the LAST group position, logit window
  shifted one position earlier) as an option.  Our fork predates this and always
  uses the "observed" rule (marginal at the FIRST position x scalar conditional
  probabilities) -- which is exactly base d71a6ca behavior and what ours's
  --gold-uld-token-merge-strategy observed mirrors.
===========================================================================
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# SECTION 0.  minimal `pad`  (verbatim math from trl/trainer/utils.py::pad)
#   Cited: trl/trainer/utils.py  def pad(...)
#   Only the right/left first-dim padding path is reproduced (all this oracle uses).
# =============================================================================
def pad(tensors, padding_value=0, padding_side="right", pad_to_multiple_of=None):
    output_shape = np.max([t.shape for t in tensors], 0).tolist()
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder
    output = torch.full(
        (len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device
    )
    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        output[i][seq_slice] = t
    return output


# =============================================================================
# SECTION 1.  Byte-offset utilities
#   VERBATIM from trl/experimental/utils.py
#     is_byte_level_tokenizer      : utils.py:142-145
#     piece_byte_len               : utils.py:148-153
#     pad_byte_offsets             : utils.py:131-139
#     _bytes_to_unicode            : utils.py:156-166
#     _byte_level_piece_len        : utils.py:172-180
#     _split_repeated_byte_offsets : utils.py:183-202
#     _normalize_byte_offsets      : utils.py:205-237
#     encode_with_byte_offsets     : utils.py:240-259
#   The ours mirror of these lives in gold_utils.py:20-158 (identical math).
# =============================================================================
def is_byte_level_tokenizer(backend) -> bool:  # utils.py:142
    return "ByteLevel" in repr(backend.pre_tokenizer) or "ByteLevel" in repr(backend.decoder)


def piece_byte_len(piece: str) -> int:  # utils.py:148
    return len(piece)


def pad_byte_offsets(offsets, target_length, padding_side):  # utils.py:131
    offs = torch.tensor(offsets, dtype=torch.long).reshape(-1, 2)
    pad_len = target_length - offs.size(0)
    if pad_len <= 0:
        return offs
    pad_block = torch.zeros(pad_len, 2, dtype=torch.long)
    return torch.cat([pad_block, offs], dim=0) if padding_side == "left" else torch.cat([offs, pad_block], dim=0)


def _bytes_to_unicode():  # utils.py:156
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs], strict=True))


_BYTE_LEVEL_DECODER = {ch: b for b, ch in _bytes_to_unicode().items()}


def _byte_level_piece_len(piece, text_bytes, start):  # utils.py:172
    piece_bytes = []
    for ch in piece:
        if ch not in _BYTE_LEVEL_DECODER:
            return None
        piece_bytes.append(_BYTE_LEVEL_DECODER[ch])
    if piece_bytes and piece_bytes[0] == ord(" ") and text_bytes[start : start + 1] != b" ":
        piece_bytes = piece_bytes[1:]
    return len(piece_bytes)


def _split_repeated_byte_offsets(byte_offsets, tokens):  # utils.py:183
    normalized = list(byte_offsets)
    i = 0
    while i < len(byte_offsets):
        j = i + 1
        while j < len(byte_offsets) and byte_offsets[j] == byte_offsets[i]:
            j += 1
        if j - i > 1:
            start, end = byte_offsets[i]
            piece_lengths = [piece_byte_len(token) for token in tokens[i:j]]
            if sum(piece_lengths) == end - start:
                cursor = start
                for offset_idx, length in enumerate(piece_lengths, start=i):
                    normalized[offset_idx] = (cursor, cursor + length)
                    cursor += length
        i = j
    return normalized


def _normalize_byte_offsets(byte_offsets, tokens, text_bytes):  # utils.py:205
    byte_offsets = _split_repeated_byte_offsets(byte_offsets, tokens)
    normalized = []
    cursor = 0
    for idx, (start, end) in enumerate(byte_offsets):
        if start == end:
            normalized.append((cursor, cursor))
            continue
        piece_len = _byte_level_piece_len(tokens[idx], text_bytes, start)
        next_start = byte_offsets[idx + 1][0] if idx + 1 < len(byte_offsets) else None
        has_overlap = start < cursor or (next_start is not None and next_start < end)
        if piece_len is not None and (has_overlap or piece_len == end - start):
            candidate_start = max(start, cursor)
            candidate_end = candidate_start + piece_len
            if candidate_end <= end:
                start = candidate_start
                end = candidate_end
        if start < cursor or end < start:
            raise ValueError("Tokenizer produced overlapping byte offsets that could not be normalized.")
        normalized.append((start, end))
        cursor = end
    return normalized


def encode_with_byte_offsets(backend, texts, add_special_tokens=False):  # utils.py:240
    if not is_byte_level_tokenizer(backend):
        raise NotImplementedError("Cross-tokenizer ULD supports only ByteLevel BPE tokenizers.")
    encs = backend.encode_batch(texts, add_special_tokens=add_special_tokens)
    out = []
    for text, enc in zip(texts, encs, strict=True):
        char_to_byte = [0]
        for ch in text:
            char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))
        byte_offsets = [(char_to_byte[s], char_to_byte[e]) for s, e in enc.offsets]
        byte_offsets = _normalize_byte_offsets(byte_offsets, enc.tokens, text.encode("utf-8"))
        out.append((list(enc.ids), byte_offsets))
    return out


# =============================================================================
# SECTION 2.  build_teacher_inputs_from_texts
#   VERBATIM from trl/experimental/gold/gold_trainer.py:180-258
#   (this is how TRL turns teacher prompt/completion TEXT into
#    teacher input_ids / labels / attention_mask / completion-relative byte offsets;
#    ours replaces it with SGLang teacher hidden-state prefill -- see DIVERGENCE notes.)
# =============================================================================
def build_teacher_inputs_from_texts(tokenizer, prompt_texts, completion_texts):  # gold_trainer.py:180
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    backend = tokenizer.backend_tokenizer

    prompt_token_ids = tokenizer(prompt_texts, add_special_tokens=True)["input_ids"]
    completion_encs = encode_with_byte_offsets(backend, completion_texts, add_special_tokens=False)

    sequences, attention_masks, labels_list, offsets_list = [], [], [], []
    for prompt_ids, (enc_ids, enc_offs), completion_text in zip(
        prompt_token_ids, completion_encs, completion_texts, strict=True
    ):
        if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
            prompt_ids = prompt_ids[:-1]
        completion_ids = list(enc_ids)
        completion_offs = list(enc_offs)
        content_len = len(completion_text.encode("utf-8"))
        sequence = list(prompt_ids) + completion_ids
        offsets = [(0, 0)] * len(prompt_ids) + completion_offs
        if eos_token_id is not None:
            sequence.append(eos_token_id)
            offsets.append((content_len, content_len))
        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        sequences.append(seq_tensor)
        attention_masks.append(torch.ones_like(seq_tensor))
        offsets_list.append(offsets)
        labels = seq_tensor.clone()
        labels[: len(prompt_ids)] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        labels_list.append(labels)

    teacher_input_ids = pad(
        sequences, padding_side="right", padding_value=pad_token_id if pad_token_id is not None else 0
    )
    teacher_attention_mask = pad(attention_masks, padding_side="right", padding_value=0).bool()
    teacher_labels = pad(labels_list, padding_side="right", padding_value=-100)

    if eos_token_id is not None:
        for row in range(teacher_attention_mask.size(0)):
            valid = (
                teacher_input_ids[row] != pad_token_id
                if pad_token_id is not None
                else teacher_attention_mask[row].bool()
            )
            if valid.any():
                last_idx = valid.nonzero(as_tuple=True)[0][-1]
                teacher_attention_mask[row, last_idx + 1 :] = False

    target_len = teacher_input_ids.size(1)
    teacher_byte_offsets = torch.stack(
        [pad_byte_offsets(offs, target_len, padding_side="right") for offs in offsets_list], dim=0
    )
    return teacher_input_ids, teacher_labels, teacher_attention_mask, teacher_byte_offsets


# =============================================================================
# SECTION 3.  generalized_jsd_loss
#   VERBATIM from GOLDTrainer.generalized_jsd_loss
#   trl/experimental/gold/gold_trainer.py:2397-2486  (the ":2398 anchor")
#   ours mirror: gold_utils.py::_gold_generalized_jsd_from_probs (:682-709) uses the
#   logits_are_probs=True + reduction over columns path (sum(dim=-1)); identical math.
# =============================================================================
def generalized_jsd_loss(
    student_logits,
    teacher_logits,
    labels=None,
    beta=0.5,
    temperature=1.0,
    reduction="batchmean",
    logits_are_probs=False,
    num_items_in_batch=None,
):
    if logits_are_probs:
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta * kl_teacher + (1 - beta) * kl_student

    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    if num_items_in_batch is not None:
        jsd_sum = jsd.sum()
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(jsd_sum.device)
        return jsd_sum / num_items_in_batch
    if reduction == "batchmean":
        denom = mask.sum().clamp_min(1) if labels is not None else max(jsd.size(0), 1)
        return jsd.sum() / denom
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    else:
        return jsd


# =============================================================================
# SECTION 4.  ULDLoss  (Universal Logit Distillation)
#   VERBATIM (loss-relevant methods) from
#   trl/experimental/gold/gold_trainer.py:261-1284
#   Dropped: hidden-state / deepspeed-zero3 / OOM-retry / debug-timing paths
#   (they do not change any loss value at TP=1/CP=1; noted at each call site).
# =============================================================================
class MockGOLDConfig:
    """Stand-in for GOLDConfig carrying only the fields ULDLoss reads.
    Field defaults mirror gold_config.py:307-394; per-arm values are the paper flags
    from examples/baselines/arms/{run_uld,run_gold_matched,run_gold_unmatched}.sh
    + _launcher_gold.sh OPD_ARGS.
    """

    def __init__(self, **kw):
        self.uld_crossentropy_weight = kw.get("uld_crossentropy_weight", 0.0)      # --gold-ce-weight 0.0
        self.uld_distillation_weight = kw.get("uld_distillation_weight", 1.0)      # --gold-distillation-weight 1.0
        self.uld_student_temperature = kw.get("uld_student_temperature", 1.0)      # --gold-student-temperature 1.0
        self.uld_teacher_temperature = kw.get("uld_teacher_temperature", 1.0)      # --gold-teacher-temperature 1.0
        self.uld_skip_student_eos = kw.get("uld_skip_student_eos", True)           # --gold-skip-student-eos
        self.uld_skip_teacher_eos = kw.get("uld_skip_teacher_eos", True)           # --gold-skip-teacher-eos
        self.use_extended_uld = kw.get("use_extended_uld", True)                   # --gold-use-extended-uld
        self.uld_use_hybrid_loss = kw.get("uld_use_hybrid_loss", False)            # --gold-use-hybrid-loss
        self.uld_hybrid_matched_weight = kw.get("uld_hybrid_matched_weight", None) # --gold-hybrid-matched-weight
        self.uld_hybrid_unmatched_weight = kw.get("uld_hybrid_unmatched_weight", None)
        self.uld_hybrid_unmatched_chunk_size = kw.get("uld_hybrid_unmatched_chunk_size", 256)
        self.uld_hybrid_matched_chunk_size = kw.get("uld_hybrid_matched_chunk_size", 256)
        self.beta = kw.get("beta", 0.5)                                            # --gold-beta 0.5


def empty_cache():  # no-op stand-in for trl.experimental.utils.empty_cache
    pass


class ULDLoss(nn.Module):
    # gold_trainer.py:266
    def __init__(self, config, student_tokenizer=None, teacher_tokenizer=None, device=None):
        super().__init__()
        self.device = device
        self.crossentropy_weight = config.uld_crossentropy_weight
        self.distillation_weight = config.uld_distillation_weight
        self.student_temperature = config.uld_student_temperature
        self.teacher_temperature = config.uld_teacher_temperature
        self.skip_student_eos = config.uld_skip_student_eos
        self.skip_teacher_eos = config.uld_skip_teacher_eos
        self.use_extended_uld = config.use_extended_uld
        self.ignore_index = -100
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.use_hybrid_loss = getattr(config, "uld_use_hybrid_loss", False)
        self.hybrid_matched_weight = getattr(config, "uld_hybrid_matched_weight", None)
        self.hybrid_unmatched_weight = getattr(config, "uld_hybrid_unmatched_weight", None)
        self.beta = getattr(config, "beta", 1.0)
        self.hybrid_unmatched_chunk_size = getattr(config, "uld_hybrid_unmatched_chunk_size", 256)
        self.hybrid_matched_chunk_size = getattr(config, "uld_hybrid_matched_chunk_size", 256)
        self._vocab_mapping = None
        self._teacher_matched_ids = None
        self._student_matched_ids = None
        self.last_matched_loss = None
        self.last_unmatched_loss = None
        if self.use_hybrid_loss and student_tokenizer is not None and teacher_tokenizer is not None:
            self._initialize_vocabulary_mapping()

    def __call__(  # gold_trainer.py:297
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
        student_byte_offsets=None,
        teacher_byte_offsets=None,
    ):
        if self.crossentropy_weight > 0:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = student_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            crossentropy_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            crossentropy_loss = self.crossentropy_weight * crossentropy_loss
        else:
            crossentropy_loss = 0.0
        distillation_loss = self._compute_distillation_loss(
            student_logits, teacher_logits, student_labels, teacher_labels,
            student_input_ids, teacher_input_ids,
            student_byte_offsets=student_byte_offsets, teacher_byte_offsets=teacher_byte_offsets,
        )
        return crossentropy_loss + distillation_loss

    def _initialize_vocabulary_mapping(self):  # gold_trainer.py:388
        student_vocab = self.student_tokenizer.get_vocab()
        teacher_vocab = self.teacher_tokenizer.get_vocab()
        student_token_to_id = dict(student_vocab.items())
        vocab_mapping, teacher_matched_ids, student_matched_ids = {}, set(), set()
        for token_str, teacher_id in teacher_vocab.items():
            if token_str in student_token_to_id:
                student_id = student_token_to_id[token_str]
                vocab_mapping[teacher_id] = student_id
                teacher_matched_ids.add(teacher_id)
                student_matched_ids.add(student_id)
        self._vocab_mapping = vocab_mapping
        self._teacher_matched_ids = teacher_matched_ids
        self._student_matched_ids = student_matched_ids
        max_matched_teacher_id = max(self._vocab_mapping.keys())
        self.mapping_tensor = torch.full((max_matched_teacher_id + 1,), -1, dtype=torch.long)
        for k, v in self._vocab_mapping.items():
            self.mapping_tensor[k] = v
        if self.device is not None:
            self.mapping_tensor = self.mapping_tensor.to(self.device)

    def _compute_distillation_loss(  # gold_trainer.py:420
        self, student_logits, teacher_logits, student_labels, teacher_labels,
        student_input_ids, teacher_input_ids, student_byte_offsets=None, teacher_byte_offsets=None,
    ):
        student_answer_index, student_answer_size = self._get_start_and_size_answers(student_labels)
        teacher_answer_index, teacher_answer_size = self._get_start_and_size_answers(teacher_labels)
        if self.skip_student_eos:
            student_answer_size = [size - 1 for size in student_answer_size]
        if self.skip_teacher_eos:
            teacher_answer_size = [size - 1 for size in teacher_answer_size]
        if (
            not student_answer_size or not teacher_answer_size
            or max(max(student_answer_size), max(teacher_answer_size)) <= 0
        ):
            return torch.zeros(1, device=student_logits.device, requires_grad=True) * student_logits.sum() * 1e-8

        batch_size = student_logits.size(0)
        distillation_losses = []
        for i in range(batch_size):
            student_start = student_answer_index[i]
            student_size = student_answer_size[i]
            teacher_start = teacher_answer_index[i]
            teacher_size = teacher_answer_size[i]
            if student_size <= 0 or teacher_size <= 0:
                distillation_losses.append(student_logits[i].sum() * 0.0)
                continue
            student_answer_logits = student_logits[i, student_start : student_start + student_size]
            teacher_answer_logits = teacher_logits[i, teacher_start : teacher_start + teacher_size]
            student_token_ids = student_input_ids[i, student_start : student_start + student_size].tolist()
            teacher_token_ids = teacher_input_ids[i, teacher_start : teacher_start + teacher_size].tolist()

            if self.use_extended_uld:
                if student_byte_offsets is None or teacher_byte_offsets is None:
                    raise ValueError("Byte offsets are required when use_extended_uld=True.")
                s_answer = student_byte_offsets[i, student_start : student_start + student_size].tolist()
                t_answer = teacher_byte_offsets[i, teacher_start : teacher_start + teacher_size].tolist()
                student_groups, teacher_groups = self._align_by_byte_offsets(s_answer, t_answer)
                paired = [
                    (sg, tg) for sg, tg in zip(student_groups, teacher_groups, strict=False) if sg and tg
                ]
                student_groups = [sg for sg, _ in paired]
                teacher_groups = [tg for _, tg in paired]

                if self.use_hybrid_loss and self._vocab_mapping is not None:
                    aligned_loss = self._compute_extended_hybrid_uld_loss_streaming(
                        student_answer_logits, teacher_answer_logits,
                        student_groups, teacher_groups, student_token_ids, teacher_token_ids,
                    )
                    distillation_losses.append(aligned_loss)
                    continue

                student_probs = F.softmax(student_answer_logits / self.student_temperature, dim=-1)
                teacher_probs = F.softmax(teacher_answer_logits / self.teacher_temperature, dim=-1)
                student_aligned = self._merge_probabilities_with_alignment_groups(
                    student_probs, student_groups, student_token_ids
                )
                teacher_aligned = self._merge_probabilities_with_alignment_groups(
                    teacher_probs, teacher_groups, teacher_token_ids
                )
            else:
                student_probs = F.softmax(student_answer_logits / self.student_temperature, dim=-1)
                teacher_probs = F.softmax(teacher_answer_logits / self.teacher_temperature, dim=-1)
                min_length = min(len(student_token_ids), len(teacher_token_ids))
                student_aligned = student_probs[:min_length, :]
                teacher_aligned = teacher_probs[:min_length, :]

            if self.use_hybrid_loss and self._vocab_mapping is not None:
                aligned_loss = self._compute_hybrid_uld_loss(student_aligned, teacher_aligned)
            else:
                student_sorted = student_aligned.sort(dim=-1, descending=True).values
                teacher_sorted = teacher_aligned.sort(dim=-1, descending=True).values
                student_vocab_size = student_sorted.size(-1)
                teacher_vocab_size = teacher_sorted.size(-1)
                max_vocab_size = max(student_vocab_size, teacher_vocab_size)
                if student_vocab_size < max_vocab_size:
                    student_sorted = F.pad(student_sorted, (0, max_vocab_size - student_vocab_size))
                if teacher_vocab_size < max_vocab_size:
                    teacher_sorted = F.pad(teacher_sorted, (0, max_vocab_size - teacher_vocab_size))
                aligned_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
                aligned_loss /= student_aligned.size(0)  # normalize by number of groups
            distillation_losses.append(aligned_loss)

        distillation_loss = torch.stack(distillation_losses).mean()  # per-sample-mean-then-batch-mean
        return self.distillation_weight * distillation_loss

    @staticmethod
    def _align_by_byte_offsets(s_offsets, t_offsets):  # gold_trainer.py:625
        s_groups, t_groups = [], []
        s_start = t_start = s = t = 0
        n_s, n_t = len(s_offsets), len(t_offsets)
        while s < n_s and t < n_t:
            s_end, t_end = s_offsets[s][1], t_offsets[t][1]
            if s_end < t_end:
                s += 1
            elif s_end > t_end:
                t += 1
            else:
                s += 1
                t += 1
                s_groups.append(list(range(s_start, s)))
                t_groups.append(list(range(t_start, t)))
                s_start, t_start = s, t
        if s < n_s or t < n_t:
            s_groups.append(list(range(s_start, n_s)))
            t_groups.append(list(range(t_start, n_t)))
        return s_groups, t_groups

    def _merge_probabilities_with_alignment_groups(self, probs, alignment_groups, token_ids=None):  # gold_trainer.py:651
        if not alignment_groups:
            return probs
        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device, dtype=probs.dtype)
        eps = 1e-8
        for group_idx, group in enumerate(alignment_groups):
            if len(group) > 1:
                if token_ids is None:
                    raise ValueError("token_ids must be provided when merging multi-token groups.")
                first_pos = group[0]  # observed: FIRST-position marginal base
                marginal_probs = probs[first_pos]
                conditional_prob_product = 1.0
                for idx in group[1:]:  # scalars = group[1:]
                    actual_token_id = token_ids[idx]
                    token_prob = probs[idx, actual_token_id].clamp_min(eps)  # 1e-8 clamp on each scalar
                    conditional_prob_product *= token_prob
                aligned_probs[group_idx] = marginal_probs * conditional_prob_product
            elif len(group) == 1:
                aligned_probs[group_idx] = probs[group[0]]
            else:
                aligned_probs[group_idx] = torch.zeros_like(probs[0])
        return aligned_probs

    def _compute_aligned_probability_chunk_from_logits(self, logits, alignment_groups, token_ids, temperature):  # gold_trainer.py:728
        vocab_size = logits.size(-1)
        if not alignment_groups:
            return logits.new_empty((0, vocab_size))
        first_positions = torch.tensor([group[0] for group in alignment_groups], dtype=torch.long, device=logits.device)
        aligned_probs = F.softmax(logits.index_select(0, first_positions) / temperature, dim=-1)
        if any(len(group) > 1 for group in alignment_groups):
            eps = 1e-8
            extra_positions, extra_token_ids, extra_group_indices = [], [], []
            for group_idx, group in enumerate(alignment_groups):
                for idx in group[1:]:
                    extra_positions.append(idx)
                    extra_token_ids.append(token_ids[idx])
                    extra_group_indices.append(group_idx)
            multiplier_values = [[] for _ in alignment_groups]
            if extra_positions:
                extra_positions_tensor = torch.tensor(extra_positions, dtype=torch.long, device=logits.device)
                extra_token_ids_tensor = torch.tensor(extra_token_ids, dtype=torch.long, device=logits.device)
                extra_probs = F.softmax(logits.index_select(0, extra_positions_tensor) / temperature, dim=-1)
                extra_actual_probs = extra_probs.gather(1, extra_token_ids_tensor.unsqueeze(1)).squeeze(1).clamp_min(eps)
                for group_idx, token_prob in zip(extra_group_indices, extra_actual_probs, strict=True):
                    multiplier_values[group_idx].append(token_prob)
            multipliers = [
                torch.stack(values).prod() if values else aligned_probs.new_tensor(1.0)
                for values in multiplier_values
            ]
            aligned_probs = aligned_probs * torch.stack(multipliers).unsqueeze(-1)
        return aligned_probs

    def _get_hybrid_vocab_tensors(self, student_vocab_size, teacher_vocab_size, device):  # gold_trainer.py:827
        if self._teacher_matched_ids:
            teacher_matched_indices = torch.tensor(sorted(self._teacher_matched_ids), dtype=torch.long, device=device)
            student_matched_indices = self.mapping_tensor[teacher_matched_indices]
        else:
            teacher_matched_indices = torch.tensor([], dtype=torch.long, device=device)
            student_matched_indices = torch.tensor([], dtype=torch.long, device=device)
        teacher_matched_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(student_vocab_size, dtype=torch.bool, device=device)
        if len(teacher_matched_indices) > 0:
            teacher_matched_mask[teacher_matched_indices] = True
            student_matched_mask[student_matched_indices] = True
        return (
            student_matched_indices, teacher_matched_indices,
            ~student_matched_mask, ~teacher_matched_mask,
            teacher_matched_indices.numel(),
        )

    def _compute_extended_hybrid_uld_loss_streaming(  # gold_trainer.py:869
        self, student_answer_logits, teacher_answer_logits,
        student_groups, teacher_groups, student_token_ids, teacher_token_ids,
    ):
        device = student_answer_logits.device
        aligned_len = len(student_groups)
        if aligned_len == 0:
            zero = torch.tensor(0.0, device=device)
            self.last_matched_loss = zero
            self.last_unmatched_loss = zero
            return zero
        (student_matched_indices, teacher_matched_indices, student_unmatched_mask,
         teacher_unmatched_mask, matched_token_count) = self._get_hybrid_vocab_tensors(
            student_answer_logits.size(-1), teacher_answer_logits.size(-1), device
        )
        matched_loss_sum = torch.tensor(0.0, device=device)
        unmatched_loss_sum = torch.tensor(0.0, device=device)
        stream_chunk_size = max(1, min(int(self.hybrid_unmatched_chunk_size), int(self.hybrid_matched_chunk_size)))
        for start in range(0, aligned_len, stream_chunk_size):
            end = min(start + stream_chunk_size, aligned_len)
            student_chunk = self._compute_aligned_probability_chunk_from_logits(
                student_answer_logits, student_groups[start:end], student_token_ids, self.student_temperature
            )
            teacher_chunk = self._compute_aligned_probability_chunk_from_logits(
                teacher_answer_logits, teacher_groups[start:end], teacher_token_ids, self.teacher_temperature
            )
            chunk_matched, chunk_unmatched = self._compute_chunked_hybrid_loss_components(
                student_chunk, teacher_chunk, student_matched_indices, teacher_matched_indices,
                student_unmatched_mask, teacher_unmatched_mask,
                chunk_size=self.hybrid_unmatched_chunk_size, matched_chunk_size=self.hybrid_matched_chunk_size,
            )
            chunk_len = end - start
            matched_loss_sum = matched_loss_sum + chunk_matched * chunk_len
            unmatched_loss_sum = unmatched_loss_sum + chunk_unmatched * chunk_len
        matched_loss = matched_loss_sum / aligned_len
        unmatched_loss = unmatched_loss_sum / aligned_len
        if self.hybrid_matched_weight is None:
            hybrid_matched_weight = matched_token_count / max(1, teacher_answer_logits.size(-1))
            hybrid_unmatched_weight = 1.0 - hybrid_matched_weight
        else:
            hybrid_matched_weight = self.hybrid_matched_weight
            hybrid_unmatched_weight = self.hybrid_unmatched_weight
        self.last_matched_loss = matched_loss
        self.last_unmatched_loss = unmatched_loss
        return hybrid_matched_weight * matched_loss + hybrid_unmatched_weight * unmatched_loss

    def _compute_hybrid_uld_loss(self, student_aligned, teacher_aligned):  # gold_trainer.py:1057
        device = student_aligned.device
        student_vocab_size = student_aligned.size(-1)
        teacher_vocab_size = teacher_aligned.size(-1)
        (student_matched_indices, teacher_matched_indices, student_unmatched_mask,
         teacher_unmatched_mask, matched_token_count) = self._get_hybrid_vocab_tensors(
            student_vocab_size, teacher_vocab_size, device
        )
        matched_loss, unmatched_loss = self._compute_chunked_hybrid_loss_components(
            student_aligned, teacher_aligned, student_matched_indices, teacher_matched_indices,
            student_unmatched_mask, teacher_unmatched_mask,
            chunk_size=self.hybrid_unmatched_chunk_size, matched_chunk_size=self.hybrid_matched_chunk_size,
        )
        if self.hybrid_matched_weight is None:
            hybrid_matched_weight = matched_token_count / max(1, teacher_vocab_size)
            hybrid_unmatched_weight = 1.0 - hybrid_matched_weight
        else:
            hybrid_matched_weight = self.hybrid_matched_weight
            hybrid_unmatched_weight = self.hybrid_unmatched_weight
        total_loss = hybrid_matched_weight * matched_loss + hybrid_unmatched_weight * unmatched_loss
        self.last_matched_loss = matched_loss
        self.last_unmatched_loss = unmatched_loss
        return total_loss

    def _compute_chunked_hybrid_loss_components(  # gold_trainer.py:1111
        self, student_aligned, teacher_aligned, student_matched_indices, teacher_matched_indices,
        student_unmatched_mask, teacher_unmatched_mask, chunk_size, matched_chunk_size=None,
    ):
        device = student_aligned.device
        seq_len = student_aligned.size(0)
        if seq_len == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero
        matched_loss = torch.tensor(0.0, device=device)
        matched_chunk_size = chunk_size if matched_chunk_size is None else matched_chunk_size
        matched_chunk_size = max(1, int(matched_chunk_size))
        if teacher_matched_indices.numel() > 0:
            matched_loss_sum = torch.tensor(0.0, device=device)
            for start in range(0, seq_len, matched_chunk_size):
                end = min(start + matched_chunk_size, seq_len)
                matched_loss_sum = matched_loss_sum + generalized_jsd_loss(
                    student_aligned[start:end, student_matched_indices],
                    teacher_aligned[start:end, teacher_matched_indices],
                    labels=None, beta=self.beta, temperature=1.0, reduction="sum", logits_are_probs=True,
                )
            matched_loss = matched_loss_sum / seq_len
        unmatched_loss = torch.tensor(0.0, device=device)
        if teacher_unmatched_mask.any() and student_unmatched_mask.any():
            unmatched_loss = self._compute_chunked_unmatched_loss(
                student_aligned, teacher_aligned, student_unmatched_mask, teacher_unmatched_mask, chunk_size=chunk_size,
            )
        return matched_loss, unmatched_loss

    def _compute_chunked_unmatched_loss(  # gold_trainer.py:1159 (OOM-retry stripped; pure math kept)
        self, student_aligned, teacher_aligned, student_unmatched_mask, teacher_unmatched_mask, chunk_size,
    ):
        device = student_aligned.device
        seq_len = student_aligned.size(0)
        unmatched_chunk_size = max(1, int(chunk_size))
        loss_sum = torch.tensor(0.0, device=device)
        start = 0
        while start < seq_len:
            end = start + min(unmatched_chunk_size, seq_len - start)
            student_sorted = student_aligned[start:end, student_unmatched_mask].sort(dim=-1, descending=True).values
            teacher_sorted = teacher_aligned[start:end, teacher_unmatched_mask].sort(dim=-1, descending=True).values
            student_size = student_sorted.size(-1)
            teacher_size = teacher_sorted.size(-1)
            max_size = max(student_size, teacher_size)
            if student_size < max_size:
                student_sorted = F.pad(student_sorted, (0, max_size - student_size))
            if teacher_size < max_size:
                teacher_sorted = F.pad(teacher_sorted, (0, max_size - teacher_size))
            loss_sum = loss_sum + F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
            start = end
        return loss_sum / seq_len

    def _get_start_and_size_answers(self, answer_tensors):  # gold_trainer.py:1270
        answers_index, answers_size = [], []
        for answer in answer_tensors:
            answer_mask = answer.ne(self.ignore_index)
            if not answer_mask.any():
                answers_index.append(0)
                answers_size.append(0)
                continue
            valid_indices = answer_mask.nonzero(as_tuple=True)[0]
            answers_index.append(int(valid_indices[0].item()))
            answers_size.append(int(answer_mask.sum().item()))
        return answers_index, answers_size


# =============================================================================
# SECTION 5.  IMPLEMENTATION CORE MIRROR
#   VERBATIM pure-torch per-group math from ours
#   slime_plugins/baselines/gold/gold_kernels.py
#   These are exactly the functions gold_loss.py calls per aligned group at
#   TP=1/CP=1; the megatron TP/CP wrappers around them cannot run on CPU but are
#   no-ops at TP=1/CP=1 (see gold_utils.py:502-505, differentiable all-gather skipped).
# =============================================================================
def ours_align_by_byte_offsets(student_offsets, teacher_offsets):  # gold_utils.py:254 (identical to TRL:625)
    s_groups, t_groups = [], []
    s_start = t_start = s = t = 0
    n_s, n_t = len(student_offsets), len(teacher_offsets)
    while s < n_s and t < n_t:
        s_end, t_end = student_offsets[s][1], teacher_offsets[t][1]
        if s_end < t_end:
            s += 1
        elif s_end > t_end:
            t += 1
        else:
            s += 1
            t += 1
            s_groups.append(list(range(s_start, s)))
            t_groups.append(list(range(t_start, t)))
            s_start, t_start = s, t
    if s < n_s or t < n_t:
        s_groups.append(list(range(s_start, n_s)))
        t_groups.append(list(range(t_start, n_t)))
    return s_groups, t_groups


def ours_merge_log_probs_with_alignment_groups(log_probs, alignment_groups, token_ids, *, clamp_min_prob=None, bayesian=False):
    # gold_utils.py:280
    if not alignment_groups:
        return log_probs[:0]
    log_clamp = math.log(clamp_min_prob) if clamp_min_prob is not None else None
    merged_rows = []
    for group in alignment_groups:
        if len(group) == 1:
            merged_rows.append(log_probs[group[0]])
        elif len(group) > 1:
            if bayesian:
                base_pos = int(group[-1])
                scalar_positions = group[:-1]
            else:
                base_pos = int(group[0])
                scalar_positions = group[1:]
            cond = log_probs.new_zeros(())
            for idx in scalar_positions:
                token_id = int(token_ids[idx])
                tail_term = log_probs[int(idx), token_id]
                if log_clamp is not None:
                    tail_term = tail_term.clamp_min(log_clamp)
                cond = cond + tail_term
            merged_rows.append(log_probs[base_pos] + cond)
        else:
            merged_rows.append(log_probs.new_full((log_probs.shape[-1],), float("-inf")))
    return torch.stack(merged_rows, dim=0)


def ours_sorted_l1_from_log_probs(student_log_probs, teacher_log_probs):  # gold_utils.py:613
    student_sorted = student_log_probs.exp().sort(dim=-1, descending=True).values
    teacher_sorted = teacher_log_probs.exp().sort(dim=-1, descending=True).values
    max_vocab = max(student_sorted.shape[-1], teacher_sorted.shape[-1])
    if student_sorted.shape[-1] < max_vocab:
        student_sorted = F.pad(student_sorted, (0, max_vocab - student_sorted.shape[-1]))
    if teacher_sorted.shape[-1] < max_vocab:
        teacher_sorted = F.pad(teacher_sorted, (0, max_vocab - teacher_sorted.shape[-1]))
    return F.l1_loss(student_sorted, teacher_sorted, reduction="none").sum(dim=-1)


def ours_generalized_jsd_from_probs(student_probs, teacher_probs, *, beta):  # gold_utils.py:682
    student_log_probs = torch.log(student_probs.clamp_min(1e-8))
    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-8))
    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]), dim=0
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1.0 - beta_t) * kl_student
    return jsd.sum(dim=-1)


def ours_make_hybrid_vocab_mapping(student_tokenizer, teacher_tokenizer, *, device, student_vocab_dim,
                                  teacher_vocab_dim, student_real_vocab_size, teacher_real_vocab_size):
    # gold_utils.py:628
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    pairs = []
    for token, teacher_id_raw in teacher_vocab.items():
        student_id_raw = student_vocab.get(token)
        if student_id_raw is None:
            continue
        teacher_id = int(teacher_id_raw)
        student_id = int(student_id_raw)
        if (0 <= teacher_id < int(teacher_real_vocab_size) and 0 <= student_id < int(student_real_vocab_size)
                and teacher_id < int(teacher_vocab_dim) and student_id < int(student_vocab_dim)):
            pairs.append((teacher_id, student_id))
    pairs.sort()
    if pairs:
        teacher_matched_ids = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
        student_matched_ids = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    else:
        teacher_matched_ids = torch.empty((0,), dtype=torch.long, device=device)
        student_matched_ids = torch.empty((0,), dtype=torch.long, device=device)
    teacher_unmatched_mask = torch.zeros(int(teacher_vocab_dim), dtype=torch.bool, device=device)
    student_unmatched_mask = torch.zeros(int(student_vocab_dim), dtype=torch.bool, device=device)
    teacher_unmatched_mask[: int(teacher_real_vocab_size)] = True
    student_unmatched_mask[: int(student_real_vocab_size)] = True
    if teacher_matched_ids.numel() > 0:
        teacher_unmatched_mask[teacher_matched_ids] = False
        student_unmatched_mask[student_matched_ids] = False
    return {
        "teacher_matched_ids": teacher_matched_ids, "student_matched_ids": student_matched_ids,
        "teacher_unmatched_mask": teacher_unmatched_mask, "student_unmatched_mask": student_unmatched_mask,
        "teacher_vocab_size": int(teacher_real_vocab_size), "matched_count": int(teacher_matched_ids.numel()),
    }


def ours_hybrid_loss_from_log_probs(student_log_probs, teacher_log_probs, *, hybrid_vocab, beta,
                                   matched_weight, unmatched_weight):  # gold_utils.py:712
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    device = student_probs.device
    rows = int(student_probs.shape[0])
    matched_loss = torch.zeros((rows,), dtype=torch.float32, device=device)
    unmatched_loss = torch.zeros((rows,), dtype=torch.float32, device=device)
    teacher_matched_ids = hybrid_vocab["teacher_matched_ids"]
    student_matched_ids = hybrid_vocab["student_matched_ids"]
    if isinstance(teacher_matched_ids, torch.Tensor) and teacher_matched_ids.numel() > 0:
        matched_loss = ours_generalized_jsd_from_probs(
            student_probs.index_select(-1, student_matched_ids.to(device)),
            teacher_probs.index_select(-1, teacher_matched_ids.to(device)),
            beta=beta,
        )
    student_unmatched_mask = hybrid_vocab["student_unmatched_mask"]
    teacher_unmatched_mask = hybrid_vocab["teacher_unmatched_mask"]
    student_unmatched = student_probs[:, student_unmatched_mask.to(device)]
    teacher_unmatched = teacher_probs[:, teacher_unmatched_mask.to(device)]
    if student_unmatched.shape[-1] > 0 and teacher_unmatched.shape[-1] > 0:
        student_sorted = student_unmatched.sort(dim=-1, descending=True).values
        teacher_sorted = teacher_unmatched.sort(dim=-1, descending=True).values
        max_vocab = max(student_sorted.shape[-1], teacher_sorted.shape[-1])
        if student_sorted.shape[-1] < max_vocab:
            student_sorted = F.pad(student_sorted, (0, max_vocab - student_sorted.shape[-1]))
        if teacher_sorted.shape[-1] < max_vocab:
            teacher_sorted = F.pad(teacher_sorted, (0, max_vocab - teacher_sorted.shape[-1]))
        unmatched_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="none").sum(dim=-1)
    if matched_weight is None:
        matched_count = int(hybrid_vocab["matched_count"])
        teacher_vocab_size = max(int(hybrid_vocab["teacher_vocab_size"]), 1)
        hmw = matched_count / teacher_vocab_size
        huw = 1.0 - hmw
    else:
        hmw = float(matched_weight)
        huw = float(unmatched_weight)
    total_loss = hmw * matched_loss + huw * unmatched_loss
    return total_loss, matched_loss, unmatched_loss


# =============================================================================
# SECTION 6.  ours DEFAULT (non-faithful) merge, replicating gold_loss.py without
#   --gold-trl-faithful: SHIFTED pairing is upstream (not modeled here since we feed
#   identical answer windows), but the MERGE differs: no 1e-8 clamp, first-position base.
#   Modeled by clamp_min_prob=None, bayesian=False in ours_merge_log_probs_*.
# =============================================================================


def _print_hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _build_synthetic_single(seed=0):
    """One aligned answer region. Student has a 2-token group that merges to one
    teacher token (exercises the chain-rule merge). Shared small vocab so
    student/teacher token strings can overlap for the hybrid arms."""
    torch.manual_seed(seed)
    V = 12  # shared toy vocab
    # student answer tokens (4 tokens): bytes  [0,2)[2,4)[4,5)[5,7)
    # teacher answer tokens (3 tokens): bytes  [0,4)[4,5)[5,7)
    # => group0: student{0,1} <-> teacher{0}   (merge!), group1: {2}<->{1}, group2:{3}<->{2}
    stu_offsets = [(0, 2), (2, 4), (4, 5), (5, 7)]
    tea_offsets = [(0, 4), (4, 5), (5, 7)]
    stu_ids = [3, 5, 7, 9]
    tea_ids = [3, 7, 9]
    stu_logits = torch.randn(len(stu_ids), V) * 1.3
    tea_logits = torch.randn(len(tea_ids), V) * 1.3
    return V, stu_logits, tea_logits, stu_ids, tea_ids, stu_offsets, tea_offsets


def _make_batch_tensors(V, stu_logits, tea_logits, stu_ids, tea_ids, stu_offsets, tea_offsets):
    """Wrap a single answer region into full [batch=1, seq] tensors with a 1-token
    prompt prefix and a trailing EOS (skip_*_eos=True will trim it), matching what
    compute_loss feeds ULDLoss.__call__."""
    EOS = 11
    PROMPT = 1
    # student sequence: [prompt] + answer_ids + [eos]
    s_seq = [PROMPT] + stu_ids + [EOS]
    t_seq = [PROMPT] + tea_ids + [EOS]
    s_labels = torch.tensor([[-100] + stu_ids + [EOS]])       # answer region = labels!=-100
    t_labels = torch.tensor([[-100] + tea_ids + [EOS]])
    s_input_ids = torch.tensor([s_seq])
    t_input_ids = torch.tensor([t_seq])
    # logits: prompt row + answer rows + eos row (prompt/eos rows unused for loss)
    s_full = torch.cat([torch.randn(1, V), stu_logits, torch.randn(1, V)], dim=0).unsqueeze(0)
    t_full = torch.cat([torch.randn(1, V), tea_logits, torch.randn(1, V)], dim=0).unsqueeze(0)
    # byte offsets: prompt (0,0), answer offsets, eos (content_len,content_len)
    s_content = stu_offsets[-1][1]
    t_content = tea_offsets[-1][1]
    s_off = torch.tensor([[(0, 0)] + stu_offsets + [(s_content, s_content)]])
    t_off = torch.tensor([[(0, 0)] + tea_offsets + [(t_content, t_content)]])
    return s_full, t_full, s_labels, t_labels, s_input_ids, t_input_ids, s_off, t_off


class _ToyTokenizer:
    """Minimal get_vocab() provider so ULDLoss / ours hybrid mapping can build the
    matched/unmatched split on the toy vocab. Overlap: ids {3,7,9,11} shared."""

    def __init__(self, mapping):
        self._vocab = mapping

    def get_vocab(self):
        return dict(self._vocab)


def _toy_tokenizers():
    # student & teacher share token strings for a subset of ids to create matches.
    stu = {f"tok{i}": i for i in range(12)}
    # teacher uses DIFFERENT strings for ids 0,1,2,4,6,8,10 but SAME for 3,7,9,11 and 5.
    tea = {}
    for i in range(12):
        if i in (3, 5, 7, 9, 11):
            tea[f"tok{i}"] = i          # matched string -> mapped id
        else:
            tea[f"teaonly{i}"] = i       # unmatched
    return _ToyTokenizer(stu), _ToyTokenizer(tea)


def main():
    torch.set_printoptions(precision=8)
    device = torch.device("cpu")

    # ---- shared synthetic answer region ----
    V, stu_logits, tea_logits, stu_ids, tea_ids, stu_offsets, tea_offsets = _build_synthetic_single(seed=7)
    (s_full, t_full, s_labels, t_labels, s_input_ids, t_input_ids, s_off, t_off) = _make_batch_tensors(
        V, stu_logits, tea_logits, stu_ids, tea_ids, stu_offsets, tea_offsets
    )
    stu_tok, tea_tok = _toy_tokenizers()

    # groups that the byte-offset aligner produces on the *answer* region
    sg, tg = ULDLoss._align_by_byte_offsets(stu_offsets, tea_offsets)
    _print_hdr("SETUP")
    print(f"vocab={V}  student answer tokens={stu_ids}  teacher answer tokens={tea_ids}")
    print(f"student byte offsets={stu_offsets}")
    print(f"teacher byte offsets={tea_offsets}")
    print(f"_align_by_byte_offsets -> student_groups={sg}  teacher_groups={tg}")
    print("(group0 merges student tokens {0,1} to teacher token {0} -> chain-rule multi-token merge)")

    # =====================================================================
    # ARM 1: uld-trl  (no hybrid, --gold-trl-faithful, merge-strategy observed)
    # =====================================================================
    _print_hdr("ARM 1  uld-trl : TRL ULDLoss (extended, non-hybrid) vs ours core (observed + 1e-8 clamp)")
    cfg = MockGOLDConfig(uld_use_hybrid_loss=False)
    uld = ULDLoss(cfg, student_tokenizer=stu_tok, teacher_tokenizer=tea_tok, device=device)
    trl_loss = uld(
        s_full, t_full, s_labels, t_labels, s_input_ids, t_input_ids,
        student_byte_offsets=s_off, teacher_byte_offsets=t_off,
    )
    print(f"TRL ULDLoss.__call__ distillation loss = {float(trl_loss):.10f}")

    # ours core: log-space merge (observed base=group[0], scalars=group[1:], clamp log(1e-8)) + sorted L1 / n_groups
    stu_lp = F.log_softmax(stu_logits / cfg.uld_student_temperature, dim=-1)
    tea_lp = F.log_softmax(tea_logits / cfg.uld_teacher_temperature, dim=-1)
    sg2, tg2 = ours_align_by_byte_offsets(stu_offsets, tea_offsets)
    stu_merged = ours_merge_log_probs_with_alignment_groups(stu_lp, sg2, stu_ids, clamp_min_prob=1e-8, bayesian=False)
    tea_merged = ours_merge_log_probs_with_alignment_groups(tea_lp, tg2, tea_ids, clamp_min_prob=1e-8, bayesian=False)
    per_group = ours_sorted_l1_from_log_probs(stu_merged, tea_merged)
    ours_loss = per_group.sum() / per_group.shape[0]  # batch=1 -> pooled == per-sample mean
    print(f"ours core   sorted-L1 mean-over-groups   = {float(ours_loss):.10f}")
    print(f"|TRL - ours| = {abs(float(trl_loss) - float(ours_loss)):.3e}   "
          f"(expect ~1e-6: prob-space vs log-space merge)")

    # =====================================================================
    # ARM 2: gold-matched  (hybrid, beta=0.5, matched_w=1.0, unmatched_w=0.0)
    # =====================================================================
    _print_hdr("ARM 2  gold-matched : TRL hybrid (mw=1,uw=0) matched-JSD vs ours core")
    cfg_m = MockGOLDConfig(uld_use_hybrid_loss=True, beta=0.5,
                           uld_hybrid_matched_weight=1.0, uld_hybrid_unmatched_weight=0.0)
    uld_m = ULDLoss(cfg_m, student_tokenizer=stu_tok, teacher_tokenizer=tea_tok, device=device)
    trl_m = uld_m(s_full, t_full, s_labels, t_labels, s_input_ids, t_input_ids,
                  student_byte_offsets=s_off, teacher_byte_offsets=t_off)
    print(f"TRL hybrid distillation loss = {float(trl_m):.10f}  "
          f"(last_matched={float(uld_m.last_matched_loss):.8f}, last_unmatched={float(uld_m.last_unmatched_loss):.8f})")

    hv = ours_make_hybrid_vocab_mapping(stu_tok, tea_tok, device=device,
                                       student_vocab_dim=V, teacher_vocab_dim=V,
                                       student_real_vocab_size=V, teacher_real_vocab_size=V)
    # ours core with CLAMP+observed (models the TRL-faithful merge, to prove the MATH matches)
    total, matched, unmatched = ours_hybrid_loss_from_log_probs(
        stu_merged, tea_merged, hybrid_vocab=hv, beta=0.5, matched_weight=1.0, unmatched_weight=0.0
    )
    ours_m = matched.mean()  # mw=1,uw=0 -> only matched, mean over groups
    print(f"ours core matched-JSD mean-over-groups = {float(ours_m):.10f}   "
          f"(matched_count={hv['matched_count']})")
    print(f"|TRL - ours| = {abs(float(trl_m) - float(ours_m)):.3e}")

    # =====================================================================
    # ARM 3: gold-unmatched  (hybrid, beta=0.5, matched_w=0.0, unmatched_w=1.0)
    # =====================================================================
    _print_hdr("ARM 3  gold-unmatched : TRL hybrid (mw=0,uw=1) unmatched-ULD vs ours core")
    cfg_u = MockGOLDConfig(uld_use_hybrid_loss=True, beta=0.5,
                           uld_hybrid_matched_weight=0.0, uld_hybrid_unmatched_weight=1.0)
    uld_u = ULDLoss(cfg_u, student_tokenizer=stu_tok, teacher_tokenizer=tea_tok, device=device)
    trl_u = uld_u(s_full, t_full, s_labels, t_labels, s_input_ids, t_input_ids,
                  student_byte_offsets=s_off, teacher_byte_offsets=t_off)
    print(f"TRL hybrid distillation loss = {float(trl_u):.10f}  "
          f"(last_matched={float(uld_u.last_matched_loss):.8f}, last_unmatched={float(uld_u.last_unmatched_loss):.8f})")
    ours_u = unmatched.mean()
    print(f"ours core unmatched-ULD mean-over-groups = {float(ours_u):.10f}")
    print(f"|TRL - ours| = {abs(float(trl_u) - float(ours_u)):.3e}")

    # =====================================================================
    # SANITY: generalized_jsd_loss (:2398 anchor) direct values
    # =====================================================================
    _print_hdr("SANITY  generalized_jsd_loss (gold_trainer.py:2398) beta semantics")
    torch.manual_seed(1)
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    for beta in (0.0, 0.5, 1.0):
        v = generalized_jsd_loss(a, b, labels=None, beta=beta, temperature=1.0, reduction="sum")
        # cross-check the probs-path used by hybrid matched (logits_are_probs=True)
        pa = F.softmax(a, -1)
        pb = F.softmax(b, -1)
        vp = generalized_jsd_loss(pa, pb, labels=None, beta=beta, temperature=1.0, reduction="sum",
                                  logits_are_probs=True)
        vk = ours_generalized_jsd_from_probs(pa, pb, beta=beta).sum()
        print(f"beta={beta}: jsd(logits,sum)={float(v):.6f}  jsd(probs,sum)={float(vp):.6f}  "
              f"ours(probs,sum)={float(vk):.6f}  |probs-ours|={abs(float(vp)-float(vk)):.2e}")

    # =====================================================================
    # REAL TOKENIZERS: end-to-end teacher build + student offsets + alignment
    # =====================================================================
    _print_hdr("REAL TOKENIZERS  build_teacher_inputs_from_texts + cross-tokenizer alignment")
    try:
        from transformers import AutoTokenizer
        stu_path = os.environ.get("GOLD_ORACLE_STUDENT", "")
        tea_path = os.environ.get("GOLD_ORACLE_TEACHER", "")
        if not (stu_path and tea_path):
            raise RuntimeError(
                "set GOLD_ORACLE_STUDENT and GOLD_ORACLE_TEACHER to local HF checkpoints "
                "to run the real-tokenizer section"
            )
        stok = AutoTokenizer.from_pretrained(stu_path, trust_remote_code=True)
        ttok = AutoTokenizer.from_pretrained(tea_path, trust_remote_code=True)
        if ttok.pad_token is None:
            ttok.pad_token = ttok.eos_token
        prompt = "Solve: what is 2+2?"
        completion = " The answer is 4 because two plus two equals four."
        # teacher tensors from text
        t_ids, t_lab, t_am, t_boff = build_teacher_inputs_from_texts(ttok, [prompt], [completion])
        # student offsets from the SAME completion text (completion-relative), mirroring
        # tokenize_with_original_text (gold_trainer.py:2314-2326) for the completion part only
        [(s_ids_c, s_off_c)] = encode_with_byte_offsets(stok.backend_tokenizer, [completion], add_special_tokens=False)
        # teacher completion region: strip prompt+eos to get completion offsets
        t_lab0 = t_lab[0]
        tstart = int((t_lab0 != -100).nonzero()[0])
        tsize = int((t_lab0 != -100).sum()) - 1  # skip_teacher_eos
        t_off_c = t_boff[0, tstart:tstart + tsize].tolist()
        s_off_only = [tuple(x) for x in s_off_c]           # skip_student_eos: no eos appended here
        sgr, tgr = ULDLoss._align_by_byte_offsets(s_off_only, t_off_c)
        print(f"student completion: {len(s_ids_c)} tokens, teacher completion: {tsize} tokens")
        print(f"student offsets[:6]={s_off_only[:6]}")
        print(f"teacher offsets[:6]={t_off_c[:6]}")
        print(f"cross-tokenizer alignment produced {len(sgr)} shared byte-boundary groups")
        multi = [(len(a), len(b)) for a, b in zip(sgr, tgr) if len(a) > 1 or len(b) > 1]
        print(f"non-1:1 groups (student_len,teacher_len)={multi[:8]}  (total non-1:1={len(multi)})")
    except Exception as exc:  # noqa
        print(f"[skipped real-tokenizer smoke: {exc!r}]")

    # =====================================================================
    # REDUCTION DIVERGENCE: TRL per-sample-mean-then-batch-mean vs ours per_token pooled
    # =====================================================================
    _print_hdr("REDUCTION  batch=2 unequal groups: TRL per-sample-mean vs ours per_token pooled")
    # sample A: reuse the region above (3 groups). sample B: a SHORTER region (1 group).
    # Build a batch=2 by right-padding to equal seq len.
    V2, sA_logits, tA_logits, sA_ids, tA_ids, sA_off, tA_off = _build_synthetic_single(seed=7)
    # sample B: 1 student token, 1 teacher token (1 group)
    sB_ids = [4]
    tB_ids = [4]
    sB_off = [(0, 3)]
    tB_off = [(0, 3)]
    torch.manual_seed(99)
    sB_logits = torch.randn(1, V2) * 1.3
    tB_logits = torch.randn(1, V2) * 1.3

    def _one(V, sl, tl, sid, tid, soff, toff):
        return _make_batch_tensors(V, sl, tl, sid, tid, soff, toff)

    bA = _one(V2, sA_logits, tA_logits, sA_ids, tA_ids, sA_off, tA_off)
    bB = _one(V2, sB_logits, tB_logits, sB_ids, tB_ids, sB_off, tB_off)

    # assemble batch=2 with right padding on seq dim
    def _cat_batch(b1, b2):
        outs = []
        for x1, x2 in zip(b1, b2):
            L = max(x1.shape[1], x2.shape[1])
            def _pad(x):
                if x.shape[1] == L:
                    return x
                padw = L - x.shape[1]
                if x.dim() == 3:
                    p = torch.zeros(1, padw, x.shape[2], dtype=x.dtype)
                    return torch.cat([x, p], dim=1)
                elif x.dim() == 3 - 1 and x.shape[-1] == 2:  # offsets [1,seq,2]
                    p = torch.zeros(1, padw, 2, dtype=x.dtype)
                    return torch.cat([x, p], dim=1)
                else:
                    # labels/input_ids [1,seq]; pad labels with -100, ids with 0
                    pv = -100 if x.eq(-100).any() else 0
                    p = torch.full((1, padw), pv, dtype=x.dtype)
                    return torch.cat([x, p], dim=1)
            outs.append(torch.cat([_pad(x1), _pad(x2)], dim=0))
        return outs

    (s_full2, t_full2, s_labels2, t_labels2, s_ids2, t_ids2, s_off2, t_off2) = _cat_batch(bA, bB)
    # fix offsets tensor dim detection: offsets are [B,seq,2]; rebuild cleanly
    # (the helper above handles it, but ensure labels padded with -100)
    cfg_r = MockGOLDConfig(uld_use_hybrid_loss=False)
    uld_r = ULDLoss(cfg_r, student_tokenizer=stu_tok, teacher_tokenizer=tea_tok, device=device)
    trl_batch = uld_r(s_full2, t_full2, s_labels2, t_labels2, s_ids2, t_ids2,
                      student_byte_offsets=s_off2, teacher_byte_offsets=t_off2)

    # per-sample losses (what TRL averages)
    def _sample_loss(sl, tl, sid, tid, soff, toff):
        slp = F.log_softmax(sl, -1)
        tlp = F.log_softmax(tl, -1)
        g1, g2 = ours_align_by_byte_offsets(soff, toff)
        sm = ours_merge_log_probs_with_alignment_groups(slp, g1, sid, clamp_min_prob=1e-8)
        tm = ours_merge_log_probs_with_alignment_groups(tlp, g2, tid, clamp_min_prob=1e-8)
        pg = ours_sorted_l1_from_log_probs(sm, tm)
        return pg  # per-group L1 vector

    pgA = _sample_loss(sA_logits, tA_logits, sA_ids, tA_ids, sA_off, tA_off)
    pgB = _sample_loss(sB_logits, tB_logits, sB_ids, tB_ids, sB_off, tB_off)
    trl_reduction = 0.5 * (pgA.sum() / pgA.shape[0] + pgB.sum() / pgB.shape[0])
    ours_per_token = (pgA.sum() + pgB.sum()) / (pgA.shape[0] + pgB.shape[0])
    print(f"sample A groups={pgA.shape[0]}  sample B groups={pgB.shape[0]}")
    print(f"TRL ULDLoss(batch=2) [per-sample-mean-then-batch-mean] = {float(trl_batch):.10f}")
    print(f"   reconstructed per-sample-mean                       = {float(trl_reduction):.10f}")
    print(f"ours per_token pooled mean-over-all-groups              = {float(ours_per_token):.10f}")
    print(f"|TRL - ours_per_token| = {abs(float(trl_batch) - float(ours_per_token)):.3e}  "
          f"<-- DIVERGENCE unless ours uses --opd-loss-reduction per_sample")

    _print_hdr("DONE")


if __name__ == "__main__":
    main()
