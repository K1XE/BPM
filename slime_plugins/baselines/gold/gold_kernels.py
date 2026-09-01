"""Pure GOLD/ULD alignment and probability helpers for the GOLD baseline.

The byte-offset helpers intentionally mirror TRL's experimental GOLD ULD
contract: cross-tokenizer extended ULD is ByteLevel-BPE only, and offsets are
UTF-8 byte offsets in completion-relative coordinates.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce

from slime_plugins.bpm.loss.bpm_loss_utils import _mask_invalid_vocab_rows_


def _gold_is_byte_level_tokenizer(tokenizer) -> bool:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return False
    return "ByteLevel" in repr(getattr(backend, "pre_tokenizer", "")) or "ByteLevel" in repr(
        getattr(backend, "decoder", "")
    )


def _gold_require_byte_level_tokenizer(tokenizer, *, context: str) -> None:
    if not _gold_is_byte_level_tokenizer(tokenizer):
        raise NotImplementedError(
            "[OPD][gold] Cross-tokenizer GOLD/ULD extended alignment "
            "currently supports only ByteLevel BPE tokenizers, matching TRL "
            f"GOLD. The {context} tokenizer is not ByteLevel."
        )


def _gold_piece_byte_len(piece: str) -> int:
    """UTF-8 byte length in ByteLevel's byte-to-unicode token space."""
    return len(piece)


def _gold_bytes_to_unicode() -> dict[int, str]:
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


_GOLD_BYTE_LEVEL_DECODER = {ch: b for b, ch in _gold_bytes_to_unicode().items()}


def _gold_byte_level_piece_len(piece: str, text_bytes: bytes, start: int) -> int | None:
    piece_bytes = []
    for ch in piece:
        if ch not in _GOLD_BYTE_LEVEL_DECODER:
            return None
        piece_bytes.append(_GOLD_BYTE_LEVEL_DECODER[ch])
    if piece_bytes and piece_bytes[0] == ord(" ") and text_bytes[start : start + 1] != b" ":
        piece_bytes = piece_bytes[1:]
    return len(piece_bytes)


def _gold_split_repeated_byte_offsets(
    byte_offsets: list[tuple[int, int]],
    tokens: list[str],
) -> list[tuple[int, int]]:
    normalized = list(byte_offsets)
    i = 0
    while i < len(byte_offsets):
        j = i + 1
        while j < len(byte_offsets) and byte_offsets[j] == byte_offsets[i]:
            j += 1

        if j - i > 1:
            start, end = byte_offsets[i]
            piece_lengths = [_gold_piece_byte_len(token) for token in tokens[i:j]]
            if sum(piece_lengths) == end - start:
                cursor = start
                for offset_idx, length in enumerate(piece_lengths, start=i):
                    normalized[offset_idx] = (cursor, cursor + length)
                    cursor += length

        i = j
    return normalized


def _gold_normalize_byte_offsets(
    byte_offsets: list[tuple[int, int]],
    tokens: list[str],
    text_bytes: bytes,
) -> list[tuple[int, int]]:
    """Normalize fast-tokenizer character spans to monotonic byte spans.

    This ports the important edge handling from TRL GOLD's
    ``encode_with_byte_offsets``: repeated spans are split across byte-level
    pieces, and overlapping/trimmed ByteLevel offsets are corrected when the
    token piece length proves the intended byte width.
    """
    byte_offsets = _gold_split_repeated_byte_offsets(byte_offsets, tokens)
    normalized: list[tuple[int, int]] = []
    cursor = 0

    for idx, (start, end) in enumerate(byte_offsets):
        if start == end:
            normalized.append((cursor, cursor))
            continue

        piece_len = _gold_byte_level_piece_len(tokens[idx], text_bytes, start)
        next_start = byte_offsets[idx + 1][0] if idx + 1 < len(byte_offsets) else None
        has_overlap = start < cursor or (next_start is not None and next_start < end)

        if piece_len is not None and (has_overlap or piece_len == end - start):
            candidate_start = max(start, cursor)
            candidate_end = candidate_start + piece_len
            if candidate_end <= end:
                start = candidate_start
                end = candidate_end

        if start < cursor or end < start:
            raise ValueError(
                "[OPD][gold] tokenizer produced overlapping byte offsets "
                "that could not be normalized. Cross-tokenizer ULD requires "
                "monotonic byte offsets."
            )

        normalized.append((start, end))
        cursor = end
    return normalized


def _gold_encode_with_byte_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Encode text and return token ids plus UTF-8 byte offsets.

    TRL GOLD supports cross-tokenizer extended ULD only for ByteLevel BPE
    tokenizers.  Keep that fail-fast contract here; a decoded-text fallback can
    produce plausible-looking but shifted offsets and silently train the wrong
    token groups.
    """
    text = text or ""
    _gold_require_byte_level_tokenizer(tokenizer, context="GOLD")
    backend = tokenizer.backend_tokenizer
    enc = backend.encode(text, add_special_tokens=False)
    ids = [int(x) for x in enc.ids]
    char_to_byte = [0]
    for ch in text:
        char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))
    offsets: list[tuple[int, int]] = []
    for start_char, end_char in enc.offsets:
        offsets.append((char_to_byte[start_char], char_to_byte[end_char]))
    offsets = _gold_normalize_byte_offsets(offsets, list(enc.tokens), text.encode("utf-8"))
    return ids, offsets


def _gold_offsets_from_actual_token_ids(
    tokenizer,
    token_ids: list[int],
    text: str,
    *,
    eos_token_id: int | None = None,
    prefer_encoded_offsets: bool = False,
) -> list[tuple[int, int]]:
    """Return completion-relative byte offsets for already-sampled token IDs.

    GOLD/ULD must align byte offsets with the exact token rows used by the loss.
    Re-tokenizing ``text`` alone can merge across the prompt/completion boundary
    and produce token IDs that no longer match the in-context logits.  This
    mirrors TRL GOLD's on-policy path: offsets are derived from the sampled token
    pieces themselves, with EOS represented as a zero-width span at the end.
    """
    text = text or ""
    text_bytes = text.encode("utf-8")
    content_len = len(text_bytes)
    ids = [int(tid) for tid in token_ids]
    if not ids:
        return []

    non_eos_len = len(ids)
    if eos_token_id is not None and ids and ids[-1] == int(eos_token_id):
        non_eos_len -= 1

    _gold_require_byte_level_tokenizer(tokenizer, context="GOLD")

    if prefer_encoded_offsets:
        encoded_ids, encoded_offsets = _gold_encode_with_byte_offsets(tokenizer, text)
        if encoded_ids != ids[:non_eos_len]:
            raise ValueError(
                "[OPD][gold] encoded-text GOLD offsets do not match "
                "the actual teacher token rows. This usually means the teacher "
                "prefill did not tokenize prompt and completion with GOLD's "
                "separate-completion boundary contract. "
                f"encoded_prefix={encoded_ids[:8]}, actual_prefix={ids[:8]}"
            )
        offsets = list(encoded_offsets)
        if non_eos_len < len(ids):
            offsets.append((content_len, content_len))
        return offsets

    # TRL GOLD on-policy path: byte offsets come straight from the sampled token
    # pieces' ByteLevel byte widths, accumulated.  No tokenizer.decode() round
    # trip and no piece-bytes==text-bytes assertion.  The student's sampled IDs
    # can form invalid/truncated UTF-8 (a multi-byte char split mid-codepoint),
    # which decode() would render as U+FFFD and inflate the text byte count;
    # reconstructing from the pieces themselves stays byte-exact and matches the
    # teacher side (built the same way, also from actual token IDs), so the
    # cross-tokenizer aligner re-syncs at shared byte boundaries.  ``text`` is no
    # longer used here (only the off-policy encoded-offsets path consumes it),
    # matching trl ULDLoss._maybe_add_completion_byte_offsets.
    pieces = tokenizer.convert_ids_to_tokens(ids[:non_eos_len])
    piece_lens = [_gold_piece_byte_len(piece) for piece in pieces]
    offsets = []
    cursor = 0
    for length in piece_lens:
        offsets.append((cursor, cursor + length))
        cursor += length
    if non_eos_len < len(ids):
        offsets.append((cursor, cursor))  # zero-width EOS span at end of bytes
    return offsets


def _gold_trim_ids_and_offsets_for_answer(
    tokenizer,
    label_ids: list[int],
    text: str,
    *,
    skip_eos: bool,
    eos_token_id: int | None,
    prefer_encoded_offsets: bool = False,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Trim optional EOS and build offsets for GOLD answer rows."""
    ids = [int(tid) for tid in label_ids]
    if skip_eos and ids:
        ids = ids[:-1]
    offsets = _gold_offsets_from_actual_token_ids(
        tokenizer,
        ids,
        text,
        eos_token_id=eos_token_id,
        prefer_encoded_offsets=prefer_encoded_offsets,
    )
    if len(offsets) != len(ids):
        raise ValueError(
            "[OPD][gold] GOLD byte-offset contract violated: "
            f"ids={len(ids)} offsets={len(offsets)}"
        )
    return ids, offsets

def _gold_align_by_byte_offsets(
    student_offsets: list[tuple[int, int]],
    teacher_offsets: list[tuple[int, int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """TRL GOLD ULDLoss._align_by_byte_offsets."""
    s_groups: list[list[int]] = []
    t_groups: list[list[int]] = []
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

def _gold_merge_log_probs_with_alignment_groups(
    log_probs: torch.Tensor,
    alignment_groups: list[list[int]],
    token_ids: list[int],
    *,
    clamp_min_prob: float | None = None,
    bayesian: bool = False,
) -> torch.Tensor:
    """Merge per-position log-probability rows using TRL GOLD's group rule.

    GOLD multiplies probabilities for multi-token groups.  We do it in log
    space for numerical stability.  Two strategies match TRL
    ``uld_token_merge_strategy``:

      observed (default, ``bayesian=False``):
        base = group[0]'s FULL distribution, scalar tokens = group[1:]
        log P_merged(y) = log P_first(y) + Σ_{j in group[1:]} log P(actual_token_j)

      bayesian (``bayesian=True``, gated on --gold-trl-faithful):
        base = group[-1]'s FULL distribution, scalar tokens = group[:-1]
        log P_merged(y) = log P_last(y) + Σ_{j in group[:-1]} log P(actual_token_j)

    ``clamp_min_prob`` (TRL GOLD default 1e-8, gated on --gold-trl-faithful) clamps
    each scalar conditional PROBABILITY to >= clamp_min_prob before the product, i.e.
    each scalar log-prob to >= log(clamp_min_prob).  The base-position marginal is
    never clamped (matches TRL ``_merge_probabilities_with_alignment_groups``).
    """
    if not alignment_groups:
        return log_probs[:0]
    log_clamp = math.log(clamp_min_prob) if clamp_min_prob is not None else None
    merged_rows: list[torch.Tensor] = []
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
                if token_id < 0 or token_id >= log_probs.shape[-1]:
                    raise ValueError(
                        f"[OPD][gold] token_id={token_id} outside vocab shard/full size {log_probs.shape[-1]}"
                    )
                tail_term = log_probs[int(idx), token_id]
                if log_clamp is not None:
                    tail_term = tail_term.clamp_min(log_clamp)
                cond = cond + tail_term
            merged_rows.append(log_probs[base_pos] + cond)
        else:
            merged_rows.append(log_probs.new_full((log_probs.shape[-1],), float("-inf")))
    return torch.stack(merged_rows, dim=0)

def _gold_merge_log_probs_from_first_rows_and_tail_scalars(
    first_row_log_probs: torch.Tensor,
    alignment_groups: list[list[int]],
    tail_label_log_probs_full: torch.Tensor,
    *,
    clamp_min_prob: float | None = None,
    bayesian: bool = False,
) -> torch.Tensor:
    """Merge GOLD groups when only the base row has a full distribution.

    This is the CP-safe student-side variant of
    ``_gold_merge_log_probs_with_alignment_groups``.  GOLD's merged
    distribution for a multi-token group is, for the two TRL strategies:

      observed (default, ``bayesian=False``): base = group[0], scalars = group[1:]
        log P_merged(v) = log P_first(v) + Σ_{j in group[1:]} log P_scalar(actual_token_j)

      bayesian (``bayesian=True``, gated on --gold-trl-faithful): base = group[-1], scalars = group[:-1]
        log P_merged(v) = log P_last(v) + Σ_{j in group[:-1]} log P_scalar(actual_token_j)

    ``first_row_log_probs[row_idx]`` is the FULL distribution materialized for the
    group's BASE position (the caller must materialize group[0] in observed mode
    and group[-1] in bayesian mode).  The scalar positions' label log-probs come
    from ``tail_label_log_probs_full`` (indexed by GLOBAL response position).

    Under context parallelism, the base student row and the scalar student rows
    can live on different CP ranks.  We therefore keep the expensive full-vocab
    row only on the rank that owns the base token, and use a CP-gathered vector
    of scalar label log-probs for the other tokens.  This preserves exact GOLD
    semantics without all-gathering ``[response_len, vocab]`` tensors.
    """
    if not alignment_groups:
        return first_row_log_probs[:0]
    if first_row_log_probs.shape[0] != len(alignment_groups):
        raise ValueError(
            "[OPD][gold] first-row/group mismatch while merging GOLD groups: "
            f"first_rows={first_row_log_probs.shape[0]} groups={len(alignment_groups)}"
        )

    log_clamp = math.log(clamp_min_prob) if clamp_min_prob is not None else None
    merged_rows: list[torch.Tensor] = []
    for row_idx, group in enumerate(alignment_groups):
        if len(group) <= 1:
            merged_rows.append(first_row_log_probs[row_idx])
            continue
        scalar_positions = group[:-1] if bayesian else group[1:]
        tail_positions = torch.tensor(
            [int(pos) for pos in scalar_positions],
            dtype=torch.long,
            device=tail_label_log_probs_full.device,
        )
        tail_terms = tail_label_log_probs_full.index_select(0, tail_positions)
        # TRL GOLD clamps each scalar conditional prob to >= clamp_min_prob (log-prob >= log clamp)
        # BEFORE the chain-rule product; the base-position marginal is never clamped.
        if log_clamp is not None:
            tail_terms = tail_terms.clamp_min(log_clamp)
        cond = tail_terms.sum()
        merged_rows.append(first_row_log_probs[row_idx] + cond.to(device=first_row_log_probs.device))
    return torch.stack(merged_rows, dim=0)



def _gold_global_log_softmax_denominator(
    logits_shard: torch.Tensor,
    *,
    temperature: float,
    real_vocab_size: int,
    vocab_start: int,
    tp_size: int,
    tp_group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked/scaled logits plus global vocab-parallel log-softmax terms."""
    logits_shard = _mask_invalid_vocab_rows_(
        logits_shard.float() / float(temperature),
        real_vocab_size=int(real_vocab_size),
        vocab_start=int(vocab_start),
    )
    with torch.no_grad():
        max_v = logits_shard.detach().max(dim=-1, keepdim=True).values
        if int(tp_size) > 1:
            torch.distributed.all_reduce(max_v, op=torch.distributed.ReduceOp.MAX, group=tp_group)

    exp_sum_local = (logits_shard - max_v).exp().sum(dim=-1, keepdim=True)
    if int(tp_size) > 1:
        exp_sum = differentiable_all_reduce(exp_sum_local, group=tp_group)
    else:
        exp_sum = exp_sum_local
    return logits_shard, max_v, exp_sum.log()


def _gold_full_log_probs_from_vocab_parallel_logits(
    logits_shard: torch.Tensor,
    *,
    temperature: float,
    real_vocab_size: int,
    vocab_start: int,
    tp_size: int,
    tp_group,
) -> torch.Tensor:
    """Exact full-vocabulary log-probs from TP vocab-parallel logits.

    GOLD/ULD sorts whole probability vectors, which is not decomposable by vocab
    shard.  For TP>1 we therefore compute the exact global softmax denominator,
    then all-gather the per-shard log-probability columns in global vocab order.
    The gather is differentiable so student gradients route back to the owning
    TP shard; teacher callers should detach/no_grad before calling if desired.
    """
    logits_shard, max_v, log_sum_exp = _gold_global_log_softmax_denominator(
        logits_shard,
        temperature=temperature,
        real_vocab_size=real_vocab_size,
        vocab_start=vocab_start,
        tp_size=tp_size,
        tp_group=tp_group,
    )
    log_probs_shard = logits_shard - max_v - log_sum_exp
    if int(tp_size) <= 1:
        return log_probs_shard
    gathered = differentiable_all_gather(log_probs_shard.contiguous(), group=tp_group)
    return torch.cat(tuple(gathered), dim=-1)


def _gold_sample_log_probs_from_vocab_parallel_logits(
    logits_shard: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float,
    real_vocab_size: int,
    vocab_start: int,
    tp_size: int,
    tp_group,
) -> torch.Tensor:
    """Exact sampled-token log-probs from TP vocab-parallel logits.

    Used for GOLD multi-token group tails.  It preserves the full-vocab softmax
    denominator while gathering only the selected actual-token numerators.
    """
    logits_shard, max_v, log_sum_exp = _gold_global_log_softmax_denominator(
        logits_shard,
        temperature=temperature,
        real_vocab_size=real_vocab_size,
        vocab_start=vocab_start,
        tp_size=tp_size,
        tp_group=tp_group,
    )
    if token_ids.device != logits_shard.device:
        token_ids = token_ids.to(device=logits_shard.device)
    token_ids = token_ids.long()
    if token_ids.numel() == 0:
        return logits_shard.new_empty((0,))
    bad = (token_ids < 0) | (token_ids >= int(real_vocab_size))
    if bool(bad.any().item()):
        preview = token_ids[bad][:8].detach().cpu().tolist()
        raise ValueError(f"[OPD][gold] sampled token id outside vocab: {preview}, vocab={int(real_vocab_size)}")

    shard_vocab = int(logits_shard.shape[-1])
    local_idx = (token_ids - int(vocab_start)).clamp(0, max(shard_vocab - 1, 0))
    in_shard = (token_ids >= int(vocab_start)) & (token_ids < int(vocab_start) + shard_vocab)
    local_logits = logits_shard.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)
    local_logits = torch.where(in_shard, local_logits, torch.zeros_like(local_logits))
    if int(tp_size) > 1:
        sample_logits = differentiable_all_reduce(local_logits, group=tp_group)
    else:
        sample_logits = local_logits
    return sample_logits - max_v.squeeze(-1) - log_sum_exp.squeeze(-1)


def _gold_sparse_tail_label_log_probs_tp(
    logits_shard: torch.Tensor,
    *,
    label_ids_full: list[int],
    local_pos_by_global: dict[int, int],
    tail_global_positions: list[int],
    temperature: float,
    real_vocab_size: int,
    vocab_start: int,
    tp_size: int,
    tp_group,
) -> torch.Tensor:
    """TP-aware sparse tail label log-probs for GOLD chain-rule groups."""
    out = logits_shard.new_zeros((int(logits_shard.shape[0]),), dtype=torch.float32)
    if not tail_global_positions or logits_shard.numel() == 0:
        return out
    rows: list[int] = []
    labels: list[int] = []
    for global_pos in tail_global_positions:
        local_row = local_pos_by_global.get(int(global_pos))
        if local_row is None or local_row >= int(logits_shard.shape[0]) or global_pos >= len(label_ids_full):
            continue
        rows.append(int(local_row))
        labels.append(int(label_ids_full[int(global_pos)]))
    if not rows:
        return out

    rows_t = torch.tensor(rows, dtype=torch.long, device=logits_shard.device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=logits_shard.device)
    vals = _gold_sample_log_probs_from_vocab_parallel_logits(
        logits_shard.index_select(0, rows_t),
        labels_t,
        temperature=temperature,
        real_vocab_size=real_vocab_size,
        vocab_start=vocab_start,
        tp_size=tp_size,
        tp_group=tp_group,
    )
    return out.scatter(0, rows_t, vals)



def _gold_sorted_l1_from_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute GOLD/ULD sorted-probability L1 from pre-normalized log-probs."""
    student_sorted = student_log_probs.exp().sort(dim=-1, descending=True).values
    teacher_sorted = teacher_log_probs.exp().sort(dim=-1, descending=True).values
    max_vocab = max(student_sorted.shape[-1], teacher_sorted.shape[-1])
    if student_sorted.shape[-1] < max_vocab:
        student_sorted = F.pad(student_sorted, (0, max_vocab - student_sorted.shape[-1]))
    if teacher_sorted.shape[-1] < max_vocab:
        teacher_sorted = F.pad(teacher_sorted, (0, max_vocab - teacher_sorted.shape[-1]))
    return F.l1_loss(student_sorted, teacher_sorted, reduction="none").sum(dim=-1)


def _gold_make_hybrid_vocab_mapping(
    student_tokenizer,
    teacher_tokenizer,
    *,
    device: torch.device,
    student_vocab_dim: int,
    teacher_vocab_dim: int,
    student_real_vocab_size: int,
    teacher_real_vocab_size: int,
) -> dict[str, torch.Tensor | int]:
    """Build tensors for GOLD hybrid matched-JSD + unmatched-ULD loss."""
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    pairs: list[tuple[int, int]] = []
    for token, teacher_id_raw in teacher_vocab.items():
        student_id_raw = student_vocab.get(token)
        if student_id_raw is None:
            continue
        teacher_id = int(teacher_id_raw)
        student_id = int(student_id_raw)
        if (
            0 <= teacher_id < int(teacher_real_vocab_size)
            and 0 <= student_id < int(student_real_vocab_size)
            and teacher_id < int(teacher_vocab_dim)
            and student_id < int(student_vocab_dim)
        ):
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
        "teacher_matched_ids": teacher_matched_ids,
        "student_matched_ids": student_matched_ids,
        "teacher_unmatched_mask": teacher_unmatched_mask,
        "student_unmatched_mask": student_unmatched_mask,
        "teacher_vocab_size": int(teacher_real_vocab_size),
        "matched_count": int(teacher_matched_ids.numel()),
    }


def _gold_generalized_jsd_from_probs(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Per-row generalized JSD used by TRL GOLD hybrid matched-token loss."""
    student_log_probs = torch.log(student_probs.clamp_min(1e-8))
    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-8))
    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-beta_t),
                    teacher_log_probs + torch.log(beta_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1.0 - beta_t) * kl_student
    return jsd.sum(dim=-1)


def _gold_hybrid_loss_from_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    hybrid_vocab: dict[str, torch.Tensor | int],
    beta: float,
    matched_weight: float | None,
    unmatched_weight: float | None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row GOLD hybrid matched-JSD + unmatched sorted-L1 loss.

    With ``return_components=True`` also returns the raw (UNWEIGHTED) per-row
    ``matched_loss`` (generalized JSD on shared-vocab tokens) and
    ``unmatched_loss`` (sorted-L1/ULD on the non-overlapping tokens), mirroring
    TRL GOLD's ``last_matched_loss``/``last_unmatched_loss`` diagnostics so the
    two hybrid components can be logged independently.
    """
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    device = student_probs.device
    rows = int(student_probs.shape[0])
    matched_loss = torch.zeros((rows,), dtype=torch.float32, device=device)
    unmatched_loss = torch.zeros((rows,), dtype=torch.float32, device=device)

    teacher_matched_ids = hybrid_vocab["teacher_matched_ids"]
    student_matched_ids = hybrid_vocab["student_matched_ids"]
    if isinstance(teacher_matched_ids, torch.Tensor) and teacher_matched_ids.numel() > 0:
        matched_loss = _gold_generalized_jsd_from_probs(
            student_probs.index_select(-1, student_matched_ids.to(device=student_probs.device)),
            teacher_probs.index_select(-1, teacher_matched_ids.to(device=teacher_probs.device)),
            beta=beta,
        )

    student_unmatched_mask = hybrid_vocab["student_unmatched_mask"]
    teacher_unmatched_mask = hybrid_vocab["teacher_unmatched_mask"]
    if isinstance(student_unmatched_mask, torch.Tensor) and isinstance(teacher_unmatched_mask, torch.Tensor):
        student_unmatched = student_probs[:, student_unmatched_mask.to(device=student_probs.device)]
        teacher_unmatched = teacher_probs[:, teacher_unmatched_mask.to(device=teacher_probs.device)]
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
        hybrid_matched_weight = matched_count / teacher_vocab_size
        hybrid_unmatched_weight = 1.0 - hybrid_matched_weight
    else:
        hybrid_matched_weight = float(matched_weight)
        hybrid_unmatched_weight = float(unmatched_weight)
    total_loss = hybrid_matched_weight * matched_loss + hybrid_unmatched_weight * unmatched_loss
    if return_components:
        return total_loss, matched_loss, unmatched_loss
    return total_loss
