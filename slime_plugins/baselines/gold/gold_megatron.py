"""GOLD/ULD cross-tokenizer OPD loss backend."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import numpy as np
import torch
from megatron.core import mpu

from slime.utils.types import RolloutBatch

from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, get_logits_and_tokens_offset_with_cp
from slime_plugins.bpm.backend.loss_helpers import get_responses
from .gold_kernels import (
    _gold_align_by_byte_offsets,
    _gold_full_log_probs_from_vocab_parallel_logits,
    _gold_hybrid_loss_from_log_probs,
    _gold_is_byte_level_tokenizer,
    _gold_make_hybrid_vocab_mapping,
    _gold_merge_log_probs_from_first_rows_and_tail_scalars,
    _gold_merge_log_probs_with_alignment_groups,
    _gold_sample_log_probs_from_vocab_parallel_logits,
    _gold_sparse_tail_label_log_probs_tp,
    _gold_sorted_l1_from_log_probs,
    _gold_trim_ids_and_offsets_for_answer,
)
from slime_plugins.bpm.backend.projection import (
    _get_bpm_teacher_lm_head as _get_gold_teacher_lm_head,
)
from slime_plugins.bpm.backend.vocab import (
    _get_bpm_local_response_indices as _get_gold_local_response_indices,
    _get_bpm_student_tokenizer as _get_gold_student_tokenizer,
    _get_bpm_teacher_tokenizer as _get_gold_teacher_tokenizer,
)
from slime_plugins.bpm.loss.bpm_loss_utils import (
    _any_cp_rank_failed,
    _count_cp_unique_samples_with_alignment,
    _raise_if_any_dp_cp_rank_failed,
    _reduce_cp_float_counts,
    _reduce_dp_cp_float_counts,
)


def gold_core_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    sum_of_sample: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GOLD baseline: TRL GOLD/ULD sorted-probability cross-tokenizer loss.

    This keeps the same rollout/teacher-hidden-state plumbing as the other
    cross-tokenizer backends, but the mismatch is handled by GOLD's ULD:
    align response tokens by UTF-8 byte offsets, merge split-token groups, sort
    both vocab distributions, then apply L1 distance.  TP>1 is supported: the
    full-vocab probabilities GOLD sorted-L1 requires are reconstructed exactly
    from vocab-parallel logits via a differentiable all-gather (see
    ``_gold_full_log_probs_from_vocab_parallel_logits``), so the method is not
    approximated shard-locally.  CP is supported by assigning each variable-width
    GOLD group to the rank that owns the first student token and CP-gathering
    only scalar tail-token log-probs, never full ``[response_len, vocab]``
    distributions.
    """
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()
    use_hybrid_loss = bool(getattr(args, "gold_use_hybrid_loss", False))
    # TRL GOLD faithfulness, gated on the gold backend AND --gold-trl-faithful.  This applies
    # TRL's per-tail 1e-8 clamp on the merged conditional probabilities, (via teacher_request) TRL's
    # Qwen-rendered teacher prompt, and TRL's uld_token_merge_strategy semantics selected by
    # --gold-uld-token-merge-strategy (same name/values/DEFAULT as TRL gold_config):
    #   "observed" (TRL DEFAULT): UNSHIFTED pairing (distribution-at-pos <-> token-AT-pos, TRL's
    #       student_logits[start:start+size]) + FIRST-position merge base + later-token scalars.
    #       CP-safe via logits_shift=0 offsets (see get_logits_and_tokens_offset_with_cp).
    #   "bayesian" (TRL PR #5905 option): NEXT-TOKEN-SHIFTED pairing (slime's historical shift)
    #       + LAST-position merge base + earlier-token scalars.
    # --gold-trl-faithful OFF keeps slime's historical hybrid (shifted + first-pos + no clamp).
    gold_trl_faithful = bool(getattr(args, "gold_trl_faithful", False)) and (
        getattr(args, "opd_backend", "") == "gold"
    )
    gold_merge_clamp_min_prob = 1e-8 if gold_trl_faithful else None
    gold_merge_strategy = str(getattr(args, "gold_uld_token_merge_strategy", "observed") or "observed")
    gold_bayesian = gold_trl_faithful and gold_merge_strategy == "bayesian"
    gold_observed_unshift = gold_trl_faithful and gold_merge_strategy == "observed"

    teacher_hidden_states_list = batch.get("teacher_hidden_states")
    teacher_token_ids_list = batch.get("teacher_token_ids")
    if teacher_hidden_states_list is None or teacher_token_ids_list is None:
        raise ValueError("gold_opd_loss requires teacher_hidden_states and teacher_token_ids")

    lm_head = _get_gold_teacher_lm_head(args)
    teacher_tokenizer = _get_gold_teacher_tokenizer(args)
    student_tokenizer = _get_gold_student_tokenizer(args)
    device = logits.device
    student_real_vocab_size = len(student_tokenizer.get_vocab())
    teacher_real_vocab_size = int(getattr(lm_head, "_opd_vocab_size", len(teacher_tokenizer.get_vocab())))
    student_shard_vocab = int(logits.shape[-1])
    teacher_shard_vocab = int(lm_head.weight.shape[0])
    student_vocab_start = tp_rank * student_shard_vocab
    teacher_vocab_start = int(getattr(lm_head, "_opd_vocab_start", tp_rank * teacher_shard_vocab))

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    loss_masks = batch["loss_masks"]
    unconcat_tokens = batch.get("unconcat_tokens")
    if unconcat_tokens is None:
        raise ValueError("gold_opd_loss requires unconcat_tokens")

    stu_logits_list: list[torch.Tensor] = []
    stu_label_chunks: list[torch.Tensor] = []
    if gold_observed_unshift:
        # TRL GOLD "observed" (TRL DEFAULT): pair the student distribution AT a response position
        # with the token AT that position (no next-token shift), matching TRL's
        # student_logits[start:start+size] paired with student_input_ids[start:...].  get_responses
        # returns the SHIFTED window logits[start-1:end-1]; here we take the UNSHIFTED window
        # logits[start:end].  CP-AWARE: with cp_size>1 the sequence is zigzag-sharded into 2 chunks
        # per rank; we mirror get_responses' CP branch with logits_shift=0 offsets (window
        # [prompt_len, total_len) instead of [prompt_len-1, total_len-1); tokens = logits + 0).
        assert logits.size(0) == 1, f"{logits.shape}"
        assert logits.dtype == torch.float32, f"{logits.dtype}"
        _squeezed = logits.squeeze(0)
        _end = 0
        for tokens, total_length, response_length in zip(
            unconcat_tokens, total_lengths, response_lengths, strict=False
        ):
            total_length = int(total_length)
            response_length = int(response_length)
            if cp_size == 1:
                _end += total_length
                _start = _end - response_length
                stu_logits_list.append(_squeezed[_start:_end])
                stu_label_chunks.append(tokens[-response_length:])
            else:
                chunk_sz, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                    total_length, response_length, logits_shift=0
                )
                logits_0 = _squeezed[_end : _end + chunk_sz]
                logits_1 = _squeezed[_end + chunk_sz : _end + 2 * chunk_sz]
                _end += 2 * chunk_sz
                logits_0 = logits_0[
                    logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]
                ]
                tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]
                logits_1 = logits_1[
                    logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]
                ]
                tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]
                assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
                assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"
                stu_logits_list.append(torch.cat([logits_0, logits_1], dim=0))
                stu_label_chunks.append(torch.cat([tokens_0, tokens_1], dim=0))
    else:
        for logits_chunk, tokens_chunk in get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        ):
            stu_logits_list.append(logits_chunk)
            stu_label_chunks.append(tokens_chunk)

    use_extended = bool(getattr(args, "gold_use_extended_uld", True))
    skip_student_eos = bool(getattr(args, "gold_skip_student_eos", True))
    skip_teacher_eos = bool(getattr(args, "gold_skip_teacher_eos", True))
    stu_temp = float(getattr(args, "gold_student_temperature", 1.0) or 1.0)
    tea_temp = float(getattr(args, "gold_teacher_temperature", 1.0) or 1.0)
    ce_weight = float(getattr(args, "gold_ce_weight", 0.0) or 0.0)
    distill_weight = float(getattr(args, "gold_distillation_weight", 1.0))
    chunk_size = max(1, int(getattr(args, "gold_chunk_size", 32) or 32))
    hybrid_beta = float(getattr(args, "gold_beta", 0.5))
    hybrid_matched_weight = getattr(args, "gold_hybrid_matched_weight", None)
    hybrid_unmatched_weight = getattr(args, "gold_hybrid_unmatched_weight", None)
    teacher_eos_id = teacher_tokenizer.eos_token_id
    if teacher_eos_id is None:
        raise ValueError("teacher tokenizer eos_token_id is required for gold")
    if bool(use_extended):
        if not _gold_is_byte_level_tokenizer(student_tokenizer):
            raise NotImplementedError(
                "[OPD][gold] --gold-use-extended-uld requires a ByteLevel "
                "student tokenizer, matching TRL GOLD's cross-tokenizer ULD contract."
            )
        if not _gold_is_byte_level_tokenizer(teacher_tokenizer):
            raise NotImplementedError(
                "[OPD][gold] --gold-use-extended-uld requires a ByteLevel "
                "teacher tokenizer, matching TRL GOLD's cross-tokenizer ULD contract."
            )
    hybrid_vocab = None
    if use_hybrid_loss:
        if not (0.0 <= hybrid_beta <= 1.0):
            raise ValueError("--gold-beta must be in [0, 1] when --gold-use-hybrid-loss is enabled")
        if (hybrid_matched_weight is None) != (hybrid_unmatched_weight is None):
            raise ValueError(
                "--gold-hybrid-matched-weight and --gold-hybrid-unmatched-weight "
                "must both be unset or both be set."
            )
        if hybrid_matched_weight is not None and (
            float(hybrid_matched_weight) < 0.0 or float(hybrid_unmatched_weight) < 0.0
        ):
            raise ValueError("GOLD hybrid weights must be non-negative")
        # The hybrid vocab mapping is IMMUTABLE (depends only on the two tokenizers + vocab dims),
        # but building it scans the full ~151k teacher vocab + ~143k dict lookups + a sort + CUDA
        # mask alloc (~100ms). It was rebuilt every microbatch -> pure per-step CPU tax. Cache it on
        # args, keyed by device so a device move rebuilds.
        _hybrid_cache_key = str(device)
        if getattr(args, "_gold_hybrid_vocab_key", None) == _hybrid_cache_key:
            hybrid_vocab = args._gold_hybrid_vocab
        else:
            hybrid_vocab = _gold_make_hybrid_vocab_mapping(
                student_tokenizer,
                teacher_tokenizer,
                device=device,
                student_vocab_dim=int(getattr(args, "padded_vocab_size", student_shard_vocab * tp_size)),
                teacher_vocab_dim=int(getattr(lm_head, "_opd_padded_vocab_size", teacher_shard_vocab * tp_size)),
                student_real_vocab_size=student_real_vocab_size,
                teacher_real_vocab_size=teacher_real_vocab_size,
            )
            args._gold_hybrid_vocab = hybrid_vocab
            args._gold_hybrid_vocab_key = _hybrid_cache_key

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_distill = torch.tensor(0.0, dtype=torch.float32, device=device)
    # Raw (unweighted) hybrid components, accumulated only when hybrid is on, so
    # the JSD (matched) and ULD (unmatched) terms can be logged separately like
    # TRL GOLD's matched_loss/unmatched_loss.
    total_matched = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_unmatched = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_aligned = 0.0
    total_label_tokens = 0.0
    total_groups = 0.0
    sample_alignment_flags: list[float] = []
    skipped_none_hidden = 0
    skipped_short_hidden = 0
    skipped_empty_gold_region = 0
    skipped_no_gold_groups = 0
    total_ce = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_ce_tokens = 0.0
    total_tea_entropy = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_stu_entropy = torch.tensor(0.0, dtype=torch.float32, device=device)
    fatal_gold_msg: str | None = None
    tp_loss_scale = 1.0 / float(max(tp_size, 1))

    is_logging_rank = (
        mpu.is_pipeline_last_stage()
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )
    if is_logging_rank and not getattr(args, "_gold_config_logged", False):
        _logger.info(
            f"[OPD][gold] extended_uld={use_extended} hybrid={getattr(args, 'gold_use_hybrid_loss', False)} "
            f"gold_tp_exact={tp_size > 1} chunk_size={chunk_size} "
            f"student_vocab={student_real_vocab_size}/{student_shard_vocab * tp_size} "
            f"teacher_vocab={teacher_real_vocab_size}/{teacher_shard_vocab * tp_size} "
            f"stu_temp={stu_temp} tea_temp={tea_temp} ce_weight={ce_weight} "
            f"distill_weight={distill_weight} skip_eos=({skip_student_eos},{skip_teacher_eos}) "
            f"hybrid_beta={hybrid_beta} tp={tp_size} cp={cp_size}"
        )
        args._gold_config_logged = True

    def _to_int_list(x) -> list[int]:
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().tolist()]
        return [int(v) for v in x]

    # --opd-loss-reduction per_sample (matches TRL GOLD per-sample-mean-then-batch-mean):
    # accumulate each ORIGINAL sample's CP-LOCAL distill SUM (grad-carrying) and its aligned-group
    # count, keyed by sample index i.  Skipped samples are simply absent from the dicts (-> 0), so
    # we never touch the 8 skip paths and per_token/per_rank stay bit-identical (dicts stay empty).
    track_per_sample = getattr(args, "opd_loss_reduction", "per_token") == "per_sample"
    per_sample_distill: dict = {}
    per_sample_aligned: dict = {}
    if track_per_sample:
        # Always define the per-sample normalizer so the dispatcher's num_tokens override sees a
        # value even if this microbatch bails on an early zero-loss return below.  The main
        # per_sample branch overwrites it with the real sample count.
        args._opd_per_sample_normalizer = 0.0

    for i in range(len(response_lengths)):
        response_len = int(response_lengths[i])
        total_len = int(total_lengths[i])
        stu_logits_shard = stu_logits_list[i]
        stu_mask_full = loss_masks[i]
        if stu_mask_full.numel() != response_len:
            stu_mask_full = stu_mask_full[-response_len:]
        stu_mask_full = stu_mask_full.to(device=device)

        local_response_indices = _get_gold_local_response_indices(
            total_len, response_len, logits_shift=0 if gold_observed_unshift else 1
        )
        if len(local_response_indices) != int(stu_logits_shard.shape[0]):
            msg = (
                f"[OPD][gold] sample {i}: local response index/logit length mismatch: "
                f"indices={len(local_response_indices)} logits={int(stu_logits_shard.shape[0])}."
            )
            if not getattr(args, "allow_opd_prefix_truncation", False):
                fatal_gold_msg = fatal_gold_msg or (
                    msg
                    + " Refusing to truncate fresh GOLD data after synchronizing "
                    "this failure across DP/CP ranks. Pass --allow-opd-prefix-truncation "
                    "only for legacy/debug replay."
                )
            min_local = min(len(local_response_indices), int(stu_logits_shard.shape[0]))
            if is_logging_rank and getattr(args, "allow_opd_prefix_truncation", False):
                _logger.warning(msg + f" Legacy fallback enabled: truncating to {min_local}.")
            local_response_indices = local_response_indices[:min_local]
            stu_logits_shard = stu_logits_shard[:min_local]
        local_pos_by_global = {gidx: lidx for lidx, gidx in enumerate(local_response_indices)}
        local_label_tokens = sum(
            float(stu_mask_full[gidx].detach().item())
            for gidx in local_response_indices
            if 0 <= gidx < stu_mask_full.numel()
        )
        total_label_tokens += local_label_tokens

        tea_hidden = teacher_hidden_states_list[i]
        if tea_hidden is None:
            skipped_none_hidden += 1
            sample_alignment_flags.append(0.0)
            continue
        if isinstance(tea_hidden, np.ndarray):
            tea_hidden = torch.from_numpy(tea_hidden)
        tea_hidden = tea_hidden.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)

        tea_input_ids_at_hidden = _to_int_list(teacher_token_ids_list[i])
        tea_hidden_len = int(tea_hidden.shape[0])
        tea_token_len = len(tea_input_ids_at_hidden)
        if tea_hidden_len != tea_token_len:
            msg = (
                f"[OPD][gold] sample {i}: teacher hidden/token length mismatch: "
                f"hidden={tea_hidden_len} token_ids={tea_token_len}."
            )
            if not getattr(args, "allow_opd_prefix_truncation", False):
                fatal_gold_msg = fatal_gold_msg or (
                    msg
                    + " Refusing to truncate fresh GOLD data after synchronizing "
                    "this failure across DP/CP ranks. Pass --allow-opd-prefix-truncation "
                    "only for legacy/debug replay."
                )
            common_teacher_len = min(tea_hidden_len, tea_token_len)
            if is_logging_rank and getattr(args, "allow_opd_prefix_truncation", False):
                _logger.warning(msg + f" Legacy fallback enabled: truncating to {common_teacher_len}.")
            tea_hidden = tea_hidden[:common_teacher_len]
            tea_input_ids_at_hidden = tea_input_ids_at_hidden[:common_teacher_len]
        if int(tea_hidden.shape[0]) <= 1:
            skipped_short_hidden += 1
            sample_alignment_flags.append(0.0)
            continue

        stu_label_ids_full = _to_int_list(unconcat_tokens[i][-response_len:])
        if gold_observed_unshift:
            # TRL GOLD "observed": pair teacher-hidden row j with the token AT its own slot.
            # teacher_token_ids/hidden are captured at masked positions [prompt_len-1 .. seq_len-2],
            # so row 0 is the LAST PROMPT position (predicts the first response token) and the eos
            # position (seq_len-1) is not captured.  Drop the leading last-prompt row on BOTH the
            # hidden states and the ids so row j' now holds the OBSERVED distribution at response
            # token r_{j'}'s slot, aligned to r_{j'}'s bytes.  No eos is appended (the eos hidden was
            # never captured), and skip_teacher_eos must NOT trim (there is no trailing eos id here).
            tea_hidden = tea_hidden[1:]
            tea_label_ids = list(tea_input_ids_at_hidden[1:])
            tea_skip_eos = False
        else:
            tea_label_ids = tea_input_ids_at_hidden[1:] + [int(teacher_eos_id)]
            tea_skip_eos = skip_teacher_eos

        student_response_text = batch.get("teacher_response_text", [None] * len(response_lengths))[i]
        if student_response_text is None:
            fatal_gold_msg = fatal_gold_msg or (
                "gold_opd_loss requires teacher_response_text/original completion text. "
                "Fresh rollout should set Sample.teacher_response_text in _inject_teacher_hidden_states; "
                "recorded failure and will synchronize across DP/CP ranks before raising."
            )
            sample_alignment_flags.append(0.0)
            continue
        # GOLD/ULD byte offsets must describe the exact in-context token rows
        # used below.  Do not retokenize the response-only text and then pair
        # those IDs with rollout/teacher rows: ByteLevel tokenizers can merge at
        # the prompt/completion boundary.  Derive offsets from actual sampled
        # IDs, matching TRL GOLD's on-policy path.
        try:
            stu_resp_ids, stu_offsets = _gold_trim_ids_and_offsets_for_answer(
                student_tokenizer,
                stu_label_ids_full,
                student_response_text,
                skip_eos=skip_student_eos,
                eos_token_id=student_tokenizer.eos_token_id,
            )
            # TRL GOLD on-policy: build teacher byte offsets from the teacher's
            # actual completion token IDs (raw ByteLevel piece widths), the same
            # space as the student side above -- NOT by re-encoding the decoded
            # text (which would reintroduce the decode round trip / U+FFFD byte
            # inflation and a teacher/student byte-space mismatch).
            tea_resp_ids, tea_offsets = _gold_trim_ids_and_offsets_for_answer(
                teacher_tokenizer,
                tea_label_ids[: int(tea_hidden.shape[0])],
                student_response_text,
                skip_eos=tea_skip_eos,
                eos_token_id=teacher_eos_id,
                prefer_encoded_offsets=False,
            )
        except Exception as exc:
            fatal_gold_msg = fatal_gold_msg or (
                f"[OPD][gold] sample {i}: failed to build GOLD byte offsets: {exc}. "
                "Recorded failure and will synchronize across DP/CP ranks before raising."
            )
            if _any_cp_rank_failed(failed=True, device=device):
                sample_alignment_flags.append(0.0)
                continue

        stu_gold_len = len(stu_resp_ids)
        tea_gold_len = len(tea_resp_ids)
        if stu_gold_len <= 0 or tea_gold_len <= 0:
            skipped_empty_gold_region += 1
            sample_alignment_flags.append(0.0)
            continue

        if use_extended:
            stu_groups, tea_groups = _gold_align_by_byte_offsets(stu_offsets, tea_offsets)
            paired = [(sg, tg) for sg, tg in zip(stu_groups, tea_groups, strict=False) if sg and tg]
            stu_groups = [sg for sg, _ in paired]
            tea_groups = [tg for _, tg in paired]
        else:
            min_len = min(stu_gold_len, tea_gold_len)
            stu_groups = [[j] for j in range(min_len)]
            tea_groups = [[j] for j in range(min_len)]
        if not stu_groups or not tea_groups:
            skipped_no_gold_groups += 1
            sample_alignment_flags.append(0.0)
            continue

        group_mask_values: list[float] = []
        valid_stu_groups: list[list[int]] = []
        valid_stu_first_rows: list[int] = []
        valid_tea_groups: list[list[int]] = []
        for sg, tg in zip(stu_groups, tea_groups, strict=False):
            sg = [int(pos) for pos in sg]
            if not sg:
                continue
            # observed: base = FIRST student row; bayesian (--gold-trl-faithful): base = LAST.
            base_global_pos = sg[-1] if gold_bayesian else sg[0]
            first_local_row = local_pos_by_global.get(int(base_global_pos))
            # Under CP a GOLD group can straddle ranks.  The rank owning the
            # BASE student token owns the full merged distribution; the scalar
            # (non-base) token log-probs are gathered below, so we no longer
            # require every student row in the group to be CP-local.
            if first_local_row is None:
                continue
            if any(int(pos) >= int(stu_mask_full.numel()) for pos in sg):
                continue
            filtered_tg = [int(pos) for pos in tg if int(pos) < int(tea_hidden.shape[0])]
            if not filtered_tg:
                continue
            mask_val = min(float(stu_mask_full[int(pos)].detach().item()) for pos in sg)
            valid_stu_groups.append(sg)
            valid_stu_first_rows.append(int(first_local_row))
            valid_tea_groups.append(filtered_tg)
            group_mask_values.append(mask_val)
        if _any_cp_rank_failed(failed=fatal_gold_msg is not None, device=device):
            sample_alignment_flags.append(0.0)
            continue
        if not valid_stu_groups:
            if cp_size == 1 and not getattr(args, "allow_opd_prefix_truncation", False):
                fatal_gold_msg = fatal_gold_msg or (
                    f"[OPD][gold] sample {i}: no valid GOLD alignment groups after CP/local-row filtering. "
                    "This indicates tokenizer/offset alignment failure; recorded failure and will "
                    "synchronize across DP/CP ranks before raising."
                )
                skipped_no_gold_groups += 1
                sample_alignment_flags.append(0.0)
                continue

        # Scalar (non-base) positions whose actual-token log-prob feeds the chain-rule product.
        # observed -> group[1:]; bayesian (--gold-trl-faithful) -> group[:-1].
        tail_global_positions = sorted(
            {int(pos) for group in stu_groups for pos in (group[:-1] if gold_bayesian else group[1:])}
        )
        if tail_global_positions:
            local_tail_log_probs = _gold_sparse_tail_label_log_probs_tp(
                stu_logits_shard,
                label_ids_full=stu_resp_ids,
                local_pos_by_global=local_pos_by_global,
                tail_global_positions=tail_global_positions,
                temperature=stu_temp,
                real_vocab_size=student_real_vocab_size,
                vocab_start=student_vocab_start,
                tp_size=tp_size,
                tp_group=tp_group,
            )
            full_tail_log_probs = all_gather_with_cp(
                local_tail_log_probs,
                total_len,
                response_len,
                logits_shift=0 if gold_observed_unshift else 1,
            )
            # The real GOLD loss for a CP-split group is computed only on the
            # rank that owns the first student token, but the gathered tail
            # scalars may have come from other CP ranks.  Keep the gather in
            # every rank's autograd graph with a zero-weight anchor so the
            # differentiable all-reduce backward runs collectively and returns
            # the owner-rank gradient to the tail-token ranks.
            total_loss = total_loss + full_tail_log_probs.sum() * 0.0
        else:
            full_tail_log_probs = stu_logits_shard.new_zeros((response_len,), dtype=torch.float32)

        sample_has_alignment = False
        for start in range(0, len(valid_stu_groups), chunk_size):
            sg_chunk = valid_stu_groups[start:start + chunk_size]
            first_rows_chunk = valid_stu_first_rows[start:start + chunk_size]
            tg_chunk = valid_tea_groups[start:start + chunk_size]
            mask_chunk = torch.tensor(group_mask_values[start:start + chunk_size], dtype=torch.float32, device=device)
            flat_stu_rows = sorted(set(first_rows_chunk))
            flat_tea_rows = sorted({row for group in tg_chunk for row in group})
            stu_row_map = {row: pos for pos, row in enumerate(flat_stu_rows)}
            tea_row_map = {row: pos for pos, row in enumerate(flat_tea_rows)}
            first_row_positions = [stu_row_map[row] for row in first_rows_chunk]
            remap_tg = [[tea_row_map[row] for row in group] for group in tg_chunk]

            stu_rows_t = torch.tensor(flat_stu_rows, dtype=torch.long, device=stu_logits_shard.device)
            tea_rows_t = torch.tensor(flat_tea_rows, dtype=torch.long, device=lm_head.weight.device)
            stu_log_probs = _gold_full_log_probs_from_vocab_parallel_logits(
                stu_logits_shard.index_select(0, stu_rows_t),
                temperature=stu_temp,
                real_vocab_size=student_real_vocab_size,
                vocab_start=student_vocab_start,
                tp_size=tp_size,
                tp_group=tp_group,
            )
            with torch.no_grad():
                tea_log_probs = _gold_full_log_probs_from_vocab_parallel_logits(
                    # fp8 teacher (MiniMax-M2.7 block-quant) can emit +inf lm_head logits -> finite
                    # forward but NaN grad in backward; cap to a finite peak (matches the BPM/SimCT guard).
                    torch.nan_to_num(lm_head(tea_hidden[tea_rows_t]), nan=0.0, posinf=1e4, neginf=-1e4),
                    temperature=tea_temp,
                    real_vocab_size=teacher_real_vocab_size,
                    vocab_start=teacher_vocab_start,
                    tp_size=tp_size,
                    tp_group=tp_group,
                )

            if use_extended:
                tea_token_ids_for_rows = [int(tea_resp_ids[row]) for row in flat_tea_rows]
                stu_first_log_probs = stu_log_probs[
                    torch.tensor(first_row_positions, dtype=torch.long, device=stu_log_probs.device)
                ]
                stu_aligned_log_probs = _gold_merge_log_probs_from_first_rows_and_tail_scalars(
                    stu_first_log_probs,
                    sg_chunk,
                    full_tail_log_probs,
                    clamp_min_prob=gold_merge_clamp_min_prob,
                    bayesian=gold_bayesian,
                )
                tea_aligned_log_probs = _gold_merge_log_probs_with_alignment_groups(
                    tea_log_probs,
                    remap_tg,
                    tea_token_ids_for_rows,
                    clamp_min_prob=gold_merge_clamp_min_prob,
                    bayesian=gold_bayesian,
                )
                if use_hybrid_loss:
                    per_group_loss, matched_comp, unmatched_comp = _gold_hybrid_loss_from_log_probs(
                        stu_aligned_log_probs,
                        tea_aligned_log_probs,
                        hybrid_vocab=hybrid_vocab,
                        beta=hybrid_beta,
                        matched_weight=hybrid_matched_weight,
                        unmatched_weight=hybrid_unmatched_weight,
                        return_components=True,
                    )
                else:
                    per_group_loss = _gold_sorted_l1_from_log_probs(stu_aligned_log_probs, tea_aligned_log_probs)
                with torch.no_grad():
                    # Extended GOLD merged rows may have probability mass < 1
                    # because multi-token groups multiply conditional
                    # probabilities.  Normalize only for the entropy diagnostic;
                    # the ULD loss above intentionally uses the raw merged masses.
                    stu_aligned_probs = stu_aligned_log_probs.detach().exp()
                    tea_aligned_probs = tea_aligned_log_probs.detach().exp()
                    stu_mass = stu_aligned_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    tea_mass = tea_aligned_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    stu_probs_for_entropy = stu_aligned_probs / stu_mass
                    tea_probs_for_entropy = tea_aligned_probs / tea_mass
                    stu_entropy = -(
                        stu_probs_for_entropy
                        * torch.log(stu_probs_for_entropy.clamp_min(1e-12))
                    ).sum(dim=-1)
                    tea_entropy = -(
                        tea_probs_for_entropy
                        * torch.log(tea_probs_for_entropy.clamp_min(1e-12))
                    ).sum(dim=-1)
            else:
                # Positional GOLD: groups are one row each.
                with torch.no_grad():
                    stu_probs = stu_log_probs.exp()
                    tea_probs = tea_log_probs.exp()
                    stu_entropy = -(stu_probs * stu_log_probs).sum(dim=-1)
                    tea_entropy = -(tea_probs * tea_log_probs).sum(dim=-1)
                if use_hybrid_loss:
                    per_group_loss, matched_comp, unmatched_comp = _gold_hybrid_loss_from_log_probs(
                        stu_log_probs,
                        tea_log_probs,
                        hybrid_vocab=hybrid_vocab,
                        beta=hybrid_beta,
                        matched_weight=hybrid_matched_weight,
                        unmatched_weight=hybrid_unmatched_weight,
                        return_components=True,
                    )
                else:
                    per_group_loss = _gold_sorted_l1_from_log_probs(stu_log_probs, tea_log_probs)

            masked = per_group_loss * mask_chunk
            total_distill = total_distill + masked.sum()
            if use_hybrid_loss:
                with torch.no_grad():
                    total_matched = total_matched + (matched_comp.detach() * mask_chunk).sum()
                    total_unmatched = total_unmatched + (unmatched_comp.detach() * mask_chunk).sum()
            sample_term = distill_weight * masked.sum() * tp_loss_scale
            total_loss = total_loss + sample_term
            mask_sum = float(mask_chunk.sum().detach().item())
            total_aligned += mask_sum
            if track_per_sample:
                per_sample_distill[i] = per_sample_distill.get(i, 0.0) + sample_term
                per_sample_aligned[i] = per_sample_aligned.get(i, 0.0) + mask_sum
            total_groups += float(len(mask_chunk))
            sample_has_alignment = sample_has_alignment or (mask_sum > 0.0)
            with torch.no_grad():
                total_tea_entropy = total_tea_entropy + (tea_entropy * mask_chunk).sum()
                total_stu_entropy = total_stu_entropy + (stu_entropy * mask_chunk).sum()

        if ce_weight > 0:
            local_rows_for_ce = []
            labels_for_ce = []
            masks_for_ce = []
            for gidx in local_response_indices:
                if skip_student_eos and gidx >= stu_gold_len:
                    continue
                lidx = local_pos_by_global.get(int(gidx))
                if lidx is None or gidx >= len(stu_label_ids_full) or gidx >= int(stu_mask_full.numel()):
                    continue
                local_rows_for_ce.append(lidx)
                labels_for_ce.append(int(stu_label_ids_full[gidx]))
                masks_for_ce.append(float(stu_mask_full[gidx].detach().item()))
            if local_rows_for_ce:
                rows_t = torch.tensor(local_rows_for_ce, dtype=torch.long, device=stu_logits_shard.device)
                labels_t = torch.tensor(labels_for_ce, dtype=torch.long, device=stu_logits_shard.device)
                mask_t = torch.tensor(masks_for_ce, dtype=torch.float32, device=device)
                ce_log_probs = _gold_sample_log_probs_from_vocab_parallel_logits(
                    stu_logits_shard.index_select(0, rows_t),
                    labels_t,
                    temperature=1.0,
                    real_vocab_size=student_real_vocab_size,
                    vocab_start=student_vocab_start,
                    tp_size=tp_size,
                    tp_group=tp_group,
                )
                per_ce = -ce_log_probs
                ce_sum = (per_ce * mask_t).sum()
                total_ce = total_ce + ce_sum
                total_ce_tokens += float(mask_t.sum().detach().item())

        sample_alignment_flags.append(float(sample_has_alignment))

    distill_loss_sum = total_loss
    ce_loss_sum = (
        ce_weight * total_ce * tp_loss_scale
        if ce_weight > 0
        else torch.tensor(0.0, dtype=torch.float32, device=device)
    )
    _raise_if_any_dp_cp_rank_failed(
        failed=fatal_gold_msg is not None,
        local_message=fatal_gold_msg,
        device=device,
        context="[OPD][gold] per-sample validation",
    )
    global_aligned, global_label_tokens = _reduce_dp_cp_float_counts(
        [float(total_aligned), float(total_label_tokens)],
        device=device,
    )
    # CP-shared (per-DP-rank) aligned-group count for the calculate_per_token_loss=False
    # backward denominator.  total_aligned is CP-LOCAL; the schedule SUMS the CP ranks'
    # returns, so a CP-local denom inflates the summed loss/grad by ~cp_size.  Mirror
    # cp_utils mask_sum (CP-shared, per-DP-rank); do NOT use global_aligned here (that
    # is DP+CP-summed and would under-normalize by ~dp_size vs the dispatcher).
    cp_aligned = _reduce_cp_float_counts([float(total_aligned)], device=device)[0]
    align_ratio_mean = total_aligned / max(total_label_tokens, 1.0)
    global_align_ratio_mean = global_aligned / max(global_label_tokens, 1.0)
    min_align_ratio = float(getattr(args, "gold_min_align_ratio", 0.0) or 0.0)
    if global_label_tokens <= 0:
        zero_loss = logits.sum() * 0.0
        if track_per_sample:
            args._opd_per_sample_normalizer = 0.0
        if is_logging_rank:
            _logger.info(
                "[OPD][gold] microbatch has no unmasked GOLD/ULD tokens; "
                "returning zero loss."
            )
        return zero_loss, {"loss": zero_loss.detach(), "gold_uld_loss": zero_loss.detach()}
    if global_aligned <= 0:
        raise ValueError(
            "[OPD][gold] no aligned GOLD/ULD groups in this DP/CP microbatch group. "
            "This usually means teacher/student response text, token IDs, byte offsets, "
            "or chat-template boundaries are mismatched. "
            f"label_tokens={global_label_tokens:.0f}, "
            f"skips(none_hidden={skipped_none_hidden}, short_hidden={skipped_short_hidden}, "
            f"empty_gold_region={skipped_empty_gold_region}, no_gold_groups={skipped_no_gold_groups}), "
            "teacher_tokenizer="
            f"{getattr(args, 'bpm_teacher_tokenizer_path', None) or getattr(args, 'bpm_teacher_model_path', None)}, "
            f"student_tokenizer={getattr(args, 'hf_checkpoint', None)}."
        )
    if min_align_ratio > 0 and global_align_ratio_mean < min_align_ratio:
        raise ValueError(
            f"[OPD][gold] global_align_ratio={global_align_ratio_mean:.4f} below "
            f"--gold-min-align-ratio={min_align_ratio:.4f}. Refusing to train a near-zero "
            "or badly shifted GOLD/ULD KD loss."
        )
    denom = max(total_aligned, 1.0)
    distill_mean = total_distill / denom
    matched_mean = (total_matched / denom).detach() if use_hybrid_loss else None
    unmatched_mean = (total_unmatched / denom).detach() if use_hybrid_loss else None
    mean_tea_entropy = (total_tea_entropy / denom).detach()
    mean_stu_entropy = (total_stu_entropy / denom).detach()
    sample_count = _count_cp_unique_samples_with_alignment(sample_alignment_flags, device=device)
    if track_per_sample:
        # TRL GOLD per-sample-mean-then-batch-mean: each sample normalized by its OWN aligned-group
        # count, then averaged over the samples that have alignment.  per_sample_distill[i] stays
        # CP-LOCAL (keeps the autograd graph on this rank's tokens); only the scalar group COUNTS
        # are CP-reduced to their full per-DP-rank value.  The schedule SUMS CP ranks' returns, so
        #   Σ_cp Σ_s (sd_cplocal,s / full_aligned_s) / N  =  Σ_s (full_sd_s / full_aligned_s) / N
        # = exact per-sample mean.  Identical at CP=1 (_reduce_cp is a no-op there).
        if ce_weight > 0:
            raise NotImplementedError(
                "--opd-loss-reduction per_sample with --gold-ce-weight>0 is not implemented yet "
                "(per-sample CE normalization). Use --gold-ce-weight 0 (the default) for per_sample."
            )
        n_orig = len(response_lengths)
        full_aligned_vec = _reduce_cp_float_counts(
            [float(per_sample_aligned.get(i, 0.0)) for i in range(n_orig)], device=device
        )
        n_samples_ps = float(sum(1 for a in full_aligned_vec if a > 0.0))
        sample_mean_sum = logits.new_zeros(())
        for i in range(n_orig):
            fa = full_aligned_vec[i]
            sd = per_sample_distill.get(i, None)
            if fa > 0.0 and sd is not None:
                sample_mean_sum = sample_mean_sum + sd / fa
        # EXACT per-sample-mean-then-batch-mean via the calculate_per_token_loss=True path: return the
        # per-sample NUMERATOR Σ_s(sd_s/aligned_s) and stash the sample COUNT as Megatron's per-token
        # normalizer.  The schedule DP+CP-sums both, so grads = Σ_all(sd_s/aligned_s)/N_global exactly,
        # even under --use-dynamic-batch-size (multiple samples per microbatch).
        loss = sample_mean_sum
        args._opd_per_sample_normalizer = n_samples_ps
        # Blocker-1 anchor: 0.0*total_loss keeps EVERY sample's cross-CP full_tail_log_probs
        # all-reduce node in this rank's graph (owner AND non-owner ranks, via gold_loss.py:407's
        # `total_loss + full_tail_log_probs.sum()*0.0`), so the differentiable all_gather_with_cp
        # backward stays collective/symmetric and cannot NCCL-deadlock at CP>1.
        loss = loss + 0.0 * total_loss
    elif args.calculate_per_token_loss:
        # The dispatcher normalizes by response-label tokens, but GOLD/ULD
        # distillation is accumulated over aligned byte-offset groups.  Preserve a
        # per-aligned-token distillation objective while leaving the optional CE
        # branch on the normal response-token denominator.
        aligned_normalizer_scale = global_label_tokens / max(global_aligned, 1.0)
        loss = distill_loss_sum * aligned_normalizer_scale + ce_loss_sum
    else:
        loss = (distill_loss_sum + ce_loss_sum) / max(cp_aligned, 1.0)
    loss = loss + 0.0 * logits.sum()

    if is_logging_rank:
        hybrid_summary = (
            f"matched_jsd={matched_mean.item():.4f} unmatched_uld={unmatched_mean.item():.4f} "
            if use_hybrid_loss
            else ""
        )
        _logger.info(
            f"[OPD][gold][step summary] loss={(distill_weight * total_distill / denom).detach().item():.4f} "
            f"uld_l1={distill_mean.detach().item():.4f} {hybrid_summary}aligned_groups={total_aligned:.0f} "
            f"global_aligned_groups={global_aligned:.0f} "
            f"label_tokens={total_label_tokens:.0f}/{global_label_tokens:.0f} "
            f"align_ratio={align_ratio_mean:.3f} global_align_ratio={global_align_ratio_mean:.3f} "
            f"tea_ent={mean_tea_entropy.item():.3f} stu_ent={mean_stu_entropy.item():.3f} "
            f"n_samples={sample_count} gold_tp_exact={tp_size > 1} tp={tp_size} cp={cp_size}"
        )

    if mpu.is_pipeline_last_stage():
        if not hasattr(args, '_gold_opd_metrics') or args._gold_opd_metrics is None:
            args._gold_opd_metrics = {
                "gold_uld_loss_sum": 0.0,
                "gold_tea_entropy_sum": 0.0,
                "gold_stu_entropy_sum": 0.0,
                "gold_overlap_count_sum": 0.0,
                "gold_align_ratio_sum": 0.0,
                "gold_aligned_tokens_sum": 0.0,
                "gold_num_tokens_sum": 0.0,
                "gold_num_samples_with_alignment_sum": 0.0,
                "_gold_opd_loss_type": "gold_uld",
            }
            if use_hybrid_loss:
                # Raw hybrid components (TRL GOLD parity); reduced/published as
                # gold_matched_loss (JSD) and gold_unmatched_loss (ULD).
                args._gold_opd_metrics["gold_matched_loss_sum"] = 0.0
                args._gold_opd_metrics["gold_unmatched_loss_sum"] = 0.0
        acc = args._gold_opd_metrics
        acc["gold_uld_loss_sum"] += float(distill_mean.detach().item()) * float(total_aligned)
        if use_hybrid_loss:
            acc["gold_matched_loss_sum"] += float(matched_mean.item()) * float(total_aligned)
            acc["gold_unmatched_loss_sum"] += float(unmatched_mean.item()) * float(total_aligned)
        acc["gold_tea_entropy_sum"] += float(mean_tea_entropy.item()) * float(total_aligned)
        acc["gold_stu_entropy_sum"] += float(mean_stu_entropy.item()) * float(total_aligned)
        acc["gold_overlap_count_sum"] += float(total_groups)
        acc["gold_align_ratio_sum"] += float(total_aligned)
        acc["gold_aligned_tokens_sum"] += float(total_aligned)
        acc["gold_num_tokens_sum"] += float(total_label_tokens)
        acc["gold_num_samples_with_alignment_sum"] += sample_count
    else:
        args._gold_opd_metrics = None

    return (
        loss,
        {
            "loss": loss.clone().detach(),
            "gold_uld_loss": loss.clone().detach(),
        },
    )
