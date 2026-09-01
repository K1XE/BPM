"""SimCT baseline: overlap-vocabulary cross-tokenizer distillation loss."""

from __future__ import annotations

import inspect
from argparse import Namespace
from collections.abc import Callable

import torch
import torch.nn.functional as F
from megatron.core import mpu

from slime.backends.megatron_utils.cp_utils import all_gather_with_cp
from slime.utils.types import RolloutBatch
from slime_plugins.bpm.backend.alignment import (
    _bpm_ids_to_decoded_texts as _ids_to_decoded_texts,
    _bpm_use_byte_alignment as _use_byte_alignment,
)
from slime_plugins.bpm.backend.loss_helpers import get_responses
from slime_plugins.bpm.backend.projection import (
    _get_bpm_teacher_lm_head,
    _select_hidden_rows_to_device,
)
from slime_plugins.bpm.backend.vocab import (
    _get_bpm_local_response_indices as _get_simct_local_response_indices,
    _get_bpm_student_tokenizer as _get_simct_student_tokenizer,
    _get_bpm_teacher_tokenizer as _get_simct_teacher_tokenizer,
)
from slime_plugins.bpm.loss.bpm_loss_utils import (
    _any_cp_rank_failed,
    _count_cp_unique_samples_with_alignment,
    _raise_if_any_dp_cp_rank_failed,
    _reduce_cp_float_counts,
    _reduce_dp_cp_float_counts,
    _resolve_opd_distill_scope,
    _validate_token_ids_in_vocab_msg,
    _validate_tp_global_ids_owned_msg,
)

from .simct_alignment import (
    _align_simct_texts_with_spans,
    _get_simct_piece_cache,
    _simct_pieces_from_cache,
)
from .simct_kernels import (
    _divergence_from_log_probs,
    _effective_simct_compile_bucket_size,
    _pad_simct_chunk_for_compile,
    _simct_full_virtual_vocab_loss_and_entropy_fused,
    _simct_full_virtual_vocab_loss_only_fused,
    _simct_streaming_full_virtual_vocab_loss_only_from_chunks,
    _simct_virtual_vocab_loss_from_logits,
    _simct_virtual_vocab_loss_only_from_logits,
)
from .simct_metrics import accumulate_simct_train_metrics, log_simct_step_summary
from .simct_projection import (
    _hidden_rows_len,
    _hidden_rows_shape,
    _sample_token_logits_from_logits,
    _simct_lm_head_at_global_ids,
    _simct_lm_head_token_logits,
)
from .simct_tp import (
    _gather_tp_logits_at_global_ids_prevalidated,
    _gather_tp_logits_at_rows_global_ids_prevalidated,
    _gather_tp_logits_at_rows_global_ids_subset,
)
from .simct_vocab import (
    _find_simct_overlap_tokens,
    _get_simct_overlap_pair_map,
    _get_simct_teacher_to_student_id_map,
    _simct_teacher_overlap_topk_candidates,
)


_SIMCT_TRAIN_METRICS_ACCEPTS_METRIC_WEIGHT = (
    "metric_weight" in inspect.signature(accumulate_simct_train_metrics).parameters
)


def _is_terminal_student_eos_segment(stu_start, stu_end, stu_label_ids, student_eos_id):
    """True iff a SimCT segment is the pure terminal single-token student-EOS unit.

    Used by ``--simct-skip-eos`` to drop ONLY the final student-EOS segment: a
    cross-tokenizer teacher prefilled on the student's own (often non-terminated)
    text puts ~0 mass on the student EOS, so reverse-KL would actively suppress the
    student's terminal stop-prob.  The condition is intentionally tight (terminal
    position AND single token AND exactly the student EOS id) so it can never drop a
    content span that merely ends in EOS.
    """
    return (
        student_eos_id is not None
        and (stu_end - stu_start) == 1
        and stu_end == len(stu_label_ids)
        and int(stu_label_ids[stu_start]) == int(student_eos_id)
    )




def simct_core_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    sum_of_sample: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """simct OPD loss using overlap-vocab token-piece alignment.

    Contract implemented here:
      1. Teacher SGLang returns hidden states at teacher loss-mask positions:
         [last teacher prompt hidden, teacher response hidden..., before EOS].
      2. ``teacher_token_ids`` are the input token IDs at the same positions.
         Therefore teacher *label* IDs for alignment are ``teacher_token_ids[1:] + [teacher_eos]``.
      3. Student ``get_responses`` returns logits at [last student prompt hidden,
         student response hidden..., before EOS] and ``tokens_chunk`` is already the
         student *label* sequence [student response tokens..., student_eos].
      4. Alignment converts label IDs to tokenizer pieces, strips Ġ/▁,
         greedy cumulative-text 1:1 matching, then compute KL on overlap vocab.

    Important distributed detail:
      - TP: student and teacher lm_head logits are vocab-sharded. We gather only the
        overlap-token columns with a straight-through all-reduce, so gradients flow
        to the owning student TP shard and no c10d autograd warning is triggered.
      - CP: only local student response positions are trained on each CP rank. The
        global student label IDs are still used to compute the simct alignment; then
        pairs are filtered to the CP-local positions. This avoids silently slicing
        teacher-tokenizer hidden states with student-tokenizer offsets.
    """
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    teacher_hidden_states_list = batch.get("teacher_hidden_states")
    teacher_token_ids_list = batch.get("teacher_token_ids")

    if teacher_hidden_states_list is None:
        raise ValueError(
            "simct_opd_loss requires teacher_hidden_states in batch. "
            "Use simct rollout pipeline."
        )
    if teacher_token_ids_list is None:
        raise ValueError(
            "simct_opd_loss requires teacher_token_ids in batch. "
            "Ensure _inject_teacher_hidden_states returns teacher token IDs."
        )

    lm_head = _get_bpm_teacher_lm_head(args)
    teacher_tokenizer = _get_simct_teacher_tokenizer(args)
    student_tokenizer = _get_simct_student_tokenizer(args)

    device = logits.device
    if not hasattr(args, "_simct_stu_overlap_ids") or args._simct_stu_overlap_ids is None:
        stu_overlap_ids, tea_overlap_ids = _find_simct_overlap_tokens(
            student_tokenizer, teacher_tokenizer, device
        )
        args._simct_stu_overlap_ids = stu_overlap_ids
        args._simct_tea_overlap_ids = tea_overlap_ids
    else:
        stu_overlap_ids = args._simct_stu_overlap_ids.to(device=device)
        tea_overlap_ids = args._simct_tea_overlap_ids.to(device=device)

    num_overlap = int(stu_overlap_ids.numel())
    if num_overlap <= 0:
        raise ValueError("[OPD][simct][SimCT] overlap vocabulary is empty")
    overlap_ratio = num_overlap / max(len(student_tokenizer.get_vocab()), 1)
    distill_scope, effective_k = _resolve_opd_distill_scope(args, num_overlap)
    sample_mode = distill_scope == "sample"
    topk_mode = distill_scope == "topk"
    simct_alignment_mode = getattr(args, "simct_alignment_mode", "span")
    if simct_alignment_mode not in ("simple", "span"):
        raise ValueError(f"Unsupported --simct-alignment-mode: {simct_alignment_mode}")
    span_alignment_enabled = simct_alignment_mode == "span"

    is_logging_rank = (
        mpu.is_pipeline_last_stage()
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )
    if not getattr(args, "_simct_config_logged", False) and is_logging_rank:
        _logger.info(
            f"[OPD][simct] overlap_tokens={num_overlap} "
            f"overlap_ratio={overlap_ratio*100:.1f}% "
            f"lm_head_shape={tuple(lm_head.weight.shape)} "
            f"opd_loss_type={getattr(args, 'opd_loss_type', 'fkl')} "
            f"temperature={getattr(args, 'opd_temperature', 1.0)} "
            f"scope={distill_scope} topk_mode={topk_mode} k={effective_k} "
            f"alignment_mode={simct_alignment_mode} "
            f"simct_chunk={getattr(args, 'simct_chunk_size', 64)} "
            f"compile_bucket={getattr(args, 'simct_compile_bucket_size', 0)}"
        )
        args._simct_config_logged = True

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    loss_masks = batch["loss_masks"]
    unconcat_tokens = batch.get("unconcat_tokens")
    if unconcat_tokens is None:
        raise ValueError("simct_opd_loss requires unconcat_tokens for label alignment")

    stu_logits_list: list[torch.Tensor] = []
    stu_label_chunks: list[torch.Tensor] = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    ):
        stu_logits_list.append(logits_chunk)
        # tokens_chunk is already labels for the corresponding logits positions.
        stu_label_chunks.append(tokens_chunk)

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank() if tp_size > 1 else 0
    tp_group = mpu.get_tensor_model_parallel_group() if tp_size > 1 else None
    cp_size = mpu.get_context_parallel_world_size()

    teacher_shard_vocab = int(getattr(lm_head, "_opd_shard_vocab_size", lm_head.weight.shape[0]))
    teacher_vocab_start = int(getattr(lm_head, "_opd_vocab_start", tp_rank * teacher_shard_vocab))
    teacher_real_vocab_size = int(getattr(lm_head, "_opd_vocab_size", len(teacher_tokenizer.get_vocab())))
    student_shard_vocab = int(logits.shape[-1])
    student_vocab_start = tp_rank * student_shard_vocab
    # `logits` is TP-local vocab-parallel output here.  Use its last dimension
    # only as the TP shard width for ownership; use model/Megatron vocab size as
    # the real-token bound to avoid confusing padded ownership slots with real
    # tokenizer IDs.
    student_real_vocab_size = int(getattr(args, "vocab_size", 0) or len(student_tokenizer.get_vocab()))
    overlap_validation_msg = (
        _validate_tp_global_ids_owned_msg(
            tea_overlap_ids.to(device=lm_head.weight.device),
            shard_vocab=teacher_shard_vocab,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            context="[OPD][simct][SimCT] teacher overlap ids",
            real_vocab_size=teacher_real_vocab_size,
        )
        or _validate_tp_global_ids_owned_msg(
            stu_overlap_ids.to(device=device),
            shard_vocab=student_shard_vocab,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            context="[OPD][simct][SimCT] student overlap ids",
            real_vocab_size=student_real_vocab_size,
        )
    )
    _raise_if_any_dp_cp_rank_failed(
        failed=overlap_validation_msg is not None,
        local_message=overlap_validation_msg,
        device=device,
        context="[OPD][simct][SimCT] TP overlap-id validation",
    )

    opd_temperature = float(getattr(args, "opd_temperature", 1.0) or 1.0)
    opd_loss_type = getattr(args, "opd_loss_type", "fkl")
    simct_skip_eos = bool(getattr(args, "simct_skip_eos", False))
    student_eos_id = student_tokenizer.eos_token_id
    simct_loss_metric_name = f"simct_{opd_loss_type}"
    opd_jsd_beta = float(getattr(args, "opd_jsd_beta", 0.5))
    opd_ce_weight = float(getattr(args, "opd_ce_weight", 0.0) or 0.0)
    if opd_loss_type not in ("fkl", "kl", "rkl", "jsd"):
        raise ValueError(f"Unsupported opd_loss_type for simct: {opd_loss_type}")
    if opd_loss_type == "jsd" and not (0.0 < opd_jsd_beta < 1.0):
        raise ValueError(f"opd_jsd_beta must be in (0, 1) for jsd, got {opd_jsd_beta}")

    # --- Cross-family stop-token bridge (opt-in, default OFF) -----------------
    # Aggregate the teacher's FULL stop set (model-agnostic detection) into the
    # single student-EOS overlap column via log-sum-exp, so a cross-family
    # teacher whose chat turn-end token (e.g. GLM <|user|>) is NOT the column the
    # student EOS was mapped to (teacher.eos_token_id == <|endoftext|>) still
    # transfers a "stop" signal.  Supports TP=1 + full scope only; forces the
    # dense exact path (see use_streaming_full_loss below).
    stop_bridge = bool(getattr(args, "simct_stop_token_bridge", False))
    teacher_stop_ids_t = None
    student_stop_ids_t = None
    eos_bridge_col_idx = None
    if stop_bridge:
        if tp_size > 1:
            raise NotImplementedError(
                "--simct-stop-token-bridge currently supports TP=1 only "
                f"(got tensor-model-parallel-size={tp_size})."
            )
        if distill_scope != "full" or topk_mode or sample_mode:
            raise NotImplementedError(
                "--simct-stop-token-bridge requires the full distill scope "
                f"(got scope={distill_scope}); topk/sample scopes are not supported yet."
            )
        if getattr(args, "_simct_teacher_stop_ids", None) is None:
            from slime_plugins.bpm.backend.special_tokens import detect_stop_token_ids

            tea_eos_id = teacher_tokenizer.eos_token_id
            if tea_eos_id is None:
                raise ValueError("[OPD][simct][stop-bridge] teacher tokenizer has no eos_token_id")
            stop_set = detect_stop_token_ids(
                teacher_tokenizer, getattr(args, "bpm_teacher_tokenizer_path", None) or getattr(args, "bpm_teacher_model_path", None)
            )
            stop_list = sorted(
                int(t) for t in stop_set if 0 <= int(t) < teacher_real_vocab_size
            )
            ov = args._simct_tea_overlap_ids.detach().cpu().tolist()
            ov_set = set(ov)
            if int(tea_eos_id) not in ov_set:
                raise ValueError(
                    "[OPD][simct][stop-bridge] teacher eos id is not in the overlap; "
                    "the EOS column is required for the stop bridge."
                )
            # No double-counting: aggregate exactly the EOS column itself plus the
            # stop tokens that are NOT already their own (content) overlap column.
            # A stop token that surface-matched a student token is already in the
            # virtual softmax as its own column; summing it into the EOS column too
            # would count its mass twice in the teacher partition.  For GLM/Qwen the
            # extras (<|user|>/<|observation|>) are not in the overlap, so all are kept.
            agg_ids = [int(tea_eos_id)] + [
                s for s in stop_list if s != int(tea_eos_id) and s not in ov_set
            ]
            agg_ids = sorted(set(agg_ids))
            dropped = [s for s in stop_list if s != int(tea_eos_id) and s in ov_set]
            args._simct_teacher_stop_ids = torch.tensor(agg_ids, dtype=torch.long)
            args._simct_eos_bridge_col_idx = int(ov.index(int(tea_eos_id)))
            if is_logging_rank:
                _logger.info(
                    "[OPD][simct][stop-bridge] aggregating %d teacher stop ids into EOS "
                    "overlap column %d: %s%s",
                    len(agg_ids),
                    args._simct_eos_bridge_col_idx,
                    agg_ids,
                    f" (skipped {dropped} already in content overlap to avoid double-count)"
                    if dropped
                    else "",
                )
            # Degeneracy guard: if only the tokenizer eos survived, the teacher's
            # real turn-end token(s) (e.g. GLM <|user|>/<|observation|>) were NOT
            # detected -- generation_config.json likely missing at the teacher path
            # or its eos_token_id is a bare scalar.  The bridge then aggregates a
            # single id and is a SILENT no-op for cross-family teachers.  Warn loud.
            if len(agg_ids) == 1:
                _logger.warning(
                    "[OPD][simct][stop-bridge] DEGENERATE stop set: only the tokenizer "
                    "eos (%d) was detected for teacher at '%s'. The bridge will be a NO-OP. "
                    "Check that generation_config.json exists there with eos_token_id as the "
                    "FULL stop list (e.g. GLM [<|endoftext|>,<|user|>,<|observation|>]); a "
                    "cross-family teacher whose chat turn-end is out-of-overlap will get NO "
                    "stop signal.",
                    agg_ids[0],
                    getattr(args, "bpm_teacher_tokenizer_path", None) or getattr(args, "bpm_teacher_model_path", None),
                )
        # Symmetric STUDENT side: the student also declares its own stop set (e.g.
        # Qwen ships <|im_end|> AND <|endoftext|>); aggregate it the same way so we
        # compare the student's TOTAL stop probability against the teacher's, rather
        # than assuming the student stops on a single token.  Same EOS column index
        # (the overlap is paired, so stu_overlap_ids[e] is the student EOS).
        if getattr(args, "_simct_student_stop_ids", None) is None:
            from slime_plugins.bpm.backend.special_tokens import detect_stop_token_ids

            stu_eos_id = student_tokenizer.eos_token_id
            if stu_eos_id is None:
                raise ValueError("[OPD][simct][stop-bridge] student tokenizer has no eos_token_id")
            stu_stop_set = detect_stop_token_ids(
                student_tokenizer, getattr(args, "hf_checkpoint", None)
            )
            stu_stop_list = sorted(
                int(t) for t in stu_stop_set if 0 <= int(t) < student_real_vocab_size
            )
            stu_ov = args._simct_stu_overlap_ids.detach().cpu().tolist()
            eos_idx = int(args._simct_eos_bridge_col_idx)
            if eos_idx >= len(stu_ov) or int(stu_ov[eos_idx]) != int(stu_eos_id):
                raise ValueError(
                    "[OPD][simct][stop-bridge] student EOS column mismatch: "
                    f"stu_overlap_ids[{eos_idx}]={stu_ov[eos_idx] if eos_idx < len(stu_ov) else None} "
                    f"!= student eos {stu_eos_id}. The overlap pairing is inconsistent."
                )
            stu_ov_set = set(stu_ov)
            stu_agg = [int(stu_eos_id)] + [
                s for s in stu_stop_list if s != int(stu_eos_id) and s not in stu_ov_set
            ]
            stu_agg = sorted(set(stu_agg))
            args._simct_student_stop_ids = torch.tensor(stu_agg, dtype=torch.long)
            if is_logging_rank and len(stu_agg) > 1:
                _logger.info(
                    "[OPD][simct][stop-bridge] student stop set (%d ids) aggregated into "
                    "EOS overlap column %d: %s",
                    len(stu_agg),
                    eos_idx,
                    stu_agg,
                )
        # Cache the device-resident stop-id tensors so their data_ptr is STABLE
        # across calls.  The projection helper caches selected lm_head weights keyed
        # by global_ids.data_ptr(); recreating these tensors every call (via .to())
        # would change the key each step and defeat the cache (re-index_select the
        # stop weight every chunk).
        _tdev = lm_head.weight.device
        if getattr(args, "_simct_teacher_stop_ids_dev", None) is None or (
            args._simct_teacher_stop_ids_dev.device != _tdev
        ):
            args._simct_teacher_stop_ids_dev = args._simct_teacher_stop_ids.to(device=_tdev)
        teacher_stop_ids_t = args._simct_teacher_stop_ids_dev
        if getattr(args, "_simct_student_stop_ids_dev", None) is None or (
            args._simct_student_stop_ids_dev.device != device
        ):
            args._simct_student_stop_ids_dev = args._simct_student_stop_ids.to(device=device)
        student_stop_ids_t = args._simct_student_stop_ids_dev
        eos_bridge_col_idx = int(args._simct_eos_bridge_col_idx)
    if opd_ce_weight > 0 and tp_size > 1:
        # CE over a TP-sharded vocab needs Megatron vocab-parallel CE. Do not
        # silently all_gather with a non-differentiable c10d op.
        raise NotImplementedError(
            "simct --opd-ce-weight > 0 with tensor parallelism is not "
            "implemented safely yet. Use --opd-ce-weight 0.0 or TP=1."
        )

    opd_diagnostics_mode = getattr(args, "opd_diagnostics_mode", "basic")
    if opd_diagnostics_mode not in ("off", "basic", "full"):
        raise ValueError(f"Unsupported --opd-diagnostics-mode: {opd_diagnostics_mode}")
    # ``off`` means loss-only: do not spend the full-overlap hot path on
    # monitoring tensors, and do not report placeholder entropy zeros to W&B.
    # ``basic``/``full`` keep real entropy diagnostics computed from the same
    # normalized distributions as the KL loss.
    loss_only_diagnostics = opd_diagnostics_mode == "off"

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_kd = torch.tensor(0.0, dtype=torch.float32, device=device)
    # Token-weighted KD sum used ONLY for the per-token-loss backward objective.
    # The per-token denominator (num_tokens = sum of student-token masks) counts
    # M student tokens for an M-token span, but the per-segment numerator emits
    # ONE KD scalar per span -> spans get 1/M weight exactly where cross-tokenizer
    # boundaries concentrate the signal.  Weighting each segment's KD by its
    # student-token count makes numerator and denominator the same per-token unit.
    total_loss_tokwt = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_tea_entropy = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_stu_entropy = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_aligned_tokens = 0.0
    total_label_tokens = 0.0
    total_overlap_count = 0.0
    total_aligned_rows = 0.0
    # Count samples that are non-degenerate enough to actually reach the aligner
    # (tea_hidden present, tea_hidden_rows > 1, non-empty teacher ids).  Used to
    # distinguish "no aligned tokens because ALL samples were empty/EOS-only"
    # (benign zero loss) from "real samples failed to align" (genuine mismatch).
    total_alignable_samples = 0.0
    sample_alignment_flags: list[float] = []
    fatal_simct_msg: str | None = None

    teacher_eos_id = teacher_tokenizer.eos_token_id
    if teacher_eos_id is None:
        raise ValueError("teacher tokenizer eos_token_id is required for simct")

    def _to_int_list(x) -> list[int]:
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().tolist()]
        return [int(v) for v in x]

    tea_piece_cache = _get_simct_piece_cache(args, "_simct_teacher_piece_cache")
    stu_piece_cache = _get_simct_piece_cache(args, "_simct_student_piece_cache")
    overlap_pairs = _get_simct_overlap_pair_map(args, stu_overlap_ids, tea_overlap_ids)
    # Joint byte-level decision (both tokenizers ByteLevel) so teacher and student
    # alignment texts share the same byte-exact representation.
    use_byte_align = _use_byte_alignment(teacher_tokenizer, student_tokenizer)

    # --opd-loss-reduction per_sample (mean-of-per-sample-means, like TRL GOLD's
    # per-sample-mean-then-batch-mean): accumulate each ORIGINAL sample's CP-LOCAL
    # grad-carrying distill SUM (the same masked_loss.sum() term that feeds total_loss /
    # the False per_rank backward) and its aligned-row count, keyed by sample index i.
    # Skipped samples are simply absent from the dicts (-> 0), so we never touch the skip
    # paths and per_token/per_rank stay bit-identical (dicts stay empty when the flag is off).
    track_per_sample = getattr(args, "opd_loss_reduction", "per_token") == "per_sample"
    per_sample_distill: dict = {}
    per_sample_aligned: dict = {}
    if track_per_sample:
        # Define the per-sample normalizer up front so the dispatcher's num_tokens override always
        # sees a value even on the early zero-loss returns; the per_sample branch below overwrites
        # it with the real sample count.
        args._opd_per_sample_normalizer = 0.0

    for i in range(len(response_lengths)):
        response_len = int(response_lengths[i])
        total_len = int(total_lengths[i])

        stu_logits_shard = stu_logits_list[i]  # CP-local [R_local, V_stu/tp]
        stu_mask_full = loss_masks[i]
        if stu_mask_full.numel() != response_len:
            stu_mask_full = stu_mask_full[-response_len:]
        # The alignment path below is Python-side by construction (tokenizer
        # pieces/spans).  Keep mask reads Python-side too: calling .item() on a
        # CUDA mask for every token/span creates thousands of host-device syncs
        # per 32k sample and was a major simct-vs-stoken performance gap.
        stu_mask_values = [float(v) for v in stu_mask_full.detach().cpu().float().tolist()]
        stu_mask_len = len(stu_mask_values)

        # Hardening guard: the span aligner builds segments over the FULL response
        # and then trains a segment if max(mask)>0 (below), folding any
        # masked-token self-logit into the span mean.  Reference SimCT instead
        # filters labels by the loss mask *before* alignment, so a span never
        # mixes masked and unmasked tokens.  The two agree only when the response
        # mask is uniform (all-ones, or fully-zero which is skipped/dropped
        # upstream) -- which is exactly what the OPD rollout path emits today.
        # A partial (mixed 0/1) within-response mask -- e.g. future multi-turn or
        # tool-output masking -- would silently train a subtly-wrong objective, so
        # fail loudly (synchronized across DP/CP) instead.  Implement
        # mask-before-align to lift this restriction.
        nonzero_mask = sum(1 for v in stu_mask_values if v > 0.0)
        if 0 < nonzero_mask < stu_mask_len:
            fatal_simct_msg = fatal_simct_msg or (
                f"[OPD][simct][SimCT] sample {i}: partial within-response loss mask "
                f"({nonzero_mask}/{stu_mask_len} unmasked) is not supported by the span "
                "aligner, which would mix masked/unmasked tokens within a span. SimCT "
                "filters labels by the loss mask before alignment; implement "
                "mask-before-align for multi-turn/partial-mask OPD. Recorded failure "
                "and will synchronize across DP/CP ranks before raising."
            )
            sample_alignment_flags.append(0.0)
            continue

        # Full student label IDs: response tokens + student EOS.  This mirrors
        # the label sequence selected by input_ids.roll(-1)[student_loss_mask].
        stu_label_ids_full = _to_int_list(unconcat_tokens[i][-response_len:])

        # Map CP-local rows in stu_logits_shard to global response-label indices.
        local_response_indices = _get_simct_local_response_indices(total_len, response_len)
        if len(local_response_indices) != int(stu_logits_shard.shape[0]):
            msg = (
                f"[OPD][simct][SimCT] sample {i}: local index/logit length mismatch: "
                f"indices={len(local_response_indices)} logits={int(stu_logits_shard.shape[0])}."
            )
            if not getattr(args, "allow_opd_prefix_truncation", False):
                fatal_simct_msg = fatal_simct_msg or (
                    msg
                    + " Refusing to truncate fresh simct data after synchronizing "
                    "this failure across DP/CP ranks. Pass --allow-opd-prefix-truncation "
                    "only for legacy/debug replay."
                )
            else:
                min_local = min(len(local_response_indices), int(stu_logits_shard.shape[0]))
                if is_logging_rank and not getattr(args, "_simct_local_len_warned", False):
                    _logger.warning(msg + f" Legacy fallback enabled: truncating to {min_local}.")
                    args._simct_local_len_warned = True
                local_response_indices = local_response_indices[:min_local]
                stu_logits_shard = stu_logits_shard[:min_local]
        if _any_cp_rank_failed(failed=fatal_simct_msg is not None, device=device):
            sample_alignment_flags.append(0.0)
            continue

        local_pos_by_global = {gidx: lidx for lidx, gidx in enumerate(local_response_indices)}
        local_label_tokens = sum(
            stu_mask_values[gidx]
            for gidx in local_response_indices
            if 0 <= gidx < stu_mask_len
        )
        total_label_tokens += local_label_tokens

        tea_hidden = teacher_hidden_states_list[i]
        if tea_hidden is None:
            if is_logging_rank and i < 3:
                _logger.warning(f"[OPD][simct][SimCT] sample {i}: teacher_hidden_states is None, skipping")
            sample_alignment_flags.append(0.0)
            continue
        tea_hidden_rows = _hidden_rows_len(tea_hidden)

        tea_input_ids_at_hidden = teacher_token_ids_list[i]
        tea_input_ids_at_hidden = _to_int_list(tea_input_ids_at_hidden)
        if not tea_input_ids_at_hidden:
            if is_logging_rank and i < 3:
                _logger.warning(f"[OPD][simct][SimCT] sample {i}: empty teacher_token_ids, skipping")
            sample_alignment_flags.append(0.0)
            continue

        if tea_hidden_rows != len(tea_input_ids_at_hidden):
            msg = (
                f"[OPD][simct][SimCT] sample {i}: teacher hidden/token length mismatch: "
                f"hidden={tea_hidden_rows} token_ids={len(tea_input_ids_at_hidden)}."
            )
            if not getattr(args, "allow_opd_prefix_truncation", False):
                fatal_simct_msg = fatal_simct_msg or (
                    msg
                    + " Refusing to truncate fresh simct data after synchronizing "
                    "this failure across DP/CP ranks. Pass --allow-opd-prefix-truncation "
                    "only for legacy/debug replay."
                )
            else:
                common_teacher_len = min(tea_hidden_rows, len(tea_input_ids_at_hidden))
                if is_logging_rank and not getattr(args, "_simct_teacher_len_warned", False):
                    _logger.warning(msg + f" Legacy fallback enabled: truncating to {common_teacher_len}.")
                    args._simct_teacher_len_warned = True
                tea_hidden = tea_hidden[:common_teacher_len]
                tea_input_ids_at_hidden = tea_input_ids_at_hidden[:common_teacher_len]
                tea_hidden_rows = common_teacher_len
        if _any_cp_rank_failed(failed=fatal_simct_msg is not None, device=device):
            sample_alignment_flags.append(0.0)
            continue
        if tea_hidden_rows <= 1:
            sample_alignment_flags.append(0.0)
            continue

        # This sample is non-degenerate and will be aligned: any zero-alignment
        # from here is a REAL alignment failure, not an empty/EOS-only rollout.
        total_alignable_samples += 1.0

        # Teacher label IDs: teacher_input_ids.roll(-1)[teacher_loss_mask].
        # Since teacher_token_ids are input IDs *at masked hidden positions*, this
        # shift is exactly: drop last-prompt input, append teacher EOS target.
        tea_label_ids = tea_input_ids_at_hidden[1:] + [int(teacher_eos_id)]
        stu_label_ids = stu_label_ids_full

        segments = _align_simct_texts_with_spans(
            _simct_pieces_from_cache(teacher_tokenizer, tea_label_ids, tea_piece_cache, use_byte_align),
            _simct_pieces_from_cache(student_tokenizer, stu_label_ids, stu_piece_cache, use_byte_align),
            teacher_tokenizer.eos_token,
            student_tokenizer.eos_token,
            # Peel the terminal EOS into its own 1:1 segment ONLY when the stop bridge
            # is on, so the EOS column it rewrites is populated even on the ~21% of
            # terminals whose last content token tokenizes differently. Off (default)
            # for non-bridge runs -> byte-identical to the verified legacy alignment.
            isolate_terminal_eos=stop_bridge,
        )

        if len(segments) == 0:
            if is_logging_rank and i < 3:
                _logger.warning(
                    f"[OPD][simct][SimCT][sample={i}] alignment FAILED: "
                    f"tea_len={len(tea_label_ids)} stu_len={len(stu_label_ids)} "
                    f"tea_ids[:5]={tea_label_ids[:5]} stu_ids[:5]={stu_label_ids[:5]} "
                    f"tea_texts[:5]={_ids_to_decoded_texts(teacher_tokenizer, tea_label_ids[:5])} "
                    f"stu_texts[:5]={_ids_to_decoded_texts(student_tokenizer, stu_label_ids[:5])}"
                )
            sample_alignment_flags.append(0.0)
            continue

        selected_segments: list[tuple[int, int, int, int]] = []
        selected_first_rows: list[int] = []
        selected_mask_values: list[float] = []
        selected_token_counts: list[int] = []
        virtual_unit_map: dict[int, int] = {}
        sample_dims: list[int] = []
        sample_objective = sample_mode
        # Align-ratio numerator for THIS sample, accumulated over selected
        # segments below.  Counts only CP-local, masked student positions so it
        # shares the exact scope of the total_label_tokens denominator (see the
        # numerator comment at the append site) -> a true <=1 coverage fraction.
        sample_aligned_masked = 0.0

        for tea_start, tea_end, stu_start, stu_end in segments:
            if tea_end > tea_hidden_rows or stu_end > len(stu_label_ids):
                continue
            if stu_end > stu_mask_len:
                continue
            # Cross-tokenizer teachers prefilled on the student's own (often
            # non-terminated) text put ~0 probability on the student EOS, so
            # reverse-KL drives the student's terminal stop-prob to 0 and blocks
            # termination.  Drop ONLY the pure terminal single-token student-EOS
            # segment so the stop signal is not actively suppressed.  Gated by
            # --simct-skip-eos (store_true, DEFAULT OFF = reference span_ctkd
            # behavior; the paper SimCT arm leaves it off for fidelity).
            if simct_skip_eos and _is_terminal_student_eos_segment(
                stu_start, stu_end, stu_label_ids, student_eos_id
            ):
                continue
            first_row = local_pos_by_global.get(stu_start)
            if first_row is None or first_row >= int(stu_logits_shard.shape[0]):
                # SimCT makes one virtual prediction per segment at the first
                # student token.  The CP rank owning that first token owns the
                # segment loss; sparse CP gathers below provide span-tail logits.
                continue

            stu_seg_ids = stu_label_ids[stu_start:stu_end]
            tea_seg_ids = tea_label_ids[tea_start:tea_end]
            is_span = (tea_end - tea_start) > 1 or (stu_end - stu_start) > 1
            if is_span and not span_alignment_enabled:
                # Official SimCT simple_ctkd baseline keeps only 1:1 aligned
                # token positions and computes KD on the shared overlap vocab.
                # Span virtual tokens belong to span_ctkd and are opt-in because
                # they require extra CP gathers and per-span self-logit work.
                continue
            overlap_dim = None
            if not is_span:
                overlap_dim = overlap_pairs.get((int(stu_seg_ids[0]), int(tea_seg_ids[0])))

            if is_span:
                sample_dim_value = None
            elif overlap_dim is not None:
                sample_dim_value = int(overlap_dim)
            elif sample_objective:
                # There is no observed SimCT virtual unit for a 1:1 non-overlap
                # token; sample-token OPD cannot form [p(y), 1-p(y)] for it.
                continue
            else:
                # Official span_ctkd still computes the overlap/virtual-vocab
                # distribution for this aligned position.
                sample_dim_value = 0

            mask_values = stu_mask_values[stu_start:stu_end]
            mask_value = max(mask_values) if mask_values else 0.0
            if mask_value <= 0.0:
                continue
            selected_idx = len(selected_segments)
            if is_span:
                virtual_unit_map[selected_idx] = len(virtual_unit_map)
                sample_dims.append(num_overlap + virtual_unit_map[selected_idx])
            else:
                sample_dims.append(sample_dim_value)
            selected_segments.append((tea_start, tea_end, stu_start, stu_end))
            selected_first_rows.append(int(first_row))
            selected_mask_values.append(mask_value)
            # Student-token count of the segment: the unit the per-token loss
            # denominator (num_tokens = sum of student loss-mask tokens) counts,
            # so weight by the STUDENT side, not max(teacher, student).  This
            # full span still drives total_loss_tokwt (the KD backward weight).
            selected_token_counts.append(stu_end - stu_start)
            # Align-ratio numerator (FIX): the old metric credited this segment's
            # FULL student span (stu_end - stu_start), mask-ignorant, to the CP
            # rank owning its first token, while the denominator total_label_tokens
            # counts only CP-local masked positions -> the ratio could exceed 1 and
            # drift across CP ranks (a CP-attribution artifact, not real coverage).
            # Count instead only the positions of THIS segment that are both
            # CP-local (in local_pos_by_global) and masked (mask value mv), giving
            # the same scope as the denominator -> numerator subset of denominator
            # -> a provably <=1 fraction.  mv (not 1.0) matches the denominator's
            # stu_mask_values weighting for non-binary masks.
            sample_aligned_masked += sum(
                mv
                for off, mv in enumerate(mask_values)
                if (stu_start + off) in local_pos_by_global
            )

        # Commit this sample's CP-local masked aligned coverage to the global
        # numerator.  Same scope as the total_label_tokens denominator added at
        # the top of the per-sample loop (both CP-local + masked).
        total_aligned_tokens += sample_aligned_masked

        target_validation_msg: str | None = None
        span_stu_ids = [
            int(tok)
            for seg_idx, (_, _, stu_start, stu_end) in enumerate(selected_segments)
            if seg_idx in virtual_unit_map
            for tok in stu_label_ids[stu_start:stu_end]
        ]
        span_tea_ids = [
            int(tok)
            for seg_idx, (tea_start, tea_end, _, _) in enumerate(selected_segments)
            if seg_idx in virtual_unit_map
            for tok in tea_label_ids[tea_start:tea_end]
        ]
        if span_stu_ids:
            target_validation_msg = _validate_token_ids_in_vocab_msg(
                torch.tensor(span_stu_ids, dtype=torch.long, device=device),
                real_vocab_size=student_real_vocab_size,
                context=f"[OPD][simct][SimCT] sample {i} student virtual-unit targets",
            )
        if target_validation_msg is None and span_tea_ids:
            target_validation_msg = _validate_token_ids_in_vocab_msg(
                torch.tensor(span_tea_ids, dtype=torch.long, device=lm_head.weight.device),
                real_vocab_size=teacher_real_vocab_size,
                context=f"[OPD][simct][SimCT] sample {i} teacher virtual-unit targets",
            )
        if target_validation_msg is not None:
            fatal_simct_msg = fatal_simct_msg or (
                target_validation_msg
                + ". Recorded failure and will synchronize across DP/CP ranks before raising."
            )
        if _any_cp_rank_failed(failed=fatal_simct_msg is not None, device=device):
            sample_alignment_flags.append(0.0)
            continue

        selected_count = torch.tensor([len(selected_segments)], dtype=torch.int64, device=device)
        if cp_size > 1:
            torch.distributed.all_reduce(selected_count, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        if int(selected_count.item()) <= 0:
            sample_alignment_flags.append(0.0)
            continue

        # Span virtual units may be owned by only a subset of CP ranks because
        # the segment owner is the rank that holds the first student token.  The
        # tail-token logits are recovered through all_gather_with_cp(), whose
        # differentiable all-reduce must be entered by every rank in the CP
        # group.  Therefore gate that CP collective on a group-level "any span"
        # flag, not on this rank's local virtual_unit_map only.
        virtual_units_any = bool(virtual_unit_map)
        if cp_size > 1:
            virtual_units_flag = torch.tensor(
                [1 if virtual_units_any else 0],
                dtype=torch.int32,
                device=device,
            )
            torch.distributed.all_reduce(
                virtual_units_flag,
                op=torch.distributed.ReduceOp.MAX,
                group=mpu.get_context_parallel_group(),
            )
            virtual_units_any = bool(int(virtual_units_flag.item()))

        if virtual_units_any:
            local_label_ids = torch.tensor(
                [int(stu_label_ids[gidx]) for gidx in local_response_indices[: int(stu_logits_shard.shape[0])]],
                dtype=torch.long,
                device=device,
            )
            # Gather only the observed student-token logits needed by span virtual
            # units.  The previous code converted the entire [tokens, vocab]
            # shard to fp32 before gather, and did so whenever CP>1 even if the
            # batch had no span units.  For 32k responses this created multi-GB
            # temporaries per microbatch.
            local_self_logits = _sample_token_logits_from_logits(
                stu_logits_shard[: local_label_ids.numel()],
                local_label_ids,
                vocab_start=student_vocab_start,
                tp_size=tp_size,
            ).float() / opd_temperature
            if cp_size > 1:
                full_stu_self_logits = all_gather_with_cp(local_self_logits, total_len, response_len)
                # Keep the differentiable CP gather in every rank's backward graph
                # when span virtual units need cross-CP tail logits.  Attach the
                # zero-value keepalive to BOTH backward accumulators (total_loss
                # for the non-per-token path, total_loss_tokwt for the per-token
                # path) so the per-token backward does not drop this CP edge.
                total_loss = total_loss + full_stu_self_logits.sum() * 0.0
                total_loss_tokwt = total_loss_tokwt + full_stu_self_logits.sum() * 0.0
            else:
                full_stu_self_logits = local_self_logits
        else:
            full_stu_self_logits = stu_logits_shard.new_empty((response_len,), dtype=torch.float32)

        virtual_dims = len(virtual_unit_map)
        chunk_size = max(int(getattr(args, "simct_chunk_size", 64) or 64), 1)
        compile_bucket_size = 0
        if distill_scope == "full":
            compile_bucket_size = _effective_simct_compile_bucket_size(
                bucket_size=int(getattr(args, "simct_compile_bucket_size", 0) or 0),
                chunk_size=chunk_size,
                distill_scope=distill_scope,
            )
        for start in range(0, len(selected_segments), chunk_size):
            end = min(start + chunk_size, len(selected_segments))
            seg_chunk = selected_segments[start:end]
            first_rows = selected_first_rows[start:end]
            first_tea_rows = [seg[0] for seg in seg_chunk]
            sample_dim_values = [int(x) for x in sample_dims[start:end]]
            (
                seg_chunk,
                first_rows,
                first_tea_rows,
                mask_values_chunk,
                token_counts_chunk,
                sample_dim_values,
                real_rows_in_chunk,
            ) = _pad_simct_chunk_for_compile(
                seg_chunk=seg_chunk,
                first_rows=first_rows,
                first_tea_rows=first_tea_rows,
                mask_values=selected_mask_values[start:end],
                token_counts=selected_token_counts[start:end],
                sample_dims=sample_dim_values,
                bucket_size=compile_bucket_size,
            )
            first_rows_t = torch.tensor(first_rows, dtype=torch.long, device=device)
            tea_first_logits = None
            if topk_mode:
                tea_first_hidden = _select_hidden_rows_to_device(
                    tea_hidden,
                    first_tea_rows,
                    device=lm_head.weight.device,
                    dtype=lm_head.weight.dtype,
                )
                tea_first_logits = lm_head(tea_first_hidden).float() / opd_temperature

            sample_dim_t = torch.tensor(sample_dim_values, dtype=torch.long, device=device)
            aligned_mask_float = torch.tensor(mask_values_chunk, dtype=torch.float32, device=device)
            segment_token_counts_t = torch.tensor(token_counts_chunk, dtype=torch.float32, device=device)

            def _build_row_local_span_extra_logits() -> tuple[torch.Tensor, torch.Tensor]:
                """Return the single row-local SimCT virtual span column."""
                stu_extra = torch.full((len(seg_chunk), 1), -1.0e9, dtype=torch.float32, device=device)
                tea_extra = torch.full((len(seg_chunk), 1), -1.0e9, dtype=torch.float32, device=device)
                flat_tea_rows: list[int] = []
                flat_tea_ids: list[int] = []
                span_slices: dict[int, tuple[int, int]] = {}
                for offset, (tea_start, tea_end, _, _) in enumerate(seg_chunk):
                    global_seg_idx = start + offset
                    if global_seg_idx not in virtual_unit_map:
                        continue
                    span_slices[offset] = (len(flat_tea_rows), len(flat_tea_rows) + (tea_end - tea_start))
                    flat_tea_rows.extend(range(tea_start, tea_end))
                    flat_tea_ids.extend(int(tok) for tok in tea_label_ids[tea_start:tea_end])
                if flat_tea_rows:
                    tea_span_ids = torch.tensor(flat_tea_ids, dtype=torch.long, device=lm_head.weight.device)
                    tea_span_hidden = _select_hidden_rows_to_device(
                        tea_hidden,
                        flat_tea_rows,
                        device=lm_head.weight.device,
                        dtype=lm_head.weight.dtype,
                    )
                    if tp_size == 1:
                        tea_self_all = (
                            _simct_lm_head_token_logits(
                                lm_head,
                                tea_span_hidden,
                                tea_span_ids,
                                vocab_start=teacher_vocab_start,
                            ).to(device=device)
                            / opd_temperature
                        )
                    else:
                        tea_span_logits = lm_head(tea_span_hidden).float() / opd_temperature
                        tea_self_all = _sample_token_logits_from_logits(
                            tea_span_logits,
                            tea_span_ids,
                            vocab_start=teacher_vocab_start,
                            tp_size=tp_size,
                        ).to(device=device)
                else:
                    tea_self_all = stu_logits_shard.new_empty((0,), dtype=torch.float32)
                for offset, (_, _, stu_start, stu_end) in enumerate(seg_chunk):
                    global_seg_idx = start + offset
                    if global_seg_idx not in virtual_unit_map:
                        continue
                    tea_slice = span_slices[offset]
                    stu_self_logits = full_stu_self_logits[stu_start:stu_end]
                    tea_self_logits = tea_self_all[tea_slice[0]:tea_slice[1]]
                    stu_extra[offset, 0] = stu_self_logits.mean()
                    tea_extra[offset, 0] = tea_self_logits.detach().mean()
                return stu_extra, tea_extra

            # Fast top-K path: select exact teacher top-K over overlap tokens using
            # shard-local teacher top-K candidates, then exchange only K values per
            # TP rank.  This avoids the old [segments, overlap_vocab] teacher gather
            # (143k columns in the GLM/Qwen run) and only gathers K student logits.
            if topk_mode:
                k = min(int(effective_k), int(num_overlap + virtual_dims))
                if k <= 1:
                    raise ValueError("[OPD][simct][SimCT] top-K virtual vocab scope requires K >= 2")

                tea_topk_vals, tea_topk_ids = _simct_teacher_overlap_topk_candidates(
                    tea_first_logits,
                    tea_overlap_ids.to(device=lm_head.weight.device),
                    k=k,
                    args=args,
                    vocab_start=teacher_vocab_start,
                    shard_vocab=teacher_shard_vocab,
                    real_vocab_size=teacher_real_vocab_size,
                    tp_size=tp_size,
                    tp_group=tp_group,
                    temperature=opd_temperature,
                )
                tea_topk_vals = tea_topk_vals.to(device=device)
                tea_topk_ids = tea_topk_ids.to(device=device)
                stu_topk_ids = _get_simct_teacher_to_student_id_map(
                    args,
                    tea_overlap_ids,
                    stu_overlap_ids,
                    teacher_real_vocab_size=teacher_real_vocab_size,
                    device=device,
                ).gather(0, tea_topk_ids.reshape(-1)).reshape_as(tea_topk_ids)
                topk_stu_vals = _gather_tp_logits_at_rows_global_ids_subset(
                    stu_logits_shard,
                    first_rows_t,
                    stu_topk_ids,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    tp_group=tp_group,
                ).float() / opd_temperature

                if virtual_dims:
                    # Other samples' span units are -inf for the current row and
                    # therefore contribute exactly zero probability.  Appending a
                    # single row-local virtual column preserves the SimCT full
                    # support while keeping the compiled shape stable at
                    # overlap_vocab + 1 instead of overlap_vocab + num_spans_in_sample.
                    stu_extra, tea_extra = _build_row_local_span_extra_logits()
                    virtual_k = min(k, int(stu_extra.shape[-1]))
                    tea_virtual_vals, virtual_pos = tea_extra.detach().topk(virtual_k, dim=-1)
                    stu_virtual_vals = stu_extra.gather(-1, virtual_pos)
                    candidate_tea_vals = torch.cat([tea_topk_vals, tea_virtual_vals], dim=-1)
                    candidate_stu_vals = torch.cat([topk_stu_vals, stu_virtual_vals], dim=-1)
                    final_k = min(k, int(candidate_tea_vals.shape[-1]))
                    tea_topk_vals, final_pos = candidate_tea_vals.detach().topk(final_k, dim=-1)
                    topk_stu_vals = candidate_stu_vals.gather(-1, final_pos)

                stu_log_probs = torch.log_softmax(topk_stu_vals, dim=-1, dtype=torch.float32)
                tea_log_probs = torch.log_softmax(tea_topk_vals, dim=-1, dtype=torch.float32)
                per_aligned_loss = _divergence_from_log_probs(
                    stu_log_probs,
                    tea_log_probs,
                    loss_type=opd_loss_type,
                    jsd_beta=opd_jsd_beta,
                )
                if loss_only_diagnostics:
                    tea_entropy = stu_entropy = None
                else:
                    with torch.no_grad():
                        tea_probs = tea_log_probs.exp()
                        stu_probs = stu_log_probs.exp()
                        tea_entropy = -(tea_probs * tea_log_probs).sum(dim=-1)
                        stu_entropy = -(stu_probs * stu_log_probs).sum(dim=-1)
            else:
                # Full overlap-vocab path.  The loss-only fKL/RKL case uses the
                # same algebra as stoken full-vocab KL: compute partition
                # functions and weighted sums directly instead of materializing
                # every log-probability/probability tensor.  Positive
                # --opd-simct-overlap-chunk-size additionally streams overlap
                # columns to lower peak memory; JSD/entropy diagnostics keep the
                # dense path in this surgical pass because avoiding their
                # intermediates needs a separate two-pass streaming helper.
                raw_overlap_chunk_size = int(getattr(args, "simct_overlap_chunk_size", 0) or 0)
                use_streaming_full_loss = (
                    distill_scope == "full"
                    and loss_only_diagnostics
                    and opd_loss_type in ("fkl", "kl", "rkl")
                    and 0 < raw_overlap_chunk_size < num_overlap
                    # The stop bridge rewrites the materialized EOS column on the
                    # dense path; force dense so the chunked streamer never bypasses it.
                    and not stop_bridge
                )
                if use_streaming_full_loss:
                    overlap_chunk_size = raw_overlap_chunk_size or num_overlap
                    overlap_chunk_size = max(overlap_chunk_size, 1)
                    tea_first_hidden = _select_hidden_rows_to_device(
                        tea_hidden,
                        first_tea_rows,
                        device=lm_head.weight.device,
                        dtype=lm_head.weight.dtype,
                    )
                    tea_first_logits = None
                    if tp_size > 1:
                        tea_first_logits = lm_head(tea_first_hidden).float() / opd_temperature

                    def _iter_full_overlap_logit_chunks():
                        for col_start in range(0, num_overlap, overlap_chunk_size):
                            col_end = min(col_start + overlap_chunk_size, num_overlap)
                            stu_ids_chunk = stu_overlap_ids[col_start:col_end]
                            tea_ids_chunk = tea_overlap_ids[col_start:col_end]
                            stu_logits_chunk = _gather_tp_logits_at_rows_global_ids_prevalidated(
                                stu_logits_shard,
                                first_rows_t,
                                stu_ids_chunk,
                                tp_size=tp_size,
                                tp_rank=tp_rank,
                                tp_group=tp_group,
                            ).float() / opd_temperature
                            if tp_size == 1:
                                tea_logits_chunk = (
                                    _simct_lm_head_at_global_ids(
                                        lm_head,
                                        tea_first_hidden,
                                        tea_ids_chunk.to(device=lm_head.weight.device),
                                        vocab_start=teacher_vocab_start,
                                    ).to(device=device)
                                    / opd_temperature
                                )
                            else:
                                tea_logits_chunk = _gather_tp_logits_at_global_ids_prevalidated(
                                    tea_first_logits,
                                    tea_ids_chunk.to(device=lm_head.weight.device),
                                    tp_size=tp_size,
                                    tp_rank=tp_rank,
                                    tp_group=tp_group,
                                ).to(device=device)
                            yield stu_logits_chunk, tea_logits_chunk

                    def _iter_full_virtual_logit_chunks():
                        yield from _iter_full_overlap_logit_chunks()
                        if virtual_dims:
                            yield _build_row_local_span_extra_logits()

                    per_aligned_loss = _simct_streaming_full_virtual_vocab_loss_only_from_chunks(
                        _iter_full_virtual_logit_chunks(),
                        loss_type=opd_loss_type,
                        jsd_beta=opd_jsd_beta,
                    )
                    tea_entropy = stu_entropy = None
                else:
                    # Dense fallback for JSD, sample/top-k fallback handling, and
                    # explicit entropy diagnostics.  It keeps the historical exact
                    # path and should be used only when those non-separable outputs
                    # are required.
                    stu_overlap_logits = _gather_tp_logits_at_rows_global_ids_prevalidated(
                        stu_logits_shard,
                        first_rows_t,
                        stu_overlap_ids,
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                        tp_group=tp_group,
                    ).float() / opd_temperature
                    tea_first_hidden = _select_hidden_rows_to_device(
                        tea_hidden,
                        first_tea_rows,
                        device=lm_head.weight.device,
                        dtype=lm_head.weight.dtype,
                    )
                    if tp_size == 1:
                        # Full simct support is the overlap vocab, not the teacher's
                        # entire vocab.  Project hidden states directly to overlap
                        # columns instead of computing [chunk, teacher_vocab] and
                        # discarding non-overlap logits.
                        tea_overlap_logits = (
                            _simct_lm_head_at_global_ids(
                                lm_head,
                                tea_first_hidden,
                                tea_overlap_ids.to(device=lm_head.weight.device),
                                vocab_start=teacher_vocab_start,
                            ).to(device=device)
                            / opd_temperature
                        )
                    else:
                        tea_first_logits = lm_head(tea_first_hidden).float() / opd_temperature
                        tea_overlap_logits = _gather_tp_logits_at_global_ids_prevalidated(
                            tea_first_logits,
                            tea_overlap_ids.to(device=lm_head.weight.device),
                            tp_size=tp_size,
                            tp_rank=tp_rank,
                            tp_group=tp_group,
                        ).to(device=device)

                    if stop_bridge and teacher_stop_ids_t is not None and tea_overlap_logits.numel():
                        # Cross-family stop bridge: replace the single-token EOS column
                        # with the log-sum-exp of the teacher's FULL stop set (same
                        # post-temperature space as tea_overlap_logits), so a teacher
                        # that ends its turn on a non-mapped token (GLM <|user|>) still
                        # transfers stop mass to the student EOS column.  TP=1 only
                        # (guarded at setup); reuses the same overlap projection helper.
                        stop_logits = (
                            _simct_lm_head_at_global_ids(
                                lm_head,
                                tea_first_hidden,
                                teacher_stop_ids_t,
                                vocab_start=teacher_vocab_start,
                            ).to(device=device)
                            / opd_temperature
                        )
                        eos_col = torch.logsumexp(stop_logits, dim=-1, keepdim=True)
                        tea_overlap_logits = torch.cat(
                            [
                                tea_overlap_logits[:, :eos_bridge_col_idx],
                                eos_col,
                                tea_overlap_logits[:, eos_bridge_col_idx + 1 :],
                            ],
                            dim=-1,
                        )

                    if (
                        stop_bridge
                        and student_stop_ids_t is not None
                        and student_stop_ids_t.numel() > 1
                        and stu_overlap_logits.numel()
                    ):
                        # Symmetric student side: replace the student EOS column with the
                        # log-sum-exp of the student's OWN stop set, so the KL compares the
                        # student's TOTAL stop probability to the teacher's.  Gradient still
                        # flows through the student (only the EOS column is reshaped).  No-op
                        # when the student has a single stop token (guarded by numel()>1).
                        stu_stop_logits = (
                            _gather_tp_logits_at_rows_global_ids_prevalidated(
                                stu_logits_shard,
                                first_rows_t,
                                student_stop_ids_t,
                                tp_size=tp_size,
                                tp_rank=tp_rank,
                                tp_group=tp_group,
                            ).float()
                            / opd_temperature
                        )
                        stu_eos_col = torch.logsumexp(stu_stop_logits, dim=-1, keepdim=True)
                        stu_overlap_logits = torch.cat(
                            [
                                stu_overlap_logits[:, :eos_bridge_col_idx],
                                stu_eos_col,
                                stu_overlap_logits[:, eos_bridge_col_idx + 1 :],
                            ],
                            dim=-1,
                        )

                    if virtual_dims:
                        # Other samples' span units are -inf for the current row and
                        # therefore contribute exactly zero probability.  Appending a
                        # single row-local virtual column preserves the SimCT full
                        # support while keeping the compiled shape stable at
                        # overlap_vocab + 1 instead of overlap_vocab + num_spans_in_sample.
                        stu_extra, tea_extra = _build_row_local_span_extra_logits()
                        stu_virtual_logits = torch.cat([stu_overlap_logits, stu_extra], dim=-1)
                        tea_virtual_logits = torch.cat([tea_overlap_logits, tea_extra], dim=-1)
                        sample_dim_t = torch.tensor(
                            [
                                num_overlap if (start + offset) in virtual_unit_map else int(sample_dim_values[offset])
                                for offset in range(len(seg_chunk))
                            ],
                            dtype=torch.long,
                            device=device,
                        )
                    else:
                        stu_virtual_logits = stu_overlap_logits
                        tea_virtual_logits = tea_overlap_logits

                    if distill_scope == "full":
                        if loss_only_diagnostics:
                            per_aligned_loss = _simct_full_virtual_vocab_loss_only_fused(
                                stu_virtual_logits,
                                tea_virtual_logits,
                                loss_type=opd_loss_type,
                                jsd_beta=opd_jsd_beta,
                            )
                            tea_entropy = stu_entropy = None
                        else:
                            per_aligned_loss, tea_entropy, stu_entropy = _simct_full_virtual_vocab_loss_and_entropy_fused(
                                stu_virtual_logits,
                                tea_virtual_logits,
                                loss_type=opd_loss_type,
                                jsd_beta=opd_jsd_beta,
                            )
                    else:
                        if loss_only_diagnostics:
                            per_aligned_loss = _simct_virtual_vocab_loss_only_from_logits(
                                stu_virtual_logits,
                                tea_virtual_logits,
                                loss_type=opd_loss_type,
                                jsd_beta=opd_jsd_beta,
                                distill_scope=distill_scope,
                                effective_k=effective_k,
                                sample_dim=sample_dim_t,
                            )
                            tea_entropy = stu_entropy = None
                        else:
                            per_aligned_loss, tea_entropy, stu_entropy, _ = _simct_virtual_vocab_loss_from_logits(
                                stu_virtual_logits,
                                tea_virtual_logits,
                                loss_type=opd_loss_type,
                                jsd_beta=opd_jsd_beta,
                                distill_scope=distill_scope,
                                effective_k=effective_k,
                                sample_dim=sample_dim_t,
                            )

            # KDFlow/SimCT constructs one virtual-token prediction per aligned
            # segment (the geometric-mean span logit already combines a span's
            # student tokens into one prediction).  total_loss / total_kd keep
            # this per-SEGMENT sum for the reported per-segment means and the
            # non-per-token backward path.  total_loss_tokwt additionally weights
            # each segment by its student-token count so the per-token-loss
            # backward objective matches its per-student-token denominator
            # (otherwise an M-token span is silently down-weighted 1/M).
            masked_loss = per_aligned_loss * aligned_mask_float
            sample_term = masked_loss.sum()
            total_loss = total_loss + sample_term
            total_kd = total_kd + sample_term
            total_loss_tokwt = total_loss_tokwt + (masked_loss * segment_token_counts_t).sum()
            row_mask_sum = float(aligned_mask_float.sum().detach().item())
            total_aligned_rows += row_mask_sum
            if track_per_sample:
                # per_sample mirrors the False/per_rank backward (total_loss / cp_aligned_rows),
                # but normalizes each sample by its OWN aligned-row count: keep this ORIGINAL
                # sample i's CP-LOCAL grad-carrying distill SUM and aligned-row (segment) count.
                per_sample_distill[i] = per_sample_distill.get(i, 0.0) + sample_term
                per_sample_aligned[i] = per_sample_aligned.get(i, 0.0) + row_mask_sum
            # NOTE: total_aligned_tokens (align-ratio numerator) is accumulated in
            # the per-sample selection loop above as CP-local MASKED coverage, NOT
            # here as the full-span (segment_token_counts) sum, so it shares the
            # denominator's scope and stays <=1.  segment_token_counts still drives
            # total_loss_tokwt (the KD backward weight) at line above.

            with torch.no_grad():
                if tea_entropy is not None and stu_entropy is not None:
                    total_tea_entropy += (tea_entropy * aligned_mask_float).sum()
                    total_stu_entropy += (stu_entropy * aligned_mask_float).sum()
                total_overlap_count += float(aligned_mask_float.sum().detach().item())

        sample_alignment_flags.append(float(sum(
            count * float(mask)
            for count, mask in zip(selected_token_counts, selected_mask_values, strict=False)
        ) > 0.0))

        if not getattr(args, "_simct_first_logged", False) and is_logging_rank:
            # Same CP-local masked coverage as the fixed global metric (was the
            # full-span count * mask sum, which could exceed local_label_tokens).
            local_mask_sum = sample_aligned_masked
            align_ratio = local_mask_sum / max(local_label_tokens, 1.0)
            n_span_units = sum(1 for seg in selected_segments if (seg[1] - seg[0]) > 1 or (seg[3] - seg[2]) > 1)
            _logger.info(
                f"[OPD][simct][SimCT][sample=0 diag] "
                f"stu_logits_shard={tuple(stu_logits_shard.shape)} tea_hidden={_hidden_rows_shape(tea_hidden)} "
                f"segments={len(selected_segments)}/{len(segments)} span_units={n_span_units} "
                f"virtual_dim={num_overlap + virtual_dims} valid_tokens={local_mask_sum:.0f} "
                f"local_label_tokens={local_label_tokens:.0f} align_ratio={align_ratio:.3f} "
                f"overlap={num_overlap} scope={distill_scope} topk_mode={topk_mode} k={effective_k} tp={tp_size} cp={cp_size} "
            )
            _logger.info(
                f"[OPD][simct][SimCT][sample=0 align] "
                f"tea_label_ids[:5]={tea_label_ids[:5]} "
                f"stu_label_ids[:5]={stu_label_ids[:5]} "
                f"segments[:5]={segments[:5]} "
                f"local_response_indices[:5]={local_response_indices[:5]}"
            )
            args._simct_first_logged = True
    # The reference OPD objective divides the summed KD loss by a token count
    # before backward.  In slime's Megatron wrapper, calculate_per_token_loss=True
    # returns num_tokens to the schedule, so the loss tensor returned here must be
    # the token-summed loss.  The schedule/global normalizer then supplies the
    # global-token denominator.  Metrics below still report per-token means.
    _raise_if_any_dp_cp_rank_failed(
        failed=fatal_simct_msg is not None,
        local_message=fatal_simct_msg,
        device=device,
        context="[OPD][simct] per-sample validation",
    )
    global_aligned_tokens, global_label_tokens, global_alignable_samples = _reduce_dp_cp_float_counts(
        [float(total_aligned_tokens), float(total_label_tokens), float(total_alignable_samples)],
        device=device,
    )
    # CP-shared (per-DP-rank) aligned-row count for the calculate_per_token_loss=False
    # backward denominator.  total_aligned_rows is CP-LOCAL; the schedule SUMS the
    # CP ranks' returns, so dividing each by its CP-local count inflates the summed
    # loss/grad by ~cp_size.  Mirror cp_utils mask_sum (CP-shared, per-DP-rank).
    cp_aligned_rows = _reduce_cp_float_counts([float(total_aligned_rows)], device=device)[0]
    kd_loss_mean = total_loss / max(total_aligned_rows, 1.0)
    # total_aligned_tokens and total_label_tokens now share the SAME scope
    # (CP-local + masked student positions), so both ratios are true coverage
    # fractions in [0, 1] (numerator is a subset of the denominator).  Both the
    # local and the CP-DP-reduced global ratio slightly UNDER-estimate coverage
    # for spans straddling a CP-chunk boundary: such a span is owned by the rank
    # holding its first token, so its tail positions living on another CP rank
    # are counted by neither rank's numerator (the owner can't -- they are not
    # CP-local to it; the other rank never selected the span).  This is a benign
    # lower bound (minority M:N spans only); it can never push the ratio above 1.
    align_ratio_mean = total_aligned_tokens / max(total_label_tokens, 1.0)
    global_align_ratio_mean = global_aligned_tokens / max(global_label_tokens, 1.0)
    min_align_ratio = float(getattr(args, "simct_min_align_ratio", 0.0) or 0.0)
    if global_label_tokens <= 0:
        zero_loss = logits.sum() * 0.0
        if track_per_sample:
            args._opd_per_sample_normalizer = 0.0
        if is_logging_rank:
            _logger.info(
                "[OPD][simct] microbatch has no unmasked OPD tokens; "
                "returning zero loss. This is expected when standalone OPD "
                "masks truncated/repetitive rollout samples."
            )
        return zero_loss, {"loss": zero_loss.detach(), "simct_opd_loss": zero_loss.detach()}
    if global_aligned_tokens <= 0:
        if global_alignable_samples <= 0:
            # No sample in this DP/CP group was even alignable (all empty/EOS-only
            # or degenerate rollouts) -> benign zero loss, NOT a tokenizer mismatch.
            # Common at early steps of a base (non-SFT-warm) student.
            zero_loss = logits.sum() * 0.0
            if track_per_sample:
                args._opd_per_sample_normalizer = 0.0
            if is_logging_rank:
                _logger.info(
                    "[OPD][simct] microbatch group has only degenerate "
                    "(empty/EOS-only) rollout samples; returning zero loss."
                )
            return zero_loss, {"loss": zero_loss.detach(), "simct_opd_loss": zero_loss.detach()}
        raise ValueError(
            "[OPD][simct] no aligned simct tokens despite "
            f"{int(global_alignable_samples)} alignable sample(s) in this DP/CP "
            "microbatch group. This usually means teacher/student token IDs or "
            "chat templates are mismatched. Check raw_messages, teacher_token_ids, "
            "and tokenizer paths."
        )
    if min_align_ratio > 0 and global_align_ratio_mean < min_align_ratio:
        raise ValueError(
            f"[OPD][simct] global_align_ratio={global_align_ratio_mean:.4f} below "
            f"--simct-min-align-ratio={min_align_ratio:.4f}. This optional debug "
            "guard is disabled by default; lower the threshold to continue with "
            "SimCT's token-normalized objective."
        )
    if track_per_sample:
        # TRL GOLD per-sample-mean-then-batch-mean: each sample normalized by its OWN aligned
        # (segment/row) count, then averaged over the samples that have alignment.
        # per_sample_distill[i] stays CP-LOCAL (keeps the autograd graph on this rank's tokens);
        # only the scalar row COUNTS are CP-reduced to their full per-DP-rank value.  The schedule
        # SUMS CP ranks' returns, so
        #   Σ_cp Σ_s (sd_cplocal,s / full_rows_s) / N  =  Σ_s (full_sd_s / full_rows_s) / N
        # = exact per-sample mean.  Identical at CP=1 (_reduce_cp_float_counts is a no-op there).
        if opd_ce_weight > 0:
            raise NotImplementedError(
                "--opd-loss-reduction per_sample with --opd-ce-weight>0 is not implemented yet "
                "(per-sample CE normalization). Use --opd-ce-weight 0 (the default) for per_sample."
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
        # EXACT per-sample-mean-then-batch-mean via calculate_per_token_loss=True: return the
        # per-sample NUMERATOR and stash the sample COUNT as Megatron's per-token normalizer, so the
        # schedule DP+CP-sums both -> grads = Σ_all(sd_s/aligned_s)/N_global exactly (also correct
        # under --use-dynamic-batch-size).
        loss = sample_mean_sum
        args._opd_per_sample_normalizer = n_samples_ps
        # CP tail anchor (CRITICAL): per_sample_distill only references the OWNER rank's tokens
        # per sample, so on a non-owner CP rank a sample's cross-CP all_gather_with_cp node
        # (folded into total_loss at simct_loss.py's `total_loss + full_stu_self_logits.sum()*0.0`
        # keepalive) would be ABSENT from this rank's autograd graph -> the differentiable all-reduce
        # backward fires asymmetrically -> NCCL hang at CP>1.  total_loss aggregates those anchors
        # over ALL samples on ALL ranks, so `+ 0.0*total_loss` (zero value, zero grad) keeps them in
        # the graph on every rank.
        loss = loss + 0.0 * total_loss
    else:
        if args.calculate_per_token_loss:
            if getattr(args, "simct_span_ctkd_norm", False):
                # span_ctkd-FAITHFUL normalization: weight-1-per-SEGMENT numerator
                # (total_loss) with Megatron's per-token denominator.  The reference
                # divides the segment RKL sum by avg_micro_batch_token_num per
                # microbatch and by accumulated_gradient at the optimizer
                # (span_ctkd.py:430, on_policy_kd_trainer.py:172, fsdp_strategy.py:351),
                # which telescopes to (sum over ALL segments)/(global-batch student
                # loss-mask token count) -- exactly what Megatron computes when we
                # return the CP-local segment sum here under per_token reduction.
                # Neither default matched: total_loss_tokwt token-weights segments;
                # per_rank divides by SEGMENT count (factor = mean span length,
                # span-density-dependent -- fidelity audit 2026-07-07).
                kd_loss_for_backward = total_loss
            else:
                # Megatron's per-token-loss path divides the returned scalar by the
                # dispatcher-provided response-token count (sum of student-token masks).
                # Use the token-weighted KD sum so numerator and denominator share the
                # per-student-token unit (uniform per-aligned-token weight; multi-token
                # spans no longer down-weighted 1/M).  We still do NOT rescale by the
                # alignment ratio: a low cross-tokenizer alignment ratio reduces this
                # microbatch's KD mass (unaligned tokens stay in the denominator) rather
                # than being amplified back to an aligned-token mean.
                kd_loss_for_backward = total_loss_tokwt
        else:
            # Divide by the CP-SHARED full aligned-row count (cp-summed), not the
            # CP-local partial count (kd_loss_mean), else the schedule's CP-sum makes
            # the backward loss ~cp_size too big.  kd_loss_mean stays the per-rank
            # reported metric below.
            kd_loss_for_backward = total_loss / max(cp_aligned_rows, 1.0)
        loss = kd_loss_for_backward

    if opd_ce_weight > 0:
        ce_total = torch.tensor(0.0, dtype=torch.float32, device=device)
        ce_denom = 0.0
        for i in range(len(response_lengths)):
            stu_logits_i = stu_logits_list[i]
            labels_i = stu_label_chunks[i][: stu_logits_i.shape[0]]
            mask_i = loss_masks[i]
            if mask_i.numel() != stu_logits_i.shape[0]:
                if cp_size > 1:
                    idxs = _get_simct_local_response_indices(int(total_lengths[i]), int(response_lengths[i]))
                    idxs = idxs[: stu_logits_i.shape[0]]
                    mask_i = mask_i[idxs]
                else:
                    mask_i = mask_i[-int(response_lengths[i]):]
            mask_i = mask_i[: stu_logits_i.shape[0]].to(device=device)
            # TP>1 is guarded above.  TP=1 logits are full vocab.  Use the same
            # temperature convention as standard CE: no OPD temperature scaling.
            per_ce = F.cross_entropy(
                stu_logits_i.float(),
                labels_i.to(device=stu_logits_i.device),
                reduction="none",
            )
            ce_total += (per_ce * mask_i.float()).sum()
            ce_denom += float(mask_i.float().sum().detach().item())
        # CP-shared CE denominator for the same reason as the KD term above.
        cp_ce_denom = _reduce_cp_float_counts([float(ce_denom)], device=device)[0]
        ce_loss_mean = ce_total / max(cp_ce_denom, 1.0)
        ce_loss = ce_total if args.calculate_per_token_loss else ce_loss_mean
        loss = opd_ce_weight * ce_loss + (1.0 - opd_ce_weight) * kd_loss_for_backward

    # Keep a gradient edge even if all alignments fail.
    loss = loss + 0.0 * logits.sum()

    denom = float(max(total_aligned_rows, 1.0))
    mean_kd_loss = (total_kd / denom).detach()
    entropy_diagnostics_enabled = not loss_only_diagnostics
    if entropy_diagnostics_enabled:
        mean_tea_entropy = (total_tea_entropy / denom).detach()
        mean_stu_entropy = (total_stu_entropy / denom).detach()
    else:
        mean_tea_entropy = mean_stu_entropy = total_loss.new_tensor(float("nan")).detach()
    sample_count = _count_cp_unique_samples_with_alignment(sample_alignment_flags, device=device)

    log_simct_step_summary(
        args,
        is_logging_rank=is_logging_rank,
        kd_loss_mean=kd_loss_mean,
        mean_kd_loss=mean_kd_loss,
        mean_tea_entropy=mean_tea_entropy,
        mean_stu_entropy=mean_stu_entropy,
        entropy_diagnostics_enabled=entropy_diagnostics_enabled,
        opd_loss_type=opd_loss_type,
        num_overlap=num_overlap,
        align_ratio_mean=align_ratio_mean,
        global_align_ratio_mean=global_align_ratio_mean,
        total_aligned_tokens=total_aligned_tokens,
        global_aligned_tokens=global_aligned_tokens,
        total_label_tokens=total_label_tokens,
        global_label_tokens=global_label_tokens,
        sample_count=sample_count,
        topk_mode=topk_mode,
        effective_k=effective_k,
        tp_size=tp_size,
        cp_size=cp_size,
    )

    metric_kwargs = {
        "device": device,
        "simct_loss_metric_name": simct_loss_metric_name,
        "mean_kd_loss": mean_kd_loss,
        "mean_tea_entropy": mean_tea_entropy,
        "mean_stu_entropy": mean_stu_entropy,
        "entropy_diagnostics_enabled": entropy_diagnostics_enabled,
        "total_overlap_count": float(total_overlap_count),
        "align_ratio_mean": align_ratio_mean,
        "total_aligned_tokens": total_aligned_tokens,
        "total_label_tokens": total_label_tokens,
        "sample_count": sample_count,
    }
    if _SIMCT_TRAIN_METRICS_ACCEPTS_METRIC_WEIGHT:
        metric_kwargs["metric_weight"] = total_aligned_rows
    accumulate_simct_train_metrics(args, **metric_kwargs)

    return (
        loss,
        {
            "loss": loss.clone().detach(),
            "simct_opd_loss": loss.clone().detach(),
        },
    )
