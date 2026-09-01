"""OPD train-metric reduction for the BPM release.

Loss functions accumulate raw per-microbatch sums on `args`; this reduces them over
the DP-with-CP group and writes the public bpm_* payload into loss_reduced.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterable

import torch
from megatron.core import mpu


def _reduce_tensor_or_float_metrics(metrics: dict, keys: Iterable[str]) -> None:
    """All-reduce tensor/float metric sums in-place across DP-with-CP ranks."""
    reducible_keys = []
    values = []
    for key in keys:
        value = metrics[key]
        if torch.is_tensor(value):
            values.append(value.detach().to(device=torch.cuda.current_device(), dtype=torch.float64))
            reducible_keys.append(key)
        elif isinstance(value, (int, float)):
            values.append(torch.tensor(float(value), dtype=torch.float64, device=torch.cuda.current_device()))
            reducible_keys.append(key)
    if not values:
        return
    values_to_reduce = torch.stack([v.reshape(()) for v in values])
    reduce_group = mpu.get_data_parallel_group(with_context_parallel=True)
    if torch.distributed.get_world_size(group=reduce_group) > 1:
        torch.distributed.all_reduce(
            values_to_reduce,
            op=torch.distributed.ReduceOp.SUM,
            group=reduce_group,
        )
    for key, value in zip(reducible_keys, values_to_reduce.tolist(), strict=False):
        metrics[key] = value


def _add_bpm_opd_metrics(args: Namespace, loss_reduced: dict[str, float]) -> None:
    """Reduce and publish BPM cross-tokenizer distillation diagnostics."""
    bpm_opd_metrics = getattr(args, "_bpm_opd_metrics", None)
    if bpm_opd_metrics is None:
        return

    numeric_keys = [key for key in bpm_opd_metrics if key != "_bpm_opd_loss_type"]
    _reduce_tensor_or_float_metrics(bpm_opd_metrics, numeric_keys)

    total_aligned = max(bpm_opd_metrics["bpm_aligned_tokens_sum"], 1)
    total_label = max(bpm_opd_metrics["bpm_num_tokens_sum"], 1)
    loss_weight = max(bpm_opd_metrics.get("bpm_loss_weight_sum", total_aligned), 1)
    bpm_sample_count = max(bpm_opd_metrics.get("bpm_num_samples_with_alignment_sum", 1), 1)
    loss_reduced["bpm_loss"] = bpm_opd_metrics["bpm_loss_sum"] / loss_weight

    entropy_tokens = bpm_opd_metrics.get("bpm_entropy_token_sum", loss_weight)
    # teacher entropy is fast-rows-only, so divide by the fast row count
    tea_entropy_tokens = bpm_opd_metrics.get("bpm_tea_entropy_token_sum", 0.0) or entropy_tokens
    if entropy_tokens > 0:
        loss_reduced["bpm_tea_entropy"] = bpm_opd_metrics["bpm_tea_entropy_sum"] / max(tea_entropy_tokens, 1)
        loss_reduced["bpm_stu_entropy"] = bpm_opd_metrics["bpm_stu_entropy_sum"] / entropy_tokens
        loss_reduced["bpm_entropy_diagnostics_enabled"] = 1.0
    else:
        loss_reduced["bpm_entropy_diagnostics_enabled"] = 0.0

    loss_reduced["bpm_align_ratio"] = bpm_opd_metrics["bpm_align_ratio_sum"] / total_label
    loss_reduced["bpm_overlap_count"] = bpm_opd_metrics["bpm_overlap_count_sum"] / bpm_sample_count
    if getattr(args, "opd_diagnostics_mode", "basic") == "full":
        loss_reduced["bpm_aligned_tokens"] = bpm_opd_metrics["bpm_aligned_tokens_sum"]
        loss_reduced["bpm_num_tokens"] = bpm_opd_metrics["bpm_num_tokens_sum"]

    bpm_weight = max(bpm_opd_metrics.get("bpm_metric_weight_sum", loss_weight), 1)
    bpm_label = max(bpm_opd_metrics.get("bpm_label_rows_sum", total_label), 1)
    bpm_samples = max(bpm_opd_metrics.get("bpm_sample_count_sum", 1), 1)
    # bpm_div: the per-row trained objective -- CE at beta=0, JSD/skew-RKL above
    loss_reduced["bpm_div"] = bpm_opd_metrics.get("bpm_div_sum", 0.0) / bpm_weight
    _q_other = bpm_opd_metrics.get("bpm_q_other_sum", 0.0) / bpm_weight
    loss_reduced["bpm_overlap_mass"] = 1.0 - _q_other
    loss_reduced["bpm_q_other"] = _q_other
    loss_reduced["bpm_coverage"] = bpm_opd_metrics.get("bpm_rows_sum", 0.0) / bpm_label
    _stop_rows_denom = max(bpm_opd_metrics.get("bpm_stop_rows_sum", 0.0), 1.0)
    loss_reduced["bpm_stop_mass_teacher"] = bpm_opd_metrics.get("bpm_stop_mass_teacher_sum", 0.0) / _stop_rows_denom
    loss_reduced["bpm_stop_prob_student"] = bpm_opd_metrics.get("bpm_stop_prob_student_sum", 0.0) / _stop_rows_denom
    loss_reduced["bpm_rep_frac"] = bpm_opd_metrics.get("bpm_rep_frac_sum", 0.0) / bpm_samples
    if bpm_opd_metrics.get("bpm_masked_ws_rows_sum", 0.0) > 0.0:
        loss_reduced["bpm_masked_ws_rows"] = bpm_opd_metrics.get("bpm_masked_ws_rows_sum", 0.0)
    if bpm_opd_metrics.get("bpm_masked_random_rows_sum", 0.0) > 0.0:
        loss_reduced["bpm_masked_random_rows"] = bpm_opd_metrics.get("bpm_masked_random_rows_sum", 0.0)

    if getattr(args, "opd_diagnostics_mode", "basic") == "full":
        loss_reduced["bpm_ce"] = bpm_opd_metrics.get("bpm_ce_sum", 0.0) / bpm_weight
        loss_reduced["bpm_rows"] = bpm_opd_metrics.get("bpm_rows_sum", 0.0)
        loss_reduced["bpm_fast_rows"] = bpm_opd_metrics.get("bpm_fast_rows_sum", 0.0)
        loss_reduced["bpm_tail_rows"] = bpm_opd_metrics.get("bpm_tail_rows_sum", 0.0)
        loss_reduced["bpm_chain_rows"] = bpm_opd_metrics.get("bpm_chain_rows_sum", 0.0)
        loss_reduced["bpm_stop_rows"] = bpm_opd_metrics.get("bpm_stop_rows_sum", 0.0)
        loss_reduced["bpm_chain_rerouted_rows"] = bpm_opd_metrics.get("bpm_chain_rerouted_rows_sum", 0.0)
        loss_reduced["bpm_midspan_rows"] = bpm_opd_metrics.get("bpm_midspan_rows_sum", 0.0)
        loss_reduced["bpm_merge_corrected_rows"] = bpm_opd_metrics.get("bpm_merge_corrected_rows_sum", 0.0)
        loss_reduced["bpm_midspan_corrected_rows"] = bpm_opd_metrics.get("bpm_midspan_corrected_rows_sum", 0.0)
        loss_reduced["bpm_tail_degenerate_rows"] = bpm_opd_metrics.get("bpm_tail_degenerate_rows_sum", 0.0)
        loss_reduced["bpm_route_dropped_rows"] = bpm_opd_metrics.get("bpm_route_dropped_rows_sum", 0.0)
        loss_reduced["bpm_skipped_byte_mismatch"] = bpm_opd_metrics.get("bpm_skipped_byte_mismatch_sum", 0.0)
        loss_reduced["bpm_skipped_broken_code"] = bpm_opd_metrics.get("bpm_skipped_broken_code_sum", 0.0)
    args._bpm_opd_metrics = None


def _add_baseline_opd_metrics(args: Namespace, loss_reduced: dict[str, float], *, attr: str, prefix: str) -> None:
    """Reduce and publish a baseline accumulator; both baselines share the key layout."""
    metrics = getattr(args, attr, None)
    if metrics is None:
        return

    loss_type_key = f"_{prefix}_opd_loss_type"
    numeric_keys = [key for key in metrics if key != loss_type_key]
    _reduce_tensor_or_float_metrics(metrics, numeric_keys)

    loss_type = metrics.get(loss_type_key, f"{prefix}_rkl")
    total_label = max(metrics[f"{prefix}_num_tokens_sum"], 1)
    total_aligned = max(metrics[f"{prefix}_aligned_tokens_sum"], 1)
    loss_weight = max(metrics.get(f"{prefix}_loss_weight_sum", total_aligned), 1)
    sample_count = max(metrics.get(f"{prefix}_num_samples_with_alignment_sum", 1), 1)

    loss_reduced[f"{loss_type}_loss"] = metrics[f"{loss_type}_loss_sum"] / loss_weight

    entropy_tokens = metrics.get(f"{prefix}_entropy_token_sum", loss_weight)
    # teacher entropy covers fewer rows than student entropy, so each gets its own denom
    tea_entropy_tokens = metrics.get(f"{prefix}_tea_entropy_token_sum", 0.0) or entropy_tokens
    if entropy_tokens > 0:
        loss_reduced[f"{prefix}_tea_entropy"] = metrics[f"{prefix}_tea_entropy_sum"] / max(tea_entropy_tokens, 1)
        loss_reduced[f"{prefix}_stu_entropy"] = metrics[f"{prefix}_stu_entropy_sum"] / entropy_tokens
        loss_reduced[f"{prefix}_entropy_diagnostics_enabled"] = 1.0
    else:
        loss_reduced[f"{prefix}_entropy_diagnostics_enabled"] = 0.0

    loss_reduced[f"{prefix}_align_ratio"] = metrics[f"{prefix}_align_ratio_sum"] / total_label
    loss_reduced[f"{prefix}_overlap_count"] = metrics[f"{prefix}_overlap_count_sum"] / sample_count
    if getattr(args, "opd_diagnostics_mode", "basic") == "full":
        loss_reduced[f"{prefix}_aligned_tokens"] = metrics[f"{prefix}_aligned_tokens_sum"]
        loss_reduced[f"{prefix}_num_tokens"] = metrics[f"{prefix}_num_tokens_sum"]

    # GOLD hybrid only: raw JSD (matched) and ULD (unmatched) components, mirroring TRL GOLD.
    if f"{prefix}_matched_loss_sum" in metrics:
        loss_reduced[f"{prefix}_matched_loss"] = metrics[f"{prefix}_matched_loss_sum"] / loss_weight
    if f"{prefix}_unmatched_loss_sum" in metrics:
        loss_reduced[f"{prefix}_unmatched_loss"] = metrics[f"{prefix}_unmatched_loss_sum"] / loss_weight

    setattr(args, attr, None)


def reduce_bpm_metrics(args: Namespace, loss_reduced: dict[str, float]) -> None:
    """Append BPM diagnostics to ``loss_reduced`` and clear one-step caches."""
    _add_bpm_opd_metrics(args, loss_reduced)
    _add_baseline_opd_metrics(args, loss_reduced, attr="_simct_opd_metrics", prefix="simct")
    _add_baseline_opd_metrics(args, loss_reduced, attr="_gold_opd_metrics", prefix="gold")
