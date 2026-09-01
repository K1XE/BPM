"""Per-step metric accumulation and logging for the SimCT baseline."""

from __future__ import annotations

import logging
from argparse import Namespace

import torch
from megatron.core import mpu

logger = logging.getLogger(__name__)


def log_simct_step_summary(
    args: Namespace,
    *,
    is_logging_rank: bool,
    kd_loss_mean: torch.Tensor,
    mean_kd_loss: torch.Tensor,
    mean_tea_entropy: torch.Tensor,
    mean_stu_entropy: torch.Tensor,
    entropy_diagnostics_enabled: bool,
    opd_loss_type: str,
    num_overlap: int,
    align_ratio_mean: float,
    global_align_ratio_mean: float,
    total_aligned_tokens: float,
    global_aligned_tokens: float,
    total_label_tokens: float,
    global_label_tokens: float,
    sample_count: float,
    topk_mode: bool,
    effective_k: int,
    tp_size: int,
    cp_size: int,
) -> None:
    """Log simct OPD step summaries with microbatch progress."""
    if not is_logging_rank:
        return

    microbatch_index = int(getattr(args, "_simct_train_microbatch_index", 0) or 0) + 1
    args._simct_train_microbatch_index = microbatch_index
    num_microbatches_hint = int(getattr(args, "_simct_train_num_microbatches", 0) or 0)
    log_interval = max(int(getattr(args, "simct_train_log_interval", 0) or 0), 0)
    should_log_summary = (
        log_interval <= 0
        or microbatch_index == 1
        or (num_microbatches_hint > 0 and microbatch_index == num_microbatches_hint)
        or microbatch_index % log_interval == 0
    )
    if not should_log_summary:
        return

    progress = (
        f"mb={microbatch_index}/{num_microbatches_hint}"
        if num_microbatches_hint > 0
        else f"mb={microbatch_index}"
    )
    entropy_msg = (
        f"tea_ent={mean_tea_entropy.item():.3f} stu_ent={mean_stu_entropy.item():.3f} "
        if entropy_diagnostics_enabled
        else "tea_ent=off stu_ent=off "
    )
    logger.info(
        f"[OPD][simct][step summary] {progress} "
        f"loss={kd_loss_mean.detach().item():.4f} {opd_loss_type}={mean_kd_loss.item():.4f} "
        f"{entropy_msg}"
        + f"overlap={num_overlap} align_ratio={align_ratio_mean:.3f} "
        f"global_align_ratio={global_align_ratio_mean:.3f} "
        f"aligned_tokens={total_aligned_tokens:.0f}/{global_aligned_tokens:.0f} "
        f"label_tokens={total_label_tokens:.0f}/{global_label_tokens:.0f} "
        f"n_samples={sample_count} topk_mode={topk_mode} k={effective_k} tp={tp_size} cp={cp_size}"
    )

def accumulate_simct_train_metrics(
    args: Namespace,
    *,
    device: torch.device,
    simct_loss_metric_name: str,
    mean_kd_loss: torch.Tensor,
    mean_tea_entropy: torch.Tensor,
    mean_stu_entropy: torch.Tensor,
    entropy_diagnostics_enabled: bool,
    total_overlap_count: float,
    align_ratio_mean: float,
    total_aligned_tokens: float,
    total_label_tokens: float,
    sample_count: float,
    metric_weight: float | None = None,
) -> None:
    """Accumulate simct OPD diagnostics without per-microbatch GPU syncs."""
    if not mpu.is_pipeline_last_stage():
        args._simct_opd_metrics = None
        return

    if not hasattr(args, "_simct_opd_metrics") or args._simct_opd_metrics is None:
        args._simct_opd_metrics = {
            f"{simct_loss_metric_name}_loss_sum": torch.tensor(0.0, dtype=torch.float32, device=device),
            "simct_tea_entropy_sum": torch.tensor(0.0, dtype=torch.float32, device=device),
            "simct_stu_entropy_sum": torch.tensor(0.0, dtype=torch.float32, device=device),
            "simct_overlap_count_sum": 0.0,
            "simct_entropy_token_sum": 0.0,
            "simct_align_ratio_sum": 0.0,
            "simct_aligned_tokens_sum": 0.0,
            "simct_num_tokens_sum": 0.0,
            "simct_num_samples_with_alignment_sum": 0.0,
            "simct_loss_weight_sum": 0.0,
            "_simct_opd_loss_type": simct_loss_metric_name,
        }
    acc = args._simct_opd_metrics
    n_aligned = float(total_aligned_tokens)
    n_label = float(total_label_tokens)
    loss_weight = n_aligned if metric_weight is None else float(metric_weight)
    acc[f"{simct_loss_metric_name}_loss_sum"] = (
        acc[f"{simct_loss_metric_name}_loss_sum"] + mean_kd_loss * loss_weight
    )
    if entropy_diagnostics_enabled:
        acc["simct_tea_entropy_sum"] = acc["simct_tea_entropy_sum"] + mean_tea_entropy * loss_weight
        acc["simct_stu_entropy_sum"] = acc["simct_stu_entropy_sum"] + mean_stu_entropy * loss_weight
        acc["simct_entropy_token_sum"] += loss_weight
    acc["simct_overlap_count_sum"] += float(total_overlap_count)
    acc["simct_align_ratio_sum"] += align_ratio_mean * n_label
    acc["simct_aligned_tokens_sum"] += n_aligned
    acc["simct_num_tokens_sum"] += n_label
    acc["simct_num_samples_with_alignment_sum"] += sample_count
    acc["simct_loss_weight_sum"] += loss_weight
