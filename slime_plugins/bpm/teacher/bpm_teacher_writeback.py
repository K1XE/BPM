"""BPM teacher payload writeback: Sample mutation, fail-fast validation, and the hook
that frees teacher payload on non-loss pipeline stages.
"""

from __future__ import annotations

import logging

import numpy as np

from slime.utils.types import Sample

from . import bpm_teacher_tokens as teacher_tokens

logger = logging.getLogger(__name__)


def masked_teacher_token_ids(
    teacher_full_token_ids: list[list[int]], loss_masks: list[np.ndarray]
) -> list[list[int]]:
    """Apply teacher-side loss masks to exact prefill token ids."""
    token_ids_list: list[list[int]] = []
    for i, (token_ids, mask) in enumerate(zip(teacher_full_token_ids, loss_masks, strict=True)):
        token_ids_masked = [int(tid) for tid, keep in zip(token_ids, mask.tolist(), strict=False) if keep]
        if len(token_ids_masked) != int(np.count_nonzero(mask)):
            raise RuntimeError(
                f"[OPD] local teacher_token_ids mask length mismatch for sample {i}: "
                f"token_ids={len(token_ids)} mask={mask.shape[0]} "
                f"masked_ids={len(token_ids_masked)} masked_rows={int(np.count_nonzero(mask))}"
            )
        token_ids_list.append(token_ids_masked)
    return token_ids_list


def ensure_teacher_payload_complete(hidden_states_list: list, token_ids_list: list | None) -> None:
    missing_hidden = [i for i, h in enumerate(hidden_states_list) if h is None]
    if missing_hidden:
        raise RuntimeError(
            f"[OPD] missing teacher hidden_states for {len(missing_hidden)} samples; "
            f"first missing indices: {missing_hidden[:16]}"
        )
    if token_ids_list is not None:
        missing_tids = [i for i, tids in enumerate(token_ids_list) if tids is None]
        if missing_tids:
            raise RuntimeError(
                f"[OPD] missing teacher_token_ids for {len(missing_tids)} samples; "
                f"first missing indices: {missing_tids[:16]}"
            )


def stripped_teacher_response_text(sample: Sample, eos_token: str, mask_eos_token: str) -> str:
    return teacher_tokens.alignment_response_text(sample, eos_token, mask_eos_token)


def inject_teacher_payload(
    self,
    data: list[Sample],
    *,
    hidden_states_list: list,
    token_ids_list: list | None,
    teacher_full_token_ids: list[list[int]],
    loss_masks: list[np.ndarray],
    eos_token: str,
    mask_eos_token: str,
) -> None:
    """Validate and write teacher hidden states / token ids into samples. Ids are derived
    locally; the SGLang-supplied fallback exists only for legacy replay.
    """
    assert len(hidden_states_list) == len(data), (
        f"[OPD] hidden_states count mismatch: got {len(hidden_states_list)}, expected {len(data)}"
    )

    for i, sample in enumerate(data):
        sample.teacher_hidden_states = hidden_states_list[i]
        sample.teacher_response_text = stripped_teacher_response_text(sample, eos_token, mask_eos_token)

    if token_ids_list is not None:
        assert len(token_ids_list) == len(data), (
            f"[OPD] token_ids count mismatch: got {len(token_ids_list)}, expected {len(data)}"
        )
        for i, sample in enumerate(data):
            sample.teacher_token_ids = token_ids_list[i]
        logger.info(f"[OPD]   injected teacher_token_ids into {len(data)} samples")
        return

    raise ValueError(
        "[OPD] teacher token ids were requested but not returned. "
        "Refusing to re-tokenize teacher text because token ids must be selected by the "
        "exact same teacher-side loss mask as hidden_states."
    )


def teacher_perf_metrics(
    self,
    *,
    t0: float,
    t_offload: float,
    t_wakeup: float,
    t_build: float,
    t_generate: float,
    t_sleep: float,
    t_onload: float,
    seq_lens: list[int],
    data: list[Sample],
    hidden_states_list: list,
) -> dict[str, float]:
    return {
        "perf/opd_teacher_total_time": t_onload - t0,
        "perf/opd_teacher_offload_time": t_offload - t0,
        "perf/opd_teacher_wakeup_time": t_wakeup - t_offload,
        "perf/opd_teacher_prompt_build_time": t_build - t_wakeup,
        "perf/opd_teacher_generate_time": t_generate - t_build,
        "perf/opd_teacher_sleep_time": t_sleep - t_generate,
        "perf/opd_teacher_onload_time": t_onload - t_sleep,
        "perf/opd_teacher_tokens_per_sec": sum(seq_lens) / (t_generate - t_build + 1e-6),
        "perf/opd_teacher_num_samples": len(data),
        "perf/opd_teacher_dropped_oversize": float(getattr(self, "_opd_oversize_dropped", 0)),
        "perf/opd_teacher_seq_tokens": sum(seq_lens),
        "perf/opd_teacher_hidden_mib": sum(_tensor_nbytes(h) for h in hidden_states_list) / (1024**2),
    }


def _tensor_nbytes(obj) -> int:
    nbytes = getattr(obj, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return int(np.asarray(obj).nbytes)


def materialize_teacher_rollout_tensors(args, rollout_id, rollout_data) -> None:
    """rollout_data_postprocess hook: keep teacher payload only where the loss runs.

    The loss materializes hidden rows lazily, so the last pipeline stage keeps the raw
    payload; earlier stages drop the references instead of holding the whole block.
    """
    from ..args.bpm_utils import is_bpm_enabled

    if not is_bpm_enabled(args):
        return
    from megatron.core import parallel_state as mpu

    if mpu.is_pipeline_last_stage():
        return
    for key in ("teacher_hidden_states", "teacher_token_ids"):
        if key in rollout_data:
            rollout_data[key] = None
