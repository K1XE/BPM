"""Per-microbatch wall-clock probe for the training loop. Diagnostic only, default off.

Records forward / loss / backward+engine time per microbatch and prints the whole
distribution each step, so a data-dependent tail is distinguishable from a uniform
slowdown.

BPM_MB_TIMING=1 synchronizes at every section boundary, so the sections are real GPU
time but the step is slower than an unprobed one -- read the shape, not the level.
BPM_MB_TIMING=nosync never synchronizes and reports CPU-side time instead.
BPM_MB_TIMING_RANKS (default 0, or `all`) selects which ranks print.

When off, model.py never imports this module and the hot path is unchanged. Boundary
totals assume PP=1; under an interleaved schedule the totals mix microbatches.
"""

from __future__ import annotations

import logging
import math
import os
import time

import torch
from megatron.core import mpu

logger = logging.getLogger(__name__)

_TRUE_VALUES = ("1", "on", "true", "yes")
_OFF_VALUES = ("", "0", "off", "false", "no")

_UNSET = object()
_TIMER = _UNSET


def get_microbatch_timer() -> _MicrobatchTimer | None:
    """Return the process-wide probe, or None when BPM_MB_TIMING is off. Built once and
    cached, None included.
    """
    global _TIMER
    if _TIMER is _UNSET:
        mode = _resolve_mode()
        if mode is None:
            _TIMER = None
        else:
            rank = _current_rank()
            _TIMER = _MicrobatchTimer(mode=mode, rank=rank, log_this_rank=_rank_should_log(rank))
    return _TIMER


def _resolve_mode() -> str | None:
    """Return ``"sync"``, ``"nosync"``, or ``None`` when the probe is disabled."""
    raw = os.environ.get("BPM_MB_TIMING", "").strip().lower()
    if raw in _TRUE_VALUES:
        return "sync"
    if raw == "nosync":
        return "nosync"
    if raw not in _OFF_VALUES:
        logger.warning("BPM_MB_TIMING=%r is not one of 1/on/nosync/off; the probe stays off", raw)
    return None


def _current_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _rank_should_log(rank: int) -> bool:
    spec = os.environ.get("BPM_MB_TIMING_RANKS", "0").strip().lower()
    if spec == "all":
        return True
    return str(rank) in {piece.strip() for piece in spec.split(",") if piece.strip()}


def _logit_rows(output_tensor) -> int | None:
    """Packed token rows in this microbatch's logits, or None if unreadable. Reading
    .shape costs nothing and proxies how much work the microbatch carried.
    """
    if not isinstance(output_tensor, torch.Tensor) or output_tensor.dim() < 2:
        return None
    # packed THD keeps the other axis at 1, so the product is the token count
    return int(output_tensor.shape[0]) * int(output_tensor.shape[1])


def _pipeline_size() -> int:
    """Pipeline width for the header line, or -1 before the parallel state exists: mpu
    asserts rather than defaulting, and a probe must not abort the run it measures.
    """
    try:
        return mpu.get_pipeline_model_parallel_world_size()
    except Exception:
        return -1


class _MicrobatchTimer:
    """Records one row per microbatch and prints the distribution at step end."""

    def __init__(self, mode: str, rank: int, log_this_rank: bool):
        self._mode = mode
        self._sync = mode == "sync"
        self._rank = rank
        self._log_this_rank = log_this_rank
        self._label = ""
        self._scheduled: int | None = None
        # One (total, forward, loss, rows) tuple per completed microbatch, seconds.
        self._records: list[tuple[float, float, float, int | None]] = []
        self._boundary: float | None = None
        self._forward_end: float | None = None
        self._loss_start: float | None = None
        self._loss_end: float | None = None
        self._rows: int | None = None

    def begin_step(self, tag: str, rollout_id, step_id, scheduled_microbatches) -> None:
        """Start collecting for one ``forward_backward_func`` call."""
        rollout_part = "" if rollout_id is None else f"rollout={rollout_id} "
        self._label = f"{tag} {rollout_part}step={step_id}"
        self._scheduled = scheduled_microbatches
        self._records = []
        self._reset_microbatch(None)

    def boundary(self) -> None:
        """Called at the leading edge of every microbatch, before its data is fetched."""
        now = self._now()
        if self._boundary is not None:
            self._close(now)
        self._reset_microbatch(now)

    def mark_forward_end(self, result) -> None:
        """Close the forward section the instant ``forward_step_func`` returns."""
        if self._boundary is None:
            return
        if isinstance(result, tuple) and len(result) == 2:
            self._rows = _logit_rows(result[0])
        self._forward_end = self._now()

    def wrap_loss(self, result):
        """Wrap the loss callable so the loss section is timed on its own.

        `result` is the (output_tensor, loss_func) pair the engine expects; the wrapper is
        transparent. The loss clock starts here, not at mark_forward_end, so the progress-bar
        update between them is charged to bwd+.
        """
        if self._boundary is None or not isinstance(result, tuple) or len(result) != 2:
            return result

        output_tensor, loss_func = result
        self._loss_start = self._now()

        def timed_loss_func(*args, **kwargs):
            outputs = loss_func(*args, **kwargs)
            self._loss_end = self._now()
            return outputs

        return output_tensor, timed_loss_func

    def end_step(self) -> None:
        """Close the final microbatch and emit the distribution. That microbatch's `total`
        also contains the step epilogue, so expect a spike at the last index; fwd and loss
        are unaffected.
        """
        if self._boundary is not None:
            self._close(self._now())
        self._reset_microbatch(None)
        if self._log_this_rank:
            self._emit()
        self._records = []

    def _now(self) -> float:
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _reset_microbatch(self, boundary: float | None) -> None:
        self._boundary = boundary
        self._forward_end = None
        self._loss_start = None
        self._loss_end = None
        self._rows = None

    def _close(self, now: float) -> None:
        forward = (self._forward_end - self._boundary) if self._forward_end is not None else math.nan
        if self._loss_start is not None and self._loss_end is not None:
            loss = self._loss_end - self._loss_start
        else:
            # Non-last pipeline stages never run the loss; it then falls into `bwd+`.
            loss = math.nan
        self._records.append((now - self._boundary, forward, loss, self._rows))

    def _emit(self) -> None:
        if not self._records:
            return
        prefix = f"[bpm-mbtime] rank={self._rank}"
        totals = [record[0] * 1e3 for record in self._records]
        forwards = [record[1] * 1e3 for record in self._records]
        losses = [record[2] * 1e3 for record in self._records]
        rests = [(record[0] - _zero_if_nan(record[1]) - _zero_if_nan(record[2])) * 1e3 for record in self._records]

        scheduled = "" if self._scheduled is None else f" scheduled={self._scheduled}"
        logger.info(
            "%s %s n_mb=%d%s mode=%s pp=%d note=last-mb-total-includes-step-epilogue",
            prefix,
            self._label,
            len(self._records),
            scheduled,
            self._mode,
            _pipeline_size(),
        )
        for name, series in (("total", totals), ("fwd", forwards), ("loss", losses), ("bwd+", rests)):
            logger.info("%s   %-5s ms: %s", prefix, name, _summary(series))

        slowest = sorted(range(len(totals)), key=lambda index: totals[index], reverse=True)[:3]
        logger.info(
            "%s   slowest: %s",
            prefix,
            " | ".join(
                f"#{index} tot={totals[index]:.1f} fwd={forwards[index]:.1f} "
                f"loss={losses[index]:.1f} bwd+={rests[index]:.1f} rows={self._records[index][3]}"
                for index in slowest
            ),
        )
        # the raw series shows the shape; a tail spike and a raised floor share a mean
        logger.info("%s   per-mb total ms: %s", prefix, ",".join(f"{value:.1f}" for value in totals))
        logger.info("%s   per-mb rows    : %s", prefix, ",".join(str(record[3]) for record in self._records))


def _zero_if_nan(value: float) -> float:
    return 0.0 if math.isnan(value) else value


def _percentile(sorted_values: list[float], quantile: float) -> float:
    """Nearest-rank percentile of an already-sorted list. Nearest rank rather than
    interpolation: with ten-to-forty microbatches, p90 exists to surface the tail.
    """
    index = min(len(sorted_values) - 1, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[max(0, index)]


def _summary(series: list[float]) -> str:
    finite = sorted(value for value in series if not math.isnan(value))
    if not finite:
        return "n/a"
    return (
        f"sum={sum(finite):.1f} mean={sum(finite) / len(finite):.1f} "
        f"p50={_percentile(finite, 0.5):.1f} p90={_percentile(finite, 0.9):.1f} "
        f"min={finite[0]:.1f} max={finite[-1]:.1f}"
    )
