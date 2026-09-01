"""Teacher lm_head projection and hidden-row selection for the BPM backend."""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from .loss_helpers import load_teacher_lm_head


def _get_bpm_teacher_lm_head(args: Namespace) -> torch.nn.Linear:
    """Cached frozen teacher lm_head (bfloat16), TP-sharded via ``load_teacher_lm_head``.

    The cache is dropped on every train-actor sleep, so this reloads once per OPD sync step.
    """
    if not hasattr(args, "_bpm_teacher_lm_head") or args._bpm_teacher_lm_head is None:
        lm_head_path = getattr(args, "bpm_teacher_model_path", None)
        if lm_head_path is None:
            raise ValueError(
                "BPM requires --bpm-teacher-model-path to load teacher lm_head weights."
            )
        device = torch.cuda.current_device()
        args._bpm_teacher_lm_head = load_teacher_lm_head(
            lm_head_path,
            device=device,
            dtype=torch.bfloat16,
        )
    return args._bpm_teacher_lm_head


def _select_hidden_rows_to_device(
    hidden: torch.Tensor | np.ndarray,
    row_indices: list[int],
    *,
    device: torch.device | int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Copy only the selected teacher hidden rows to the training device.

    The BPM loss needs only teacher rows that govern CP-local student positions; selecting rows
    before the transfer avoids copying the whole response hidden block on every CP rank.
    """
    hidden_dim = int(hidden.shape[-1]) if len(hidden.shape) > 1 else 0
    if not row_indices:
        return torch.empty((0, hidden_dim), dtype=dtype, device=device)

    if isinstance(hidden, np.ndarray):
        rows_np = np.asarray(row_indices, dtype=np.int64)
        selected = torch.from_numpy(hidden[rows_np])
        return selected.to(device=device, dtype=dtype, non_blocking=True)

    rows = torch.tensor(row_indices, dtype=torch.long, device=hidden.device)
    selected = hidden.index_select(0, rows)
    return selected.to(device=device, dtype=dtype, non_blocking=(hidden.device.type == "cpu"))


def _batched_lm_head_rows(
    lm_head: torch.nn.Module,
    hidden: torch.Tensor,
    row_indices: list[int],
    *,
    temperature: float,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Project selected hidden rows through a frozen lm_head in bounded chunks."""
    if not row_indices:
        return hidden.new_empty((0, int(getattr(lm_head, "_opd_shard_vocab_size", lm_head.weight.shape[0]))))
    outs: list[torch.Tensor] = []
    row_t = torch.tensor(row_indices, dtype=torch.long, device=hidden.device)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, int(row_t.numel()), chunk_size):
        rows = row_t[start:start + chunk_size]
        outs.append(lm_head(hidden[rows]).float() / temperature)
    return torch.cat(outs, dim=0)
