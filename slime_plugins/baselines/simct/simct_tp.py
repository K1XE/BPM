"""TP-aware logit gather helpers for the SimCT baseline."""

from __future__ import annotations

import torch


def _gather_tp_logits_at_global_ids_prevalidated(
    logits_shard: torch.Tensor,
    global_ids: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
    tp_group,
) -> torch.Tensor:
    """Gather static, already-validated global token IDs from TP shards.

    simct full overlap-vocab loss gathers the same overlap ID list for every
    chunk.  The caller validates ownership once before the hot loop, so repeating
    ``valid.all().item()`` / owner-count all-reduces in every chunk only adds GPU
    synchronizations.  This helper preserves the same straight-through gradient
    behavior as ``_gather_tp_logits_at_global_ids`` without the repeated checks.
    """
    if global_ids.device != logits_shard.device:
        global_ids = global_ids.to(device=logits_shard.device)
    global_ids = global_ids.long()
    if global_ids.numel() == 0:
        return logits_shard.new_empty((logits_shard.shape[0], 0), dtype=logits_shard.dtype)

    shard_vocab = logits_shard.shape[-1]
    if tp_size == 1:
        # Avoid materializing/reading a [rows, overlap] expanded gather index.
        # index_select is the stoken-like contiguous hot path for TP=1.
        return logits_shard.index_select(-1, global_ids)

    shard_start = tp_rank * shard_vocab
    shard_end = shard_start + shard_vocab
    in_shard = (global_ids >= shard_start) & (global_ids < shard_end)
    local_ids = (global_ids - shard_start).clamp(0, max(shard_vocab - 1, 0))
    local_vals = logits_shard.gather(-1, local_ids.unsqueeze(0).expand(logits_shard.shape[0], -1))
    local_vals = torch.where(in_shard.unsqueeze(0), local_vals, torch.zeros_like(local_vals))
    with torch.no_grad():
        gathered = local_vals.detach().clone()
        torch.distributed.all_reduce(gathered, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    return local_vals + (gathered - local_vals.detach())

def _gather_tp_logits_at_rows_global_ids_prevalidated(
    logits_shard: torch.Tensor,
    row_ids_1d: torch.Tensor,
    global_ids: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
    tp_group,
) -> torch.Tensor:
    """Gather static global token IDs for selected rows without full-row copy.

    This is the full simct analogue of the row-subset top-K helper.  It returns
    ``logits_shard[row_ids_1d, global_ids]`` (with TP all-reduce when needed)
    directly as ``[rows, ids]`` and avoids materializing
    ``logits_shard.index_select(0, row_ids_1d)`` as ``[rows, full_student_vocab]``.
    """
    if row_ids_1d.device != logits_shard.device:
        row_ids_1d = row_ids_1d.to(device=logits_shard.device)
    if global_ids.device != logits_shard.device:
        global_ids = global_ids.to(device=logits_shard.device)
    row_ids_1d = row_ids_1d.long()
    global_ids = global_ids.long()
    if global_ids.numel() == 0:
        return logits_shard.new_empty((row_ids_1d.numel(), 0), dtype=logits_shard.dtype)

    shard_vocab = logits_shard.shape[-1]
    if tp_size == 1:
        # Full simct supervises only the shared/virtual support.  Do NOT select
        # rows first: ``index_select(0, rows)`` materializes
        # [rows, full_student_vocab] before dropping non-overlap columns.  In the
        # GLM-4.7-Flash -> Qwen3.5-2B run from the 20260613_113211 log this means
        # copying ~8k x 248k logits per CP-local sample just to keep ~8k x 143k.
        #
        # Prefer a zero-copy row slice for the common contiguous-row case, then
        # select overlap columns.  For sparse rows, use direct 2-D advanced
        # indexing so work scales with rows x overlap_vocab, not rows x full_vocab.
        if row_ids_1d.numel() == 0:
            return logits_shard.new_empty((0, global_ids.numel()), dtype=logits_shard.dtype)
        first = int(row_ids_1d[0].detach().item())
        n_rows = int(row_ids_1d.numel())
        if n_rows == 1:
            return logits_shard.narrow(0, first, 1).index_select(-1, global_ids)
        # Avoid synchronizing the common arange case through .all().item() when
        # rows were created from a Python range; the tensor is tiny compared with
        # the vocab projection, so this check is still cheap and outside the
        # differentiable logits path.
        expected = torch.arange(first, first + n_rows, device=row_ids_1d.device, dtype=row_ids_1d.dtype)
        if bool(torch.equal(row_ids_1d, expected)):
            return logits_shard.narrow(0, first, n_rows).index_select(-1, global_ids)
        return logits_shard[row_ids_1d.unsqueeze(-1), global_ids.unsqueeze(0)]

    rows = row_ids_1d.unsqueeze(-1).expand(row_ids_1d.numel(), global_ids.numel())
    shard_start = tp_rank * shard_vocab
    shard_end = shard_start + shard_vocab
    in_shard = (global_ids >= shard_start) & (global_ids < shard_end)
    local_ids = (global_ids - shard_start).clamp(0, max(shard_vocab - 1, 0))
    local_vals = logits_shard[rows, local_ids.unsqueeze(0).expand_as(rows)]
    local_vals = torch.where(in_shard.unsqueeze(0), local_vals, torch.zeros_like(local_vals))
    with torch.no_grad():
        gathered = local_vals.detach().clone()
        torch.distributed.all_reduce(gathered, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    return local_vals + (gathered - local_vals.detach())

def _gather_tp_logits_at_rows_global_ids_subset(
    logits_shard: torch.Tensor,
    row_ids_1d: torch.Tensor,
    global_ids_2d: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
    tp_group,
) -> torch.Tensor:
    """Gather per-row token subsets without copying full vocab rows.

    ``row_ids_1d`` is [rows] indexing into ``logits_shard`` and
    ``global_ids_2d`` is [rows, K].  This is stricter than
    _gather_tp_logits_at_global_ids_subset because it avoids first materializing
    ``logits_shard[row_ids_1d, :]``.  For 32k simct runs that removes a large
    [chunk, student_vocab] temporary from the top-K path.
    """
    if row_ids_1d.device != logits_shard.device:
        row_ids_1d = row_ids_1d.to(device=logits_shard.device)
    if global_ids_2d.device != logits_shard.device:
        global_ids_2d = global_ids_2d.to(device=logits_shard.device)
    row_ids_1d = row_ids_1d.long()
    global_ids_2d = global_ids_2d.long()
    if global_ids_2d.numel() == 0:
        return logits_shard.new_empty(global_ids_2d.shape, dtype=logits_shard.dtype)

    shard_vocab = logits_shard.shape[-1]
    rows = row_ids_1d.unsqueeze(-1).expand_as(global_ids_2d)
    if tp_size == 1:
        valid = (global_ids_2d >= 0) & (global_ids_2d < shard_vocab)
        if not bool(valid.all().item()):
            bad = global_ids_2d[~valid][:10].detach().cpu().tolist()
            raise ValueError(
                f"[OPD][simct] requested token id outside TP=1 logits vocab row subset: "
                f"bad_ids={bad} shard_vocab={shard_vocab}"
            )
        return logits_shard[rows, global_ids_2d]

    shard_start = tp_rank * shard_vocab
    shard_end = shard_start + shard_vocab
    in_shard = (global_ids_2d >= shard_start) & (global_ids_2d < shard_end)
    local_ids = (global_ids_2d - shard_start).clamp(0, max(shard_vocab - 1, 0))
    local_vals = logits_shard[rows, local_ids]
    local_vals = torch.where(in_shard, local_vals, torch.zeros_like(local_vals))

    with torch.no_grad():
        gathered = local_vals.detach().clone()
        torch.distributed.all_reduce(gathered, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        owner_count = in_shard.to(dtype=torch.int32).clone()
        torch.distributed.all_reduce(owner_count, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        if not bool((owner_count == 1).all().item()):
            bad = global_ids_2d[(owner_count != 1).to(device=global_ids_2d.device)]
            raise ValueError(
                f"[OPD][simct] TP vocab-shard ownership check failed for row subset gather. "
                f"bad_ids={bad[:10].detach().cpu().tolist()} tp_size={tp_size} shard_vocab={shard_vocab}"
            )
    return local_vals + (gathered - local_vals.detach())
