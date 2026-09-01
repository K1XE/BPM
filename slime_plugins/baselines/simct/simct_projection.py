"""Teacher projection and hidden-row helpers for the SimCT baseline."""

from __future__ import annotations

import numpy as np
import torch
from megatron.core import mpu


def _sample_token_logits_from_logits(
    logits_shard: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    vocab_start: int,
    tp_size: int,
) -> torch.Tensor:
    """Straight-through gather of raw logits for per-row token IDs."""
    shard_vocab = logits_shard.shape[-1]
    if token_ids.numel() == 0:
        return logits_shard.new_empty((0,))
    local_idx = (token_ids - vocab_start).clamp(0, shard_vocab - 1)
    in_shard = (token_ids >= vocab_start) & (token_ids < vocab_start + shard_vocab)
    local = logits_shard.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)
    local = torch.where(in_shard, local, torch.zeros_like(local))
    if tp_size <= 1:
        return local
    gathered = local.detach().clone()
    torch.distributed.all_reduce(
        gathered,
        op=torch.distributed.ReduceOp.SUM,
        group=mpu.get_tensor_model_parallel_group(),
    )
    return local + (gathered - local.detach())

def _simct_lm_head_at_global_ids(
    lm_head: torch.nn.Module,
    hidden: torch.Tensor,
    global_ids: torch.Tensor,
    *,
    vocab_start: int,
) -> torch.Tensor:
    """Project hidden rows directly to arbitrary teacher-vocab ids.

    Full SimCT compares only the overlap vocabulary, not the teacher's whole
    vocab.  For GLM/Qwen this is ~143k columns vs a larger padded teacher vocab.
    This helper avoids computing and then discarding non-overlap teacher logits.
    It is used on the TP=1 fast path; TP>1 callers must preserve the existing
    owner/all-reduce path.
    """
    if hidden.numel() == 0 or global_ids.numel() == 0:
        return hidden.new_empty((hidden.shape[0], int(global_ids.numel())), dtype=torch.float32)
    if global_ids.device != hidden.device:
        global_ids = global_ids.to(device=hidden.device)
    cache_key = (
        int(lm_head.weight.data_ptr()),
        str(lm_head.weight.device),
        int(vocab_start),
        int(global_ids.data_ptr()),
        int(global_ids.numel()),
    )
    # SMALL bounded dict cache (not a single slot): the stop-token bridge projects
    # BOTH the ~143k-col overlap ids AND the tiny stop-set ids through this helper
    # every chunk.  A single-slot cache keyed by global_ids would alternate between
    # them and re-index_select the ~1.2GB overlap weight on every call.  A dict keyed
    # by cache_key keeps both selected weights resident; capacity is bounded (only a
    # couple of distinct, args-cached id tensors exist) with simple FIFO eviction.
    cache = getattr(lm_head, "_simct_overlap_weight_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        lm_head._simct_overlap_weight_cache = cache
    cached = cache.get(cache_key)
    if cached is None:
        local_ids = (global_ids.long() - int(vocab_start)).clamp(0, lm_head.weight.shape[0] - 1)
        weight = lm_head.weight.index_select(0, local_ids).contiguous()
        bias = getattr(lm_head, "bias", None)
        bias_selected = bias.index_select(0, local_ids).contiguous() if bias is not None else None
        cached = (weight, bias_selected)
        if len(cache) >= 8:  # bound memory: evict oldest (insertion order)
            cache.pop(next(iter(cache)))
        cache[cache_key] = cached
    weight, bias_selected = cached
    out = hidden.matmul(weight.t())
    if bias_selected is not None:
        out = out + bias_selected.unsqueeze(0)
    return out.float()

def _simct_lm_head_token_logits(
    lm_head: torch.nn.Module,
    hidden: torch.Tensor,
    global_ids: torch.Tensor,
    *,
    vocab_start: int,
) -> torch.Tensor:
    """Project each hidden row only to its matching teacher token id."""
    if hidden.numel() == 0 or global_ids.numel() == 0:
        return hidden.new_empty((0,), dtype=torch.float32)
    if global_ids.device != hidden.device:
        global_ids = global_ids.to(device=hidden.device)
    local_ids = (global_ids.long() - int(vocab_start)).clamp(0, lm_head.weight.shape[0] - 1)
    weight = lm_head.weight.index_select(0, local_ids)
    out = (hidden * weight).sum(dim=-1)
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        out = out + bias.index_select(0, local_ids)
    return out.float()

def _hidden_rows_len(hidden: torch.Tensor | np.ndarray) -> int:
    """Return the row count for a tensor/ndarray hidden-state block."""
    return int(hidden.shape[0])

def _hidden_rows_shape(hidden: torch.Tensor | np.ndarray) -> tuple[int, ...]:
    """Return a stable shape tuple without moving hidden states across devices."""
    return tuple(int(x) for x in hidden.shape)
