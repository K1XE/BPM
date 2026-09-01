"""Context-parallel scalar reduction for the BPM loss path. Only _reduce_cp_float_counts
is needed here; distributed_masked_whiten lives on slime.utils.distributed_utils.
"""
from __future__ import annotations

from argparse import Namespace

import torch
from megatron.core import mpu


def _to_int_list(x) -> list[int]:
    """Coerce a tensor or iterable of token ids into a plain host ``list[int]``."""
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.detach().cpu().tolist()]
    return [int(v) for v in x]


def _reduce_cp_float_counts(values: list[float], device: torch.device | int) -> list[float]:
    """Sum scalar counters over the context-parallel group only, not DP.

    The calculate_per_token_loss=False path returns one pre-normalized scalar per CP
    rank and the schedule sums them, so the denominator must be the CP-summed count.
    Reducing over DP+CP would under-normalize by ~dp_size.
    """
    counts = torch.tensor(values, dtype=torch.float64, device=device)
    if torch.distributed.is_initialized() and mpu.get_context_parallel_world_size() > 1:
        torch.distributed.all_reduce(
            counts, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group()
        )
    return [float(v) for v in counts.detach().cpu().tolist()]


def _resolve_opd_distill_scope(args: Namespace, support_size: int) -> tuple[str, int]:
    """Resolve OPD range into ``sample``/``topk``/``full`` plus effective K."""
    scope = getattr(args, "opd_distill_scope", "auto")
    opd_topk = int(getattr(args, "opd_topk", 0) or 0)
    if scope == "sample":
        return "sample", 1
    if scope == "full":
        return "full", support_size
    if scope == "topk":
        if opd_topk <= 1:
            raise ValueError("--opd-distill-scope topk requires --opd-topk >= 2")
        if opd_topk >= support_size:
            raise ValueError("--opd-distill-scope topk requires --opd-topk smaller than the support size")
        return "topk", opd_topk
    if scope != "auto":
        raise ValueError(f"Unsupported --opd-distill-scope: {scope}")
    if 0 < opd_topk < support_size:
        return "topk", opd_topk
    return "full", support_size


def _validate_token_ids_in_vocab_msg(
    token_ids: torch.Tensor,
    *,
    real_vocab_size: int,
    context: str,
) -> str | None:
    """Return an error message if sampled token IDs are outside a real vocab.

    This is intentionally side-effect free and does not raise.  BPM
    OPD callers use it inside data-dependent per-sample loops so failures can be
    synchronized across DP/CP ranks before any later DP/CP collective.
    """
    if token_ids.numel() == 0:
        return None
    bad_mask = (token_ids < 0) | (token_ids >= int(real_vocab_size))
    if not bool(bad_mask.any().item()):
        return None
    bad = token_ids[bad_mask][:8].detach().cpu().tolist()
    return f"{context}: token id outside vocab: {bad}, vocab={int(real_vocab_size)}"


def _validate_tp_global_ids_owned_msg(
    global_ids: torch.Tensor,
    *,
    shard_vocab: int,
    tp_size: int,
    tp_rank: int,
    tp_group,
    context: str,
    real_vocab_size: int | None = None,
) -> str | None:
    """Validate that global token IDs are owned by exactly one TP vocab shard.

    The validation mirrors `_gather_tp_logits_at_global_ids` but returns a
    message instead of raising.  It is used before BPM per-sample loops
    so config/tokenizer errors are routed through the DP/CP synchronized
    fail-fast path rather than surfacing as helper-internal partial-rank exits.
    """
    if global_ids.numel() == 0:
        return None
    if global_ids.device.type != "cuda" and torch.cuda.is_available():
        global_ids = global_ids.to(device=torch.cuda.current_device())
    global_ids = global_ids.long()
    shard_vocab = int(shard_vocab)
    tp_size = int(tp_size)
    tp_rank = int(tp_rank)
    padded_vocab = shard_vocab * max(tp_size, 1)
    bad_mask = (global_ids < 0) | (global_ids >= padded_vocab)
    if real_vocab_size is not None:
        bad_mask = bad_mask | (global_ids >= int(real_vocab_size))
    if bool(bad_mask.any().item()):
        bad = global_ids[bad_mask][:10].detach().cpu().tolist()
        return (
            f"{context}: requested token id outside TP vocab range: bad_ids={bad}, "
            f"shard_vocab={shard_vocab}, tp_size={tp_size}, "
            f"padded_vocab={padded_vocab}, real_vocab_size={real_vocab_size}"
        )

    if tp_size <= 1:
        return None

    shard_start = tp_rank * shard_vocab
    shard_end = shard_start + shard_vocab
    owner_count = ((global_ids >= shard_start) & (global_ids < shard_end)).to(dtype=torch.int32)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            owner_count,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
    if not bool((owner_count == 1).all().item()):
        bad_mask = owner_count != 1
        bad = global_ids[bad_mask][:10].detach().cpu().tolist()
        counts = owner_count[bad_mask][:10].detach().cpu().tolist()
        return (
            f"{context}: TP vocab-shard ownership check failed; each requested token id "
            f"must be owned by exactly one TP rank. bad_ids={bad} owner_counts={counts} "
            f"tp_size={tp_size} shard_vocab={shard_vocab}"
        )
    return None


def _reduce_dp_cp_float_counts(values: list[float], device: torch.device | int) -> list[float]:
    """Sum scalar counters over the DP-with-CP group.

    BPM losses can legitimately have zero local aligned rows on one
    CP rank when all response rows for a short sample live on another CP slice.
    Alignment guards therefore need group-level counts, not only local counts.
    """
    counts = torch.tensor(values, dtype=torch.float64, device=device)
    if torch.distributed.is_initialized():
        group = mpu.get_data_parallel_group(with_context_parallel=True)
        if torch.distributed.get_world_size(group=group) > 1:
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=group)
    return [float(v) for v in counts.detach().cpu().tolist()]


def _raise_if_any_dp_cp_rank_failed(
    *,
    failed: bool,
    local_message: str | None,
    device: torch.device | int,
    context: str,
) -> None:
    """Synchronize fail-fast guards before later DP/CP collectives.

    Some tokenization/alignment checks are data-dependent, so one DP rank can hit
    a bad sample while another rank has clean local samples.  If the bad rank
    raises immediately and the clean rank proceeds into a later all-reduce, the
    job can hang until the distributed timeout.  Call this at a common point that
    every DP/CP rank reaches after recording local failures.
    """
    flag = torch.tensor([1 if failed else 0], dtype=torch.int32, device=device)
    if torch.distributed.is_initialized():
        group = mpu.get_data_parallel_group(with_context_parallel=True)
        if torch.distributed.get_world_size(group=group) > 1:
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=group)
    if int(flag.item()) == 0:
        return
    if failed and local_message:
        raise ValueError(local_message)
    raise ValueError(
        f"{context} failed on another DP/CP rank before a collective. "
        "Check the corresponding rank logs for the detailed local error."
    )


def _any_cp_rank_failed(*, failed: bool, device: torch.device | int) -> bool:
    """Synchronize a local per-sample failure flag only inside the CP group.

    SimCT may need CP collectives inside a sample to gather sparse span-tail
    logits.  If one CP rank has already detected bad local data, every CP rank
    for that same sample must skip those collectives and defer the user-facing
    exception to the existing DP-with-CP post-loop synchronizer.
    """
    flag = torch.tensor([1 if failed else 0], dtype=torch.int32, device=device)
    if torch.distributed.is_initialized() and mpu.get_context_parallel_world_size() > 1:
        torch.distributed.all_reduce(
            flag,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_context_parallel_group(),
        )
    return bool(int(flag.item()))


def _count_cp_unique_samples_with_alignment(local_flags: list[float], device: torch.device | int) -> float:
    """Count original samples with any local aligned/valid token once across CP.

    Callers must pass one flag per original sample, in the same sample order on
    every CP rank.  DP-with-CP metric all-reduce happens later in model.py.  If
    every CP rank contributes a per-sample indicator, sample-count denominators
    are inflated by up to cp_size.  Reduce indicators inside the CP group first,
    then let only CP rank 0 contribute the per-sample union count to the later
    DP-with-CP reduction.
    """
    if not local_flags:
        return 0.0
    flags = torch.tensor(local_flags, dtype=torch.float32, device=device)
    cp_size = mpu.get_context_parallel_world_size()
    if torch.distributed.is_initialized() and cp_size > 1:
        cp_group = mpu.get_context_parallel_group()
        local_len = torch.tensor([flags.numel()], dtype=torch.int64, device=device)
        min_len = local_len.clone()
        max_len = local_len.clone()
        torch.distributed.all_reduce(min_len, op=torch.distributed.ReduceOp.MIN, group=cp_group)
        torch.distributed.all_reduce(max_len, op=torch.distributed.ReduceOp.MAX, group=cp_group)
        if int(min_len.item()) != int(max_len.item()):
            raise ValueError(
                "CP sample-alignment metric contract violated: every CP rank must pass "
                "one flag per original sample in the same order; "
                f"local_len={int(local_len.item())} min={int(min_len.item())} max={int(max_len.item())}"
            )
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.SUM, group=cp_group)
        if mpu.get_context_parallel_rank() != 0:
            return 0.0
    return float((flags > 0).float().sum().detach().cpu().item())


def _mask_invalid_vocab_rows_(
    logits_shard: torch.Tensor,
    *,
    real_vocab_size: int,
    vocab_start: int,
) -> torch.Tensor:
    """Mask fake padded vocab rows in a vocab-parallel logits shard.

    Megatron pads vocab to make shards equal.  Teacher lm_head rows beyond the HF
    real vocab are zero-padded for shape parity, but they are not valid tokens and
    must not enter softmax denominators or top-k selection.
    """
    shard_vocab = logits_shard.shape[-1]
    local_ids = torch.arange(shard_vocab, device=logits_shard.device)
    global_ids = local_ids + int(vocab_start)
    invalid = global_ids >= int(real_vocab_size)
    if bool(invalid.any().item()):
        logits_shard = logits_shard.clone()
        logits_shard[..., invalid] = -torch.finfo(logits_shard.dtype).max
    return logits_shard
