"""Overlap-vocabulary construction for the SimCT baseline."""

from __future__ import annotations

from argparse import Namespace

import torch


def _find_simct_overlap_tokens(student_tokenizer, teacher_tokenizer, device):
    """Find overlap tokens between teacher and student tokenizers.

    Normalizes prefix characters (Ġ and ▁ are different BPE whitespace markers)
    before matching, so tokens like "Ġhello" and "▁hello" are considered the same.

    Returns:
        stu_overlap_ids: tensor of student token IDs for overlap tokens [num_overlap]
        tea_overlap_ids: tensor of teacher token IDs for overlap tokens [num_overlap]
    """
    stu_vocab = {k.replace("Ġ", "▁"): v for k, v in student_tokenizer.get_vocab().items()}
    tea_vocab = {k.replace("Ġ", "▁"): v for k, v in teacher_tokenizer.get_vocab().items()}
    # Distributed correctness: never iterate a Python set directly here.
    # Python hash randomization may give different overlap order per process,
    # which would make TP all-reduce columns refer to different token IDs.
    #
    # Runtime performance: use a deterministic student-id order instead of token
    # string order.  The SimCT full objective is invariant to the virtual
    # vocabulary column order as long as student/teacher columns stay paired, and
    # student-id order makes the hot student gather much closer to a coalesced
    # index-select for the Qwen student.
    overlap_items = sorted(
        ((int(stu_vocab[t]), int(tea_vocab[t])) for t in (set(stu_vocab.keys()) & set(tea_vocab.keys()))),
        key=lambda pair: (pair[0], pair[1]),
    )

    stu_ids = [stu_id for stu_id, _ in overlap_items]
    tea_ids = [tea_id for _, tea_id in overlap_items]

    # Ensure EOS is included in overlap
    stu_eos = student_tokenizer.eos_token_id
    tea_eos = teacher_tokenizer.eos_token_id
    if stu_eos is not None and tea_eos is not None and (stu_eos, tea_eos) not in set(zip(stu_ids, tea_ids, strict=False)):
        stu_ids.append(stu_eos)
        tea_ids.append(tea_eos)

    # Deduplicate by TEACHER id.  A teacher id appearing in two columns is
    # double-counted in the teacher virtual-vocab softmax: its exp() enters the
    # partition twice, halving its per-column mass and shifting every other
    # column by -ln(1+p).  This triggers for the teacher EOS whenever the teacher
    # EOS string also matches a NON-eos student token by surface form -- e.g. GLM
    # eos "<|endoftext|>" (a real stop token) collides with Qwen "<|endoftext|>"
    # (the PAD token, id != Qwen eos "<|im_end|>").  The natural-overlap pair
    # (stu_pad, tea_eos) then coexists with the appended (stu_eos, tea_eos),
    # leaking the teacher's stop mass onto the student PAD and halving the real
    # EOS signal.  Keep one column per teacher id, preferring the pair that maps
    # the teacher EOS to the student's real EOS.
    if len(set(tea_ids)) != len(tea_ids):
        seen_tea: dict[int, int] = {}
        dedup_stu: list[int] = []
        dedup_tea: list[int] = []
        for s_id, t_id in zip(stu_ids, tea_ids, strict=False):
            if t_id in seen_tea:
                # On a conflict for the teacher EOS id, keep the student-EOS pair.
                if t_id == tea_eos and s_id == stu_eos:
                    dedup_stu[seen_tea[t_id]] = s_id
                continue
            seen_tea[t_id] = len(dedup_tea)
            dedup_stu.append(s_id)
            dedup_tea.append(t_id)
        stu_ids, tea_ids = dedup_stu, dedup_tea

    import logging as _logging
    _logging.getLogger(__name__).info(
        f"[OPD][simct] overlap_tokens: {len(stu_ids)} "
        f"(stu_vocab={len(stu_vocab)}, tea_vocab={len(tea_vocab)}, "
        f"overlap_ratio={len(stu_ids)/max(len(stu_vocab),1)*100:.1f}%)"
    )

    return (
        torch.tensor(stu_ids, dtype=torch.long, device=device),
        torch.tensor(tea_ids, dtype=torch.long, device=device),
    )

def _get_simct_teacher_to_student_id_map(
    args: Namespace,
    tea_overlap_ids: torch.Tensor,
    stu_overlap_ids: torch.Tensor,
    *,
    teacher_real_vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a GPU lookup table mapping teacher token id -> student token id.

    simct top-K is selected in the teacher vocabulary.  The student loss needs
    the corresponding student token id for each shared token string.  Building
    this dense map once avoids Python dict lookups in the per-microbatch hot
    path and lets the exact top-K path gather student logits by teacher ids.
    """
    cache = getattr(args, "_simct_tea_to_stu_id_map", None)
    if (
        cache is None
        or cache.device != device
        or int(cache.numel()) != int(teacher_real_vocab_size)
    ):
        cache = torch.full(
            (int(teacher_real_vocab_size),),
            -1,
            dtype=torch.long,
            device=device,
        )
        tea_ids = tea_overlap_ids.to(device=device, dtype=torch.long)
        stu_ids = stu_overlap_ids.to(device=device, dtype=torch.long)
        valid = (tea_ids >= 0) & (tea_ids < int(teacher_real_vocab_size))
        cache[tea_ids[valid]] = stu_ids[valid]
        args._simct_tea_to_stu_id_map = cache
    return cache

def _get_simct_teacher_overlap_mask(
    args: Namespace,
    tea_overlap_ids: torch.Tensor,
    *,
    vocab_start: int,
    shard_vocab: int,
    real_vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return local teacher-vocab mask for token ids in the simct overlap set."""
    key = (int(vocab_start), int(shard_vocab), int(real_vocab_size), str(device))
    cache_key = getattr(args, "_simct_teacher_overlap_mask_key", None)
    cache = getattr(args, "_simct_teacher_overlap_mask", None)
    if cache is None or cache.device != device or cache_key != key:
        local_ids = tea_overlap_ids.to(device=device, dtype=torch.long) - int(vocab_start)
        valid = (local_ids >= 0) & (local_ids < int(shard_vocab))
        mask = torch.zeros((int(shard_vocab),), dtype=torch.bool, device=device)
        if bool(valid.any().item()):
            mask[local_ids[valid]] = True
        # Exclude Megatron padding slots.
        real_end = max(0, min(int(real_vocab_size) - int(vocab_start), int(shard_vocab)))
        if real_end < int(shard_vocab):
            mask[real_end:] = False
        args._simct_teacher_overlap_mask_key = key
        args._simct_teacher_overlap_mask = mask
        cache = mask
    return cache

def _simct_teacher_overlap_topk_candidates(
    teacher_logits_shard: torch.Tensor,
    tea_overlap_ids: torch.Tensor,
    *,
    k: int,
    args: Namespace,
    vocab_start: int,
    shard_vocab: int,
    real_vocab_size: int,
    tp_size: int,
    tp_group,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact teacher top-K over the simct overlap vocab without materializing it.

    Previous simct top-K first gathered ``[rows, overlap_vocab]`` teacher
    logits (143k columns for GLM/Qwen) and then selected top-K.  That made
    top-K nearly as expensive as full overlap-vocab KL.  This helper instead
    masks non-overlap columns in the already-computed teacher vocab shard, takes
    shard-local top-K, and only exchanges ``K`` candidates across TP ranks.
    """
    if teacher_logits_shard.numel() == 0:
        empty_vals = teacher_logits_shard.new_empty((teacher_logits_shard.shape[0], 0), dtype=torch.float32)
        empty_ids = torch.empty((teacher_logits_shard.shape[0], 0), dtype=torch.long, device=teacher_logits_shard.device)
        return empty_vals, empty_ids

    k = max(1, min(int(k), int(tea_overlap_ids.numel())))
    local_k = min(k, int(shard_vocab))
    overlap_mask = _get_simct_teacher_overlap_mask(
        args,
        tea_overlap_ids,
        vocab_start=vocab_start,
        shard_vocab=shard_vocab,
        real_vocab_size=real_vocab_size,
        device=teacher_logits_shard.device,
    )
    neg_inf = torch.finfo(teacher_logits_shard.dtype).min
    # In-place is safe: teacher logits are frozen, only used for top-K candidate
    # selection in this branch, and avoiding an extra [rows, vocab] allocation is
    # the point of this fast path.
    masked_logits = teacher_logits_shard.masked_fill_(~overlap_mask.unsqueeze(0), neg_inf)
    local_vals, local_pos = masked_logits.topk(local_k, dim=-1)
    local_ids = local_pos.to(dtype=torch.long) + int(vocab_start)

    if tp_size > 1:
        all_vals = [torch.empty_like(local_vals) for _ in range(tp_size)]
        all_ids = [torch.empty_like(local_ids) for _ in range(tp_size)]
        torch.distributed.all_gather(all_vals, local_vals.contiguous(), group=tp_group)
        torch.distributed.all_gather(all_ids, local_ids.contiguous(), group=tp_group)
        vals = torch.cat(all_vals, dim=-1)
        ids = torch.cat(all_ids, dim=-1)
        final_k = min(k, vals.shape[-1])
        vals, pos = vals.topk(final_k, dim=-1)
        ids = ids.gather(-1, pos)
    else:
        vals, ids = local_vals, local_ids

    # ``teacher_logits_shard`` is already temperature-scaled by _batched_lm_head_rows.
    return vals.float(), ids

def _get_simct_overlap_pair_map(
    args: Namespace,
    stu_overlap_ids: torch.Tensor,
    tea_overlap_ids: torch.Tensor,
) -> dict[tuple[int, int], int]:
    cache_key = (
        int(stu_overlap_ids.numel()),
        int(tea_overlap_ids.numel()),
        str(stu_overlap_ids.device),
        str(tea_overlap_ids.device),
        int(stu_overlap_ids.data_ptr()),
        int(tea_overlap_ids.data_ptr()),
    )
    if getattr(args, "_simct_overlap_pairs_key", None) != cache_key:
        args._simct_overlap_pairs = {
            (int(stu_id), int(tea_id)): dim
            for dim, (stu_id, tea_id) in enumerate(
                zip(
                    stu_overlap_ids.detach().cpu().tolist(),
                    tea_overlap_ids.detach().cpu().tolist(),
                    strict=False,
                )
            )
        }
        args._simct_overlap_pairs_key = cache_key
    return args._simct_overlap_pairs
