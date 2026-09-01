"""Reference loss assembly: alignment -> per-position targets -> forward-KL.
build_position_targets is pure Python; bpm_loss_from_logits is the torch reference.
"""
from __future__ import annotations

from .bpm_chunks import build_chunks
from .bpm_position import position_target


def build_position_targets(
    stu_ids: list[int],
    tea_ids: list[int],
    stu_byte_map: dict[int, bytes],
    tea_byte_map: dict[int, bytes],
    tea_dists: list,
    student_trie,
    student_eos_id: int | None = None,
    teacher_stop_ids: set[int] | None = None,
    eps: float = 1e-12,
) -> dict[int, dict]:
    """{global_student_index: {student_token_id: prob}} for every covered position. A
    position is covered iff it lies in a byte-aligned chunk.
    """
    targets: dict[int, dict] = {}
    for cp in build_chunks(stu_ids, tea_ids, stu_byte_map, tea_byte_map, tea_dists):
        for (gidx, offset, realized_sid, at_boundary) in cp.positions:
            targets[int(gidx)] = position_target(
                cp.chunk,
                offset,
                student_trie,
                student_eos_id=student_eos_id,
                teacher_stop_ids=teacher_stop_ids,
                eps=eps,
            )
    return targets


def bpm_loss_from_logits(
    student_logits,            # [R, V] student logits for the R response rows this rank owns
    position_targets: dict,    # {row_index in [0,R): {student_token_id: prob}}
    *,
    max_k: int | None = None,
):
    """Pack sparse targets and run the CP-shared mass-conserving forward-KL. Rows absent
    from position_targets are masked out. Returns (loss_sum, cp_rows, metrics).
    """
    import torch

    from .bpm_kernels import sparse_forward_kl_rows

    r = int(student_logits.shape[0])
    device = student_logits.device
    rows = [position_targets.get(i) for i in range(r)]
    k = max_k or max((len(t) for t in rows if t), default=1)
    k = max(int(k), 1)

    target_ids = torch.zeros((r, k), dtype=torch.long, device=device)
    target_probs = torch.zeros((r, k), dtype=torch.float32, device=device)
    target_mask = torch.zeros((r, k), dtype=torch.bool, device=device)
    row_mask = torch.zeros((r,), dtype=torch.bool, device=device)
    other = torch.zeros((r,), dtype=torch.float32, device=device)

    for i, t in enumerate(rows):
        if not t:
            continue
        row_mask[i] = True
        items = sorted(t.items(), key=lambda kv: -kv[1])[:k]   # keep top-k by prob if over-full
        s = 0.0
        for j, (tid, p) in enumerate(items):
            target_ids[i, j] = int(tid)
            target_probs[i, j] = float(p)
            target_mask[i, j] = True
            s += float(p)
        other[i] = max(0.0, 1.0 - s)   # complement (truncation tail / unrepresented mass)

    return sparse_forward_kl_rows(
        student_logits,
        target_ids,
        target_probs,
        target_mask,
        row_mask,
        other_prob=other,
    )
