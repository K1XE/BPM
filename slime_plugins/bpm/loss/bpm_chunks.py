"""Build chunks from a realized (student, teacher) token alignment (no torch).

A chunk is a maximal run between two sync points -- byte offsets that are both a
student and a teacher boundary. Each chunk carries the student positions it covers.
"""
from __future__ import annotations

from .bpm_seg import Chunk


class ChunkPositions:
    """A chunk plus the student positions inside it.

    positions: (global_student_index, offset_in_chunk, realized_student_id,
    at_teacher_boundary), indexing the response token sequence.
    """

    __slots__ = ("chunk", "positions")

    def __init__(self, chunk: Chunk, positions: list):
        self.chunk = chunk
        self.positions = positions


def _cumulative(byte_lens: list[int]) -> list[int]:
    off = [0]
    for n in byte_lens:
        off.append(off[-1] + n)
    return off


def build_chunks(
    stu_ids: list[int],
    tea_ids: list[int],
    stu_byte_map: dict[int, bytes],
    tea_byte_map: dict[int, bytes],
    tea_dists: list,
) -> list:
    """Partition the response into byte-aligned chunks. tea_dists is aligned with tea_ids;
    tea_dists[j] predicts tea_ids[j].
    """
    if len(tea_dists) != len(tea_ids):
        raise ValueError("tea_dists must align 1:1 with tea_ids")
    stu_bytes = [stu_byte_map[int(i)] for i in stu_ids]
    tea_bytes = [tea_byte_map[int(i)] for i in tea_ids]
    stu_off = _cumulative([len(b) for b in stu_bytes])
    tea_off = _cumulative([len(b) for b in tea_bytes])
    if stu_off[-1] != tea_off[-1]:
        raise ValueError(f"student/teacher byte length mismatch: {stu_off[-1]} vs {tea_off[-1]}")

    sync = sorted(set(stu_off) & set(tea_off))
    tea_off_set = set(tea_off)

    out: list = []
    for s_idx in range(len(sync) - 1):
        a, b = sync[s_idx], sync[s_idx + 1]
        tea_j = [j for j in range(len(tea_ids)) if a <= tea_off[j] < b]
        chunk = Chunk([tea_bytes[j] for j in tea_j], [tea_dists[j] for j in tea_j], tea_byte_map)
        positions = []
        for i in range(len(stu_ids)):
            if a <= stu_off[i] < b:
                offset_in_chunk = stu_off[i] - a
                at_boundary = (stu_off[i] in tea_off_set)
                positions.append((i, offset_in_chunk, int(stu_ids[i]), at_boundary))
        out.append(ChunkPositions(chunk, positions))
    return out
