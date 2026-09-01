"""Byte-prefix role-bridge map phi -- the vectorizable core (no torch).

    phi[t] = the longest student-token byte-prefix of teacher token t's bytes, or -1

At a boundary the target is the scatter-add target[u] = sum_{phi(t)=u} q[t], so the
forward-KL CE collapses to a gather + dot. Exact for 1:1 and N:1; for spanning rows
phi is a fast-path approximation and those rows route to the chain path.
"""
from __future__ import annotations

from .bpm_position import build_student_byte_trie


def student_first_token(student_trie, tea_bytes: bytes) -> int | None:
    """Deepest student-token id on tea_bytes' path (longest student-token prefix), or None."""
    node = student_trie
    deepest = node.token_id
    for byte in tea_bytes:
        node = node.children.get(byte)
        if node is None:
            break
        if node.token_id is not None:
            deepest = node.token_id
    return deepest


def student_first_token_capped(student_trie, tea_bytes: bytes, max_len: int) -> int | None:
    """Longest student-token prefix within `max_len` bytes, so a token longer than the
    chunk boundary cannot steal mass from the shorter in-chunk token.
    """
    node = student_trie
    deepest = node.token_id
    for i, byte in enumerate(tea_bytes):
        if i >= int(max_len):
            break
        node = node.children.get(byte)
        if node is None:
            break
        if node.token_id is not None:
            deepest = node.token_id
    return deepest


def build_phi(stu_byte_map: dict[int, bytes], tea_byte_map: dict[int, bytes]) -> dict[int, int]:
    """{teacher_id: student_id} longest student-token byte-prefix map (-1 if none)."""
    trie = build_student_byte_trie(stu_byte_map)
    phi: dict[int, int] = {}
    for tid, tbytes in tea_byte_map.items():
        u = student_first_token(trie, tbytes)
        phi[int(tid)] = int(u) if u is not None else -1
    return phi


def build_conditional_tail_phi(
    stu_byte_map: dict[int, bytes],
    tea_byte_map: dict[int, bytes],
    realized_prefix: bytes,
    *,
    remaining_len: int,
) -> dict[int, int]:
    """Teacher-id -> student-id map for an exact mid-teacher-token tail chunk.

    Valid only when the chunk ends at the realized teacher token's end. Teacher tokens
    equal to the realized prefix have no next byte and map to -1.
    """
    trie = build_student_byte_trie(stu_byte_map)
    out: dict[int, int] = {}
    rem = max(int(remaining_len), 0)
    for tid, tbytes in tea_byte_map.items():
        if not tbytes.startswith(realized_prefix):
            continue
        suffix = tbytes[len(realized_prefix):]
        u = student_first_token_capped(trie, suffix, rem)
        out[int(tid)] = int(u) if u is not None else -1
    return out


def scatter_target(phi: dict[int, int], q: dict[int, float]) -> dict[int, float]:
    """Reference token-marginal target {student_id: mass} = sum_{phi(t)=u} q[t]
    (teacher tokens with phi=-1 fall into the complement mass)."""
    target: dict[int, float] = {}
    for tid, p in q.items():
        u = phi.get(int(tid), -1)
        if u < 0:
            continue
        target[u] = target.get(u, 0.0) + p
    return target
