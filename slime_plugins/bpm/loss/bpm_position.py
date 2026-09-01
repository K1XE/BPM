"""Per-student-position target by byte-prefix marginalization (no torch).

    target[u] = mass of teacher byte continuations whose greedy student token is u.

Walk the student byte trie; at prefix p the mass ending exactly at p is
seg_prob(a, p) - sum_children seg_prob(a, p + byte), assigned to the deepest
student-token ancestor. The telescope conserves mass with no leak at internal nodes.
Control tokens go to the stopping bridge -- pass their ids in exclude_ids.
"""
from __future__ import annotations


class StudentTrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: dict[int, "StudentTrieNode"] = {}
        self.token_id: int | None = None


def build_student_byte_trie(token_bytes: dict[int, bytes], exclude_ids: set[int] | None = None) -> StudentTrieNode:
    """Build a byte trie of student tokens. Empty-byte / excluded ids are skipped."""
    exclude = exclude_ids or set()
    root = StudentTrieNode()
    for tid, b in token_bytes.items():
        if int(tid) in exclude or not b:
            continue
        node = root
        for byte in b:
            node = node.children.setdefault(byte, StudentTrieNode())
        node.token_id = int(tid)
    return root


def first_token_partition(chunk, a: int, trie: StudentTrieNode, eps: float = 1e-12) -> dict[int, float]:
    """Distribution over student tokens by greedy first token at boundary offset `a`.

    Returns {student_token_id: prob}; use partition_residual to track the root leak.
    Branches with cylinder mass <= eps are pruned (safe by prefix monotonicity).
    """
    target: dict[int, float] = {}

    def visit(node: StudentTrieNode, p: bytes, m_p: float, last_tok: int | None) -> None:
        cur_tok = node.token_id if node.token_id is not None else last_tok
        child_mass = 0.0
        for byte, child in node.children.items():
            cp = p + bytes([byte])
            m_child = chunk.seg_prob(a, cp)
            # Count every child's mass (even sub-eps) so this node's residual is not
            # over-attributed to its ancestor token.
            child_mass += m_child
            if m_child > eps:
                visit(child, cp, m_child, cur_tok)
        exact_here = m_p - child_mass
        if exact_here > eps and cur_tok is not None:
            target[cur_tok] = target.get(cur_tok, 0.0) + exact_here

    visit(trie, b"", chunk.seg_prob(a, b""), None)
    return target


def partition_residual(chunk, a: int, target: dict[int, float]) -> float:
    """Mass not assigned to any student token (should be ~0 for a complete byte vocab)."""
    return float(chunk.seg_prob(a, b"") - sum(target.values()))


def position_target(
    chunk,
    a: int,
    trie: StudentTrieNode,
    *,
    student_eos_id: int | None = None,
    teacher_stop_ids: set[int] | None = None,
    eps: float = 1e-12,
) -> dict[int, float]:
    """Target distribution at student byte offset `a`. Mid-token offsets use the same
    conditional partition; boundary stop mass goes to the student EOS via the bridge.
    """
    target = first_token_partition(chunk, a, trie, eps=eps)
    if student_eos_id is not None:
        stop_mass = chunk.stop_prob(a, teacher_stop_ids)
        if stop_mass > eps:
            sid = int(student_eos_id)
            target[sid] = target.get(sid, 0.0) + stop_mass
    return target
