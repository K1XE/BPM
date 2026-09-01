"""Per-position teacher byte trie for O(len) byte-prefix marginal queries (no torch).

    prefix_mass(p) = sum q[t] over teacher tokens whose bytes start with p
    exact_mass(t)  = sum q[t] over teacher tokens whose bytes equal t

Built once per teacher position from that position's next-token distribution.
"""
from __future__ import annotations


class _Node:
    __slots__ = ("children", "subtree_mass", "exact_mass")

    def __init__(self):
        self.children: dict[int, "_Node"] = {}
        self.subtree_mass: float = 0.0
        self.exact_mass: float = 0.0


class TeacherDistTrie:
    """Byte trie of one teacher next-token distribution, for O(len) mass queries."""

    def __init__(self, dist: dict[int, float], byte_map: dict[int, bytes]):
        self.root = _Node()
        for tid, p in dist.items():
            b = byte_map.get(int(tid))
            if b is None:
                # Control / non-byte token: no byte content, excluded from the byte walk.
                continue
            node = self.root
            node.subtree_mass += p
            for byte in b:
                node = node.children.setdefault(byte, _Node())
                node.subtree_mass += p
            node.exact_mass += p

    def _node(self, prefix: bytes):
        node = self.root
        for byte in prefix:
            node = node.children.get(byte)
            if node is None:
                return None
        return node

    def prefix_mass(self, prefix: bytes) -> float:
        if not prefix:
            return self.root.subtree_mass
        n = self._node(prefix)
        return n.subtree_mass if n is not None else 0.0

    def exact_mass(self, target: bytes) -> float:
        n = self._node(target)
        return n.exact_mass if n is not None else 0.0
