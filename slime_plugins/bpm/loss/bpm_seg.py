"""Realized-segmentation byte walk over one byte-aligned chunk (no torch).

seg_prob(a, c) is the byte-prefix marginal that the teacher stream continues with
byte string c at offset a. At a boundary a candidate ending inside t_k uses
prefix_mass(q[k], c); a spanning candidate consumes exact_mass(q[k], t_k) and
recurses into q[k+1]. Mid-token, the mass is conditioned on the realized prefix.
Spanning over-count is removed by the trie residualization in bpm_position.
"""
from __future__ import annotations

from bisect import bisect_right

from .bpm_trie import TeacherDistTrie


class Chunk:
    """One byte-aligned chunk.

    tea_bytes  realized teacher tokens' byte strings (all nonempty)
    tea_dists  tea_dists[j] predicts tea token j
    byte_map   teacher token id -> bytes
    """

    def __init__(self, tea_bytes: list[bytes], tea_dists: list[dict], byte_map: dict[int, bytes]):
        if len(tea_bytes) != len(tea_dists):
            raise ValueError("chunk needs one teacher dist per realized teacher token")
        if not all(len(b) > 0 for b in tea_bytes):
            # zero-length tokens would duplicate byte offsets and drop a bisect factor
            raise ValueError("chunk has a zero-length teacher token (exclude control tokens first)")
        self.tea_bytes = tea_bytes
        self.tea_dists = tea_dists
        self.byte_map = byte_map
        # callers may pass a torch-backed object exposing prefix_mass/exact_mass
        self.tries = [
            q if hasattr(q, "prefix_mass") and hasattr(q, "exact_mass") else TeacherDistTrie(q, byte_map)
            for q in tea_dists
        ]
        self.off = [0]
        for b in tea_bytes:
            self.off.append(self.off[-1] + len(b))
        self.chunk_len = self.off[-1]

    def _token_at(self, pos: int) -> int:
        return bisect_right(self.off, pos) - 1

    def is_chunk_end(self, a: int) -> bool:
        """True iff offset a is exactly the chunk end (routes to the stopping bridge)."""
        return a == self.chunk_len

    def _chain_prefix_mass_from_boundary(self, a: int, c: bytes) -> float:
        """Unconditional chain prefix mass for bytes ``c`` starting at teacher boundary ``a``."""
        if a == self.chunk_len:
            return 1.0 if not c else 0.0
        p = 1.0
        ci = 0
        pos = a
        n = len(c)
        while ci < n:
            k = self._token_at(pos)
            within = pos - self.off[k]
            if within == 0:
                rem = c[ci:]
                lk = len(self.tea_bytes[k])
                if len(rem) <= lk:
                    p *= self.tries[k].prefix_mass(rem)
                    return p
                if c[ci:ci + lk] != self.tea_bytes[k]:
                    return 0.0
                p *= self.tries[k].exact_mass(self.tea_bytes[k])
                ci += lk
                pos += lk
            else:
                forced = self.tea_bytes[k][within:]
                take = min(len(forced), n - ci)
                if c[ci:ci + take] != forced[:take]:
                    return 0.0
                ci += take
                pos += take
        return p

    def seg_prob(self, a: int, c: bytes) -> float:
        """Conditional prob the teacher continues with bytes `c` at offset `a`, conditioned
        on the token prefix already emitted before `a`.
        """
        if not (0 <= a <= self.chunk_len):
            raise ValueError(f"seg_prob offset out of range: a={a} chunk_len={self.chunk_len}")
        if a + len(c) > self.chunk_len:
            # Bytes past the chunk boundary are the next chunk's business.
            return 0.0
        if not c:
            return 1.0
        if a == self.chunk_len:
            return 0.0

        k = self._token_at(a)
        within = a - self.off[k]
        if within == 0:
            return self._chain_prefix_mass_from_boundary(a, c)

        boundary = self.off[k]
        realized_prefix = self.tea_bytes[k][:within]
        denom = self._chain_prefix_mass_from_boundary(boundary, realized_prefix)
        if denom <= 0.0:
            return 0.0
        num = self._chain_prefix_mass_from_boundary(boundary, realized_prefix + c)
        return num / denom

    def stop_prob(self, a: int, stop_ids: set[int] | None) -> float:
        """Conditional teacher stop mass at byte offset `a`. Stop tokens have no byte
        continuation, so they are possible only at a decision boundary.
        """
        if not stop_ids or not (0 <= a < self.chunk_len):
            return 0.0
        k = self._token_at(a)
        if a != self.off[k]:
            return 0.0
        q = self.tea_dists[k]
        if hasattr(q, "stop_mass"):
            return float(q.stop_mass(stop_ids))
        return float(sum(float(q.get(int(t), 0.0)) for t in stop_ids))
