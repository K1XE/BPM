"""Small metric / reference helpers (no torch)."""
from __future__ import annotations


def bpm_repetition_fraction(token_ids) -> float:
    """Adjacent token repetition fraction in [0, 1]."""
    ids = [int(x) for x in token_ids]
    if len(ids) < 2:
        return 0.0
    reps = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    return float(reps) / float(len(ids) - 1)


def bpm_coverage_fraction(covered_rows: int | float, total_rows: int | float) -> float:
    denom = float(total_rows)
    if denom <= 0.0:
        return 0.0
    return float(covered_rows) / denom


def bpm_mean(values) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))
