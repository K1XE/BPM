"""GPU-free behaviour-equivalence gate for the BPM objective core.

Asserts the pure core reproduces the golden baseline: trie masses, phi, first-token
queries and partition targets. The core never imports torch (proved below).
Run: python3 -X dev slime_plugins/bpm/tests/test_bpm_core.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_SLIME_ROOT = _HERE.parents[3]          # .../bpm-release/slime
_GOLDEN_DIR = Path(os.environ.get("BPM_GOLDEN_DIR", _HERE.parent / "golden"))

if str(_SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLIME_ROOT))

from slime_plugins.bpm.loss.bpm_phi import build_phi, student_first_token
from slime_plugins.bpm.loss.bpm_position import (
    build_student_byte_trie,
    first_token_partition,
    partition_residual,
)
from slime_plugins.bpm.loss.bpm_seg import Chunk
from slime_plugins.bpm.loss.bpm_trie import TeacherDistTrie

TOL = 1e-9
LOSS_TOL = 1e-6

# which student vocab + exclusion each partition case used
_PARTITION_SETUP = {
    "digits": ("all", True),
    "multibyte_cjk": ("all", True),
    "codex_supertoken": ("super", False),
}


def _load_golden() -> dict:
    return json.loads((_GOLDEN_DIR / "golden_core.json").read_text())


def _byte_maps(meta: dict):
    def m(key):
        return {int(k): bytes.fromhex(v) for k, v in meta[key].items()}

    tea = m("tea_byte_map_hex")
    content = m("stu_content_hex")
    special = m("stu_special_hex")
    super_map = m("stu_super_hex")
    return tea, content, special, super_map


def check_build_phi(g) -> None:
    _, content, _, _ = _byte_maps(g["_meta"])
    tea = {int(k): bytes.fromhex(v) for k, v in g["_meta"]["tea_byte_map_hex"].items()}
    got = build_phi(content, tea)
    want = {int(k): int(v) for k, v in g["build_phi"].items()}
    assert got == want, f"build_phi mismatch: {got} vs {want}"


def check_student_first_token(g) -> None:
    tea, content, special, _ = _byte_maps(g["_meta"])
    stu_all = {**content, **special}
    special_ids = set(special)
    trie_all = build_student_byte_trie(stu_all)
    trie_excl = build_student_byte_trie(stu_all, exclude_ids=special_ids)
    for q in g["student_first_token_queries"]:
        qb = bytes.fromhex(q["query_bytes_hex"])
        a = student_first_token(trie_all, qb)
        b = student_first_token(trie_excl, qb)
        a = None if a is None else int(a)
        b = None if b is None else int(b)
        assert a == q["with_specials"], f"with_specials {qb.hex()}: {a} vs {q['with_specials']}"
        assert b == q["specials_excluded"], f"excluded {qb.hex()}: {b} vs {q['specials_excluded']}"


def check_teacher_dist_trie(g) -> None:
    tea = {int(k): bytes.fromhex(v) for k, v in g["_meta"]["tea_byte_map_hex"].items()}
    tdt_dist = {int(k): v for k, v in g["teacher_dist_trie"]["dist"].items()}
    tdt = TeacherDistTrie(tdt_dist, tea)
    for e in g["teacher_dist_trie"]["prefix_mass"]:
        got = tdt.prefix_mass(bytes.fromhex(e["prefix_hex"]))
        assert abs(got - e["trie"]) <= TOL, f"prefix_mass {e['prefix_hex']}: {got} vs {e['trie']}"
        assert abs(got - e["reference"]) <= TOL, f"prefix_mass vs ref {e['prefix_hex']}"
    for e in g["teacher_dist_trie"]["exact_mass"]:
        got = tdt.exact_mass(bytes.fromhex(e["target_hex"]))
        assert abs(got - e["trie"]) <= TOL, f"exact_mass {e['target_hex']}: {got} vs {e['trie']}"
        assert abs(got - e["reference"]) <= TOL, f"exact_mass vs ref {e['target_hex']}"


def check_first_token_partition(g) -> None:
    tea, content, special, super_map = _byte_maps(g["_meta"])
    stu_all = {**content, **special}
    special_ids = set(special)
    for case in g["first_token_partition"]:
        which, do_exclude = _PARTITION_SETUP[case["name"]]
        stu_map = stu_all if which == "all" else super_map
        exclude = special_ids if do_exclude else None
        tea_bytes = [bytes.fromhex(h) for h in case["tea_bytes_hex"]]
        tea_dist = {int(k): v for k, v in case["tea_dist"].items()}
        ch = Chunk(tea_bytes, [tea_dist], tea)
        trie = build_student_byte_trie(stu_map, exclude_ids=exclude)
        tgt = first_token_partition(ch, case["offset"], trie)
        want = {int(k): v for k, v in case["target"].items()}
        assert set(tgt) == set(want), f"{case['name']} support: {set(tgt)} vs {set(want)}"
        for u in want:
            assert abs(tgt[u] - want[u]) <= TOL, f"{case['name']}[{u}]: {tgt[u]} vs {want[u]}"
        res = partition_residual(ch, case["offset"], tgt)
        assert abs(res - case["residual"]) <= TOL, f"{case['name']} residual: {res} vs {case['residual']}"


def check_loss_reference_if_torch() -> str:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # torch absent -> core gate already passed torch-free
        return f"SKIP loss reference (torch unavailable: {type(exc).__name__})"
    loss_path = _GOLDEN_DIR / "golden_loss.json"
    if not loss_path.exists():
        return "SKIP loss reference (golden_loss.json absent)"
    try:
        from slime_plugins.bpm.loss.bpm_loss_ref import bpm_loss_from_logits
    except Exception as exc:
        return f"SKIP loss reference (import failed: {type(exc).__name__}: {exc})"
    import torch

    gl = json.loads(loss_path.read_text())
    logits = torch.tensor(gl["student_logits"], dtype=torch.float32)
    position_targets = {
        int(r): {int(t): float(p) for t, p in tgt.items()}
        for r, tgt in gl["position_targets"].items()
    }
    loss_sum, cp_rows, metrics = bpm_loss_from_logits(logits, position_targets)
    ls = float(loss_sum.detach().cpu().item())
    ce = float(metrics["bpm_ce_mean"].detach().cpu().item())
    assert abs(ls - gl["loss_sum"]) <= LOSS_TOL, f"loss_sum: {ls} vs {gl['loss_sum']}"
    assert abs(float(cp_rows) - gl["cp_rows"]) <= LOSS_TOL, f"cp_rows: {cp_rows} vs {gl['cp_rows']}"
    assert abs(ce - gl["bpm_ce_mean"]) <= LOSS_TOL, f"ce_mean: {ce} vs {gl['bpm_ce_mean']}"
    assert abs(float(metrics["bpm_local_rows"]) - gl["bpm_local_rows"]) <= LOSS_TOL
    return f"OK loss reference (loss_sum={ls:.6f}, ce_mean={ce:.6f})"


# pytest entry points
def test_build_phi():
    check_build_phi(_load_golden())


def test_student_first_token():
    check_student_first_token(_load_golden())


def test_teacher_dist_trie():
    check_teacher_dist_trie(_load_golden())


def test_first_token_partition():
    check_first_token_partition(_load_golden())


def test_core_path_is_torch_free():
    # Sibling test modules import torch at pytest collection time, so check in a
    # clean subprocess: importing the pure core must not pull in torch.
    import subprocess

    code = (
        "import sys; "
        "import slime_plugins.bpm.loss.bpm_phi, slime_plugins.bpm.loss.bpm_position; "
        "assert 'torch' not in sys.modules, 'BPM core imported torch (must be torch-free)'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(_SLIME_ROOT))


def main() -> int:
    g = _load_golden()
    checks = [
        ("build_phi", lambda: check_build_phi(g)),
        ("student_first_token", lambda: check_student_first_token(g)),
        ("teacher_dist_trie", lambda: check_teacher_dist_trie(g)),
        ("first_token_partition", lambda: check_first_token_partition(g)),
    ]
    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"OK   {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    # torch-free proof before any torch import
    if "torch" in sys.modules:
        failed += 1
        print("FAIL core_path_is_torch_free: torch imported by the pure core")
    else:
        print("OK   core_path_is_torch_free")
    print(check_loss_reference_if_torch())
    print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {len(checks) + 1} core checks, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
