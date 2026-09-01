"""GPU-free wiring test for the SimCT and GOLD baselines.

Asserts each backend resolves to its own entry, that a cross-wired entry path is
rejected, and that every published arm launcher emits the flags its arm is defined by.
Run: python3 slime_plugins/baselines/test_baseline_args.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slime_plugins.bpm.args.bpm_arguments import add_bpm_arguments  # noqa: E402
from slime_plugins.bpm.args.bpm_argument_validation import (  # noqa: E402
    _ENTRY_PATHS,
    _require_entry_path,
)

ARM_CONTRACT = {
    "simct": (
        ["--opd-backend", "simct", "--opd-loss-reduction", "per_token",
         "--simct-alignment-mode", "span", "--simct-span-ctkd-norm", _ENTRY_PATHS["simct"]],
        ["--gold-use-hybrid-loss", "--bpm-beta"],
    ),
    "uld": (
        ["--opd-backend", "gold", "--opd-loss-reduction", "per_sample",
         "--gold-trl-faithful", "--gold-use-extended-uld", _ENTRY_PATHS["gold"]],
        ["--gold-use-hybrid-loss", "--gold-hybrid-matched-weight"],
    ),
    "gold": (
        ["--gold-use-hybrid-loss", "--gold-beta", "0.5"],
        ["--gold-hybrid-matched-weight", "--gold-hybrid-unmatched-weight"],
    ),
    "gold_matched": (["--gold-hybrid-matched-weight", "1.0", "--gold-hybrid-unmatched-weight", "0.0"], []),
    "gold_unmatched": (["--gold-hybrid-matched-weight", "0.0", "--gold-hybrid-unmatched-weight", "1.0"], []),
}

BASELINE_DEFAULTS = {
    "simct_alignment_mode": "span",
    "simct_span_ctkd_norm": False,
    "simct_chunk_size": 64,
    "gold_use_extended_uld": True,
    "gold_use_hybrid_loss": False,
    "gold_beta": 0.5,
    "gold_chunk_size": 32,
    "gold_skip_student_eos": True,
    "gold_skip_teacher_eos": True,
    "gold_trl_faithful": False,
    "gold_hybrid_matched_weight": None,
    "gold_hybrid_unmatched_weight": None,
}


def test_baseline_defaults() -> int:
    defaults = vars(add_bpm_arguments(argparse.ArgumentParser()).parse_args([]))
    for key, want in BASELINE_DEFAULTS.items():
        assert key in defaults, f"{key} is not registered"
        assert defaults[key] == want, f"{key}: {defaults[key]!r} != {want!r}"
    return len(BASELINE_DEFAULTS)


def test_entry_path_gate() -> bool:
    for backend in ("simct", "gold"):
        _require_entry_path(argparse.Namespace(custom_loss_function_path=_ENTRY_PATHS[backend]), backend)
        _require_entry_path(argparse.Namespace(custom_loss_function_path=None), backend)
        for other in set(_ENTRY_PATHS) - {backend}:
            try:
                _require_entry_path(
                    argparse.Namespace(custom_loss_function_path=_ENTRY_PATHS[other]), backend
                )
            except ValueError:
                continue
            raise AssertionError(f"backend={backend} accepted the {other} entry path")
    return True


def _dry_run(arm: str) -> list[str]:
    env = dict(
        os.environ,
        DRY_RUN="1",
        BASELINE=arm,
        STUDENT_MODEL_PATH="/x",
        STUDENT_REF_LOAD="/y",
        TEACHER_MODEL_PATH="/z",
        DATA_PATH="/d",
    )
    out = subprocess.run(
        ["bash", os.path.join(_REPO_ROOT, "examples/baselines/reproduce/run_p2_glm_z1_9b.sh")],
        cwd=_REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    return out.stdout.split()


def test_arm_launchers_emit_their_contract() -> int:
    for arm, (required, forbidden) in ARM_CONTRACT.items():
        args = _dry_run(arm)
        for flag in required:
            assert flag in args, f"{arm}: missing {flag}"
        for flag in forbidden:
            assert flag not in args, f"{arm}: unexpected {flag}"
    return len(ARM_CONTRACT)


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in [
        ("baseline_defaults", test_baseline_defaults),
        ("entry_path_gate", test_entry_path_gate),
        ("arm_launchers_emit_their_contract", test_arm_launchers_emit_their_contract),
    ]:
        try:
            print(f"PASS {name} -> {fn()}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"FAIL {name} -> {exc!r}")
            failed += 1
    print(f"RESULT: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
