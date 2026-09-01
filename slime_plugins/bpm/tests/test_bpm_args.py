"""GPU-free arg-surface test for the BPM argument package.

Asserts every registered key carries its expected default and that validate_bpm_args
raises on the documented invalid combinations.
Run: python3 slime_plugins/bpm/tests/test_bpm_args.py
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slime_plugins.bpm.args.bpm_arguments import add_bpm_arguments  # noqa: E402
from slime_plugins.bpm.args.bpm_argument_validation import validate_bpm_args  # noqa: E402


# every registered key and its authoritative default
EXPECTED_DEFAULTS = {
    # backend-agnostic OPD infrastructure (--opd-*)
    "opd_mode": "off",
    "opd_backend": "bpm",
    "opd_loss_reduction": "per_token",
    "opd_diagnostics_mode": "basic",
    # teacher engine placement (--opd-teacher-*)
    "opd_teacher_model_path": None,
    "opd_teacher_tp_size": 1,
    "opd_teacher_mem_fraction": 0.8,
    "opd_teacher_dp_size": 1,
    "opd_teacher_ep_size": 1,
    "opd_teacher_placement": None,
    "opd_teacher_prefill_chunk_size": 2,
    "opd_teacher_max_prefill_len": 0,
    "opd_teacher_generate_rpc_timeout": 900.0,
    "opd_teacher_hidden_recv_timeout": 300.0,
    # BPM method knobs (--bpm-*)
    "bpm_teacher_model_path": None,
    "bpm_teacher_tokenizer_path": None,
    "bpm_alignment_mode": "bpm",
    "bpm_beta": 0.5,
    "bpm_rkl_lambda": 0.0,
    "bpm_proj_chunk": 65536,
    "bpm_chain_mode": "scatter",
    "bpm_routes": "11,n1,1n",
    "bpm_stop_bridge_mode": "bridge",
    "bpm_joined_prefill": False,
    # baseline-shared divergence knobs (--opd-*, consumed by the simct backend)
    "opd_loss_type": "rkl",
    "opd_temperature": 1.0,
    "opd_ce_weight": 0.0,
    "opd_topk": 0,
    "opd_distill_scope": "auto",
    "opd_jsd_beta": 0.5,
    "opd_aux_loss_coef": 1.0,
    "allow_opd_prefix_truncation": False,
    # SimCT baseline (--simct-*)
    "simct_alignment_mode": "span",
    "simct_span_ctkd_norm": False,
    "simct_chunk_size": 64,
    "simct_compile_bucket_size": 0,
    "simct_overlap_chunk_size": 0,
    "simct_min_align_ratio": 0.0,
    "simct_train_log_interval": 0,
    "simct_skip_eos": False,
    "simct_stop_token_bridge": False,
    # GOLD / ULD baselines (--gold-*)
    "gold_use_extended_uld": True,
    "gold_use_hybrid_loss": False,
    "gold_beta": 0.5,
    "gold_hybrid_matched_weight": None,
    "gold_hybrid_unmatched_weight": None,
    "gold_ce_weight": 0.0,
    "gold_distillation_weight": 1.0,
    "gold_student_temperature": 1.0,
    "gold_teacher_temperature": 1.0,
    "gold_skip_student_eos": True,
    "gold_skip_teacher_eos": True,
    "gold_chunk_size": 32,
    "gold_trl_faithful": False,
    "gold_uld_token_merge_strategy": "observed",
    "gold_min_align_ratio": 0.0,
}


def _parse(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_bpm_arguments(parser)
    return parser.parse_args(argv)


def _upstream_defaults():
    """The handful of upstream slime args that validate_bpm_args reads via getattr."""
    return {
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "advantage_estimator": "grpo",
        "n_samples_per_prompt": 8,
        "entropy_coef": 0.0,
        "use_kl_loss": False,
        "kl_loss_coef": 0.0,
        "recompute_loss_function": False,
        "agent_type": None,
        "hf_checkpoint": None,
        "loss_type": "custom_loss",
        "custom_loss_function_path": "slime_plugins.bpm.entry.custom_loss.bpm_custom_loss_function",
        "loss_mask_type": "qwen",
        "seed": 1234,
        # BPM requires both offloads (the teacher engine is launched outside Ray's
        # GPU accounting), so a valid enabled namespace always has them on.
        "offload_train": True,
        "offload_rollout": True,
    }


def _valid_enabled_args():
    """A parsed, valid, BPM-enabled namespace (standalone_loss) with upstream attrs filled in."""
    args = _parse(
        [
            "--opd-mode", "standalone_loss",
            "--opd-backend", "bpm",
            "--bpm-teacher-model-path", "/tmp/teacher",
            "--bpm-alignment-mode", "bpm",
        ]
    )
    for k, v in _upstream_defaults().items():
        setattr(args, k, v)
    return args


def test_defaults_present_and_correct():
    args = _parse([])
    ns = vars(args)
    for key, expected in EXPECTED_DEFAULTS.items():
        assert key in ns, f"missing registered key: {key}"
        got = ns[key]
        assert got == expected and type(got) is type(expected), (
            f"default mismatch for {key}: got {got!r} ({type(got).__name__}), "
            f"expected {expected!r} ({type(expected).__name__})"
        )
    # bidirectional: a new flag nobody pinned here fails instead of drifting in
    unpinned = set(ns) - set(EXPECTED_DEFAULTS)
    assert not unpinned, f"registered but unpinned in EXPECTED_DEFAULTS: {sorted(unpinned)}"
    return len(EXPECTED_DEFAULTS)


def test_representative_cli_captures_values():
    args = _parse(
        [
            "--opd-mode", "standalone_loss",
            "--opd-backend", "bpm",
            "--opd-loss-reduction", "per_rank",
            "--bpm-teacher-model-path", "/tmp/teacher",
            "--bpm-teacher-tokenizer-path", "/tmp/teacher_tok",
            "--bpm-beta", "1.0",
            "--bpm-routes", "11,n1",
            "--bpm-chain-mode", "bytewalk",
            "--bpm-stop-bridge-mode", "skip",
        ]
    )
    assert args.opd_mode == "standalone_loss"
    assert args.opd_backend == "bpm"
    assert args.opd_loss_reduction == "per_rank"
    assert args.bpm_teacher_model_path == "/tmp/teacher"
    assert args.bpm_teacher_tokenizer_path == "/tmp/teacher_tok"
    assert args.bpm_beta == 1.0
    assert args.bpm_routes == "11,n1"
    assert args.bpm_chain_mode == "bytewalk"
    assert args.bpm_stop_bridge_mode == "skip"
    return True


def test_backend_value_rejects_non_bpm():
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            _parse(["--opd-backend", "not_a_backend"])
    except SystemExit:
        # argparse converts ArgumentTypeError into a SystemExit(2).
        return True
    raise AssertionError("--opd-backend not_a_backend should have been rejected")


def test_valid_enabled_args_pass():
    args = _valid_enabled_args()
    validate_bpm_args(args)  # must not raise
    # per_token reduction resolves calculate_per_token_loss=True.
    assert args.calculate_per_token_loss is True
    # teacher paths resolved / mirrored.
    assert args.bpm_teacher_tokenizer_path == "/tmp/teacher"
    return True


def test_off_mode_is_noop():
    args = _parse([])
    validate_bpm_args(args)  # opd_mode off -> immediate return, no teacher path required
    return True


def _assert_raises(mutate, needle):
    args = _valid_enabled_args()
    mutate(args)
    try:
        validate_bpm_args(args)
    except (ValueError, NotImplementedError) as exc:
        assert needle in str(exc), f"raised but message missing {needle!r}: {exc}"
        return True
    raise AssertionError(f"expected a raise containing {needle!r}")


def test_per_sample_reduction_accepted():
    args = _valid_enabled_args()
    args.opd_loss_reduction = "per_sample"
    validate_bpm_args(args)
    assert args.calculate_per_token_loss is True, args.calculate_per_token_loss
    return args.calculate_per_token_loss


def test_invalid_combos_raise():
    # Documented combo 1: BPM requires student TP=1.
    _assert_raises(lambda a: setattr(a, "tensor_model_parallel_size", 2), "tensor-model-parallel-size 1")
    # Documented combo 2: routes must be a subset of {11,n1,1n}.
    _assert_raises(lambda a: setattr(a, "bpm_routes", "11,zz"), "--bpm-routes")
    # Enable guard: teacher path required when mode != off.
    def _no_teacher(a):
        a.opd_teacher_model_path = None
        a.bpm_teacher_model_path = None
    _assert_raises(_no_teacher, "teacher-model-path")
    # standalone_loss ignores entropy_coef, so a non-zero value is rejected
    _assert_raises(lambda a: setattr(a, "entropy_coef", 0.01), "entropy-coef")
    # standalone_loss must be wired through the custom_loss entry.
    _assert_raises(lambda a: setattr(a, "loss_type", "policy_loss"), "custom_loss")
    return True


if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        ("defaults_present_and_correct", test_defaults_present_and_correct),
        ("representative_cli_captures_values", test_representative_cli_captures_values),
        ("backend_value_rejects_non_bpm", test_backend_value_rejects_non_bpm),
        ("valid_enabled_args_pass", test_valid_enabled_args_pass),
        ("off_mode_is_noop", test_off_mode_is_noop),
        ("invalid_combos_raise", test_invalid_combos_raise),
        ("per_sample_reduction_accepted", test_per_sample_reduction_accepted),
    ]
    for name, fn in tests:
        try:
            out = fn()
            print(f"PASS {name} -> {out}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"FAIL {name} -> {exc!r}")
            failed += 1
    print(f"RESULT: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
