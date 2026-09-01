"""Fail-fast validation and default rewrites for BPM command-line arguments.

Parser registration lives in bpm_argument_groups.py. Unsupported configuration
raises with a clear message; nothing is silently ignored.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _resolve_teacher_paths(args) -> None:
    """Resolve the BPM teacher model/tokenizer paths, falling back to --opd-teacher-model-path."""
    bpm_path = getattr(args, "bpm_teacher_model_path", None) or getattr(args, "opd_teacher_model_path", None)
    if bpm_path is None:
        raise ValueError("--bpm-teacher-model-path (or --opd-teacher-model-path) must be set for BPM")
    if getattr(args, "bpm_teacher_model_path", None) is None:
        args.bpm_teacher_model_path = bpm_path
        logger.info("BPM: using opd_teacher_model_path as bpm_teacher_model_path")
    if getattr(args, "bpm_teacher_tokenizer_path", None) is None:
        args.bpm_teacher_tokenizer_path = args.bpm_teacher_model_path
        logger.info("BPM: using bpm_teacher_model_path as teacher tokenizer path")


def _validate_bpm_method(args) -> None:
    """Validate the BPM alignment objective knobs and its coupling constraints."""
    # BPM needs a full-vocab softmax on both sides, so the student cannot be sharded
    if int(getattr(args, "tensor_model_parallel_size", 1)) > 1:
        raise ValueError(
            "--opd-backend bpm requires --tensor-model-parallel-size 1 "
            f"(got {getattr(args, 'tensor_model_parallel_size', 1)}); BPM's full-vocab softmax does "
            "not support student vocab-parallel sharding."
        )
    routes = {r.strip() for r in str(getattr(args, "bpm_routes", "") or "11,n1,1n").split(",") if r.strip()}
    if not routes or not routes.issubset({"11", "n1", "1n"}):
        raise ValueError(
            "--bpm-routes must be a non-empty comma-set drawn from {11,n1,1n}; "
            f"got {sorted(routes)!r}."
        )


_ENTRY_PATHS = {
    "bpm": "slime_plugins.bpm.entry.custom_loss.bpm_custom_loss_function",
    "simct": "slime_plugins.baselines.simct.entry.custom_loss.simct_custom_loss_function",
    "gold": "slime_plugins.baselines.gold.entry.custom_loss.gold_custom_loss_function",
}


def _require_entry_path(args, backend: str) -> None:
    """Reject an entry path that does not belong to ``backend``.

    Every standalone_loss run goes through the same custom_loss slot, so a path pointing at
    another backend's entry would silently train that other objective.
    """
    entry = getattr(args, "custom_loss_function_path", None)
    if entry is not None and entry != _ENTRY_PATHS[backend]:
        raise ValueError(
            f"--opd-backend {backend} requires --custom-loss-function-path "
            f"{_ENTRY_PATHS[backend]}; got {entry!r}."
        )


def _validate_simct_method(args) -> None:
    """Validate the SimCT baseline knobs."""
    if int(getattr(args, "tensor_model_parallel_size", 1)) > 1:
        raise ValueError(
            "--opd-backend simct requires --tensor-model-parallel-size 1 "
            f"(got {getattr(args, 'tensor_model_parallel_size', 1)})."
        )
    mode = getattr(args, "simct_alignment_mode", "span")
    if mode not in ("simple", "span"):
        raise ValueError(f"--simct-alignment-mode must be simple or span; got {mode!r}.")
    _require_entry_path(args, "simct")


def _validate_gold_method(args) -> None:
    """Validate the GOLD / ULD baseline knobs."""
    if int(getattr(args, "tensor_model_parallel_size", 1)) > 1:
        raise ValueError(
            "--opd-backend gold requires --tensor-model-parallel-size 1 "
            f"(got {getattr(args, 'tensor_model_parallel_size', 1)})."
        )
    if getattr(args, "gold_use_hybrid_loss", False):
        for name in ("gold_hybrid_matched_weight", "gold_hybrid_unmatched_weight"):
            if getattr(args, name, None) is None:
                raise ValueError(
                    f"--gold-use-hybrid-loss requires --{name.replace('_', '-')}."
                )
    _require_entry_path(args, "gold")


def validate_bpm_args(args) -> None:
    """Validate BPM flags and perform BPM-only default rewrites on the namespace."""
    if getattr(args, "opd_mode", "off") == "off":
        return

    if getattr(args, "opd_teacher_model_path", None) is None and getattr(args, "bpm_teacher_model_path", None) is None:
        raise ValueError("--opd-teacher-model-path (or --bpm-teacher-model-path) must be set when --opd-mode is not off")

    _tp = int(getattr(args, "opd_teacher_tp_size", 1) or 1)
    _ep = int(getattr(args, "opd_teacher_ep_size", 1) or 1)
    if _ep > _tp or (_ep > 0 and _tp % _ep != 0):
        raise ValueError(
            f"--opd-teacher-ep-size {_ep} is invalid for --opd-teacher-tp-size {_tp}: "
            "expert parallelism must satisfy ep_size <= tp_size and tp_size % ep_size == 0."
        )

    if not getattr(args, "offload_train", False) or not getattr(args, "offload_rollout", False):
        raise ValueError(
            "BPM requires --offload-train and --offload-rollout. The teacher SGLang service is "
            "launched outside Ray's GPU accounting, so without both offloads it co-resides with "
            "the resident rollout/train GPUs and OOMs -- even outside --colocate mode."
        )

    if getattr(args, "recompute_loss_function", False):
        raise NotImplementedError(
            "BPM does not support --recompute-loss-function: the loss accumulates diagnostics on args "
            "during forward, and activation-checkpoint recompute can double-count them."
        )

    _resolve_teacher_paths(args)

    if args.opd_mode == "standalone_loss":
        # standalone BPM is reached through the custom_loss hook
        if getattr(args, "loss_type", "policy_loss") != "custom_loss":
            raise ValueError(
                "--opd-mode standalone_loss reaches the backend through the custom_loss entry; "
                f"set --loss-type custom_loss (got --loss-type {getattr(args, 'loss_type', None)!r}) and "
                f"--custom-loss-function-path {_ENTRY_PATHS[getattr(args, 'opd_backend', 'bpm')]}."
            )
        if getattr(args, "custom_loss_function_path", None) is None:
            raise ValueError(
                "--opd-mode standalone_loss requires --custom-loss-function-path "
                f"{_ENTRY_PATHS[getattr(args, 'opd_backend', 'bpm')]}."
            )
        if float(getattr(args, "entropy_coef", 0.0) or 0.0) != 0.0:
            raise NotImplementedError(
                "--opd-mode standalone_loss ignores --entropy-coef in this release: the custom BPM loss "
                "does not add an entropy bonus, so a non-zero coefficient is silently ineffective. "
                "Set --entropy-coef 0."
            )
    else:
        if getattr(args, "loss_type", "policy_loss") != "policy_loss":
            raise ValueError(
                f"--opd-mode {args.opd_mode} couples BPM through policy_loss; got --loss-type "
                f"{getattr(args, 'loss_type', None)!r}. Use --loss-type policy_loss."
            )

    if getattr(args, "agent_type", None) is not None:
        raise NotImplementedError(
            "BPM with --agent-type is not implemented: the agent batch_len path splits tokens/loss "
            "masks but not teacher_hidden_states/teacher_token_ids. Disable BPM or --agent-type."
        )

    backend = getattr(args, "opd_backend", "bpm")
    if backend == "bpm":
        _validate_bpm_method(args)
    elif backend == "simct":
        _validate_simct_method(args)
    elif backend == "gold":
        _validate_gold_method(args)
    else:
        raise ValueError(f"--opd-backend {backend!r} has no validator.")

    # per_token and per_sample ride calculate_per_token_loss=True; per_rank rides the False path
    reduction = getattr(args, "opd_loss_reduction", "per_token")
    args.calculate_per_token_loss = reduction in ("per_token", "per_sample")
