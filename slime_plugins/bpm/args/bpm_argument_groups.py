"""BPM argparse flag groups. Validation and default rewrites live in
bpm_argument_validation.py; the provider wrapper lives in bpm_arguments.py.

Naming: method knobs are --bpm-*, backend-agnostic OPD infrastructure is --opd-*.
"""

from __future__ import annotations

import argparse

from .bpm_utils import canonical_bpm_backend


_PUBLISHED_BACKENDS = ("bpm", "simct", "gold")


def _parse_bpm_backend(value: str) -> str:
    canonical = canonical_bpm_backend(value)
    if canonical not in _PUBLISHED_BACKENDS:
        raise argparse.ArgumentTypeError(
            f"unknown OPD backend {value!r}; published backends are {_PUBLISHED_BACKENDS}."
        )
    return canonical


def _add_opd_core_arguments(parser: argparse.ArgumentParser) -> None:
    """OPD coupling, divergence, reduction, and diagnostics (backend-agnostic)."""
    parser.add_argument(
        "--opd-mode",
        type=str,
        choices=["off", "standalone_loss", "aux_loss"],
        default="off",
        help=(
            "How distillation is coupled to training. off=normal RL; standalone_loss=pure BPM loss "
            "(reached through the custom_loss entry); aux_loss=policy loss + coef*BPM."
        ),
    )
    parser.add_argument(
        "--opd-backend",
        type=_parse_bpm_backend,
        metavar="{bpm,simct,gold}",
        default="bpm",
        help=(
            "Teacher-data/loss backend used when --opd-mode is not off. 'bpm' is the method; "
            "'simct' and 'gold' are the paper's cross-tokenizer comparison baselines."
        ),
    )
    parser.add_argument(
        "--opd-loss-reduction",
        type=str,
        default="per_token",
        choices=["per_token", "per_rank", "per_sample"],
        help=(
            "KD-loss normalization (resolves args.calculate_per_token_loss). per_token = global "
            "per-response-token mean over the DP+CP batch; per_rank = per-DP-rank pooled mean; "
            "per_sample = per-sample mean then batch mean (each sample weighted 1/N)."
        ),
    )
    parser.add_argument(
        "--opd-diagnostics-mode",
        type=str,
        default="basic",
        choices=["off", "basic", "full"],
        help=(
            "How many diagnostics the BPM loss computes and publishes. off = loss only (cheapest, "
            "skips the entropy diagnostics); basic = the default panel (alignment/coverage/stop/"
            "divergence); full = additionally publishes the route-census and repair counters "
            "(bpm_rows, bpm_fast_rows, bpm_tail_rows, ...). The reference "
            "runs used basic."
        ),
    )


    parser.add_argument(
        "--opd-loss-type",
        type=str,
        default="rkl",
        choices=["rkl", "jsd", "kl", "fkl"],
        help="Divergence used by the simct backend. Ignored by bpm (see --bpm-beta) and gold.",
    )
    parser.add_argument("--opd-temperature", type=float, default=1.0, help="Softmax temperature applied to both sides.")
    parser.add_argument("--opd-ce-weight", type=float, default=0.0, help="Weight on the ground-truth CE term.")
    parser.add_argument("--opd-topk", type=int, default=0, help="Restrict the target to the teacher's top-k; 0 = full vocabulary.")
    parser.add_argument(
        "--opd-distill-scope",
        type=str,
        default="auto",
        choices=["auto", "sample", "topk", "full"],
        help="Which vocabulary slice carries the divergence; auto resolves from --opd-topk.",
    )
    parser.add_argument("--opd-jsd-beta", type=float, default=0.5, help="Interpolation weight for --opd-loss-type jsd.")
    parser.add_argument("--opd-aux-loss-coef", type=float, default=1.0, help="Coefficient on the distillation term when --opd-mode aux_loss.")
    parser.add_argument(
        "--allow-opd-prefix-truncation",
        action="store_true",
        default=False,
        help="Allow the teacher prefill to be truncated instead of dropping the sample.",
    )


def _add_opd_teacher_arguments(parser: argparse.ArgumentParser) -> None:
    """Teacher engine placement / timeout flags (backend-agnostic infrastructure)."""
    parser.add_argument(
        "--opd-teacher-model-path",
        type=str,
        default=None,
        help="HF path for the teacher model served by SGLang for prefill-only hidden-state extraction.",
    )
    parser.add_argument(
        "--opd-teacher-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for the teacher SGLang engine.",
    )
    parser.add_argument(
        "--opd-teacher-mem-fraction",
        type=float,
        default=0.8,
        help="Static memory fraction for the teacher SGLang engine.",
    )
    parser.add_argument(
        "--opd-teacher-dp-size",
        type=int,
        default=1,
        help="Data parallel size for the teacher: number of independent engine replicas.",
    )
    parser.add_argument(
        "--opd-teacher-ep-size",
        type=int,
        default=1,
        help="Expert parallel size for the teacher SGLang engine. Must satisfy ep_size <= tp_size and tp_size %% ep_size == 0.",
    )
    parser.add_argument(
        "--opd-teacher-placement",
        type=str,
        default=None,
        help=(
            "Explicit teacher replica placement, overriding the default round-robin over all "
            "alive Ray nodes. Format: one ';'-separated group per replica (so the group count "
            "must equal --opd-teacher-dp-size), each group being '<node>:<gpu>[,<gpu>...]' with "
            "as many GPU ids as one replica needs (tp_size). <node> may be a Ray NodeID, node IP, "
            "or hostname; leave it empty to mean the node running the training actors. "
            "Example for dp=8/tp=1 pinned to the local node: ':0;:1;:2;:3;:4;:5;:6;:7'. "
            "Required on a Ray cluster shared with other jobs -- the default round-robin will "
            "otherwise scatter teacher replicas onto nodes this job does not own."
        ),
    )
    parser.add_argument(
        "--opd-teacher-prefill-chunk-size",
        type=int,
        default=2,
        help="Max samples per teacher prefill-only generate() chunk per DP replica. 0 disables chunking.",
    )
    parser.add_argument(
        "--opd-teacher-max-prefill-len",
        type=int,
        default=0,
        help=(
            "Max teacher prefill length (prompt+response, teacher tokens) before a sample is dropped "
            "from KD this step instead of crashing the rollout. 0 auto-derives from the teacher config."
        ),
    )
    parser.add_argument(
        "--opd-teacher-generate-rpc-timeout",
        type=float,
        default=900.0,
        help="Optional hard timeout (seconds) waiting on the teacher subprocess generate response. 0 disables.",
    )
    parser.add_argument(
        "--opd-teacher-hidden-recv-timeout",
        type=float,
        default=300.0,
        help="Optional hard timeout (seconds) for receiving teacher hidden-state tensors. 0 disables.",
    )


    parser.add_argument(
        "--opd-teacher-tokenizer-path",
        dest="bpm_teacher_tokenizer_path",
        type=str,
        default=None,
        help="Teacher tokenizer path; falls back to --opd-teacher-model-path.",
    )


def _add_bpm_method_arguments(parser: argparse.ArgumentParser) -> None:
    """BPM objective weights, routing, stop-bridge, and teacher paths."""
    parser.add_argument(
        "--bpm-teacher-model-path",
        type=str,
        default=None,
        help="Path to the teacher model for BPM. Used to load the teacher lm_head and tokenizer (falls back to --opd-teacher-model-path).",
    )
    parser.add_argument(
        "--bpm-teacher-tokenizer-path",
        type=str,
        default=None,
        help="Path to the teacher tokenizer for BPM (falls back to --bpm-teacher-model-path).",
    )
    parser.add_argument(
        "--bpm-alignment-mode",
        type=str,
        choices=["bpm"],
        default="bpm",
        help=(
            "Cross-tokenizer alignment objective. 'bpm' (default, only published mode): byte-prefix "
            "marginalization of the teacher's full next-token distribution onto the student vocab, "
            "with chain-exact correction for coarser student tokens and tail positions. Requires TP=1."
        ),
    )
    parser.add_argument(
        "--bpm-beta",
        type=float,
        default=0.5,
        help=(
            "Single divergence axis over BPM's byte-marginal target vs the student softmax (TRL GOLD "
            "convention): 0.0 = forward-KL (mass-covering, exact fast path); 0.5 (default) = JSD; "
            "1.0 = reverse-KL. Forced-delta rows always train CE."
        ),
    )
    parser.add_argument(
        "--bpm-rkl-lambda",
        type=float,
        default=0.0,
        help=(
            "Skew floor for the reverse-KL endpoint (--bpm-beta >= 1); ignored otherwise. 0.0 = pure "
            "reverse-KL with q floored to eps; >0 = skew reverse-KL toward a smoother moving target."
        ),
    )
    parser.add_argument(
        "--bpm-proj-chunk",
        type=int,
        default=65536,
        help=(
            "Row block size for the teacher lm_head projection + token-marginal CE over boundary rows "
            "(primary throughput/memory knob). Raise to fill a large GPU; lower only if BPM OOMs."
        ),
    )
    parser.add_argument(
        "--bpm-chain-mode",
        type=str,
        default="scatter",
        choices=["fast", "scatter", "bytewalk"],
        help=(
            "How to handle 1:N chain rows (a student token spanning past the realized teacher token). "
            "'fast' = vectorized phi gather; 'scatter' = fast + the realized-path first-token correction; "
            "'bytewalk' = the exact pure-Python chain byte-walk on spanning rows (slow)."
        ),
    )
    parser.add_argument(
        "--bpm-routes",
        type=str,
        default="11,n1,1n",
        help=(
            "Route-ablation mask: comma-set of alignment classes to train, drawn from {11,n1,1n}. "
            "'11' = byte-exact 1:1 fast rows; 'n1' = N:1 rows; '1n' = spanning 1:N/M:N rows. "
            "Stop rows are always trained. Default trains all."
        ),
    )
    parser.add_argument(
        "--bpm-stop-bridge-mode",
        type=str,
        default="bridge",
        choices=["bridge", "floor", "skip"],
        help=(
            "How the explicit student stop row is supervised. 'bridge' (default) uses the teacher's "
            "stopping probability at the aligned row; 'floor' floors it on genuinely-stopped samples; "
            "'skip' excludes the stop row from the loss."
        ),
    )
    parser.add_argument(
        "--bpm-joined-prefill",
        action="store_true",
        default=False,
        help=(
            "Tokenize the joined prompt+response once and derive the teacher boundary by "
            "subtraction, mirroring the reference cross-tokenizer trainer. Off by default: the "
            "shipped path tokenizes prompt and response separately."
        ),
    )


def _add_simct_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    """SimCT baseline (--opd-backend simct): overlap-based candidate-set distillation."""
    parser.add_argument(
        "--simct-alignment-mode",
        type=str,
        default="span",
        choices=["simple", "span"],
        help="span = span-aligned shared-surface vocabulary (the paper arm); simple = position-wise overlap.",
    )
    parser.add_argument("--simct-span-ctkd-norm", action="store_true", default=False, help="Normalize the span-aligned target as in span-CTKD.")
    parser.add_argument("--simct-chunk-size", type=int, default=64, help="Row block size for the SimCT projection.")
    parser.add_argument("--simct-compile-bucket-size", type=int, default=0, help="torch.compile bucket size; 0 disables bucketing.")
    parser.add_argument("--simct-overlap-chunk-size", type=int, default=0, help="Chunk size for building the shared-surface overlap; 0 = one shot.")
    parser.add_argument("--simct-min-align-ratio", type=float, default=0.0, help="Drop a sample from the loss below this alignment ratio.")
    parser.add_argument("--simct-train-log-interval", type=int, default=0, help="Steps between SimCT diagnostic logs; 0 disables.")
    parser.add_argument("--simct-skip-eos", action="store_true", default=False, help="Exclude the EOS row from the loss.")
    parser.add_argument("--simct-stop-token-bridge", action="store_true", default=False, help="Isolate the terminal EOS row; off in the paper arm.")


def _add_gold_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    """GOLD / ULD baselines (--opd-backend gold): rank-matched transport losses."""
    parser.add_argument("--gold-use-extended-uld", action="store_true", default=True, help="TRL-extended ULD (the published GOLD default).")
    parser.add_argument("--disable-gold-extended-uld", dest="gold_use_extended_uld", action="store_false", help="Classic ULD sorted-L1 only.")
    parser.add_argument("--gold-use-hybrid-loss", action="store_true", default=False, help="Hybrid GOLD: exact match on surface-shared tokens, sorted loss elsewhere.")
    parser.add_argument("--gold-beta", type=float, default=0.5, help="Generalized Jensen-Shannon interpolation weight.")
    parser.add_argument("--gold-hybrid-matched-weight", type=float, default=None, help="Weight on the matched-token term when hybrid is on.")
    parser.add_argument("--gold-hybrid-unmatched-weight", type=float, default=None, help="Weight on the sorted term when hybrid is on.")
    parser.add_argument("--gold-ce-weight", type=float, default=0.0, help="Weight on the ground-truth CE term.")
    parser.add_argument("--gold-distillation-weight", type=float, default=1.0, help="Weight on the distillation term.")
    parser.add_argument("--gold-student-temperature", type=float, default=1.0, help="Student softmax temperature.")
    parser.add_argument("--gold-teacher-temperature", type=float, default=1.0, help="Teacher softmax temperature.")
    parser.add_argument("--gold-skip-student-eos", action="store_true", default=True, help="Exclude the student EOS row.")
    parser.add_argument("--disable-gold-skip-student-eos", dest="gold_skip_student_eos", action="store_false", help="Keep the student EOS row.")
    parser.add_argument("--gold-skip-teacher-eos", action="store_true", default=True, help="Exclude the teacher EOS row.")
    parser.add_argument("--disable-gold-skip-teacher-eos", dest="gold_skip_teacher_eos", action="store_false", help="Keep the teacher EOS row.")
    parser.add_argument("--gold-chunk-size", type=int, default=32, help="Row block size for the sorted-L1 kernel.")
    parser.add_argument("--gold-trl-faithful", action="store_true", default=False, help="Reproduce the reference trainer's normalization exactly.")
    parser.add_argument(
        "--gold-uld-token-merge-strategy",
        type=str,
        default="observed",
        choices=["observed", "bayesian"],
        help="How multi-token surface forms are merged before sorting.",
    )
    parser.add_argument("--gold-min-align-ratio", type=float, default=0.0, help="Drop a sample from the loss below this alignment ratio.")


def add_bpm_argument_groups(parser: argparse.ArgumentParser) -> None:
    """Register all BPM-specific CLI flags in grouped, reviewable sections."""
    _add_opd_core_arguments(parser)
    _add_opd_teacher_arguments(parser)
    _add_bpm_method_arguments(parser)
    _add_simct_baseline_arguments(parser)
    _add_gold_baseline_arguments(parser)
