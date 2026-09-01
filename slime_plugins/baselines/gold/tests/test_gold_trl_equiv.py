"""EQUIVALENCE HARNESS  --  TRL GOLD/ULD  vs  ours gold, identical inputs.

Purpose
-------
Prove numerically (not by argument) that, under the exact flag sets used by the
paper's three GOLD/ULD baseline arms
(scripts/opd/paper/p1-glm47flash-qwen35-2b/{run-uld-trl,run-gold-matched,run-gold-unmatched}.sh),
ours's REAL loss kernels compute THE SAME loss TRL would on identical inputs --
or localize every divergence to file:line on both sides with magnitude.

Both sides are invoked on IDENTICAL inputs:

  (a) TRL GOLD ORACLE  -- tests/fidelity/trl_gold_oracle.py, which VERBATIM-copies
      (with file:line citations) TRL ULDLoss / generalized_jsd_loss (:2398) /
      build_teacher_inputs_from_texts / byte-offset utils from
      trl/experimental/gold/gold_trainer.py (+ utils.py).
      We call the REAL TRL classes (ULDLoss.__call__ etc.), not a reimpl.

  (b) ours REAL kernels -- the ACTUAL functions from
      slime/backends/megatron_utils/opd/gold_utils.py, loaded by importlib
      (the package __init__ drags megatron/vllm, so we bypass it and satisfy the
      single ``.loss_utils`` relative import with a verbatim stub of
      _mask_invalid_vocab_rows_, gold_utils.py:413-432 -- a no-op at TP=1).
      These functions are orchestrated EXACTLY as gold_loss.py
      (gold_core_loss_function) orchestrates them at TP=1/CP=1, honoring
      every paper flag via an argparse.Namespace built from the arm scripts +
      _launcher_gold.sh OPD_ARGS + slime/utils/opd/argument_groups.py defaults.

The megatron TP/CP wrappers in gold_loss.py cannot run on a CPU box, but at
TP=1/CP=1 they are provable no-ops (tp_size<=1 short-circuits the all-gather:
gold_utils.py:502-505; cp_size==1 takes the direct-slice window branch:
gold_loss.py:133-137 observed / loss_helpers.get_responses:286-290 shifted), and
EVERY per-group number the backend produces flows through the pure gold_utils
kernels we call here.  The harness replicates only that pure-python windowing +
per-sample accumulation from gold_loss.py, citing the exact source lines.

PAPER ARM FLAGS (common: --gold-distillation-weight 1.0 --gold-ce-weight 0.0
--gold-student-temperature 1.0 --gold-teacher-temperature 1.0
--gold-use-extended-uld --gold-skip-student-eos --gold-skip-teacher-eos
--gold-chunk-size 32 --gold-beta 0.5; POST-FIDELITY-FIX: all arms also pass
--gold-trl-faithful --gold-uld-token-merge-strategy observed and the launchers pass
--opd-loss-reduction per_sample (TRL batch normalizer)):
  uld-trl       : (no hybrid)
  gold-matched  : --gold-use-hybrid-loss --gold-hybrid-matched-weight 1.0 --gold-hybrid-unmatched-weight 0.0
  gold-unmatched: --gold-use-hybrid-loss --gold-hybrid-matched-weight 0.0 --gold-hybrid-unmatched-weight 1.0
  gold-hybrid   : --gold-use-hybrid-loss (NO matched/unmatched weight flags) -> ADAPTIVE weighting.
                  Both sides resolve the SAME weights from the SAME vocab split:
                    matched_weight   = matched_count / teacher_vocab_size
                    unmatched_weight = 1 - matched_weight
                  ours  gold_utils.py:761-765  (matched_count=:678, teacher_vocab_size=:677)
                  TRL  gold_trainer.py:934-936 (extended-hybrid streaming path; denom =
                       teacher_answer_logits.size(-1); at Vt_dim==Vt this == teacher real vocab).

Run:  python3 test_gold_trl_equiv.py

CERTIFIED MAXIMA  (re-recorded 2026-07-03; DETERMINISTIC -- identical across re-runs
after seeding random/torch at import + FIXED per-arm seed map _ARM_SEED_OFFSET;
mixed criterion: PASS if max_rel<1e-5 OR max_abs<1e-6):
  A synthetic uld-trl      max_rel = 4.39e-06  max_abs = 2.38e-07  PASS
  A synthetic gold-matched max_rel = 1.82e-06  max_abs = 8.94e-08  PASS
  A synthetic gold-unmatched max_rel = 3.60e-07 max_abs = 2.38e-07 PASS
  A synthetic gold-hybrid  max_rel = 1.98e-03  max_abs = 1.79e-07  PASS  (ADAPTIVE weighting;
      as-config max_rel is a NEAR-ZERO-LOSS artifact: one case has loss ~5.9e-5 with adaptive
      matched_weight 1/43 -> |diff| only 1.16e-7 [same float32 merge noise as all arms]
      amplified by the tiny denom; max_rel restricted to loss>1e-3 is 4.21e-07.)
  PADDED-VOCAB mask-path   max_rel = 8.66e-08   PASS  (width>real vocab, pad cols +-inf/NaN)
  BATCH>=2 per_sample      max_rel = 0.00e+00   PASS  (sum_i(sd_i/aligned_i)/N == TRL mean)
  B real-tokenizer uld-trl max_rel = 7.14e-07   PASS
  D1 NEG-CTRL per_token    rel     = 9.72e-02   EXPECTED-DIVERGE (per_token vs TRL per_sample)
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import types

import torch
import torch.nn.functional as F

# DETERMINISM: seed both RNGs at MODULE IMPORT so every run (and the edge harness
# that imports this module) starts from an identical state, independent of
# PYTHONHASHSEED / import order.  Re-running the harness 3x yields identical maxima.
random.seed(20260703)
torch.manual_seed(20260703)

# FIXED per-arm seed-offset map (replaces the non-deterministic ``hash(arm)`` that
# varied run-to-run under hash randomization).  Distinct small integers keep each
# arm's synthetic stream well separated and reproducible.
_ARM_SEED_OFFSET = {"uld-trl": 101, "gold-matched": 202, "gold-unmatched": 303, "gold-hybrid": 404}

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trl_gold_oracle as O  # noqa: E402  the VERBATIM TRL copy (self-contained)

# --------------------------------------------------------------------------- #
#  Load the REAL gold_kernels.py (bypassing the megatron-heavy package init). #
# --------------------------------------------------------------------------- #
_GOLD_PKG_DIR = os.path.abspath(os.path.join(HERE, ".."))


def _load_real_gold_utils():
    pkg = "ours_gold_real"
    parent = types.ModuleType(pkg)
    parent.__path__ = [_GOLD_PKG_DIR]
    sys.modules[pkg] = parent

    # Verbatim stub of loss_utils._mask_invalid_vocab_rows_ (gold_kernels' only
    # intra-package import).  Copied verbatim from slime_plugins/bpm/loss/bpm_loss_utils.py.
    # At TP=1 vocab_start=0 and real_vocab_size==shard_vocab -> `invalid` is all
    # False -> this is an identity passthrough (no clone, no masking).
    lu = types.ModuleType(pkg + ".bpm_loss_utils")

    def _mask_invalid_vocab_rows_(logits_shard, *, real_vocab_size, vocab_start):
        shard_vocab = logits_shard.shape[-1]
        local_ids = torch.arange(shard_vocab, device=logits_shard.device)
        global_ids = local_ids + int(vocab_start)
        invalid = global_ids >= int(real_vocab_size)
        if bool(invalid.any().item()):
            logits_shard = logits_shard.clone()
            logits_shard[..., invalid] = -torch.finfo(logits_shard.dtype).max
        return logits_shard

    lu._mask_invalid_vocab_rows_ = _mask_invalid_vocab_rows_
    sys.modules[pkg + ".bpm_loss_utils"] = lu

    spec = importlib.util.spec_from_file_location(
        pkg + ".gold_kernels", os.path.join(_GOLD_PKG_DIR, "gold_kernels.py")
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[pkg + ".gold_kernels"] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load_real_gold_utils()  # REAL ours kernels live here


# --------------------------------------------------------------------------- #
#  Toy tokenizer with get_vocab() (for hybrid matched/unmatched vocab split).   #
# --------------------------------------------------------------------------- #
class ToyTok:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)

    def get_vocab(self):
        return dict(self._vocab)


# --------------------------------------------------------------------------- #
#  Arm -> argparse.Namespace, built from the arm scripts + launcher OPD_ARGS.   #
# --------------------------------------------------------------------------- #
def make_arm_args(arm: str) -> argparse.Namespace:
    """Namespace mirroring the flags each paper arm actually passes.

    Common OPD_ARGS (_launcher_gold.sh) + defaults from argument_groups.py.
    POST-FIDELITY-FIX: all three arms export the faithful pair and the launchers
    pass --opd-loss-reduction per_sample (see comments below).
    """
    ns = argparse.Namespace(
        opd_backend="gold",
        gold_distillation_weight=1.0,   # --gold-distillation-weight 1.0
        gold_ce_weight=0.0,             # --gold-ce-weight 0.0
        gold_student_temperature=1.0,   # --gold-student-temperature 1.0
        gold_teacher_temperature=1.0,   # --gold-teacher-temperature 1.0
        gold_use_extended_uld=True,     # --gold-use-extended-uld
        gold_skip_student_eos=True,     # --gold-skip-student-eos
        gold_skip_teacher_eos=True,     # --gold-skip-teacher-eos
        gold_chunk_size=32,             # --gold-chunk-size 32
        gold_beta=0.5,                  # --gold-beta 0.5
        # POST-FIDELITY-FIX (2026-07): all gold launchers now pass --opd-loss-reduction per_sample
        # (TRL GOLDTrainer batch normalizer; was per_token before the fix -> D1 divergence).
        opd_loss_reduction="per_sample",
        # hybrid / faithful flags set per arm below:
        gold_use_hybrid_loss=False,
        gold_hybrid_matched_weight=None,
        gold_hybrid_unmatched_weight=None,
        # POST-FIDELITY-FIX: ALL THREE arms now export GOLD_TRL_FAITHFUL=1 + observed merge
        # (the hybrid arms previously lacked it -> shifted windowing, 23-100% divergence).
        gold_trl_faithful=True,
        gold_uld_token_merge_strategy="observed",
    )
    if arm == "uld-trl":
        pass                                           # faithful pair already set above
    elif arm == "gold-matched":
        ns.gold_use_hybrid_loss = True                 # --gold-use-hybrid-loss
        ns.gold_hybrid_matched_weight = 1.0            # --gold-hybrid-matched-weight 1.0
        ns.gold_hybrid_unmatched_weight = 0.0          # --gold-hybrid-unmatched-weight 0.0
    elif arm == "gold-unmatched":
        ns.gold_use_hybrid_loss = True
        ns.gold_hybrid_matched_weight = 0.0            # --gold-hybrid-matched-weight 0.0
        ns.gold_hybrid_unmatched_weight = 1.0          # --gold-hybrid-unmatched-weight 1.0
    elif arm == "gold-hybrid":
        # TRL-default ADAPTIVE weighting: --gold-use-hybrid-loss with NO weight flags.
        # matched/unmatched weights stay None -> gold_utils.py:761-765 resolves
        # matched_weight = matched_count/teacher_vocab_size, unmatched = 1 - that
        # from the SAME hybrid_vocab split TRL derives on the same tokenizers.
        ns.gold_use_hybrid_loss = True                 # --gold-use-hybrid-loss
        ns.gold_hybrid_matched_weight = None           # (no flag) -> adaptive
        ns.gold_hybrid_unmatched_weight = None         # (no flag) -> adaptive
    else:
        raise ValueError(arm)
    return ns


def mock_cfg_for_arm(arm: str) -> O.MockGOLDConfig:
    """TRL-side config carrying the same per-arm values."""
    if arm == "uld-trl":
        return O.MockGOLDConfig(uld_use_hybrid_loss=False)
    if arm == "gold-matched":
        return O.MockGOLDConfig(
            uld_use_hybrid_loss=True, beta=0.5,
            uld_hybrid_matched_weight=1.0, uld_hybrid_unmatched_weight=0.0,
            uld_hybrid_matched_chunk_size=32, uld_hybrid_unmatched_chunk_size=32,
        )
    if arm == "gold-unmatched":
        return O.MockGOLDConfig(
            uld_use_hybrid_loss=True, beta=0.5,
            uld_hybrid_matched_weight=0.0, uld_hybrid_unmatched_weight=1.0,
            uld_hybrid_matched_chunk_size=32, uld_hybrid_unmatched_chunk_size=32,
        )
    if arm == "gold-hybrid":
        # ADAPTIVE: both hybrid weights None -> TRL resolves them from the vocab
        # overlap in _compute_extended_hybrid_uld_loss_streaming (gold_trainer.py:934-936),
        # matching gold_kernels.py:761-765 on the SAME tokenizer split.
        return O.MockGOLDConfig(
            uld_use_hybrid_loss=True, beta=0.5,
            uld_hybrid_matched_weight=None, uld_hybrid_unmatched_weight=None,
            uld_hybrid_matched_chunk_size=32, uld_hybrid_unmatched_chunk_size=32,
        )
    raise ValueError(arm)


# --------------------------------------------------------------------------- #
#  ours per-sample GOLD/ULD scalar, using the REAL gold_utils kernels,           #
#  orchestrated as gold_loss.py does at TP=1/CP=1.                              #
# --------------------------------------------------------------------------- #
def ours_gold_sample_scalar(
    args: argparse.Namespace,
    *,
    stu_answer_logits: torch.Tensor,   # [R, Vs]  student distribution rows for the R answer tokens
    tea_answer_logits: torch.Tensor,   # [Rt, Vt] teacher distribution rows for the Rt answer tokens
    stu_ids: list[int],                # R student answer token ids
    tea_ids: list[int],                # Rt teacher answer token ids
    stu_offsets: list[tuple[int, int]],
    tea_offsets: list[tuple[int, int]],
    hybrid_vocab=None,
    return_groups: bool = False,
    real_vs: int | None = None,
    real_vt: int | None = None,
):
    """Reproduce gold_loss.py:414-625 per-sample math at TP=1/CP=1 via REAL kernels.

    The caller has already applied the arm's WINDOWING (observed=unshift vs
    non-faithful=next-token shift, gold_loss.py:116-166) to produce
    stu_answer_logits / tea_answer_logits, and EOS trimming (skip_*_eos ->
    _gold_trim_ids_and_offsets_for_answer:238-239 -> ids[:-1]) to produce ids +
    offsets.  This function performs alignment + merge + kernel exactly as the
    backend does.  Returns distill_weight * (per-group loss).sum()/num_groups,
    i.e. the per-SAMPLE distillation scalar (== gold_loss distill_mean at batch=1,
    == TRL _compute_distillation_loss:494 per-sample-mean at batch=1).
    """
    # real_vs/real_vt override the softmax support so a PADDED tensor (width >
    # real vocab, pad columns poisoned) exercises ours's _mask_invalid_vocab_rows_
    # mask path; default = full tensor width (no masking).
    Vs = int(real_vs) if real_vs is not None else int(stu_answer_logits.shape[-1])
    Vt = int(real_vt) if real_vt is not None else int(tea_answer_logits.shape[-1])
    stu_temp = float(args.gold_student_temperature)
    tea_temp = float(args.gold_teacher_temperature)
    distill_weight = float(args.gold_distillation_weight)

    gold_trl_faithful = bool(args.gold_trl_faithful) and args.opd_backend == "gold"
    clamp = 1e-8 if gold_trl_faithful else None                                   # gold_loss.py:86
    strat = str(args.gold_uld_token_merge_strategy or "observed")
    bayesian = gold_trl_faithful and strat == "bayesian"                          # gold_loss.py:88

    # --- alignment (gold_loss.py:422-425) via the REAL kernel ---
    stu_groups, tea_groups = G._gold_align_by_byte_offsets(stu_offsets, tea_offsets)
    paired = [(sg, tg) for sg, tg in zip(stu_groups, tea_groups, strict=False) if sg and tg]
    stu_groups = [sg for sg, _ in paired]
    tea_groups = [tg for _, tg in paired]
    if not stu_groups:
        z = torch.zeros((), dtype=torch.float32)
        return (z, stu_groups, tea_groups) if return_groups else z

    # --- full-vocab log-probs (gold_loss.py:524-540) via the REAL kernel ---
    stu_lp = G._gold_full_log_probs_from_vocab_parallel_logits(
        stu_answer_logits.float(), temperature=stu_temp,
        real_vocab_size=Vs, vocab_start=0, tp_size=1, tp_group=None,
    )
    tea_lp = G._gold_full_log_probs_from_vocab_parallel_logits(
        tea_answer_logits.float(), temperature=tea_temp,
        real_vocab_size=Vt, vocab_start=0, tp_size=1, tp_group=None,
    )

    # --- student merge via the CP-safe first-rows+tail-scalars kernel ---
    #     (gold_loss.py:478-560; base=group[-1] if bayesian else group[0]).
    base_positions = [int(g[-1] if bayesian else g[0]) for g in stu_groups]
    stu_first_lp = stu_lp[torch.tensor(base_positions, dtype=torch.long)]
    tail_positions = sorted(
        {int(p) for g in stu_groups for p in (g[:-1] if bayesian else g[1:])}
    )
    full_tail = G._gold_sparse_tail_label_log_probs_tp(
        stu_answer_logits.float(),
        label_ids_full=stu_ids,
        local_pos_by_global={j: j for j in range(stu_answer_logits.shape[0])},
        tail_global_positions=tail_positions,
        temperature=stu_temp, real_vocab_size=Vs, vocab_start=0,
        tp_size=1, tp_group=None,
    )
    stu_aligned = G._gold_merge_log_probs_from_first_rows_and_tail_scalars(
        stu_first_lp, stu_groups, full_tail, clamp_min_prob=clamp, bayesian=bayesian,
    )
    tea_aligned = G._gold_merge_log_probs_with_alignment_groups(
        tea_lp, tea_groups, tea_ids, clamp_min_prob=clamp, bayesian=bayesian,
    )

    # --- per-group kernel (gold_loss.py:561-572) via the REAL kernel ---
    if bool(args.gold_use_hybrid_loss):
        total, matched, unmatched = G._gold_hybrid_loss_from_log_probs(
            stu_aligned, tea_aligned, hybrid_vocab=hybrid_vocab,
            beta=float(args.gold_beta),
            matched_weight=args.gold_hybrid_matched_weight,
            unmatched_weight=args.gold_hybrid_unmatched_weight,
            return_components=True,
        )
        per_group = total
    else:
        per_group = G._gold_sorted_l1_from_log_probs(stu_aligned, tea_aligned)

    # mask == all-ones (fully unmasked synthetic answer region).  gold_loss.py:612-618
    # accumulates masked.sum(); distill_mean = total_distill/total_aligned
    # (gold_loss.py:717-718) == per-sample mean over groups at batch=1.
    n_groups = per_group.shape[0]
    scalar = distill_weight * per_group.sum() / max(n_groups, 1)
    if return_groups:
        return scalar, stu_groups, tea_groups
    return scalar


# --------------------------------------------------------------------------- #
#  TRL per-sample scalar via the REAL ULDLoss on a batch=1 windowed sample.     #
# --------------------------------------------------------------------------- #
def trl_gold_sample_scalar(
    arm: str,
    *,
    sfull: torch.Tensor,   # [1, P+R+1, Vs]  prompt rows + R answer rows + eos row
    tfull: torch.Tensor,   # [1, P+Rt+1, Vt]
    stu_ids: list[int], tea_ids: list[int],
    stu_offsets: list[tuple[int, int]], tea_offsets: list[tuple[int, int]],
    P: int, stu_tok: ToyTok, tea_tok: ToyTok,
    eos_s: int, eos_t: int,
):
    """Call the REAL TRL ULDLoss.__call__ (from the verbatim oracle) on a
    batch=1 sample.  ULDLoss internally windows to the labels!=-100 answer
    region (UNSHIFTED student_logits[start:start+size], gold_trainer.py:438-441),
    applies skip_*_eos (:418-421), aligns by byte offsets (:448) and merges/sorts.
    Byte offsets are the SAME arrays fed to the ours side."""
    R = len(stu_ids)
    Rt = len(tea_ids)
    s_labels = torch.tensor([[-100] * P + stu_ids + [eos_s]])
    t_labels = torch.tensor([[-100] * P + tea_ids + [eos_t]])
    s_input_ids = torch.tensor([[0] * P + stu_ids + [eos_s]])
    t_input_ids = torch.tensor([[0] * P + tea_ids + [eos_t]])
    s_content = stu_offsets[-1][1] if stu_offsets else 0
    t_content = tea_offsets[-1][1] if tea_offsets else 0
    s_off = torch.tensor([[(0, 0)] * P + stu_offsets + [(s_content, s_content)]])
    t_off = torch.tensor([[(0, 0)] * P + tea_offsets + [(t_content, t_content)]])
    cfg = mock_cfg_for_arm(arm)
    uld = O.ULDLoss(cfg, student_tokenizer=stu_tok, teacher_tokenizer=tea_tok, device=torch.device("cpu"))
    loss = uld(sfull, tfull, s_labels, t_labels, s_input_ids, t_input_ids,
               student_byte_offsets=s_off, teacher_byte_offsets=t_off)
    return loss


# --------------------------------------------------------------------------- #
#  Random synthetic case generator (identical byte offsets to BOTH sides).      #
# --------------------------------------------------------------------------- #
def gen_case(rng: torch.Generator, *, hybrid: bool):
    """One randomized answer region.  Returns everything both sides need.

    Byte offsets are constructed group-by-group so student & teacher share the
    group boundaries (exercising multi-token merge groups), with an occasional
    unshared trailing token to exercise the dropped-empty-pair path.
    """
    def ri(lo, hi):
        return int(torch.randint(lo, hi + 1, (1,), generator=rng).item())

    Vs = ri(8, 48)
    Vt = ri(8, 48)
    G_groups = ri(1, 7)
    stu_offsets: list[tuple[int, int]] = []
    tea_offsets: list[tuple[int, int]] = []
    cursor = 0
    for _ in range(G_groups):
        width = ri(2, 6)
        sc = ri(1, 3)  # student tokens in this group
        tc = ri(1, 3)  # teacher tokens in this group
        # split `width` bytes across sc student tokens (each >=1), monotonic
        def split(n):
            if n == 1:
                return [width]
            cuts = sorted(ri(1, width - 1) for _ in range(n - 1))
            pts = [0] + cuts + [width]
            # ensure strictly increasing so each piece >=1; if collision, fall back to even
            segs = [pts[k + 1] - pts[k] for k in range(n)]
            if any(s <= 0 for s in segs):
                base = width // n
                segs = [base] * (n - 1) + [width - base * (n - 1)]
            return segs
        s_segs = split(sc)
        t_segs = split(tc)
        c = cursor
        for w in s_segs:
            stu_offsets.append((c, c + w))
            c += w
        c = cursor
        for w in t_segs:
            tea_offsets.append((c, c + w))
            c += w
        cursor += width
    # occasionally add an unshared trailing token on one side (dropped empty pair)
    if ri(0, 3) == 0:
        w = ri(1, 4)
        if ri(0, 1) == 0:
            stu_offsets.append((cursor, cursor + w))
        else:
            tea_offsets.append((cursor, cursor + w))

    R = len(stu_offsets)
    Rt = len(tea_offsets)
    stu_ids = [ri(0, Vs - 1) for _ in range(R)]
    tea_ids = [ri(0, Vt - 1) for _ in range(Rt)]
    scale = 0.1 + (10.0 - 0.1) * float(torch.rand(1, generator=rng).item())

    P = 2  # prompt length (>=1 so the next-token-shift window has a real prompt row)
    # EOS is a valid in-vocab id living ONLY in the trailing eos ROW, which
    # skip_*_eos trims on BOTH sides (never enters the loss columns).  We do NOT
    # add an extra vocab COLUMN: that would give TRL an un-masked padding column
    # while ours masks it via real_vocab_size, an asymmetric-support artifact the
    # real pipeline never exposes (both operate on the real vocab).  So tensor
    # width == real vocab size, and ours's _mask_invalid_vocab_rows_ is a no-op.
    eos_s, eos_t = 0, 0
    Vs_dim, Vt_dim = Vs, Vt
    sfull = torch.randn(P + R + 1, Vs_dim, generator=rng) * scale
    tfull = torch.randn(P + Rt + 1, Vt_dim, generator=rng) * scale

    # toy tokenizers with a matched/unmatched string split (hybrid arms)
    if hybrid:
        # distinct string per student id; a random subset of teacher ids reuse a
        # student string (matched, string identity == TRL/ours matched rule).
        stu_vocab = {f"s{i}": i for i in range(Vs)}
        tea_vocab = {}
        n_match = ri(1, min(Vs, Vt))
        match_students = list(range(Vs))
        # pick n_match distinct teacher ids to be matched
        tea_ids_all = list(range(Vt))
        for j in tea_ids_all:
            tea_vocab[f"t{j}"] = j  # default teacher-only
        # assign matches
        for k in range(n_match):
            tj = tea_ids_all[k]
            si = match_students[k % Vs]
            tea_vocab[f"s{si}"] = tj  # same STRING as student id si -> matched pair
        stu_tok = ToyTok(stu_vocab)
        tea_tok = ToyTok(tea_vocab)
    else:
        stu_tok = ToyTok({f"s{i}": i for i in range(Vs)})
        tea_tok = ToyTok({f"t{j}": j for j in range(Vt)})

    return dict(
        Vs=Vs, Vt=Vt, Vs_dim=Vs_dim, Vt_dim=Vt_dim, P=P, R=R, Rt=Rt,
        stu_ids=stu_ids, tea_ids=tea_ids, stu_offsets=stu_offsets, tea_offsets=tea_offsets,
        sfull=sfull, tfull=tfull, stu_tok=stu_tok, tea_tok=tea_tok, eos_s=eos_s, eos_t=eos_t,
    )


def build_hybrid_vocab(args, case):
    if not args.gold_use_hybrid_loss:
        return None
    return G._gold_make_hybrid_vocab_mapping(
        case["stu_tok"], case["tea_tok"], device=torch.device("cpu"),
        student_vocab_dim=case["Vs_dim"], teacher_vocab_dim=case["Vt_dim"],
        student_real_vocab_size=case["Vs"], teacher_real_vocab_size=case["Vt"],
    )


def window_student(args, case):
    """Apply the arm's student windowing (gold_loss.py:116-166) at CP=1.

    observed / --gold-trl-faithful (uld-trl): UNSHIFTED  logits[P : P+R]  (gold_loss.py:133-137).
    non-faithful (hybrid arms):               SHIFTED    logits[P-1 : P-1+R] (get_responses:286-290).
    Both then use the SAME R answer token ids/offsets."""
    P, R = case["P"], case["R"]
    faithful_obs = bool(args.gold_trl_faithful) and args.gold_uld_token_merge_strategy == "observed"
    if faithful_obs:
        return case["sfull"][P:P + R]
    return case["sfull"][P - 1:P - 1 + R]


def window_teacher(args, case):
    """Teacher windowing mirror (gold_loss.py:355-368).

    observed drops the leading last-prompt hidden row (tea_hidden[1:]) so teacher
    rows are UNSHIFTED (distribution AT each teacher token's slot) == TRL
    teacher_logits[start:start+size].  non-faithful keeps the shifted predictive
    rows.  We model both from the same tfull tensor."""
    P, Rt = case["P"], case["Rt"]
    faithful_obs = bool(args.gold_trl_faithful) and args.gold_uld_token_merge_strategy == "observed"
    if faithful_obs:
        return case["tfull"][P:P + Rt]
    return case["tfull"][P - 1:P - 1 + Rt]


# --------------------------------------------------------------------------- #
#  Comparison driver.                                                           #
# --------------------------------------------------------------------------- #
def rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def run_synthetic(n_per_arm=220, seed=1234):
    arms = ["uld-trl", "gold-matched", "gold-unmatched", "gold-hybrid"]
    results = {}
    # per arm collect: as-configured max_rel, kernel-aligned max_rel, alignment mismatches
    for arm in arms:
        args = make_arm_args(arm)
        # a "kernel-aligned" variant = SAME arm but forced TRL-faithful window+clamp,
        # to isolate the JSD/L1 KERNEL from the windowing/clamp pipeline choice.
        args_kernel = make_arm_args(arm)
        args_kernel.gold_trl_faithful = True
        args_kernel.gold_uld_token_merge_strategy = "observed"

        rng = torch.Generator().manual_seed(seed + _ARM_SEED_OFFSET[arm])
        max_rel_cfg = 0.0
        max_rel_ker = 0.0
        max_abs_cfg = 0.0            # absolute-diff track (for near-zero-loss cases,
        max_rel_cfg_big = 0.0        # e.g. adaptive gold-hybrid, where rel amplifies)
        worst_cfg = None
        align_mismatch = 0
        n_ok = 0
        for _ in range(n_per_arm):
            case = gen_case(rng, hybrid=(arm != "uld-trl"))
            if case["R"] == 0 or case["Rt"] == 0:
                continue
            hv = build_hybrid_vocab(args, case)

            # --- TRL side (REAL ULDLoss) ---
            try:
                trl = float(trl_gold_sample_scalar(
                    arm, sfull=case["sfull"].unsqueeze(0), tfull=case["tfull"].unsqueeze(0),
                    stu_ids=case["stu_ids"], tea_ids=case["tea_ids"],
                    stu_offsets=case["stu_offsets"], tea_offsets=case["tea_offsets"],
                    P=case["P"], stu_tok=case["stu_tok"], tea_tok=case["tea_tok"],
                    eos_s=case["eos_s"], eos_t=case["eos_t"],
                ))
            except Exception as exc:
                print(f"[{arm}] TRL error: {exc!r}")
                continue

            # --- ours AS-CONFIGURED (REAL kernels) ---
            ours_cfg, sg, tg = ours_gold_sample_scalar(
                args, stu_answer_logits=window_student(args, case),
                tea_answer_logits=window_teacher(args, case),
                stu_ids=case["stu_ids"], tea_ids=case["tea_ids"],
                stu_offsets=case["stu_offsets"], tea_offsets=case["tea_offsets"],
                hybrid_vocab=hv, return_groups=True,
            )
            ours_cfg = float(ours_cfg)

            # --- ours KERNEL-ALIGNED (faithful window+clamp) ---
            hv_k = build_hybrid_vocab(args_kernel, case)
            ours_ker = float(ours_gold_sample_scalar(
                args_kernel, stu_answer_logits=window_student(args_kernel, case),
                tea_answer_logits=window_teacher(args_kernel, case),
                stu_ids=case["stu_ids"], tea_ids=case["tea_ids"],
                stu_offsets=case["stu_offsets"], tea_offsets=case["tea_offsets"],
                hybrid_vocab=hv_k,
            ))

            # --- alignment structure (both use identical offsets -> must match) ---
            trl_sg, trl_tg = O.ULDLoss._align_by_byte_offsets(case["stu_offsets"], case["tea_offsets"])
            trl_paired = [(a, b) for a, b in zip(trl_sg, trl_tg) if a and b]
            trl_sg = [a for a, _ in trl_paired]
            trl_tg = [b for _, b in trl_paired]
            if trl_sg != sg or trl_tg != tg:
                align_mismatch += 1

            rc = rel_diff(trl, ours_cfg)
            rk = rel_diff(trl, ours_ker)
            ac = abs(trl - ours_cfg)
            if rc > max_rel_cfg:
                max_rel_cfg = rc
                worst_cfg = (trl, ours_cfg, ac)
            max_abs_cfg = max(max_abs_cfg, ac)
            # rel restricted to non-trivial losses (>1e-3): isolates the kernel
            # agreement from small-denominator amplification on near-zero losses.
            if max(abs(trl), abs(ours_cfg)) > 1e-3:
                max_rel_cfg_big = max(max_rel_cfg_big, rc)
            max_rel_ker = max(max_rel_ker, rk)
            n_ok += 1
        results[arm] = dict(
            n=n_ok, max_rel_cfg=max_rel_cfg, max_rel_ker=max_rel_ker,
            max_abs_cfg=max_abs_cfg, max_rel_cfg_big=max_rel_cfg_big,
            align_mismatch=align_mismatch, worst_cfg=worst_cfg,
        )
    return results


# --------------------------------------------------------------------------- #
#  Reduction divergence D1: batch>1 per_token(default) vs TRL per-sample-mean.  #
# --------------------------------------------------------------------------- #
def run_reduction_check(seed=77):
    """Two samples of unequal group counts, uld-trl arm.  TRL batches as
    per-sample-mean-then-batch-mean (gold_trainer.py:494); ours default
    --opd-loss-reduction per_token (argument_groups.py:125) pools all groups.
    Reports both so the D1 magnitude is explicit."""
    args = make_arm_args("uld-trl")
    rng = torch.Generator().manual_seed(seed)
    cA = gen_case(rng, hybrid=False)
    cB = gen_case(rng, hybrid=False)

    # per-sample ours scalars (each already /num_groups)
    def per_sample(case):
        _, sg, tg = ours_gold_sample_scalar(
            args, stu_answer_logits=window_student(args, case),
            tea_answer_logits=window_teacher(args, case),
            stu_ids=case["stu_ids"], tea_ids=case["tea_ids"],
            stu_offsets=case["stu_offsets"], tea_offsets=case["tea_offsets"],
            return_groups=True,
        )
        # recover per-group vector: recompute sum for pooled reduction
        Vs = case["Vs_dim"]
        stu_lp = G._gold_full_log_probs_from_vocab_parallel_logits(
            window_student(args, case).float(), temperature=1.0, real_vocab_size=case["Vs"],
            vocab_start=0, tp_size=1, tp_group=None)
        tea_lp = G._gold_full_log_probs_from_vocab_parallel_logits(
            window_teacher(args, case).float(), temperature=1.0, real_vocab_size=case["Vt"],
            vocab_start=0, tp_size=1, tp_group=None)
        base = [int(g[0]) for g in sg]
        stu_first = stu_lp[torch.tensor(base, dtype=torch.long)]
        tails = sorted({int(p) for g in sg for p in g[1:]})
        ft = G._gold_sparse_tail_label_log_probs_tp(
            window_student(args, case).float(), label_ids_full=case["stu_ids"],
            local_pos_by_global={j: j for j in range(window_student(args, case).shape[0])},
            tail_global_positions=tails, temperature=1.0, real_vocab_size=case["Vs"],
            vocab_start=0, tp_size=1, tp_group=None)
        sa = G._gold_merge_log_probs_from_first_rows_and_tail_scalars(stu_first, sg, ft, clamp_min_prob=1e-8)
        ta = G._gold_merge_log_probs_with_alignment_groups(tea_lp, tg, case["tea_ids"], clamp_min_prob=1e-8)
        pg = G._gold_sorted_l1_from_log_probs(sa, ta)
        return pg
    pgA = per_sample(cA)
    pgB = per_sample(cB)
    trl_reduction = 0.5 * (pgA.sum() / pgA.shape[0] + pgB.sum() / pgB.shape[0])
    ours_per_token = (pgA.sum() + pgB.sum()) / (pgA.shape[0] + pgB.shape[0])
    return dict(
        nA=pgA.shape[0], nB=pgB.shape[0],
        trl_per_sample=float(trl_reduction), ours_per_token=float(ours_per_token),
        rel=rel_diff(float(trl_reduction), float(ours_per_token)),
    )


# --------------------------------------------------------------------------- #
#  Real-tokenizer cases: each side builds offsets/alignment with its OWN code.   #
# --------------------------------------------------------------------------- #
REAL_TEXTS = [
    " The answer is 4 because two plus two equals four.",
    " def solve(n):\n    return sum(range(n))  # O(n)",
    " 積分 ∫x^2 dx = x^3/3 + C, 微分は 2x です。",
    " Hello, world! 🌍🚀 Let's test emoji handling 😀.",
    " The limit as x→0 of sin(x)/x is exactly 1.",
    " 中文分词与英文 tokenization 很不一样。",
    " for (int i = 0; i < n; i++) { arr[i] *= 2; }",
    " Théorème de Pythagore: a² + b² = c².",
    " 확률 P(A∩B) = P(A)P(B|A) 조건부확률.",
    " x = [1,2,3]; y = x[::-1]  # reversed",
    " Σ_{k=1}^{n} k = n(n+1)/2, a classic identity.",
    " ¿Cómo estás? Muy bien, gracias. café ☕",
    " SELECT * FROM users WHERE age > 21;",
    " ∀ε>0 ∃δ>0 such that |x-a|<δ ⇒ |f(x)-L|<ε.",
    " 🎯 Goal: 100% accuracy on the eval set.",
    " print(f'{x:.4f}')  # four decimals",
    " これはテストです。トークン化を確認します。",
    " The matrix A⁻¹ exists iff det(A) ≠ 0.",
    " grep -rn 'pattern' . | awk '{print $1}'",
    " Ω(n log n) is the lower bound for comparison sorts.",
    " 3.14159265358979 is π to 14 decimal places.",
    " Möbius strip has only one side and one boundary.",
]


def run_real_tokenizer(max_cases=22):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        return dict(skipped=f"transformers import failed: {exc!r}")
    stu_path = os.environ.get("GOLD_ORACLE_STUDENT", "")
    tea_path = os.environ.get("GOLD_ORACLE_TEACHER", "")
    try:
        stok = AutoTokenizer.from_pretrained(stu_path, trust_remote_code=True)
        ttok = AutoTokenizer.from_pretrained(tea_path, trust_remote_code=True)
    except Exception as exc:
        return dict(skipped=f"tokenizer load failed: {exc!r}")
    if ttok.pad_token is None:
        ttok.pad_token = ttok.eos_token
    # The reference builds byte offsets from the fast tokenizer's offset_mapping. Some
    # tokenizers report every token as zero-width there, which leaves the reference
    # aligner with nothing to pair; this class is then not comparable.
    [(_probe_ids, _probe_off)] = O.encode_with_byte_offsets(
        ttok.backend_tokenizer, [" probe text"], add_special_tokens=False)
    if not any(b > a for a, b in _probe_off):
        return dict(skipped="teacher offset_mapping is degenerate (all zero-width), "
                            "so the reference aligner has no spans to pair")

    rows = []
    align_group_mismatch = 0
    n = 0
    max_rel_uld = 0.0
    worst = None
    rng = torch.Generator().manual_seed(2024)
    prompt = "Solve the following problem."
    for text in REAL_TEXTS[:max_cases]:
        # ---- TRL side offsets: build_teacher_inputs_from_texts (teacher) +
        #      encode_with_byte_offsets (student), gold_trainer.py:180 / :2314.
        try:
            t_ids_full, t_lab, t_am, t_boff = O.build_teacher_inputs_from_texts(ttok, [prompt], [text])
            [(s_ids_trl, s_off_trl)] = O.encode_with_byte_offsets(stok.backend_tokenizer, [text], add_special_tokens=False)
        except Exception as exc:
            rows.append((text[:22], "TRL-offset-ERR", repr(exc)[:40]))
            continue
        t_lab0 = t_lab[0]
        tstart = int((t_lab0 != -100).nonzero()[0])
        tsize = int((t_lab0 != -100).sum()) - 1  # skip_teacher_eos
        t_off_trl = [tuple(x) for x in t_boff[0, tstart:tstart + tsize].tolist()]
        t_ids_trl = t_ids_full[0, tstart:tstart + tsize].tolist()
        s_off_trl = [tuple(x) for x in s_off_trl]  # skip_student_eos: no eos appended
        s_ids_trl = list(s_ids_trl)

        # ---- ours side offsets: _gold_offsets_from_actual_token_ids from the
        #      ACTUAL token ids (raw ByteLevel piece widths), gold_utils.py:161-224.
        #      Student ids = same student encoding; teacher ids = same teacher
        #      completion ids -- but offsets come from ours's OWN builder.
        try:
            s_ids_krl, s_off_krl = G._gold_trim_ids_and_offsets_for_answer(
                stok, list(s_ids_trl) + [stok.eos_token_id], text,
                skip_eos=True, eos_token_id=stok.eos_token_id)
            t_ids_krl, t_off_krl = G._gold_trim_ids_and_offsets_for_answer(
                ttok, list(t_ids_trl) + [ttok.eos_token_id], text,
                skip_eos=True, eos_token_id=ttok.eos_token_id)
        except Exception as exc:
            rows.append((text[:22], "ours-offset-ERR", repr(exc)[:40]))
            continue

        # ---- alignment structure comparison (EACH side's own offsets) ----
        trl_sg, trl_tg = O.ULDLoss._align_by_byte_offsets(s_off_trl, t_off_trl)
        trl_sg = [a for a, b in zip(trl_sg, trl_tg) if a and b]
        trl_tg = [b for a, b in zip(*O.ULDLoss._align_by_byte_offsets(s_off_trl, t_off_trl)) if a and b]
        ours_sg, ours_tg = G._gold_align_by_byte_offsets(s_off_krl, t_off_krl)
        ours_paired = [(a, b) for a, b in zip(ours_sg, ours_tg) if a and b]
        ours_sg = [a for a, _ in ours_paired]
        ours_tg = [b for _, b in ours_paired]
        groups_match = (trl_sg == ours_sg and trl_tg == ours_tg)
        offsets_match = (s_off_trl == s_off_krl and t_off_trl == t_off_krl)
        if not groups_match:
            align_group_mismatch += 1

        # ---- uld-trl loss on the SAME offsets each side built (seeded logits
        #      over the real vocab dims).  We give BOTH sides identical answer
        #      logits + each its own offsets, so any loss gap == alignment gap. ----
        Vs = len(stok.get_vocab())
        Vt = len(ttok.get_vocab())
        Rs = len(s_ids_trl)
        Rt = len(t_ids_trl)
        # use small dense synthetic logits over a REDUCED support to keep CPU sort
        # cheap but exercise the real ids: place random logits on the real vocab
        # dims is 150k-wide sort per row -> still fine for ~20 rows.
        s_answer = torch.randn(Rs, Vs, generator=rng)
        t_answer = torch.randn(Rt, Vt, generator=rng)

        # TRL: wrap into batch=1 with its own offsets
        P = 1
        eos_s, eos_t = stok.eos_token_id, ttok.eos_token_id
        sfull = torch.cat([torch.randn(P, Vs, generator=rng), s_answer, torch.randn(1, Vs, generator=rng)], 0)
        tfull = torch.cat([torch.randn(P, Vt, generator=rng), t_answer, torch.randn(1, Vt, generator=rng)], 0)
        args = make_arm_args("uld-trl")
        try:
            trl_loss = float(trl_gold_sample_scalar(
                "uld-trl", sfull=sfull.unsqueeze(0), tfull=tfull.unsqueeze(0),
                stu_ids=s_ids_trl, tea_ids=t_ids_trl,
                stu_offsets=s_off_trl, tea_offsets=t_off_trl,
                P=P, stu_tok=stok, tea_tok=ttok, eos_s=eos_s, eos_t=eos_t))
            # ours uses its OWN offsets + the UNSHIFTED answer window (observed).
            ours_loss = float(ours_gold_sample_scalar(
                args, stu_answer_logits=s_answer, tea_answer_logits=t_answer,
                stu_ids=s_ids_krl, tea_ids=t_ids_krl,
                stu_offsets=s_off_krl, tea_offsets=t_off_krl))
        except Exception as exc:
            rows.append((text[:22], "loss-ERR", repr(exc)[:50]))
            continue
        r = rel_diff(trl_loss, ours_loss)
        if r > max_rel_uld:
            max_rel_uld = r
            worst = (text[:30], trl_loss, ours_loss)
        n += 1
        rows.append((
            text[:22], f"S{Rs}/T{Rt}",
            f"grp {len(ours_sg)}", "off=" + ("Y" if offsets_match else "N"),
            "grp=" + ("Y" if groups_match else "N"), f"rel={r:.1e}",
        ))
    return dict(rows=rows, n=n, align_group_mismatch=align_group_mismatch,
                max_rel_uld=max_rel_uld, worst=worst)


# --------------------------------------------------------------------------- #
#  PADDED-VOCAB mask-path parity: ours tensor WIDER than real vocab, pad columns   #
#  poisoned with +-inf / NaN, masked via real_vocab_size; TRL sees real vocab.    #
# --------------------------------------------------------------------------- #
def run_padded_vocab_check(seed=4242):
    """ours receives an answer-logit tensor WIDER than the real vocab, whose pad
    columns are POISONED with +inf / -inf / NaN.  real_vs/real_vt route ours through
    _mask_invalid_vocab_rows_ (gold_utils), which must overwrite every pad column to
    -finfo.max BEFORE the softmax so the poison never leaks.  TRL gets ONLY the real
    vocab columns.  The masked ours sorted-L1 ULD loss MUST equal TRL's real-vocab loss
    (rtol 1e-5) AND be finite -> proves ours's mask path neutralizes padded/poisoned
    columns exactly (uld-trl arm; the arm whose tensors can carry lm-head padding)."""
    args = make_arm_args("uld-trl")
    rng = torch.Generator().manual_seed(seed)
    case = gen_case(rng, hybrid=False)
    while case["R"] == 0 or case["Rt"] == 0:
        case = gen_case(rng, hybrid=False)
    Vs, Vt = case["Vs"], case["Vt"]
    R, Rt = case["R"], case["Rt"]
    s_real = window_student(args, case)   # [R, Vs]  real-vocab answer logits
    t_real = window_teacher(args, case)   # [Rt, Vt]

    pad_s, pad_t = 5, 7
    _POISON = [float("inf"), float("-inf"), float("nan")]

    def poison(rows, pad):
        blk = torch.empty(rows, pad, dtype=torch.float32)
        for r in range(rows):
            for c in range(pad):
                blk[r, c] = _POISON[(r + c) % 3]
        return blk

    s_pad = torch.cat([s_real, poison(R, pad_s)], dim=1)    # [R, Vs+pad_s]
    t_pad = torch.cat([t_real, poison(Rt, pad_t)], dim=1)   # [Rt, Vt+pad_t]

    # ours on padded+poisoned tensor, real_vocab_size = real vocab -> mask path.
    ours = float(ours_gold_sample_scalar(
        args, stu_answer_logits=s_pad, tea_answer_logits=t_pad,
        stu_ids=case["stu_ids"], tea_ids=case["tea_ids"],
        stu_offsets=case["stu_offsets"], tea_offsets=case["tea_offsets"],
        real_vs=Vs, real_vt=Vt,
    ))

    # TRL reference on ONLY the real vocab columns (mask-equivalent support).
    P = case["P"]
    sfull = torch.cat([torch.zeros(P, Vs), s_real, torch.zeros(1, Vs)], 0)
    tfull = torch.cat([torch.zeros(P, Vt), t_real, torch.zeros(1, Vt)], 0)
    trl = float(trl_gold_sample_scalar(
        "uld-trl", sfull=sfull.unsqueeze(0), tfull=tfull.unsqueeze(0),
        stu_ids=case["stu_ids"], tea_ids=case["tea_ids"],
        stu_offsets=case["stu_offsets"], tea_offsets=case["tea_offsets"],
        P=P, stu_tok=case["stu_tok"], tea_tok=case["tea_tok"],
        eos_s=case["eos_s"], eos_t=case["eos_t"],
    ))
    return dict(
        trl=trl, ours=ours, rel=rel_diff(trl, ours),
        finite=bool(math.isfinite(ours)),
        width_s=Vs + pad_s, real_s=Vs, width_t=Vt + pad_t, real_t=Vt,
    )


# --------------------------------------------------------------------------- #
#  BATCH>=2 per_sample aggregation: assert sum_i(sd_i/aligned_i)/N == TRL         #
#  per-sample-mean (gold_trainer.py:494) via a REAL batched ULDLoss call.         #
# --------------------------------------------------------------------------- #
def run_batch_persample_check(seed=8888, N=3):
    """Build N unequal-group samples, call the REAL TRL ULDLoss ONCE on a right-padded
    batch of N, and assert its scalar == mean of the N ours per-sample scalars.
    TRL's _compute_distillation_loss stacks per-sample means then batch-means
    (gold_trainer.py:494): batch = (1/N) sum_i (sd_i / aligned_i); each ours per-sample
    scalar = per_group.sum()/n_groups = sd_i/aligned_i.  Samples are padded to a common
    vocab width with -inf columns (softmax->0, loss-invariant on BOTH sides; ours also
    masks them via real_vs/real_vt), so batching does not perturb the per-sample loss."""
    args = make_arm_args("uld-trl")
    rng = torch.Generator().manual_seed(seed)
    cases = []
    while len(cases) < N:
        c = gen_case(rng, hybrid=False)
        if c["R"] > 0 and c["Rt"] > 0:
            cases.append(c)
    P = cases[0]["P"]
    Vs = max(c["Vs"] for c in cases)
    Vt = max(c["Vt"] for c in cases)

    def padV(x, target):
        if x.shape[-1] == target:
            return x
        pad = torch.full((x.shape[0], target - x.shape[-1]), float("-inf"), dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    # --- per-sample ours scalars (real support via real_vs/real_vt mask) ---
    ours_scalars = []
    for c in cases:
        ours_scalars.append(float(ours_gold_sample_scalar(
            args, stu_answer_logits=padV(window_student(args, c), Vs),
            tea_answer_logits=padV(window_teacher(args, c), Vt),
            stu_ids=c["stu_ids"], tea_ids=c["tea_ids"],
            stu_offsets=c["stu_offsets"], tea_offsets=c["tea_offsets"],
            real_vs=c["Vs"], real_vt=c["Vt"],
        )))
    ours_batch_mean = sum(ours_scalars) / N

    # --- build a REAL right-padded batch and call TRL ULDLoss once ---
    s_seqs, t_seqs, s_lab, t_lab, s_in, t_in, s_off, t_off = ([] for _ in range(8))
    for c in cases:
        R, Rt = c["R"], c["Rt"]
        s_ans = padV(window_student(args, c), Vs)
        t_ans = padV(window_teacher(args, c), Vt)
        s_seqs.append(torch.cat([torch.zeros(P, Vs), s_ans, torch.zeros(1, Vs)], 0))
        t_seqs.append(torch.cat([torch.zeros(P, Vt), t_ans, torch.zeros(1, Vt)], 0))
        s_lab.append(torch.tensor([-100] * P + c["stu_ids"] + [c["eos_s"]]))
        t_lab.append(torch.tensor([-100] * P + c["tea_ids"] + [c["eos_t"]]))
        s_in.append(torch.tensor([0] * P + c["stu_ids"] + [c["eos_s"]]))
        t_in.append(torch.tensor([0] * P + c["tea_ids"] + [c["eos_t"]]))
        sc = c["stu_offsets"][-1][1] if c["stu_offsets"] else 0
        tc = c["tea_offsets"][-1][1] if c["tea_offsets"] else 0
        s_off.append(torch.tensor([(0, 0)] * P + c["stu_offsets"] + [(sc, sc)]))
        t_off.append(torch.tensor([(0, 0)] * P + c["tea_offsets"] + [(tc, tc)]))
    sb = O.pad(s_seqs, padding_value=0.0)
    tb = O.pad(t_seqs, padding_value=0.0)
    s_labels = O.pad(s_lab, padding_value=-100)
    t_labels = O.pad(t_lab, padding_value=-100)
    s_input = O.pad(s_in, padding_value=0)
    t_input = O.pad(t_in, padding_value=0)
    s_offb = O.pad(s_off, padding_value=0)
    t_offb = O.pad(t_off, padding_value=0)
    cfg = mock_cfg_for_arm("uld-trl")
    uld = O.ULDLoss(cfg, student_tokenizer=cases[0]["stu_tok"], teacher_tokenizer=cases[0]["tea_tok"],
                    device=torch.device("cpu"))
    trl_batch = float(uld(sb, tb, s_labels, t_labels, s_input, t_input,
                          student_byte_offsets=s_offb, teacher_byte_offsets=t_offb))
    return dict(N=N, ours_scalars=ours_scalars, ours_batch_mean=ours_batch_mean,
                trl_batch=trl_batch, rel=rel_diff(trl_batch, ours_batch_mean))


# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    print("=" * 78)
    print("EQUIVALENCE HARNESS  TRL GOLD/ULD  vs  ours gold (REAL kernels)")
    print("=" * 78)

    # ---- (i) synthetic randomized cases ----
    print("\n### CLASS A: synthetic randomized cases (identical byte offsets both sides)")
    # 220 per arm is the published sweep; GOLD_EQUIV_CASES shortens it for a quick pass.
    syn = run_synthetic(n_per_arm=int(os.environ.get("GOLD_EQUIV_CASES", 220)))
    print(f"\n{'arm':<16}{'n':>5}{'max_rel(cfg)':>15}{'max_abs(cfg)':>15}"
          f"{'max_rel(loss>1e-3)':>20}{'max_rel(kernel)':>17}{'align_mism':>12}")
    for arm, r in syn.items():
        print(f"{arm:<16}{r['n']:>5}{r['max_rel_cfg']:>15.3e}{r['max_abs_cfg']:>15.3e}"
              f"{r['max_rel_cfg_big']:>20.3e}{r['max_rel_ker']:>17.3e}{r['align_mismatch']:>12}")
    for arm, r in syn.items():
        if r["worst_cfg"] is not None:
            print(f"   [{arm}] worst-rel as-configured pair (TRL, ours, |diff|) = "
                  f"({r['worst_cfg'][0]:.6f}, {r['worst_cfg'][1]:.6f}, {r['worst_cfg'][2]:.2e})")

    # ---- reduction divergence D1 ----
    print("\n### D1 REDUCTION (batch=2, uld-trl): default per_token vs TRL per-sample-mean")
    red = run_reduction_check()
    print(f"   sample groups: A={red['nA']} B={red['nB']}")
    print(f"   TRL per-sample-mean = {red['trl_per_sample']:.8f}   "
          f"ours per_token(default) = {red['ours_per_token']:.8f}   rel_diff = {red['rel']:.3e}")

    # ---- padded-vocab mask-path parity ----
    print("\n### PADDED-VOCAB (mask-path): ours width>real vocab, pad cols poisoned +-inf/NaN")
    pad = run_padded_vocab_check()
    print(f"   student width={pad['width_s']} (real {pad['real_s']})  teacher width={pad['width_t']} "
          f"(real {pad['real_t']})  poison=+inf/-inf/NaN")
    print(f"   TRL(real vocab) = {pad['trl']:.8f}   ours(masked padded) = {pad['ours']:.8f}   "
          f"finite={pad['finite']}   rel_diff = {pad['rel']:.3e}")
    assert pad["finite"], "PADDED-VOCAB FAIL: ours produced non-finite loss -> poison leaked past the mask"
    assert pad["rel"] < 1e-5, (
        f"PADDED-VOCAB FAIL: masked-padded ours != real-vocab TRL, rel={pad['rel']:.3e}")

    # ---- batch>=2 per_sample aggregation ----
    print("\n### BATCH>=2 per_sample: sum_i(sd_i/aligned_i)/N  ==  TRL per-sample-mean (real batch)")
    bat = run_batch_persample_check()
    print(f"   N={bat['N']}  ours per-sample scalars = "
          f"[{', '.join(f'{x:.6f}' for x in bat['ours_scalars'])}]")
    print(f"   ours mean = {bat['ours_batch_mean']:.8f}   TRL batch = {bat['trl_batch']:.8f}   "
          f"rel_diff = {bat['rel']:.3e}")
    assert bat["rel"] < 1e-5, (
        f"BATCH per_sample FAIL: mean(ours per-sample) != TRL per-sample-mean, rel={bat['rel']:.3e}")

    # ---- (ii) real-tokenizer cases ----
    print("\n### CLASS B: real-tokenizer cases (each side builds its OWN offsets/alignment)")
    real = run_real_tokenizer()
    if real.get("skipped"):
        print(f"   [SKIPPED] {real['skipped']}")
    else:
        print(f"   {'text':<24}{'S/T':<10}{'groups':<9}{'offs':<7}{'grp':<7}{'loss_rel':<12}")
        for row in real["rows"]:
            print("   " + "".join(f"{str(c):<{w}}" for c, w in zip(
                row, [24, 10, 9, 7, 7, 12][:len(row)])))
        print(f"\n   real cases n={real['n']}  alignment-group mismatches (own-offset builders)="
              f"{real['align_group_mismatch']}  max_rel(uld-trl loss)={real['max_rel_uld']:.3e}")
        if real["worst"] is not None:
            print(f"   worst uld-trl: {real['worst'][0]!r} TRL={real['worst'][1]:.6f} ours={real['worst'][2]:.6f}")

    # ---- verdict table ----
    print("\n" + "=" * 78)
    print("VERDICT TABLE  (rtol 1e-5)")
    print("=" * 78)
    # Certify on a mixed rel/abs criterion: PASS if rel<1e-5 OR |diff|<1e-6.
    # The abs fallback matters ONLY for the adaptive gold-hybrid arm, whose seed
    # stream hits a near-zero-loss case (loss ~6e-5, adaptive matched_weight 1/43)
    # where the SAME ~1e-7 float32 merge noise as the certified arms amplifies
    # into a ~2e-3 relative number.  All arms' kernels agree to ~1e-7 absolute.
    def _syn_verdict(r):
        return "PASS" if (r["max_rel_cfg"] < 1e-5 or r["max_abs_cfg"] < 1e-6) else "DIVERGES"
    print(f"{'case-class':<34}{'n':>5}{'max_rel':>13}{'max_abs':>12}{'verdict':>12}")
    v_uld = syn["uld-trl"]
    print(f"{'A synthetic uld-trl (as-config)':<34}{v_uld['n']:>5}{v_uld['max_rel_cfg']:>13.2e}"
          f"{v_uld['max_abs_cfg']:>12.2e}{_syn_verdict(v_uld):>12}")
    for arm in ("gold-matched", "gold-unmatched", "gold-hybrid"):
        r = syn[arm]
        print(f"{'A synthetic ' + arm + ' (as-config)':<34}{r['n']:>5}{r['max_rel_cfg']:>13.2e}"
              f"{r['max_abs_cfg']:>12.2e}{_syn_verdict(r):>12}")
        print(f"{'A synthetic ' + arm + ' (kernel-only)':<34}{r['n']:>5}{r['max_rel_ker']:>13.2e}"
              f"{r['max_abs_cfg']:>12.2e}{_syn_verdict(r):>12}")
    print(f"{'D1 NEG-CTRL per_token (per_sample=launch)':<34}{'2':>5}{red['rel']:>13.2e}{'':>12}"
          f"{('PASS' if red['rel'] < 1e-5 else 'EXP-DIVERGE'):>12}")
    print(f"{'PADDED-VOCAB mask-path (uld-trl)':<34}{'1':>5}{pad['rel']:>13.2e}{'':>12}"
          f"{('PASS' if (pad['rel'] < 1e-5 and pad['finite']) else 'FAIL'):>12}")
    print(f"{'BATCH>=2 per_sample aggregation':<34}{bat['N']:>5}{bat['rel']:>13.2e}{'':>12}"
          f"{('PASS' if bat['rel'] < 1e-5 else 'FAIL'):>12}")
    if not real.get("skipped"):
        print(f"{'B real-tokenizer uld-trl loss':<34}{real['n']:>5}{real['max_rel_uld']:>13.2e}{'':>12}"
              f"{('PASS' if real['max_rel_uld'] < 1e-5 else 'DIVERGES'):>12}")
        print(f"{'B real-tokenizer alignment groups':<34}{real['n']:>5}"
              f"{('mism=' + str(real['align_group_mismatch'])):>13}{'':>12}"
              f"{('PASS' if real['align_group_mismatch'] == 0 else 'DIVERGES'):>12}")
    kernel_rows = [_syn_verdict(syn[a]) for a in ("uld-trl", "gold-matched", "gold-unmatched", "gold-hybrid")]
    kernel_rows += ["PASS" if (pad["rel"] < 1e-5 and pad["finite"]) else "FAIL"]
    kernel_rows += ["PASS" if bat["rel"] < 1e-5 else "FAIL"]
    n_pass = sum(1 for v in kernel_rows if v == "PASS")
    print(f"\n{n_pass}/{len(kernel_rows)} kernel comparisons PASS. D1 is a negative control: "
          f"harness default per_token vs reference per-sample; the launchers set per_sample.")
    print("DONE")


if __name__ == "__main__":
    main()
