"""EDGE-CASE HARNESS -- adversarial breakers for TRL GOLD/ULD vs ours gold.

Companion to test_gold_trl_equiv.py.  Where that file runs broad randomized
cases, this file hand-builds pathological cases explicitly DESIGNED to break
equivalence, per the review mandate:

  single-token response; EOS-only (empty answer) response; teacher seq much
  longer / shorter than student; all-mass-on-one-token logits (+-30); duplicate
  tokens; empty alignment groups; beta in {0, 1} hybrid extremes; chunk boundary
  31/32/33 groups vs chunk-size 32; both temperatures 0.7.

Each case is fed to BOTH sides on identical inputs:
  * TRL   = the REAL O.ULDLoss.__call__ from trl_gold_oracle.py (verbatim copy).
  * ours   = the REAL kernels in slime_plugins/baselines/gold/gold_kernels.py, orchestrated exactly
            as gold_loss.py drives them (via H.ours_gold_sample_scalar).

We compare under the TRL-FAITHFUL / kernel-aligned config (--gold-trl-faithful
+ observed merge + 1e-8 clamp) so windowing matches on both sides and the test
ISOLATES the JSD / sorted-L1 / merge KERNELS.  Any FAIL here is a genuine kernel
divergence.  We ALSO record the as-configured hybrid (SHIFTED, no clamp) number
to reconfirm the windowing-shift divergence that the main harness flagged.

Run:  python3 test_gold_trl_edge.py

CERTIFIED MAXIMA  (re-recorded 2026-07-03; DETERMINISTIC -- identical across 3 re-runs
after seeding random/torch at import + FIXED per-case-name seed map _fixed_seed; rtol 1e-5):
  E1..E15, E17  : all PASS (max kernel rel ~5.90e-07 at E8 hybrid matched beta=0.0);
                  kernel FAILs / one-sided ERRs = 0.
  E16 (PRE-FIX-SHIFT negative control, gold_trl_faithful=False + shifted merge):
                  rel = 2.276e-01  DIVERGES(expected)  (asserts rel > 1e-2).
  E17 (FORCED-FAITHFUL):  rel = 1.04e-07  PASS.
  E18 (ADAPTIVE gold-hybrid weights at extreme overlap fractions, Vt=10):
                  m=0    -> ours matched_weight 0.0000 (TRL cannot construct a zero-overlap
                            hybrid: gold_trainer.py:413 max() of empty vocab_mapping);
                  m=1    -> weight 1/10=0.1000, ours==TRL, loss PASS;
                  m=all  -> weight 10/10=1.0000, ours==TRL, loss PASS.
                  Asserts adaptive-weight parity (gold_kernels.py:761-765 ==
                  TRL gold_trainer.py:934-936) at both live extremes.
"""

from __future__ import annotations

import argparse
import copy
import random
import zlib

import torch

import test_gold_trl_equiv as H  # reuse loader (O, G) + helpers
O = H.O
G = H.G
ToyTok = H.ToyTok

# DETERMINISM: seed both RNGs at MODULE IMPORT (H already seeds on its own import,
# but seed again here so this harness is reproducible when run standalone,
# independent of PYTHONHASHSEED).  Re-running 3x yields identical maxima.
random.seed(20260703)
torch.manual_seed(20260703)


def _fixed_seed(name: str) -> int:
    """Deterministic per-case-name integer seed (a FIXED map: same name -> same
    int on every run and every machine).  Replaces the hash-randomized
    ``hash(name) % 2**31`` so the per-case logits are reproducible."""
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


# --------------------------------------------------------------------------- #
def faithful_args(*, hybrid=False, matched_w=None, unmatched_w=None,
                  beta=0.5, stu_temp=1.0, tea_temp=1.0, chunk=32):
    """Namespace forced TRL-faithful (observed + 1e-8 clamp) so windowing
    matches TRL and only the kernels are exercised."""
    ns = argparse.Namespace(
        opd_backend="gold",
        gold_distillation_weight=1.0, gold_ce_weight=0.0,
        gold_student_temperature=stu_temp, gold_teacher_temperature=tea_temp,
        gold_use_extended_uld=True, gold_skip_student_eos=True,
        gold_skip_teacher_eos=True, gold_chunk_size=chunk, gold_beta=beta,
        opd_loss_reduction="per_token",
        gold_use_hybrid_loss=hybrid,
        gold_hybrid_matched_weight=matched_w,
        gold_hybrid_unmatched_weight=unmatched_w,
        gold_trl_faithful=True,               # force observed+clamp
        gold_uld_token_merge_strategy="observed",
    )
    return ns


def mock_cfg(*, hybrid=False, matched_w=None, unmatched_w=None,
             beta=0.5, stu_temp=1.0, tea_temp=1.0, chunk=32):
    return O.MockGOLDConfig(
        uld_use_hybrid_loss=hybrid, beta=beta,
        uld_hybrid_matched_weight=matched_w, uld_hybrid_unmatched_weight=unmatched_w,
        uld_hybrid_matched_chunk_size=chunk, uld_hybrid_unmatched_chunk_size=chunk,
        uld_student_temperature=stu_temp, uld_teacher_temperature=tea_temp,
    )


def trl_scalar(cfg, *, sfull, tfull, stu_ids, tea_ids, stu_offsets, tea_offsets,
               P, stu_tok, tea_tok, eos_s=0, eos_t=0):
    """REAL O.ULDLoss on a batch=1 windowed sample with an arbitrary cfg."""
    R, Rt = len(stu_ids), len(tea_ids)
    s_labels = torch.tensor([[-100] * P + stu_ids + [eos_s]])
    t_labels = torch.tensor([[-100] * P + tea_ids + [eos_t]])
    s_input_ids = torch.tensor([[0] * P + stu_ids + [eos_s]])
    t_input_ids = torch.tensor([[0] * P + tea_ids + [eos_t]])
    s_content = stu_offsets[-1][1] if stu_offsets else 0
    t_content = tea_offsets[-1][1] if tea_offsets else 0
    s_off = torch.tensor([[(0, 0)] * P + stu_offsets + [(s_content, s_content)]])
    t_off = torch.tensor([[(0, 0)] * P + tea_offsets + [(t_content, t_content)]])
    uld = O.ULDLoss(cfg, student_tokenizer=stu_tok, teacher_tokenizer=tea_tok,
                    device=torch.device("cpu"))
    loss = uld(sfull.unsqueeze(0), tfull.unsqueeze(0), s_labels, t_labels,
               s_input_ids, t_input_ids,
               student_byte_offsets=s_off, teacher_byte_offsets=t_off)
    return float(loss)


def ours_scalar(args, *, stu_answer_logits, tea_answer_logits, stu_ids, tea_ids,
               stu_offsets, tea_offsets, hybrid_vocab=None):
    return float(H.ours_gold_sample_scalar(
        args, stu_answer_logits=stu_answer_logits, tea_answer_logits=tea_answer_logits,
        stu_ids=stu_ids, tea_ids=tea_ids, stu_offsets=stu_offsets, tea_offsets=tea_offsets,
        hybrid_vocab=hybrid_vocab))


def hybrid_toks(Vs, Vt, matched_pairs):
    """matched_pairs: list of (student_id, teacher_id) sharing a STRING."""
    stu_vocab = {f"s{i}": i for i in range(Vs)}
    tea_vocab = {f"t{j}": j for j in range(Vt)}
    for si, tj in matched_pairs:
        tea_vocab[f"s{si}"] = tj  # teacher id tj now shares student si's string
    return ToyTok(stu_vocab), ToyTok(tea_vocab)


def build_hv(args, stu_tok, tea_tok, Vs, Vt):
    if not args.gold_use_hybrid_loss:
        return None
    return G._gold_make_hybrid_vocab_mapping(
        stu_tok, tea_tok, device=torch.device("cpu"),
        student_vocab_dim=Vs, teacher_vocab_dim=Vt,
        student_real_vocab_size=Vs, teacher_real_vocab_size=Vt)


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


# --------------------------------------------------------------------------- #
#  The full-vocab-answer window builder.  The observed/faithful window is
#  logits[P:P+R]; we build sfull with P prompt rows + R answer rows + 1 eos row,
#  so window_student(faithful)=sfull[P:P+R] == the answer rows we pass to ours.
# --------------------------------------------------------------------------- #
def mk_logits(rows, V, *, gen, scale=1.0, spike=None):
    """rows x V logit tensor.  spike=(col, val) puts +val on col, -val elsewhere
    for an all-mass-on-one-token distribution."""
    if spike is not None:
        col, val = spike
        t = torch.full((rows, V), -abs(val), dtype=torch.float32)
        t[:, col % V] = abs(val)
        return t
    return torch.randn(rows, V, generator=gen) * scale


def run_case(name, *, args, cfg, P, Vs, Vt, stu_ids, tea_ids, stu_offsets,
             tea_offsets, s_answer, t_answer, stu_tok, tea_tok, results,
             expect_zero=False):
    """Compute both sides on the answer window and record rel diff."""
    # TRL needs the full [P + R + 1, V] tensor; prepend P prompt rows + append eos row.
    gen = torch.Generator().manual_seed(_fixed_seed(name))
    R, Rt = len(stu_ids), len(tea_ids)
    sfull = torch.cat([mk_logits(P, Vs, gen=gen), s_answer, mk_logits(1, Vs, gen=gen)], 0)
    tfull = torch.cat([mk_logits(P, Vt, gen=gen), t_answer, mk_logits(1, Vt, gen=gen)], 0)
    hv = build_hv(args, stu_tok, tea_tok, Vs, Vt)
    try:
        trl = trl_scalar(cfg, sfull=sfull, tfull=tfull, stu_ids=stu_ids, tea_ids=tea_ids,
                         stu_offsets=stu_offsets, tea_offsets=tea_offsets, P=P,
                         stu_tok=stu_tok, tea_tok=tea_tok)
    except Exception as exc:
        trl = f"ERR:{type(exc).__name__}"
    try:
        ours = ours_scalar(args, stu_answer_logits=s_answer, tea_answer_logits=t_answer,
                         stu_ids=stu_ids, tea_ids=tea_ids, stu_offsets=stu_offsets,
                         tea_offsets=tea_offsets, hybrid_vocab=hv)
    except Exception as exc:
        ours = f"ERR:{type(exc).__name__}"
    if isinstance(trl, str) or isinstance(ours, str):
        r = None
        verdict = "BOTH-ERR" if (isinstance(trl, str) and isinstance(ours, str)) else "ONE-ERR"
    else:
        r = rel(trl, ours)
        if expect_zero:
            verdict = "PASS(0)" if (abs(trl) < 1e-9 and abs(ours) < 1e-9) else "CHECK"
        else:
            verdict = "PASS" if r < 1e-5 else "FAIL"
    results.append((name, trl, ours, r, verdict))
    return trl, ours, r


# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(7)
    results: list[tuple] = []

    # ---------- E1: single-token response, uld-trl kernel ----------
    a = faithful_args(); c = mock_cfg()
    P, Vs, Vt = 2, 16, 20
    run_case("E1 single-token uld", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[3], tea_ids=[5], stu_offsets=[(0, 4)], tea_offsets=[(0, 4)],
             s_answer=mk_logits(1, Vs, gen=gen), t_answer=mk_logits(1, Vt, gen=gen),
             stu_tok=ToyTok({f"s{i}": i for i in range(Vs)}),
             tea_tok=ToyTok({f"t{j}": j for j in range(Vt)}), results=results)

    # ---------- E2: EOS-only response (empty answer after skip_eos) ----------
    # stu_ids empty -> TRL answer_size = 1 (just eos) -> skip -> 0 -> zero loss.
    # ours: stu offsets empty -> no groups -> zero.
    a = faithful_args(); c = mock_cfg()
    run_case("E2 eos-only empty", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[], tea_ids=[], stu_offsets=[], tea_offsets=[],
             s_answer=mk_logits(0, Vs, gen=gen), t_answer=mk_logits(0, Vt, gen=gen),
             stu_tok=ToyTok({f"s{i}": i for i in range(Vs)}),
             tea_tok=ToyTok({f"t{j}": j for j in range(Vt)}), results=results,
             expect_zero=True)

    # ---------- E3: teacher MUCH longer than student (5 tea tokens -> 1 stu) ----------
    a = faithful_args(); c = mock_cfg()
    stu_off = [(0, 10)]
    tea_off = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    run_case("E3 tea>>stu (1<-5)", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[7], tea_ids=[1, 2, 3, 4, 5], stu_offsets=stu_off, tea_offsets=tea_off,
             s_answer=mk_logits(1, Vs, gen=gen), t_answer=mk_logits(5, Vt, gen=gen),
             stu_tok=ToyTok({f"s{i}": i for i in range(Vs)}),
             tea_tok=ToyTok({f"t{j}": j for j in range(Vt)}), results=results)

    # ---------- E4: student MUCH longer than teacher (5 stu -> 1 tea) ----------
    a = faithful_args(); c = mock_cfg()
    run_case("E4 stu>>tea (5->1)", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[1, 2, 3, 4, 5], tea_ids=[7], stu_offsets=tea_off, tea_offsets=stu_off,
             s_answer=mk_logits(5, Vs, gen=gen), t_answer=mk_logits(1, Vt, gen=gen),
             stu_tok=ToyTok({f"s{i}": i for i in range(Vs)}),
             tea_tok=ToyTok({f"t{j}": j for j in range(Vt)}), results=results)

    # ---------- E5: all-mass-on-one-token logits (+-30) ----------
    a = faithful_args(); c = mock_cfg()
    run_case("E5 spike +-30", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[3, 9], tea_ids=[4, 8], stu_offsets=[(0, 3), (3, 6)],
             tea_offsets=[(0, 3), (3, 6)],
             s_answer=mk_logits(2, Vs, gen=gen, spike=(3, 30.0)),
             t_answer=mk_logits(2, Vt, gen=gen, spike=(4, 30.0)),
             stu_tok=ToyTok({f"s{i}": i for i in range(Vs)}),
             tea_tok=ToyTok({f"t{j}": j for j in range(Vt)}), results=results)

    # ---------- E6: duplicate token ids inside a merged group ----------
    # multi-token group where the tail tokens repeat the same id -> squared prob.
    a = faithful_args(); c = mock_cfg()
    stu_off = [(0, 2), (2, 4), (4, 6)]   # 3 student tokens, one group (shared end at 6)
    tea_off = [(0, 6)]                    # 1 teacher token same span
    run_case("E6 duplicate ids merge", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[5, 5, 5], tea_ids=[9], stu_offsets=stu_off, tea_offsets=tea_off,
             s_answer=mk_logits(3, Vs, gen=gen), t_answer=mk_logits(1, Vt, gen=gen),
             stu_tok=ToyTok({f"s{i}": i for i in range(Vs)}),
             tea_tok=ToyTok({f"t{j}": j for j in range(Vt)}), results=results)

    # ---------- E7: empty alignment groups (offsets never share a boundary) ----------
    a = faithful_args(); c = mock_cfg()
    # student ends at 3,7 ; teacher ends at 5,9 -> no shared interior boundary until end.
    run_case("E7 no-shared-boundary", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[1, 2], tea_ids=[3, 4], stu_offsets=[(0, 3), (3, 7)],
             tea_offsets=[(0, 5), (5, 9)],
             s_answer=mk_logits(2, Vs, gen=gen), t_answer=mk_logits(2, Vt, gen=gen),
             stu_tok=ToyTok({f"s{i}": i for i in range(Vs)}),
             tea_tok=ToyTok({f"t{j}": j for j in range(Vt)}), results=results)

    # ---------- E8/E9: hybrid beta extremes (matched-weight only) ----------
    for beta in (0.0, 1.0):
        a = faithful_args(hybrid=True, matched_w=1.0, unmatched_w=0.0, beta=beta)
        c = mock_cfg(hybrid=True, matched_w=1.0, unmatched_w=0.0, beta=beta)
        stu_tok, tea_tok = hybrid_toks(Vs, Vt, [(3, 4), (9, 8), (1, 2)])
        run_case(f"E8 hybrid matched beta={beta}", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
                 stu_ids=[3, 9], tea_ids=[4, 8], stu_offsets=[(0, 3), (3, 6)],
                 tea_offsets=[(0, 3), (3, 6)],
                 s_answer=mk_logits(2, Vs, gen=gen), t_answer=mk_logits(2, Vt, gen=gen),
                 stu_tok=stu_tok, tea_tok=tea_tok, results=results)

    # ---------- E10: hybrid unmatched, beta irrelevant ----------
    a = faithful_args(hybrid=True, matched_w=0.0, unmatched_w=1.0, beta=0.5)
    c = mock_cfg(hybrid=True, matched_w=0.0, unmatched_w=1.0, beta=0.5)
    stu_tok, tea_tok = hybrid_toks(Vs, Vt, [(3, 4)])
    run_case("E10 hybrid unmatched", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[3, 9], tea_ids=[4, 8], stu_offsets=[(0, 3), (3, 6)],
             tea_offsets=[(0, 3), (3, 6)],
             s_answer=mk_logits(2, Vs, gen=gen), t_answer=mk_logits(2, Vt, gen=gen),
             stu_tok=stu_tok, tea_tok=tea_tok, results=results)

    # ---------- E11/E12/E13: chunk-boundary 31/32/33 groups vs chunk=32 ----------
    for ngrp in (31, 32, 33):
        a = faithful_args(chunk=32); c = mock_cfg(chunk=32)
        stu_off = [(k * 2, k * 2 + 2) for k in range(ngrp)]
        tea_off = [(k * 2, k * 2 + 2) for k in range(ngrp)]
        sid = [(k % Vs) for k in range(ngrp)]
        tid = [((k + 1) % Vt) for k in range(ngrp)]
        run_case(f"E11 chunk-bound n={ngrp}", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
                 stu_ids=sid, tea_ids=tid, stu_offsets=stu_off, tea_offsets=tea_off,
                 s_answer=mk_logits(ngrp, Vs, gen=gen), t_answer=mk_logits(ngrp, Vt, gen=gen),
                 stu_tok=ToyTok({f"s{i}": i for i in range(Vs)}),
                 tea_tok=ToyTok({f"t{j}": j for j in range(Vt)}), results=results)

    # ---------- E14: both temperatures 0.7 (uld) ----------
    a = faithful_args(stu_temp=0.7, tea_temp=0.7); c = mock_cfg(stu_temp=0.7, tea_temp=0.7)
    run_case("E14 temp=0.7 uld", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[3, 9, 2], tea_ids=[4, 8], stu_offsets=[(0, 2), (2, 4), (4, 6)],
             tea_offsets=[(0, 3), (3, 6)],
             s_answer=mk_logits(3, Vs, gen=gen), t_answer=mk_logits(2, Vt, gen=gen),
             stu_tok=ToyTok({f"s{i}": i for i in range(Vs)}),
             tea_tok=ToyTok({f"t{j}": j for j in range(Vt)}), results=results)

    # ---------- E15: both temperatures 0.7 (hybrid matched) ----------
    a = faithful_args(hybrid=True, matched_w=1.0, unmatched_w=0.0, beta=0.5,
                      stu_temp=0.7, tea_temp=0.7)
    c = mock_cfg(hybrid=True, matched_w=1.0, unmatched_w=0.0, beta=0.5,
                 stu_temp=0.7, tea_temp=0.7)
    stu_tok, tea_tok = hybrid_toks(Vs, Vt, [(3, 4), (9, 8)])
    run_case("E15 temp=0.7 hybrid-m", args=a, cfg=c, P=P, Vs=Vs, Vt=Vt,
             stu_ids=[3, 9], tea_ids=[4, 8], stu_offsets=[(0, 3), (3, 6)],
             tea_offsets=[(0, 3), (3, 6)],
             s_answer=mk_logits(2, Vs, gen=gen), t_answer=mk_logits(2, Vt, gen=gen),
             stu_tok=stu_tok, tea_tok=tea_tok, results=results)

    # ---------- E16: PRE-FIX SHIFT negative control -- gold-matched NON-FAITHFUL ----------
    # Explicitly exercise the PRE-FIDELITY-FIX path: ours with --gold-trl-faithful DISABLED
    # and a NON-observed (shifted) merge, so ours takes the shifted next-token window
    # [P2-1:P2-1+Rg] + NO 1e-8 clamp while TRL uses the unshifted observed window
    # [P2:P2+Rg].  This MUST diverge (> E16_MIN_DIVERGENCE); a small rel means the
    # windowing-shift bug is silently masked and is a genuine FAIL of this control.
    E16_MIN_DIVERGENCE = 1e-2
    a_cfg = H.make_arm_args("gold-matched")
    a_cfg.gold_trl_faithful = False               # <-- PRE-FIX: disable faithful windowing
    a_cfg.gold_uld_token_merge_strategy = "shifted"  # <-- non-observed merge (shifted window)
    c = mock_cfg(hybrid=True, matched_w=1.0, unmatched_w=0.0, beta=0.5)
    stu_tok, tea_tok = hybrid_toks(Vs, Vt, [(3, 4), (9, 8)])
    P2 = 2
    Rg = 3
    sid = [3, 9, 2]; tid = [4, 8, 1]
    soff = [(0, 2), (2, 4), (4, 6)]; toff = [(0, 2), (2, 4), (4, 6)]
    gg = torch.Generator().manual_seed(999)
    sfull = mk_logits(P2 + Rg + 1, Vs, gen=gg)
    tfull = mk_logits(P2 + Rg + 1, Vt, gen=gg)
    # TRL uses unshifted answer window [P2:P2+Rg]; as-configured ours uses shifted [P2-1:P2-1+Rg]
    trl = trl_scalar(c, sfull=sfull, tfull=tfull, stu_ids=sid, tea_ids=tid,
                     stu_offsets=soff, tea_offsets=toff, P=P2, stu_tok=stu_tok, tea_tok=tea_tok)
    case = dict(Vs=Vs, Vt=Vt, Vs_dim=Vs, Vt_dim=Vt, P=P2, R=Rg, Rt=Rg,
                stu_ids=sid, tea_ids=tid, stu_offsets=soff, tea_offsets=toff,
                sfull=sfull, tfull=tfull, stu_tok=stu_tok, tea_tok=tea_tok, eos_s=0, eos_t=0)
    hv = build_hv(a_cfg, stu_tok, tea_tok, Vs, Vt)
    ours_shift = float(H.ours_gold_sample_scalar(
        a_cfg, stu_answer_logits=H.window_student(a_cfg, case),
        tea_answer_logits=H.window_teacher(a_cfg, case),
        stu_ids=sid, tea_ids=tid, stu_offsets=soff, tea_offsets=toff, hybrid_vocab=hv))
    r = rel(trl, ours_shift)
    e16_rel = r
    results.append(("E16 gold-matched PRE-FIX-SHIFT ctrl", trl, ours_shift, r,
                    "DIVERGES(expected)" if r > E16_MIN_DIVERGENCE else "FAIL(unexpectedly-close)"))

    # ---------- E17: same, but ours forced faithful -> should match ----------
    a_faith = H.make_arm_args("gold-matched")
    a_faith.gold_trl_faithful = True
    a_faith.gold_uld_token_merge_strategy = "observed"
    hv = build_hv(a_faith, stu_tok, tea_tok, Vs, Vt)
    ours_faith = float(H.ours_gold_sample_scalar(
        a_faith, stu_answer_logits=H.window_student(a_faith, case),
        tea_answer_logits=H.window_teacher(a_faith, case),
        stu_ids=sid, tea_ids=tid, stu_offsets=soff, tea_offsets=toff, hybrid_vocab=hv))
    r = rel(trl, ours_faith)
    results.append(("E17 gold-matched FORCED-FAITHFUL", trl, ours_faith, r,
                    "PASS" if r < 1e-5 else "FAIL"))

    # ---------- E18: ADAPTIVE hybrid weights at extreme overlap fractions ----------
    # gold-hybrid arm (TRL-default): NO matched/unmatched weight flags -> BOTH sides
    # resolve the weights from the vocabulary overlap:
    #     matched_weight   = matched_count / teacher_vocab_size
    #     unmatched_weight = 1 - matched_weight
    #   gold_kernels.py:761-765 (matched_count=:678, teacher_vocab_size=:677)
    #   TRL gold_trainer.py:934-936 (extended-hybrid streaming path; denom =
    #        teacher_answer_logits.size(-1), which == Vt here since no lm-head padding)
    # Drive matched_count to its EXTREMES (0, 1, ~all of teacher vocab) from the SAME
    # toy vocab split fed to both sides, and assert the resolved adaptive weights agree
    # at the extremes (plus loss parity where TRL can construct).
    E18_TOL = 1e-5
    Vs_h, Vt_h = 12, 10
    P_h = 2
    sid_h = [0, 1]; tid_h = [0, 1]
    soff_h = [(0, 3), (3, 6)]; toff_h = [(0, 3), (3, 6)]
    e18 = {}
    for label, n_match in (("m=0", 0), ("m=1", 1), (f"m=all({Vt_h})", Vt_h)):
        # n_match distinct teacher ids share a student token STRING (matched pairs).
        matched_pairs = [(j % Vs_h, j) for j in range(n_match)]
        stu_tok, tea_tok = hybrid_toks(Vs_h, Vt_h, matched_pairs)
        a = faithful_args(hybrid=True, matched_w=None, unmatched_w=None, beta=0.5)  # adaptive
        c = mock_cfg(hybrid=True, matched_w=None, unmatched_w=None, beta=0.5)
        hv = build_hv(a, stu_tok, tea_tok, Vs_h, Vt_h)
        ours_mw = int(hv["matched_count"]) / max(int(hv["teacher_vocab_size"]), 1)  # ours adaptive matched_weight
        ge = torch.Generator().manual_seed(_fixed_seed("E18 " + label))
        s_ans = mk_logits(2, Vs_h, gen=ge); t_ans = mk_logits(2, Vt_h, gen=ge)
        try:
            uld = O.ULDLoss(c, student_tokenizer=stu_tok, teacher_tokenizer=tea_tok,
                            device=torch.device("cpu"))
            trl_mw = len(uld._teacher_matched_ids) / Vt_h        # TRL adaptive matched_weight
            sfull = torch.cat([mk_logits(P_h, Vs_h, gen=ge), s_ans, mk_logits(1, Vs_h, gen=ge)], 0)
            tfull = torch.cat([mk_logits(P_h, Vt_h, gen=ge), t_ans, mk_logits(1, Vt_h, gen=ge)], 0)
            trl_loss = trl_scalar(c, sfull=sfull, tfull=tfull, stu_ids=sid_h, tea_ids=tid_h,
                                  stu_offsets=soff_h, tea_offsets=toff_h, P=P_h,
                                  stu_tok=stu_tok, tea_tok=tea_tok)
            ours_loss = ours_scalar(a, stu_answer_logits=s_ans, tea_answer_logits=t_ans,
                                  stu_ids=sid_h, tea_ids=tid_h, stu_offsets=soff_h,
                                  tea_offsets=toff_h, hybrid_vocab=hv)
            w_ok = abs(ours_mw - trl_mw) < 1e-12
            l_rel = rel(trl_loss, ours_loss)
            verdict = "PASS" if (w_ok and l_rel < E18_TOL) else "FAIL"
            results.append((f"E18 adaptive {label} w={ours_mw:.4f}", trl_loss, ours_loss, l_rel, verdict))
        except Exception as exc:
            # TRL's _initialize_vocabulary_mapping (gold_trainer.py:413, max() of an EMPTY
            # vocab_mapping) REJECTS a zero-overlap hybrid at CONSTRUCTION -> TRL cannot run
            # matched_count=0.  ours degrades gracefully to matched_weight=0 (pure unmatched).
            trl_mw = f"ERR:{type(exc).__name__}"
            w_ok = (ours_mw == 0.0)
            verdict = "TRL-CANT-BUILD/ours-w=0" if w_ok else "FAIL"
            results.append((f"E18 adaptive {label} w={ours_mw:.4f}", trl_mw, "n/a", None, verdict))
        e18[label] = (ours_mw, trl_mw)

    # ------------------------------------------------------------------ #
    print("=" * 92)
    print("EDGE-CASE HARNESS  TRL GOLD/ULD vs ours (faithful/kernel-aligned unless noted)")
    print("=" * 92)
    print(f"{'case':<38}{'TRL':>16}{'ours':>16}{'rel':>12}   verdict")
    print("-" * 92)
    n_fail = 0
    for name, trl, ours, r, v in results:
        ts = f"{trl:.8f}" if isinstance(trl, float) else trl
        ks = f"{ours:.8f}" if isinstance(ours, float) else ours
        rs = f"{r:.2e}" if isinstance(r, float) else "-"
        print(f"{name:<38}{ts:>16}{ks:>16}{rs:>12}   {v}")
        if v.startswith("FAIL") or v == "ONE-ERR":
            n_fail += 1
    print("-" * 92)
    print(f"kernel FAILs / one-sided ERRs: {n_fail}")
    # Hard assertion for the E16 negative control: the pre-fix shifted/non-observed
    # window MUST diverge from TRL.  If it does not, the shift bug is masked.
    assert e16_rel > E16_MIN_DIVERGENCE, (
        f"E16 negative control FAILED: pre-fix shifted path rel={e16_rel:.3e} "
        f"<= {E16_MIN_DIVERGENCE:.0e}; the windowing-shift divergence is not being exercised."
    )
    print(f"E16 negative-control divergence OK: rel={e16_rel:.3e} > {E16_MIN_DIVERGENCE:.0e}")

    # E18 adaptive-weight extremes: hard parity assertions at the overlap boundaries.
    all_label = f"m=all({Vt_h})"
    assert e18["m=0"][0] == 0.0, (
        f"E18 m=0 FAILED: ours adaptive matched_weight != 0 ({e18['m=0'][0]}); "
        f"zero overlap must give matched_weight=0 (pure unmatched)."
    )
    assert abs(e18["m=1"][0] - 1.0 / Vt_h) < 1e-12 and e18["m=1"][0] == e18["m=1"][1], (
        f"E18 m=1 adaptive-weight parity FAILED: ours={e18['m=1'][0]} trl={e18['m=1'][1]} "
        f"(expected 1/{Vt_h})."
    )
    assert e18[all_label][0] == 1.0 and e18[all_label][0] == e18[all_label][1], (
        f"E18 m=all adaptive-weight parity FAILED: ours={e18[all_label][0]} trl={e18[all_label][1]} "
        f"(expected 1.0)."
    )
    print(f"E18 adaptive-weight extremes OK: m=0 w={e18['m=0'][0]:.4f} (TRL cant-build, ours->0); "
          f"m=1 w={e18['m=1'][0]:.4f}==TRL; {all_label} w={e18[all_label][0]:.4f}==TRL")
    print("DONE")


if __name__ == "__main__":
    main()
