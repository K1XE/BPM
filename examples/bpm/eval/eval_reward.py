"""Eval-only custom RM for Path-A code benchmarks (MBPP+, LiveCodeBench).

Path A (`train.py --debug-rollout-only`) scores eval samples through the rollout
reward hub. Wire this module in with:

    --custom-rm-path examples.bpm.eval.eval_reward.reward_func --reward-key acc --eval-reward-key acc

Each sample must carry `metadata.rm_type == "code_sandbox"` plus a `metadata.code_grader`
tag (see examples/bpm/eval/prepare_code_eval_data.py):

    code_grader="mbpp" : score_mbpp(response, metadata["test"])
    code_grader="lcb"  : score_lcb(response, metadata["problem"])   # {testtype,tests,func_name}
    code_grader="bcb"  : score_bcb(response, metadata["test"])       # unittest suite (BCB-Hard)

WHY a separate module (not reward_adapter.py): the training reward adapter's
`code_sandbox` branch only calls `score_taco` (TACO {inputs,outputs,fn_name}). MBPP+
(assert test body) and LCB (stdin/functional) need their own graders. Keeping this in a
dedicated eval module leaves the training reward path byte-for-byte unchanged.

Return shape mirrors the stock dapo RM (`{"acc": bool, "score": float}`) so the eval
pass@k path reads `sample.reward["acc"]` under `--reward-key acc` exactly like the math
benchmarks. The code graders (examples/bpm/reward/code_eval.py) already run each candidate inside
a killable subprocess sandbox (RLIMIT + SIGKILL on timeout), so a student infinite-loop
cannot wedge the reward path; we only move the blocking call off the event loop with a
thread so many samples grade concurrently.
"""
import asyncio
import logging
import os
import sys
from typing import List

from slime.utils.types import Sample

logger = logging.getLogger("slime")




def _grade_one(sample: Sample, args) -> dict:
    """Blocking: grade one code sample in the sandbox. Returns {"acc","score","pred"}."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    grader = md.get("code_grader")
    response = sample.response or ""
    is_code = grader in ("mbpp", "lcb", "taco", "bcb") or md.get("rm_type") == "code_sandbox"
    ok = False
    try:
        if not is_code:
            # MATH (aime24/aime25/math500): marker-based verifier aligned to the OFFICIAL graders
            # (lm-eval minerva, lighteval math_verify). Extracts the answer from the LAST \boxed{} /
            # "Answer:" marker (NO last-number fallback, NO greedy full-text parse) then checks
            # equivalence via DeepScaler sympy + math_verify on that span, with a percent guard.
            # Credits 1/2==0.5, \frac{408}{2}==204, 2\sqrt2, (2,3) equivalence while rejecting stray
            # -number false-positives ("give up 34"==34, "50%"==50). Verified zero-bias on real
            # AIME24 rollouts. Replaces the old --rm-type dapo path that UNDER-scored MATH-500.
            from examples.bpm.reward.math_verifier import check_math
            ok = bool(check_math(response, str(sample.label if sample.label is not None else "")))
        elif not md.get("has_tests"):
            ok = False  # ungradeable row (no recoverable tests) -> 0, same as training
        elif grader == "mbpp":
            from examples.bpm.reward.code_eval import score_mbpp
            ok = bool(score_mbpp(response, md.get("test", ""), timeout=90.0, mem_mb=4096))
        elif grader == "lcb":
            from examples.bpm.reward.code_eval import score_lcb
            # timeout=6.0 per test matches the official lcb_runner signal.alarm(6); score_lcb now
            # dispatches functional-vs-stdin on problem["func_name"] presence (official semantics).
            ok = bool(score_lcb(response, md.get("problem") or {}, timeout=6.0))
        elif grader == "bcb":
            from examples.bpm.reward.code_eval import score_bcb
            # BigCodeBench-Hard: unittest suite, unhardened sandbox + pylibs deps + Agg backend;
            # mem_mb=16384 is a virtual-address cap (scipy/matplotlib thread stacks EAGAIN at 4G).
            ok = bool(score_bcb(response, md.get("test", "")))
        elif grader == "taco":
            from examples.bpm.reward.code_eval import score_taco
            _t = md.get("tests") or {}
            if isinstance(_t, str):  # taco_test_em stores tests as a JSON string (bigint-safe)
                import json as _json
                _t = _json.loads(_t)
            ok = bool(score_taco(response, _t))
        else:
            logger.warning("[eval_reward] sample missing/unknown code_grader=%r; scoring 0", grader)
            ok = False
    except Exception as e:  # a broken grader/payload must not crash the whole eval
        logger.warning("[eval_reward] grader=%r raised %r -> acc 0", grader, e)
        ok = False
    return {"acc": ok, "score": 1.0 if ok else 0.0, "pred": None}


async def reward_func(args, samples, **kwargs):
    """Custom RM entry (batch + single). Returns {"acc","score","pred"} per sample."""
    is_single = isinstance(samples, Sample)
    sample_list = [samples] if is_single else list(samples)
    # to_thread: the graders block on subprocess.communicate; running them in threads keeps
    # the rollout event loop responsive and lets the sandbox spawns overlap.
    results = await asyncio.gather(*[asyncio.to_thread(_grade_one, s, args) for s in sample_list])
    return results[0] if is_single else list(results)
