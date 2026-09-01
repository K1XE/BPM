"""Reward adapter for OPD rollout/eval runs.

RolloutManager owns the OPD teacher lifecycle: it offloads rollout engines,
wakes the teacher, runs prefill-only hidden-state extraction, sleeps the
teacher, and injects ``sample.teacher_hidden_states`` when needed.  This module must not create another teacher service.

``reward_func`` computes real task rewards through the configured ``--rm-type``
so rollout/eval/pass-rate metrics remain meaningful. ``post_process_rewards``
keeps standalone OPD gradient rewards at zero, while coupled OPD modes return
real rewards so normal GRPO/PPO advantage computation remains available before
policy_loss applies the requested OPD coupling.

Usage:
  --custom-rm-path examples.bpm.reward.reward_adapter.reward_func
  --custom-reward-post-process-path examples.bpm.reward.reward_adapter.post_process_rewards
  --rm-type dapo|deepscaler|math|gpqa|...
  --advantage-estimator grpo
  --loss-type policy_loss
  --opd-mode standalone_loss|aux_loss|pg_advantage
  --opd-backend bpm
  --opd-teacher-model-path <teacher-hf-path>
"""

import logging
from typing import List

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _strip_terminal_special_tokens(text: str, args) -> str:
    """Remove terminal chat/eos special-token text before rule-based grading."""
    if not text:
        return ""

    specials = ["<|im_end|>", "<|endoftext|>"]
    tokenizer = getattr(args, "tokenizer", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        specials.insert(0, eos_token)

    out = text
    changed = True
    while changed:
        changed = False
        stripped = out.rstrip()
        for special in specials:
            if special and stripped.endswith(special):
                out = stripped[: -len(special)]
                changed = True
                break
    return out


import asyncio as _asyncio
import concurrent.futures as _futures

# ONE shared, size-capped pool for math_verify -- NOT a fresh ThreadPoolExecutor per call (256/step
# churn) and NOT unbounded. A hung sympy occupies at most one of these workers (the 200-char guard
# in the caller makes a true infinite hang unreachable in practice; the watchdog catches any stall).
_MV_EXEC = None


def _mv_exec():
    global _MV_EXEC
    if _MV_EXEC is None:
        _MV_EXEC = _futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="mv")
    return _MV_EXEC


def _import_math_verify():
    """Import math_verify, falling back to the SHARED ceph copy when a node has no system install.

    math_verify is installed PER-NODE, but a run executes on whichever node it was launched from.
    A run that landed on a node without it produced HOURS of silently-zero raw_reward, because the
    ImportError was swallowed into acc=False (pred=='label' still scored wrong). Try the system copy
    first; then the shared /home copy (works on every node without per-node installs); then FAIL LOUD.
    sympy is already in sys.modules (torch imported it at startup), so the shared path only supplies
    math_verify itself -- it does NOT shadow the node's sympy."""
    try:
        from math_verify import parse, verify
        return parse, verify
    except ImportError:
        import sys, os
        _shared = __import__("os").environ.get("BPM_REWARD_PYLIBS", "")
        if os.path.isdir(_shared) and _shared not in sys.path:
            sys.path.insert(0, _shared)
        try:
            from math_verify import parse, verify
            return parse, verify
        except ImportError as e:
            import socket, logging
            logging.getLogger("slime").critical(
                "[OPD][deepmath] math_verify NOT importable on host=%s (nor shared %s): %r. "
                "raw_reward would be silently 0 for the ENTIRE run -- failing loud instead. "
                "Install it: pip install math_verify (on this node, or into %s).",
                socket.gethostname(), _shared, e, _shared)
            raise


class _MVWorker:
    """One killable math_verify worker PROCESS (see examples/bpm/reward/mv_worker.py for why).

    A thread-based verify with sympy's internal timeouts disabled froze three earlier rollout runs:
    a model-generated pred (short enough to pass the 200-char guard, e.g. a huge-power latex)
    sank sympy into a GIL-holding C computation; asyncio.wait_for cannot interrupt that, so the
    ENTIRE RolloutManager process (event loop, wandb, everything) froze mid-rollout with engines
    idle and zero errors. A process can be SIGKILLed; a thread cannot."""

    def __init__(self):
        import multiprocessing as _mp
        import os as _os
        # spawn children do NOT inherit parent sys.path — make sure they can import
        # examples.bpm.reward.mv_worker regardless of how the parent got its path set up.
        _repo_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        _pp = _os.environ.get("PYTHONPATH", "")
        if _repo_root not in _pp.split(":"):
            _os.environ["PYTHONPATH"] = f"{_repo_root}:{_pp}" if _pp else _repo_root
        ctx = _mp.get_context("spawn")  # never fork a threaded ray actor
        self.conn, child = ctx.Pipe()
        from examples.bpm.reward import mv_worker as _mvw
        self.proc = ctx.Process(target=_mvw.worker_main, args=(child,), daemon=True)
        self.proc.start()
        child.close()

    def wait_ready(self, timeout: float = 45.0) -> None:
        # first reply is the "ready" handshake (child imports math_verify; shared-ceph fallback time)
        if not self.conn.poll(timeout) or self.conn.recv()[0] != "ready":
            self.kill()
            raise RuntimeError("math_verify worker failed to start (math_verify importable?)")

    def verify(self, label: str, response: str, timeout: float):
        """Full reward path (extract+guards+parse+verify) in the child.

        Returns (acc, pred) or None on timeout (caller must kill+replace this worker)."""
        self.conn.send((label, response))
        if self.conn.poll(timeout):
            _st, acc, pred = self.conn.recv()
            return bool(acc), pred
        return None

    def kill(self):
        try:
            self.proc.kill()
            self.proc.join(2)
            self.conn.close()
        except Exception:
            pass


_MV_POOL = None  # queue.Queue[_MVWorker]
_MV_POOL_SIZE = 8


def _mv_pool():
    global _MV_POOL
    if _MV_POOL is None:
        import queue as _queue
        _MV_POOL = _queue.Queue()
        workers = [_MVWorker() for _ in range(_MV_POOL_SIZE)]  # start all in parallel...
        for w in workers:
            w.wait_ready()                                      # ...then handshake
            _MV_POOL.put(w)
    return _MV_POOL


def _mv_verify_sync(label: str, response: str, timeout: float = 8.0):
    """Blocking full reward path with a HARD, GIL-proof wall-clock cap (runs in an _MV_EXEC thread).

    On timeout the stuck worker process is killed and replaced — the computation dies with it,
    so a pathological sample costs one worker respawn instead of freezing the training job.
    Returns (acc, pred)."""
    pool = _mv_pool()
    w = pool.get()
    try:
        r = w.verify(label, response, timeout)
    except Exception:
        r = None
    if r is None:
        w.kill()
        try:
            nw = _MVWorker()
            nw.wait_ready()
            pool.put(nw)
        except Exception:
            # keep pool size honest even if respawn fails once; next call will retry
            import logging
            logging.getLogger("slime").critical("[OPD][deepmath] mv worker respawn FAILED")
            raise
        c = getattr(_mv_verify_sync, "_n_kill", 0) + 1
        _mv_verify_sync._n_kill = c
        if c & (c - 1) == 0:  # powers of two
            import logging
            logging.getLogger("slime").warning(
                "[OPD][deepmath] reward path exceeded %.1fs -> worker KILLED+respawned (#%d) label=%r resp_tail=%r",
                timeout, c, str(label)[:60], (response or "")[-80:])
        return False, ""
    pool.put(w)
    return r


async def _deepmath_verify_async(label: str, response: str, timeout: float = 8.0):
    """NON-BLOCKING, GIL-proof deepmath reward: extraction + math_verify in a KILLABLE process.

    History of this function (each iteration fixed a production incident):
      v1 thread + math_verify signal timeouts  -> signal.alarm raises off-main-thread: all-zero rewards.
      v2 thread + timeouts disabled + wait_for -> a generated pred (e.g. ``10^{10^{9}}``) sank sympy
         into a GIL-holding C computation; wait_for cannot interrupt it; the WHOLE RolloutManager
         (event loop, wandb, all threads) froze mid-rollout — three DeepMath runs hung at rollout 45,
         whose slice contains a prompt (label ``2^{2006}``) that reliably elicits such preds.
      v3 (this): the entire per-sample reward computation runs in a worker PROCESS that is simply
         SIGKILLed+respawned on wall-clock timeout. Also moves the extraction regex (measured ~2s on
         degenerate 80k-char responses) OFF the event loop. Returns (acc, pred); timeout -> (False, "").
    """
    loop = _asyncio.get_running_loop()
    # _mv_verify_sync enforces the REAL cap by killing its worker process (GIL-proof). The outer
    # wait_for is only a loose belt for pool contention / respawn stalls — NOT the primary guard.
    fut = loop.run_in_executor(_mv_exec(), _mv_verify_sync, label, response, timeout)
    try:
        acc, pred = await _asyncio.wait_for(fut, timeout=timeout * 4 + 120)
        return bool(acc), pred
    except _asyncio.TimeoutError:
        c = getattr(_deepmath_verify_async, "_n_timeout", 0) + 1
        _deepmath_verify_async._n_timeout = c
        if c & (c - 1) == 0:  # powers of two -> visible but not spammy
            import logging
            logging.getLogger("slime").warning(
                "[OPD][deepmath] reward OUTER belt tripped (%.1fs) (#%d) label=%r -> 0",
                timeout * 4 + 120, c, str(label)[:60])
        return False, ""
    except ImportError:
        # A MISSING package must crash loud, never masquerade as a wrong answer (observed in practice:
        # hours of silent acc=0). The worker handshake already failed loud in that case.
        raise
    except Exception as e:
        # A genuine error on ONE sample -> 0, but LOG it (was silently swallowed before,
        # which is exactly how a systematic grader failure could hide as "model is just wrong").
        c = getattr(_deepmath_verify_async, "_n_err", 0) + 1
        _deepmath_verify_async._n_err = c
        if c & (c - 1) == 0:  # powers of two
            import logging
            logging.getLogger("slime").warning(
                "[OPD][deepmath] reward error (#%d) label=%r: %r -> 0", c, str(label)[:60], e)
        return False, ""


# ---------------------------------------------------------------------------
# DEFENSE-IN-DEPTH: generic killable-worker reward path
# ---------------------------------------------------------------------------
# The deepmath branch above (_deepmath_verify_async) has run each per-sample
# reward in a SIGKILL-able spawn PROCESS ever since a GIL-holding sympy
# computation froze three runs whole. The OTHER rule-based branches (dapo,
# deepscaler, math, f1, gpqa, ifbench) used to run INLINE on the event loop —
# and one of them (dapo's O(n^2) normalize_final_answer regex) froze the whole
# run. This generalizes the deepmath mechanism so ANY
# picklable (fn, args) reward computation gets the same GIL-proof wall-clock
# cap. It is a SEPARATE pool/executor from the deepmath machinery above, so the
# deepmath branch behavior is entirely unchanged.
class _KWorker:
    """One GENERIC killable reward worker PROCESS (see examples/bpm/reward/killable_worker.py).

    Unlike _MVWorker (which pre-imports math_verify for the deepmath path), this
    runs an ARBITRARY picklable (fn, args): the fn's module imports lazily in the
    child on first use. Same spawn + SIGKILL-on-timeout contract — a thread or
    asyncio.wait_for cannot interrupt a GIL-holding regex/sympy call, a SIGKILL can."""

    def __init__(self):
        import multiprocessing as _mp
        import os as _os
        # spawn children do NOT inherit parent sys.path — make sure they can import
        # examples.bpm.reward.killable_worker regardless of how the parent set its path up.
        _repo_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        _pp = _os.environ.get("PYTHONPATH", "")
        if _repo_root not in _pp.split(":"):
            _os.environ["PYTHONPATH"] = f"{_repo_root}:{_pp}" if _pp else _repo_root
        ctx = _mp.get_context("spawn")  # never fork a threaded ray actor
        self.conn, child = ctx.Pipe()
        from examples.bpm.reward import killable_worker as _kw
        self.proc = ctx.Process(target=_kw.worker_main, args=(child,), daemon=True)
        self.proc.start()
        child.close()

    def wait_ready(self, timeout: float = 45.0) -> None:
        # first reply is the "ready" handshake (spawn + main-module fixup import time)
        if not self.conn.poll(timeout) or self.conn.recv()[0] != "ready":
            self.kill()
            raise RuntimeError("killable reward worker failed to start")

    def run(self, fn, args, timeout: float):
        """Send (fn, args); return ('ok'|'err', payload, extra) or None on wall-clock timeout.

        Raises only if pickling/sending fn+args fails (caller kills+replaces the worker)."""
        self.conn.send((fn, args))
        if self.conn.poll(timeout):
            return self.conn.recv()
        return None

    def kill(self):
        try:
            self.proc.kill()
            self.proc.join(2)
            self.conn.close()
        except Exception:
            pass


_KW_POOL = None  # queue.Queue[_KWorker]
_KW_POOL_SIZE = 8
_KW_EXEC = None  # ThreadPoolExecutor that blocks on the worker processes


def _kw_pool():
    global _KW_POOL
    if _KW_POOL is None:
        import queue as _queue
        _KW_POOL = _queue.Queue()
        workers = [_KWorker() for _ in range(_KW_POOL_SIZE)]  # start all in parallel...
        for w in workers:
            w.wait_ready()                                     # ...then handshake
            _KW_POOL.put(w)
    return _KW_POOL


def _kw_exec():
    global _KW_EXEC
    if _KW_EXEC is None:
        _KW_EXEC = _futures.ThreadPoolExecutor(max_workers=_KW_POOL_SIZE, thread_name_prefix="kw")
    return _KW_EXEC


def _run_killable_sync(fn, args, timeout: float, default):
    """Blocking: run fn(*args) in a killable worker PROCESS with a HARD wall-clock cap.

    FAST PATH returns fn(*args) BIT-IDENTICAL (round-tripped through the same pickle
    the deepmath path uses). On wall-clock timeout the stuck worker is SIGKILLed and
    replaced and ``default`` is returned. A genuine per-sample grader exception also
    returns ``default`` but is LOGGED (never silently swallowed); a missing-package
    ImportError in the child is re-raised LOUD (a missing grader
    dep must never masquerade as reward 0 for a whole run)."""
    pool = _kw_pool()
    w = pool.get()
    send_failed = False
    try:
        r = w.run(fn, args, timeout)
    except Exception as e:
        # pickling/sending (fn, args) failed, or the pipe broke — recover by
        # killing+respawning the worker and returning the default (logged below).
        r = None
        send_failed = True
        _send_err = e
    if r is None:
        w.kill()
        try:
            nw = _KWorker()
            nw.wait_ready()
            pool.put(nw)
        except Exception:
            import logging
            logging.getLogger("slime").critical("[OPD][reward-guard] worker respawn FAILED")
            raise
        c = getattr(_run_killable_sync, "_n_kill", 0) + 1
        _run_killable_sync._n_kill = c
        if c & (c - 1) == 0:  # powers of two -> visible but not spammy
            import logging
            reason = "send/pickle failed (%r)" % _send_err if send_failed else "exceeded %.1fs" % timeout
            logging.getLogger("slime").warning(
                "[OPD][reward-guard] %s -> worker KILLED+respawned (#%d) fn=%s",
                reason, c, getattr(fn, "__qualname__", fn))
        return default
    pool.put(w)
    status, payload, extra = r
    if status == "ok":
        return payload
    # status == "err": the child raised. payload=type_name, extra=repr(exc).
    if payload in ("ImportError", "ModuleNotFoundError"):
        # crash LOUD: a missing grader dependency must never look like a wrong answer.
        raise RuntimeError(f"[OPD][reward-guard] grader import failed in worker: {extra}")
    c = getattr(_run_killable_sync, "_n_err", 0) + 1
    _run_killable_sync._n_err = c
    if c & (c - 1) == 0:  # powers of two
        import logging
        logging.getLogger("slime").warning(
            "[OPD][reward-guard] grader error (#%d) fn=%s: %s(%s) -> default",
            c, getattr(fn, "__qualname__", fn), payload, extra)
    return default


async def _run_killable(fn, args, default, timeout: float = 8.0):
    """NON-BLOCKING wrapper around _run_killable_sync (mirrors _deepmath_verify_async).

    Offloads the blocking pool round-trip to a thread so the event loop stays
    responsive and per-call caps OVERLAP under gather. The inner worker SIGKILL is
    the REAL guard; the outer wait_for is only a loose belt for pool contention /
    respawn stalls. Returns fn(*args) on the fast path, ``default`` on any cap."""
    loop = _asyncio.get_running_loop()
    fut = loop.run_in_executor(_kw_exec(), _run_killable_sync, fn, args, timeout, default)
    try:
        return await _asyncio.wait_for(fut, timeout=timeout * 4 + 120)
    except _asyncio.TimeoutError:
        c = getattr(_run_killable, "_n_timeout", 0) + 1
        _run_killable._n_timeout = c
        if c & (c - 1) == 0:  # powers of two
            import logging
            logging.getLogger("slime").warning(
                "[OPD][reward-guard] OUTER belt tripped (%.1fs) (#%d) fn=%s -> default",
                timeout * 4 + 120, c, getattr(fn, "__qualname__", fn))
        return default
    except ImportError:
        # A missing package must crash loud, never masquerade as a default reward.
        raise


def _score_asserts_kw(response: str, test_body: str) -> bool:
    """Killable-worker entry: grade a code response against a flat assert/unit-test body.

    For simple function-style code data (MBPP, KodCode-simple) whose grading contract is an
    assert body executed AFTER the generated function in ONE namespace (not TACO I/O cases).
    Reuses the SAME code_eval.score_mbpp already validated on the MBPP+/HumanEval+ evals, so the
    training and eval graders share one path. `test_body` is the flat, import-stripped,
    call-appended assert body baked at dataset-build time (build_kodmbpp.py). Spawned SIGKILL-able
    worker (a student solution can infinite-loop); tools/eval added to sys.path lazily.
    """
    from examples.bpm.reward.code_eval import score_mbpp

    if not test_body:
        return False
    return bool(score_mbpp(response, test_body, timeout=12.0, mem_mb=1024))


def _score_taco_kw(response: str, tests) -> bool:
    """Killable-worker entry: grade a code response against a curated TACO tests payload.

    `tests` is a JSON STRING (metadata.tests is stored stringified so pandas/ujson read_json never
    tries to parse the fn cases' huge competitive-programming integers -> "Value is too big!").
    Bridges the CURATED schema {"type","cases":[{"input","output"}],"fn_name"} to the RAW shape
    code_eval.score_taco expects ({"inputs","outputs","fn_name"}). Passing the curated dict straight
    through would hit score_taco's tests.get("inputs")->None and return False for EVERY code row (the
    exact "code always scores 0" bug this wiring fixes). Runs in a spawned SIGKILL-able worker (a
    student solution can infinite-loop). Imported lazily so the spawned worker resolves it
    through the repo root it inherits on PYTHONPATH.
    """
    import json as _json

    from examples.bpm.reward.code_eval import score_taco

    if isinstance(tests, str):
        tests = _json.loads(tests) if tests else {}
    cases = (tests or {}).get("cases") or []
    if not cases:
        return False
    raw = {
        "inputs": [c.get("input") for c in cases],
        "outputs": [c.get("output") for c in cases],
        "fn_name": tests.get("fn_name"),
    }
    return bool(score_taco(response, raw))


async def _rule_based_reward_no_custom_recursion(args, sample: Sample, **kwargs):
    """Compute the configured rule-based reward without mutating args.custom_rm_path."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or getattr(args, "rm_type", None) or "").strip()
    response = _strip_terminal_special_tokens(sample.response or "", args)
    label = sample.label

    if rm_type.startswith("boxed_"):
        from slime.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer

        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    if rm_type == "remote_rm":
        from slime.rollout.rm_hub import remote_rm

        return await remote_rm(args, sample)
    if rm_type == "deepscaler":
        from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward

        # sympy-backed grader on the event loop -> run in a killable worker (returns int 0/1).
        return await _run_killable(get_deepscaler_rule_based_reward, (response, label), default=0)
    if rm_type == "dapo":
        from slime.rollout.rm_hub.math_dapo_utils import compute_score as compute_score_dapo

        # This is the branch that used to freeze rollouts (O(n^2) normalize_final_answer regex). The primary
        # fix is the solution_str[-2048:] guard in compute_score; this killable-worker wrapper is
        # defense-in-depth. Fast path returns the SAME {score,acc,pred} dict, bit-identical.
        return await _run_killable(
            compute_score_dapo, (response, label),
            default={"score": -1.0, "acc": False, "pred": None},
        )
    if rm_type == "code_sandbox":
        # Cross-tokenizer code KD: grade by EXECUTING the response against the row's recovered TACO
        # test cases (stdin/stdout or fn_name), NOT by math answer-extraction (which returns
        # [INVALID] on every code solution). tests live in metadata.tests; ~5.5% of code rows have
        # no recoverable cases (source shipped none / all oversized) -> acc False (ungradeable, and
        # excluded from the rollout/raw_reward_code mean downstream). Runs in the same SIGKILL-able
        # worker as the other graders; returns the {score,acc,pred} dict shape --reward-key acc reads.
        if not metadata.get("has_tests"):
            return {"score": 0.0, "acc": False, "pred": None}
        r = await _run_killable(
            _score_taco_kw, (response, metadata.get("tests")),
            default={"score": 0.0, "acc": False, "pred": None},
        )
        if isinstance(r, bool):
            return {"score": 1.0 if r else 0.0, "acc": r, "pred": None}
        return r
    if rm_type == "code_asserts":
        # Simple function-style code KD (MBPP / KodCode-simple): grade by EXECUTING the response
        # then the row's flat assert body in one namespace (score_mbpp), not by TACO I/O cases.
        # test body in metadata.test; rows without one are ungradeable -> acc False (excluded from
        # the raw_reward_code mean downstream). Same SIGKILL-able worker + {score,acc,pred} shape.
        if not metadata.get("has_tests") or not metadata.get("test"):
            return {"score": 0.0, "acc": False, "pred": None}
        r = await _run_killable(
            _score_asserts_kw, (response, metadata.get("test")),
            default={"score": 0.0, "acc": False, "pred": None},
        )
        if isinstance(r, bool):
            return {"score": 1.0 if r else 0.0, "acc": r, "pred": None}
        return r
    if rm_type == "deepmath":
        # DeepMath-103K answers are often latex/symbolic (\dfrac, \infty, expressions). The dapo
        # grader only normalizes string forms, so it FALSE-NEGATIVES correct answers in a different
        # form (1/2 vs \frac{1}{2} vs 0.5). Use dapo's robust EXTRACTION (Answer:/boxed) + math_verify
        # (sympy) for robust EQUIVALENCE. Returns the same {score,acc,pred} shape as dapo.
        # The ENTIRE reward computation (extraction regex + guards + math_verify) runs inside a
        # killable worker PROCESS with a hard wall-clock cap — see _deepmath_verify_async for the
        # incident history (a GIL-holding sympy computation on a generated pred froze THREE runs
        # whole at rollout 45; nothing thread- or asyncio-based can interrupt that, SIGKILL can).
        acc, pred = await _deepmath_verify_async(str(label), response, timeout=8.0)
        # One-time-ish sample diagnostic: log the first few (pred,label,acc,response-tail) so a
        # systematic 0 reward can be diagnosed as "model produced no parseable answer" vs "answers
        # wrong" vs "grader bug". Set OPD_DEEPMATH_DEBUG=0 to silence.
        import os as _os
        _dbg = int(_os.environ.get("OPD_DEEPMATH_DEBUG", "8"))
        _seen = getattr(_rule_based_reward_no_custom_recursion, "_dm_dbg", 0)
        if _seen < _dbg:
            _rule_based_reward_no_custom_recursion._dm_dbg = _seen + 1
            import logging
            tail = (response or "")[-120:].replace("\n", "\\n")
            logging.getLogger("slime").warning(
                "[OPD][deepmath-dbg %d] acc=%s pred=%r label=%r resp_tail=%r",
                _seen, acc, pred[:60], str(label)[:60], tail,
            )
        return {"score": 1.0 if acc else -1.0, "acc": acc, "pred": pred}
    if rm_type == "math":
        from slime.rollout.rm_hub.math_utils import grade_answer_verl

        # grade_answer_verl runs sympy -> killable worker. Post-process is unchanged (1/0).
        ok = await _run_killable(grade_answer_verl, (response, label), default=False)
        return 1 if ok else 0
    if rm_type == "f1":
        from slime.rollout.rm_hub.f1 import f1_score

        # f1_score returns (f1, precision, recall); take [0] exactly as before.
        res = await _run_killable(f1_score, (response, label), default=(0.0, 0.0, 0.0))
        return res[0]
    if rm_type == "gpqa":
        from slime.rollout.rm_hub.gpqa import compute_gpqa_reward

        # metadata passed positionally (compute_gpqa_reward(response, label, metadata)).
        return await _run_killable(compute_gpqa_reward, (response, label, metadata), default=0.0)
    if rm_type == "ifbench":
        from slime.rollout.rm_hub.ifbench import compute_ifbench_reward

        return await _run_killable(compute_ifbench_reward, (response, label, metadata), default=0.0)
    if rm_type == "random":
        import random

        return random.randint(0, 1)
    if rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    raise NotImplementedError("Rule-based RM type is not specified.")


async def reward_func(args, samples, **kwargs):
    """Compute real correctness rewards through the configured rule-based RM."""
    rm_type = getattr(args, "rm_type", None)
    is_single = isinstance(samples, Sample)
    sample_list = [samples] if is_single else list(samples)

    if not rm_type:
        return 0.0 if is_single else [0.0] * len(sample_list)

    # Run the per-sample graders CONCURRENTLY (not a sequential await loop): the deepmath grader
    # offloads its sympy work to a bounded pool and awaits it, so with gather the event loop stays
    # responsive and the per-call timeouts OVERLAP. A sequential loop of blocking graders froze the
    # rollout loop for n_samples*timeout (the 8h stall). Order is preserved by gather.
    rewards = await _asyncio.gather(
        *[_rule_based_reward_no_custom_recursion(args, sample, **kwargs) for sample in sample_list]
    )
    return rewards[0] if is_single else list(rewards)


def post_process_rewards(args, samples: List[Sample], **kwargs):
    """Return gradient rewards according to OPD coupling mode.

    Standalone OPD returns zero gradient rewards because the standalone OPD loss
    is the only training signal.  ``aux_loss`` and ``pg_advantage`` return real
    rewards so slime's normal RL advantage path remains available before
    policy_loss adds/replaces the OPD signal.
    """
    n = len(samples)

    missing_hs = sum(1 for s in samples if not hasattr(s, "teacher_hidden_states") or s.teacher_hidden_states is None)
    if missing_hs > 0:
        logger.warning(
            f"[OPD] post_process_rewards: {missing_hs}/{n} samples missing "
            "teacher_hidden_states; RolloutManager should inject them before reward post-processing."
        )

    def _to_float(r):
        if r is None:
            return 0.0
        if isinstance(r, (int, float)):
            return float(r)
        if isinstance(r, dict):
            reward_key = getattr(args, "reward_key", None) or getattr(args, "eval_reward_key", None)
            if reward_key and reward_key in r:
                return float(r[reward_key])
            return float(next(iter(r.values()), 0.0))
        return 0.0

    raw_rewards = [_to_float(getattr(s, "reward", None)) for s in samples]
    n_correct = sum(1 for r in raw_rewards if r > 0)
    avg = sum(raw_rewards) / max(len(raw_rewards), 1)
    logger.info(
        f"[OPD] post_process_rewards: {n} samples, real reward avg={avg:.3f} "
        f"({n_correct}/{n} correct), opd_mode={getattr(args, 'opd_mode', 'off')}"
    )

    if getattr(args, "opd_mode", "off") in ("aux_loss", "pg_advantage"):
        return raw_rewards, raw_rewards
    return raw_rewards, [0.0] * n
