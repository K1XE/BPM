"""Persistent deepmath-reward worker PROCESS (spawn target). Deliberately a tiny standalone
module: the spawn child imports ONLY this file + math_verify + (file-loaded) math_dapo_utils —
never torch/slime packages.

Why a process, not a thread: the deepmath reward path has TWO ways to stall the event loop:
  (1) sympy/math_verify on a pathological generated pred (e.g. ``10^{10^{9}}``, 12 chars) sinks
      into a CPython big-int/C computation that HOLDS THE GIL for hours — every thread in the
      process freezes (measured: a ticker thread got ~0 ticks/18s next to such a verify);
  (2) ``extract_answer_from_solution`` (regex over the FULL response) takes seconds on degenerate
      ~80k-char "Wait.\\nWait.\\n..." responses, and ran ON the event loop.
Observed in production: three DeepMath runs froze whole (engines idle, zero errors, wandb dead)
at rollout 45 — the slice holds a prompt (label ``2^{2006}``) whose degenerate responses trip (1).
``asyncio.wait_for``/thread timeouts cannot interrupt a GIL-holding C call; SIGKILLing a worker
process can. So the ENTIRE per-sample reward computation lives here, in a killable process.
"""
import os
import sys


def _import_math_verify():
    """system copy -> shared copy (per-node installs differ, see reward_adapter)."""
    try:
        from math_verify import parse, verify
        return parse, verify
    except ImportError as first_err:
        shared = os.environ.get("BPM_REWARD_PYLIBS", "")
        if os.path.isdir(shared) and shared not in sys.path:
            sys.path.insert(0, shared)
        try:
            from math_verify import parse, verify
            return parse, verify
        except ImportError as e:
            raise ImportError(
                "math_verify is not importable (math rewards would silently be 0). "
                "Install it with `pip install math-verify` on every node, or point "
                "BPM_REWARD_PYLIBS at a directory containing it."
            ) from (e or first_err)


def _load_math_dapo_utils():
    """Load slime/rollout/rm_hub/math_dapo_utils.py DIRECTLY by file, bypassing the rm_hub
    package __init__ (which drags aiohttp/slime deps into this deliberately-light child)."""
    import importlib.util
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo_root, "slime", "rollout", "rm_hub", "math_dapo_utils.py")
    spec = importlib.util.spec_from_file_location("_standalone_math_dapo_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker_main(conn):
    """Loop: recv (label, full_response) -> send (status, acc, pred_head).

    status: "ok" | "err". Parent kills us on wall-clock timeout (stuck extract/parse/verify)."""
    parse, verify = _import_math_verify()
    mdu = _load_math_dapo_utils()
    conn.send(("ready", True, ""))
    while True:
        try:
            item = conn.recv()
        except (EOFError, OSError):
            return
        if item is None:
            return
        label, response = item
        pred = ""
        try:
            pred = mdu.extract_answer_from_solution(response) or ""
            acc = False
            if pred and pred != "[INVALID]" and len(pred) <= 200 and len(str(label)) <= 200:
                gold = parse("\\boxed{" + str(label) + "}", parsing_timeout=None)
                answer = parse("\\boxed{" + pred + "}", parsing_timeout=None)
                acc = bool(verify(gold, answer, timeout_seconds=None))
            conn.send(("ok", acc, pred[:200]))
        except Exception:
            try:
                conn.send(("err", False, pred[:200]))
            except (BrokenPipeError, OSError):
                return
