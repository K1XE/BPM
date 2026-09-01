"""Generic killable REWARD worker PROCESS (spawn target).

Runs an ARBITRARY picklable ``(fn, args)`` and returns ``fn(*args)``, so a
GIL-holding regex / sympy computation on a pathological model completion costs
one SIGKILL + respawn instead of freezing the whole rollout event loop.

Why a PROCESS, not a thread (learned the hard way from repeated rollout freezes):
  * ``examples/bpm/reward/reward_adapter.py`` used to call the rule-based graders
    (dapo / deepscaler / math / f1 / gpqa / ifbench) INLINE on the asyncio
    event loop. A catastrophic O(n^2) regex in
    ``math_dapo_utils.normalize_final_answer`` (measured 68s on a 131k-char
    newline-free tail) then froze the ENTIRE run — engines idle, wandb dead,
    zero errors. py-spy caught the frame live.
  * A GIL-holding C computation (big-int sympy, catastrophic-backtrack regex)
    is UNINTERRUPTIBLE by threads or ``asyncio.wait_for``: every thread in the
    process stalls with it. Only SIGKILLing a separate process can stop it.
So the whole per-sample reward computation lives here, in a process the parent
kills on wall-clock timeout. This is the generalization of the deepmath-only
``mv_worker.py``; see ``reward_adapter._run_killable`` for the driver + pool.

Deliberately a tiny standalone module: the spawn child imports ONLY this file
plus (lazily, when the first ``(fn, args)`` is unpickled) whatever module the
grader lives in. It never eagerly imports torch/slime.
"""


def worker_main(conn):
    """Loop: recv ``(fn, args)`` -> send ``("ok", fn(*args), None)``.

    Protocol (always 3-tuples):
      ("ready", None, None)     handshake, sent once at startup
      ("ok",    result, None)   fn returned ``result`` (picklable)
      ("err",   type_name, repr) fn raised; parent decides (ImportError ->
                                 crash loud; else -> per-sample default)

    The parent enforces the REAL wall-clock cap by SIGKILLing this process; the
    computation dies with it. ``conn.recv()`` unpickles ``fn`` (importing its
    module in this child on first use) — a failure there is reported as an err
    so the parent can surface a missing-package bug instead of silently scoring
    zero for a whole run."""
    conn.send(("ready", None, None))
    while True:
        try:
            item = conn.recv()
        except (EOFError, OSError):
            return
        except Exception as exc:  # unpickling (fn, args) failed, e.g. ImportError in child
            try:
                conn.send(("err", type(exc).__name__, repr(exc)))
                continue
            except (BrokenPipeError, OSError):
                return
        if item is None:
            return
        fn, args = item
        try:
            conn.send(("ok", fn(*args), None))
        except Exception as exc:
            try:
                conn.send(("err", type(exc).__name__, repr(exc)))
            except (BrokenPipeError, OSError):
                return
