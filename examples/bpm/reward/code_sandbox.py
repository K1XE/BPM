#!/usr/bin/env python3
"""Safe, killable code-execution sandbox for offline code eval (LiveCodeBench / MBPP+ / HumanEval+).

Pure-KD setup: used ONLY to score eval generations, never in the training loop. Even so it executes
UNTRUSTED model-generated code, so every run is a separate killable process with hard limits — one
hung / forking / memory-bombing submission can never take down the eval (mirrors the reward_adapter
v3 killable-worker lesson: a hung reward once froze a whole run).

Isolation per run: fresh process group (SIGKILL on timeout), RLIMIT_AS memory cap, RLIMIT_CPU,
RLIMIT_NPROC (anti fork-bomb), RLIMIT_FSIZE, closed/ços network best-effort, ephemeral cwd, stdin fed
then closed. (For hard isolation add firejail/nsjail/seccomp if available; this is the portable core.)

Two grading modes cover the benchmark families:
  - run_stdin_tests  : LCB / competitive — feed stdin, compare stdout (normalized).
  - run_assert_tests : MBPP+ / HumanEval+ — append the assert/call harness, pass iff exit 0.
"""
from __future__ import annotations
import os, sys, resource, signal, subprocess, tempfile, time
from decimal import Decimal, InvalidOperation

_PY = sys.executable


_NOBODY = 65534   # drop untrusted code to nobody:nobody when we run as root


def _set_limits(mem_mb: int, cpu_s: int):
    def _apply():
        # new session so we can SIGKILL the whole group (kills child forks too)
        os.setsid()
        b = mem_mb * 1024 * 1024
        for res, lim in (
            (resource.RLIMIT_AS, (b, b)),          # address space
            (resource.RLIMIT_CPU, (cpu_s, cpu_s)), # cpu seconds (backstop for wall timeout)
            (resource.RLIMIT_FSIZE, (256 * 1024 * 1024,) * 2),  # bound child file/stdout writes
            (resource.RLIMIT_NPROC, (4096, 4096)), # anti fork-bomb; per-USER once we drop to nobody
        ):
            try:
                resource.setrlimit(res, lim)
            except (ValueError, OSError):
                pass
        # Prefer THIS sandbox for the host OOM killer over a co-tenant training job; an OOM kill
        # then surfaces as returncode -9 -> the transient retry below (the mode it was built for).
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")
        except Exception:
            pass
        # NEVER execute untrusted / unhardened (BCB) code as root: root bypasses RLIMIT_NPROC,
        # can raise its own rlimits, and (unhardened) has write/kill power over the whole node
        # incl. the neighbouring training job. Match official BigCodeBench's non-root docker user.
        if os.geteuid() == 0:
            try:
                os.setgroups([]); os.setgid(_NOBODY); os.setuid(_NOBODY)
            except Exception:
                pass
    return _apply


def _run(code: str, stdin: str, timeout: float, mem_mb: int, extra_env: dict | None = None) -> tuple[str, str, int]:
    """Run `code` as a script with `stdin`; return (stdout, stderr, status) where status:
    0 ok, 124 timeout, 137 killed/mem, else nonzero exit."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        # A hermetic, deterministic child env: HOME/TMPDIR/config dirs INSIDE the tempdir (so a
        # BCB test's tempfile/matplotlib writes are auto-cleaned and never touch shared /tmp);
        # single-thread every BLAS/OMP backend (a 16-way-parallel grader must not oversubscribe
        # the node -> load-dependent timeouts); TZ=UTC + big int-str + utf-8 io (all official).
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0",
               "HOME": d, "TMPDIR": d, "TEMP": d, "TMP": d,
               "XDG_CONFIG_HOME": os.path.join(d, ".cfg"), "MPLCONFIGDIR": os.path.join(d, ".mpl"),
               "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8", "PYTHONINTMAXSTRDIGITS": "50000",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TZ": "UTC"}
        if extra_env:
            env.update(extra_env)
        # Retry TRANSIENT failures (fork/OOM-kill under host memory pressure -- large test
        # suites like HumanEval+'s 500KB/764-case bodies flake on a busy node). A retry can
        # only turn a transient false-negative back into a pass; a genuinely wrong/looping
        # program fails the same way every attempt, so this never creates a false-positive.
        for _attempt in range(3):
            # fresh per-attempt cwd (a -9 retry must NOT inherit attempt N-1's fixture files);
            # world-writable so the dropped-to-nobody child can create files.
            cwd = os.path.join(d, f"run{_attempt}")
            try:
                os.makedirs(cwd, exist_ok=True)
                for sub in (cwd, os.path.join(d, ".cfg"), os.path.join(d, ".mpl")):
                    os.makedirs(sub, exist_ok=True); os.chmod(sub, 0o777)
                os.chmod(d, 0o777)
            except Exception:
                pass
            # Redirect child stdout/stderr to FILES (bounded by RLIMIT_FSIZE) instead of PIPEs:
            # a print-flood generation can't balloon the parent's RSS, and a forked grandchild
            # holding the pipe open can't make a finished-correct program read as a timeout.
            op = os.path.join(cwd, ".out"); ep = os.path.join(cwd, ".err")
            try:
                ofh = open(op, "wb"); efh = open(ep, "wb")
                p = subprocess.Popen(
                    [_PY, path], stdin=subprocess.PIPE, stdout=ofh, stderr=efh,
                    cwd=cwd, env=env, preexec_fn=_set_limits(mem_mb, int(timeout) + 1),
                )
                ofh.close(); efh.close()
            except Exception as e:
                if _attempt < 2:
                    time.sleep(0.5 * (_attempt + 1)); continue
                return "", f"spawn-error: {e}", 1
            timed_out = False
            try:
                p.communicate(input=(stdin or "").encode("utf-8", "ignore"), timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # reap the WHOLE group on EVERY exit path (child is a session leader; kills
                # any helper process/server a BCB suite left running on the shared node).
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    p.wait(timeout=2)
                except Exception:
                    pass
            if timed_out:
                return "", "timeout", 124
            def _read(fp, cap):
                try:
                    with open(fp, "rb") as fh:
                        return fh.read(cap).decode("utf-8", "backslashreplace")
                except Exception:
                    return ""
            out, err = _read(op, 32 * 1024 * 1024), _read(ep, 64 * 1024)
            # 137/-9 = SIGKILL (host OOM-killer under memory pressure) -> transient, retry
            if p.returncode in (-9, 137) and _attempt < 2:
                time.sleep(0.5 * (_attempt + 1)); continue
            return out, err, p.returncode
        return "", "spawn-error: retries exhausted", 1


# Run the model solution with __name__ == "__test__" (never "__main__") so a demo
# `if __name__ == "__main__": main()` / input() block does NOT execute and EOF-fail an
# otherwise-correct function; and turn an early sys.exit(0) into a HARD FAIL (exit 13) so a
# generation that exits before the asserts run can never falsely pass. Matches the exec-into-
# a-module semantics of EvalPlus / BigCodeBench / lcb_runner. Preamble/import names stay in the
# shared globals (annotations like `def f(a: List[int])` evaluate at def-time in that scope).
_SOL_SHIM = (
    "import base64 as _b64\n"
    "_g = dict(globals()); _g['__name__'] = '__test__'\n"
    "try:\n"
    "    exec(compile(_b64.b64decode('%s').decode('utf-8'), 'solution.py', 'exec'), _g)\n"
    "except SystemExit:\n"
    "    raise SystemExit(13)\n"
    "globals().update({_k: _v for _k, _v in _g.items() if _k != '__name__'})\n"
)


def wrap_solution(code: str) -> str:
    import base64 as _b64
    return _SOL_SHIM % _b64.b64encode((code or "").encode("utf-8")).decode()


def _norm(s: str) -> str:
    """Official LCB stdout normalization (get_stripped_lines): strip BOTH sides of every line, drop
    trailing blank lines. Matches lcb_runner testing_util, which strips each line before comparing."""
    lines = [ln.strip() for ln in (s or "").replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _decimal_equal(a: str, b: str) -> bool:
    """Official LCB numeric compare: line-structured, EXACT. Inputs are already _norm'd. Require the
    SAME line count and SAME tokens-per-line; a line matches iff it is byte-equal OR every token is a
    Decimal and the token lists are numerically equal (Decimal('1.0')==Decimal('1.00'), '1e3'=='1000').
    There is NO 1e-6 tolerance -- the official grader compares exactly (a tolerance credits a
    numerically-close-but-wrong answer, a false-positive). Any non-numeric token difference -> WA.
    The line structure also prevents "1\\n2\\n3" from matching "1 2 3" (a shape false-positive)."""
    la, lb = a.split("\n"), b.split("\n")
    if len(la) != len(lb):
        return False
    for lax, lbx in zip(la, lb):
        if lax == lbx:
            continue
        ta, tb = lax.split(), lbx.split()
        if len(ta) != len(tb):
            return False
        try:
            if [Decimal(x) for x in ta] != [Decimal(y) for y in tb]:
                return False
        except (InvalidOperation, ValueError):
            return False
    return True


def run_stdin_tests(code: str, tests: list[tuple[str, str]], timeout: float = 6.0,
                    mem_mb: int = 1024) -> bool:
    """All (stdin, expected_stdout) pass? Exact-after-normalization, with an EXACT Decimal numeric
    fallback (official LCB semantics; per-test wall timeout matches the official signal.alarm(6))."""
    if not code or not tests:
        return False
    for stdin, expected in tests:
        out, err, status = _run(code, stdin, timeout, mem_mb)
        if status != 0:
            return False
        no, ne = _norm(out), _norm(expected)
        if no == ne:
            continue
        if _decimal_equal(no, ne):
            continue
        return False
    return True


# Hardened preamble for ASSERT-based grading (MBPP+/HumanEval+/KodCode/MBPP). Two jobs:
#   (1) reliability_guard (evalplus-faithful): neutralize destructive / code-exec calls so an
#       untrusted student rollout can never delete/truncate a real file or spawn a process. Our
#       process isolation + RLIMITs bound RESOURCES and lifetime, but NOT filesystem writes/deletes
#       (the sandbox subprocess runs as the same user); at training scale we grade thousands of
#       generations, so an in-process guard is genuine defense-in-depth, not paranoia.
#   (2) import scaffolding: resolve bare `List/Optional` type hints (evaluated at def-time -- the
#       #1 assert-path false-negative) and common module names, without `from math import *` (which
#       would shadow the 3-arg builtin pow). Only ADDS names -> can only turn a false-fail into a
#       pass, never a pass into a fail. stdout is swallowed so a print-heavy solution cannot flood
#       the pipe (we grade on EXIT CODE; asserts raise -> nonzero exit regardless of stdout).
# Self-consistency at dataset-build time runs canonical solutions through THIS SAME harness, so any
# problem whose reference legitimately needs a disabled call (file I/O, subprocess) is dropped
# rather than silently mis-graded -- the grader and the data are validated as one unit.
_ASSERT_GUARD = (
    "import os as _os, sys as _sys\n"
    "try:\n"
    "    import shutil as _shutil, subprocess as _subprocess, builtins as _bi\n"
    "    def _blk(*a, **k):\n"
    "        raise OSError('disabled in sandbox')\n"
    # putenv/getcwd/chdir are NOT disabled: numpy et al. set os.environ (-> putenv) at import
    # time, so disabling them false-negatives every library-importing test. Only genuine
    # delete / rename / exec / kill / privilege vectors are neutralized.
    "    for _m, _ns in ((_os, ['kill','system','remove','removedirs','rmdir',"
    "'setuid','fork','forkpty','killpg','rename','renames','truncate','replace','unlink',"
    "'chroot','lchflags','lchmod','lchown']), (_shutil, ['rmtree','move','chown'])):\n"
    "        for _n in _ns:\n"
    "            try:\n"
    "                setattr(_m, _n, _blk)\n"
    "            except Exception:\n"
    "                pass\n"
    "    _subprocess.Popen = _blk\n"
    # builtins.open is intentionally NOT disabled: import-time file reads (numpy et al. in
    # HumanEval+ test bodies) need it, and disabling it false-negatives every numpy-using test.
    # Destructive WRITES are bounded by RLIMIT_FSIZE (64MB) + the ephemeral cwd; the real
    # delete/exec vectors (os.remove/system, shutil.rmtree, subprocess.Popen, os.fork) are gone.
    "except Exception:\n"
    "    pass\n"
)
_ASSERT_IMPORTS = (
    "import sys as _s2\n"
    "_s2.setrecursionlimit(100000)\n"
    "try:\n"
    "    from typing import *\n"
    "    import math, collections, itertools, functools, heapq, bisect, re, string, json\n"
    "    from collections import *\n"
    "    from itertools import *\n"
    "    from functools import *\n"
    "except Exception:\n"
    "    pass\n"
)
# Swallow the SOLUTION's stdout so a print-heavy program can't flood the parent's capture buffer
# (RLIMIT_AS caps the child's memory, not the grader parent reading the pipe). Runs BEFORE the
# guard disables builtins.open, so it can still open /dev/null. Grading is on EXIT CODE, so a
# swallowed stdout never changes pass/fail.
_ASSERT_SWALLOW = (
    "import os as _o3, sys as _s3\n"
    "try:\n"
    "    _s3.stdout = open(_o3.devnull, 'w')\n"
    "except Exception:\n"
    "    pass\n"
)


def run_assert_tests(code: str, asserts: list[str], entry_setup: str = "", timeout: float = 8.0,
                     mem_mb: int = 1024, hardened: bool = True) -> bool:
    """MBPP+/HumanEval+/KodCode style: append each assert/harness snippet; pass iff it runs with
    exit 0. All asserts run together (fail-fast inside the process). With hardened=True (default)
    the executed program is prefixed with the import scaffolding + reliability_guard above."""
    if not code:
        return False
    # imports FIRST (need real `open` for stdout swallow + before guard disables open),
    # then swallow stdout, then the guard, then the solution + asserts.
    prefix = (_ASSERT_IMPORTS + _ASSERT_SWALLOW + _ASSERT_GUARD) if hardened else ""
    harness = prefix + wrap_solution(code) + "\n" + entry_setup + "\n" + "\n".join(asserts) + "\n"
    out, err, status = _run(harness, "", timeout, mem_mb)
    return status == 0


if __name__ == "__main__":
    print("code_sandbox ready; python =", _PY)
