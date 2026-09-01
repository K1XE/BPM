#!/usr/bin/env python3
"""Benchmark-specific code scoring on top of the safe sandbox (code_sandbox).

Covers the three families used by this paper's code eval:
  - MBPP+ / HumanEval+ : append the EvalPlus test body (calls the fn by name) -> exit 0.
  - LiveCodeBench stdin   : feed each test's stdin, compare stdout (normalized).
  - LiveCodeBench functional (LeetCode-style): call Solution().<func_name>(*parsed_args) == expected.

A problem is scored pass@1 = 1 iff the extracted program passes ALL of its tests. All execution
goes through the killable sandbox, so a bad generation can never hang/booby-trap the eval.
"""
from __future__ import annotations
import base64
import json
import os
from examples.bpm.reward.code_extract import extract_code
from examples.bpm.reward.code_sandbox import run_stdin_tests, run_assert_tests, _run, wrap_solution

# LiveCodeBench-style preamble, prepended BEFORE the model's code so that (a) type
# annotations in the signature (def f(a: List[str])) resolve at def-time -- the #1 cause of
# false-negatives on the 444 functional problems -- and (b) bare names competitive solutions
# assume (inf, gcd, sqrt, deque, defaultdict, Counter, accumulate...) are in scope. `from math
# import *` shadows the builtin 3-arg pow(a,b,mod), so we restore it afterwards.
_LCB_PREAMBLE = (
    # Official lcb_runner testing_util import_string: star-import every stdlib module competitive
    # solutions call by bare name (deque/defaultdict/Counter, gcd/inf/comb, bisect_left, heappush,
    # deepcopy, mean/median, reduce/accumulate, operator names, string constants). Missing any of
    # these NameErrors an otherwise-correct solution (false-negative). `from builtins import *` comes
    # AFTER `from math import *` so it restores the 3-arg builtin pow(a,b,mod) that math.pow shadows.
    "from string import *\n"
    "from re import *\n"
    "from datetime import *\n"
    "from collections import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from copy import *\n"
    "from math import *\n"
    "from random import *\n"
    "from statistics import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from operator import *\n"
    "from io import *\n"
    "from typing import *\n"
    "from builtins import *\n"
    "import sys, math, collections, itertools, functools, heapq, bisect, string, re, json\n"
    "sys.setrecursionlimit(100000)\n"
)

# Functional harness: prefer a module-level function of that name, else Solution().{fn}
# (LCB has both class-based and bare-function solutions -- the old harness only tried the
# class -> NameError/false-negative on module-level solutions). tuple/list are normalized so a
# tuple return compares equal to a list-shaped expected output.
_FUNC_HARNESS = """
import json as _json, ast as _ast
def _parse(s):
    s = s.strip()
    try: return _json.loads(s)
    except Exception:
        try: return _ast.literal_eval(s)
        except Exception: return s
def _top(x):
    # Official lcb_runner semantics: convert ONLY the top-level tuple to a list, then compare with
    # ==. (The previous recursive list/tuple flattening was more lenient than the leaderboard.)
    return list(x) if isinstance(x, tuple) else x
_g = globals()
if "{fn}" in _g and callable(_g["{fn}"]):
    _fn = _g["{fn}"]
elif "Solution" in _g:
    _fn = getattr(Solution(), "{fn}")
else:
    raise SystemExit(11)
for _inp, _exp in _CASES:
    _args = [_parse(l) for l in _inp.split("\\n") if l.strip() != ""] if _inp.strip() else []
    _want = _parse(_exp)
    try:
        _got = _fn(*_args)
    except Exception as _e:
        raise SystemExit(3)
    if _top(_got) != _top(_want):
        raise SystemExit(4)
raise SystemExit(0)
"""


def score_mbpp(generation: str, test_body: str, timeout: float = 62.0, mem_mb: int = 1024) -> bool:
    code = extract_code(generation)
    if code is None:
        return False
    # test_body goes through entry_setup so ONLY the model solution is shimmed (__name__=__test__,
    # early sys.exit -> hard fail); a `sys.exit(0)` generation can no longer skip the asserts.
    return run_assert_tests(code, [], entry_setup=test_body, timeout=timeout, mem_mb=mem_mb)


# ---- BigCodeBench-Hard (2026-07-10) -------------------------------------------------
# BCB tests are unittest.TestCase classes (often with unittest.mock) that exercise real
# filesystem/plotting/library behavior, so they run UNHARDENED (no reliability_guard),
# with any extra deps supplied via BPM_CODEEVAL_PYLIBS and a headless MPL backend.
# Pass iff the whole suite is green (official semantics). 132/148 hard tasks are gradeable
# (mem_mb=16384 is a virtual-address-space cap: scipy/matplotlib thread stacks EAGAIN at 4G).
# in this env; a few tasks (heavy libs / pandas-2.x drift / live-network tests) are
# excluded at dataset-build time.
_BCB_RUNNER = (
    "\n\nimport unittest, sys\n"
    # Load EXACTLY the official `TestCases` class (a model that echoes its own extra TestCase
    # subclass can't dilute the suite), and REQUIRE testsRun>0 so an empty suite is not a pass.
    "_s = unittest.TestLoader().loadTestsFromTestCase(TestCases)\n"
    "_r = unittest.TextTestRunner(stream=sys.stderr, verbosity=0).run(_s)\n"
    "sys.exit(0 if (_r.wasSuccessful() and _r.testsRun > 0) else 1)\n"
)
_BCB_PYLIBS = os.environ.get("BPM_CODEEVAL_PYLIBS", "")
_BCB_ENV = {
    "PYTHONPATH": _BCB_PYLIBS,
    "MPLBACKEND": "Agg",
    "NLTK_DATA": os.path.join(_BCB_PYLIBS, "nltk_data") if _BCB_PYLIBS else "",
    "TOKENIZERS_PARALLELISM": "false",
}


# BCB solutions must run in the SAME namespace as the test (unittest.mock.patch targets the
# solution's imported names, and BCB test closures read module-level names the solution defines).
# So we DON'T isolate into a private dict like wrap_solution does; we only (a) neutralize a demo
# `if __name__ == "__main__":` block by exec'ing the solution under a name != "__main__", and
# (b) turn an early sys.exit() into a hard fail. The solution's globals ARE this module's globals.
def _bcb_wrap(code: str) -> str:
    import base64 as _b64
    b = _b64.b64encode((code or "").encode("utf-8")).decode()
    return (
        "import base64 as _b64x\n"
        "__name__ = '__bcb_solution__'\n"            # so `if __name__=='__main__'` is skipped
        "try:\n"
        "    exec(compile(_b64x.b64decode('%s').decode('utf-8'), 'solution.py', 'exec'), globals())\n"
        "except SystemExit:\n"
        "    raise SystemExit(13)\n"
        "__name__ = '__main__'\n"                     # restore so the test's own `unittest.main`/name checks behave
    ) % b


def score_bcb(generation: str, test_body: str, timeout: float = 240.0, mem_mb: int = 30720) -> bool:
    # timeout/mem match official BigCodeBench (TIMEOUT_LIMIT=240, 30GB AS): a slow-but-correct
    # sklearn/statsmodels suite on a co-loaded node must not time out (load-dependent = arms not
    # comparable). Solution is shimmed (__name__=__test__ + sys.exit->fail); the sandbox runs it
    # dropped to nobody (unhardened BCB must never touch the node as root).
    code = extract_code(generation)
    if code is None or not test_body:
        return False
    harness = _bcb_wrap(code) + "\n\n" + test_body + _BCB_RUNNER
    out, err, status = _run(harness, "", timeout, mem_mb, extra_env=_BCB_ENV)
    return status == 0


def score_lcb_stdin(generation: str, tests: list[dict], timeout: float = 6.0) -> bool:
    code = extract_code(generation)
    if code is None:
        return False
    pairs = [(t["input"], t["output"]) for t in tests]
    # Same preamble as functional so a stdin program using bare typing/collections/math names
    # (deque, defaultdict, inf, gcd...) doesn't NameError. It runs before the user program;
    # a program that re-imports these is unaffected.
    return run_stdin_tests(_LCB_PREAMBLE + code, pairs, timeout=timeout)


def score_lcb_functional(generation: str, tests: list[dict], func_name: str,
                         timeout: float = 6.0) -> bool:
    code = extract_code(generation)
    if code is None or not func_name:
        return False
    cases = [(t["input"], t["output"]) for t in tests]
    # _LCB_PREAMBLE MUST come first: `def f(a: List[str])` evaluates its annotation at def-time,
    # which runs inside the user code, before the harness body.
    harness = _LCB_PREAMBLE + wrap_solution(code) + "\n_CASES = " + repr(cases) + "\n" + _FUNC_HARNESS.replace("{fn}", func_name)
    _, _, status = _run(harness, "", timeout, 1024)
    return status == 0


def score_lcb(generation: str, problem: dict, timeout: float = 6.0) -> bool:
    # Official dispatch: a problem is FUNCTIONAL iff it carries a function name (in_outs['fn_name']
    # is not None), else it is a stdin/stdout problem -- more robust than trusting a `testtype` tag.
    func_name = problem.get("func_name") or ""
    if func_name.strip():
        return score_lcb_functional(generation, problem["tests"], func_name, timeout)
    return score_lcb_stdin(generation, problem["tests"], timeout)


# ----------------------------------------------------------------------------
# TACO grading (BAAI/TACO `input_output` field).
#
# TACO packs two contract shapes into one JSON string per problem:
#   stdin/stdout : {"inputs":[...], "outputs":[...]}                  (~87%; fn_name absent)
#   function     : {"fn_name":"f", "inputs":[[args],...], "outputs":[exp,...]}   (~13%)
# A problem scores 1 iff the extracted program passes ALL (capped) cases.
#
# Test capping (mirrors build_lcb_full.py cap_tests): grade at most _TACO_MAX_CASES
# cases and skip any single case whose input+output exceeds _TACO_MAX_CASE_BYTES, so a
# pathological max-stress test (TACO ships some) can never hang the reward path. Each case
# still runs inside the killable sandbox with a wall timeout, so this only bounds the
# *number* of sandbox spawns, not their individual safety.
# ----------------------------------------------------------------------------
_TACO_MAX_CASES = 40
_TACO_MAX_CASE_BYTES = 500_000

_TACO_FUNC_HARNESS = '''
import json as _json, base64 as _b64, math as _math
_CASES = _json.loads(_b64.b64decode("__B64__").decode())

def _taco_eq(a, b):
    if a == b:
        return True
    try:
        return _math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False

def _lit(x):
    if isinstance(x, str):
        try:
            import ast as _ast
            return _ast.literal_eval(x.strip())
        except Exception:
            return x
    return x

def _taco_match_inner(got, exp):
    # TACO wraps most functional expected-outputs in a 1-element list ([True], [25], ['MAS']);
    # some are raw scalars. Accept got==exp, got==exp[0], or [got]==exp, with float tolerance.
    if _taco_eq(got, exp):
        return True
    if isinstance(exp, list) and len(exp) == 1 and _taco_eq(got, exp[0]):
        return True
    if isinstance(got, list) and len(got) == 1 and _taco_eq(got[0], exp):
        return True
    return False

def _taco_match(got, exp):
    # A subset of TACO fn-rows store the expected output as the STRINGIFIED repr of the return
    # value wrapped in a 1-element list (fn returns [2], the row stores ['[2]']); literal_eval
    # the expected side and retry, else a correct solution scores 0. Audited 2026-07-10:
    # rescues 11/13 fn false-negatives, 0 regressions on 37 passing rows, 0 false-positives
    # on 40 wrong-solution controls. Only the EXPECTED side is evaluated, never the model's.
    if _taco_match_inner(got, exp):
        return True
    e = _lit(exp[0]) if isinstance(exp, list) and len(exp) == 1 and isinstance(exp[0], str) else _lit(exp)
    return _taco_match_inner(got, e)

_g = globals()
if "__FN__" in _g and callable(_g["__FN__"]):
    _fn = _g["__FN__"]
elif "Solution" in _g:
    _fn = getattr(Solution(), "__FN__")
else:
    raise SystemExit(11)

for _inp, _exp in _CASES:
    _args = list(_inp) if isinstance(_inp, list) else [_inp]
    try:
        _got = _fn(*_args)
    except SystemExit:
        raise
    except Exception:
        raise SystemExit(4)
    if not _taco_match(_got, _exp):
        raise SystemExit(5)
raise SystemExit(0)
'''


def _taco_cap(inputs: list, outputs: list) -> list[tuple]:
    """Bound the graded case set: skip oversized cases, keep at most _TACO_MAX_CASES."""
    cases = []
    for inp, out in zip(inputs, outputs):
        if len(str(inp)) + len(str(out)) > _TACO_MAX_CASE_BYTES:
            continue
        cases.append((inp, out))
        if len(cases) >= _TACO_MAX_CASES:
            break
    return cases


def _taco_stdin_str(x) -> str:
    """TACO stdin inputs/outputs are usually a full string, sometimes a list of lines."""
    if isinstance(x, list):
        return "\n".join(str(e) for e in x)
    return str(x)


def score_taco(generation: str, tests: dict, timeout: float = 12.0, mem_mb: int = 1024) -> bool:
    """Grade a generation against ONE TACO problem's parsed `input_output` dict.

    `tests` is the JSON-decoded input_output field (stdin or fn_name form, see module note).
    Returns True iff the extracted python program passes ALL capped cases.
    """
    code = extract_code(generation)
    if code is None:
        return False
    inputs = tests.get("inputs") or []
    outputs = tests.get("outputs") or []
    if not inputs or not outputs:
        return False
    cases = _taco_cap(inputs, outputs)
    if not cases:
        return False
    fn_name = tests.get("fn_name")
    if fn_name:
        payload = base64.b64encode(json.dumps(cases).encode()).decode()
        harness = _TACO_FUNC_HARNESS.replace("__B64__", payload).replace("__FN__", fn_name)
        return run_assert_tests(code, [], entry_setup=harness, timeout=timeout, mem_mb=mem_mb)
    pairs = [(_taco_stdin_str(i), _taco_stdin_str(o)) for i, o in cases]
    return run_stdin_tests(code, pairs, timeout=timeout, mem_mb=mem_mb)


if __name__ == "__main__":
    print("code_eval ready")
