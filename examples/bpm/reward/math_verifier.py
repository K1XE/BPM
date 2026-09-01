#!/usr/bin/env python3
"""Thorough math answer verifier for offline eval (AIME / MATH500 / AMC / HMMT).

Pure-KD setup: this is used ONLY to score eval generations, never in the training loop.

Design = MARKER-BASED extraction + two equivalence engines, tuned to match the OFFICIAL graders
(lm-eval minerva, lighteval math_verify) with NEITHER false-positives NOR false-negatives:

  extraction (extract_answer_robust): LAST \\boxed{} first (the reliable final-answer marker for
     reasoning models that state the answer after </think>), then the "Answer:" marker, stripping a
     leading in-box "Answer:" / \\text{Answer:} prefix. There is deliberately NO last-number fallback
     and NO greedy full-text math_verify parse -- both credit a stray digit in a give-up / hesitant /
     truncated generation (verified false-positives: "...I give up 34"==34, "surface area is 50%"==50,
     prose "...204..."==204). No marker -> no answer -> scored 0, exactly as the official graders
     (which require the boxed / "Final Answer:" marker) do.

  engine 1: DAPO Minerva scorer (is_correct_minerva) -- identical to the training/eval pipeline.
  engine 2: DeepScaler/verl grade_answer (sympy + Hendrycks normalization) + math_verify, BOTH run
     only on the EXTRACTED span, so a bare span like "100\\pi - 200" verifies as one expression.
     Credits fraction/decimal/sqrt/pi/tuple/interval equivalence (1/2==0.5, \\frac{408}{2}==204,
     2\\sqrt2, (2,3), [0,\\infty)) that pure string-normalization misses.

  percent guard: a percent answer and a plain number are DIFFERENT quantities (50% = 0.5 != 50) and
     both Minerva and DeepScaler strip the '%'; when exactly one of pred/gold is a percent we trust
     only math_verify (which reads 50% as 0.5), never the '%'-stripping engines.

All engines are wrapped so a parser hang/exception can never crash the batch (offline eval).
Verified on real GLM-4.7-Flash AIME24 (240 rollouts): removes the recorded grader's false-positives
(truncated no-answer rollouts) with ZERO new false-negatives; MATH500 golds 500/500 self-consistent.
"""
from __future__ import annotations
import re
import signal
from contextlib import contextmanager

# --- engine 1: math_verify (optional at import time) -------------------------------------------
import logging

logger = logging.getLogger(__name__)

try:
    from math_verify import parse as _mv_parse_raw, verify as _mv_verify_raw
    _HAVE_MV = True

    # math_verify's parse()/verify() default to a signal.alarm() timeout, which raises
    # "doesn't support threaded environment" in any non-main thread. The Path-A eval grades
    # via asyncio.to_thread, so the raw calls silently return False there -> the strongest
    # equivalence engine is lost and MATH500/AIME symbolic answers (408/2==204) are UNDER-
    # scored. parsing_timeout/timeout_seconds=None disables the internal signal so it runs in
    # ANY thread; our _time_limit(...) still provides the outer wall-clock guard on the main
    # thread (and math answers are short, so a threaded call without a guard won't hang).
    def _mv_parse(text, **kw):
        kw.setdefault("parsing_timeout", None)
        return _mv_parse_raw(text, **kw)

    def _mv_verify(gold, target, **kw):
        kw.setdefault("timeout_seconds", None)
        return _mv_verify_raw(gold, target, **kw)

    # Silence math_verify's per-parse "Timeout is disabled ..." warning (child loggers
    # math_verify.parser/.metric/.grader emit it every call once parsing_timeout=None -> it
    # would flood the eval log). Setting the parent logger alone doesn't suppress children.
    import logging as _logging
    for _ln in ("math_verify", "math_verify.parser", "math_verify.metric", "math_verify.grader"):
        _logging.getLogger(_ln).setLevel(_logging.ERROR)
except Exception as _e:
    _HAVE_MV = False
    logger.error(
        "[BPM math_verifier] math-verify unavailable (%s). Symbolic equivalence is OFF; "
        "MATH-500/AIME scores will be understated by several points. pip install math-verify",
        _e,
    )

# --- engine 2: DeepScaler/verl sympy checker (already in the repo) -------------------------------
import os, sys
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "slime", "rollout", "rm_hub")
)
try:
    from math_utils import (  # type: ignore
        grade_answer_mathd, grade_answer_sympy, extract_boxed_answer, last_boxed_only_string,
    )
    _HAVE_DS = True
except Exception as _e:
    _HAVE_DS = False
    logger.error("[BPM math_verifier] DeepScaler sympy checker unavailable (%s).", _e)

# --- engine 3: DAPO Minerva scorer -- THIS IS WHAT THE TRAINING/EVAL PIPELINE USES (rm_type=dapo).
# Using it here keeps offline table numbers consistent with the in-training eval curve, and it has the
# most complete extractor (Answer:/answer is/=/Chinese 答案/\boxed, latest-candidate-wins, im_end/eot
# stripping) + Minerva normalization + percent-aware matching.
try:
    from math_dapo_utils import is_correct_minerva, extract_answer_from_solution, _INVALID_PRED  # type: ignore
    _HAVE_DAPO = True
except Exception as _e:
    _HAVE_DAPO = False
    logger.error("[BPM math_verifier] DAPO Minerva scorer unavailable (%s).", _e)


if not (_HAVE_MV or _HAVE_DS or _HAVE_DAPO):
    raise RuntimeError(
        "[BPM math_verifier] no math grading engine available. Scores would be "
        "silently understated. Install requirements: pip install math-verify pylatexenc"
    )

@contextmanager
def _time_limit(seconds: float):
    """Best-effort wall-clock guard (main thread only); no-op if signals unavailable."""
    def _raise(signum, frame):
        raise TimeoutError()
    try:
        old = signal.signal(signal.SIGALRM, _raise)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    except (ValueError, AttributeError):
        # not in main thread / no SIGALRM -> run without the guard
        yield


# --- robust answer extraction (for the fallback engine + for logging) --------------------------
_ANSWER_PATTERNS = [
    re.compile(r"(?:final\s+answer|the\s+answer\s+is|answer)\s*[:=]?\s*\$?\\?boxed\{", re.I),  # "Answer: \boxed{"
    re.compile(r"(?:final\s+answer|the\s+answer\s+is|answer)\s*[:=]\s*", re.I),                # "Answer: X"
]
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
# a leading "Answer:" / "the answer is" the model sometimes writes INSIDE the box, e.g.
# \boxed{Answer: 25} -> the boxed content is "Answer: 25", which math_verify can't parse. Strip it.
_BOXED_ANSWER_PREFIX = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:=]?\s*", re.I)
# same, but wrapped in \text{...}: \boxed{\text{Answer: } 25}
_TEXT_ANSWER_PREFIX = re.compile(r"^\s*\\text\s*\{\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:=]?\s*\}\s*", re.I)


def _is_percent(s: str) -> bool:
    """True if the (trimmed) answer is a percentage form, e.g. '50%' or '50\\%'."""
    return bool(s) and s.strip().rstrip("$ ").endswith("%")


def _strip_answer_prefix(s: str) -> str:
    """Drop a leading 'Answer:' / 'the final answer is' the model wrote inside \\boxed{...},
    including the \\text{Answer: } wrapped form."""
    if not s:
        return s
    s2 = _TEXT_ANSWER_PREFIX.sub("", s, count=1)       # \boxed{\text{Answer: }25} -> "25"
    stripped = _BOXED_ANSWER_PREFIX.sub("", s2, count=1).strip()  # \boxed{Answer: 25} -> "25"
    # only accept the strip if it left something (don't turn "answer" alone into "")
    return stripped if stripped else s

# unicode dashes that models emit for a minus sign: en-dash, em-dash, unicode minus.
_DASH_CHARS = {"–": "-", "—": "-", "−": "-"}
_NBSP_BETWEEN_DIGITS = re.compile("(?<=\\d)\u00a0(?=\\d)")  # NBSP digit-group separator


def _normalize_surface(text: str) -> str:
    """Normalize surface artifacts that break equivalence: all unicode dashes -> ascii '-',
    strip LaTeX spacing macros (\\, \\; \\!), and drop a NBSP used as a digit-group separator
    (e.g. '1 000' -> '1000')."""
    if not text:
        return text
    for src, dst in _DASH_CHARS.items():
        text = text.replace(src, dst)
    text = text.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    text = _NBSP_BETWEEN_DIGITS.sub("", text)
    return text


def _extract_answer_span(text: str) -> str | None:
    """Extract the final answer from ONE (already surface-normalized) text segment, best source first."""
    # 1. LAST \boxed{...} FIRST -- the single most reliable final-answer marker, and what the
    #    official extractors (lm-eval, lighteval) key on. Reasoning models state the answer as
    #    \boxed{} AFTER </think>; if the DAPO text extractor runs first it can match an "answer is"
    #    phrase INSIDE the thinking and grab a malformed span across the </think> boundary
    #    (verified: 13/15 real AIME24 \boxed answers were lost that way -> a 5.4pp false-negative
    #    hit). Boxed-first eliminates it while a give-up/truncated generation with no \boxed still
    #    falls through to score 0.
    if _HAVE_DS:
        inner = extract_boxed_answer(text) if last_boxed_only_string(text) else None
        if inner is not None and inner.strip():
            return _strip_answer_prefix(inner.strip())
    # 1b. a 'boxed{...}' with the backslash dropped by the model (common typo)
    mb = list(re.finditer(r"\\?boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text))
    if mb and mb[-1].group(1).strip():
        return _strip_answer_prefix(mb[-1].group(1).strip())
    # 2. the DAPO pipeline extractor (Answer:/answer is/=/Chinese 答案, latest-wins, token strip)
    if _HAVE_DAPO:
        try:
            a = extract_answer_from_solution(text)
            if a and a != _INVALID_PRED:
                return a
        except Exception:
            pass
    # 3. explicit "Answer:" / "the answer is" (last marker) then the remainder of that line
    last_span = None
    for pat in _ANSWER_PATTERNS[1:]:
        for m in pat.finditer(text):
            last_span = m.end()
    if last_span is not None:
        tail = text[last_span:].splitlines()[0].strip().rstrip(". ").strip("$").strip()
        if tail:
            return tail
    # NO last-number fallback: the official graders (lm-eval minerva, lighteval math_verify) NEVER
    # fall back to "the last number anywhere". Doing so credits a stray digit in a give-up / hesitant
    # / mid-computation generation (verified false-positives: "...I give up. 34"==34, "the surface
    # area is 50%"==50, prose "...204..."==204). Our eval prompt REQUIRES the answer in \boxed{} or
    # "Answer:", so a compliant model always leaves a marker; a non-compliant one scores 0, exactly
    # as under the official standard. No marker -> no answer -> wrong (never guess a number).
    return None


def _strip_md_emphasis(s: str | None) -> str | None:
    """Strip markdown emphasis the model wraps around the answer line: '**Answer: 110**' has its
    'Answer:' consumed by the extractor but keeps the trailing '**' ('110**'); a bare '**110**'
    -> '110'. Bold/italic '**'/'*' appears near 'Answer' in ~5% of real rollouts and otherwise
    fails the sympy/string match (verified real AIME/HMMT/MATH500 false-negative). Strips only
    LEADING/TRAILING '*' so internal multiplication ('2*3') survives; empty result -> None."""
    if s is None:
        return None
    s = s.strip().strip("*").strip()
    return s or None


def extract_answer_robust(text: str) -> str | None:
    """Extract the final answer from a (possibly very diverse) model generation, best source first.

    Prefers the model's post-</think> final response: a stray \\boxed{} left INSIDE the reasoning
    block must not outrank the actual answer stated after thinking closes (verified false-negative
    on unboxed symbolic finals like 'Answer: 7/3' when a scratch \\boxed{2} sits in the think
    block -- boxed-first would otherwise grab the '2'). Falls back to the whole generation when
    nothing usable follows </think> (the model boxes its answer immediately before closing think,
    or never closes it under the token cap). Markdown emphasis around the answer is stripped."""
    if not text:
        return None
    text = _normalize_surface(text)
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[1]
        if tail.strip():
            span = _strip_md_emphasis(_extract_answer_span(tail))
            if span is not None:
                return span
    return _strip_md_emphasis(_extract_answer_span(text))


def _mv_parse_gold(gold: str):
    """Parse the GOLD: WRAP FIRST (same as _mv_parse_span), so a juxtaged multi-term gold like
    ``5\\sqrt{5}`` / ``288\\pi`` / ``3^{2025}`` / ``\\frac{7}{3}\\sqrt{3}`` is parsed as ONE
    expression and never silently reduced to its leading term. A bare parse of these returns a
    NON-EMPTY but WRONG sub-value (``5\\sqrt{5}`` -> [5]), which then falsely credits a give-up
    student that emits only ``5`` -- inflating HMMT2026 / MATH500 non-uniformly across arms
    (weaker arms emit more partial answers). The wrapped form ``\\boxed{5\\sqrt{5}}`` binds the
    juxtaposition into a single atom; clean integer/fraction/radical golds parse identically
    wrapped, so wrap-first only ever fixes the truncation, never breaks a valid gold."""
    for form in (f"\\boxed{{{gold}}}", f"${gold}$", gold):
        try:
            r = _mv_parse(form)
            if r:
                return r
        except Exception:
            pass
    return []


def _mv_parse_span(span: str):
    """Parse an already-EXTRACTED answer span atomically: WRAP FIRST so a multi-term bare span like
    '100\\pi - 200' is one expression, never silently reduced to its trailing '200' (bare parse can
    return a non-empty but WRONG sub-value, so wrapped forms take priority)."""
    for form in (f"\\boxed{{{span}}}", f"${span}$", span):
        try:
            r = _mv_parse(form)
            if r:
                return r
        except Exception:
            pass
    return []


def _mv_ok(pred: str, gold: str, timeout: float) -> bool:
    """math_verify on a FULL generation: let it extract the answer from `pred` (bare parse)."""
    if not _HAVE_MV:
        return False
    try:
        with _time_limit(timeout):
            g = _mv_parse_gold(gold)
            if not g:
                return False
            return bool(_mv_verify(g, _mv_parse(pred)))
    except Exception:
        return False


def _num_equal(a, b) -> bool:
    """Exact numeric equality of two parsed sympy values. Used ONLY for the percent branch,
    where math_verify's verify() is percent-blind (it credits 50% == 50)."""
    try:
        import sympy
        d = sympy.simplify(a - b)
        if d == 0 or getattr(d, "is_zero", None) is True:
            return True
    except Exception:
        pass
    try:
        return abs(float(a) - float(b)) < 1e-9
    except Exception:
        return False


def _mv_ok_span(span: str, gold: str, timeout: float) -> bool:
    """math_verify on an already-EXTRACTED answer span: parse the WHOLE span as one expression
    (wrapped), so a bare-LaTeX span like '100\\pi - 200' is not silently reduced to its trailing '200'."""
    if not _HAVE_MV:
        return False
    try:
        with _time_limit(timeout):
            if _is_percent(span) or _is_percent(gold):
                # Percent atom ("50%"): math_verify is percent-BLIND -- verify(Integer(50),
                # Rational(1,2)) returns True (it credits 50% == 50), and parse("50%") appends a
                # %-stripped STRING fallback "50" that also string-matches a plain '50' gold. So we
                # bypass verify(): keep only the NUMERIC parses and compare them for exact equality,
                # which correctly gives 50% == 1/2 (0.5 == 0.5) True and 50% == 50 False. (The LaTeX
                # form "50\\%" bare-parses to 50 -- a separate pre-existing math_verify limitation --
                # but no benchmark has a percent gold, so neither case fires on real eval data.)
                g_nums = [x for x in _mv_parse(gold) if not isinstance(x, str)]
                p_nums = [x for x in _mv_parse(span) if not isinstance(x, str)]
                return any(_num_equal(a, b) for a in g_nums for b in p_nums)
            # Non-percent: WRAP FIRST so a juxtaposed multi-term span/gold ('100\\pi - 200',
            # '5\\sqrt{5}') binds as ONE expression, never reduced to a trailing/leading term.
            g = _mv_parse_gold(gold)
            p = _mv_parse_span(span)
            return bool(g and p and _mv_verify(g, p))
    except Exception:
        return False


def _ds_ok(pred_answer: str, gold: str, timeout: float) -> bool:
    if not _HAVE_DS:
        return False
    g = gold
    if "\\boxed" in g:
        g2 = extract_boxed_answer(g)
        if g2 is not None:
            g = g2
    try:
        with _time_limit(timeout):
            return bool(grade_answer_mathd(pred_answer, g) or grade_answer_sympy(pred_answer, g))
    except Exception:
        return False


def check_math(model_output: str, gold: str, timeout: float = 8.0) -> bool:
    """True iff the generation's final answer is mathematically equal to `gold`.

    `gold` may be a bare answer ("34", "\\frac{1}{2}") or itself boxed / "Answer:"-wrapped.
    """
    if gold is None:
        return False
    gold = str(gold).strip()
    if not gold or not model_output:
        return False
    # Extract the predicted answer once (MARKER-based: last \boxed{} first, then "Answer:"; NO
    # last-number fallback). Used both for the percent guard and as engine 2's input.
    pred = extract_answer_robust(model_output)
    if pred is not None:
        m = re.fullmatch(r"\\?boxed\{(.+)\}", pred.strip())
        if m:
            pred = m.group(1).strip()
    # PERCENT GUARD: a percent answer and a plain number are DIFFERENT quantities (50% = 0.5 != 50).
    # Both the Minerva scorer (engine 1) and the DeepScaler normalizer (engine 2's sympy grade) strip
    # the '%' and would wrongly credit "50%"==50. math_verify handles it correctly (50% -> 0.5). So
    # when EXACTLY ONE of pred/gold is a percent, we trust ONLY math_verify's span check and skip the
    # two lenient engines. (Percent gold vs percent pred still matches; plain vs plain is unaffected.)
    percent_mismatch = pred is not None and (_is_percent(pred) != _is_percent(gold))

    # engine 1: DAPO Minerva scorer -- identical to the training/eval pipeline (rm_type=dapo)
    if _HAVE_DAPO and not percent_mismatch:
        try:
            with _time_limit(timeout):
                ok, _ = is_correct_minerva(model_output, gold)
                if ok:
                    return True
        except Exception:
            pass
    # NB: we deliberately do NOT run math_verify on the RAW generation with its own greedy any-match
    # extraction -- that grabs a stray number/percent from anywhere in the text and treats "50%"==50,
    # a give-up "...34" ==34 (verified false-positives). ALL grading goes through MARKER-BASED
    # extraction (engine 1's Minerva extractor above; engine 2 below runs sympy/math_verify only on
    # the extracted span). This matches the official graders, which require the boxed / "Answer:"
    # marker and never parse the whole solution greedily.
    #
    # engine 2: robust MARKER extraction -> DeepScaler sympy grade + math_verify on the extracted span
    #           (credits fraction/expression equivalence the pure-normalization engine 1 can miss)
    if pred is None:
        return False
    if not percent_mismatch and _ds_ok(pred, gold, timeout):
        return True
    if _mv_ok_span(pred, gold, timeout):      # span-mode: parse the whole extracted answer, no sub-extract
        return True
    return False


if __name__ == "__main__":
    import json
    print(json.dumps({"have_math_verify": _HAVE_MV, "have_deepscaler": _HAVE_DS, "have_dapo": _HAVE_DAPO}))
