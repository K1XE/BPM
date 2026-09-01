# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from decimal import Decimal, InvalidOperation


_INVALID_PRED = "[INVALID]"
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def _is_terminal_answer_noise(text: str) -> bool:
    """Return true when trailing text is only wrappers or punctuation."""

    return _strip_answer_noise(str(text).replace("$", "")) == ""


def maybe_remove_boxed(s: str | None) -> str | None:
    """Remove a surrounding ``\\boxed{...}`` if present, otherwise return input.

    ``remove_boxed`` intentionally asserts the exact boxed shape.  The reward
    path needs a tolerant helper because AIME labels often arrive as
    ``\\boxed{204}``, while DAPO labels are plain integers.
    """

    if s is None:
        return None
    boxed = last_boxed_only_string(str(s))
    if boxed is None:
        return s
    start = str(s).rfind(boxed)
    prefix = str(s)[:start].strip().strip("$").strip()
    suffix = str(s)[start + len(boxed) :]
    if prefix or not _is_terminal_answer_noise(suffix):
        return s
    try:
        return remove_boxed(boxed)
    except Exception:
        return s


def _strip_answer_noise(answer: str) -> str:
    """Strip wrappers that are common after answer extraction.

    This deliberately strips only terminal punctuation/markers around the final
    answer span.  Internal decimal points/fractions are preserved:
    ``3.14.`` -> ``3.14`` and ``42.`` -> ``42``.
    """

    answer = str(answer).strip()
    # Some models include a closing sentence after the final numeric answer.
    answer = re.sub(r"\s*(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", answer)
    answer = answer.strip().strip("'\"`")
    while answer and answer[-1] in ".。．,，;；:":
        answer = answer[:-1].rstrip()
    return answer


def extract_answer_from_solution(solution_str: str, answer_pattern: str | None = None) -> str:
    """Extract the model's final answer from a full response.

    Candidate sources:
      1. Explicit ``Answer:``/``Final answer:``/``答案:`` lines.
      2. ``\\boxed{...}`` expressions anywhere in the response.

    The latest candidate by character position wins, which handles responses
    that revise an earlier answer and end with a boxed final answer.

    This keeps DAPO's original ``Answer:`` contract while avoiding systematic
    false negatives on AIME-style boxed-only completions.
    """

    solution_str = str(solution_str or "")
    patterns = []
    if answer_pattern:
        patterns.append(answer_pattern)
    patterns.extend(
        [
            r"(?im)(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)",
            r"(?im)(?:final\s+answer|answer)\s*(?:is|=)\s*([^\n]+)",
            r"(?m)(?:最终答案|答案)\s*(?:是|为|[:：])\s*([^\n]+)",
        ]
    )

    candidates: list[tuple[int, str]] = []
    for pattern in patterns:
        candidates.extend((m.start(), m.group(1)) for m in re.finditer(pattern, solution_str))

    boxed = last_boxed_only_string(solution_str)
    if boxed is not None:
        boxed_start = solution_str.rfind(boxed)
        boxed_suffix = solution_str[boxed_start + len(boxed) :]
        try:
            if _is_terminal_answer_noise(boxed_suffix):
                candidates.append((boxed_start, remove_boxed(boxed)))
        except Exception:
            if _is_terminal_answer_noise(boxed_suffix):
                candidates.append((boxed_start, boxed))

    if candidates:
        return _strip_answer_noise(max(candidates, key=lambda item: item[0])[1])

    return _INVALID_PRED


# NOTE: a SIGALRM-based ``timeout`` context manager used to live here. It was DEAD CODE
# (zero call sites; grep confirmed) and, worse, a footgun: signal.alarm/SIGALRM only fires on
# the MAIN thread, so it silently no-ops when the reward runs off-thread (as it does in the
# rollout event loop / worker threads) — it could never have bounded the O(n^2) regex below.
# The real, GIL-proof wall-clock cap for the reward path is the killable worker PROCESS in
# examples/bpm/reward/killable_worker.py (driven by reward_adapter._run_killable), which SIGKILLs a
# stuck computation instead of relying on a signal that a threaded/async caller never receives.


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
    "<|im_end|>",
    "<|endoftext|>",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = _strip_answer_noise(str(final_answer))
    final_answer = final_answer.split("=")[-1]

    # Strip LaTeX inline/display math delimiters so a correct answer wrapped like
    # \(11\) or \[11\] normalizes to 11 (the $...$ delimiters are handled below).
    for _delim in ("\\(", "\\)", "\\[", "\\]"):
        final_answer = final_answer.replace(_delim, "")

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = maybe_remove_boxed(final_answer) or ""

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return _strip_answer_noise(final_answer)


def normalize_ground_truth(ground_truth: str, gt_need_extract: bool = False) -> str:
    """Normalize a DAPO/AIME ground-truth answer.

    DAPO's original scorer assumes integer labels.  Keep that behavior when the
    label is int-like, but do not crash for already-normalized boxed/string
    labels; returning the normalized string is safer for eval diagnostics.
    """

    if gt_need_extract:
        ground_truth = remove_boxed(last_boxed_only_string(str(ground_truth)))
    else:
        ground_truth = maybe_remove_boxed(str(ground_truth)) or str(ground_truth)
    gt = normalize_final_answer(ground_truth)
    # Only integer-ize an int-VALUED label (204, 204.0 -> "204"). A previous
    # unconditional str(int(float(gt))) FLOORED genuine decimal/fraction gold
    # answers (1.25 -> "1", 326.5 -> "326", $18.90 -> "18"), corrupting ~7/500
    # MATH-500 labels: both a false-negative (correct decimal marked wrong) and a
    # false-positive (an integer guess == floor(gold) marked right).
    try:
        f = float(gt)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return gt


def _decimal_from_number(text: str) -> Decimal | None:
    text = str(text).replace(",", "")
    if not _NUMBER_RE.fullmatch(text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _split_percent_answer(answer: str | None) -> tuple[Decimal, bool] | None:
    """Parse a normalized answer as a number with an optional percent unit."""

    if answer is None:
        return None
    compact = re.sub(r"\s+", "", str(answer)).lower()
    compact = compact.replace("\\%", "%")
    compact = compact.replace("\\text{%}", "%")
    compact = compact.replace("\\text{percent}", "percent")
    compact = compact.replace("\\text{percentage}", "percentage")
    for suffix in ("percentage", "percent", "%"):
        if compact.endswith(suffix):
            value = _decimal_from_number(compact[: -len(suffix)])
            return (value, True) if value is not None else None
    value = _decimal_from_number(compact)
    return (value, False) if value is not None else None


def _numeric_equal(pred: str | None, gt: str | None) -> bool:
    """True iff pred and gt are the SAME plain number under surface-format variation the string
    compare misses: leading zeros (025==25), a leading + (+25==25), a signed zero (-0==0), and
    unicode/thin-space or comma thousands separators (1 000 / 1,000 == 1000). Only fires when BOTH
    sides parse cleanly as one Decimal, so it never mislabels fractions/expressions/sets (those stay
    on the existing paths) -> no new false positives. Live for the integer-labeled train reward."""
    from decimal import Decimal, InvalidOperation

    def _dec(s):
        if s is None:
            return None
        t = str(s).strip()
        for ch in (" ", " ", " ", " ", " ", ","):
            t = t.replace(ch, "")
        t = t.lstrip("+")
        try:
            return Decimal(t)
        except (InvalidOperation, ValueError):
            return None

    a, b = _dec(pred), _dec(gt)
    return a is not None and b is not None and a == b


def _answers_match(pred: str | None, gt: str | None) -> bool:
    if pred == gt:
        return True
    if _numeric_equal(pred, gt):
        return True
    pred_percent = _split_percent_answer(pred)
    gt_percent = _split_percent_answer(gt)
    if pred_percent is None or gt_percent is None:
        return False
    pred_value, pred_has_percent = pred_percent
    gt_value, gt_has_percent = gt_percent
    return (pred_has_percent or gt_has_percent) and pred_value == gt_value


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution.  Keep the original Answer-line format as
    # the highest-priority path, but fall back to boxed-only answers used by
    # many AIME prompts and model completions.
    extracted_answer = extract_answer_from_solution(solution_str, answer_pattern)
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    gt = normalize_ground_truth(gt, gt_need_extract=gt_need_extract)

    return _answers_match(pred, gt), pred


def is_correct_strict_box(pred: str, gt: str, pause_tokens_index: list[int] | None = None) -> tuple[int, str | None]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = _strip_answer_noise(remove_boxed(boxed_pred)) if boxed_pred is not None else None
    gt = normalize_ground_truth(gt)

    return 1 if _answers_match(extracted_pred, gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: list[int] | None = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: list[int] | None = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # INCIDENT GUARD: normalize_final_answer's
    # re.sub(r"(.*?)(\$)(.*?)(\$)(.*)") is O(n^2) on a long newline-free tail with <2 '$'
    # (measured: 90k chars = 35.6s, 131k = 68s INSIDE the rollout event loop -> whole-run freeze;
    # py-spy caught the frame live). Upstream slime bounds this with solution_str[-300:]
    # ("longest MATH-500 answer has 159 chars"). We keep a
    # more generous tail (answers are extracted from the mandated final "Answer:"/boxed line, which
    # always lives in the tail) while still bounding the regex to milliseconds.
    solution_str = solution_str[-2048:]

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
