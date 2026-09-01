#!/usr/bin/env python3
"""Extract the submitted program from a model generation (think-mode friendly).

Aligned with EvalPlus's sanitize philosophy: return only a syntactically-valid Python program, never
a prose-filled block. Reasoning models put the final program in the post-</think> answer section and
sometimes wrap earlier *reasoning* in a ```python fence; grabbing that prose fence SyntaxErrors an
otherwise-correct submission (a false-negative). So we (a) look after the last </think> first,
(b) prefer the LAST fence whose body actually ast-parses, and (c) salvage the longest parseable
prefix when a block has trailing prose appended after the code.
"""
from __future__ import annotations
import ast
import re

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CODE_START = re.compile(r"(?m)^\s*(def |class |import |from |print\(|for |while |if |@|async def )")


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def _salvage(code: str) -> str | None:
    """Longest ast-parseable prefix: drop trailing lines until it parses (handles prose/tests the
    model appended AFTER a complete program). None if nothing non-empty parses."""
    if not code:
        return None
    if _parses(code):
        return code
    lines = code.split("\n")
    for end in range(len(lines), 0, -1):
        cand = "\n".join(lines[:end]).rstrip()
        if cand and _parses(cand):
            return cand
    return None


def extract_code(text: str) -> str | None:
    """Return the submitted program: the LAST syntactically-valid fenced block (python-tagged
    preferred), else a salvageable raw body. ANSI-stripped, think-mode aware."""
    if not text:
        return None
    text = _ANSI.sub("", text)
    # reasoning models: the final program lives after the last </think>. Only switch to that tail if
    # it actually carries code, so a truncated/odd generation still falls back to the whole text.
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[1]
        if _FENCE.search(tail) or _CODE_START.search(tail):
            text = tail
    blocks = _FENCE.findall(text)
    if blocks:
        py = [body for lang, body in blocks if lang.lower() in ("python", "py", "python3", "")]
        cands = py or [body for _, body in blocks]
        # prefer the LAST block that ast-parses cleanly (skip prose-filled reasoning fences)
        for body in reversed(cands):
            body = body.strip("\n")
            if _parses(body):
                return body
        # none parse: salvage the longest parseable prefix of the last candidate
        return _salvage(cands[-1].strip("\n"))
    # no fences: accept a raw body that looks like a program, salvaging trailing prose
    if _CODE_START.search(text):
        return _salvage(text.strip("\n"))
    return None


if __name__ == "__main__":
    demo = "reasoning...\n```python\nprint(sum(map(int, input().split())))\n```\ndone"
    print(repr(extract_code(demo)))
