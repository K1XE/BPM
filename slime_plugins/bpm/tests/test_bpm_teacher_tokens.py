"""GPU-free unit tests for the BPM teacher token/payload pure functions.

Covers teacher token-id alignment, the think-prefix seam, EOS/stop stripping and
payload validation. Only the import-light teacher modules are exercised.
Run: python3 slime_plugins/bpm/tests/test_bpm_teacher_tokens.py
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slime.utils.types import Sample  # noqa: E402
from slime_plugins.bpm.teacher import bpm_teacher_request as req  # noqa: E402
from slime_plugins.bpm.teacher import bpm_teacher_tokens as tok  # noqa: E402
from slime_plugins.bpm.teacher import bpm_teacher_writeback as wb  # noqa: E402


class MockTeacherTokenizer:
    """Char-level tokenizer: each character maps to a distinct id; multi-char tokens are given
    explicit ids so prompt/response boundaries can be hand-checked."""

    def __init__(self):
        self._vocab = {"a": 1, "b": 2, "c": 3, "d": 4, " ": 5, "P": 6, "Q": 7}
        self.eos_token = "<e>"
        self.eos_token_id = 0

    def __call__(self, text, return_tensors="np", add_special_tokens=False):
        ids = [self._vocab[ch] for ch in text]
        return {"input_ids": np.array([ids], dtype=np.int64)}


class MockStudentTokenizer:
    """Minimal student tokenizer for ``normalize_opd_student_tokens``: id->char decode + eos."""

    def __init__(self):
        self._inv = {1: "a", 2: "b", 3: "c", 7: "P", 9: "<s>", 0: "<e>"}
        self.eos_token = "<e>"
        self.eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._inv[int(i)] for i in ids)


def _manager(**args_kwargs):
    args = Namespace(**args_kwargs)
    return Namespace(args=args)  # a stand-in for the RolloutManager (`self`)


# ---- EOS / special-token stripping ----


def test_strip_trailing_special_text_roundtrip():
    text, n = tok.strip_trailing_special_text("hello<e><e>", "<e>")
    assert (text, n) == ("hello", 2), (text, n)
    assert tok.strip_trailing_special_text("hello", "<e>") == ("hello", 0)
    assert tok.strip_trailing_special_text("", "<e>") == ("", 0)
    assert tok.strip_trailing_special_text("hello", None) == ("hello", 0)
    return "ok"


def test_alignment_response_text_prefers_stashed():
    s = Sample()
    s.response = "raw<e>"
    s.metadata = {"_opd_response_text_for_alignment": "clean text"}
    assert tok.alignment_response_text(s, "<e>", "<e>") == "clean text"
    s.metadata = {}
    # falls back to sample.response with trailing eos stripped
    assert tok.alignment_response_text(s, "<e>", "<e>") == "raw"
    return "ok"


# ---- think-prefix seam dedup / insert ----


def test_think_seam_dedup():
    # prefill-think teacher x generation-think student: strip the prompt tag
    out = req._normalize_teacher_think_seam("<|assistant|><think>", "<think>reasoning</think>")
    assert out == "<|assistant|>", out
    return "ok"


def test_think_seam_insert():
    # generation-think teacher x prefill-think student: unopened close
    out = req._normalize_teacher_think_seam("<|assistant|>", "reasoning</think>answer")
    assert out == "<|assistant|><think>\n", out
    return "ok"


def test_think_seam_untouched():
    # both generation-think (prompt has no tag, response opens its own): unchanged.
    out = req._normalize_teacher_think_seam("<|assistant|>", "<think>reasoning</think>answer")
    assert out == "<|assistant|>", out
    # both prefill-think (prompt has tag, response does not open one): unchanged.
    out2 = req._normalize_teacher_think_seam("<|assistant|><think>", "reasoning answer")
    assert out2 == "<|assistant|><think>", out2
    return "ok"


# ---- teacher prefill boundary + token-id alignment ----


def test_build_prefill_boundary_and_row_alignment():
    mt = MockTeacherTokenizer()
    prompt_text, response_text = "Pa", "bc"  # prompt ids [6,1]; response ids [2,3]
    full_text, full_ids, prompt_len, resp_len, prefill_ids = req._build_bpm_prefill(
        prompt_text, response_text, mt, mt.eos_token, joined=False
    )
    assert full_ids == [6, 1, 2, 3, 0], full_ids  # prompt + response + eos
    assert prompt_len == 2, prompt_len
    assert resp_len == 3, resp_len  # 2 response tokens + appended eos
    assert prefill_ids == full_ids
    assert full_text == "Pabc<e>"

    # the row at prompt_len-1 predicts the first response token; eos is out
    mask = req._loss_mask(len(full_ids), prompt_len)
    assert mask.tolist() == [False, True, True, True, False], mask.tolist()
    assert int(mask.sum()) == resp_len - 1 + 1 - 0  # 3 predicting rows

    # these are at-row input ids, not next-token labels; backend/pg.py shifts
    # them. Emitting next-tokens here would double-shift the teacher labels.
    masked = wb.masked_teacher_token_ids([full_ids], [mask])[0]
    assert masked == [1, 2, 3], masked  # at-row input ids on rows 1..3 (shifted downstream)
    assert len(masked) == int(mask.sum())
    return full_ids, masked


def test_masked_teacher_token_ids_matches_mask():
    ids = [[10, 11, 12, 13, 14]]
    masks = [np.array([False, True, False, True, True])]
    out = wb.masked_teacher_token_ids(ids, masks)
    assert out == [[11, 13, 14]], out
    # a zero mask yields an empty label list (fully-masked skip)
    assert wb.masked_teacher_token_ids([[1, 2]], [np.array([False, False])]) == [[]]
    return "ok"


# ---- payload completeness validation ----


def test_ensure_payload_complete_raises():
    hs = [np.zeros((2, 4)), None]
    try:
        wb.ensure_teacher_payload_complete(hs, [[1], [2]])
        raise AssertionError("expected RuntimeError for missing hidden_states")
    except RuntimeError as e:
        assert "missing teacher hidden_states" in str(e)

    hs_ok = [np.zeros((2, 4)), np.zeros((1, 4))]
    try:
        wb.ensure_teacher_payload_complete(hs_ok, [[1], None])
        raise AssertionError("expected RuntimeError for missing token ids")
    except RuntimeError as e:
        assert "missing teacher_token_ids" in str(e)
    # complete payload validates cleanly
    wb.ensure_teacher_payload_complete(hs_ok, [[1], [2]])
    return "ok"


# ---- EOS append + stop stripping round trip ----


def test_normalize_appends_eos_and_strips_stop_ids():
    self = _manager(opd_mode="aux_loss", tokenizer=MockStudentTokenizer())
    # pre-seed the stop-id cache: turn-end 9 and eos 0 must stay out of the text
    self.args._opd_student_stop_ids_text = (0, 9)

    s = Sample()
    s.tokens = [7, 1, 2, 9]  # prompt P, response a b <s>(realized stop) -- no eos yet
    s.response_length = 3
    s.response = "ab"
    s.loss_mask = [1, 1, 1]
    s.metadata = {}
    s.rollout_log_probs = None

    metrics = tok.normalize_opd_student_tokens(self, [s])

    assert s.tokens[-1] == 0, s.tokens  # eos appended exactly once
    assert s.response_length == 4, s.response_length
    assert s.loss_mask == [1, 1, 1, 1], s.loss_mask
    assert metrics["rollout/opd_eos_appended"] == 1.0, metrics
    # alignment text drops the realized stop id and the appended eos
    assert s.metadata["_opd_response_text_for_alignment"] == "ab", s.metadata
    return s.metadata["_opd_response_text_for_alignment"]


def test_normalize_noop_when_disabled():
    self = _manager(opd_mode="off", tokenizer=MockStudentTokenizer())
    s = Sample()
    s.tokens = [7, 1]
    s.response_length = 1
    assert tok.normalize_opd_student_tokens(self, [s]) == {}
    assert s.tokens == [7, 1]  # untouched
    return "ok"


if __name__ == "__main__":
    tests = [
        ("strip_trailing_special_text_roundtrip", test_strip_trailing_special_text_roundtrip),
        ("alignment_response_text_prefers_stashed", test_alignment_response_text_prefers_stashed),
        ("think_seam_dedup", test_think_seam_dedup),
        ("think_seam_insert", test_think_seam_insert),
        ("think_seam_untouched", test_think_seam_untouched),
        ("build_prefill_boundary_and_row_alignment", test_build_prefill_boundary_and_row_alignment),
        ("masked_teacher_token_ids_matches_mask", test_masked_teacher_token_ids_matches_mask),
        ("ensure_payload_complete_raises", test_ensure_payload_complete_raises),
        ("normalize_appends_eos_and_strips_stop_ids", test_normalize_appends_eos_and_strips_stop_ids),
        ("normalize_noop_when_disabled", test_normalize_noop_when_disabled),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            out = fn()
            print(f"PASS {name} -> {out}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"FAIL {name} -> {exc!r}")
            failed += 1
    print(f"RESULT: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
