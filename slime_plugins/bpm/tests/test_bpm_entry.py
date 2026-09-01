"""CPU acceptance test for the BPM backend + entry shim.

Drives bpm_custom_loss_function end to end on a tiny cross-tokenizer batch under a
single-rank gloo parallel state, and checks the 4->5 arg shim.
Run: python3 slime_plugins/bpm/tests/test_bpm_entry.py
"""

from __future__ import annotations

import math
import os
import sys
from argparse import Namespace

import torch

# repo root, so the plugin package imports when run as a script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _init_single_rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    from megatron.core import parallel_state as mpu

    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


class _ByteLevelPreTok:
    def __repr__(self):
        return "ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)"


class _Backend:
    pre_tokenizer = _ByteLevelPreTok()
    decoder = _ByteLevelPreTok()


class MockTokenizer:
    """Minimal ByteLevel-BPE-like tokenizer over an ASCII vocab."""

    def __init__(self, vocab: dict[str, int], eos_id: int):
        self._vocab = dict(vocab)
        self._inv = {v: k for k, v in vocab.items()}
        self.eos_token_id = eos_id
        self.eos_token = self._inv[eos_id]
        self.bos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None
        self.all_special_ids = [eos_id]
        self.chat_template = None
        self.backend_tokenizer = _Backend()

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._inv[int(ids)]
        return [self._inv[int(i)] for i in ids]

    def decode(self, ids, **kwargs):
        return "".join(self._inv[int(i)] for i in ids)

    apply_chat_template = None


def _build_lm_head(hidden: int, vocab: int) -> torch.nn.Linear:
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    torch.nn.init.normal_(lm_head.weight, std=0.02)
    lm_head.eval()
    lm_head.requires_grad_(False)
    lm_head._opd_vocab_size = vocab
    lm_head._opd_padded_vocab_size = vocab
    lm_head._opd_shard_vocab_size = vocab
    lm_head._opd_vocab_start = 0
    lm_head._opd_vocab_end = vocab
    return lm_head


def _build_args_batch():
    torch.manual_seed(0)
    hidden, vocab = 8, 3  # {"a":0, "b":1, "<eos>":2}
    vocab_map = {"a": 0, "b": 1, "<eos>": 2}
    eos = 2

    student_tok = MockTokenizer(vocab_map, eos)
    teacher_tok = MockTokenizer(vocab_map, eos)
    lm_head = _build_lm_head(hidden, vocab)

    args = Namespace()
    args.calculate_per_token_loss = False
    # Pre-cache so no checkpoint/tokenizer loading happens on CPU.
    args._bpm_teacher_lm_head = lm_head
    args._bpm_teacher_tokenizer_cache = teacher_tok
    args._bpm_student_tokenizer_cache = student_tok

    prompt_len, response_len = 1, 2
    total_len = prompt_len + response_len
    # student tokens 'a','b' (bytes 'ab'); teacher labels decode to the same bytes
    unconcat = torch.tensor([9, 0, 1], dtype=torch.long)  # [prompt, a, b] (prompt id unused)
    teacher_token_ids = torch.tensor([9, 0, 1], dtype=torch.long)  # labels = [0,1,eos] -> "ab"
    teacher_hidden = torch.randn(3, hidden, dtype=torch.float32)

    batch = {
        "response_lengths": [response_len],
        "total_lengths": [total_len],
        "loss_masks": [torch.ones(response_len, dtype=torch.float32)],
        "unconcat_tokens": [unconcat],
        "teacher_token_ids": [teacher_token_ids],
        "teacher_hidden_states": [teacher_hidden],
        "rollout_mask_sums": [torch.tensor(float(response_len))],
    }
    # logits are wider than the tokenizer vocab (Megatron padding), so every row
    # has a non-empty complement
    logits = torch.randn(1, total_len, vocab + 2, dtype=torch.float32)
    return args, batch, logits


def test_entry_returns_finite_scalar_and_bpm_key():
    _init_single_rank()
    from slime_plugins.bpm.entry.custom_loss import bpm_custom_loss_function
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    args, batch, logits = _build_args_batch()
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"],
        batch["rollout_mask_sums"], args.calculate_per_token_loss,
    )
    loss, log = bpm_custom_loss_function(args, batch, logits, sum_of_sample_mean)

    assert torch.is_tensor(loss) and loss.dim() == 0, f"loss not scalar: {loss}"
    assert math.isfinite(float(loss.item())), f"loss not finite: {loss}"
    # byte-matched samples route through fast rows; a huge CE would mean an edge case
    assert abs(float(loss.item())) < 1e4, f"loss magnitude unexpectedly large: {loss}"
    bpm_keys = [k for k in log if k.startswith("bpm_")]
    assert bpm_keys, f"no bpm_* metric key in {list(log)}"
    for k, v in log.items():
        assert math.isfinite(float(v)), f"metric {k} not finite: {v}"
    return float(loss.item()), sorted(log)


def test_shim_sum_of_sample_matches_reference():
    """The reconstructed ``sum_of_sample`` equals the per-sample masked token sum."""
    _init_single_rank()
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    args, batch, _ = _build_args_batch()
    sum_of_sample = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"],
        batch.get("rollout_mask_sums"), True,
    )
    x = torch.tensor([2.0, 3.0], dtype=torch.float32)  # per response-token values
    got = float(sum_of_sample(x).item())
    ref = float((x * batch["loss_masks"][0]).sum().item())  # 2+3 = 5
    assert abs(got - ref) < 1e-9, f"shim reducer {got} != reference {ref}"
    return got, ref


if __name__ == "__main__":
    _init_single_rank()
    passed = 0
    failed = 0
    for name, fn in [
        ("entry_finite_scalar_and_bpm_key", test_entry_returns_finite_scalar_and_bpm_key),
        ("shim_sum_of_sample_matches_reference", test_shim_sum_of_sample_matches_reference),
    ]:
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
