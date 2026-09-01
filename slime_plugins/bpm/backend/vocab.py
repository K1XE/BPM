"""Tokenizer caches and CP-local response-index helpers for the BPM backend."""

from __future__ import annotations

from megatron.core import mpu

from slime.backends.megatron_utils.cp_utils import get_logits_and_tokens_offset_with_cp


def _get_bpm_teacher_tokenizer(args):
    """Cached teacher tokenizer."""
    if not hasattr(args, "_bpm_teacher_tokenizer_cache"):
        from transformers import AutoTokenizer

        teacher_tokenizer_path = (
            getattr(args, "bpm_teacher_tokenizer_path", None)
            or getattr(args, "bpm_teacher_model_path", None)
        )
        if teacher_tokenizer_path is None:
            raise ValueError("bpm_teacher_tokenizer_path/bpm_teacher_model_path required for BPM")
        args._bpm_teacher_tokenizer_cache = AutoTokenizer.from_pretrained(
            teacher_tokenizer_path, trust_remote_code=True
        )
    return args._bpm_teacher_tokenizer_cache


def _get_bpm_student_tokenizer(args):
    """Cached student tokenizer."""
    if not hasattr(args, "_bpm_student_tokenizer_cache"):
        from transformers import AutoTokenizer

        student_tokenizer_path = getattr(args, "tokenizer_model", None)
        if student_tokenizer_path is None:
            raise ValueError("tokenizer_model required for student tokenizer")
        args._bpm_student_tokenizer_cache = AutoTokenizer.from_pretrained(
            student_tokenizer_path, trust_remote_code=True
        )
    return args._bpm_student_tokenizer_cache


def _get_bpm_local_response_indices(
    total_length: int, response_length: int, *, logits_shift: int = 1
) -> list[int]:
    """Global response-label indices owned by the current CP rank.

    get_responses yields logits for [prompt_len-1, total_length-1), mapping to response
    indices [0, response_length). Each CP rank owns two sliced chunks.
    """
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return list(range(response_length))

    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, logits_shift=logits_shift
    )
    base = prompt_length - logits_shift
    indices: list[int] = []
    for start, end in logits_offset:
        if start < end:
            indices.extend(range(start - base, end - base))
    return indices
