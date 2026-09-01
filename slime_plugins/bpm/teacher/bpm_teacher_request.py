"""BPM teacher prefill request construction.

Owns prompt formatting, teacher-tokenizer boundaries and loss-mask construction.
The teacher tokenizes with its own tokenizer, so the mask is over teacher tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from slime.utils.types import Sample

from . import bpm_teacher_tokens as teacher_tokens
from ..args.bpm_utils import get_student_tokenizer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TeacherPrefillBatch:
    prompts: list[str]
    prefill_input_ids: list[list[int] | None]
    loss_masks: list[np.ndarray]
    seq_lens: list[int]
    teacher_full_token_ids: list[list[int]]
    mask_eos_token: str


def get_teacher_mask_tokenizer(self):
    """Return (teacher_tokenizer, teacher_eos_text, student_eos_text), loaded once from
    --bpm-teacher-tokenizer-path (falling back to the model path) and cached.
    """
    tokenizer = get_student_tokenizer(self.args)
    eos_token = tokenizer.eos_token
    if eos_token is None:
        raise ValueError("[OPD] student tokenizer eos_token is required for teacher prompt construction")

    teacher_tokenizer_path = getattr(self.args, "bpm_teacher_tokenizer_path", None)
    if teacher_tokenizer_path is None:
        teacher_tokenizer_path = getattr(self.args, "opd_teacher_model_path", None)
    if getattr(self, "_teacher_tokenizer_for_mask", None) is None:
        from transformers import AutoTokenizer

        # trust_remote_code mirrors every serving path: a custom architecture must
        # load with the same code or its token boundaries disagree
        self._teacher_tokenizer_for_mask = AutoTokenizer.from_pretrained(
            teacher_tokenizer_path, trust_remote_code=True
        )
        logger.info(f"[OPD] loaded teacher tokenizer for loss_mask from {teacher_tokenizer_path}")
    mask_tokenizer = self._teacher_tokenizer_for_mask
    mask_eos_token = mask_tokenizer.eos_token
    if mask_eos_token is None:
        raise ValueError("[OPD] teacher tokenizer eos_token is required for teacher prompt construction")
    return mask_tokenizer, mask_eos_token, eos_token


def _teacher_prompt_text(self, sample: Sample, mask_tokenizer) -> str:
    """Format a prompt with the teacher tokenizer."""
    template_kwargs = getattr(self.args, "apply_chat_template_kwargs", None) or {}
    if getattr(self.args, "opd_backend", "") == "gold" and bool(getattr(self.args, "gold_trl_faithful", False)):
        # TRL GOLD gives the teacher the student-rendered prompt, no teacher chat template
        if isinstance(sample.prompt, str):
            return sample.prompt
        return get_student_tokenizer(self.args).apply_chat_template(
            sample.prompt,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    if isinstance(sample.prompt, list):
        return mask_tokenizer.apply_chat_template(
            sample.prompt,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    if isinstance(sample.prompt, str):
        if sample.metadata and "raw_messages" in sample.metadata:
            return mask_tokenizer.apply_chat_template(
                sample.metadata["raw_messages"],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        raise ValueError(
            "[OPD] sample.prompt is a pre-formatted string but metadata['raw_messages'] was "
            "not found. Refusing to feed a student-formatted prompt to the teacher tokenizer "
            "because it can shift teacher hidden/token ids (e.g. an extra thinking prefix). "
            "Run with --apply-chat-template so metadata['raw_messages'] is stored."
        )
    raise ValueError(f"[OPD] Unexpected prompt type: {type(sample.prompt)}")


def _teacher_eos_id(mask_tokenizer, mask_eos_token: str) -> int:
    teacher_eos_id = getattr(mask_tokenizer, "eos_token_id", None)
    if teacher_eos_id is not None:
        return int(teacher_eos_id)
    eos_ids = mask_tokenizer.encode(mask_eos_token, add_special_tokens=False)
    if len(eos_ids) != 1:
        raise ValueError(
            f"[OPD] teacher tokenizer has no eos_token_id and EOS text encodes to "
            f"{len(eos_ids)} ids; cannot build an unambiguous teacher label boundary."
        )
    return int(eos_ids[0])


def _joined_prefill_ids(prompt_text, response_text, mask_tokenizer, teacher_eos_id):
    """Tokenize the joined prompt+response once and derive the boundary by subtraction.

    Opt-in via --bpm-joined-prefill; mirrors the reference trainer. Returns None when
    the joined boundary is degenerate.
    """
    joined_enc = mask_tokenizer(prompt_text + response_text, return_tensors="np", add_special_tokens=False)
    resp_enc = mask_tokenizer(response_text, return_tensors="np", add_special_tokens=False)
    joined_ids = [int(t) for t in joined_enc["input_ids"][0].tolist()]
    resp_count = int(resp_enc["input_ids"].shape[1])
    if joined_ids and joined_ids[-1] == teacher_eos_id:
        joined_ids = joined_ids[:-1]
    prompt_len = len(joined_ids) - resp_count
    if prompt_len <= 0 or resp_count <= 0 or resp_count > len(joined_ids):
        return None
    return joined_ids, prompt_len, resp_count


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _normalize_teacher_think_seam(prompt_text: str, response_text: str) -> str:
    """Balance the thinking-open tag at the teacher's prompt/response seam.

    (a) prefill-think teacher x generation-think student: strip the prompt-side tag,
        else the teacher sees a doubled <think><think> prefix.
    (b) generation-think teacher x prefill-think student: append <think> so the
        teacher does not see a bare closing tag.

    Prompt-side edits are safe: byte alignment covers only the response text.
    """
    rstripped = prompt_text.rstrip()
    prompt_has_open = rstripped.endswith(_THINK_OPEN)
    resp_has_open = response_text.lstrip().startswith(_THINK_OPEN)
    if prompt_has_open and resp_has_open:
        return rstripped[: -len(_THINK_OPEN)]
    if (not prompt_has_open) and (not resp_has_open) and (_THINK_CLOSE in response_text):
        return prompt_text + _THINK_OPEN + "\n"
    return prompt_text


def _build_bpm_prefill(
    prompt_text: str,
    response_text: str,
    mask_tokenizer,
    mask_eos_token: str,
    joined: bool = False,
    prompt_add_special_tokens: bool = False,
):
    """Build exact teacher-tokenized prefill ids and boundary lengths. Prompt and response
    are tokenized separately then concatenated; --bpm-joined-prefill switches paths.
    """
    tea_full_text = prompt_text + response_text + mask_eos_token
    teacher_eos_id = _teacher_eos_id(mask_tokenizer, mask_eos_token)
    if joined:
        built = _joined_prefill_ids(prompt_text, response_text, mask_tokenizer, teacher_eos_id)
        if built is not None:
            prefix_ids, prompt_len, response_count = built
            full_token_ids = prefix_ids + [teacher_eos_id]
            return tea_full_text, full_token_ids, prompt_len, response_count + 1, full_token_ids
    # TRL adds special tokens to the prompt only, never to the completion
    prompt_enc = mask_tokenizer(prompt_text, return_tensors="np", add_special_tokens=bool(prompt_add_special_tokens))
    resp_enc = mask_tokenizer(response_text, return_tensors="np", add_special_tokens=False)
    prompt_token_ids = [int(t) for t in prompt_enc["input_ids"][0].tolist()]
    response_token_ids = [int(t) for t in resp_enc["input_ids"][0].tolist()]
    if prompt_token_ids and prompt_token_ids[-1] == teacher_eos_id:
        prompt_token_ids = prompt_token_ids[:-1]
    full_token_ids = prompt_token_ids + response_token_ids + [teacher_eos_id]
    prompt_len = len(prompt_token_ids)
    resp_len = len(response_token_ids) + 1
    return tea_full_text, full_token_ids, prompt_len, resp_len, full_token_ids


def _loss_mask(seq_len: int, prompt_len: int) -> np.ndarray:
    if prompt_len <= 0:
        raise ValueError(f"[OPD] invalid prompt_len={prompt_len}, seq_len={seq_len}")
    mask = np.zeros(seq_len, dtype=bool)
    # hidden_states[prompt_len-1] predicts the first response token; the final
    # EOS position predicts nothing
    mask[prompt_len - 1 : -1] = True
    return mask


def build_teacher_prefill_batch(self, data: list[Sample]) -> TeacherPrefillBatch:
    """Build prompts, exact teacher prefill ids, and teacher loss masks."""
    mask_tokenizer, mask_eos_token, eos_token = get_teacher_mask_tokenizer(self)
    _is_gold = getattr(self.args, "opd_backend", "") == "gold"
    prompts: list[str] = []
    prefill_input_ids: list[list[int] | None] = []
    loss_masks: list[np.ndarray] = []
    seq_lens: list[int] = []
    teacher_full_token_ids: list[list[int]] = []

    _diag_on = False
    for sample in data:
        prompt_text = _teacher_prompt_text(self, sample, mask_tokenizer)
        response_text = teacher_tokens.alignment_response_text(sample, eos_token, mask_eos_token)
        # balance the thinking-open tag at the seam (_normalize_teacher_think_seam)
        prompt_text = _normalize_teacher_think_seam(prompt_text, response_text)

        # stash the rendered prompt tail so the teacher-diag dump can verify it
        if _diag_on:
            if not isinstance(sample.metadata, dict):
                sample.metadata = {}
            sample.metadata["_opd_teacher_prompt_tail"] = prompt_text[-200:]

        tea_full_text, full_token_ids, prompt_len, resp_len, input_ids_for_prefill = _build_bpm_prefill(
            prompt_text,
            response_text,
            mask_tokenizer,
            mask_eos_token,
            # GOLD aligns on response-only bytes, so its ids must skip the joined path
            joined=(not _is_gold) and bool(getattr(self.args, "bpm_joined_prefill", False)),
            prompt_add_special_tokens=_is_gold and bool(getattr(self.args, "gold_trl_faithful", False)),
        )

        seq_len = len(full_token_ids)
        if prompt_len <= 0:
            raise ValueError(
                f"[OPD] invalid prompt_len={prompt_len}, seq_len={seq_len}, resp_len={resp_len}"
            )
        prompts.append(tea_full_text)
        prefill_input_ids.append(input_ids_for_prefill)
        loss_masks.append(_loss_mask(seq_len, prompt_len))
        seq_lens.append(seq_len)
        teacher_full_token_ids.append(full_token_ids)

    return TeacherPrefillBatch(
        prompts=prompts,
        prefill_input_ids=prefill_input_ids,
        loss_masks=loss_masks,
        seq_lens=seq_lens,
        teacher_full_token_ids=teacher_full_token_ids,
        mask_eos_token=mask_eos_token,
    )
