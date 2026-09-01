"""Student prompt and token normalization for the BPM teacher path: preserve sampled
token ids, strip duplicate EOS text, append the training EOS, report the counters.
"""

from __future__ import annotations

import logging

from slime.utils.types import Sample
from ..args.bpm_utils import get_student_tokenizer

logger = logging.getLogger(__name__)


def opd_student_prompt_text(self, sample: Sample) -> str:
    """Return the exact student-side prompt text used for tokenization."""
    tokenizer = get_student_tokenizer(self.args)
    if isinstance(sample.prompt, list):
        tools = sample.metadata.get("tools") if isinstance(sample.metadata, dict) else None
        return tokenizer.apply_chat_template(
            sample.prompt,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **(getattr(self.args, "apply_chat_template_kwargs", None) or {}),
        )
    if isinstance(sample.prompt, str):
        return sample.prompt
    raise ValueError(f"[OPD] Unexpected prompt type: {type(sample.prompt)}")


def strip_trailing_special_text(text: str, special: str | None) -> tuple[str, int]:
    """Strip terminal copies of `special` from `text`; return (text, count). BPM assumes
    the response carries no EOS and adds one exactly once during tokenization.
    """
    if not text or not special:
        return text or "", 0
    count = 0
    out = text
    while out.rstrip().endswith(special):
        out = out.rstrip()
        out = out[: -len(special)]
        count += 1
    return out, count


def alignment_response_text(sample: Sample, eos_token: str, mask_eos_token: str) -> str:
    """Return the special-token-preserving response text used for byte alignment: the raw
    text decoded from the sampled ids, not sample.response (skip_special_tokens=True).
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    response_text = metadata.get("_opd_response_text_for_alignment")
    if not isinstance(response_text, str):
        response_text = sample.response or ""
    response_text, _ = strip_trailing_special_text(response_text, eos_token)
    if mask_eos_token != eos_token:
        response_text, _ = strip_trailing_special_text(response_text, mask_eos_token)
    return response_text


def _student_stop_ids_for_text(self) -> set[int]:
    """Full student stop/turn-end id set, from generation_config and the chat template.

    tokenizer.eos_token_id alone is wrong for chat students. If the realized stop
    token's text leaks into the alignment text the teacher scores it as content.
    """
    cached = getattr(self.args, "_opd_student_stop_ids_text", None)
    if cached is not None:
        return set(int(x) for x in cached)
    stop: set[int] = set()
    eos_id = getattr(get_student_tokenizer(self.args), "eos_token_id", None)
    if eos_id is not None:
        stop.add(int(eos_id))
    try:
        # imported lazily so the rollout process does not load the torch-heavy loss package
        from ..backend.special_tokens import detect_stop_token_ids

        stop |= {
            int(t)
            for t in detect_stop_token_ids(
                get_student_tokenizer(self.args), model_path=getattr(self.args, "hf_checkpoint", None)
            )
        }
    except Exception as exc:  # noqa: BLE001 - fall back to eos-only rather than crash rollout
        logger.warning(
            f"[OPD] student stop-token detection failed ({exc!r}); "
            "alignment text stripping falls back to tokenizer eos only"
        )
    self.args._opd_student_stop_ids_text = tuple(sorted(stop))
    return stop


def normalize_opd_student_tokens(self, data: list[Sample]) -> dict[str, float]:
    """Preserve rollout-sampled student token ids and add the training EOS if needed.

    The engine's output_token_logprobs ids are canonical. Re-encoding prompt+response
    could change the sampled ids and make the rollout log-probs stale.
    """
    from ..args.bpm_utils import is_bpm_enabled

    if not is_bpm_enabled(self.args) or not data:
        return {}

    tokenizer = get_student_tokenizer(self.args)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token_id is None or eos_token is None:
        raise ValueError("[OPD] tokenizer eos_token/eos_token_id is required")

    changed = 0
    eos_appended = 0
    stripped_text_eos = 0
    rollout_logprob_extends = 0
    rollout_logprob_drops = 0
    old_response_lens: list[int] = []
    new_response_lens: list[int] = []

    for sample in data:
        old_tokens = list(sample.tokens or [])
        old_response_len = int(sample.response_length)
        old_response_lens.append(old_response_len)

        response_text, n_stripped = strip_trailing_special_text(sample.response or "", eos_token)
        stripped_text_eos += n_stripped
        if n_stripped:
            sample.response = response_text

        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["_opd_old_response_length"] = old_response_len
        full_ids = old_tokens
        response_length = old_response_len
        appended_eos = False
        if not full_ids or full_ids[-1] != int(eos_token_id):
            full_ids = full_ids + [int(eos_token_id)]
            response_length += 1
            appended_eos = True
            eos_appended += 1

        if response_length < 0 or response_length > len(full_ids):
            raise ValueError(
                f"[OPD] invalid rollout response_length={response_length} "
                f"for token sequence length={len(full_ids)}"
            )

        sample.metadata["_opd_response_length"] = response_length
        sample.metadata["_opd_preserved_rollout_token_ids"] = True

        old_loss_mask = list(sample.loss_mask) if sample.loss_mask is not None else None
        if old_loss_mask is None:
            loss_mask = [1] * response_length
        elif len(old_loss_mask) == response_length:
            loss_mask = old_loss_mask
        elif appended_eos and len(old_loss_mask) == old_response_len:
            loss_mask = old_loss_mask + [1]
        else:
            logger.warning(
                f"[OPD] resetting loss_mask because length {len(old_loss_mask)} "
                f"does not match response_length={response_length}"
            )
            loss_mask = [1] * response_length

        if old_tokens != full_ids or old_response_len != response_length or old_loss_mask != loss_mask:
            changed += 1

        sample.tokens = full_ids
        sample.response_length = response_length
        sample.loss_mask = loss_mask
        response_ids_for_text = full_ids[-response_length:] if response_length > 0 else []
        # strip every trailing stop id, not just tokenizer.eos: a chat sequence can end
        # [..., <role-stop>, <eos>] and both must stay out of the alignment text
        _stop_text_ids = _student_stop_ids_for_text(self)
        while response_ids_for_text and int(response_ids_for_text[-1]) in _stop_text_ids:
            response_ids_for_text = response_ids_for_text[:-1]
        try:
            raw_response_text = tokenizer.decode(
                response_ids_for_text,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            raw_response_text = tokenizer.decode(response_ids_for_text, skip_special_tokens=False)
        raw_response_text, _ = strip_trailing_special_text(raw_response_text, eos_token)
        sample.metadata["_opd_response_text_for_alignment"] = raw_response_text

        if sample.rollout_log_probs is not None:
            old_logprob_len = len(sample.rollout_log_probs)
            if old_logprob_len == response_length:
                sample.rollout_log_probs = list(sample.rollout_log_probs)
                sample.metadata["_opd_rollout_log_probs_mask"] = [1] * response_length
            elif appended_eos and old_logprob_len == old_response_len:
                # the appended EOS was not sampled, so carry a mask for PG-side consumers; the
                # standalone loss still sees the normal loss_mask and can distill EOS
                sample.rollout_log_probs = list(sample.rollout_log_probs) + [0.0]
                sample.metadata["_opd_rollout_log_probs_mask"] = [1] * old_logprob_len + [0]
                rollout_logprob_extends += 1
            else:
                # do not synthesize zero logprobs for existing tokens: that makes stale
                # rollout_log_probs diagnostics look valid
                sample.rollout_log_probs = None
                sample.metadata.pop("_opd_rollout_log_probs_mask", None)
                rollout_logprob_drops += 1
        new_response_lens.append(response_length)

    if rollout_logprob_drops:
        logger.info(
            "[OPD] rollout_log_probs dropped for the whole batch: "
            f"{rollout_logprob_drops} sample(s) changed tokenization beyond a safe EOS append"
        )
        for sample in data:
            sample.rollout_log_probs = None

    if changed or eos_appended or stripped_text_eos or rollout_logprob_extends or rollout_logprob_drops:
        logger.info(
            f"[OPD] preserved rollout student token ids for {len(data)} samples: "
            f"changed={changed}, eos_appended={eos_appended}, "
            f"stripped_text_eos={stripped_text_eos}, "
            f"rollout_logprob_extends={rollout_logprob_extends}, "
            f"rollout_logprob_drops={rollout_logprob_drops}, "
            f"old_resp_len min/max/mean="
            f"{min(old_response_lens)}/{max(old_response_lens)}/{sum(old_response_lens)/len(old_response_lens):.1f}, "
            f"new_resp_len min/max/mean="
            f"{min(new_response_lens)}/{max(new_response_lens)}/{sum(new_response_lens)/len(new_response_lens):.1f}"
        )
    metrics = {
        "rollout/opd_tokenization_changed": float(changed),
        "rollout/opd_eos_appended": float(eos_appended),
        "rollout/opd_stripped_text_eos": float(stripped_text_eos),
        "rollout/opd_rollout_logprob_extends": float(rollout_logprob_extends),
        "rollout/opd_rollout_logprob_drops": float(rollout_logprob_drops),
        "rollout/opd_pg_retokenized_beyond_eos": 0.0,
    }
    return metrics
