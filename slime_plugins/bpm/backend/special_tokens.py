"""Model-agnostic stop-token (assistant-turn-end) detection for the stopping bridge.

Turn-end surface strings differ across families, so the stop set is derived from the
model's own generation_config and chat template rather than per-model hardcoding.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


def _as_id_set(value) -> set[int]:
    """Coerce an int / list / tuple / set of token ids into a ``set[int]``."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        out: set[int] = set()
        for v in value:
            if v is None:
                continue
            try:
                out.add(int(v))
            except (TypeError, ValueError):
                continue
        return out
    try:
        return {int(value)}
    except (TypeError, ValueError):
        return set()


def _generation_config_eos_ids(model_path: str | None) -> set[int]:
    """``eos_token_id`` from the model's ``generation_config`` (list or scalar)."""
    if not model_path:
        return set()
    try:
        from transformers import GenerationConfig

        gen = GenerationConfig.from_pretrained(model_path)
    except Exception as exc:  # noqa: BLE001 -- missing/unsupported config is benign
        _logger.info("[OPD][bpm] no usable generation_config at %s (%s)", model_path, exc)
        return set()
    return _as_id_set(getattr(gen, "eos_token_id", None))


# a sentinel no chat template contains; the specials trailing it are turn-end
_TURN_END_SENTINEL = "ⓨⓨⓨ"


def _chat_template_turn_end_ids(tokenizer) -> set[int]:
    """Detect assistant-turn-end delimiter id(s) from the tokenizer's own chat template."""
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None or getattr(tokenizer, "chat_template", None) is None:
        return set()

    def _to_id_list(x) -> list[int]:
        if isinstance(x, dict):
            x = x.get("input_ids", [])
        try:
            if len(x) > 0 and isinstance(x[0], (list, tuple)):
                x = x[0]
        except TypeError:
            pass
        out: list[int] = []
        for t in x:
            try:
                out.append(int(t))
            except (TypeError, ValueError):
                return []
        return out

    try:
        rendered = _to_id_list(
            apply(
                [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": _TURN_END_SENTINEL},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        sentinel_ids = _to_id_list(tokenizer.encode(_TURN_END_SENTINEL, add_special_tokens=False))
    except Exception:  # noqa: BLE001 -- template that rejects this shape is fine
        return set()
    if not rendered or not sentinel_ids:
        return set()
    last = sentinel_ids[-1]
    try:
        pos = len(rendered) - 1 - rendered[::-1].index(last)
    except ValueError:
        return set()
    trailing = rendered[pos + 1 :]
    special_ids = set(int(x) for x in (getattr(tokenizer, "all_special_ids", None) or []))
    # Keep only structural (special/added) delimiters, not stray whitespace content.
    return {t for t in trailing if t in special_ids}


def detect_stop_token_ids(tokenizer, model_path: str | None = None) -> set[int]:
    """Return the model's full set of assistant-turn-terminating token ids: the union of
    tokenizer.eos_token_id, generation_config.eos_token_id, and template turn-ends.
    """
    stop: set[int] = set()
    stop |= _as_id_set(getattr(tokenizer, "eos_token_id", None))
    stop |= _generation_config_eos_ids(model_path)
    stop |= _chat_template_turn_end_ids(tokenizer)
    stop = {t for t in stop if t is not None and t >= 0}
    return stop
