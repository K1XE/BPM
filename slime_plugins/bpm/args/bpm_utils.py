"""Naming and config helpers shared by BPM argument parsing and runtime code, keeping
the canonical-name and enable-gate logic out of the parser and loss modules.
"""

from __future__ import annotations

from typing import Any


# None canonicalizes to bpm so an unset --opd-backend still resolves
BPM_BACKEND_ALIASES = {
    None: "bpm",
    "bpm": "bpm",
    "simct": "simct",
    "gold": "gold",
}

# Standalone BPM distillation loss reached through the ``custom_loss`` entry.
BPM_LOSS_TYPE_ALIASES = {
    "bpm_opd_loss": "bpm_opd_loss",
}

def canonical_bpm_backend(backend: str | None) -> str:
    """Return the canonical backend name used by BPM code paths."""
    return BPM_BACKEND_ALIASES.get(backend, backend)


def canonical_bpm_loss_type(loss_type: str | None) -> str | None:
    """Return the canonical standalone-loss type for BPM."""
    return BPM_LOSS_TYPE_ALIASES.get(loss_type, loss_type)


def is_bpm_enabled(args: Any) -> bool:
    """Whether this run needs the BPM teacher hidden-state payload / loss path."""
    return getattr(args, "opd_mode", "off") != "off"


def normalize_teacher_offload_tags(tags: Any) -> list[str] | None:
    """Normalize the teacher sleep/offload tag contract.

    SGLang uses tags=None for all regions and does not special-case a literal ['all'],
    so the CLI value `all` becomes None. Other values become explicit tag lists.
    """
    if tags is None:
        return None
    if isinstance(tags, str):
        normalized = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
    else:
        normalized = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
    if not normalized or "all" in normalized:
        return None
    return normalized


def teacher_offload_tags_release_weights(tags: Any) -> bool:
    """Whether sleeping with these tags frees teacher model weights.

    Full sleep and an explicit `weights` tag do. Partial tags such as kv_cache,cuda_graph
    are profiling modes: weights stay resident and can overlap colocated actors.
    """
    normalized = normalize_teacher_offload_tags(tags)
    if normalized is None:
        return True
    return "weights" in normalized


_STUDENT_TOKENIZER_CACHE: dict = {}


def get_student_tokenizer(args):
    """Return the student tokenizer: args.tokenizer when the host provides one, otherwise
    loaded and cached from --hf-checkpoint.
    """
    tok = getattr(args, "tokenizer", None)
    if tok is not None:
        return tok
    path = getattr(args, "hf_checkpoint", None)
    if not path:
        raise ValueError("[OPD] neither args.tokenizer nor args.hf_checkpoint is available")
    tok = _STUDENT_TOKENIZER_CACHE.get(path)
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _STUDENT_TOKENIZER_CACHE[path] = tok
    return tok
