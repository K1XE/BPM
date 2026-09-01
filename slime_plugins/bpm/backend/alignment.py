"""Byte-exact per-token text helpers for BPM byte-prefix alignment.

Both tokenizers are compared in the same ByteLevel-byte representation, so a
multi-byte char split across byte-tokens still reconstructs the same UTF-8 stream
(per-token independent decode desyncs on U+FFFD and drops the response tail).
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)
_warned_lossy_decode = False


def _bpm_ids_to_pieces(tokenizer, token_ids: list[int]) -> list[str]:
    """Tokenizer pieces for the given ids (fallback path only)."""
    try:
        return tokenizer.convert_ids_to_tokens(token_ids)
    except Exception:
        return [tokenizer.decode([tid]) for tid in token_ids]


def _bpm_ids_to_decoded_texts(tokenizer, token_ids: list[int]) -> list[str]:
    """Per-token independent decode, used only as the exception fallback: lossy for
    byte-level BPE, where a split multi-byte char decodes to U+FFFD chars.
    """
    global _warned_lossy_decode
    if not _warned_lossy_decode:
        _logger.warning(
            "byte-exact token decode unavailable; falling back to per-token decode, "
            "which is lossy for multi-byte characters split across byte tokens"
        )
        _warned_lossy_decode = True
    texts: list[str] = []
    for token_id in token_ids:
        try:
            texts.append(tokenizer.decode([int(token_id)], skip_special_tokens=False))
        except TypeError:
            texts.append(tokenizer.decode([int(token_id)]))
        except Exception:
            piece = _bpm_ids_to_pieces(tokenizer, [int(token_id)])[0]
            texts.append(piece.replace("▁", "").replace("Ġ", ""))
    return texts


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=True))


_BYTE_LEVEL_DECODER = {ch: b for b, ch in _bytes_to_unicode().items()}


def _is_byte_level_tokenizer(tokenizer) -> bool:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return False
    return "ByteLevel" in repr(getattr(backend, "pre_tokenizer", "")) or "ByteLevel" in repr(
        getattr(backend, "decoder", "")
    )


def _bpm_ids_to_byte_texts(tokenizer, token_ids: list[int]) -> list[str]:
    """Per-token raw-byte text (one latin-1 char per byte) for byte-exact alignment. A
    char outside the ByteLevel table falls back to the raw piece.
    """
    try:
        pieces = tokenizer.convert_ids_to_tokens([int(t) for t in token_ids])
    except Exception:
        return _bpm_ids_to_decoded_texts(tokenizer, token_ids)
    if isinstance(pieces, str):
        pieces = [pieces]
    out: list[str] = []
    for piece in pieces:
        if not piece:
            out.append("")
            continue
        decoded_bytes: list[str] = []
        ok = True
        for ch in piece:
            b = _BYTE_LEVEL_DECODER.get(ch)
            if b is None:
                ok = False
                break
            decoded_bytes.append(chr(b))
        out.append("".join(decoded_bytes) if ok else piece)
    return out


def _bpm_use_byte_alignment(teacher_tokenizer, student_tokenizer) -> bool:
    """True iff both tokenizers are ByteLevel BPE (required to compare in one representation)."""
    return bool(_is_byte_level_tokenizer(teacher_tokenizer)) and bool(
        _is_byte_level_tokenizer(student_tokenizer)
    )
