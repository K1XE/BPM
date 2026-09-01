"""Token-piece span alignment for the SimCT baseline."""

from __future__ import annotations

from argparse import Namespace

from slime_plugins.bpm.backend.alignment import (
    _bpm_ids_to_byte_texts as _ids_to_byte_texts,
    _bpm_ids_to_decoded_texts as _ids_to_decoded_texts,
    _bpm_ids_to_pieces as _ids_to_pieces,
    _bpm_use_byte_alignment as _use_byte_alignment,
)


def _simct_ids_to_alignment_texts(tokenizer, token_ids: list[int], use_byte: bool) -> list[str]:
    """Texts for SimCT span alignment.  ``use_byte`` MUST be the joint decision
    (both tokenizers ByteLevel) so teacher and student use the same byte-exact
    representation; otherwise both fall back to the legacy per-token decode.
    """
    if use_byte:
        return _ids_to_byte_texts(tokenizer, token_ids)
    return _ids_to_decoded_texts(tokenizer, token_ids)

def _get_simct_piece_cache(args: Namespace, attr_name: str) -> dict[int, str]:
    cache = getattr(args, attr_name, None)
    if cache is None:
        cache = {}
        setattr(args, attr_name, cache)
    return cache

def _simct_pieces_from_cache(
    tokenizer, token_ids: list[int], cache: dict[int, str], use_byte: bool
) -> list[str]:
    missing = [int(tid) for tid in token_ids if int(tid) not in cache]
    if missing:
        # Byte-exact for ByteLevel BPE (avoids the U+FFFD desync that silently
        # dropped the response tail from the KD loss); legacy decode otherwise.
        # ``use_byte`` is the JOINT both-ByteLevel decision so both sides match.
        decoded = _simct_ids_to_alignment_texts(tokenizer, missing, use_byte)
        cache.update({int(tid): text for tid, text in zip(missing, decoded, strict=False)})
    return [cache[int(tid)] for tid in token_ids]

def _align_simct_texts_with_spans(
    tea_texts: list[str],
    stu_texts: list[str],
    tea_eos: str | None,
    stu_eos: str | None,
    isolate_terminal_eos: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Return SimCT minimal aligned units from already-decoded token texts.

    Keep the same greedy cumulative-text semantics as the reference SimCT
    alignment, but track only the unmatched suffix since the last boundary.  The
    previous implementation appended every token piece to full-history Python
    strings and compared those growing strings on every iteration.  With 32k
    responses that makes alignment quadratic in generated characters and can
    dominate full-vocab simct training before any GPU KL work starts.
    """
    if len(tea_texts) == 0 or len(stu_texts) == 0:
        return []

    i = j = 0
    pending_tea_parts: list[str] = []
    pending_stu_parts: list[str] = []
    pending_tea_len = 0
    pending_stu_len = 0
    boundaries: list[tuple[int, int]] = []

    def pending_equal() -> bool:
        if pending_tea_len != pending_stu_len:
            return False
        if pending_tea_len == 0:
            return True
        return "".join(pending_tea_parts) == "".join(pending_stu_parts)

    def reset_pending() -> tuple[list[str], list[str], int, int]:
        return [], [], 0, 0

    while i < len(tea_texts) and j < len(stu_texts):
        tea_piece = tea_texts[i]
        stu_piece = stu_texts[j]
        is_eos_match = tea_piece == tea_eos and stu_piece == stu_eos
        if pending_equal() and (tea_piece == stu_piece or is_eos_match):
            boundaries.append((i, j))
            i += 1
            j += 1
            pending_tea_parts, pending_stu_parts, pending_tea_len, pending_stu_len = reset_pending()
        elif pending_tea_len > pending_stu_len:
            pending_stu_parts.append(stu_piece)
            pending_stu_len += len(stu_piece)
            j += 1
        elif pending_tea_len < pending_stu_len:
            pending_tea_parts.append(tea_piece)
            pending_tea_len += len(tea_piece)
            i += 1
        else:
            pending_tea_parts.append(tea_piece)
            pending_stu_parts.append(stu_piece)
            pending_tea_len += len(tea_piece)
            pending_stu_len += len(stu_piece)
            i += 1
            j += 1

    segments: list[tuple[int, int, int, int]] = []
    for boundary_idx, (tea_end_inclusive, stu_end_inclusive) in enumerate(boundaries):
        if boundary_idx == 0:
            tea_start = 0
            stu_start = 0
        else:
            tea_start = boundaries[boundary_idx - 1][0] + 1
            stu_start = boundaries[boundary_idx - 1][1] + 1
        tea_end = tea_end_inclusive + 1
        stu_end = stu_end_inclusive + 1
        if tea_start < tea_end and stu_start < stu_end:
            segments.append((tea_start, tea_end, stu_start, stu_end))

    # Terminal-EOS isolation (opt-in; off by default = byte-identical legacy behavior).
    # A monotonic aligner CANNOT split the trailing EOS off when the last content token
    # tokenizes differently across tokenizers (e.g. teacher "25" vs student "2","5"):
    # with no shared internal boundary the pending text only re-syncs AT the eos-match,
    # so the EOS is folded into the final content span. That hides the stop signal from
    # the dedicated EOS column (the stop-bridge then never reaches ~21% of terminals).
    # Both sides ALWAYS append exactly one terminal EOS and the content before it is
    # byte-identical, so we can safely peel the last EOS token of the final span into
    # its own 1:1 segment. We only touch the LAST segment and only when it actually
    # ends in EOS on both sides and leaves non-degenerate content -> no other alignment
    # is affected.
    if isolate_terminal_eos and segments and tea_eos is not None and stu_eos is not None:
        ts, te, ss, se = segments[-1]
        is_span = (te - ts) > 1 or (se - ss) > 1
        ends_in_eos = (
            0 <= te - 1 < len(tea_texts)
            and 0 <= se - 1 < len(stu_texts)
            and tea_texts[te - 1] == tea_eos
            and stu_texts[se - 1] == stu_eos
        )
        content_nondegenerate = (te - 1) > ts and (se - 1) > ss
        if is_span and ends_in_eos and content_nondegenerate:
            segments[-1] = (ts, te - 1, ss, se - 1)  # content span, EOS removed
            segments.append((te - 1, te, se - 1, se))  # terminal EOS as its own 1:1 segment
    return segments

def _align_simct_sequences_with_spans(
    teacher_token_ids: list[int],
    student_token_ids: list[int],
    teacher_tokenizer,
    student_tokenizer,
) -> list[tuple[int, int, int, int]]:
    """Return SimCT minimal aligned units as token spans.

    This mirrors SimCT ``span_ctkd``: greedily align cumulative decoded text and
    convert every matched boundary into a segment
    ``(tea_start, tea_end, stu_start, stu_end)``.  1:1 segments train like the
    old simple_ctkd path; non-1:1 segments become virtual span tokens.
    """
    use_byte = _use_byte_alignment(teacher_tokenizer, student_tokenizer)
    return _align_simct_texts_with_spans(
        _simct_ids_to_alignment_texts(teacher_tokenizer, teacher_token_ids, use_byte),
        _simct_ids_to_alignment_texts(student_tokenizer, student_token_ids, use_byte),
        teacher_tokenizer.eos_token,
        student_tokenizer.eos_token,
    )
