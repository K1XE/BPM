"""BPM Megatron entry: the Byte-Prefix Marginalization objective (production loss).

Forward-KL of the student's full-vocab next-token distribution against a per-position
target built from the teacher's distribution by byte-prefix marginalization. Requires
student TP=1. Under context parallelism only CP-local rows are trained and the CE sum
is reduced once with the CP-shared denominator.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import random
import re
import time
from argparse import Namespace
from bisect import bisect_left, bisect_right

import torch

from slime.utils.types import RolloutBatch

from ..backend.loss_helpers import get_responses
from .bpm_kernels import (
    apply_chain_scatter_deltas,
    build_fast_scatter_target,
    build_tail_scatter_target,
    delta_ce_sum_local,
    realized_merge_candidates,
    spanning_chain_parts,
    scatter_target_divergence_sum_local,
    sparse_target_divergence_sum_local,
)
from .bpm_metrics import bpm_repetition_fraction
from .bpm_phi import build_phi, student_first_token_capped
from .bpm_position import build_student_byte_trie, position_target
from .bpm_seg import Chunk
from .bpm_loss_utils import _reduce_cp_float_counts, _to_int_list
from ..backend.alignment import _bpm_ids_to_byte_texts
from ..backend.projection import _get_bpm_teacher_lm_head, _select_hidden_rows_to_device
from ..backend.vocab import (
    _get_bpm_local_response_indices,
    _get_bpm_student_tokenizer,
    _get_bpm_teacher_tokenizer,
)

_logger = logging.getLogger(__name__)


def _control_ids(tokenizer) -> set[int]:
    """Core control tokens that do not render as response-text bytes (EOS/BOS/PAD/UNK).

    Tokens that do render as literal text (</think>) are kept on purpose. Anything else
    non-rendering is caught by the byte-stream equality check, which skips the sample.
    """
    ids: set[int] = set()
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            ids.add(int(tid))
    return ids


# BPM_FILTER_BROKEN_CODE: skip broken-code-fence samples
_BPM_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.S)
_BPM_CODE_START_RE = re.compile(r"(?m)^\s*(def |class |import |from |print\(|for |while |if |@|async def )")
# a real code block opens with a line ending in ':'
_BPM_BLOCK_COLON_RE = re.compile(
    r"(?m)^\s*(def |class |for |while |if |elif |else|try|except|finally|with |async def ).*:[ \t]*$")


def _bpm_filter_broken_code_enabled() -> bool:
    return os.environ.get("BPM_FILTER_BROKEN_CODE", "").strip().lower() in ("1", "on", "true", "yes")


def _bpm_mask_ws_rows_enabled() -> bool:
    """BPM_MASK_WS_ROWS: drop every response position whose realized student token is an
    all-whitespace (space/tab) run from the BPM loss. Defaults on, the paper's standard
    configuration; set BPM_MASK_WS_ROWS=0 to train the unmasked ablation."""
    return os.environ.get("BPM_MASK_WS_ROWS", "on").strip().lower() not in ("0", "off", "false", "no")


def _bpm_mask_random_rows_enabled() -> bool:
    """BPM_MASK_RANDOM_ROWS: matched-fraction control for the WS-row ablation.

    Drops as many random content rows as the WS mask would have dropped, keeping the
    whitespace rows trained. MUST run with BPM_MASK_WS_ROWS=0: the two filters are
    independent, so both on drops both sets.
    """
    return os.environ.get("BPM_MASK_RANDOM_ROWS", "").strip().lower() in ("1", "on", "true", "yes")


def _bpm_perf_probe_enabled() -> bool:
    """BPM_PERF_PROBE: wall-clock attribution of the loss's own sections, appended to the
    per-step log line. Off by default: it inserts cuda.synchronize at each boundary.
    """
    return os.environ.get("BPM_PERF_PROBE", "").strip().lower() in ("1", "on", "true", "yes")


def _is_ws_only_bytes(b: bytes) -> bool:
    """True iff b is a non-empty run of only ASCII space (0x20) / tab (0x09)."""
    return len(b) > 0 and all(c in (0x20, 0x09) for c in b)


def _has_broken_code_block(text: str) -> bool:
    """True iff the response has a python-looking fenced block that fails ast.parse.

    Only plausible programs are judged, so prose fences never trigger. Unknown parse
    failures conservatively do not flag.
    """
    for lang, body in _BPM_FENCE_RE.findall(text):
        if lang.lower() not in ("python", "py", "python3", ""):
            continue
        if body.count("\n") < 3 or not _BPM_CODE_START_RE.search(body):
            continue
        if not _BPM_BLOCK_COLON_RE.search(body):
            continue
        try:
            ast.parse(body)
        except (SyntaxError, ValueError):
            return True
        except Exception:
            continue
    return False


def _build_byte_map(tokenizer, ids: list[int], control: set[int]) -> dict[int, bytes]:
    """{token_id: raw bytes} for non-control, byte-encodable tokens (others skipped)."""
    byte_texts = _bpm_ids_to_byte_texts(tokenizer, ids)
    out: dict[int, bytes] = {}
    for tid, bt in zip(ids, byte_texts):
        if int(tid) in control or not bt:
            continue
        try:
            out[int(tid)] = bt.encode("latin-1")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return out


def _detect_stop_ids(
    args: Namespace,
    tokenizer,
    *,
    cache_name: str,
    model_path: str | None,
    real_vocab_size: int | None = None,
) -> set[int]:
    from ..backend.special_tokens import detect_stop_token_ids

    cached = getattr(args, cache_name, None)
    if cached is not None:
        return set(int(x) for x in cached)
    stop = detect_stop_token_ids(tokenizer, model_path=model_path)
    if real_vocab_size is not None:
        stop = {int(t) for t in stop if 0 <= int(t) < int(real_vocab_size)}
    else:
        stop = {int(t) for t in stop if int(t) >= 0}
    setattr(args, cache_name, tuple(sorted(stop)))
    return stop


def _ensure_global_maps(
    args: Namespace,
    student_tokenizer,
    teacher_tokenizer,
    *,
    teacher_stop_ids: set[int],
    student_stop_ids: set[int] | None = None,
    real_vocab_size: int,
) -> tuple:
    """Build once: full-vocab byte maps, control sets, student trie, and phi."""
    cached = getattr(args, "_bpm_global_maps", None)
    if cached is not None:
        return cached
    # byte layer needs ByteLevel BPE on both sides
    from ..backend.alignment import _bpm_use_byte_alignment

    if not _bpm_use_byte_alignment(teacher_tokenizer, student_tokenizer):
        raise RuntimeError(
            "BPM requires ByteLevel-BPE tokenizers on BOTH sides (teacher and student); "
            "got a non-ByteLevel tokenizer. The byte-prefix phi map would silently drop "
            "tokens and zero the loss. Refusing to train."
        )
    stu_control = _control_ids(student_tokenizer)
    tea_control = _control_ids(teacher_tokenizer)
    tea_control = set(tea_control) | {int(t) for t in teacher_stop_ids}
    # stop tokens are control signals, not byte-walk content
    stu_control = set(stu_control) | {int(t) for t in (student_stop_ids or set())}
    stu_ids = [int(v) for v in student_tokenizer.get_vocab().values()]
    tea_ids = [int(v) for v in teacher_tokenizer.get_vocab().values() if int(v) < int(real_vocab_size)]
    stu_byte_map = _build_byte_map(student_tokenizer, stu_ids, stu_control)
    tea_byte_map = _build_byte_map(teacher_tokenizer, tea_ids, tea_control)
    student_trie = build_student_byte_trie(stu_byte_map, exclude_ids=stu_control)
    phi = build_phi(stu_byte_map, tea_byte_map)
    # stop pump: teacher stop mass -> the realized student stop id
    _pump_on = os.environ.get("BPM_STOP_PUMP", "on").strip().lower() not in ("off", "0", "false", "skip")
    student_eos_id = getattr(student_tokenizer, "eos_token_id", None)
    if student_eos_id is not None and _pump_on:
        for tid in teacher_stop_ids:
            if 0 <= int(tid) < int(real_vocab_size):
                phi[int(tid)] = int(student_eos_id)
    elif not _pump_on:
        _logger.info("BPM_STOP_PUMP=off: per-row stop pump disabled (bridge unaffected)")
    args._bpm_stop_pump_on = _pump_on
    maps = (stu_byte_map, tea_byte_map, stu_control, tea_control, student_trie, phi)
    args._bpm_global_maps = maps
    return maps


def _ensure_phi_tensors(args, phi: dict, real_vocab_size: int, vstu: int, device) -> tuple:
    """Cache teacher ids, mapped student ids, and compact student-image ids."""
    cached = getattr(args, "_bpm_phi_tensors", None)
    cached_meta = getattr(args, "_bpm_phi_tensors_meta", None)
    meta = (int(real_vocab_size), int(vstu), torch.device(device))
    if (
        cached is not None
        and cached_meta == meta
        and cached[0].device == torch.device(device)
    ):
        return cached
    phi_list = [-1] * int(real_vocab_size)
    for tid, sid in phi.items():
        if 0 <= int(tid) < real_vocab_size and 0 <= int(sid) < vstu:
            phi_list[int(tid)] = int(sid)
    phi_t = torch.tensor(phi_list, dtype=torch.long, device=device)
    teacher_ids = torch.nonzero(phi_t >= 0, as_tuple=False).flatten()
    if int(teacher_ids.numel()) == 0:
        phi_student_ids = torch.empty((0,), dtype=torch.long, device=device)
        image_ids_t = torch.empty((0,), dtype=torch.long, device=device)
    else:
        phi_student_ids = phi_t.index_select(0, teacher_ids)
        image_ids_t = torch.unique(phi_student_ids, sorted=True)
    tensors = (teacher_ids, phi_student_ids, image_ids_t)
    args._bpm_phi_tensors = tensors
    args._bpm_phi_tensors_meta = meta
    args._bpm_phi_pump_variants = {}
    return tensors


def _phi_student_ids_for_pump(
    args, phi_student_ids, student_eos_id: int, pump_stop_id: int
):
    """phi_student_ids with the eos placeholder column redirected to the sample's
    realized stop token (the stop pump; see _ensure_global_maps). Identity when the
    realized stop is tokenizer.eos. Variants are cached per pump id (a chat student has
    at most a handful of stop ids)."""
    if pump_stop_id == student_eos_id or student_eos_id < 0:
        return phi_student_ids
    cache = getattr(args, "_bpm_phi_pump_variants", None)
    if cache is None:
        cache = {}
        args._bpm_phi_pump_variants = cache
    v = cache.get(int(pump_stop_id))
    if v is None or v.device != phi_student_ids.device:
        v = phi_student_ids.clone()
        v[v == int(student_eos_id)] = int(pump_stop_id)
        cache[int(pump_stop_id)] = v
    return v


def _stop_bridge_target_mass(
    mode: str, stop_mass: float, floor: float, truncated: bool
) -> float | None:
    """Per-row bridge mass on the realized stop token, under --bpm-stop-bridge-mode.

    'bridge' uses the teacher's raw endorsement; a weak one pushes the realized stop down.
    'floor' raises it, but only on genuinely stopped samples -- flooring a truncated row
    would teach stopping mid-thought. 'skip' returns None and the caller drops the row.
    """
    if mode == "skip":
        return None
    if mode == "floor" and not truncated:
        return max(float(stop_mass), float(floor))
    return float(stop_mass)


def _ensure_tail_phi_tensors(
    args,
    stu_byte_map: dict[int, bytes],
    tea_byte_map: dict[int, bytes],
    realized_prefix: bytes,
    remaining_len: int,
    real_vocab_size: int,
    vstu: int,
    device,
    prefix_index: _TeacherPrefixIndex,
    student_trie,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], bool]:
    """Cached selected teacher ids + phi tensors for exact conditional tail rows.

    Returns (teacher_ids, phi_selected, image_ids_t, image_ids_tuple, all_valid) where
    all_valid is True iff q_other == 0 (every support id maps to a valid student token)."""
    cache = getattr(args, "_bpm_tail_phi_tensor_cache", None)
    if cache is None:
        from collections import OrderedDict
        cache = OrderedDict()
        args._bpm_tail_phi_tensor_cache = cache
    key = (bytes(realized_prefix), int(remaining_len), int(real_vocab_size), int(vstu), torch.device(device))
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)   # LRU touch
        return cached

    del stu_byte_map
    prefix_ids = prefix_index.by_prefix.get(bytes(realized_prefix))
    ids = [] if prefix_ids is None else [int(t) for t in prefix_ids.tolist()]
    ids = sorted(t for t in ids if 0 <= t < int(real_vocab_size))
    teacher_ids = torch.tensor(ids, dtype=torch.long, device=device)
    phi_vals: list[int] = []
    for tid in ids:
        suffix = tea_byte_map[int(tid)][len(realized_prefix):]
        u = student_first_token_capped(student_trie, suffix, int(remaining_len))
        # ids >= vstu would scatter out of bounds -> complement bin
        phi_vals.append(int(u) if u is not None and 0 <= int(u) < int(vstu) else -1)
    phi_selected = torch.tensor(phi_vals, dtype=torch.long, device=device)
    # phi_vals is already on CPU; phi_selected.cpu() would sync
    image_ids = sorted({int(s) for s in phi_vals if 0 <= int(s) < int(vstu)})
    image_ids_t = torch.tensor(image_ids, dtype=torch.long, device=device)
    # forced-delta is exact only when q_other == 0
    all_valid = all(0 <= int(s) < int(vstu) for s in phi_vals)
    # image_ids (CPU) lets the degenerate-tail gate compare
    tensors = (teacher_ids, phi_selected, image_ids_t, tuple(image_ids), all_valid)
    cache[key] = tensors
    # LRU cap: entries hold GPU tensors, keys keep arriving
    if len(cache) > 65536:
        cache.popitem(last=False)
    return tensors


class _TeacherPrefixIndex:
    """Full teacher vocab byte-prefix index shared by chain rows.

    Chain queries run on CPU by design: returning Python floats per trie query would
    synchronize CUDA once per prefix node.
    """

    def __init__(self, tea_byte_map: dict[int, bytes], real_vocab_size: int, device) -> None:
        self.device = torch.device(device)
        by_prefix: dict[bytes, list[int]] = {}
        by_exact: dict[bytes, list[int]] = {}
        for tid, b in tea_byte_map.items():
            tid_i = int(tid)
            if tid_i < 0 or tid_i >= int(real_vocab_size) or not b:
                continue
            by_exact.setdefault(b, []).append(tid_i)
            for n in range(len(b) + 1):
                by_prefix.setdefault(b[:n], []).append(tid_i)
        self.by_prefix = {p: torch.tensor(ids, dtype=torch.long) for p, ids in by_prefix.items()}
        self.by_exact = {p: torch.tensor(ids, dtype=torch.long) for p, ids in by_exact.items()}


def _chain_query_parts(
    *,
    start_pos: int,
    c: bytes,
    chunk: Chunk,
    teacher_rows: list[int],
) -> list[tuple[int, str, bytes]] | None:
    if not c or start_pos == chunk.chunk_len:
        return []
    pos = int(start_pos)
    ci = 0
    n = len(c)
    parts: list[tuple[int, str, bytes]] = []
    while ci < n and pos < chunk.chunk_len:
        k = chunk._token_at(pos)
        within = pos - chunk.off[k]
        rem = c[ci:]
        lk = len(chunk.tea_bytes[k])
        if within == 0:
            if len(rem) <= lk:
                parts.append((int(teacher_rows[k]), "prefix", rem))
                return parts
            if c[ci:ci + lk] != chunk.tea_bytes[k]:
                return None
            parts.append((int(teacher_rows[k]), "exact", chunk.tea_bytes[k]))
            ci += lk
            pos += lk
        else:
            forced = chunk.tea_bytes[k][within:]
            take = min(len(forced), n - ci)
            if c[ci:ci + take] != forced[:take]:
                return None
            ci += take
            pos += take
    return parts if ci == n else None


def _parts_have_ids(parts: list[tuple[int, str, bytes]] | None, prefix_index: _TeacherPrefixIndex) -> bool:
    if parts is None:
        return False
    for _row, kind, key in parts:
        ids = prefix_index.by_prefix.get(key) if kind == "prefix" else prefix_index.by_exact.get(key)
        if ids is None or int(ids.numel()) == 0:
            return False
    return True


def _add_query_parts(needed: dict[int, dict[str, set[bytes]]], parts: list[tuple[int, str, bytes]]) -> None:
    for row, kind, key in parts:
        needed.setdefault(int(row), {"prefix": set(), "exact": set()})[kind].add(key)


def _seg_query_parts(
    *,
    chunk: Chunk,
    offset: int,
    c: bytes,
    teacher_rows: list[int],
) -> tuple[list[tuple[int, str, bytes]], list[tuple[int, str, bytes]]] | None:
    if not c:
        return ([], [])
    if int(offset) + len(c) > chunk.chunk_len:
        return None
    k = chunk._token_at(int(offset))
    if int(offset) == chunk.off[k]:
        parts = _chain_query_parts(start_pos=int(offset), c=c, chunk=chunk, teacher_rows=teacher_rows)
        return (parts or [], parts or []) if parts is not None else None
    boundary = chunk.off[k]
    realized_prefix = chunk.tea_bytes[k][: int(offset) - boundary]
    denom = _chain_query_parts(start_pos=boundary, c=realized_prefix, chunk=chunk, teacher_rows=teacher_rows)
    num = _chain_query_parts(start_pos=boundary, c=realized_prefix + c, chunk=chunk, teacher_rows=teacher_rows)
    if denom is None or num is None:
        return None
    return (denom + num, num)


def _collect_position_query_needs(
    needed: dict[int, dict[str, set[bytes]]],
    *,
    chunk: Chunk,
    offset: int,
    trie,
    teacher_rows: list[int],
    teacher_prefix_index: _TeacherPrefixIndex,
    teacher_stop_ids: set[int],
) -> None:
    if not (0 <= int(offset) < chunk.chunk_len):
        return
    k = chunk._token_at(int(offset))
    if int(offset) == chunk.off[k] and teacher_stop_ids:
        needed.setdefault(int(teacher_rows[k]), {"prefix": set(), "exact": set()}).setdefault("stop", set())
    max_len = chunk.chunk_len - int(offset)
    def visit(node, p: bytes) -> None:
        if len(p) >= max_len:
            return
        for byte, child in node.children.items():
            cp = p + bytes([byte])
            res = _seg_query_parts(chunk=chunk, offset=int(offset), c=cp, teacher_rows=teacher_rows)
            if res is None:
                continue
            parts, support_parts = res
            if not _parts_have_ids(support_parts, teacher_prefix_index):
                continue
            _add_query_parts(needed, parts)
            visit(child, cp)

    visit(trie, b"")


def _build_chain_needed_id_sets(
    *,
    chain_items: list[tuple[int, int, int, int]],
    chunk_plan: dict[tuple[int, int], list[int]],
    tea_content_rows: list[int],
    tea_content_ids: list[int],
    tea_off: list[int],
    stu_off: list[int],
    tea_byte_map: dict[int, bytes],
    student_trie,
    teacher_prefix_index: _TeacherPrefixIndex,
    teacher_stop_ids: set[int],
) -> dict[int, dict[str, dict[bytes, torch.Tensor | None] | torch.Tensor | None]]:
    out: dict[int, dict[str, set[bytes]]] = {}
    chunk_cache: dict[tuple[int, int], Chunk] = {}
    cend_by_cstart = {int(cstart): int(cend) for cstart, cend in chunk_plan}
    for _lrow, ck, n, _gidx in chain_items:
        cstart = tea_off[ck]
        cend = cend_by_cstart.get(int(cstart))
        if cend is None:
            continue
        tea_js = chunk_plan.get((int(cstart), int(cend)), [])
        if not tea_js:
            continue
        cache_key = (int(cstart), int(cend))
        chunk = chunk_cache.get(cache_key)
        if chunk is None:
            chunk = Chunk([tea_byte_map[tea_content_ids[j]] for j in tea_js], [{} for _ in tea_js], tea_byte_map)
            chunk_cache[cache_key] = chunk
        offset = stu_off[n] - cstart
        _collect_position_query_needs(
            out,
            chunk=chunk,
            offset=offset,
            trie=student_trie,
            teacher_rows=[tea_content_rows[j] for j in tea_js],
            teacher_prefix_index=teacher_prefix_index,
            teacher_stop_ids=teacher_stop_ids,
        )
    result: dict[int, dict[str, dict[bytes, torch.Tensor | None] | torch.Tensor | None]] = {}
    for row, parts in out.items():
        prefix = {
            p: teacher_prefix_index.by_prefix.get(p)
            for p in parts.get("prefix", set())
        }
        exact = {
            p: teacher_prefix_index.by_exact.get(p)
            for p in parts.get("exact", set())
        }
        stop_ids = torch.tensor(sorted(int(t) for t in teacher_stop_ids), dtype=torch.long) if "stop" in parts else None
        result[int(row)] = {"prefix": prefix, "exact": exact, "stop": stop_ids}
    return result


class _TeacherProbQuery:
    """Row-specific teacher probability mass query used by bpm_seg.Chunk."""

    def __init__(self, mass_cache: dict[str, dict[bytes, float] | float], prefix_index: _TeacherPrefixIndex) -> None:
        self._mass_cache = mass_cache
        self.prefix_index = prefix_index

    def prefix_mass(self, prefix: bytes) -> float:
        cached = self._mass_cache["prefix"].get(prefix)
        if cached is not None:
            return float(cached)
        ids = self.prefix_index.by_prefix.get(prefix)
        if ids is None or int(ids.numel()) == 0:
            return 0.0
        raise RuntimeError(f"BPM chain query missing prefetched prefix mass for {prefix!r}")

    def exact_mass(self, target: bytes) -> float:
        cached = self._mass_cache["exact"].get(target)
        if cached is not None:
            return float(cached)
        ids = self.prefix_index.by_exact.get(target)
        if ids is None or int(ids.numel()) == 0:
            return 0.0
        raise RuntimeError(f"BPM chain query missing prefetched exact mass for {target!r}")

    def stop_mass(self, _stop_ids: set[int] | None = None) -> float:
        return float(self._mass_cache.get("stop", 0.0))


def _ensure_teacher_prefix_index(
    args: Namespace,
    tea_byte_map: dict[int, bytes],
    real_vocab_size: int,
    device,
) -> _TeacherPrefixIndex:
    cached = getattr(args, "_bpm_teacher_prefix_index", None)
    if cached is not None and getattr(cached, "device", None) == torch.device(device):
        return cached
    index = _TeacherPrefixIndex(tea_byte_map, real_vocab_size, device)
    args._bpm_teacher_prefix_index = index
    return index


def _entropy_from_logits(z: torch.Tensor, log_z: torch.Tensor | None = None) -> torch.Tensor:
    zf = z.float()
    if log_z is None:
        log_z = torch.logsumexp(zf, dim=-1)
    return log_z - (torch.exp(zf - log_z.unsqueeze(-1)) * zf).sum(-1)


def _entropy_sum_from_logits_chunked(z: torch.Tensor, log_z: torch.Tensor | None = None, row_chunk: int = 256) -> torch.Tensor:
    """Sum of per-row entropy, computed in row sub-blocks so the [rows, V] fp32 temporary
    (exp(zf - log_z) * zf inside _entropy_from_logits) stays bounded to ~row_chunk * V instead
    of materializing the whole (up to ~8192-row) block at once. Diagnostics-only (caller is
    under no_grad); == _entropy_from_logits(z, log_z).sum() modulo fp32 summation order."""
    n = int(z.shape[0])
    if n == 0:
        return z.sum() * 0.0
    row_chunk = max(1, min(int(row_chunk), n))
    total = z.sum() * 0.0
    for r in range(0, n, row_chunk):
        lz = None if log_z is None else log_z[r:r + row_chunk]
        total = total + _entropy_from_logits(z[r:r + row_chunk], log_z=lz).sum()
    return total


def _auto_bpm_loss_row_chunk(vstu: int, vtea: int, *, device=None) -> int:
    """Bound the scatter-target row block. Each row holds up to ~4 dense fp32 [c, Vstu] residents
    (target_stu + stu_f + saved softmax p + the student-logit grad), so the per-row budget is
    ~16*Vstu bytes. The teacher [c, Vtea] softmax is tiled inside the no-grad builder."""
    max_vocab = max(int(vstu), int(vtea), 1)
    budget_bytes = 512_000_000
    if device is not None and torch.cuda.is_available():
        try:
            dev = torch.device(device)
            free_bytes, _total_bytes = torch.cuda.mem_get_info(dev)
            budget_bytes = max(budget_bytes, int(free_bytes * 0.12))
        except (RuntimeError, TypeError, ValueError):
            pass
    return max(1, min(2048, int(budget_bytes) // (16 * max_vocab)))


def _auto_bpm_entropy_row_chunk(vstu: int, vtea: int, *, device=None) -> int:
    """Row chunk for the entropy diagnostics: budget ~ free*0.06 // (8*V), capped [256, 2048].
    Only iteration granularity is affected, not the summed value.
    """
    max_vocab = max(int(vstu), int(vtea), 1)
    budget_bytes = 1_000_000_000
    if device is not None and torch.cuda.is_available():
        try:
            dev = torch.device(device)
            free_bytes, _total_bytes = torch.cuda.mem_get_info(dev)
            budget_bytes = max(budget_bytes, int(free_bytes * 0.06))
        except (RuntimeError, TypeError, ValueError):
            pass
    return max(256, min(2048, int(budget_bytes) // (8 * max_vocab)))


def _guard_teacher_logits(z: torch.Tensor, real_vocab_size: int, temperature: float) -> torch.Tensor:
    """Project raw teacher lm_head logits onto the BPM target space.

    fp8 teachers can emit +inf, which makes x-max = NaN in the backward. Cap to a finite
    peak, drop padded-vocab columns, then apply the softmax temperature.
    """
    z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
    if real_vocab_size and z.shape[-1] > real_vocab_size:
        z = z[:, :real_vocab_size]
    return z / temperature


def _mass_from_logits(z: torch.Tensor, ids: list[int], log_z: torch.Tensor | None = None) -> torch.Tensor:
    if not ids:
        return torch.zeros((int(z.shape[0]),), dtype=torch.float32, device=z.device)
    ids_t = torch.tensor(ids, dtype=torch.long, device=z.device)
    if log_z is None:
        log_z = torch.logsumexp(z.float(), dim=-1)
    selected = z.float().index_select(1, ids_t)
    return torch.exp(torch.logsumexp(selected, dim=-1) - log_z)


def _cumulative_offsets(ids: list[int], byte_map: dict[int, bytes]) -> list[int]:
    off = [0]
    for t in ids:
        off.append(off[-1] + len(byte_map[t]))
    return off


def _teacher_queries_from_hidden(
    *,
    lm_head,
    tea_hidden,
    rows: list[int],
    real_vocab_size: int,
    temperature: float,
    proj_chunk: int,
    lm_device,
    lm_dtype,
    prefix_index: _TeacherPrefixIndex,
    needed_id_sets: dict[int, dict[str, dict[bytes, torch.Tensor | None] | torch.Tensor | None]],
) -> list[_TeacherProbQuery]:
    """Compute row teacher probs for selected hidden rows without Python vocab dicts."""
    if not rows:
        return []
    t_hidden = _select_hidden_rows_to_device(tea_hidden, rows, device=lm_device, dtype=lm_dtype)
    out: list[_TeacherProbQuery] = []
    with torch.no_grad():
        for s in range(0, int(t_hidden.shape[0]), proj_chunk):
            z = _guard_teacher_logits(lm_head(t_hidden[s:s + proj_chunk]).float(), real_vocab_size, temperature)
            log_z = torch.logsumexp(z.float(), dim=-1)
            rows_chunk = rows[s:s + proj_chunk]
            for off, row in enumerate(rows_chunk):
                row_z = z[off:off + 1].float()
                row_log_z = log_z[off:off + 1]
                needed = needed_id_sets.get(int(row), {})
                mass_cache: dict[str, dict[bytes, float] | float] = {"prefix": {}, "exact": {}, "stop": 0.0}
                pending: list[tuple[str, bytes | None, torch.Tensor]] = []
                for kind in ("prefix", "exact"):
                    entries = needed.get(kind, {})
                    if not isinstance(entries, dict):
                        continue
                    for key, ids_cpu in entries.items():
                        if ids_cpu is None or int(ids_cpu.numel()) == 0:
                            mass_cache[kind][key] = 0.0
                        else:
                            ids = ids_cpu.to(device=row_z.device, non_blocking=True)
                            selected = row_z.index_select(1, ids)
                            val_t = torch.exp(torch.logsumexp(selected, dim=-1) - row_log_z)
                            pending.append((kind, key, val_t.squeeze(0)))
                stop_ids_cpu = needed.get("stop")
                if isinstance(stop_ids_cpu, torch.Tensor) and int(stop_ids_cpu.numel()) > 0:
                    ids = stop_ids_cpu.to(device=row_z.device, non_blocking=True)
                    selected = row_z.index_select(1, ids)
                    val_t = torch.exp(torch.logsumexp(selected, dim=-1) - row_log_z)
                    pending.append(("stop", None, val_t.squeeze(0)))
                if pending:
                    vals = torch.stack([x[2] for x in pending]).detach().cpu().tolist()
                    for (kind, key, _val_t), val in zip(pending, vals):
                        if kind == "stop":
                            mass_cache["stop"] = float(val)
                        else:
                            mass_cache[kind][key] = float(val)
                out.append(_TeacherProbQuery(mass_cache, prefix_index))
    return out


def _pack_sparse_targets(position_targets: list[dict[int, float]], *, device) -> tuple:
    k = max((len(t) for t in position_targets if t), default=1)
    k = max(int(k), 1)
    r = len(position_targets)
    # one H2D per tensor instead of O(r*k) scalar writes
    ids_rows = [[0] * k for _ in range(r)]
    probs_rows = [[0.0] * k for _ in range(r)]
    mask_rows = [[False] * k for _ in range(r)]
    row_mask_list = [False] * r
    other_list = [0.0] * r   # masked rows (row_mask=False) keep 0.0, matching the prior tensor default
    q_other_vals: list[float] = []
    for i, target in enumerate(position_targets):
        if not target:
            q_other_vals.append(1.0)
            continue
        row_mask_list[i] = True
        s = 0.0
        for j, (tid, prob) in enumerate(sorted(target.items(), key=lambda kv: -kv[1])):
            p = float(prob)
            ids_rows[i][j] = int(tid)
            probs_rows[i][j] = p
            mask_rows[i][j] = True
            s += p
        q_other = max(0.0, 1.0 - s)
        other_list[i] = q_other
        q_other_vals.append(q_other)
    target_ids = torch.tensor(ids_rows, dtype=torch.long, device=device)
    target_probs = torch.tensor(probs_rows, dtype=torch.float32, device=device)
    target_mask = torch.tensor(mask_rows, dtype=torch.bool, device=device)
    row_mask = torch.tensor(row_mask_list, dtype=torch.bool, device=device)
    other = torch.tensor(other_list, dtype=torch.float32, device=device)
    return target_ids, target_probs, target_mask, row_mask, other, q_other_vals


def _teacher_stop_mass_from_hidden(
    *,
    lm_head,
    tea_hidden,
    row: int,
    teacher_stop_ids: set[int],
    real_vocab_size: int,
    temperature: float,
    lm_device,
    lm_dtype,
) -> float:
    if not teacher_stop_ids:
        return 0.0
    t_hidden = _select_hidden_rows_to_device(tea_hidden, [row], device=lm_device, dtype=lm_dtype)
    with torch.no_grad():
        z = _guard_teacher_logits(lm_head(t_hidden).float(), real_vocab_size, temperature)
        q = torch.softmax(z, dim=-1)[0]
        ids = [int(t) for t in teacher_stop_ids if 0 <= int(t) < int(q.shape[0])]
        if not ids:
            return 0.0
        return float(q[torch.tensor(ids, dtype=torch.long, device=q.device)].sum().detach().cpu().item())


def _teacher_stop_row(tea_label_ids: list[int], teacher_stop_ids: set[int]) -> int | None:
    """Pick the teacher hidden row whose next-token distribution is the real turn-end.

    tea_label_ids contains the teacher eos exactly once, at the last index, and that row's
    hidden is the state after the last content token. So take the last stop label.
    """
    rows = [j for j, lab in enumerate(tea_label_ids) if int(lab) in teacher_stop_ids]
    return rows[-1] if rows else None


def _decode_token(tokenizer, tid: int) -> str:
    try:
        return tokenizer.decode([int(tid)], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([int(tid)])
    except Exception:
        return str(tid)


def _dump_bpm_diag(
    *,
    args: Namespace,
    sample_index: int,
    call_index: int,
    rows: list[dict],
) -> None:
    record = {
        "call_index": int(call_index),
        "rollout_id": getattr(args, "_opd_diag_rollout_id", None),
        "step_id": getattr(args, "_opd_diag_step_id", None),
        "time": round(time.time(), 3),
        "sample_index": int(sample_index),
        "rows": rows[:40],  # 32 fast-row slots + up to 8 reserved student_stop slots
    }
    out_dir = None
    if not out_dir:
        dd = getattr(args, "dump_details", None)
        out_dir = os.path.join(dd, "bpm_teacher_diag") if dd else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "bpm_teacher_diag.jsonl"), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    else:
        _logger.info(f"[OPD][bpm][teacher-diag] {json.dumps(record, ensure_ascii=False, default=str)}")


def _batched_candidate_cylinders(
    all_cands: list[dict],
    fz: torch.Tensor,          # [n_factor_rows, Vtea] fp32 tempered logits (on device)
    flse: torch.Tensor,        # [n_factor_rows] logsumexp(fz)
    row_pos: dict[int, int],   # teacher hidden row -> index into fz
    pref_dev: dict[bytes, torch.Tensor],   # unique last_prefix -> by_prefix ids (on device)
    exact_dev: dict[bytes, torch.Tensor],  # unique exact-factor bytes -> by_exact ids (on device)
    tea_byte_map: dict[int, bytes],
) -> torch.Tensor:
    """Vectorized realized-path cylinder masses for a flat candidate list.

    Each cand carries {'exact': [(row, tid)...], 'last_row', 'last_prefix'}; returns an [n]
    tensor of prod(exact) * prefix_mass(last), 0.0 where nothing extends the last prefix.
    No bare `i` in here: shadowing a caller loop index once corrupted per-sample state.
    """
    n_cands = len(all_cands)
    logp_acc = fz.new_zeros(n_cands)
    dead = torch.zeros(n_cands, dtype=torch.bool, device=fz.device)
    se_c: list[int] = []
    se_fi: list[int] = []
    se_tid: list[int] = []
    me: list[tuple[int, int, torch.Tensor]] = []
    for _ci, c in enumerate(all_cands):
        for (r, tid) in c["exact"]:
            _fi = row_pos[r]
            _ids_e = exact_dev.get(tea_byte_map[tid])
            if _ids_e is None or int(_ids_e.numel()) == 1:
                se_c.append(_ci); se_fi.append(_fi); se_tid.append(int(tid))
            else:
                me.append((_ci, _fi, _ids_e))
    if se_c:
        _fi_t = torch.tensor(se_fi, dtype=torch.long, device=fz.device)
        _tid_t = torch.tensor(se_tid, dtype=torch.long, device=fz.device)
        _ci_t = torch.tensor(se_c, dtype=torch.long, device=fz.device)
        logp_acc.index_add_(0, _ci_t, fz[_fi_t, _tid_t] - flse[_fi_t])
    for (_ci, _fi, _ids_e) in me:
        logp_acc[_ci] = logp_acc[_ci] + (
            torch.logsumexp(fz[_fi].index_select(0, _ids_e), dim=0) - flse[_fi]
        )
    pgroups: dict[bytes, list[tuple[int, int]]] = {}
    for _ci, c in enumerate(all_cands):
        ids = pref_dev.get(c["last_prefix"])
        if ids is None or int(ids.numel()) == 0:
            dead[_ci] = True
        else:
            pgroups.setdefault(c["last_prefix"], []).append((_ci, row_pos[c["last_row"]]))
    for _pfx, _lst in pgroups.items():
        ids = pref_dev[_pfx]
        _ci_t = torch.tensor([x[0] for x in _lst], dtype=torch.long, device=fz.device)
        _fi_t = torch.tensor([x[1] for x in _lst], dtype=torch.long, device=fz.device)
        logp_acc.index_add_(0, _ci_t, torch.logsumexp(fz[_fi_t][:, ids], dim=-1) - flse[_fi_t])
    return torch.where(dead, logp_acc.new_zeros(()), logp_acc.exp())


def _accumulate_bpm_train_metrics(
    args: Namespace,
    *,
    device,
    cp_size: int = 1,
    total_loss_sum,
    total_ce_sum,
    total_tea_entropy,
    total_stu_entropy,
    total_entropy_rows: float,
    total_tea_entropy_rows: float = 0.0,
    total_stop_teacher: float,
    total_stop_student: float,
    total_rep_frac: float,
    total_q_other: float,
    total_local_rows: float,
    total_fast_rows: float,
    total_tail_rows: float,
    total_chain_rows: float,
    total_stop_rows: float,
    total_label_rows: float,
    metric_sample_count: float,
    skipped_samples: float,
    total_chain_rerouted_rows: float = 0.0,
    total_midspan_rows: float = 0.0,
    total_merge_corrected_rows: float = 0.0,
    total_midspan_corrected_rows: float = 0.0,
    total_tail_degenerate_rows: float = 0.0,
    total_route_dropped_rows: float = 0.0,
    skipped_broken_code: float = 0.0,
    total_masked_ws_rows: float = 0.0,
    total_masked_random_rows: float = 0.0,
) -> None:
    from megatron.core import mpu

    if not mpu.is_pipeline_last_stage():
        args._bpm_opd_metrics = None
        return

    if not hasattr(args, "_bpm_opd_metrics") or args._bpm_opd_metrics is None:
        args._bpm_opd_metrics = {
            "bpm_loss_sum": 0.0,
            "bpm_ce_sum": 0.0,
            # per-row trained objective: CE at beta=0, JSD/skew-RKL above
            "bpm_div_sum": 0.0,
            "bpm_stop_mass_teacher_sum": 0.0,
            "bpm_stop_prob_student_sum": 0.0,
            "bpm_rep_frac_sum": 0.0,
            "bpm_q_other_sum": 0.0,
            "bpm_rows_sum": 0.0,
            "bpm_fast_rows_sum": 0.0,
            "bpm_tail_rows_sum": 0.0,
            "bpm_chain_rows_sum": 0.0,
            "bpm_stop_rows_sum": 0.0,
            "bpm_chain_rerouted_rows_sum": 0.0,
            "bpm_midspan_rows_sum": 0.0,
            "bpm_merge_corrected_rows_sum": 0.0,
            "bpm_midspan_corrected_rows_sum": 0.0,
            "bpm_tail_degenerate_rows_sum": 0.0,
            "bpm_route_dropped_rows_sum": 0.0,
            "bpm_masked_ws_rows_sum": 0.0,
            "bpm_masked_random_rows_sum": 0.0,
            "bpm_label_rows_sum": 0.0,
            "bpm_sample_count_sum": 0.0,
            "bpm_metric_weight_sum": 0.0,
            "bpm_skipped_byte_mismatch_sum": 0.0,
            "bpm_skipped_broken_code_sum": 0.0,
            "bpm_tea_entropy_sum": torch.tensor(0.0, dtype=torch.float32, device=device),
            "bpm_stu_entropy_sum": torch.tensor(0.0, dtype=torch.float32, device=device),
            "bpm_tea_entropy_token_sum": 0.0,  # fast-row count (teacher-entropy denom)
            "bpm_overlap_count_sum": 0.0,
            "bpm_entropy_token_sum": 0.0,
            "bpm_align_ratio_sum": 0.0,
            "bpm_aligned_tokens_sum": 0.0,
            "bpm_num_tokens_sum": 0.0,
            "bpm_num_samples_with_alignment_sum": 0.0,
            "bpm_loss_weight_sum": 0.0,
            "_bpm_opd_loss_type": "bpm",
        }

    acc = args._bpm_opd_metrics
    metric_weight = float(total_local_rows)
    row_denom = max(metric_weight, 1.0)
    entropy_weight = float(total_entropy_rows)
    entropy_denom = max(entropy_weight, 1.0)
    bpm_loss_mean = (total_loss_sum.detach() / row_denom)
    bpm_ce_mean = (total_ce_sum.detach() / row_denom)
    bpm_tea_entropy_mean = (total_tea_entropy.detach() / entropy_denom)
    bpm_stu_entropy_mean = (total_stu_entropy.detach() / entropy_denom)

    acc["bpm_loss_sum"] += float(bpm_loss_mean.item()) * metric_weight
    acc["bpm_ce_sum"] += float(bpm_ce_mean.item()) * metric_weight
    # bpm_div: what total_ce_sum holds; == CE only at beta=0
    acc["bpm_div_sum"] = acc.get("bpm_div_sum", 0.0) + float(bpm_ce_mean.item()) * metric_weight
    acc["bpm_stop_mass_teacher_sum"] += float(total_stop_teacher)
    acc["bpm_stop_prob_student_sum"] += float(total_stop_student)
    acc["bpm_rep_frac_sum"] += float(total_rep_frac)
    acc["bpm_q_other_sum"] += float(total_q_other)
    acc["bpm_rows_sum"] += metric_weight
    acc["bpm_fast_rows_sum"] += float(total_fast_rows)
    acc["bpm_tail_rows_sum"] += float(total_tail_rows)
    acc["bpm_chain_rows_sum"] += float(total_chain_rows)
    acc["bpm_stop_rows_sum"] += float(total_stop_rows)
    acc["bpm_chain_rerouted_rows_sum"] = acc.get("bpm_chain_rerouted_rows_sum", 0.0) + float(total_chain_rerouted_rows)
    acc["bpm_midspan_rows_sum"] = acc.get("bpm_midspan_rows_sum", 0.0) + float(total_midspan_rows)
    acc["bpm_merge_corrected_rows_sum"] = acc.get("bpm_merge_corrected_rows_sum", 0.0) + float(total_merge_corrected_rows)
    acc["bpm_midspan_corrected_rows_sum"] = acc.get("bpm_midspan_corrected_rows_sum", 0.0) + float(total_midspan_corrected_rows)
    acc["bpm_tail_degenerate_rows_sum"] = acc.get("bpm_tail_degenerate_rows_sum", 0.0) + float(total_tail_degenerate_rows)
    acc["bpm_route_dropped_rows_sum"] = acc.get("bpm_route_dropped_rows_sum", 0.0) + float(total_route_dropped_rows)
    acc["bpm_masked_ws_rows_sum"] = acc.get("bpm_masked_ws_rows_sum", 0.0) + float(total_masked_ws_rows)
    acc["bpm_masked_random_rows_sum"] = acc.get("bpm_masked_random_rows_sum", 0.0) + float(total_masked_random_rows)
    acc["bpm_label_rows_sum"] += float(total_label_rows)
    acc["bpm_sample_count_sum"] += float(metric_sample_count)  # rep_frac denom (cp-inflated, cancels)
    acc["bpm_metric_weight_sum"] += metric_weight
    acc["bpm_skipped_byte_mismatch_sum"] += float(skipped_samples)
    acc["bpm_skipped_broken_code_sum"] = acc.get("bpm_skipped_broken_code_sum", 0.0) + float(skipped_broken_code)
    if entropy_weight > 0.0:
        acc["bpm_tea_entropy_sum"] = acc["bpm_tea_entropy_sum"] + bpm_tea_entropy_mean * entropy_weight
        acc["bpm_stu_entropy_sum"] = acc["bpm_stu_entropy_sum"] + bpm_stu_entropy_mean * entropy_weight
        acc["bpm_entropy_token_sum"] += entropy_weight
        # teacher entropy denom = fast rows only
        acc["bpm_tea_entropy_token_sum"] = acc.get("bpm_tea_entropy_token_sum", 0.0) + float(total_tea_entropy_rows)
    acc["bpm_overlap_count_sum"] += metric_weight
    acc["bpm_align_ratio_sum"] += metric_weight
    acc["bpm_aligned_tokens_sum"] += metric_weight
    acc["bpm_num_tokens_sum"] += float(total_label_rows)
    # /cp_size: CP splits the sequence, not samples
    acc["bpm_num_samples_with_alignment_sum"] += float(metric_sample_count) / max(int(cp_size), 1)
    acc["bpm_loss_weight_sum"] += metric_weight


def bpm_core_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean,
    sum_of_sample,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """BPM forward-KL OPD loss (vectorized). Returns (scalar loss, log dict)."""
    from megatron.core import mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        raise ValueError(
            "[OPD][bpm] --bpm-alignment-mode bpm requires student tensor-parallel "
            f"size 1 (full-vocab student/teacher softmax); got TP={tp_size}."
        )

    teacher_hidden_states_list = batch.get("teacher_hidden_states")
    teacher_token_ids_list = batch.get("teacher_token_ids")
    if teacher_hidden_states_list is None or teacher_token_ids_list is None:
        raise ValueError("[OPD][bpm] requires teacher_hidden_states and teacher_token_ids in batch.")
    unconcat_tokens = batch.get("unconcat_tokens")
    if unconcat_tokens is None:
        raise ValueError("[OPD][bpm] requires unconcat_tokens for label alignment.")

    lm_head = _get_bpm_teacher_lm_head(args)
    teacher_tokenizer = _get_bpm_teacher_tokenizer(args)
    student_tokenizer = _get_bpm_student_tokenizer(args)

    device = logits.device
    cp_size = mpu.get_context_parallel_world_size()
    temperature = 1.0
    teacher_eos_id = teacher_tokenizer.eos_token_id
    real_vocab_size = int(getattr(lm_head, "_opd_vocab_size", 0) or len(teacher_tokenizer.get_vocab()))
    teacher_model_path = (
        getattr(args, "bpm_teacher_model_path", None)
        or getattr(args, "opd_teacher_model_path", None)
        or getattr(args, "bpm_teacher_tokenizer_path", None)
    )
    teacher_stop_ids = _detect_stop_ids(
        args,
        teacher_tokenizer,
        cache_name="_bpm_teacher_stop_ids",
        model_path=teacher_model_path,
        real_vocab_size=real_vocab_size,
    )
    # without model_path the stop set collapses to {tokenizer.eos}
    student_stop_ids = _detect_stop_ids(
        args,
        student_tokenizer,
        cache_name="_bpm_student_stop_ids",
        model_path=getattr(args, "hf_checkpoint", None),
    )
    student_eos_id = int(getattr(student_tokenizer, "eos_token_id", -1))
    stu_byte_map, tea_byte_map, stu_control, tea_control, student_trie, phi = _ensure_global_maps(
        args,
        student_tokenizer,
        teacher_tokenizer,
        teacher_stop_ids=teacher_stop_ids,
        student_stop_ids=student_stop_ids,
        real_vocab_size=real_vocab_size,
    )
    lm_dtype = lm_head.weight.dtype
    lm_device = lm_head.weight.device
    # bigger blocks batch the GEMM but hold [c, V] fp32 buffers
    proj_chunk = int(getattr(args, "bpm_proj_chunk", 0) or 0) or 1024
    vstu = int(logits.shape[-1])
    loss_row_chunk = min(proj_chunk, _auto_bpm_loss_row_chunk(vstu, real_vocab_size, device=device))
    phi_teacher_ids, phi_student_ids, phi_image_ids = _ensure_phi_tensors(
        args, phi, real_vocab_size, vstu, device
    )
    teacher_prefix_index = _ensure_teacher_prefix_index(args, tea_byte_map, real_vocab_size, lm_device)
    # ce_weight scales the identity-resolved CE; 1.0 default.
    ce_weight = 1.0
    # 0=forward-KL, 0.5=JSD, 1=reverse-KL (TRL GOLD convention)
    _cb = getattr(args, "bpm_beta", 0.0)
    bpm_beta = 0.0 if _cb is None else float(_cb)
    # fallback matches the argparse default 0.0
    _cl = getattr(args, "bpm_rkl_lambda", 0.0)
    bpm_rkl_lambda = 0.0 if _cl is None else float(_cl)
    # 'fast' = phi gather | 'scatter' += chain fix | 'bytewalk' = exact
    chain_mode = str(getattr(args, "bpm_chain_mode", "fast") or "fast")
    # '11' = 1:1, 'n1' = N:1, '1n' = spanning; stop rows always on
    _routes_raw = str(getattr(args, "bpm_routes", "") or "11,n1,1n")
    bpm_routes = {r.strip() for r in _routes_raw.split(",") if r.strip()}
    route_11 = "11" in bpm_routes
    route_n1 = "n1" in bpm_routes
    route_1n = "1n" in bpm_routes
    # 'fast' = phi gather | 'conditional' = exact, biggest cost
    tail_mode = "exact"
    # 'bridge' | 'floor' on stopped samples | 'skip' = no loss
    stop_bridge_mode = str(getattr(args, "bpm_stop_bridge_mode", "bridge") or "bridge")
    if stop_bridge_mode not in ("bridge", "floor", "skip"):
        raise ValueError(f"Unsupported --bpm-stop-bridge-mode: {stop_bridge_mode}")
    stop_bridge_floor = 0.8
    entropy_row_chunk = _auto_bpm_entropy_row_chunk(vstu, real_vocab_size, device=device)
    opd_diagnostics_mode = getattr(args, "opd_diagnostics_mode", "basic")
    if opd_diagnostics_mode not in ("off", "basic", "full"):
        raise ValueError(f"Unsupported --opd-diagnostics-mode: {opd_diagnostics_mode}")
    loss_only_diagnostics = opd_diagnostics_mode == "off"
    entropy_diagnostics_enabled = not loss_only_diagnostics

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    # mask=0 positions are not trained; None = all trainable
    loss_masks = batch.get("loss_masks")

    is_logging_rank = (
        mpu.is_pipeline_last_stage()
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )

    stu_logits_list: list[torch.Tensor] = []
    for logits_chunk, _tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    ):
        stu_logits_list.append(logits_chunk)

    # per-row objective; a true CE only at beta=0
    total_loss_sum = logits.sum() * 0.0   # gradient edge; redefined to ce_weight*div after the loop
    total_ce_sum = logits.sum() * 0.0     # distillation objective sum (CE@beta0, divergence@beta>0)
    total_local_rows = 0.0
    total_label_rows = 0.0
    skipped_samples = 0.0
    skipped_broken_code = 0.0
    filter_broken_code = _bpm_filter_broken_code_enabled()
    mask_ws_rows = _bpm_mask_ws_rows_enabled()
    total_masked_ws_rows = 0.0
    mask_random_rows = _bpm_mask_random_rows_enabled()
    total_masked_random_rows = 0.0
    alignable_samples = 0.0
    total_tea_entropy = logits.sum() * 0.0
    total_stu_entropy = logits.sum() * 0.0
    total_tea_entropy_rows = 0.0   # fast rows only (teacher-entropy denom; tea ent needs z_c)
    # on-device: the per-position loops must not .item()-sync
    total_stop_teacher = torch.zeros((), dtype=torch.float32, device=device)
    total_stop_student = torch.zeros((), dtype=torch.float32, device=device)
    total_q_other = 0.0
    total_q_other_tensor = torch.zeros((), dtype=torch.float32, device=device)
    # GPU-resident: a python float would need a per-group sync
    total_tail_degenerate_tensor = torch.zeros((), dtype=torch.float32, device=device)
    total_rep_frac = 0.0
    total_fast_rows = 0.0
    total_tail_rows = 0.0
    total_chain_rows = 0.0
    total_stop_rows = 0.0
    # alignment-signal observability (the previously-invisible classes):
    total_chain_rerouted_rows = 0.0   # boundary-start 1:N rows folded into fast (fast/scatter)
    total_midspan_rows = 0.0          # mid-start M:N spanning rows (fast: folded; scatter: dropped)
    total_route_dropped_rows = 0.0    # rows excluded by the --bpm-routes ablation mask
    total_merge_corrected_rows = 0.0  # non-spanning boundary rows given the v2 merge-candidate fix
    total_midspan_corrected_rows = 0.0  # v2.5: true mid-span rows trained via conditional chain
    # includes valid_t2==0 soft-dropped rows (exactly-0 CE)
    total_entropy_rows = 0.0
    metric_sample_count = 0.0

    # BPM_PERF_PROBE=1: sync at section boundaries; off by default
    bpm_perf_probe = _bpm_perf_probe_enabled()
    _perf = {"classify": 0.0, "fast": 0.0, "tail": 0.0, "chain": 0.0, "stop": 0.0}

    def _psync():
        if bpm_perf_probe and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        return time.time()

    # per_sample: this sample's CP-local distill sum and row count
    track_per_sample = getattr(args, "opd_loss_reduction", "per_token") == "per_sample"
    per_sample_distill: dict = {}   # CP-local grad-carrying loss sum per original sample i
    per_sample_aligned: dict = {}   # CP-local BPM-row count per original sample i
    if track_per_sample:
        # set up front so the early zero-loss return also has it
        args._opd_per_sample_normalizer = 0.0

    for i in range(len(response_lengths)):
        # snapshot the accumulators; the contribution is a delta
        if track_per_sample:
            ps_ce_start = total_ce_sum
            ps_rows_start = total_local_rows
        stu_logits_shard = stu_logits_list[i]
        response_len = int(response_lengths[i])
        total_len = int(total_lengths[i])

        local_response_indices = _get_bpm_local_response_indices(total_len, response_len)
        if len(local_response_indices) > int(stu_logits_shard.shape[0]):
            # a short shard would silently untrain this sample's tail
            raise RuntimeError(
                f"BPM CP window mismatch: {len(local_response_indices)} CP-owned response rows "
                f"but student logits shard has only {int(stu_logits_shard.shape[0])} rows "
                f"(total_len={total_len}, response_len={response_len})."
            )
        local_pos_by_global = {gidx: lidx for lidx, gidx in enumerate(local_response_indices)}

        stu_label_ids_full = _to_int_list(unconcat_tokens[i][-response_len:])
        # per-position trainability mask; mask=0 rows are excluded
        mask_i = _to_int_list(loss_masks[i]) if loss_masks is not None else None
        total_label_rows += float(
            sum(
                1
                for g in local_response_indices
                if mask_i is None or (g < len(mask_i) and mask_i[g] != 0)
            )
        )

        tea_hidden = teacher_hidden_states_list[i]
        tea_input_ids = teacher_token_ids_list[i]
        if tea_hidden is None or tea_input_ids is None:
            continue
        tea_input_ids = _to_int_list(tea_input_ids)
        if not tea_input_ids or int(tea_hidden.shape[0]) != len(tea_input_ids):
            continue
        tea_label_ids = tea_input_ids[1:] + [int(teacher_eos_id)]

        # content streams; both sides must render the same bytes
        stu_content_ids: list[int] = []
        stu_content_global: list[int] = []
        ok = True
        for gidx, tid in enumerate(stu_label_ids_full):
            # the stop row supervises these; training them here double-counts
            if int(tid) in stu_control or int(tid) in student_stop_ids:
                continue
            if int(tid) not in stu_byte_map:
                ok = False
                break
            stu_content_ids.append(int(tid))
            stu_content_global.append(gidx)
        if not ok or not stu_content_ids:
            continue

        tea_content_ids: list[int] = []
        tea_content_rows: list[int] = []
        for j, tid in enumerate(tea_label_ids):
            if int(tid) in tea_control:
                continue
            if int(tid) not in tea_byte_map:
                ok = False
                break
            tea_content_ids.append(int(tid))
            tea_content_rows.append(j)
        if not ok or not tea_content_ids:
            continue

        stu_off = _cumulative_offsets(stu_content_ids, stu_byte_map)
        tea_off = _cumulative_offsets(tea_content_ids, tea_byte_map)
        sync_points = sorted(set(stu_off) & set(tea_off))
        # byte content must match, not just length
        stu_bytes = b"".join(stu_byte_map[t] for t in stu_content_ids)
        tea_bytes = b"".join(tea_byte_map[t] for t in tea_content_ids)
        if stu_bytes != tea_bytes:
            skipped_samples += 1.0
            if is_logging_rank and skipped_samples <= 3:
                _logger.warning(
                    f"[OPD][bpm] sample {i}: student/teacher byte streams differ after control "
                    f"removal (len {len(stu_bytes)} vs {len(tea_bytes)}); skipping."
                )
            continue
        # broken code fogs every later row; whole-sample skip
        if filter_broken_code and _has_broken_code_block(stu_bytes.decode("utf-8", errors="replace")):
            skipped_broken_code += 1.0
            if is_logging_rank and skipped_broken_code <= 3:
                _logger.info(
                    f"[OPD][bpm] sample {i}: broken fenced code block; skipping "
                    f"(BPM_FILTER_BROKEN_CODE)."
                )
            continue
        tea_off_to_k = {tea_off[k]: k for k in range(len(tea_content_ids))}

        alignable_samples += 1.0
        total_rep_frac += bpm_repetition_fraction(stu_label_ids_full)
        metric_sample_count += 1.0
        diag_rows: list[dict] = []
        diag_iv = 0
        diag_enabled = bool(entropy_diagnostics_enabled and diag_iv > 0 and is_logging_rank and i == 0)

        # fast rows stay vectorized; tail and 1:N use exact paths
        fast_lrows: list[int] = []
        fast_thidden_rows: list[int] = []
        tail_groups: dict[tuple[int, bytes, int], list[tuple[int, int]]] = {}
        chain_items: list[tuple[int, int, int, int]] = []  # (lrow, chunk_start_k, n, gidx)
        chain_scatter_specs: list[dict] = []  # chain_mode=scatter realized-chain corrections
        midspan_specs: list[dict] = []        # v2.5: true mid-span rows (conditional chain)

        def _chain_rows_contiguous(parts) -> bool:
            """Control-token seam guard: the chain factors must live on consecutive teacher
            hidden rows. The content byte stream has control tokens removed, so a candidate
            walking across a removed mid-response control token would splice bytes across the
            seam and its chain factors would jump the control row's probability -- refuse."""
            j_first = parts[0][0]
            j_last = parts[-1][0]
            for _jj in range(j_first, j_last):
                if tea_content_rows[_jj + 1] != tea_content_rows[_jj] + 1:
                    return False
            return True

        def _cand_specs_for(cands: list, a_off: int, gov_k: int, extra_len: int) -> list:
            """Per-candidate chain factorizations. extra_len=0 for boundary rows; for mid-span
            rows extra_len=within so the parts factorize cyl(rho + candidate bytes) from the
            governing boundary (the conditional divides by prefix_mass(rho) later)."""
            out = []
            for (c_sid, c_len) in cands:
                if not (0 <= int(c_sid) < vstu):
                    # bounds guard: an out-of-range sid corrupts the next row
                    break
                parts = spanning_chain_parts(
                    extra_len + c_len, gov_k, tea_content_ids, lambda tid: len(tea_byte_map[tid])
                )
                if parts is None or not _chain_rows_contiguous(parts):
                    break
                c_bytes = stu_bytes[a_off:a_off + c_len]
                full_bytes = (tea_byte_map[tea_content_ids[gov_k]][:extra_len] + c_bytes) if extra_len else c_bytes
                last_take = parts[-1][2]
                out.append({
                    "u_sid": int(c_sid),
                    "exact": [
                        (int(tea_content_rows[j]), int(tea_content_ids[j]))
                        for (j, _kind, _t) in parts[:-1]
                    ],
                    "last_row": int(tea_content_rows[parts[-1][0]]),
                    "last_prefix": bytes(full_bytes[len(full_bytes) - last_take:]),
                })
            return out

        def _mk_scatter_spec(gov_k: int, a_off: int, cands: list) -> dict | None:
            """chain_mode=scatter v2: per-candidate realized-path chain factorizations for one
            boundary fast row (candidates nested along the realized bytes, increasing length).
            Candidates whose bytes run past the content stream / cross a control seam are
            dropped (with everything deeper -- nesting makes them share the hazard)."""
            cand_specs = _cand_specs_for(cands, a_off, gov_k, 0)
            if not cand_specs:
                return None
            _v_raw = int(phi.get(int(tea_content_ids[gov_k]), -1))
            return {
                "fast_idx": len(fast_lrows) - 1,
                "v_sid": _v_raw if 0 <= _v_raw < vstu else -1,
                "cands": cand_specs,
            }
        chunk_plan: dict[tuple[int, int], list[int]] = {}
        # mask as many random rows as whitespace rows, seeded
        random_mask_rows: set[int] = set()
        if mask_random_rows:
            _ws_n = 0
            _elig: list[int] = []
            for _n2, _g2 in enumerate(stu_content_global):
                if local_pos_by_global.get(_g2) is None:
                    continue
                if mask_i is not None and (_g2 >= len(mask_i) or mask_i[_g2] == 0):
                    continue
                if _is_ws_only_bytes(stu_byte_map[stu_content_ids[_n2]]):
                    _ws_n += 1
                else:
                    _elig.append(_n2)
            if _ws_n > 0 and _elig:
                _seed_base = os.environ.get("BPM_MASK_RANDOM_SEED", "1234")
                _sig = f"{_seed_base}-{len(stu_content_global)}-{int(stu_content_ids[0])}"
                _rng = random.Random(int(hashlib.md5(_sig.encode()).hexdigest()[:8], 16))
                random_mask_rows = set(_rng.sample(_elig, min(_ws_n, len(_elig))))
        _cls_t = _psync()
        for n, gidx in enumerate(stu_content_global):
            lrow = local_pos_by_global.get(gidx)
            if lrow is None:
                continue
            if mask_i is not None and (gidx >= len(mask_i) or mask_i[gidx] == 0):
                continue  # honor loss_masks: never train masked-out positions
            a = stu_off[n]
            k = tea_off_to_k.get(a)
            sid = stu_content_ids[n]
            _sid_bytes = stu_byte_map[sid]
            slen = len(_sid_bytes)
            # WS-row ablation: drop whitespace rows from every route
            if mask_ws_rows and _is_ws_only_bytes(_sid_bytes):
                total_masked_ws_rows += 1.0
                continue
            # matched-fraction control: drop the pre-selected non-WS rows
            if n in random_mask_rows:
                total_masked_random_rows += 1.0
                continue
            # exact while the student token ends inside this decision
            if k is not None and a + slen <= tea_off[k] + len(tea_byte_map[tea_content_ids[k]]):
                # route gating: 1:1 vs N:1 head (strict prefix of the token)
                _tend_k = tea_off[k] + len(tea_byte_map[tea_content_ids[k]])
                _is_11 = (a + slen) == _tend_k
                if (_is_11 and not route_11) or ((not _is_11) and not route_n1):
                    total_route_dropped_rows += 1.0
                    continue
                fast_lrows.append(lrow)
                fast_thidden_rows.append(tea_content_rows[k])
                if chain_mode == "scatter" and route_1n and ce_weight > 0.0:
                    # boundary-row merge candidates carry chain mass
                    _cands = realized_merge_candidates(student_trie, stu_bytes, a, _tend_k - a)
                    if _cands:
                        _sp = _mk_scatter_spec(k, a, _cands)
                        if _sp is not None:
                            total_merge_corrected_rows += 1.0
                            chain_scatter_specs.append(_sp)
            else:
                if k is None:
                    ck = bisect_right(tea_off, a) - 1
                    if ck < 0 or ck >= len(tea_content_ids) or not (tea_off[ck] < a < tea_off[ck + 1]):
                        continue
                else:
                    ck = k
                cstart = tea_off[ck]
                tend = tea_off[ck + 1]
                if a > cstart and a + slen <= tend and tend in sync_points:
                    if not route_n1:
                        total_route_dropped_rows += 1.0
                        continue
                    if tail_mode == "fast":
                        # mid-token row -> the cheap phi gather
                        fast_lrows.append(lrow)
                        fast_thidden_rows.append(tea_content_rows[ck])
                        continue
                    within = a - cstart
                    realized_prefix = tea_byte_map[tea_content_ids[ck]][:within]
                    remaining_len = tend - a
                    tail_groups.setdefault(
                        (int(tea_content_rows[ck]), bytes(realized_prefix), int(remaining_len)),
                        [],
                    ).append((lrow, int(gidx), int(sid)))
                    continue
                if not route_1n:
                    total_route_dropped_rows += 1.0
                    continue
                _is_midspan = a > cstart  # mid-start and spans past / unsynced end (M:N interior)
                if chain_mode == "fast":
                    # spanning row -> phi gather; scatter mode corrects the sign
                    if _is_midspan:
                        total_midspan_rows += 1.0
                    else:
                        total_chain_rerouted_rows += 1.0
                    fast_lrows.append(lrow)
                    fast_thidden_rows.append(tea_content_rows[ck])
                    continue
                if chain_mode == "scatter":
                    if _is_midspan:
                        total_midspan_rows += 1.0
                        # true mid-span rows get the exact conditional
                        if a + slen > tend and ce_weight > 0.0:
                            _within = a - cstart
                            _rho = bytes(tea_byte_map[tea_content_ids[ck]][:_within])
                            _rem_len = tend - a
                            _cands = realized_merge_candidates(student_trie, stu_bytes, a, _rem_len)
                            if _cands and not any(cl == slen for (_cs, cl) in _cands):
                                _cands = [(cs, cl) for (cs, cl) in _cands if cl < slen]
                                _cands.append((int(sid), slen))   # emitted spans by definition
                            _cs2 = _cand_specs_for(_cands, a, ck, _within) if _cands else []
                            if _cs2:
                                _rem_bytes = tea_byte_map[tea_content_ids[ck]][_within:]
                                _vr = student_first_token_capped(student_trie, _rem_bytes, _rem_len)
                                midspan_specs.append({
                                    "lrow": lrow,
                                    "gov_row": int(tea_content_rows[ck]),
                                    "rho": _rho,
                                    "rem_len": int(_rem_len),
                                    "v_sid": int(_vr) if _vr is not None and 0 <= int(_vr) < vstu else -1,
                                    "cands": _cs2,
                                })
                        continue
                    total_chain_rerouted_rows += 1.0
                    fast_lrows.append(lrow)
                    fast_thidden_rows.append(tea_content_rows[ck])
                    if ce_weight > 0.0:  # ce_weight==0 never applies deltas (mirror the precompute gate)
                        # max_cands must keep the emitted token
                        _tend_gov = tea_off[ck] + len(tea_byte_map[tea_content_ids[ck]])
                        _cands = realized_merge_candidates(student_trie, stu_bytes, a, _tend_gov - a)
                        if _cands and not any(cl == slen for (_cs, cl) in _cands):
                            _cands = [(cs, cl) for (cs, cl) in _cands if cl < slen]
                            _cands.append((int(sid), slen))       # keep nesting order
                        if _cands:
                            _sp = _mk_scatter_spec(ck, a, _cands)
                            if _sp is not None:
                                chain_scatter_specs.append(_sp)
                    continue
                sp_idx = bisect_right(sync_points, cstart)
                cend = sync_points[sp_idx] if sp_idx < len(sync_points) else tea_off[-1]
                left = bisect_left(tea_off, cstart)
                right = bisect_left(tea_off, cend)
                tea_js = [j for j in range(left, right) if j < len(tea_content_ids)]
                if not tea_js:
                    continue
                chunk_plan.setdefault((int(cstart), int(cend)), tea_js)
                chain_items.append((lrow, ck, n, gidx))

        # (lrow, teacher_hidden_row, stop_mass, stop_id, truncated)
        stop_items: list[tuple[int, int, float, int, bool]] = []
        for gidx, tid in enumerate(stu_label_ids_full):
            if int(tid) not in student_stop_ids:
                continue
            # first stop label only: later ones are the synthetic OPD EOS
            lrow = local_pos_by_global.get(gidx)
            row_trainable = lrow is not None and not (
                mask_i is not None and (gidx >= len(mask_i) or mask_i[gidx] == 0)
            )
            if row_trainable:
                # last stop label: the masked-ids contract puts it last
                stop_row = _teacher_stop_row(tea_label_ids, teacher_stop_ids)
                if stop_row is not None:
                    stop_mass = _teacher_stop_mass_from_hidden(
                        lm_head=lm_head,
                        tea_hidden=tea_hidden,
                        row=stop_row,
                        teacher_stop_ids=teacher_stop_ids,
                        real_vocab_size=real_vocab_size,
                        temperature=temperature,
                        lm_device=lm_device,
                        lm_dtype=lm_dtype,
                    )
                    if stop_mass > 0.0:
                        # truncated samples exceed the cap after the appended eos
                        _cap = int(getattr(args, "rollout_max_response_len", 0) or 0)
                        _truncated = bool(_cap > 0 and int(response_lengths[i]) > _cap)
                        stop_items.append((lrow, stop_row, stop_mass, int(tid), _truncated))
            break

        # redirect the eos placeholder to the realized stop token
        realized_stop_id = next(
            (int(t) for t in stu_label_ids_full if int(t) in student_stop_ids), None
        )
        if realized_stop_id is not None and realized_stop_id != student_eos_id:
            args._bpm_chat_stop_id = int(realized_stop_id)
        pump_stop_id = (
            int(realized_stop_id)
            if realized_stop_id is not None and realized_stop_id != student_eos_id
            else int(getattr(args, "_bpm_chat_stop_id", student_eos_id))
        )
        phi_student_ids_pump = _phi_student_ids_for_pump(
            args, phi_student_ids, int(student_eos_id), int(pump_stop_id)
        )

        if bpm_perf_probe:
            _perf["classify"] += _psync() - _cls_t
        # ---- fast boundary rows: vectorized token-marginal forward-KL ----
        _perf_t = _psync()
        if fast_lrows:
            # precompute each spanning row's chain mass once per sample
            _spec_pchain = None   # v2: [n_total_candidates] cylinder masses, spec-major order
            if chain_scatter_specs and ce_weight > 0.0:
                # reuse teacher_prefix_index: a new device key thrashes it
                _all_cands = [c for sp in chain_scatter_specs for c in sp["cands"]]
                _factor_rows = sorted({
                    r for c in _all_cands for (r, _t) in c["exact"]
                } | {c["last_row"] for c in _all_cands})
                _row_pos = {r: _p for _p, r in enumerate(_factor_rows)}
                with torch.no_grad():
                    _fh = _select_hidden_rows_to_device(
                        tea_hidden, _factor_rows, device=lm_device, dtype=lm_dtype
                    )
                    # spanning-heavy samples (~2 rows/spec)
                    _fz_parts = []
                    for _c0 in range(0, int(_fh.shape[0]), 1024):
                        _zc = _guard_teacher_logits(lm_head(_fh[_c0:_c0 + 1024]).float(), real_vocab_size, temperature)
                        _fz_parts.append(_zc.to(device))
                    _fz = torch.cat(_fz_parts, dim=0) if len(_fz_parts) > 1 else _fz_parts[0]
                    del _fz_parts
                    _flse = torch.logsumexp(_fz, dim=-1)             # [n_factor_rows]
                    # hoist the (Zipfian-repetitive) prefix/exact id tensors to device once
                    _pref_dev = {
                        p: teacher_prefix_index.by_prefix[p].to(_fz.device)
                        for p in {c["last_prefix"] for c in _all_cands}
                        if p in teacher_prefix_index.by_prefix
                    }
                    _exact_dev = {
                        b: teacher_prefix_index.by_exact[b].to(_fz.device)
                        for c in _all_cands
                        for b in (tea_byte_map[t] for (_r, t) in c["exact"])
                        if b in teacher_prefix_index.by_exact
                    }
                    # (vectorized cylinders shared with the v2.5 mid-span path)
                    _spec_pchain = _batched_candidate_cylinders(
                        _all_cands, _fz, _flse, _row_pos, _pref_dev, _exact_dev, tea_byte_map
                    ) if _all_cands else None
            b_lrows = torch.tensor(fast_lrows, dtype=torch.long, device=device)
            t_hidden = _select_hidden_rows_to_device(
                tea_hidden, fast_thidden_rows, device=lm_device, dtype=lm_dtype
            )                                                        # [Nb, H]
            nb = int(t_hidden.shape[0])
            for s in range(0, nb, loss_row_chunk):
                e = min(s + loss_row_chunk, nb)
                with torch.no_grad():
                    z = _guard_teacher_logits(lm_head(t_hidden[s:e]).float(), real_vocab_size, temperature)
                if z.device != device:
                    z = z.to(device)
                z_c = z
                stu_c = stu_logits_shard.index_select(0, b_lrows[s:e])
                if temperature != 1.0:
                    stu_c = stu_c / temperature
                # build the target once (no-grad scatter), then one dense pass
                if ce_weight > 0.0:
                    target_stu, q_other_vec, tea_log_z = build_fast_scatter_target(
                        z_c, phi_teacher_ids, phi_student_ids_pump, vstu
                    )
                    if _spec_pchain is not None:
                        # first-token correction over candidates c_1..c_m:
                        #   target[v] -= P_1; target[c_i] += P_i - P_{i+1}; target[c_m] += P_m
                        _d_rows: list[int] = []
                        _d_sids: list[int] = []
                        _d_vals = []
                        _coff = 0
                        for sp in chain_scatter_specs:
                            m_c = len(sp["cands"])
                            if s <= sp["fast_idx"] < e:
                                row_local = sp["fast_idx"] - s
                                P = _spec_pchain[_coff:_coff + m_c]          # [m_c]
                                _d_rows.append(row_local)
                                _d_sids.append(sp["v_sid"])
                                _d_vals.append(-P[0:1])
                                if m_c > 1:
                                    diffs = (P[:-1] - P[1:]).clamp_min(0.0)  # nested -> >=0
                                    for _ci in range(m_c - 1):
                                        _d_rows.append(row_local)
                                        _d_sids.append(sp["cands"][_ci]["u_sid"])
                                    _d_vals.append(diffs)
                                _d_rows.append(row_local)
                                _d_sids.append(sp["cands"][m_c - 1]["u_sid"])
                                _d_vals.append(P[m_c - 1:m_c])
                            _coff += m_c
                        if _d_rows:
                            apply_chain_scatter_deltas(
                                target_stu, q_other_vec,
                                rows=_d_rows, sids=_d_sids,
                                deltas=torch.cat(_d_vals),
                            )
                    _dv = scatter_target_divergence_sum_local(
                        stu_c, target_stu, q_other_vec,
                        beta=bpm_beta, rkl_lambda=bpm_rkl_lambda,
                    )
                    loss_c, rows_c, q_other_sum, stu_log_z = _dv[:4]
                    del target_stu
                    total_loss_sum = total_loss_sum + loss_c
                    total_ce_sum = total_ce_sum + loss_c
                    total_q_other_tensor = total_q_other_tensor + q_other_sum.to(device=device)
                else:
                    # ce_weight==0: keep only the [Nb] log-Z's for diagnostics
                    rows_c = float(stu_c.shape[0])
                    tea_log_z = torch.logsumexp(z_c.float(), dim=-1)
                    stu_log_z = torch.logsumexp(stu_c.float(), dim=-1)
                    q_other_vec = stu_log_z.new_zeros(stu_log_z.shape)
                total_local_rows += rows_c
                total_fast_rows += rows_c
                stu_prob_for_metrics = None
                q = None
                # stop-row-only: fast rows would pollute the numerator
                if entropy_diagnostics_enabled:
                    with torch.no_grad():
                        tea_ent = _entropy_sum_from_logits_chunked(z_c, log_z=tea_log_z, row_chunk=entropy_row_chunk)
                        if stu_prob_for_metrics is not None:
                            stu_prob = stu_prob_for_metrics
                            stu_ent = -(stu_prob * stu_prob.clamp_min(1e-12).log()).sum(-1).sum()
                        else:
                            stu_prob = None
                            stu_ent = _entropy_sum_from_logits_chunked(stu_c, log_z=stu_log_z, row_chunk=entropy_row_chunk)
                        total_tea_entropy = total_tea_entropy + tea_ent.to(device=device)
                        total_stu_entropy = total_stu_entropy + stu_ent.to(device=device)
                        total_entropy_rows += float(z_c.shape[0])
                        total_tea_entropy_rows += float(z_c.shape[0])  # fast-only teacher-entropy denom
                    if diag_enabled and len(diag_rows) < 32:
                        if q is None:
                            q = torch.softmax(z_c.float(), dim=-1)
                        if stu_prob is None:
                            stu_prob = torch.softmax(stu_c.float(), dim=-1)
                        argmax_ids = torch.argmax(stu_prob, dim=-1).detach().cpu().tolist()
                        # local: the old definition lived in the removed fast-row stop-metric
                        # accumulation; the per-row diag display still wants it.
                        stop_ids = [t for t in teacher_stop_ids if 0 <= t < z_c.shape[-1]]
                        stop_masses = (
                            q[:, stop_ids].sum(-1).detach().cpu().tolist()
                            if stop_ids
                            else [0.0] * int(q.shape[0])
                        )
                        for off, argmax_id in enumerate(argmax_ids):
                            if len(diag_rows) >= 32:
                                break
                            lrow = fast_lrows[s + off] if s + off < len(fast_lrows) else None
                            diag_rows.append({
                                "kind": "fast_boundary",
                                "local_row": lrow,
                                "student_argmax": int(argmax_id),
                                "student_argmax_text": _decode_token(student_tokenizer, int(argmax_id)),
                                "teacher_stop_mass": float(stop_masses[off]),
                                "q_other": float(q_other_vec[off].detach().cpu().item()),
                            })

        # ---- conditional tail rows: exact mid-token marginal ----
        if bpm_perf_probe:
            _perf["fast"] += _psync() - _perf_t
        _perf_t = _psync()
        if tail_groups:
            tail_row_to_groups: dict[int, list[tuple[bytes, int, list[tuple[int, int]]]]] = {}
            tail_forced_delta = True
            # pass 1: single-continuation groups collapse to forced-delta
            delta_lrows: list[int] = []
            delta_ids: list[int] = []
            for (teacher_row, realized_prefix, remaining_len), items in tail_groups.items():
                teacher_ids, phi_selected, tail_image_ids, py_image_ids, tail_all_valid = _ensure_tail_phi_tensors(
                    args, stu_byte_map, tea_byte_map, realized_prefix, remaining_len,
                    real_vocab_size, vstu, device,
                    prefix_index=teacher_prefix_index, student_trie=student_trie,
                )
                if int(teacher_ids.numel()) == 0:
                    continue
                # gate: q_other==0, one continuation, and it is the realized token
                if (
                    tail_forced_delta
                    and tail_all_valid
                    and len(py_image_ids) == 1
                    and all(int(sid) == int(py_image_ids[0]) for _l, _g, sid in items)
                ):
                    for _lrow, _gidx, sid in items:
                        delta_lrows.append(int(_lrow))
                        delta_ids.append(int(sid))
                else:
                    tail_row_to_groups.setdefault(int(teacher_row), []).append(
                        (bytes(realized_prefix), int(remaining_len), items,
                         teacher_ids, phi_selected, tail_image_ids)
                    )

            # pass 2a: forced-delta stays CE -- the target is a one-hot
            for s in range(0, len(delta_lrows), loss_row_chunk):
                lrows_t = torch.tensor(delta_lrows[s:s + loss_row_chunk], dtype=torch.long, device=device)
                ids_t = torch.tensor(delta_ids[s:s + loss_row_chunk], dtype=torch.long, device=device)
                stu_c = stu_logits_shard.index_select(0, lrows_t)
                if temperature != 1.0:
                    stu_c = stu_c / temperature
                loss_d, rows_d = delta_ce_sum_local(stu_c, ids_t)
                total_loss_sum = total_loss_sum + loss_d
                total_ce_sum = total_ce_sum + loss_d
                total_local_rows += rows_d
                # forced-delta rows are counted inside bpm_tail_rows
                total_tail_rows += rows_d
                if entropy_diagnostics_enabled:
                    with torch.no_grad():
                        total_stu_entropy = total_stu_entropy + _entropy_sum_from_logits_chunked(stu_c, row_chunk=entropy_row_chunk).to(device=device)
                        total_entropy_rows += float(stu_c.shape[0])

            # pass 2b: one scatter-target CE over all tail rows
            branch_rows = list(tail_row_to_groups)
            if branch_rows:
                t_hidden = _select_hidden_rows_to_device(
                    tea_hidden, branch_rows, device=lm_device, dtype=lm_dtype
                )
                # flush when full; CE is row-independent, so splits are exact
                tail_buf_lrows: list[int] = []
                tail_buf_target: list = []   # each [1, Vstu] detached
                tail_buf_qother: list = []   # each [1] detached

                def _flush_tail_buf():
                    nonlocal total_loss_sum, total_ce_sum, total_local_rows, total_tail_rows
                    nonlocal total_q_other_tensor, total_stu_entropy, total_entropy_rows
                    if not tail_buf_lrows:
                        return
                    lrows_t = torch.tensor(tail_buf_lrows, dtype=torch.long, device=device)
                    stu_c = stu_logits_shard.index_select(0, lrows_t)
                    if temperature != 1.0:
                        stu_c = stu_c / temperature
                    target_stu = torch.cat(tail_buf_target, dim=0)
                    q_other_vec = torch.cat(tail_buf_qother, dim=0)
                    _tv = scatter_target_divergence_sum_local(
                        stu_c, target_stu, q_other_vec,
                        beta=bpm_beta, rkl_lambda=bpm_rkl_lambda,
                    )
                    loss_t, rows_t, q_other_sum, _stu_log_z = _tv[:4]
                    total_loss_sum = total_loss_sum + loss_t
                    total_ce_sum = total_ce_sum + loss_t
                    total_local_rows += rows_t
                    total_tail_rows += rows_t
                    total_q_other_tensor = total_q_other_tensor + q_other_sum.to(device=device)
                    del target_stu, q_other_vec
                    if entropy_diagnostics_enabled:
                        with torch.no_grad():
                            total_stu_entropy = total_stu_entropy + _entropy_sum_from_logits_chunked(stu_c, log_z=_stu_log_z, row_chunk=entropy_row_chunk).to(device=device)
                            total_entropy_rows += float(stu_c.shape[0])
                    tail_buf_lrows.clear(); tail_buf_target.clear(); tail_buf_qother.clear()

                for s in range(0, int(t_hidden.shape[0]), loss_row_chunk):
                    e = min(s + loss_row_chunk, int(t_hidden.shape[0]))
                    with torch.no_grad():
                        z = _guard_teacher_logits(lm_head(t_hidden[s:e]).float(), real_vocab_size, temperature)
                    if z.device != device:
                        z = z.to(device)
                    for off, teacher_row in enumerate(branch_rows[s:e]):
                        z_row = z[off:off + 1]
                        for (realized_prefix, remaining_len, items,
                             teacher_ids, phi_selected, tail_image_ids) in tail_row_to_groups[int(teacher_row)]:
                            tgt_row, qother_row, tail_valid = build_tail_scatter_target(
                                z_row, teacher_ids, phi_selected, vstu
                            )
                            # prefix-support underflow: target and q_other are already zero
                            total_tail_degenerate_tensor = total_tail_degenerate_tensor + (
                                (1.0 - tail_valid[0]) * float(len(items))
                            )
                            for (lrow, _gidx, _sid) in items:
                                tail_buf_lrows.append(int(lrow))
                                tail_buf_target.append(tgt_row)
                                tail_buf_qother.append(qother_row)
                                if len(tail_buf_lrows) >= loss_row_chunk:
                                    _flush_tail_buf()
                    del z
                _flush_tail_buf()

        # ---- mid-span rows: conditional base + chain deltas ----
        if bpm_perf_probe:
            _perf["tail"] += _psync() - _perf_t
            _perf_t = _psync()
        if midspan_specs and ce_weight > 0.0:
            _ms_all_cands = [c for sp in midspan_specs for c in sp["cands"]]
            _ms_rows = sorted(
                {r for c in _ms_all_cands for (r, _t) in c["exact"]}
                | {c["last_row"] for c in _ms_all_cands}
                | {sp["gov_row"] for sp in midspan_specs}
            )
            _ms_pos = {r: _p for _p, r in enumerate(_ms_rows)}
            _ms_targets: list[torch.Tensor] = []
            _ms_qother: list[torch.Tensor] = []
            _ms_lrows: list[int] = []
            with torch.no_grad():
                _mh = _select_hidden_rows_to_device(
                    tea_hidden, _ms_rows, device=lm_device, dtype=lm_dtype
                )
                _mz_parts = []
                for _c0 in range(0, int(_mh.shape[0]), 1024):
                    _zc2 = _guard_teacher_logits(lm_head(_mh[_c0:_c0 + 1024]).float(), real_vocab_size, temperature)
                    _mz_parts.append(_zc2.to(device))
                _mz = torch.cat(_mz_parts, dim=0) if len(_mz_parts) > 1 else _mz_parts[0]
                del _mz_parts
                _mlse = torch.logsumexp(_mz, dim=-1)
                _mpref = {
                    p: teacher_prefix_index.by_prefix[p].to(_mz.device)
                    for p in ({c["last_prefix"] for c in _ms_all_cands}
                              | {sp["rho"] for sp in midspan_specs})
                    if p in teacher_prefix_index.by_prefix
                }
                _mexact = {
                    b: teacher_prefix_index.by_exact[b].to(_mz.device)
                    for c in _ms_all_cands
                    for b in (tea_byte_map[t] for (_r, t) in c["exact"])
                    if b in teacher_prefix_index.by_exact
                }
                _ms_cyl = _batched_candidate_cylinders(
                    _ms_all_cands, _mz, _mlse, _ms_pos, _mpref, _mexact, tea_byte_map
                )
                # prefix_mass denominators, batched by unique rho
                _dgroups: dict[bytes, list[int]] = {}
                for sp in midspan_specs:
                    if sp["rho"] in _mpref:
                        _dgroups.setdefault(sp["rho"], []).append(_ms_pos[sp["gov_row"]])
                _den_map: dict[tuple[int, bytes], torch.Tensor] = {}
                for _rho_b, _fi_list in _dgroups.items():
                    _fi_u = sorted(set(_fi_list))
                    _fi_t = torch.tensor(_fi_u, dtype=torch.long, device=_mz.device)
                    _dv = torch.logsumexp(_mz[_fi_t][:, _mpref[_rho_b]], dim=-1) - _mlse[_fi_t]
                    for _p2, _fi2 in enumerate(_fi_u):
                        _den_map[(_fi2, _rho_b)] = _dv[_p2]
            # flush when full, like the tail buffer
            def _flush_midspan_buf():
                nonlocal total_loss_sum, total_ce_sum, total_local_rows
                nonlocal total_midspan_corrected_rows, total_q_other_tensor
                nonlocal _ms_targets, _ms_qother, _ms_lrows
                if not _ms_lrows:
                    return
                _ms_lrows_t = torch.tensor(_ms_lrows, dtype=torch.long, device=device)
                stu_c = stu_logits_shard.index_select(0, _ms_lrows_t)
                if temperature != 1.0:
                    stu_c = stu_c / temperature
                _msv = scatter_target_divergence_sum_local(
                    stu_c, torch.cat(_ms_targets, dim=0), torch.cat(_ms_qother, dim=0),
                    beta=bpm_beta, rkl_lambda=bpm_rkl_lambda,
                )
                loss_m, rows_m, q_other_sum_m, _slz_m = _msv[:4]
                total_loss_sum = total_loss_sum + loss_m
                total_ce_sum = total_ce_sum + loss_m
                total_local_rows += rows_m
                # midspan rows are not in fast/tail/chain/stop
                total_midspan_corrected_rows += float(len(_ms_lrows))
                total_q_other_tensor = total_q_other_tensor + q_other_sum_m.to(device=device)
                _ms_targets = []
                _ms_qother = []
                _ms_lrows = []

            _coff2 = 0
            for sp in midspan_specs:
                m_c = len(sp["cands"])
                ids_rho = _mpref.get(sp["rho"])
                if ids_rho is None or int(ids_rho.numel()) == 0:
                    _coff2 += m_c
                    continue   # no prefix support: leave the row untrained (pathological)
                with torch.no_grad():   # target build only; the flush's divergence needs grad
                    _gi = _ms_pos[sp["gov_row"]]
                    log_den = _den_map[(_gi, sp["rho"])]
                    teacher_ids_t, phi_sel_t, _img_t, _img_tuple, _allv = _ensure_tail_phi_tensors(
                        args, stu_byte_map, tea_byte_map, sp["rho"], sp["rem_len"],
                        real_vocab_size, vstu, device, teacher_prefix_index, student_trie,
                    )
                    tgt_row, qo_row, valid_t2 = build_tail_scatter_target(
                        _mz[_gi:_gi + 1], teacher_ids_t, phi_sel_t, vstu
                    )
                    total_tail_degenerate_tensor = total_tail_degenerate_tensor + (1.0 - valid_t2[0])
                    # clamp_max(30) is a NaN guard; valid rows have -log_den <= 21
                    P = _ms_cyl[_coff2:_coff2 + m_c] * torch.exp((-log_den).clamp_max(30.0)) * valid_t2[0]
                    _d_sids = [sp["v_sid"]]
                    _d_vals = [-P[0:1]]
                    if m_c > 1:
                        _d_vals.append((P[:-1] - P[1:]).clamp_min(0.0))
                        _d_sids.extend(c["u_sid"] for c in sp["cands"][:-1])
                    _d_sids.append(sp["cands"][m_c - 1]["u_sid"])
                    _d_vals.append(P[m_c - 1:m_c])
                    apply_chain_scatter_deltas(
                        tgt_row, qo_row,
                        rows=[0] * len(_d_sids), sids=_d_sids, deltas=torch.cat(_d_vals),
                    )
                _ms_targets.append(tgt_row)
                _ms_qother.append(qo_row)
                _ms_lrows.append(int(sp["lrow"]))
                _coff2 += m_c
                if len(_ms_lrows) >= loss_row_chunk:
                    _flush_midspan_buf()
            _flush_midspan_buf()

        # ---- chain rows: 1:N coarser + mid-token conditional targets ----
        if bpm_perf_probe:
            _perf["midspan"] = _perf.get("midspan", 0.0) + (_psync() - _perf_t)
        _perf_t = _psync()
        if chain_items:
            chain_lrows: list[int] = []
            chain_targets: list[dict[int, float]] = []
            query_row_order: list[int] = []
            query_row_to_pos: dict[int, int] = {}
            for tea_js in chunk_plan.values():
                for j in tea_js:
                    row = int(tea_content_rows[j])
                    if row not in query_row_to_pos:
                        query_row_to_pos[row] = len(query_row_order)
                        query_row_order.append(row)
            needed_id_sets = _build_chain_needed_id_sets(
                chain_items=chain_items,
                chunk_plan=chunk_plan,
                tea_content_rows=tea_content_rows,
                tea_content_ids=tea_content_ids,
                tea_off=tea_off,
                stu_off=stu_off,
                tea_byte_map=tea_byte_map,
                student_trie=student_trie,
                teacher_prefix_index=teacher_prefix_index,
                teacher_stop_ids=teacher_stop_ids,
            )
            teacher_queries = _teacher_queries_from_hidden(
                lm_head=lm_head,
                tea_hidden=tea_hidden,
                rows=query_row_order,
                real_vocab_size=real_vocab_size,
                temperature=temperature,
                proj_chunk=loss_row_chunk,
                lm_device=lm_device,
                lm_dtype=lm_dtype,
                prefix_index=teacher_prefix_index,
                needed_id_sets=needed_id_sets,
            )
            query_by_row = {
                row: teacher_queries[pos]
                for row, pos in query_row_to_pos.items()
            }
            chunk_cache: dict[tuple[int, int], Chunk] = {}
            for lrow, ck, n, _gidx in chain_items:
                # Minimal chunk from the governing teacher token to the next sync point.
                cstart = tea_off[ck]
                sp_idx = bisect_right(sync_points, cstart)
                cend = sync_points[sp_idx] if sp_idx < len(sync_points) else tea_off[-1]
                tea_js = chunk_plan.get((int(cstart), int(cend)), [])
                if not tea_js:
                    continue
                cache_key = (int(cstart), int(cend))
                chunk = chunk_cache.get(cache_key)
                if chunk is None:
                    tea_rows = [tea_content_rows[j] for j in tea_js]
                    tea_dists = [query_by_row[int(row)] for row in tea_rows]
                    chunk = Chunk([tea_byte_map[tea_content_ids[j]] for j in tea_js], tea_dists, tea_byte_map)
                    chunk_cache[cache_key] = chunk
                    del tea_dists
                offset = stu_off[n] - cstart
                target = position_target(
                    chunk,
                    offset,
                    student_trie,
                    # chain stop mass uses the same pump column as fast
                    student_eos_id=int(pump_stop_id) if (pump_stop_id >= 0 and getattr(args, "_bpm_stop_pump_on", True)) else None,
                    teacher_stop_ids=teacher_stop_ids,
                )
                if target:
                    chain_lrows.append(lrow)
                    chain_targets.append(target)
                    if diag_enabled and len(diag_rows) < 32:
                        top = sorted(target.items(), key=lambda kv: -kv[1])[:8]
                        diag_rows.append({
                            "kind": "chain",
                            "global_pos": int(_gidx),
                            "offset": int(offset),
                            "realized_student": int(stu_content_ids[n]),
                            "realized_student_text": _decode_token(student_tokenizer, int(stu_content_ids[n])),
                            "target_top": [
                                {
                                    "id": int(tid),
                                    "prob": float(prob),
                                    "text": _decode_token(student_tokenizer, int(tid)),
                                }
                                for tid, prob in top
                            ],
                            "target_sum": float(sum(target.values())),
                        })
            chunk_cache.clear()
            del teacher_queries
            del query_by_row
            if chain_lrows:
                lrows_t = torch.tensor(chain_lrows, dtype=torch.long, device=device)
                target_ids, target_probs, target_mask, row_mask, other, q_other_vals = _pack_sparse_targets(
                    chain_targets, device=device
                )
                stu_c = stu_logits_shard.index_select(0, lrows_t)
                if temperature != 1.0:
                    stu_c = stu_c / temperature
                # beta-aware: chain trains on the fast/tail divergence axis
                _cv = sparse_target_divergence_sum_local(
                    stu_c,
                    target_ids,
                    target_probs,
                    target_mask,
                    row_mask,
                    other_prob=other,
                    beta=bpm_beta,
                    rkl_lambda=bpm_rkl_lambda,
                )
                loss_s, rows_s = _cv[:2]
                total_loss_sum = total_loss_sum + loss_s
                total_ce_sum = total_ce_sum + loss_s
                total_local_rows += rows_s
                total_chain_rows += rows_s
                total_q_other += float(sum(q_other_vals))
                if entropy_diagnostics_enabled:
                    with torch.no_grad():
                        stu_ent = _entropy_sum_from_logits_chunked(stu_c, row_chunk=entropy_row_chunk)
                        total_stu_entropy = total_stu_entropy + stu_ent.to(device=device)
                        total_entropy_rows += float(stu_c.shape[0])

        # ---- explicit student stop rows ----
        if bpm_perf_probe:
            _perf["chain"] += _psync() - _perf_t
        _perf_t = _psync()
        if not stop_items:
            pass  # no realized stop row for this sample
        elif stop_bridge_mode == "skip":
            # 'skip': no gradient on the stop decision, metrics still logged
            total_stop_rows += float(len(stop_items))
            total_stop_teacher += float(sum(x[2] for x in stop_items))
            if diag_enabled:
                for lrow, _stop_row, stop_mass, _rid, _trunc in stop_items:
                    if len(diag_rows) >= 40:
                        break
                    diag_rows.append({
                        "kind": "student_stop",
                        "local_row": int(lrow),
                        "student_eos_id": int(student_eos_id),
                        "realized_stop_id": int(_rid),
                        "realized_stop_text": _decode_token(student_tokenizer, int(_rid)),
                        "teacher_stop_mass": float(stop_mass),
                        "truncated": bool(_trunc),
                        "bridge_mode": "skip",
                    })
            if entropy_diagnostics_enabled:
                with torch.no_grad():
                    _lrows_t = torch.tensor(
                        [x[0] for x in stop_items], dtype=torch.long, device=device
                    )
                    _stu_c = stu_logits_shard.index_select(0, _lrows_t)
                    if temperature != 1.0:
                        _stu_c = _stu_c / temperature
                    _stop_ids_rows = torch.tensor(
                        [int(x[3]) for x in stop_items], dtype=torch.long, device=_stu_c.device
                    )
                    _valid = _stop_ids_rows < int(_stu_c.shape[-1])
                    if bool(_valid.any()):
                        _z = _stu_c.float()
                        _logz = torch.logsumexp(_z, dim=-1)
                        _sel = _z.gather(1, _stop_ids_rows.clamp_max(int(_stu_c.shape[-1]) - 1).unsqueeze(1)).squeeze(1)
                        _pstop = torch.exp(_sel - _logz) * _valid.float()
                        total_stop_student = total_stop_student + _pstop.sum().to(device=device)
        else:
            stop_lrows = [x[0] for x in stop_items]
            # bridge mass -> the realized stop column x[3]; x[4] = truncated
            stop_targets = [
                {int(x[3]): _stop_bridge_target_mass(stop_bridge_mode, x[2], stop_bridge_floor, x[4])}
                for x in stop_items
            ]
            target_ids, target_probs, target_mask, row_mask, other, q_other_vals = _pack_sparse_targets(
                stop_targets, device=device
            )
            lrows_t = torch.tensor(stop_lrows, dtype=torch.long, device=device)
            stu_c = stu_logits_shard.index_select(0, lrows_t)
            if temperature != 1.0:
                stu_c = stu_c / temperature
            # beta-aware: stop trains on the fast/tail divergence axis
            _sv = sparse_target_divergence_sum_local(
                stu_c,
                target_ids,
                target_probs,
                target_mask,
                row_mask,
                other_prob=other,
                # stop rows force beta=0: above it the restoring gradient dies
                beta=0.0,
                rkl_lambda=0.0,
            )
            loss_s, rows_s = _sv[:2]
            total_loss_sum = total_loss_sum + loss_s
            total_ce_sum = total_ce_sum + loss_s
            total_local_rows += rows_s
            total_stop_rows += rows_s
            total_q_other += float(sum(q_other_vals))
            total_stop_teacher += float(sum(x[2] for x in stop_items))
            if diag_enabled:
                # stop rows get reserved diag slots (40 vs the fast rows' 32)
                for lrow, _stop_row, stop_mass, _rid, _trunc in stop_items:
                    if len(diag_rows) >= 40:
                        break
                    diag_rows.append({
                        "kind": "student_stop",
                        "local_row": int(lrow),
                        "student_eos_id": int(student_eos_id),
                        "realized_stop_id": int(_rid),
                        "realized_stop_text": _decode_token(student_tokenizer, int(_rid)),
                        "teacher_stop_mass": float(stop_mass),
                        "truncated": bool(_trunc),
                        "bridge_mode": stop_bridge_mode,
                    })
            if entropy_diagnostics_enabled:
                with torch.no_grad():
                    total_stu_entropy = total_stu_entropy + (
                        _entropy_sum_from_logits_chunked(stu_c, row_chunk=entropy_row_chunk)
                    ).to(device=device)
                    total_entropy_rows += float(stu_c.shape[0])
                    # the realized stop column, not the fixed eos id
                    _stop_ids_rows = torch.tensor(
                        [int(x[3]) for x in stop_items], dtype=torch.long, device=stu_c.device
                    )
                    _valid = _stop_ids_rows < int(stu_c.shape[-1])
                    if bool(_valid.any()):
                        _z = stu_c.float()
                        _logz = torch.logsumexp(_z, dim=-1)
                        _sel = _z.gather(1, _stop_ids_rows.clamp_max(int(stu_c.shape[-1]) - 1).unsqueeze(1)).squeeze(1)
                        _pstop = torch.exp(_sel - _logz) * _valid.float()
                        total_stop_student = total_stop_student + _pstop.sum().to(device=device)

        if bpm_perf_probe:
            _perf["stop"] += _psync() - _perf_t

        if diag_enabled and diag_rows:
            call_index = int(getattr(args, "_bpm_teacher_diag_calls", 0)) + 1
            args._bpm_teacher_diag_calls = call_index
            if call_index % diag_iv == 0:
                try:
                    _dump_bpm_diag(args=args, sample_index=i, call_index=call_index, rows=diag_rows)
                except Exception as exc:  # noqa: BLE001 - diagnostic must not break training
                    _logger.warning(f"[OPD][bpm][teacher-diag] dump skipped: {exc}")

        # per_sample capture: this sample's distill sum and row count
        if track_per_sample:
            rows_delta = total_local_rows - ps_rows_start
            if rows_delta > 0.0:
                per_sample_aligned[i] = rows_delta
                per_sample_distill[i] = ce_weight * (total_ce_sum - ps_ce_start)

    # ----- reductions + final loss in the bpm contract -----
    if args.calculate_per_token_loss:
        cp_rows = total_local_rows
        cp_fast_rows = total_fast_rows
        cp_tail_rows = total_tail_rows
        cp_chain_rows = total_chain_rows
        cp_stop_rows = total_stop_rows
    else:
        # total_local_rows is the single loss denominator
        cp_rows, cp_fast_rows, cp_tail_rows, cp_chain_rows, cp_stop_rows = _reduce_cp_float_counts(
            [
                float(total_local_rows),
                float(total_fast_rows),
                float(total_tail_rows),
                float(total_chain_rows),
                float(total_stop_rows),
            ],
            device=device,
        )
    global_rows = total_local_rows if args.calculate_per_token_loss else cp_rows
    global_alignable = alignable_samples
    total_q_other += float(total_q_other_tensor.detach().item())

    # final loss = ce_weight * CE, computed before the metrics
    total_loss_sum = ce_weight * total_ce_sum

    _accumulate_bpm_train_metrics(
        args,
        device=device,
        cp_size=cp_size,
        total_loss_sum=total_loss_sum,
        total_ce_sum=total_ce_sum,
        total_tea_entropy=total_tea_entropy,
        total_stu_entropy=total_stu_entropy,
        total_entropy_rows=total_entropy_rows if entropy_diagnostics_enabled else 0.0,
        total_tea_entropy_rows=total_tea_entropy_rows if entropy_diagnostics_enabled else 0.0,
        total_stop_teacher=total_stop_teacher,
        total_stop_student=total_stop_student,
        total_rep_frac=total_rep_frac,
        total_q_other=total_q_other,
        total_local_rows=total_local_rows,
        total_fast_rows=total_fast_rows,
        total_tail_rows=total_tail_rows,
        total_chain_rows=total_chain_rows,
        total_stop_rows=total_stop_rows,
        total_label_rows=total_label_rows,
        metric_sample_count=metric_sample_count,
        skipped_samples=skipped_samples,
        total_chain_rerouted_rows=total_chain_rerouted_rows,
        total_midspan_rows=total_midspan_rows,
        total_merge_corrected_rows=total_merge_corrected_rows,
        total_midspan_corrected_rows=total_midspan_corrected_rows,
        total_tail_degenerate_rows=float(total_tail_degenerate_tensor.detach().item()),
        total_route_dropped_rows=total_route_dropped_rows,
        skipped_broken_code=skipped_broken_code,
        total_masked_ws_rows=total_masked_ws_rows,
        total_masked_random_rows=total_masked_random_rows,
    )

    if global_rows <= 0:
        zero_loss = logits.sum() * 0.0
        if track_per_sample:
            args._opd_per_sample_normalizer = 0.0
        if is_logging_rank:
            _logger.info(
                f"[OPD][bpm] microbatch group produced no BPM rows "
                f"(alignable={int(global_alignable)} skipped_byte_mismatch={int(skipped_samples)} "
                f"skipped_broken_code={int(skipped_broken_code)}); "
                "returning zero loss."
            )
        return zero_loss, {"loss": zero_loss.detach(), "bpm_opd_loss": zero_loss.detach()}

    if track_per_sample:
        # per-sample-mean-then-batch-mean; only row counts are CP-reduced
        if ce_weight > 0:
            raise NotImplementedError(
                "--opd-loss-reduction per_sample with --bpm-ce-weight>0 is not implemented "
                "yet (per-sample CE normalization)."
            )
        n_orig = len(response_lengths)
        full_aligned_vec = _reduce_cp_float_counts(
            [float(per_sample_aligned.get(i, 0.0)) for i in range(n_orig)], device=device
        )
        n_samples_ps = float(sum(1 for a in full_aligned_vec if a > 0.0))
        sample_mean_sum = logits.new_zeros(())
        for i in range(n_orig):
            fa = full_aligned_vec[i]
            sd = per_sample_distill.get(i, None)
            if fa > 0.0 and sd is not None:
                sample_mean_sum = sample_mean_sum + sd / fa
        # return the per-sample numerator, stash the count as normalizer
        loss = sample_mean_sum
        args._opd_per_sample_normalizer = n_samples_ps
        # CP tail anchor: no differentiable cross-CP collective in grad
        loss = loss + 0.0 * total_loss_sum
    elif args.calculate_per_token_loss:
        loss = total_loss_sum                       # schedule divides by global response-token count
    else:
        loss = total_loss_sum / max(cp_rows, 1.0)   # CP-shared denominator
    loss = loss + 0.0 * logits.sum()

    bpm_ce_mean = (total_ce_sum / max(total_local_rows, 1.0)).detach()

    if is_logging_rank:
        # the Python chain byte-walk is the dominant step-time risk
        chain_frac_warn = 0.05
        if cp_chain_rows > 0 and cp_rows > 0 and (cp_chain_rows / cp_rows) > chain_frac_warn:
            _logger.warning(
                f"[OPD][bpm] CHAIN byte-walk fraction {cp_chain_rows / cp_rows:.1%} exceeds "
                f"{chain_frac_warn:.0%} ({cp_chain_rows:.0f}/{cp_rows:.0f} rows) -- the Python "
                "byte-walk may dominate step time; investigate coarser/atomic-token routing."
            )
        perf_str = ""
        if bpm_perf_probe:
            perf_str = (
                f" perf_s(classify={_perf['classify']:.2f} fast={_perf['fast']:.2f} "
                f"tail={_perf['tail']:.2f} midspan={_perf.get('midspan', 0.0):.2f} "
                f"chain={_perf['chain']:.2f} "
                f"stop={_perf['stop']:.2f})"
            )
        _logger.info(
            f"[OPD][bpm] rows(cp_shared)={cp_rows:.0f} rows(global)={global_rows:.0f} "
            f"fast={cp_fast_rows:.0f} tail={cp_tail_rows:.0f} chain={cp_chain_rows:.0f} stop={cp_stop_rows:.0f} "
            # chain= counts bytewalk rows only; fast/scatter fold into fast=
            f"1n_in_fast={total_chain_rerouted_rows:.0f} merge_corr={total_merge_corrected_rows:.0f} "
            f"midspan={total_midspan_corrected_rows:.0f}/{total_midspan_rows:.0f} "
            f"ce_mean={float(bpm_ce_mean):.4f} temp={temperature} vocab=FULL "
            f"alignable={int(global_alignable)} skipped_byte_mismatch={int(skipped_samples)} "
            f"skipped_broken_code={int(skipped_broken_code)} "
            f"cp={cp_size} tp={tp_size}{perf_str}"
        )

    # same key set on every return path: model.py asserts the count
    return (
        loss,
        {
            "loss": loss.clone().detach(),
            "bpm_opd_loss": loss.clone().detach(),
        },
    )
