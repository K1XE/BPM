"""SimCT ORACLE — numerically-faithful, self-contained reference for the paper's
SimCT baseline arm, extracted VERBATIM from the reference repo
github.com/sunjie279/SimCT-, and compared against the span path this repo runs.

WHY THIS FILE EXISTS
--------------------
The SimCT arm (examples/baselines/arms/run_simct.sh) claims to reproduce SimCT
exactly.  Its flag set is:

    --simct-alignment-mode span    --simct-span-ctkd-norm
    --opd-loss-type rkl            --opd-temperature 1.0
    --opd-ce-weight 0.0            --opd-topk 0
    (--simct-skip-eos / --simct-stop-token-bridge DEFAULT OFF)

The reference config it must match is SimCT `--kd_algorithm span_ctkd
--kd_loss_fn rkl --kd_temperature 1.0 --kd_ratio 1.0`
(scripts/ctopd/qwen25_gemma2_span_mix10k_lr5e-7.sh etc.; kd_temperature default
1.0 per kdflow/arguments/distillation_args.py:16-18).

This module:
  1. Extracts the SimCT span_ctkd alignment + virtual-vocab + rkl-loss math
     VERBATIM (with SimCT- file:line citations) as standalone functions.
  2. Imports this repo's span path (simct_alignment.py aligner + simct_kernels.py
     loss kernel) unmodified.
  3. On identical inputs proves (a) SAME span alignment units, (b) SAME per-row
     and summed loss — or localizes each divergence to file:line + magnitude.

TWO DELIBERATE DEVIATIONS FROM THE REFERENCE
--------------------------------------------
D1  This implementation aligns on raw bytes; the reference decodes each id in
    isolation (`decode([tid])`).  Identical on ASCII, but the reference drops a
    whole sample when a multi-byte character (emoji, CJK, math unicode) is split
    across byte tokens.
D2  This implementation de-duplicates the teacher-EOS overlap column.

Run:  python slime_plugins/baselines/simct/tests/test_simct_fidelity.py
CPU torch only. The real-tokenizer smoke is skipped unless SIMCT_ORACLE_STUDENT
and SIMCT_ORACLE_TEACHER point at local HF checkpoints.

================================================================================
COMPARATOR SPEC (exact reference semantics, prose)
================================================================================
(1) ALIGNMENT UNIT CONSTRUCTION  [SimCT span_ctkd._align_sequences_with_spans,
    span_ctkd.py:75-173]
    - Operates on LABEL ids = input_ids rolled left by 1, taken at loss_mask
      positions (next-token semantics).  span_ctkd.py:349-350,393-394.
    - Each token id -> text via `tokenizer.decode([tid])` (span_ctkd.py:109-110).
      This is per-token Unicode decode.  CHAR-LEVEL cumulative text matching.
    - Greedy monotone aligner: keep two growing history strings; when the FULL
      histories are equal AND the current teacher/student pieces are equal (or
      both are their own EOS), emit a boundary and advance both; else advance
      whichever side has the SHORTER history (by len()); if equal length but
      pieces differ, advance both.  span_ctkd.py:121-141.
    - A boundary (i,j) closes a segment.  Segment k spans teacher
      [prev_boundary_tea+1 .. boundary_tea] and student
      [prev_boundary_stu+1 .. boundary_stu], stored as half-open
      (tea_start, tea_end, stu_start, stu_end).  span_ctkd.py:162-171.
    - Multi-token span = (te-ts)>1 or (se-ss)>1; else 1:1 unit.
      span_ctkd.py:222-224.
    - EOS is matched by surface equality of each side's own eos_token
      (span_ctkd.py:113-114,122); no EOS is skipped/isolated.

(2) TOKEN-PROB AGGREGATION WITHIN A SPAN  [span_ctkd._build_virtual_vocab_logits,
    span_ctkd.py:257-274]
    - Span logit = ARITHMETIC MEAN of each constituent token's logit at its OWN
      token id: mean_k logits[pos_k, label_k].  span_ctkd.py:262-267 (student),
      269-274 (teacher).  After softmax this is the GEOMETRIC mean of per-token
      probabilities (docstring span_ctkd.py:257-261).  NOT sum-of-logprobs, NOT
      product, NOT first-token.
    - Overlap-vocab logits for the row are taken from the FIRST token position of
      the segment only (span_ctkd.py:239-248), for BOTH 1:1 and span rows.

(3) KD LOSS  [reverse_kl_div.compute_reverse_kl_div, reverse_kl_div.py:6-30;
    driver span_ctkd.training_step:315-446]
    - Virtual common vocab per aligned segment: columns =
      [num_overlap overlap tokens] ++ [num_spans span self-logit dims].
      1:1 rows fill all span dims with -1e9; a span row fills ITS OWN span dim
      with the mean self-logit and all other span dims with -1e9.
      span_ctkd.py:231-310.
    - RKL direction: p = STUDENT, q = TEACHER (teacher detached).
      rkl = sum_v p_s(v) * (log p_s(v) - log p_t(v)).  reverse_kl_div.py:24.
      SUPPORT = the virtual common vocab (overlap UNION spans), NOT student-only
      or teacher-only.
    - TEMPERATURE divides BOTH logits BEFORE softmax (reverse_kl_div.py:14-15);
      kd_temperature default 1.0 -> no-op for the paper arm.
    - NORMALIZATION: per-segment rkl summed with reduction="none".sum()
      (span_ctkd.py:417-422), summed over all samples, then divided ONCE by
      avg_micro_batch_token_num (span_ctkd.py:430).  So the objective is
      PER-TOKEN over the whole micro-batch, one virtual prediction PER SEGMENT.
    - MASKING/EOS: alignment already runs on loss_mask positions; the terminal
      EOS row is KEPT (no skip).  Segments with no boundaries are dropped and
      their tokens counted only toward total_response_tokens (span_ctkd.py:401-403).

(4) WEIGHTING / FILTERING OF SPANS
    - span_ctkd applies NO per-span weighting and NO span filtering: every
      aligned segment (1:1 and span) contributes one rkl term with weight 1.
      (span_ctkd_no_span_loss / simple_ctkd_random_span are separate ablation
      algorithms with span_mask_ratio / random_span_ratio; NOT used by the
      paper arm which is --kd_algorithm span_ctkd.)

(5) SimCT OWN DEFAULTS / HPARAMS (kdflow/arguments/distillation_args.py)
    - kd_ratio        default 0.5  (paper span scripts set 1.0; on-policy forces 1.0)
    - kd_temperature  default 1.0
    - kd_algorithm    default vanilla_kd  (paper span scripts set span_ctkd)
    - kd_loss_fn      default kl   (paper span scripts set rkl)
    - jsd_beta 0.5, skew_lambda 0.1, adaptive_alpha 0.5  (unused by rkl)
    => ours paper flag set (span / rkl / T=1.0 / ce_weight=0.0 / topk=0) MATCHES
       SimCT span_ctkd + rkl + T=1.0 + kd_ratio=1.0.
================================================================================
"""

from __future__ import annotations

import os
import sys

import torch

# repo root, so the plugin package imports when run as a script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slime_plugins.baselines.simct import simct_alignment as _ours_align  # noqa: E402
from slime_plugins.baselines.simct import simct_kernels as _ours_kernels  # noqa: E402
from slime_plugins.bpm.backend import alignment as _byte_align  # noqa: E402

# SMOKE 2 needs real checkpoints; it is skipped when they are absent.
STUDENT_MODEL = os.environ.get("SIMCT_ORACLE_STUDENT", "")
TEACHER_MODEL = os.environ.get("SIMCT_ORACLE_TEACHER", "")


# =============================================================================
# PART A — SimCT REFERENCE, extracted VERBATIM from the published SimCT reference
# =============================================================================

def simct_find_overlap_tokens(student_tokenizer, teacher_tokenizer, device):
    """VERBATIM: SimCT span_ctkd.py:53-69 (SpanCrossTokenizerKD._find_overlap_tokens).

    Ġ->▁ normalized surface-string intersection; EOS pair appended if absent.
    Returns (student_overlap_ids, teacher_overlap_ids) as paired 1-D tensors.
    NOTE: SimCT iterates a Python `set`, so the COLUMN ORDER is nondeterministic;
    the virtual-vocab softmax is column-permutation-invariant so this does not
    change the loss value (only which column index a token lands in).
    """
    student_vocab = {k.replace("Ġ", "▁"): v for k, v in student_tokenizer.get_vocab().items()}
    teacher_vocab = {k.replace("Ġ", "▁"): v for k, v in teacher_tokenizer.get_vocab().items()}
    overlap_tokens = set(student_vocab.keys()) & set(teacher_vocab.keys())
    student_ids = [student_vocab[token] for token in overlap_tokens]
    teacher_ids = [teacher_vocab[token] for token in overlap_tokens]
    stu_eos, tea_eos = student_tokenizer.eos_token_id, teacher_tokenizer.eos_token_id
    if stu_eos not in student_ids or tea_eos not in teacher_ids:
        student_ids.append(stu_eos)
        teacher_ids.append(tea_eos)
    return (
        torch.tensor(student_ids, dtype=torch.long, device=device),
        torch.tensor(teacher_ids, dtype=torch.long, device=device),
    )


def simct_align_sequences_with_spans(tea_label_ids, stu_label_ids, teacher_tokenizer, student_tokenizer):
    """VERBATIM: SimCT span_ctkd.py:75-173
    (SpanCrossTokenizerKD._align_sequences_with_spans), debug logging removed.

    Returns (segments, tea_ids_list, stu_ids_list).
    """
    if len(tea_label_ids) == 0 or len(stu_label_ids) == 0:
        return [], [], []

    tea_ids_list = tea_label_ids if isinstance(tea_label_ids, list) else tea_label_ids.cpu().tolist()
    stu_ids_list = stu_label_ids if isinstance(stu_label_ids, list) else stu_label_ids.cpu().tolist()

    # span_ctkd.py:109-110 — per-token Unicode decode.
    tea_token_texts = [teacher_tokenizer.decode([tid]) for tid in tea_ids_list]
    stu_token_texts = [student_tokenizer.decode([tid]) for tid in stu_ids_list]

    tea_eos = teacher_tokenizer.eos_token
    stu_eos = student_tokenizer.eos_token

    i, j = 0, 0
    boundaries = []
    history_tea = ""
    history_stu = ""

    while i < len(tea_token_texts) and j < len(stu_token_texts):
        is_eos_match = (tea_token_texts[i] == tea_eos and stu_token_texts[j] == stu_eos)
        if history_tea == history_stu and (
            tea_token_texts[i] == stu_token_texts[j] or is_eos_match
        ):
            boundaries.append((i, j))
            history_tea += tea_token_texts[i]
            history_stu += stu_token_texts[j]
            i += 1
            j += 1
        elif len(history_tea) > len(history_stu):
            history_stu += stu_token_texts[j]
            j += 1
        elif len(history_tea) < len(history_stu):
            history_tea += tea_token_texts[i]
            i += 1
        else:
            history_tea += tea_token_texts[i]
            history_stu += stu_token_texts[j]
            i += 1
            j += 1

    if len(boundaries) == 0:
        return [], tea_ids_list, stu_ids_list

    segments = []
    for idx in range(len(boundaries)):
        if idx == 0:
            local_tea_start, local_stu_start = 0, 0
        else:
            local_tea_start = boundaries[idx - 1][0] + 1
            local_stu_start = boundaries[idx - 1][1] + 1
        local_tea_end = boundaries[idx][0] + 1
        local_stu_end = boundaries[idx][1] + 1
        segments.append((local_tea_start, local_tea_end, local_stu_start, local_stu_end))

    return segments, tea_ids_list, stu_ids_list


def simct_build_virtual_vocab_logits(
    segments,
    stu_logits_aligned,
    tea_logits_aligned,
    stu_label_ids_list,
    tea_label_ids_list,
    student_overlap_token_ids,
    teacher_overlap_token_ids,
):
    """VERBATIM: SimCT span_ctkd.py:178-310
    (SpanCrossTokenizerKD._build_virtual_vocab_logits), debug logging removed.

    Returns (stu_virtual_logits, tea_virtual_logits) each
    [num_segments, num_overlap + num_spans].
    """
    num_overlap = student_overlap_token_ids.shape[0]
    device = stu_logits_aligned.device

    span_indices = []
    for seg_idx, (ts, te, ss, se) in enumerate(segments):
        if (te - ts) > 1 or (se - ss) > 1:
            span_indices.append(seg_idx)
    num_spans = len(span_indices)
    seg_to_span_dim = {}
    for dim_idx, seg_idx in enumerate(span_indices):
        seg_to_span_dim[seg_idx] = dim_idx

    virtual_dim = num_overlap + num_spans  # noqa: F841 (kept for parity/clarity)

    stu_rows = []
    tea_rows = []

    for seg_idx, (ts, te, ss, se) in enumerate(segments):
        stu_seg_logits = stu_logits_aligned[ss:se]
        stu_first_logits = stu_seg_logits[0]
        stu_overlap = stu_first_logits[student_overlap_token_ids]

        tea_seg_logits = tea_logits_aligned[ts:te]
        tea_first_logits = tea_seg_logits[0]
        tea_overlap = tea_first_logits[teacher_overlap_token_ids]

        if num_spans > 0:
            stu_span_dims = torch.full((num_spans,), -1e9, device=device, dtype=stu_overlap.dtype)
            tea_span_dims = torch.full((num_spans,), -1e9, device=device, dtype=tea_overlap.dtype)

            if seg_idx in seg_to_span_dim:
                dim_pos = seg_to_span_dim[seg_idx]
                stu_span_token_ids = stu_label_ids_list[ss:se]
                stu_self_logits = torch.stack([
                    stu_seg_logits[k, tid]
                    for k, tid in enumerate(stu_span_token_ids)
                ])
                stu_span_dims[dim_pos] = stu_self_logits.mean()

                tea_span_token_ids = tea_label_ids_list[ts:te]
                tea_self_logits = torch.stack([
                    tea_seg_logits[k, tid]
                    for k, tid in enumerate(tea_span_token_ids)
                ])
                tea_span_dims[dim_pos] = tea_self_logits.mean()

            stu_row = torch.cat([stu_overlap, stu_span_dims])
            tea_row = torch.cat([tea_overlap, tea_span_dims])
        else:
            stu_row = stu_overlap
            tea_row = tea_overlap

        stu_rows.append(stu_row)
        tea_rows.append(tea_row)

    stu_virtual_logits = torch.stack(stu_rows)
    tea_virtual_logits = torch.stack(tea_rows)
    return stu_virtual_logits, tea_virtual_logits


def simct_compute_reverse_kl_div(student_logits, teacher_logits, temperature=1.0, reduction="none"):
    """VERBATIM: SimCT reverse_kl_div.py:6-30 (compute_reverse_kl_div), minus
    @torch.compile.  p=student, q=teacher(detached upstream)."""
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    student_probs = student_log_probs.exp()
    rkl_div = (student_probs * (student_log_probs - teacher_log_probs)).sum(-1)
    if reduction == "mean":
        rkl_div = rkl_div.mean()
    elif reduction == "sum":
        rkl_div = rkl_div.sum()
    return rkl_div


def simct_span_ctkd_sample_loss(
    stu_logits_aligned,
    tea_logits_aligned,
    stu_label_ids,
    tea_label_ids,
    student_tokenizer,
    teacher_tokenizer,
    student_overlap_token_ids,
    teacher_overlap_token_ids,
    temperature=1.0,
):
    """Reference per-sample SimCT span_ctkd numerator, mirroring
    span_ctkd.training_step:395-427 for ONE sample (no batching / avg_token_num
    normalization — that final divide is a scalar the trainer applies once,
    span_ctkd.py:430).

    stu_logits_aligned : [num_stu_loss_tokens, vocab_s] (loss_mask positions)
    tea_logits_aligned : [num_tea_loss_tokens, vocab_t] (loss_mask positions)

    Returns dict with segments, per-segment rkl, and the segment-summed loss.
    """
    segments, tea_ids_list, stu_ids_list = simct_align_sequences_with_spans(
        tea_label_ids, stu_label_ids, teacher_tokenizer, student_tokenizer
    )
    if len(segments) == 0:
        return {"segments": [], "per_segment": torch.zeros(0), "loss_sum": torch.tensor(0.0)}

    stu_virtual, tea_virtual = simct_build_virtual_vocab_logits(
        segments, stu_logits_aligned, tea_logits_aligned,
        stu_ids_list, tea_ids_list,
        student_overlap_token_ids, teacher_overlap_token_ids,
    )
    per_segment = simct_compute_reverse_kl_div(
        stu_virtual, tea_virtual.detach(), temperature=temperature, reduction="none"
    )
    return {
        "segments": segments,
        "stu_virtual": stu_virtual,
        "tea_virtual": tea_virtual,
        "per_segment": per_segment,
        "loss_sum": per_segment.sum(),
    }


# =============================================================================
# PART B — release span path, imported from the implementation under test.
# =============================================================================
# Aligner: simct_alignment._align_simct_sequences_with_spans (SimCT span parity,
#   BUT uses byte-exact texts for ByteLevel-BPE tokenizers, see divergence D1).
# Loss kernel: simct_kernels._simct_full_virtual_vocab_loss_and_entropy_fused
#   (the diagnostics-on path; identical per-row RKL math to reverse_kl_div).
#
# The overlap builder is re-derived here without its Megatron import.

def ours_find_overlap_tokens(student_tokenizer, teacher_tokenizer, device):
    """Build overlap IDs in deterministic student-ID order with teacher-ID dedup."""
    stu_vocab = {k.replace("Ġ", "▁"): v for k, v in student_tokenizer.get_vocab().items()}
    tea_vocab = {k.replace("Ġ", "▁"): v for k, v in teacher_tokenizer.get_vocab().items()}
    overlap_items = sorted(
        ((int(stu_vocab[t]), int(tea_vocab[t])) for t in (set(stu_vocab.keys()) & set(tea_vocab.keys()))),
        key=lambda pair: (pair[0], pair[1]),
    )
    stu_ids = [s for s, _ in overlap_items]
    tea_ids = [t for _, t in overlap_items]
    stu_eos = student_tokenizer.eos_token_id
    tea_eos = teacher_tokenizer.eos_token_id
    if stu_eos is not None and tea_eos is not None and (stu_eos, tea_eos) not in set(zip(stu_ids, tea_ids)):
        stu_ids.append(stu_eos)
        tea_ids.append(tea_eos)
    if len(set(tea_ids)) != len(tea_ids):
        seen_tea = {}
        dedup_stu, dedup_tea = [], []
        for s_id, t_id in zip(stu_ids, tea_ids):
            if t_id in seen_tea:
                if t_id == tea_eos and s_id == stu_eos:
                    dedup_stu[seen_tea[t_id]] = s_id
                continue
            seen_tea[t_id] = len(dedup_tea)
            dedup_stu.append(s_id)
            dedup_tea.append(t_id)
        stu_ids, tea_ids = dedup_stu, dedup_tea
    return (
        torch.tensor(stu_ids, dtype=torch.long, device=device),
        torch.tensor(tea_ids, dtype=torch.long, device=device),
    )


def ours_align_segments(tea_label_ids, stu_label_ids, teacher_tokenizer, student_tokenizer):
    """Run the release SimCT span aligner."""
    return _ours_align._align_simct_sequences_with_spans(
        list(tea_label_ids), list(stu_label_ids), teacher_tokenizer, student_tokenizer
    )


def ours_build_virtual_from_simct_layout(
    segments,
    stu_logits_aligned,
    tea_logits_aligned,
    stu_label_ids,
    tea_label_ids,
    student_overlap_token_ids,
    teacher_overlap_token_ids,
):
    """Build per-row virtual logits with one row-local span column.

    A 1:1 row uses -1e9 in the extra column; a span row uses its mean
    self-logit.  Overlap columns preserve the SimCT order.
    This is numerically identical to SimCT's [num_overlap + num_spans] layout
    because every OTHER span's column is -1e9 (prob 0), so it never affects the
    row's softmax.  Column order here mirrors SimCT so the comparison isolates
    the LOSS MATH, not the (permutation-invariant) overlap ordering.
    """
    num_overlap = student_overlap_token_ids.shape[0]
    device = stu_logits_aligned.device
    stu_rows, tea_rows = [], []
    for (ts, te, ss, se) in segments:
        stu_first = stu_logits_aligned[ss]
        tea_first = tea_logits_aligned[ts]
        stu_overlap = stu_first[student_overlap_token_ids]
        tea_overlap = tea_first[teacher_overlap_token_ids]
        is_span = (te - ts) > 1 or (se - ss) > 1
        if is_span:
            stu_ids_span = stu_label_ids[ss:se]
            tea_ids_span = tea_label_ids[ts:te]
            stu_self = torch.stack([stu_logits_aligned[ss + k, tid] for k, tid in enumerate(stu_ids_span)])
            tea_self = torch.stack([tea_logits_aligned[ts + k, tid] for k, tid in enumerate(tea_ids_span)])
            stu_extra = stu_self.mean().reshape(1)
            tea_extra = tea_self.detach().mean().reshape(1)
        else:
            stu_extra = torch.full((1,), -1e9, device=device, dtype=stu_overlap.dtype)
            tea_extra = torch.full((1,), -1e9, device=device, dtype=tea_overlap.dtype)
        stu_rows.append(torch.cat([stu_overlap, stu_extra]))
        tea_rows.append(torch.cat([tea_overlap, tea_extra]))
    return torch.stack(stu_rows), torch.stack(tea_rows)


def ours_kernel_rkl(stu_virtual, tea_virtual):
    """Release full-scope RKL kernel (non-loss-only path,
    simct_kernels._simct_full_virtual_vocab_loss_and_entropy_fused,
    simct_kernels.py:342-411).  Returns per-row loss."""
    per_loss, _tea_ent, _stu_ent = _ours_kernels._simct_full_virtual_vocab_loss_and_entropy_fused(
        stu_virtual, tea_virtual, loss_type="rkl", jsd_beta=0.5
    )
    return per_loss


def ours_kernel_rkl_loss_only(stu_virtual, tea_virtual):
    """Release loss-only fused RKL kernel (simct_kernels._simct_full_virtual_vocab_loss_only_fused,
    simct_kernels.py:207-261) — the algebraic RKL form used in diagnostics-off runs."""
    return _ours_kernels._simct_full_virtual_vocab_loss_only_fused(
        stu_virtual, tea_virtual, loss_type="rkl", jsd_beta=0.5
    )


# =============================================================================
# Comparison harness
# =============================================================================

def _fmt_segments(segs, n=12):
    return str(segs[:n]) + (" ..." if len(segs) > n else "")


def compare_case(
    name,
    stu_label_ids,
    tea_label_ids,
    student_tokenizer,
    teacher_tokenizer,
    student_overlap_ids,
    teacher_overlap_ids,
    vocab_s,
    vocab_t,
    seed=0,
):
    """Run SimCT ref + ours path on identical logits; print units + loss + deltas."""
    print("=" * 78)
    print(f"CASE: {name}")
    print("=" * 78)
    print(f"  stu_label_ids ({len(stu_label_ids)}): {stu_label_ids[:20]}{' ...' if len(stu_label_ids)>20 else ''}")
    print(f"  tea_label_ids ({len(tea_label_ids)}): {tea_label_ids[:20]}{' ...' if len(tea_label_ids)>20 else ''}")

    # ---- (a) alignment units ----
    simct_segs, _, _ = simct_align_sequences_with_spans(
        tea_label_ids, stu_label_ids, teacher_tokenizer, student_tokenizer
    )
    ours_segs = ours_align_segments(tea_label_ids, stu_label_ids, teacher_tokenizer, student_tokenizer)
    segs_equal = simct_segs == ours_segs
    print(f"\n  [ALIGN] SimCT segments ({len(simct_segs)}): {_fmt_segments(simct_segs)}")
    print(f"  [ALIGN] ours   segments ({len(ours_segs)}): {_fmt_segments(ours_segs)}")
    print(f"  [ALIGN] IDENTICAL: {segs_equal}")
    if not segs_equal:
        # localize first mismatch
        for k in range(max(len(simct_segs), len(ours_segs))):
            a = simct_segs[k] if k < len(simct_segs) else None
            b = ours_segs[k] if k < len(ours_segs) else None
            if a != b:
                print(f"  [ALIGN] first mismatch at unit {k}: SimCT={a} ours={b}")
                break

    # Use SimCT's segments as the common ground for the loss comparison so both
    # loss kernels see identical inputs (alignment divergence, if any, is reported
    # separately above).
    segments = simct_segs
    if len(segments) == 0:
        print("  [LOSS] no segments; skipping loss comparison")
        return segs_equal, True

    # ---- deterministic synthetic logits at the needed positions ----
    g = torch.Generator().manual_seed(seed)
    n_stu = len(stu_label_ids)
    n_tea = len(tea_label_ids)
    stu_logits = torch.randn(n_stu, vocab_s, generator=g, dtype=torch.float32)
    tea_logits = torch.randn(n_tea, vocab_t, generator=g, dtype=torch.float32)

    # ---- SimCT reference ----
    simct_out = simct_span_ctkd_sample_loss(
        stu_logits, tea_logits, stu_label_ids, tea_label_ids,
        student_tokenizer, teacher_tokenizer,
        student_overlap_ids, teacher_overlap_ids, temperature=1.0,
    )
    simct_per = simct_out["per_segment"]
    simct_sum = simct_out["loss_sum"]

    # ---- ours path (same segments, same logits, same overlap layout) ----
    ours_stu_v, ours_tea_v = ours_build_virtual_from_simct_layout(
        segments, stu_logits, tea_logits, stu_label_ids, tea_label_ids,
        student_overlap_ids, teacher_overlap_ids,
    )
    ours_per = ours_kernel_rkl(ours_stu_v, ours_tea_v)
    ours_per_lo = ours_kernel_rkl_loss_only(ours_stu_v, ours_tea_v)
    ours_sum = ours_per.sum()
    ours_sum_lo = ours_per_lo.sum()

    per_delta = (simct_per - ours_per).abs().max().item()
    per_delta_lo = (simct_per - ours_per_lo).abs().max().item()
    sum_delta = (simct_sum - ours_sum).abs().item()
    n_span = sum(1 for (ts, te, ss, se) in segments if (te - ts) > 1 or (se - ss) > 1)

    print(f"\n  [LOSS] num_overlap={student_overlap_ids.numel()} segments={len(segments)} spans={n_span}")
    print(f"  [LOSS] SimCT per-segment rkl[:6]     = {[round(x,6) for x in simct_per[:6].tolist()]}")
    print(f"  [LOSS] ours   per-segment rkl[:6]     = {[round(x,6) for x in ours_per[:6].tolist()]}")
    print(f"  [LOSS] SimCT segment-summed loss      = {simct_sum.item():.8f}")
    print(f"  [LOSS] ours(entropy-fused) summed loss = {ours_sum.item():.8f}   |Δ|={sum_delta:.3e}")
    print(f"  [LOSS] ours(loss-only)     summed loss = {ours_sum_lo.item():.8f}   |Δ|={(simct_sum-ours_sum_lo).abs().item():.3e}")
    print(f"  [LOSS] max |Δ| per-segment (fused)     = {per_delta:.3e}")
    print(f"  [LOSS] max |Δ| per-segment (loss-only) = {per_delta_lo:.3e}")
    loss_ok = per_delta < 1e-4 and per_delta_lo < 1e-4
    print(f"  [LOSS] NUMERICALLY IDENTICAL (tol 1e-4): {loss_ok}")
    return segs_equal, loss_ok


# =============================================================================
# Tiny synthetic tokenizers (byte-level OFF => ours uses legacy decode == SimCT)
# =============================================================================

class TinyTokenizer:
    """Minimal tokenizer stub: get_vocab/decode/convert_ids_to_tokens/eos.
    backend_tokenizer=None so _is_byte_level_tokenizer -> False, i.e. ours
    takes the SAME per-token decode path as SimCT (isolates the loss math)."""

    def __init__(self, id_to_text: dict[int, str], eos_id: int, eos_text: str):
        self._id2t = dict(id_to_text)
        self._id2t[eos_id] = eos_text
        self.eos_token_id = eos_id
        self.eos_token = eos_text
        self.backend_tokenizer = None

    def get_vocab(self):
        return {t: i for i, t in self._id2t.items()}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._id2t.get(int(i), "") for i in ids)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._id2t.get(int(ids), "")
        return [self._id2t.get(int(i), "") for i in ids]


def run_tiny_synthetic():
    print("\n" + "#" * 78)
    print("# SMOKE 1 — tiny synthetic tokenizers (legacy decode path == SimCT)")
    print("#" * 78)
    # Text to encode: "abcdefg" + EOS, tokenized differently by two toy tokenizers.
    # Teacher pieces:  ["ab","cd","ef","g", EOS]
    # Student pieces:  ["a","bc","de","fg", EOS]  -> forces multi-token spans.
    tea_map = {10: "ab", 11: "cd", 12: "ef", 13: "g", 14: "x", 15: "de"}
    stu_map = {20: "a", 21: "bc", 22: "de", 23: "fg", 24: "y", 25: "cd"}
    TEA_EOS, STU_EOS = 99, 199
    teacher_tok = TinyTokenizer(tea_map, TEA_EOS, "<eos_t>")
    student_tok = TinyTokenizer(stu_map, STU_EOS, "<eos_s>")

    # Give the two vocabs a real overlap on surface strings "cd" and "de".
    #   teacher "cd"=11, student "cd"=25 ; teacher "de"=15, student "de"=22
    tea_label_ids = [10, 11, 12, 13, TEA_EOS]   # ab cd ef g <eos_t>
    stu_label_ids = [20, 21, 22, 23, STU_EOS]   # a  bc de fg <eos_s>

    vocab_s = max(stu_map) + 200 + 1
    vocab_t = max(tea_map) + 200 + 1

    simct_stu_ov, simct_tea_ov = simct_find_overlap_tokens(student_tok, teacher_tok, torch.device("cpu"))
    ours_stu_ov, ours_tea_ov = ours_find_overlap_tokens(student_tok, teacher_tok, torch.device("cpu"))
    print(f"  overlap surface set (SimCT count={simct_stu_ov.numel()}, ours count={ours_stu_ov.numel()})")
    print(f"    SimCT stu_overlap_ids={simct_stu_ov.tolist()} tea_overlap_ids={simct_tea_ov.tolist()}")
    print(f"    ours   stu_overlap_ids={ours_stu_ov.tolist()} tea_overlap_ids={ours_tea_ov.tolist()}")

    return compare_case(
        "tiny synthetic (forced spans)",
        stu_label_ids, tea_label_ids, student_tok, teacher_tok,
        simct_stu_ov, simct_tea_ov, vocab_s, vocab_t, seed=1,
    )


def run_real_tokenizers():
    print("\n" + "#" * 78)
    print("# SMOKE 2 — real tokenizers (set SIMCT_ORACLE_STUDENT / SIMCT_ORACLE_TEACHER)")
    print("#" * 78)
    if not (STUDENT_MODEL and TEACHER_MODEL):
        raise RuntimeError(
            "set SIMCT_ORACLE_STUDENT and SIMCT_ORACLE_TEACHER to local HF checkpoints "
            "to run the real-tokenizer smoke; the synthetic proof above needs neither"
        )
    from transformers import AutoTokenizer

    student_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    print(f"  student vocab={len(student_tok.get_vocab())} eos={student_tok.eos_token!r}({student_tok.eos_token_id})")
    print(f"  teacher vocab={len(teacher_tok.get_vocab())} eos={teacher_tok.eos_token!r}({teacher_tok.eos_token_id})")
    print(f"  student byte-level={_byte_align._is_byte_level_tokenizer(student_tok)}  "
          f"teacher byte-level={_byte_align._is_byte_level_tokenizer(teacher_tok)}  "
          f"ours use_byte={_ours_align._use_byte_alignment(teacher_tok, student_tok)}")

    simct_stu_ov, simct_tea_ov = simct_find_overlap_tokens(student_tok, teacher_tok, torch.device("cpu"))
    ours_stu_ov, ours_tea_ov = ours_find_overlap_tokens(student_tok, teacher_tok, torch.device("cpu"))
    print(f"  overlap count: SimCT={simct_stu_ov.numel()}  ours={ours_stu_ov.numel()}  "
          f"(Δ={abs(simct_stu_ov.numel()-ours_stu_ov.numel())})")

    vocab_s = int(max(student_tok.get_vocab().values())) + 1
    vocab_t = int(max(teacher_tok.get_vocab().values())) + 1

    texts = [
        "The derivative of x^2 is 2x, so f'(3) = 6.",           # ASCII math (no multibyte)
        "定理: 若 x>0 则 sqrt(x)>0。证毕。",                       # CJK multibyte (stresses byte vs decode)
    ]
    results = []
    for t in texts:
        stu_ids = student_tok.encode(t, add_special_tokens=False)
        tea_ids = teacher_tok.encode(t, add_special_tokens=False)
        # Append each side's terminal EOS (mirrors label-id construction).
        stu_label_ids = stu_ids + [student_tok.eos_token_id]
        tea_label_ids = tea_ids + [teacher_tok.eos_token_id]
        r = compare_case(
            f"real: {t!r}",
            stu_label_ids, tea_label_ids, student_tok, teacher_tok,
            simct_stu_ov, simct_tea_ov, vocab_s, vocab_t, seed=7,
        )
        results.append((t, r))
    return results


def main():
    torch.manual_seed(0)
    print("SimCT ORACLE — fidelity proof for the paper SimCT (span_ctkd/rkl/T=1.0) arm")
    print(f"  torch={torch.__version__}  device=cpu")
    print(f"  aligner : {_ours_align.__file__}")
    print(f"  kernels : {_ours_kernels.__file__}")

    tiny = run_tiny_synthetic()
    try:
        real = run_real_tokenizers()
    except Exception as exc:  # pragma: no cover
        import traceback
        print(f"\n[SMOKE 2 ERROR] {exc}")
        traceback.print_exc()
        real = None

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  tiny synthetic: align_equal={tiny[0]}  loss_equal={tiny[1]}")
    if real:
        for t, (se, lo) in real:
            print(f"  real {t[:40]!r:42}: align_equal={se}  loss_equal={lo}")


if __name__ == "__main__":
    main()
