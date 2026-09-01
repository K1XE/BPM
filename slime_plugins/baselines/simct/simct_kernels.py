"""SimCT divergence kernels and chunk helpers.

These helpers operate on tensors only.  Keep tokenizer alignment and decoded
piece caching in `opd_simct_alignment.py`; keep batch orchestration in the
the SimCT backend.
"""

from __future__ import annotations

from collections.abc import Iterable

import os

import torch


def _sample_complement_divergence_from_logps(
    stu_sample_logp: torch.Tensor,
    tea_sample_logp: torch.Tensor,
    *,
    loss_type: str,
    jsd_beta: float,
) -> torch.Tensor:
    """Sample-token OPD over ``{sampled token, complement}``.

    A singleton top-1 renormalized KL is always zero.  The sample-token OPD
    scope instead keeps the full-vocab probability of the sampled label and
    compares the two-bin distributions ``[p(y), 1-p(y)]``.  This gives
    fkl/rkl/jsd a non-degenerate, differentiable low-memory objective while
    preserving the exact full-softmax denominator for the sampled token.
    """
    loss_type = "fkl" if loss_type == "kl" else loss_type
    if loss_type not in ("fkl", "rkl", "jsd"):
        raise ValueError(f"Unsupported sample-token OPD loss_type: {loss_type}")

    eps = 1e-6
    stu_sample_p = stu_sample_logp.exp().clamp(min=eps, max=1.0 - eps)
    tea_sample_p = tea_sample_logp.detach().exp().clamp(min=eps, max=1.0 - eps)
    stu_probs = torch.stack((stu_sample_p, 1.0 - stu_sample_p), dim=-1)
    tea_probs = torch.stack((tea_sample_p, 1.0 - tea_sample_p), dim=-1)
    stu_log_probs = stu_probs.log()
    tea_log_probs = tea_probs.log()

    if loss_type == "fkl":
        return (tea_probs.detach() * (tea_log_probs.detach() - stu_log_probs)).sum(dim=-1)
    if loss_type == "rkl":
        return (stu_probs * (stu_log_probs - tea_log_probs.detach())).sum(dim=-1)

    beta = torch.tensor(jsd_beta, dtype=torch.float32, device=stu_log_probs.device)
    log_m = torch.logaddexp(
        stu_log_probs + beta.log(),
        tea_log_probs.detach() + (1.0 - beta).log(),
    )
    return (
        jsd_beta * (stu_probs * (stu_log_probs - log_m)).sum(dim=-1)
        + (1.0 - jsd_beta) * (tea_probs.detach() * (tea_log_probs.detach() - log_m)).sum(dim=-1)
    )

def _pad_simct_chunk_for_compile(
    *,
    seg_chunk: list[tuple[int, int, int, int]],
    first_rows: list[int],
    first_tea_rows: list[int],
    mask_values: list[float],
    token_counts: list[int],
    sample_dims: list[int],
    bucket_size: int,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[int],
    list[int],
    list[float],
    list[int],
    list[int],
    int,
]:
    """Pad a SimCT chunk to a compile bucket with zero-mask dummy rows.

    The dummy rows duplicate an existing row so all indexes remain valid; their
    mask/token-count are zero, so they do not affect loss or metrics.  The only
    purpose is to avoid a new torch.compile graph for every tail response length.
    """
    rows = len(seg_chunk)
    if rows == 0 or bucket_size <= 0:
        return seg_chunk, first_rows, first_tea_rows, mask_values, token_counts, sample_dims, rows
    padded_rows = ((rows + bucket_size - 1) // bucket_size) * bucket_size
    # Do not turn a tiny tail over the bucket boundary into a near-2x GEMM/KL
    # tax (e.g. 8193 -> 16384 for bucket=8192).  In that case the occasional
    # exact-shape compile is cheaper than doubling the hot full-vocab work.
    if padded_rows - rows > max(128, rows // 8):
        return seg_chunk, first_rows, first_tea_rows, mask_values, token_counts, sample_dims, rows
    if padded_rows == rows:
        return seg_chunk, first_rows, first_tea_rows, mask_values, token_counts, sample_dims, rows

    pad = padded_rows - rows
    dummy_seg = seg_chunk[-1]
    dummy_stu_row = int(first_rows[-1])
    dummy_tea_row = int(first_tea_rows[-1])
    return (
        seg_chunk + [dummy_seg] * pad,
        first_rows + [dummy_stu_row] * pad,
        first_tea_rows + [dummy_tea_row] * pad,
        mask_values + [0.0] * pad,
        token_counts + [0] * pad,
        sample_dims + [0] * pad,
        rows,
    )

def _effective_simct_compile_bucket_size(*, bucket_size: int, chunk_size: int, distill_scope: str) -> int:
    """Return a safe compile bucket for SimCT full-overlap chunks.

    The bucket is a row-count padding quantum, not a token-budget knob.  Padding a
    CP-local ~8k row chunk to 32k rows multiplies the [rows, overlap_vocab] KL and
    teacher projection work by roughly 4x.  Keep the optimization local to the
    actual chunk size so it cannot silently dominate full-vocab SimCT training.
    """
    bucket = max(int(bucket_size or 0), 0)
    if bucket <= 0 or distill_scope != "full":
        return 0
    chunk = max(int(chunk_size or 1), 1)
    if bucket >= chunk:
        return 0
    return bucket

def _divergence_from_log_probs(
    stu_log_probs: torch.Tensor,
    tea_log_probs: torch.Tensor,
    *,
    loss_type: str,
    jsd_beta: float,
) -> torch.Tensor:
    """Per-row divergence with frozen teacher target."""
    loss_type = "fkl" if loss_type == "kl" else loss_type
    tea_log_probs = tea_log_probs.detach()
    if loss_type == "fkl":
        return (tea_log_probs.exp() * (tea_log_probs - stu_log_probs)).sum(dim=-1)
    if loss_type == "rkl":
        stu_probs = stu_log_probs.exp()
        return (stu_probs * (stu_log_probs - tea_log_probs)).sum(dim=-1)
    if loss_type == "jsd":
        beta = torch.tensor(jsd_beta, dtype=torch.float32, device=stu_log_probs.device)
        log_m = torch.logaddexp(
            stu_log_probs + beta.log(),
            tea_log_probs + (1.0 - beta).log(),
        )
        return (
            jsd_beta * (stu_log_probs.exp() * (stu_log_probs - log_m)).sum(dim=-1)
            + (1.0 - jsd_beta) * (tea_log_probs.exp() * (tea_log_probs - log_m)).sum(dim=-1)
        )
    raise ValueError(f"Unsupported OPD loss_type: {loss_type}")

@torch.compile(fullgraph=False, dynamic=True)
def _simct_virtual_vocab_loss_only_from_logits(
    stu_virtual_logits: torch.Tensor,
    tea_virtual_logits: torch.Tensor,
    *,
    loss_type: str,
    jsd_beta: float,
    distill_scope: str,
    effective_k: int,
    sample_dim: torch.Tensor,
) -> torch.Tensor:
    """Compute SimCT OPD loss without monitoring-only entropy tensors."""
    if stu_virtual_logits.numel() == 0:
        return stu_virtual_logits.new_empty((0,))

    # Sanitize the detached, frozen teacher logits to finite for all scopes
    # (sample/topk/full); gradient-free, no-op when finite. Keeps a non-finite
    # teacher row from NaN-ing the student gradient via 0*inf in backward.
    tea_virtual_logits = torch.nan_to_num(
        tea_virtual_logits.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    if distill_scope == "sample":
        stu_log_probs_full = torch.log_softmax(stu_virtual_logits, dim=-1, dtype=torch.float32)
        tea_log_probs_full = torch.log_softmax(tea_virtual_logits.detach(), dim=-1, dtype=torch.float32)
        row = torch.arange(stu_virtual_logits.shape[0], device=stu_virtual_logits.device)
        sample_dim = sample_dim.to(device=stu_virtual_logits.device)
        return _sample_complement_divergence_from_logps(
            stu_log_probs_full[row, sample_dim],
            tea_log_probs_full[row, sample_dim],
            loss_type=loss_type,
            jsd_beta=jsd_beta,
        )

    if distill_scope == "topk":
        if effective_k <= 1:
            raise ValueError("[OPD][simct] top-K virtual vocab scope requires K >= 2")
        k = min(int(effective_k), int(tea_virtual_logits.shape[-1]))
        if k <= 1:
            raise ValueError("[OPD][simct] virtual vocab has fewer than two dimensions for top-K")
        tea_topk_vals, topk_pos = tea_virtual_logits.detach().topk(k, dim=-1)
        stu_vals = stu_virtual_logits.gather(-1, topk_pos)
        stu_log_probs = torch.log_softmax(stu_vals, dim=-1, dtype=torch.float32)
        tea_log_probs = torch.log_softmax(tea_topk_vals, dim=-1, dtype=torch.float32)
    else:
        stu_log_probs = torch.log_softmax(stu_virtual_logits, dim=-1, dtype=torch.float32)
        tea_log_probs = torch.log_softmax(tea_virtual_logits.detach(), dim=-1, dtype=torch.float32)

    return _divergence_from_log_probs(
        stu_log_probs,
        tea_log_probs,
        loss_type=loss_type,
        jsd_beta=jsd_beta,
    )

@torch.compile(fullgraph=False, dynamic=True)
def _simct_full_virtual_vocab_loss_only_fused(
    stu_virtual_logits: torch.Tensor,
    tea_virtual_logits: torch.Tensor,
    *,
    loss_type: str,
    jsd_beta: float,
) -> torch.Tensor:
    """Stoken-style loss-only KL over the SimCT virtual vocabulary.

    This is the hot path for full-overlap SimCT runs.  Keep the whole
    denominator + divergence schedule in one compiled graph; the previous
    Python wrapper computed row-wise max/sum outside compile and then called a
    second compiled helper, which created extra launches/shape-specialization
    overhead on 32k-response batches.
    """
    if stu_virtual_logits.numel() == 0:
        return stu_virtual_logits.new_empty((0,))

    stu_logits = stu_virtual_logits.float()
    # Sanitize the detached, frozen teacher logits to finite before the loss
    # math (see _simct_full_virtual_vocab_loss_and_entropy_fused); gradient-free,
    # no-op when finite.
    tea_logits = torch.nan_to_num(
        tea_virtual_logits.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    stu_max = stu_logits.max(dim=-1, keepdim=True).values
    tea_max = tea_logits.max(dim=-1, keepdim=True).values
    stu_exp = (stu_logits - stu_max).exp()
    tea_exp = (tea_logits - tea_max).exp()
    stu_exp_sum = stu_exp.sum(dim=-1, keepdim=True)
    tea_exp_sum = tea_exp.sum(dim=-1, keepdim=True)
    stu_log_z = stu_max + stu_exp_sum.log()
    tea_log_z = tea_max + tea_exp_sum.log()

    if loss_type in ("fkl", "kl"):
        tea_weighted_diff = (tea_exp * (tea_logits - stu_logits)).sum(dim=-1, keepdim=True)
        return (tea_weighted_diff / tea_exp_sum - tea_log_z + stu_log_z).squeeze(-1)
    if loss_type == "rkl":
        stu_weighted_diff = (stu_exp * (stu_logits - tea_logits)).sum(dim=-1, keepdim=True)
        return (stu_weighted_diff / stu_exp_sum - stu_log_z + tea_log_z).squeeze(-1)

    stu_log_probs = stu_logits - stu_log_z
    tea_log_probs = tea_logits - tea_log_z
    beta = torch.tensor(jsd_beta, dtype=torch.float32, device=stu_log_probs.device)
    log_m = torch.logaddexp(
        stu_log_probs + beta.log(),
        tea_log_probs.detach() + (1.0 - beta).log(),
    )
    stu_probs = stu_log_probs.exp()
    tea_probs = tea_log_probs.detach().exp()
    return (
        jsd_beta * (stu_probs * (stu_log_probs - log_m)).sum(dim=-1)
        + (1.0 - jsd_beta)
        * (tea_probs * (tea_log_probs.detach() - log_m)).sum(dim=-1)
    )


def _simct_streaming_full_virtual_vocab_loss_only_from_chunks(
    logit_chunks: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    loss_type: str,
    jsd_beta: float,
) -> torch.Tensor:
    """Exact full SimCT loss from overlap-column chunks.

    This is the SimCT analogue of the same-tokenizer full-vocab shard formula.  It
    computes the same fKL/RKL as materializing ``[rows, overlap_vocab]`` logits,
    but streams overlap columns and stores only per-row reductions:

    ``RKL = E_{p_s}[s - t] - log Z_s + log Z_t``
    ``FKL = E_{p_t}[t - s] - log Z_t + log Z_s``

    JSD is intentionally not handled here because the mixture term is not
    additively separable from the two softmax denominators.
    """
    del jsd_beta  # Kept for a uniform OPD-kernel call signature.
    loss_type = "fkl" if loss_type == "kl" else loss_type
    if loss_type not in ("fkl", "rkl"):
        raise ValueError(f"Streaming full SimCT supports fkl/kl/rkl only, got {loss_type}")

    stu_max = tea_max = None
    stu_exp_sum = tea_exp_sum = None
    stu_weighted_diff = tea_weighted_diff = None

    for stu_chunk, tea_chunk in logit_chunks:
        if stu_chunk.numel() == 0:
            continue
        stu = stu_chunk.float()
        # Sanitize the detached, frozen teacher chunk to finite (gradient-free,
        # no-op when finite) so a non-finite row cannot NaN the student gradient.
        tea = torch.nan_to_num(
            tea_chunk.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4
        )
        chunk_stu_max = stu.max(dim=-1, keepdim=True).values
        chunk_tea_max = tea.max(dim=-1, keepdim=True).values
        chunk_stu_exp = (stu - chunk_stu_max).exp()
        chunk_tea_exp = (tea - chunk_tea_max).exp()
        chunk_stu_sum = chunk_stu_exp.sum(dim=-1, keepdim=True)
        chunk_tea_sum = chunk_tea_exp.sum(dim=-1, keepdim=True)
        chunk_stu_weight = (chunk_stu_exp * (stu - tea)).sum(dim=-1, keepdim=True)
        chunk_tea_weight = (chunk_tea_exp * (tea - stu)).sum(dim=-1, keepdim=True)

        if stu_max is None:
            stu_max = chunk_stu_max
            tea_max = chunk_tea_max
            stu_exp_sum = chunk_stu_sum
            tea_exp_sum = chunk_tea_sum
            stu_weighted_diff = chunk_stu_weight
            tea_weighted_diff = chunk_tea_weight
            continue

        new_stu_max = torch.maximum(stu_max, chunk_stu_max)
        old_stu_scale = (stu_max - new_stu_max).exp()
        new_stu_scale = (chunk_stu_max - new_stu_max).exp()
        stu_exp_sum = stu_exp_sum * old_stu_scale + chunk_stu_sum * new_stu_scale
        stu_weighted_diff = stu_weighted_diff * old_stu_scale + chunk_stu_weight * new_stu_scale
        stu_max = new_stu_max

        new_tea_max = torch.maximum(tea_max, chunk_tea_max)
        old_tea_scale = (tea_max - new_tea_max).exp()
        new_tea_scale = (chunk_tea_max - new_tea_max).exp()
        tea_exp_sum = tea_exp_sum * old_tea_scale + chunk_tea_sum * new_tea_scale
        tea_weighted_diff = tea_weighted_diff * old_tea_scale + chunk_tea_weight * new_tea_scale
        tea_max = new_tea_max

    if stu_max is None:
        return torch.empty((0,), dtype=torch.float32)

    stu_log_z = stu_max + stu_exp_sum.log()
    tea_log_z = tea_max + tea_exp_sum.log()
    if loss_type == "rkl":
        return (stu_weighted_diff / stu_exp_sum - stu_log_z + tea_log_z).squeeze(-1)
    return (tea_weighted_diff / tea_exp_sum - tea_log_z + stu_log_z).squeeze(-1)


def _simct_full_virtual_vocab_loss_and_entropy_fused(
    stu_virtual_logits: torch.Tensor,
    tea_virtual_logits: torch.Tensor,
    *,
    loss_type: str,
    jsd_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-overlap SimCT loss plus entropy in one compiled pass.

    This is the hot full-overlap SimCT path.  Keep it in eager/native torch
    rather than torch.compile: the GLM/Qwen run has many different row counts
    even after bucketing, and compiling this multi-output [rows, 143k] helper
    caused repeated graph specialization while the entropy outputs could
    collapse to literal zeros.  Native log-sum-exp kernels are stable here and
    avoid compile warmup on every new tail shape.
    """
    if stu_virtual_logits.numel() == 0:
        empty = stu_virtual_logits.new_empty((0,))
        return empty, empty, empty

    stu_logits = stu_virtual_logits.float()
    # Sanitize the (detached, frozen) teacher logits to finite BEFORE the
    # divergence math: a non-finite teacher row (e.g. an fp16-overflow inf that
    # slipped past the capture clamp, or a bad teacher prefill) would otherwise
    # produce inf/nan in logsumexp/exp whose SAVED tensors NaN the student
    # gradient in backward (0*inf).  nan_to_num here is gradient-free (teacher is
    # detached) and a no-op when all rows are finite.
    tea_logits = torch.nan_to_num(
        tea_virtual_logits.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    stu_lse = torch.logsumexp(stu_logits, dim=-1, keepdim=True)
    tea_lse = torch.logsumexp(tea_logits, dim=-1, keepdim=True)
    stu_log_probs = stu_logits - stu_lse
    tea_log_probs = tea_logits - tea_lse
    stu_probs = stu_log_probs.exp()

    if loss_type in ("fkl", "kl"):
        tea_probs = tea_log_probs.exp()
        per_loss = (tea_probs * (tea_log_probs - stu_log_probs)).sum(dim=-1)
    elif loss_type == "rkl":
        per_loss = (stu_probs * (stu_log_probs - tea_log_probs)).sum(dim=-1)
    else:
        beta = torch.tensor(jsd_beta, dtype=torch.float32, device=stu_log_probs.device)
        log_m = torch.logaddexp(
            stu_log_probs + beta.log(),
            tea_log_probs + (1.0 - beta).log(),
        )
        tea_probs = tea_log_probs.exp()
        per_loss = (
            jsd_beta * (stu_probs * (stu_log_probs - log_m)).sum(dim=-1)
            + (1.0 - jsd_beta) * (tea_probs * (tea_log_probs - log_m)).sum(dim=-1)
        )

    with torch.no_grad():
        # H(softmax(x)) = logsumexp(x) - E_p[x].  This avoids multiplying
        # probabilities by very negative log-probabilities and gives non-zero
        # diagnostics for the same normalized distribution used by the loss.
        stu_probs_detached = stu_log_probs.detach().exp()
        tea_probs_detached = tea_log_probs.detach().exp()
        stu_entropy = (
            stu_lse.detach().squeeze(-1)
            - (stu_probs_detached * stu_logits.detach()).sum(dim=-1)
        )
        tea_entropy = (
            tea_lse.detach().squeeze(-1)
            - (tea_probs_detached * tea_logits.detach()).sum(dim=-1)
        )
        stu_entropy = torch.nan_to_num(stu_entropy, nan=0.0, posinf=0.0, neginf=0.0)
        tea_entropy = torch.nan_to_num(tea_entropy, nan=0.0, posinf=0.0, neginf=0.0)
    return per_loss, tea_entropy, stu_entropy


# Optional env-gated compile of the entropy kernel (SIMCT_COMPILE_ENTROPY=1), OFF by
# default.  The docstring above documents that an earlier compile attempt produced
# zero/NaN entropy on some shape buckets; dynamic=True may neutralize that on current
# torch (the loss-only sibling at L198 compiles fine with it).  Opt in and CONFIRM
# train/simct_tea_entropy & simct_stu_entropy are non-zero on the first steps before
# relying on it.  Saves ~100-150s/step when stable (basic diagnostics eager path).
if os.environ.get("SIMCT_COMPILE_ENTROPY", "0") == "1":
    _simct_full_virtual_vocab_loss_and_entropy_fused = torch.compile(
        fullgraph=False, dynamic=True
    )(_simct_full_virtual_vocab_loss_and_entropy_fused)


@torch.compile(fullgraph=False, dynamic=True)
def _simct_virtual_vocab_loss_from_logits(
    stu_virtual_logits: torch.Tensor,
    tea_virtual_logits: torch.Tensor,
    *,
    loss_type: str,
    jsd_beta: float,
    distill_scope: str,
    effective_k: int,
    sample_dim: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute OPD over SimCT virtual common vocab.

    ``stu_virtual_logits``/``tea_virtual_logits`` are [segments, overlap+spans].
    Scope semantics:
      - full: whole virtual vocabulary
      - topk: teacher top-K virtual dimensions
      - sample: observed segment virtual dimension vs complement bucket
    """
    if stu_virtual_logits.numel() == 0:
        empty = stu_virtual_logits.new_empty((0,))
        return empty, empty, empty, empty

    # Sanitize the detached, frozen teacher logits to finite for all scopes;
    # gradient-free, no-op when finite (see the sibling kernels).
    tea_virtual_logits = torch.nan_to_num(
        tea_virtual_logits.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    if distill_scope == "sample":
        stu_log_probs_full = torch.log_softmax(stu_virtual_logits, dim=-1, dtype=torch.float32)
        tea_log_probs_full = torch.log_softmax(tea_virtual_logits.detach(), dim=-1, dtype=torch.float32)
        row = torch.arange(stu_virtual_logits.shape[0], device=stu_virtual_logits.device)
        stu_sample_logp = stu_log_probs_full[row, sample_dim.to(device=stu_virtual_logits.device)]
        tea_sample_logp = tea_log_probs_full[row, sample_dim.to(device=stu_virtual_logits.device)]
        per_loss = _sample_complement_divergence_from_logps(
            stu_sample_logp,
            tea_sample_logp,
            loss_type=loss_type,
            jsd_beta=jsd_beta,
        )
        with torch.no_grad():
            stu_p = stu_sample_logp.exp().clamp(min=1e-6, max=1.0 - 1e-6)
            tea_p = tea_sample_logp.exp().clamp(min=1e-6, max=1.0 - 1e-6)
            stu_entropy = -(stu_p * stu_p.log() + (1.0 - stu_p) * (1.0 - stu_p).log())
            tea_entropy = -(tea_p * tea_p.log() + (1.0 - tea_p) * (1.0 - tea_p).log())
        return per_loss, tea_entropy, stu_entropy, tea_sample_logp.detach()

    if distill_scope == "topk":
        if effective_k <= 1:
            raise ValueError("[OPD][simct] top-K virtual vocab scope requires K >= 2")
        k = min(int(effective_k), int(tea_virtual_logits.shape[-1]))
        if k <= 1:
            raise ValueError("[OPD][simct] virtual vocab has fewer than two dimensions for top-K")
        tea_topk_vals, topk_pos = tea_virtual_logits.detach().topk(k, dim=-1)
        stu_vals = stu_virtual_logits.gather(-1, topk_pos)
        stu_log_probs = torch.log_softmax(stu_vals, dim=-1, dtype=torch.float32)
        tea_log_probs = torch.log_softmax(tea_topk_vals, dim=-1, dtype=torch.float32)
    else:
        stu_log_probs = torch.log_softmax(stu_virtual_logits, dim=-1, dtype=torch.float32)
        tea_log_probs = torch.log_softmax(tea_virtual_logits.detach(), dim=-1, dtype=torch.float32)

    per_loss = _divergence_from_log_probs(
        stu_log_probs,
        tea_log_probs,
        loss_type=loss_type,
        jsd_beta=jsd_beta,
    )
    with torch.no_grad():
        tea_probs = tea_log_probs.exp()
        stu_probs = stu_log_probs.exp()
        tea_entropy = -(tea_probs * tea_log_probs).sum(dim=-1)
        stu_entropy = -(stu_probs * stu_log_probs).sum(dim=-1)
        tea_sample_logp = tea_log_probs.gather(
            -1,
            torch.zeros((tea_log_probs.shape[0], 1), dtype=torch.long, device=tea_log_probs.device),
        ).squeeze(-1)
    return per_loss, tea_entropy, stu_entropy, tea_sample_logp.detach()

def _simct_virtual_sample_logps(
    stu_virtual_logits: torch.Tensor,
    tea_virtual_logits: torch.Tensor,
    sample_dim: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of the observed SimCT virtual unit in each row."""
    stu_log_probs = torch.log_softmax(stu_virtual_logits, dim=-1, dtype=torch.float32)
    # Sanitize the detached, frozen teacher logits to finite (gradient-free,
    # no-op when finite) before the softmax.
    tea_virtual_logits = torch.nan_to_num(
        tea_virtual_logits.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    tea_log_probs = torch.log_softmax(tea_virtual_logits.detach(), dim=-1, dtype=torch.float32)
    row = torch.arange(stu_virtual_logits.shape[0], device=stu_virtual_logits.device)
    sample_dim = sample_dim.to(device=stu_virtual_logits.device)
    return stu_log_probs[row, sample_dim], tea_log_probs[row, sample_dim].detach()
