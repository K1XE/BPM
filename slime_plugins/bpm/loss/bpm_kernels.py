"""BPM forward-KL over a sparse per-position target (torch + pure-Python reference).

The teacher target is detached and sparse; an explicit complement bin carries the
uncovered mass. Target ids get dCE/dz_v = p(v) - q(v), so the gradient sums to zero.
"""
from __future__ import annotations

import math


def forward_kl_reference(target: dict[int, float], student_logits: dict[int, float]) -> dict:
    """Pure-Python single-row reference, for tests. Returns CE, KL, the per-token grad
    p(v)-q(v) and its sum.
    """
    ids = list(student_logits)
    m = max(student_logits.values())
    exps = {i: math.exp(student_logits[i] - m) for i in ids}
    z = sum(exps.values())
    p = {i: exps[i] / z for i in ids}
    logp = {i: (student_logits[i] - m) - math.log(z) for i in ids}

    q_sum = sum(target.values())
    q_other = 1.0 - q_sum
    ce = -sum(target.get(i, 0.0) * logp[i] for i in ids)
    # complement bin: only contributes when q_other > 0
    if q_other > 1e-12:
        p_in = sum(p[i] for i in target if i in p)
        p_other = max(1.0 - p_in, 1e-12)
        ce += -q_other * math.log(p_other)
    # KL = CE - H(target), including the complement bin
    neg_entropy = sum(target.get(i, 0.0) * math.log(max(target.get(i, 0.0), 1e-12)) for i in ids)
    if q_other > 1e-12:
        neg_entropy += q_other * math.log(q_other)
    kl = ce + neg_entropy
    target_ids = {int(i) for i in target}
    p_in = sum(p[i] for i in target if i in p)
    p_other = max(1.0 - p_in, 1e-12)
    grad = {}
    for i in ids:
        if i in target_ids:
            grad[i] = p[i] - target.get(i, 0.0)
        elif q_other > 1e-12:
            grad[i] = p[i] * (1.0 - q_other / p_other)
        else:
            grad[i] = p[i]
    return {
        "ce": ce,
        "kl": kl,
        "grad": grad,
        "grad_sum": sum(grad.values()),
        "q_sum": q_sum,
    }


def forward_kl_ce_sum_local(
    logits,            # [R, V] full-vocab student logits (torch.Tensor)
    target_ids,        # [R, K] padded sparse student token ids (long)
    target_probs,      # [R, K] padded sparse target probs (float)
    target_mask,       # [R, K] bool, valid entries
    row_mask,          # [R] bool, valid rows
    other_prob=None,   # [R] complement mass (0 if full-vocab byte-BPE + bridge complete)
):
    """CP-local forward-KL CE sum over rows. Returns (loss_sum, local_rows); the caller
    does one CP reduction at the end. Requires full-vocab logits (TP=1).
    """
    import torch

    logits_f = logits.float()
    log_z = torch.logsumexp(logits_f, dim=-1)
    qmask = target_mask.float()
    gathered_logits = logits_f.gather(-1, target_ids.clamp_min(0))
    gathered_logp = gathered_logits - log_z.unsqueeze(-1)
    ce = -(target_probs * qmask * gathered_logp).sum(-1)
    if other_prob is not None:
        # log1mexp is stable as p_in->1, unlike (1-p_in) in fp32
        neg = torch.finfo(logits_f.dtype).min
        masked_logp = torch.where(target_mask, gathered_logp, torch.full_like(gathered_logp, neg))
        log_p_in = torch.logsumexp(masked_logp, dim=-1)                         # log(p_in)
        # clamp below 0 so a full p_in gives a finite penalty, not -inf
        log_p_in = torch.clamp_max(log_p_in, -torch.finfo(logits_f.dtype).eps)
        log_p_other = torch.log(-torch.expm1(log_p_in))                         # log(1 - p_in)
        log_p_other = torch.where(other_prob > 0, log_p_other, torch.zeros_like(log_p_other))
        ce = ce - other_prob * log_p_other
    rm = row_mask.float()
    per_row = ce * rm
    loss_sum = per_row.sum()
    local_rows = float(rm.sum().detach().item())
    return loss_sum, local_rows


def token_marginal_ce_sum_local(
    student_logits,    # [R, Vstu] student logits at boundary rows (already temperature-divided)
    teacher_logits,    # [R, Vtea] teacher logits at boundary rows (already temperature-divided)
    phi_teacher_ids,   # [M] teacher ids where phi>=0
    phi_student_ids,   # [M] mapped student id per selected teacher id
    phi_image_ids,     # [U] student ids in image(phi)
    teacher_tile: int = 4096,
    image_tile: int = 8192,
):
    """CP-local token-marginal forward-KL CE sum:

        CE = -sum_t q[t]*log p[phi[t]] - q_other*log(p_other)

    A gather by phi plus a dot with the teacher row. Returns (loss_sum, local_rows).
    """
    import torch

    if int(student_logits.shape[0]) == 0:
        zero = student_logits.sum() * 0.0
        empty = student_logits.new_empty((0,), dtype=torch.float32)
        return zero, 0.0, zero.detach(), empty, empty

    stu_f = student_logits.float()
    log_z = torch.logsumexp(stu_f, dim=-1)                      # [R]
    tea_f = teacher_logits.float()
    tea_log_z = torch.logsumexp(tea_f, dim=-1)                  # [R]

    if int(phi_image_ids.numel()) == 0 or int(phi_teacher_ids.numel()) == 0:
        ce_main = torch.zeros_like(log_z)
        q_other = torch.ones_like(log_z)
        p_in = torch.zeros_like(log_z)
    else:
        teacher_tile = max(1, int(teacher_tile))
        ce_main = torch.zeros_like(log_z)
        q_valid_sum = torch.zeros_like(log_z)
        for t0 in range(0, int(phi_teacher_ids.numel()), teacher_tile):
            t1 = min(t0 + teacher_tile, int(phi_teacher_ids.numel()))
            ids_t = phi_teacher_ids[t0:t1]
            stu_ids_t = phi_student_ids[t0:t1]
            selected_teacher_logits = tea_f.index_select(1, ids_t)
            q_valid = torch.exp(selected_teacher_logits - tea_log_z.unsqueeze(-1))
            mapped_logits = stu_f.index_select(1, stu_ids_t)
            mapped_logp = mapped_logits - log_z.unsqueeze(-1)
            ce_main = ce_main - (q_valid * mapped_logp).sum(-1)
            q_valid_sum = q_valid_sum + q_valid.sum(-1)
        q_other = (1.0 - q_valid_sum).clamp_min(0.0)
        image_tile = max(1, int(image_tile))
        p_in = torch.zeros_like(log_z)
        for i0 in range(0, int(phi_image_ids.numel()), image_tile):
            i1 = min(i0 + image_tile, int(phi_image_ids.numel()))
            image_logits = stu_f.index_select(1, phi_image_ids[i0:i1])
            p_in = p_in + torch.exp(torch.logsumexp(image_logits, dim=-1) - log_z)
    p_other = (1.0 - p_in).clamp_min(1e-12)                     # [R]
    ce = ce_main - q_other * torch.log(p_other)                # [R]
    loss_sum = ce.sum()
    local_rows = float(student_logits.shape[0])
    return loss_sum, local_rows, q_other.detach(), log_z.detach(), tea_log_z.detach()


def selected_token_marginal_ce_sum_local(
    student_logits,     # [R, Vstu]
    teacher_logits,     # [1 or R, Vtea], already temperature-divided
    teacher_ids,        # [M] selected teacher ids, e.g. conditional prefix support
    phi_selected,       # [M] student id (>=0) or -1 for selected ids
    phi_image_ids,      # [U] student ids in image(phi_selected)
    teacher_tile: int = 4096,
    image_tile: int = 8192,
):
    """Conditional token-marginal CE over a selected teacher-token subset, for exact
    mid-token tail rows. The selected mass is normalized by its row-wise denominator.
    """
    import torch

    if int(student_logits.shape[0]) == 0 or int(teacher_ids.numel()) == 0:
        zero = student_logits.sum() * 0.0
        return zero, 0.0, zero.detach()

    stu_f = student_logits.float()
    log_z = torch.logsumexp(stu_f, dim=-1)
    tea_f = teacher_logits.float()
    tea_log_z = torch.logsumexp(tea_f, dim=-1)
    teacher_tile = max(1, int(teacher_tile))
    q_mass = torch.zeros_like(tea_log_z)
    valid_mass = torch.zeros_like(tea_log_z)
    ce_num = torch.zeros_like(log_z)
    for t0 in range(0, int(teacher_ids.numel()), teacher_tile):
        t1 = min(t0 + teacher_tile, int(teacher_ids.numel()))
        ids_t = teacher_ids[t0:t1]
        phi_t = phi_selected[t0:t1]
        q_sel = torch.exp(tea_f.index_select(1, ids_t) - tea_log_z.unsqueeze(-1))
        valid_t = (phi_t >= 0).float()
        gathered_logits = stu_f.index_select(1, phi_t.clamp_min(0))
        gathered = gathered_logits - log_z.unsqueeze(-1)
        q_mass = q_mass + q_sel.sum(-1)
        valid_mass = valid_mass + (q_sel * valid_t).sum(-1)
        ce_num = ce_num + (q_sel * valid_t * gathered).sum(-1)
    denom = q_mass.clamp_min(1e-12)
    if int(denom.shape[0]) == 1 and int(student_logits.shape[0]) != 1:
        denom = denom.expand(int(student_logits.shape[0]))
        valid_mass = valid_mass.expand(int(student_logits.shape[0]))
    ce_main = -(ce_num / denom)
    q_other = (1.0 - (valid_mass / denom)).clamp_min(0.0)
    if int(phi_image_ids.numel()) == 0:
        p_in = torch.zeros_like(log_z)
    else:
        image_tile = max(1, int(image_tile))
        p_in = torch.zeros_like(log_z)
        for i0 in range(0, int(phi_image_ids.numel()), image_tile):
            i1 = min(i0 + image_tile, int(phi_image_ids.numel()))
            image_logits = stu_f.index_select(1, phi_image_ids[i0:i1])
            p_in = p_in + torch.exp(torch.logsumexp(image_logits, dim=-1) - log_z)
    p_other = (1.0 - p_in).clamp_min(1e-12)
    ce = ce_main - q_other * torch.log(p_other)
    return ce.sum(), float(student_logits.shape[0]), q_other.detach().sum()


def build_fast_scatter_target(
    teacher_logits,    # [R, Vtea] teacher logits at boundary rows (temperature-divided)
    phi_teacher_ids,   # [M] teacher ids where phi>=0
    phi_student_ids,   # [M] mapped student id per selected teacher id
    vstu: int,
    teacher_tile: int = 4096,
):
    """No-grad fast target: target_stu[r,u] = sum_{phi(t)=u} softmax(teacher)[r,t].
    Tiled over teacher cols. Returns (target_stu, q_other, tea_log_z), all detached.
    """
    import torch

    with torch.no_grad():
        tea_f = teacher_logits.float()
        tea_log_z = torch.logsumexp(tea_f, dim=-1)               # [R]
        r = int(tea_f.shape[0])
        target = tea_f.new_zeros((r, int(vstu)))
        q_valid_sum = tea_f.new_zeros((r,))
        m = int(phi_teacher_ids.numel())
        tt = max(1, int(teacher_tile))
        for t0 in range(0, m, tt):
            t1 = min(t0 + tt, m)
            ids_t = phi_teacher_ids[t0:t1]
            stu_ids_t = phi_student_ids[t0:t1]
            q = torch.exp(tea_f.index_select(1, ids_t) - tea_log_z.unsqueeze(-1))   # [R,tile]
            target.scatter_add_(1, stu_ids_t.unsqueeze(0).expand(r, -1), q)
            q_valid_sum = q_valid_sum + q.sum(-1)
        q_other = (1.0 - q_valid_sum).clamp_min(0.0)
    return target, q_other, tea_log_z.detach()


def build_tail_scatter_target(
    teacher_logits,    # [1, Vtea] shared teacher row for the group (temperature-divided)
    teacher_ids,       # [M] selected teacher ids (prefix support)
    phi_selected,      # [M] student id (>=0) or -1
    vstu: int,
    teacher_tile: int = 4096,
):
    """No-grad exact conditional tail target for one group, normalized by denom.

    Returns (target_stu, q_other, valid). When the support mass underflows the target
    is masked to zero, so the row contributes zero CE and zero gradient.
    """
    import torch

    with torch.no_grad():
        tea_f = teacher_logits.float()
        tea_log_z = torch.logsumexp(tea_f, dim=-1)               # [1]
        target = tea_f.new_zeros((1, int(vstu)))
        q_mass = tea_f.new_zeros((1,))
        valid_mass = tea_f.new_zeros((1,))
        m = int(teacher_ids.numel())
        tt = max(1, int(teacher_tile))
        for t0 in range(0, m, tt):
            t1 = min(t0 + tt, m)
            ids_t = teacher_ids[t0:t1]
            phi_t = phi_selected[t0:t1]
            q_sel = torch.exp(tea_f.index_select(1, ids_t) - tea_log_z.unsqueeze(-1))  # [1,tile]
            valid = (phi_t >= 0)
            q_mass = q_mass + q_sel.sum(-1)
            qv = q_sel * valid.float()
            valid_mass = valid_mass + qv.sum(-1)
            target.scatter_add_(1, phi_t.clamp_min(0).unsqueeze(0).expand(1, -1), qv)
        valid_t = (q_mass > 1e-9).float()                        # [1], stays on device
        denom = q_mass.clamp_min(1e-12)
        target = (target / denom) * valid_t.unsqueeze(-1)        # soft drop: all-zero row = 0 CE
        q_other = ((1.0 - valid_mass / denom).clamp_min(0.0)) * valid_t
    return target, q_other, valid_t


def scatter_target_ce_sum_local(
    student_logits,    # [R, Vstu] student logits (temperature-divided), GRAD
    target_stu,        # [R, Vstu] detached fp32 identity-resolved target distribution
    q_other,           # [R]       detached fp32 complement mass
    complement: bool = True,
):
    """Forward-KL CE with the identity-resolved scatter target: one dense backward, no
    atomic scatters.

        CE   = q_valid_sum*logsumexp(z) - (target_stu*z).sum(-1) - q_other*log(p_other)
        d/dz = q_valid_sum*softmax(z) - target_stu

    Returns (loss_sum, rows, q_other_sum, stu_log_z).
    """
    import torch

    r = int(student_logits.shape[0])
    if r == 0:
        zero = student_logits.sum() * 0.0
        empty = student_logits.new_empty((0,), dtype=torch.float32)
        return zero, 0.0, zero.detach(), empty
    stu_f = student_logits.float()
    log_z = torch.logsumexp(stu_f, dim=-1)                       # [R]
    q_valid_sum = target_stu.sum(-1)                            # [R] detached
    ce = q_valid_sum * log_z - (target_stu * stu_f).sum(-1)     # [R] grad-carrying, no softmax
    if complement:
        # log-space complement: 1-p_in cancels in fp32 as p_in->1
        neg = torch.finfo(stu_f.dtype).min                      # finite sentinel (not -inf)
        comp_logits = torch.where(target_stu > 0, torch.full_like(stu_f, neg), stu_f)
        log_p_other = torch.logsumexp(comp_logits, dim=-1) - log_z          # [R], stable
        # empty complement -> huge-negative logsumexp; mask so 0*x==0
        log_p_other = torch.where(q_other > 0, log_p_other, torch.zeros_like(log_p_other))
        ce = ce - q_other * log_p_other
    loss_sum = ce.sum()
    return loss_sum, float(r), q_other.sum().detach(), log_z.detach()


def scatter_target_divergence_sum_local(
    student_logits,    # [R, Vstu] student logits (temperature-divided), GRAD
    target_stu,        # [R, Vstu] detached fp32 byte-marginalized teacher dist q (sums to 1-q_other)
    q_other,           # [R]       detached fp32 complement (other) mass
    beta: float = 0.0,         # TRL GOLD generalized-JSD axis: 0=forward-KL, 0.5=JSD, 1=reverse-KL
    rkl_lambda: float = 0.1,   # BPM-only skew floor for the reverse-KL endpoint (beta>=1); see below
    complement: bool = True,
):
    """BPM divergence on one beta axis, matching TRL GOLD's generalized_jsd_loss:

        beta <= 0    forward-KL D(q||p), delegated to scatter_target_ce_sum_local
        0 < beta < 1 JSD_beta = beta*KL(q||M) + (1-beta)*KL(p||M), M = beta*q+(1-beta)*p
        beta >= 1    reverse-KL D(p||q); rkl_lambda>0 skews the target, 0 floors q to eps

    Returns the scatter_target_ce_sum_local contract.
    """
    import torch

    beta_f = float(beta)
    if beta_f <= 0.0:  # forward-KL == the historical default: exact, light fast path.
        return scatter_target_ce_sum_local(
            student_logits, target_stu, q_other, complement=complement
        )

    import torch.utils.checkpoint as _ckpt

    r = int(student_logits.shape[0])
    V = int(student_logits.shape[1])
    if r == 0:
        zero = student_logits.sum() * 0.0
        empty = student_logits.new_empty((0,), dtype=torch.float32)
        return zero, 0.0, zero.detach(), empty
    eps = 1e-12
    lam_f = float(rkl_lambda)
    is_reverse = beta_f >= 1.0

    def _div_rows(z, q, qo):
        # jsd/rkl hold ~8 dense [n,V] temps, so this runs checkpointed
        stu_f = z.float()
        log_z = torch.logsumexp(stu_f, dim=-1)
        logp = stu_f - log_z.unsqueeze(-1)
        p = logp.exp()
        mask = q > 0
        z2 = torch.zeros_like(p)
        qoc = qo.clamp_min(0.0)
        # log-space complement, as in scatter_target_ce
        neg = torch.finfo(stu_f.dtype).min
        comp = torch.where(mask, torch.full_like(stu_f, neg), stu_f)
        # the p-term other contribution is not gated by q_other
        log_p_other = torch.logsumexp(comp, dim=-1) - log_z
        p_other = log_p_other.exp()
        # soft-dropped rows must contribute exactly zero
        row_live = (mask.any(-1) | (qo > 0)).to(p.dtype)
        if is_reverse:
            # reverse-KL D(p||q) endpoint (beta>=1).
            if lam_f > 0.0:
                # skew reverse-KL D(p || (1-lambda)q + lambda p): bounded
                t = (1.0 - lam_f) * q + lam_f * p
                t_other = ((1.0 - lam_f) * qoc + lam_f * p_other).clamp_min(eps)
                log_t = torch.log(t.clamp_min(eps))
                return (torch.where(mask, p * (logp - log_t), z2).sum(-1)
                        + p_other * (log_p_other - torch.log(t_other))) * row_live
            # pure reverse-KL: q floored to eps, so the penalty <= log(1/eps)
            # summed unmasked, so the complement is covered token-wise
            log_q = torch.log(q.clamp_min(eps))
            return (p * (logp - log_q)).sum(-1) * row_live
        # generalized JSD, TRL GOLD convention (arXiv 2306.13649 Eq.1):
        #   jsd = beta*KL(q||M) + (1-beta)*KL(p||M),  M = beta*q+(1-beta)*p
        m = beta_f * q + (1.0 - beta_f) * p
        m_other = (beta_f * qoc + (1.0 - beta_f) * p_other).clamp_min(eps)
        log_m = torch.log(m.clamp_min(eps))
        log_m_other = torch.log(m_other)
        q_term = beta_f * (torch.where(mask, q * (torch.log(q.clamp_min(eps)) - log_m), z2).sum(-1)
                           + qoc * (torch.log(qoc.clamp_min(eps)) - log_m_other))
        p_term = (1.0 - beta_f) * (torch.where(mask, p * (logp - log_m), z2).sum(-1)
                                   + p_other * (log_p_other - log_m_other))
        return (q_term + p_term) * row_live

    # ~1.5GB transient cap; checkpointing bounds the peak
    SUB = max(32, min(r, int(1.5e9 / (V * 4 * 12))))
    parts = []
    log_z_parts = []
    for s in range(0, r, SUB):
        zc = student_logits[s:s + SUB]
        qc = target_stu[s:s + SUB]
        qoc = q_other[s:s + SUB]
        if zc.requires_grad:
            d = _ckpt.checkpoint(_div_rows, zc, qc, qoc, use_reentrant=False)
        else:
            d = _div_rows(zc, qc, qoc)
        parts.append(d)
        log_z_parts.append(torch.logsumexp(zc.float(), dim=-1).detach())
    per_row = torch.cat(parts)
    loss_sum = per_row.sum()
    log_z = torch.cat(log_z_parts)
    return loss_sum, float(r), q_other.clamp_min(0.0).sum().detach(), log_z


def sparse_target_divergence_sum_local(
    logits,            # [R, V] full-vocab student logits (torch.Tensor), GRAD
    target_ids,        # [R, K] padded sparse student token ids (long)
    target_probs,      # [R, K] padded sparse target probs (float, sum + other_prob == 1)
    target_mask,       # [R, K] bool, valid entries
    row_mask,          # [R]    bool, valid rows
    other_prob=None,   # [R]    complement (other) mass; 0 for masked rows
    beta: float = 0.0,         # same TRL-GOLD axis as scatter_target_divergence_sum_local
    rkl_lambda: float = 0.1,   # reverse-KL skew floor (beta>=1)
):
    """Beta-aware divergence over a sparse (ids, probs, mask) target -- the chain/stop
    route's drop-in for forward_kl_ce_sum_local.

    beta <= 0 delegates to forward_kl_ce_sum_local unchanged; beta > 0 densifies and
    routes through scatter_target_divergence_sum_local. Return contract is unchanged.
    """
    import torch

    beta_f = float(beta)
    if beta_f <= 0.0:
        # forward-KL == today's behavior, by literal delegation.
        return forward_kl_ce_sum_local(
            logits, target_ids, target_probs, target_mask, row_mask,
            other_prob=other_prob,
        )
    logits_f = logits.float()
    r = int(logits_f.shape[0])
    v = int(logits_f.shape[1])
    rm2 = row_mask.unsqueeze(-1)
    valid = target_mask & rm2
    vals = torch.where(valid, target_probs.float(), torch.zeros_like(target_probs, dtype=torch.float32))
    target_stu = logits_f.new_zeros((r, v))            # detached dense byte-marginal target
    target_stu.scatter_add_(1, target_ids.clamp_min(0), vals)   # unique ids/row -> no intra-row collision
    if other_prob is not None:
        q_other = other_prob.float() * row_mask.float()   # masked rows -> 0 (all-zero row = 0 contrib)
    else:
        q_other = logits_f.new_zeros((r,))
    dv = scatter_target_divergence_sum_local(
        logits, target_stu, q_other, beta=beta_f, rkl_lambda=rkl_lambda,
        complement=True,
    )
    # row_mask count, matching forward_kl_ce_sum_local's local_rows
    local_rows = float(row_mask.float().sum().detach().item())
    return dv[0], local_rows


def spanning_chain_parts(u_len: int, j0: int, tea_ids, byte_len_of):
    """Realized-path chain factorization for a boundary-start spanning (1:N) token.

    Returns [(j, kind, take)] with kind in {'exact','prefix'} such that
        P_chain = prod_exact q_j[t_j] * prefix_mass(q_last, last_take)
    None if u's bytes run past the content stream (caller falls back to fast).
    """
    parts = []
    ci = 0
    j = int(j0)
    n = len(tea_ids)
    while ci < int(u_len):
        if j >= n:
            return None
        lk = int(byte_len_of(tea_ids[j]))
        rem = int(u_len) - ci
        if rem <= lk:
            parts.append((j, "prefix", rem))
            return parts
        parts.append((j, "exact", lk))
        ci += lk
        j += 1
    return None  # unreachable for a true spanning row (u_len > first token's lk)


def realized_merge_candidates(trie_root, resp_bytes, a: int, min_len_exclusive: int, max_cands: int = 8):
    """Walk the student byte-trie from offset `a` and return every student token that
    prefixes the realized continuation and is longer than `min_len_exclusive`.
    Returns [(student_id, byte_len)] in increasing length order.
    """
    node = trie_root
    out = []
    n = len(resp_bytes)
    pos = a
    while pos < n:
        node = node.children.get(resp_bytes[pos])
        if node is None:
            break
        pos += 1
        if node.token_id is not None and (pos - a) > int(min_len_exclusive):
            out.append((int(node.token_id), pos - a))
            if len(out) >= int(max_cands):
                break
    return out


def apply_chain_scatter_deltas(target_stu, q_other, rows, sids, deltas):
    """Signed in-place update of the detached fast target; sid == -1 routes to q_other.
    Per boundary row with nested candidates c_1..c_m:

        target[phi(t_gov)] -= cyl(c_1)
        target[c_i]        += cyl(c_i) - cyl(c_{i+1})
        target[c_m]        += cyl(c_m)

    Negative results are clamped to 0 (fp noise only).
    """
    import torch

    if not rows:
        return
    # host-side partition: a tensor.any() gate would sync per spec
    dev = target_stu.device
    in_pairs = [(r, s, k) for k, (r, s) in enumerate(zip(rows, sids)) if s >= 0]
    q_pairs = [(r, k) for k, (r, s) in enumerate(zip(rows, sids)) if s < 0]
    d_t = deltas.to(device=dev, dtype=target_stu.dtype) if torch.is_tensor(deltas) else torch.tensor(
        deltas, dtype=target_stu.dtype, device=dev
    )
    vstu = int(target_stu.shape[1])
    flat = target_stu.view(-1)
    if in_pairs:
        idx = torch.tensor([r * vstu + s for (r, s, _k) in in_pairs], dtype=torch.long, device=dev)
        sel = torch.tensor([k for (_r, _s, k) in in_pairs], dtype=torch.long, device=dev)
        flat.scatter_add_(0, idx, d_t.index_select(0, sel))
        flat.index_put_((idx,), flat.index_select(0, idx).clamp_min(0.0))
    if q_pairs:
        r_q = torch.tensor([r for (r, _k) in q_pairs], dtype=torch.long, device=dev)
        sel_q = torch.tensor([k for (_r, k) in q_pairs], dtype=torch.long, device=dev)
        q_other.index_put_((r_q,), (q_other.index_select(0, r_q) + d_t.index_select(0, sel_q)).clamp_min(0.0))


def apply_chain_scatter_correction(target_stu, q_other, rows, u_sids, v_sids, p_chain):
    """Move the realized-chain mass P_chain from the governing token's phi head (or the
    complement bin when v_sid < 0) onto the realized spanning token u -- the fix for
    the chain_mode=fast sign flip. Mass is moved, not created.
    """
    import torch

    if not rows:
        return
    dev = target_stu.device
    rows_t = torch.tensor(rows, dtype=torch.long, device=dev)
    u_t = torch.tensor(u_sids, dtype=torch.long, device=dev)
    v_t = torch.tensor(v_sids, dtype=torch.long, device=dev)
    p_t = p_chain.to(device=dev, dtype=target_stu.dtype) if torch.is_tensor(p_chain) else torch.tensor(
        p_chain, dtype=target_stu.dtype, device=dev
    )
    flat = target_stu.view(-1)
    vstu = int(target_stu.shape[1])
    flat.scatter_add_(0, rows_t * vstu + u_t, p_t)
    has_v = v_t >= 0
    if bool(has_v.any()):
        idx_v = rows_t[has_v] * vstu + v_t[has_v]
        flat.scatter_add_(0, idx_v, -p_t[has_v])
        flat.index_put_((idx_v,), flat.index_select(0, idx_v).clamp_min(0.0))
    no_v = ~has_v
    if bool(no_v.any()):
        r_nv = rows_t[no_v]
        q_other.index_put_((r_nv,), (q_other.index_select(0, r_nv) - p_t[no_v]).clamp_min(0.0))


def delta_ce_sum_local(student_logits, target_ids):
    """CP-local forced-delta CE sum: the target is a delta on the realized student token.
    Uses F.cross_entropy, so no [N, Vstu] intermediate is retained. Returns (loss_sum, rows).
    """
    import torch
    import torch.nn.functional as F

    if int(student_logits.shape[0]) == 0:
        zero = student_logits.sum() * 0.0
        return zero, 0.0
    loss_sum = F.cross_entropy(student_logits.float(), target_ids, reduction="sum")
    return loss_sum, float(student_logits.shape[0])


def sparse_forward_kl_rows(
    logits,
    target_ids,
    target_probs,
    target_mask,
    row_mask,
    other_prob=None,
):
    """Single-call forward-KL: CP-local CE sum plus one CP reduction.

    Returns (loss_sum, cp_rows, metrics). cp_rows is CP-shared -- the CP-local count
    would inflate the cp-summed loss by ~cp_size.
    """
    from .bpm_loss_utils import _reduce_cp_float_counts

    loss_sum, local_rows = forward_kl_ce_sum_local(
        logits, target_ids, target_probs, target_mask, row_mask, other_prob=other_prob
    )
    cp_rows = _reduce_cp_float_counts([local_rows], device=logits.device)[0]
    metrics = {
        "bpm_rows": cp_rows,
        "bpm_local_rows": local_rows,
        "bpm_ce_mean": (loss_sum / max(local_rows, 1.0)).detach(),
    }
    return loss_sum, cp_rows, metrics
