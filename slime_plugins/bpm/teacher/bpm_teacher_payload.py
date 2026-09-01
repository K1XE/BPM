"""BPM teacher payload orchestration.

Owns one rollout step's teacher lifecycle: wake the replicas, build prompts and
prefill ids, run prefill-only generate across DP replicas, sleep, then inject.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any

import numpy as np

from slime.utils.types import Sample

from . import bpm_teacher_request as teacher_request
from . import bpm_teacher_rollout as teacher_rollout
from . import bpm_teacher_writeback as teacher_writeback
from ..args.bpm_utils import get_student_tokenizer

logger = logging.getLogger(__name__)


def chunk_sample_indices(indices: list[int], chunk_size: int) -> list[list[int]]:
    if not indices:
        return []
    if chunk_size <= 0 or chunk_size >= len(indices):
        return [indices]
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]


def tensor_nbytes(obj) -> int:
    nbytes = getattr(obj, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return int(np.asarray(obj).nbytes)


def _resolve_teacher_prefill_limit(args) -> int:
    """Max teacher prefill length before SGLang hard-rejects: --opd-teacher-max-prefill-len,
    else the teacher config.json context. Returns 0 (guard off) if neither is available.
    """
    override = int(getattr(args, "opd_teacher_max_prefill_len", 0) or 0)
    if override > 0:
        logger.info(f"[OPD] teacher over-length guard: limit={override} (--opd-teacher-max-prefill-len)")
        return override
    model_path = getattr(args, "opd_teacher_model_path", None)
    if not model_path:
        return 0
    import json
    import os

    cfg_path = os.path.join(model_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as e:  # missing/unreadable config -> disable guard, never crash the run
        logger.warning(
            f"[OPD] teacher over-length guard DISABLED: could not read {cfg_path} ({e}). "
            "Set --opd-teacher-max-prefill-len to enable it."
        )
        return 0
    for container in (cfg, cfg.get("text_config") or {}, cfg.get("llm_config") or {}):
        for key in ("max_position_embeddings", "seq_length", "max_sequence_length", "n_positions"):
            v = container.get(key)
            if isinstance(v, int) and v > 0:
                # rope_scaling can extend the served context well beyond this base value
                if cfg.get("rope_scaling") or container.get("rope_scaling"):
                    logger.warning(
                        f"[OPD] teacher config has rope_scaling; the auto-derived over-length "
                        f"limit={v} (base {key}) may be BELOW the actually-served context, which "
                        "would drop healthy samples. If the teacher serves an extended context, "
                        "set --opd-teacher-max-prefill-len to the real served length."
                    )
                logger.info(f"[OPD] teacher over-length guard: limit={v} (teacher config.json:{key})")
                return v
    logger.warning(
        f"[OPD] teacher over-length guard DISABLED: {cfg_path} has no max_position_embeddings. "
        "Set --opd-teacher-max-prefill-len to enable it."
    )
    return 0


def inject_teacher_hidden_states(self, data: list[Sample]) -> list[Sample]:
    """Run teacher prefill and inject hidden states + teacher token ids into samples.

    Teacher ids are derived locally rather than from SGLang's input_token_logprobs,
    which needs logits for every prefill token and can OOM on long rollouts.
    Order: wakeup -> generate -> sleep; rollout stays offloaded until train.py onloads.
    """
    t0 = time.time()
    logger.info(
        f"[OPD] teacher_fwd start — {len(data)} samples, "
        f"avg_resp_len={sum(len(s.response) for s in data)/max(len(data),1):.0f} chars"
    )

    # rollout engines are offloaded right after generation; no extra release
    if self.args.offload_rollout:
        logger.info("[OPD]   rollout engines already offloaded before teacher forward")
    t_offload = t0

    assert self._teacher_engine_services, "[OPD] teacher engine services list is empty"
    dp_size = len(self._teacher_engine_services)

    # wake all replicas concurrently; each owns its own service
    teacher_rollout.ensure_teacher_awake(self)
    t_wakeup = time.time()
    logger.info(f"[OPD]   all {dp_size} replicas wakeup done ({t_wakeup-t_offload:.2f}s)")

    tokenizer = get_student_tokenizer(self.args)
    eos_token = tokenizer.eos_token
    if eos_token is None:
        raise ValueError("[OPD] student tokenizer eos_token is required for teacher prompt construction")

    prefill_batch = teacher_request.build_teacher_prefill_batch(self, data)
    prompts = prefill_batch.prompts
    prefill_input_ids = prefill_batch.prefill_input_ids
    loss_masks = prefill_batch.loss_masks
    seq_lens = prefill_batch.seq_lens
    teacher_full_token_ids = prefill_batch.teacher_full_token_ids
    mask_eos_token = prefill_batch.mask_eos_token

    # over-length guard: the engine hard-rejects a prefill longer than the teacher
    # context. Zero the loss_mask so the sample skips KD but still trains the policy.
    # perf/opd_teacher_dropped_oversize surfaces a systematic many-drop.
    if not hasattr(self, "_opd_teacher_prefill_limit_cached"):
        self._opd_teacher_prefill_limit_cached = _resolve_teacher_prefill_limit(self.args)
    teacher_prefill_limit = self._opd_teacher_prefill_limit_cached
    n_oversize_dropped = 0
    if teacher_prefill_limit and teacher_prefill_limit > 0:
        oversized = [i for i, sl in enumerate(seq_lens) if sl >= teacher_prefill_limit]
        n_oversize_dropped = len(oversized)
        if oversized:
            for i in oversized:
                loss_masks[i][:] = False
            logger.warning(
                f"[OPD] over-length guard: dropping {len(oversized)}/{len(data)} sample(s) from teacher "
                f"KD this step (teacher prefill seq_len >= limit {teacher_prefill_limit}); "
                f"offending seq_lens={sorted((seq_lens[i] for i in oversized), reverse=True)[:8]}. "
                "loss_mask zeroed -> fully-masked skip path (no prefill, empty payload, no KD)."
            )
    self._opd_oversize_dropped = n_oversize_dropped

    masked_request_indices = [i for i, mask in enumerate(loss_masks) if int(np.count_nonzero(mask)) == 0]
    masked_request_index_set = set(masked_request_indices)
    active_indices = [i for i in range(len(data)) if i not in masked_request_index_set]
    if masked_request_indices:
        logger.info(
            "[OPD]   skipping teacher prefill for fully masked samples: "
            f"{len(masked_request_indices)}/{len(data)}"
        )

    t_build = time.time()
    logger.info(
        f"[OPD]   prompt build done ({t_build-t_wakeup:.2f}s) — "
        f"seq_lens min={min(seq_lens)} max={max(seq_lens)} mean={sum(seq_lens)/len(seq_lens):.0f} "
        f"total_tokens={sum(seq_lens)} active_tokens={sum(seq_lens[i] for i in active_indices)}"
    )

    logger.info(
        "[OPD] teacher_token_ids will be derived locally from teacher tokenizer + loss_mask; "
        "not requesting SGLang input_token_logprobs, which can OOM on long prefill batches."
    )

    # bound per-request IPC size; chunk only by whole sample
    teacher_chunk_size = int(getattr(self.args, "opd_teacher_prefill_chunk_size", 8) or 0)
    teacher_soft_timeout = 300.0
    teacher_rpc_timeout_raw = float(getattr(self.args, "opd_teacher_generate_rpc_timeout", 0.0) or 0.0)
    teacher_hidden_timeout_raw = float(getattr(self.args, "opd_teacher_hidden_recv_timeout", 0.0) or 0.0)
    teacher_rpc_timeout = teacher_rpc_timeout_raw if teacher_rpc_timeout_raw > 0 else None
    teacher_hidden_timeout = teacher_hidden_timeout_raw if teacher_hidden_timeout_raw > 0 else None

    # token-greedy balancing across replicas; fully masked samples skip prefill
    replica_indices: list[list[int]] = [[] for _ in range(dp_size)]
    replica_tokens: list[int] = [0] * dp_size
    for sample_idx in active_indices:
        seq_len = seq_lens[sample_idx]
        min_r = min(range(dp_size), key=lambda r: replica_tokens[r])
        replica_indices[min_r].append(sample_idx)
        replica_tokens[min_r] += seq_len

    replica_chunks = [chunk_sample_indices(replica_indices[r], teacher_chunk_size) for r in range(dp_size)]
    replica_mask_rows = [
        sum(int(np.count_nonzero(loss_masks[i])) for i in replica_indices[r]) for r in range(dp_size)
    ]

    logger.info(
        "[OPD]   DP load balance: "
        + " ".join(
            f"replica{r}={len(replica_indices[r])}samples"
            f"/{replica_tokens[r]}tokens/{replica_mask_rows[r]}masked_rows/{len(replica_chunks[r])}chunks"
            for r in range(dp_size)
        )
        + (f" chunk_size={teacher_chunk_size}" if teacher_chunk_size > 0 else " chunk_size=disabled")
    )
    if teacher_rpc_timeout is None or teacher_hidden_timeout is None:
        logger.info(
            "[OPD]   teacher generate hard timeout disabled; soft watchdog will log only and "
            "will not cancel in-flight chunks."
        )

    # chunks within one replica stay serial so responses never interleave
    def _replica_generate(replica_idx: int):
        idxs = replica_indices[replica_idx]
        chunks = replica_chunks[replica_idx]
        if not idxs:
            return replica_idx, [], []

        svc = self._teacher_engine_services[replica_idx]
        r_hidden_states = []
        replica_t0 = time.time()
        logger.info(
            f"[OPD]   replica{replica_idx} generate start — "
            f"{len(idxs)} samples, {replica_tokens[replica_idx]} seq_tokens, "
            f"{replica_mask_rows[replica_idx]} masked_rows, {len(chunks)} chunks"
        )

        for chunk_idx, chunk_idxs in enumerate(chunks, start=1):
            chunk_t0 = time.time()
            chunk_seq_tokens = sum(seq_lens[i] for i in chunk_idxs)
            chunk_mask_rows = sum(int(np.count_nonzero(loss_masks[i])) for i in chunk_idxs)
            logger.info(
                f"[OPD]   replica{replica_idx} chunk {chunk_idx}/{len(chunks)} start — "
                f"{len(chunk_idxs)} samples, {chunk_seq_tokens} seq_tokens, {chunk_mask_rows} masked_rows"
            )
            hs, _ = svc.generate(
                prompt=[prompts[i] for i in chunk_idxs],
                input_ids=(
                    [prefill_input_ids[i] for i in chunk_idxs]
                    if any(prefill_input_ids[i] is not None for i in chunk_idxs)
                    else None
                ),
                loss_masks=[loss_masks[i] for i in chunk_idxs],
                sampling_params={"max_new_tokens": 0},
                return_hidden_states=True,
                return_token_ids=False,
                response_timeout=teacher_rpc_timeout,
                hidden_recv_timeout=teacher_hidden_timeout,
            )
            if hs is None or len(hs) != len(chunk_idxs):
                raise RuntimeError(
                    f"[OPD] replica{replica_idx} chunk {chunk_idx}: hidden_states count mismatch got "
                    f"{None if hs is None else len(hs)}, expected {len(chunk_idxs)}"
                )
            r_hidden_states.extend(hs)
            chunk_dt = time.time() - chunk_t0
            chunk_hidden_mib = sum(tensor_nbytes(h) for h in (hs or [])) / (1024**2)
            logger.info(
                f"[OPD]   replica{replica_idx} chunk {chunk_idx}/{len(chunks)} done "
                f"({chunk_dt:.2f}s) — hidden={chunk_hidden_mib:.1f} MiB"
            )

        logger.info(
            f"[OPD]   replica{replica_idx} generate done "
            f"({time.time()-replica_t0:.2f}s) — {len(r_hidden_states)} samples"
        )
        return replica_idx, idxs, r_hidden_states

    hidden_size = int(getattr(getattr(self.args, "opd_teacher_config", None), "hidden_size", 0) or 0)
    hidden_states_list = [
        np.empty((0, hidden_size), dtype=np.float16) if i in masked_request_indices else None
        for i in range(len(prompts))
    ]
    future_to_replica: dict[Any, int] = {}
    submit_t = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=dp_size) as executor:
        pending = set()
        for r in range(dp_size):
            fut = executor.submit(_replica_generate, r)
            pending.add(fut)
            future_to_replica[fut] = r

        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=teacher_soft_timeout if teacher_soft_timeout > 0 else None,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                elapsed = time.time() - submit_t
                pending_desc = " ".join(
                    f"replica{future_to_replica[f]}="
                    f"{len(replica_indices[future_to_replica[f]])}samples/"
                    f"{replica_tokens[future_to_replica[f]]}tokens/"
                    f"{len(replica_chunks[future_to_replica[f]])}chunks"
                    for f in pending
                )
                logger.warning(
                    f"[OPD]   teacher generate still running after {elapsed:.1f}s; "
                    f"pending: {pending_desc}. This is a soft watchdog only: no in-flight "
                    "SGLang request is cancelled or killed."
                )
                continue

            for fut in done:
                r_idx, r_idxs, r_hs = fut.result()
                if r_hs is None or len(r_hs) != len(r_idxs):
                    raise RuntimeError(
                        f"[OPD] replica{r_idx}: hidden_states count mismatch "
                        f"got {None if r_hs is None else len(r_hs)}, expected {len(r_idxs)}"
                    )
                for local_pos, global_pos in enumerate(r_idxs):
                    hidden_states_list[global_pos] = r_hs[local_pos]

    # teacher ids are derived locally from the tokenizer + the same loss masks
    token_ids_list = teacher_writeback.masked_teacher_token_ids(teacher_full_token_ids, loss_masks)

    teacher_writeback.ensure_teacher_payload_complete(hidden_states_list, token_ids_list)
    t_generate = time.time()
    hs_shapes = [f"{h.shape}" for h in hidden_states_list[:3]]
    logger.info(
        f"[OPD]   generate done ({t_generate-t_build:.2f}s) — first_shapes={hs_shapes}, "
        f"throughput={sum(seq_lens)/(t_generate-t_build+1e-6):.0f} tok/s"
    )

    # sleep all replicas concurrently before rollout engines are restored
    teacher_rollout.ensure_teacher_asleep(self)
    t_sleep = time.time()
    logger.info(f"[OPD]   all {dp_size} replicas sleep done ({t_sleep-t_generate:.2f}s)")

    # keep rollout engines released: train.py's onload cycle restores them together.
    # Marking them released avoids a double pause, which wedges torch_memory_saver.
    self._bpm_prereleased = True
    logger.info("[OPD]   rollout stays offloaded (phase pattern)")
    t_onload = t_sleep

    teacher_writeback.inject_teacher_payload(
        self,
        data,
        hidden_states_list=hidden_states_list,
        token_ids_list=token_ids_list,
        teacher_full_token_ids=teacher_full_token_ids,
        loss_masks=loss_masks,
        eos_token=eos_token,
        mask_eos_token=mask_eos_token,
    )

    logger.info(
        f"[OPD] teacher_fwd DONE — total={t_onload-t0:.2f}s "
        f"(wakeup={t_wakeup-t_offload:.1f}s build={t_build-t_wakeup:.1f}s "
        f"generate={t_generate-t_build:.1f}s sleep={t_sleep-t_generate:.1f}s)"
    )
    self._last_opd_teacher_perf = teacher_writeback.teacher_perf_metrics(
        self,
        t0=t0,
        t_offload=t_offload,
        t_wakeup=t_wakeup,
        t_build=t_build,
        t_generate=t_generate,
        t_sleep=t_sleep,
        t_onload=t_onload,
        seq_lens=[seq_lens[i] for i in active_indices],
        data=data,
        hidden_states_list=hidden_states_list,
    )
    return data
