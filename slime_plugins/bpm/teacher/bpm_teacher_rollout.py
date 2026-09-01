"""BPM teacher engine placement, startup, and sleep/wake lifecycle.

Functions take the RolloutManager as `self`, so the teacher contracts stay out of
ray/rollout.py; the manager only holds the state and calls these entry points.
"""

from __future__ import annotations

import logging
import socket

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from slime.utils.logging_utils import configure_logger

from ..args.bpm_utils import (
    is_bpm_enabled,
    normalize_teacher_offload_tags,
    teacher_offload_tags_release_weights,
)

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=0)
class TeacherEngineRayActor:
    """Runs one BPM teacher engine replica on a selected Ray node."""

    def __init__(self, config, replica_idx: int):
        configure_logger()
        from .bpm_teacher_engine import SGLangTeacherEngineService

        self.replica_idx = replica_idx
        self.service = SGLangTeacherEngineService(config)
        self.service.start()

    def ready(self):
        return True

    def generate(self, **kwargs):
        return self.service.generate(**kwargs)

    def sleep(self, **kwargs):
        return self.service.sleep(**kwargs)

    def wakeup(self, **kwargs):
        return self.service.wakeup(**kwargs)

    def shutdown(self):
        self.service.shutdown()
        return True


class RayTeacherReplicaClient:
    """Synchronous client wrapper matching SGLangTeacherEngineService methods."""

    def __init__(self, actor):
        self.actor = actor

    def generate(self, **kwargs):
        return ray.get(self.actor.generate.remote(**kwargs))

    def sleep(self, **kwargs):
        return ray.get(self.actor.sleep.remote(**kwargs))

    def wakeup(self, **kwargs):
        return ray.get(self.actor.wakeup.remote(**kwargs))

    def shutdown(self):
        return ray.get(self.actor.shutdown.remote())


def init_bpm_teacher_engines(self) -> None:
    """Initialize BPM teacher replicas for a RolloutManager instance."""
    args = self.args
    self._teacher_engine_service = None  # kept as an alias for _services[0] or None
    self._teacher_engine_services: list = []  # all replicas; len == dp_size
    self._teacher_is_asleep = False
    if not is_bpm_enabled(args):
        return

    from .bpm_teacher_engine import SGLangTeacherEngineService, TeacherEngineConfig

    dp_size = getattr(args, "opd_teacher_dp_size", 1)
    tp_size = args.opd_teacher_tp_size
    ep_size = getattr(args, "opd_teacher_ep_size", 1)
    pp_size = 1
    gpus_per_replica = tp_size * pp_size
    base_gpu_id = 0

    rollout_engines = getattr(self, "rollout_engines", None) or []
    if args.offload_rollout and rollout_engines:
        logger.info(
            f"[OPD] Offloading {len(rollout_engines)} rollout engines "
            "before teacher initialization to free GPU memory..."
        )
        # a just-ready engine may still be capturing CUDA graphs; a release then can
        # 504 even though it lands engine-side. Gate on one generation, then retry.
        import time as _time

        ray.get([engine.health_generate.remote(timeout=300.0) for engine in rollout_engines])
        for attempt in range(3):
            try:
                self.offload()
                break
            except Exception as exc:  # late 504: the release may still have landed
                if attempt == 2:
                    raise
                logger.warning(f"[OPD] rollout offload attempt {attempt + 1} failed ({exc}); retrying in 30s")
                _time.sleep(30)
        # create_rollout_manager offloads again right after __init__; mark the engines
        # pre-released so that call is a no-op
        self._bpm_prereleased = True
        logger.info("[OPD] Rollout engines offloaded successfully.")

    teacher_placements = build_teacher_placements(self, dp_size, gpus_per_replica, base_gpu_id)
    # phase 1: dispatch remote replicas without blocking -- disjoint GPU sets, so
    # their loads run in parallel. Local replicas start serially because start()
    # mutates the parent CUDA_VISIBLE_DEVICES.
    services: list = [None] * len(teacher_placements)
    remote_pending: list[tuple[int, object, object]] = []  # (replica_idx, actor, ready_ref)
    local_pending: list[tuple[int, object]] = []  # (replica_idx, teacher_config)
    for replica_idx, placement in enumerate(teacher_placements):
        replica_cvd = placement["cuda_visible_devices"]
        teacher_config = TeacherEngineConfig(
            model_path=args.opd_teacher_model_path,
            tokenizer_path=getattr(args, "bpm_teacher_tokenizer_path", None),
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=pp_size,
            chunked_prefill_size=-1,
            mem_fraction_static=args.opd_teacher_mem_fraction,
            offload_tags="all",
            base_gpu_id=0,
            cuda_visible_devices=replica_cvd,
            allow_prefix_truncation=False,
        )
        logger.info(
            f"[OPD] Initializing teacher replica {replica_idx}/{dp_size}: "
            f"model={teacher_config.model_path}, tokenizer={teacher_config.tokenizer_path}, "
            f"tp={tp_size}, ep={ep_size}, pp={pp_size}, "
            f"mem_frac={teacher_config.mem_fraction_static}, "
            f"node={placement['node_idx']}({placement['address']}), "
            f"CUDA_VISIBLE_DEVICES={replica_cvd}"
        )
        if placement["is_local"]:
            local_pending.append((replica_idx, teacher_config))
        else:
            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}
            actor = TeacherEngineRayActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=placement["node_id"], soft=False),
                runtime_env={"env_vars": env_vars},
            ).remote(teacher_config, replica_idx)
            remote_pending.append((replica_idx, actor, actor.ready.remote()))

    def _cleanup_partial_teacher_startup() -> None:
        # parallel startup leaks partial replicas more easily, so roll everything back
        for svc in services:
            if svc is not None:
                try:
                    svc.shutdown()
                except Exception:
                    logger.exception("[OPD] teacher replica shutdown during startup-failure cleanup failed")
        # ray.kill is an unconditional backstop; safe after a clean shutdown because
        # it only removes the num_gpus=0 wrapper actor
        for _, actor, _ in remote_pending:
            try:
                ray.kill(actor)
            except Exception:
                logger.exception("[OPD] teacher actor kill during startup-failure cleanup failed")
        self._teacher_engine_services = []
        self._teacher_is_asleep = False

    try:
        # start local replicas serially while remote actors load on their own nodes
        for replica_idx, teacher_config in local_pending:
            svc = SGLangTeacherEngineService(teacher_config)
            svc.start()
            services[replica_idx] = svc

        # phase 2: await remote replicas concurrently; assign by replica_idx so routing
        # is independent of completion order
        ref_to_meta = {ready_ref: (idx, actor) for idx, actor, ready_ref in remote_pending}
        pending = list(ref_to_meta.keys())
        completed = 0
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=30)
            for ref in ready:
                ray.get(ref)  # propagate real startup exceptions
                idx, actor = ref_to_meta[ref]
                services[idx] = RayTeacherReplicaClient(actor)
                completed += 1
            if remote_pending:
                logger.info(
                    f"[OPD] teacher remote replica startup progress: "
                    f"completed={completed}/{len(remote_pending)}, pending={len(pending)}"
                )

        self._teacher_engine_services = services

        # phase 3: initial sleep is identical across replicas, so parallelize it. Inside
        # the try, so a partial sleep also rolls back via _cleanup_partial_teacher_startup.
        teacher_tags = teacher_offload_tags(self)
        if not teacher_sleep_enabled(self):
            logger.info("[OPD] Teacher replicas started; teacher sleep disabled, keeping them resident on GPU.")
        elif teacher_tags_release_weights(self, teacher_tags):
            import concurrent.futures

            def _initial_sleep(idx, svc):
                return idx, svc.sleep(tags=teacher_tags, return_response=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(services)) as executor:
                futs = [executor.submit(_initial_sleep, idx, svc) for idx, svc in enumerate(services)]
                for fut in concurrent.futures.as_completed(futs):
                    idx, response = fut.result()
                    logger.info(
                        f"[OPD] Teacher replica {idx} started and slept "
                        f"(elapsed={response.get('elapsed', -1):.2f}s "
                        f"empty_cache={response.get('empty_cache_time', -1):.2f}s "
                        f"release={response.get('release_time', -1):.2f}s "
                        f"tags={response.get('tags')!r})."
                    )
            self._teacher_is_asleep = True
        else:
            logger.info(
                f"[OPD] Teacher replicas started; skip initial sleep for non-weight "
                f"offload_tags={teacher_tags!r}."
            )
    except Exception:
        _cleanup_partial_teacher_startup()
        raise

    if self._teacher_engine_services:
        self._teacher_engine_service = self._teacher_engine_services[0]
    logger.info(
        f"[OPD] {len(self._teacher_engine_services)} teacher engine(s) ready "
        f"(dp={dp_size}, tp={tp_size}, ep={ep_size}, pp={pp_size})."
    )


def shutdown_teacher_engines(self) -> None:
    """Shut down all teacher replicas (best effort) and clear manager state."""
    for idx, svc in enumerate(getattr(self, "_teacher_engine_services", []) or []):
        try:
            logger.info(f"[OPD] Shutting down teacher engine replica {idx}")
            svc.shutdown()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[OPD] Error shutting down teacher engine replica {idx}: {e}")
    self._teacher_engine_services = []
    self._teacher_engine_service = None


def teacher_ray_nodes(self) -> tuple[list[dict], int]:
    nodes = [node for node in ray.nodes() if node.get("Alive", False)]
    nodes.sort(key=lambda n: (n.get("NodeManagerAddress", ""), n.get("NodeManagerHostname", "")))
    current_node_id = str(ray.get_runtime_context().get_node_id())
    current_ip = ray.util.get_node_ip_address()
    current_hostname = socket.gethostname()
    current_idx = next(
        (
            i
            for i, node in enumerate(nodes)
            if (
                str(node.get("NodeID")) == current_node_id
                or str(node.get("NodeManagerAddress")) == current_ip
                or str(node.get("NodeManagerHostname")) == current_hostname
            )
        ),
        None,
    )
    if current_idx is None:
        node_map = " ".join(
            f"{i}={node.get('NodeManagerHostname')}@{node.get('NodeManagerAddress')}"
            for i, node in enumerate(nodes)
        )
        raise RuntimeError(
            "[OPD] cannot match RolloutManager Ray node for teacher placement: "
            f"hostname={current_hostname} ip={current_ip} node_id={current_node_id}; "
            f"alive_nodes={node_map}"
        )
    logger.info(
        "[OPD] Ray node map for teacher placement: "
        + " ".join(
            f"{i}={node.get('NodeManagerHostname')}@{node.get('NodeManagerAddress')}"
            for i, node in enumerate(nodes)
        )
        + f" current={current_idx}({current_hostname}@{current_ip}, node_id={current_node_id})"
    )
    return nodes, current_idx


def resolve_teacher_node(self, token: str | None, nodes: list[dict], current_idx: int) -> int:
    if token is None or token == "":
        return current_idx
    token = token.strip()
    if token.isdigit():
        idx = int(token)
        if 0 <= idx < len(nodes):
            return idx
        raise ValueError(f"[OPD] teacher placement node index {idx} out of range 0..{len(nodes)-1}")
    for idx, node in enumerate(nodes):
        if token in {
            str(node.get("NodeID")),
            str(node.get("NodeManagerAddress")),
            str(node.get("NodeManagerHostname")),
        }:
            return idx
    raise ValueError(f"[OPD] cannot resolve teacher placement node {token!r}")


def build_teacher_placements(self, dp_size: int, gpus_per_replica: int, base_gpu_id: int) -> list[dict]:
    nodes, current_idx = teacher_ray_nodes(self)
    if not nodes:
        raise RuntimeError("[OPD] no alive Ray nodes for teacher placement")

    explicit = getattr(self.args, "opd_teacher_placement", None)
    placements: list[dict] = []
    if explicit:
        groups = [g.strip() for g in explicit.split(";") if g.strip()]
        if len(groups) != dp_size:
            raise ValueError(f"[OPD] --opd-teacher-placement has {len(groups)} groups, expected dp={dp_size}")
        for group in groups:
            node_token = None
            gpu_part = group
            if ":" in group:
                node_token, gpu_part = group.split(":", 1)
            node_idx = resolve_teacher_node(self, node_token, nodes, current_idx)
            gpu_ids = [int(x.strip()) for x in gpu_part.split(",") if x.strip()]
            if len(gpu_ids) != gpus_per_replica:
                raise ValueError(
                    f"[OPD] placement group {group!r} has {len(gpu_ids)} GPUs, expected {gpus_per_replica}"
                )
            if len(set(gpu_ids)) != len(gpu_ids):
                raise ValueError(f"[OPD] duplicate GPU ids in placement group {group!r}")
            placements.append({"node_idx": node_idx, "gpu_ids": gpu_ids})
    else:
        next_gpu = [base_gpu_id] * len(nodes)
        for replica_idx in range(dp_size):
            placed = False
            for offset in range(len(nodes)):
                node_idx = (replica_idx + offset) % len(nodes)
                start = next_gpu[node_idx]
                end = start + gpus_per_replica
                if end <= self.args.num_gpus_per_node:
                    placements.append({"node_idx": node_idx, "gpu_ids": list(range(start, end))})
                    next_gpu[node_idx] = end
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    "[OPD] cannot place teacher replicas across Ray nodes. "
                    "Set --opd-teacher-placement explicitly."
                )

    resolved = []
    used_slots: set[tuple[int, int]] = set()
    for replica_idx, placement in enumerate(placements):
        node = nodes[placement["node_idx"]]
        gpu_ids = placement["gpu_ids"]
        for gpu_id in gpu_ids:
            if gpu_id < 0 or gpu_id >= self.args.num_gpus_per_node:
                raise ValueError(
                    f"[OPD] teacher placement replica{replica_idx} uses GPU {gpu_id}, "
                    f"expected 0 <= id < num_gpus_per_node={self.args.num_gpus_per_node}"
                )
            slot = (placement["node_idx"], gpu_id)
            if slot in used_slots:
                raise ValueError(
                    f"[OPD] duplicate teacher GPU placement: node{placement['node_idx']} gpu{gpu_id}. "
                    "Teacher DP replicas must not overlap on the same node."
                )
            used_slots.add(slot)
        resolved.append(
            {
                "replica_idx": replica_idx,
                "node_idx": placement["node_idx"],
                "node_id": node.get("NodeID"),
                "hostname": node.get("NodeManagerHostname"),
                "address": node.get("NodeManagerAddress"),
                "gpu_ids": gpu_ids,
                "cuda_visible_devices": ",".join(str(i) for i in gpu_ids),
                "is_local": placement["node_idx"] == current_idx,
            }
        )
    logger.info(
        "[OPD] teacher placements: "
        + " ".join(
            f"replica{p['replica_idx']}=node{p['node_idx']}({p['address']}):gpu[{p['cuda_visible_devices']}]"
            for p in resolved
        )
    )
    return resolved


def teacher_offload_tags(self):
    return normalize_teacher_offload_tags("all")


def teacher_sleep_enabled(self) -> bool:
    return True


def teacher_tags_release_weights(self, tags=None) -> bool:
    """Whether a teacher sleep with these tags actually releases weights.

    resume_memory_occupation(tags=...) is not idempotent: resuming a tag that was never
    released can crash the scheduler, and kv_cache/cuda_graph are absent right after init.
    """
    if tags is None:
        tags = teacher_offload_tags(self)
    return teacher_offload_tags_release_weights(tags)


def ensure_teacher_awake(self) -> None:
    if not self._teacher_engine_services:
        return
    if not teacher_sleep_enabled(self):
        self._teacher_is_asleep = False
        logger.info("[OPD]   teacher sleep disabled; skip wakeup")
        return
    if not self._teacher_is_asleep:
        logger.info("[OPD]   teacher already awake; skip wakeup")
        return
    tags = teacher_offload_tags(self)
    # mark before the wakeup RPCs: a later failure must sleep all replicas rather
    # than trust stale state
    self._teacher_is_asleep = False
    import concurrent.futures

    def _wakeup(idx, svc):
        return idx, svc.wakeup(tags=tags, return_response=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._teacher_engine_services)) as executor:
        futs = [executor.submit(_wakeup, idx, svc) for idx, svc in enumerate(self._teacher_engine_services)]
        for fut in concurrent.futures.as_completed(futs):
            idx, response = fut.result()
            logger.info(
                f"[OPD]   replica {idx}/{len(self._teacher_engine_services)} wakeup done "
                f"(elapsed={response.get('elapsed', -1):.2f}s "
                f"empty_cache={response.get('empty_cache_time', -1):.2f}s "
                f"resume={response.get('resume_time', -1):.2f}s tags={response.get('tags')!r})"
            )
    self._teacher_is_asleep = False


def best_effort_teacher_sleep_after_error(self, reason: str) -> None:
    if (
        not self._teacher_engine_services
        or not teacher_sleep_enabled(self)
        or self._teacher_is_asleep
        or not teacher_tags_release_weights(self, teacher_offload_tags(self))
    ):
        return
    try:
        logger.warning(f"[OPD] teacher path failed; best-effort sleep before propagating error: {reason}")
        ensure_teacher_asleep(self)
    except Exception:
        logger.exception("[OPD] best-effort teacher sleep failed after teacher-path error")


def ensure_teacher_asleep(self) -> None:
    if not self._teacher_engine_services:
        return
    if not teacher_sleep_enabled(self):
        self._teacher_is_asleep = False
        logger.info("[OPD]   teacher sleep disabled; keep teacher resident")
        return
    if self._teacher_is_asleep:
        logger.info("[OPD]   teacher already asleep; skip sleep")
        return
    tags = teacher_offload_tags(self)
    if not teacher_tags_release_weights(self, tags):
        logger.info(
            f"[OPD]   skip teacher sleep for non-weight offload_tags={tags!r}; this avoids a "
            "SGLang error on partial kv_cache/cuda_graph resume. Use offload_tags='all' or "
            "include 'weights' if real teacher sleep is required."
        )
        return

    import concurrent.futures

    def _sleep(idx, svc):
        return idx, svc.sleep(tags=tags, return_response=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._teacher_engine_services)) as executor:
        futs = [executor.submit(_sleep, idx, svc) for idx, svc in enumerate(self._teacher_engine_services)]
        for fut in concurrent.futures.as_completed(futs):
            idx, response = fut.result()
            logger.info(
                f"[OPD]   replica {idx}/{len(self._teacher_engine_services)} sleep done "
                f"(elapsed={response.get('elapsed', -1):.2f}s "
                f"empty_cache={response.get('empty_cache_time', -1):.2f}s "
                f"release={response.get('release_time', -1):.2f}s tags={response.get('tags')!r})"
            )
    self._teacher_is_asleep = True
