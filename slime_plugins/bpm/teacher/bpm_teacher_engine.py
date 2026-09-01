"""SGLang teacher engine subprocess used by BPM.

The RolloutManager sends prefill-only requests over multiprocessing queues and gets
response-token hidden states back as shared-memory tensors. BPM-only.
"""

import logging
import os
import queue as _queue_mod  # stdlib, avoid clash with mp.Queue
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue as MPQueue


def _patch_hf_hub_bucket_not_found_error() -> None:
    """Backfill an old huggingface_hub error class some bundled SGLang builds import.
    The teacher only uses local paths, so the alias just lets SGLang finish importing.
    """
    try:
        import huggingface_hub.errors as hf_errors
    except Exception:
        return
    if hasattr(hf_errors, "BucketNotFoundError"):
        return

    base_error = getattr(hf_errors, "HfHubHTTPError", Exception)

    class BucketNotFoundError(base_error):
        pass

    hf_errors.BucketNotFoundError = BucketNotFoundError


_patch_hf_hub_bucket_not_found_error()

from sglang.srt.entrypoints.engine import Engine as _SglEngine
from sglang.srt.managers.scheduler import run_scheduler_process as _original_run_scheduler_process

logger = logging.getLogger(__name__)

from .bpm_teacher_handlers import engine_worker as _engine_worker

os.environ.setdefault("SGLANG_JIT_DEEPGEMM_FAST_WARMUP", "true")


def _patched_run_scheduler_process(*args, **kwargs):
    """Scheduler process entry that applies the hidden_states patch in the subprocess."""
    from .bpm_sglang_patch import apply_patch

    if not apply_patch():
        raise RuntimeError(
            f"[PatchedEngine] Failed to apply required hidden_states patch in scheduler "
            f"subprocess (PID={os.getpid()}). Refusing to continue because the teacher "
            "hidden-state layout would be unverified."
        )
    return _original_run_scheduler_process(*args, **kwargs)


class PatchedEngine(_SglEngine):
    """SGLang Engine that applies the hidden_states patch in scheduler subprocesses.
    The stock path converts them with .tolist(); the patch uses .numpy() instead.
    """

    run_scheduler_process_func = staticmethod(_patched_run_scheduler_process)


@dataclass
class TeacherEngineConfig:
    """Configuration for the BPM teacher SGLang engine."""

    model_path: str
    tokenizer_path: Optional[str] = None
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    chunked_prefill_size: int = -1
    disable_radix_cache: bool = True
    enable_return_hidden_states: bool = True
    enable_memory_saver: bool = True
    enable_weights_cpu_backup: bool = True
    mem_fraction_static: float = 0.8
    quantization: Optional[str] = None
    offload_tags: Optional[str] = "all"
    base_gpu_id: int = 0
    # for multi-node tp/pp
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: Optional[str] = None
    # Ray sets CUDA_VISIBLE_DEVICES to '' for num_gpus=0 actors
    cuda_visible_devices: Optional[str] = None
    # opt-in for replaying legacy debug dumps only; fresh rollouts must align
    allow_prefix_truncation: bool = False


class SGLangTeacherEngineService:
    """Teacher SGLang Engine in a subprocess, over three multiprocessing queues:
    request_queue (requests), response_queue (status), hidden_queue (shared-memory
    tensors). Spawned without force=True to avoid Ray start-method conflicts.
    """

    def __init__(self, config: TeacherEngineConfig):
        self.config = config
        self.process: Optional[mp.Process] = None
        self.request_queue: Optional[MPQueue] = None
        self.response_queue: Optional[MPQueue] = None
        self.hidden_queue: Optional[MPQueue] = None
        self._started = False
        self._next_request_id = 0

    def _new_request_id(self, req_type: str) -> str:
        self._next_request_id += 1
        return f"{req_type}-{self._next_request_id}"

    def start(self, timeout: float | None = None):
        """Start the teacher SGLang Engine in a subprocess.

        `timeout` waits for the child's init_done handshake (env OPD_TEACHER_INIT_TIMEOUT,
        else 5400s): large FP8 MoE teachers need a long one-time kernel warmup.
        """
        if timeout is None:
            timeout = float(os.environ.get("OPD_TEACHER_INIT_TIMEOUT", "5400"))
        if self._started:
            raise RuntimeError("Service already started")

        # spawn, not fork: the parent has CUDA initialized. RolloutManager locks
        # the global start_method to fork, so use an explicit spawn context.
        _spawn_ctx = mp.get_context("spawn")

        self.request_queue = _spawn_ctx.Queue()
        self.response_queue = _spawn_ctx.Queue()
        self.hidden_queue = _spawn_ctx.Queue()

        self.process = _spawn_ctx.Process(
            target=_engine_worker,
            args=(self.config, self.request_queue, self.response_queue, self.hidden_queue),
        )

        # the spawn child caches CUDA_VISIBLE_DEVICES at `import torch`, so set it
        # in the parent before start() and restore right after
        _saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self.config.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
            logger.info(
                f"[TeacherEngine] Parent CUDA_VISIBLE_DEVICES temporarily set to "
                f"{self.config.cuda_visible_devices!r} (was: {_saved_cvd!r}) before spawn"
            )
        self.process.start()
        if self.config.cuda_visible_devices is not None:
            if _saved_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cvd
            logger.info(
                f"[TeacherEngine] Parent CUDA_VISIBLE_DEVICES restored to "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')!r}"
            )

        try:
            response = self.response_queue.get(timeout=timeout)
            if response.get("type") == "init_done" and response.get("success"):
                self._started = True
                logger.info(f"[TeacherEngine] Started successfully (PID={self.process.pid})")
            else:
                raise RuntimeError(f"Init failed: {response.get('error')}")
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Engine initialization failed: {e}")

    def generate(
        self,
        prompt: List[str],
        loss_masks: List[np.ndarray],
        input_ids: Optional[List[List[int]]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        return_hidden_states: bool = True,
        return_token_ids: bool = False,
        image_data=None,
        response_timeout: Optional[float] = 600.0,
        hidden_recv_timeout: Optional[float] = 300.0,
        check_interval: float = 10.0,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[List[int]]]]:
        """Run prefill-only generation and return (hidden_states, token_ids).

        `input_ids` is optional; SGLang tokenizes `prompt` when it is absent. `loss_masks`
        selects the response hidden states. response_timeout / hidden_recv_timeout of None
        or <=0 disable the deadline but keep the liveness checks. Entries are None when
        not requested.
        """
        if not self._started:
            raise RuntimeError("Service not started")

        if self.process is not None and not self.process.is_alive():
            raise RuntimeError(
                f"[TeacherEngine] Engine subprocess (PID={self.process.pid}) is dead! "
                f"exitcode={self.process.exitcode}"
            )

        if sampling_params is None:
            sampling_params = {"max_new_tokens": 0}  # prefill-only by default

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "input_ids": input_ids,
            "loss_masks": loss_masks,
            "sampling_params": sampling_params,
            "return_hidden_states": return_hidden_states,
            "return_token_ids": return_token_ids,
        }
        if image_data is not None:
            kwargs["image_data"] = image_data

        request_id = self._new_request_id("generate")
        self.request_queue.put({"type": "generate", "request_id": request_id, "kwargs": kwargs})

        response = self._get_response(
            req_type="generate",
            request_id=request_id,
            timeout=response_timeout,
            check_interval=check_interval,
        )
        if not response.get("success"):
            raise RuntimeError(f"Generate failed: {response.get('error')}")

        num_samples = response["num_samples"]
        hidden_states_by_index: Optional[List[Optional[np.ndarray]]] = (
            [None] * num_samples if return_hidden_states else None
        )
        received_by_index = [False] * num_samples

        hidden_deadline_enabled = hidden_recv_timeout is not None and hidden_recv_timeout > 0
        hidden_timeout = float(hidden_recv_timeout) if hidden_deadline_enabled else None
        check_interval = max(float(check_interval), 0.1)

        # hidden_recv_timeout=None/0 drops the deadline, keeps liveness checks
        received = 0
        while received < num_samples:
            start_t = time.monotonic()
            while True:
                wait_timeout = check_interval
                if hidden_timeout is not None:
                    remaining = hidden_timeout - (time.monotonic() - start_t)
                    if remaining <= 0:
                        raise RuntimeError(
                            f"Hidden state recv timeout after {hidden_timeout}s, "
                            f"request_id={request_id}, received={received}/{num_samples}"
                        )
                    wait_timeout = min(wait_timeout, remaining)
                try:
                    hidden_item = self.hidden_queue.get(timeout=wait_timeout)
                    break
                except _queue_mod.Empty:
                    if self.process is not None and not self.process.is_alive():
                        raise RuntimeError(
                            f"[TeacherEngine] Subprocess died during hidden_states recv! "
                            f"exitcode={self.process.exitcode}"
                        )
            if isinstance(hidden_item, dict):
                item_request_id = hidden_item.get("request_id")
                sample_idx = hidden_item.get("sample_idx")
                hs_tensor = hidden_item.get("tensor")
            else:
                # untagged tensors count as current; only smoke tests patch the queue
                item_request_id = request_id
                sample_idx = received
                hs_tensor = hidden_item

            if item_request_id != request_id:
                logger.warning(
                    "[TeacherEngine] Dropping stale hidden_states payload: "
                    f"got request_id={item_request_id!r}, expected={request_id!r}"
                )
                continue
            if not isinstance(sample_idx, int) or sample_idx < 0 or sample_idx >= num_samples:
                raise RuntimeError(
                    f"[TeacherEngine] Invalid hidden_states sample_idx={sample_idx!r} "
                    f"for request_id={request_id}, num_samples={num_samples}"
                )
            if received_by_index[sample_idx]:
                raise RuntimeError(
                    f"[TeacherEngine] Duplicate payload sample_idx={sample_idx} for request_id={request_id}"
                )
            received_by_index[sample_idx] = True
            if hidden_states_by_index is not None:
                if hs_tensor is None:
                    raise RuntimeError(
                        f"[TeacherEngine] Missing hidden_states tensor for sample_idx={sample_idx} "
                        f"request_id={request_id}"
                    )
                # detach from the shared segment before Ray copies it: holding both can
                # SIGBUS when /dev/shm is tight
                hidden_states_by_index[sample_idx] = np.array(hs_tensor.numpy(), copy=True)
            received += 1

        token_ids_list: Optional[List[List[int]]] = None
        if return_token_ids and "token_ids_list" in response:
            token_ids_list = response["token_ids_list"]

        if hidden_states_by_index is not None:
            missing = [i for i, value in enumerate(hidden_states_by_index) if value is None]
            if missing:
                raise RuntimeError(
                    f"[TeacherEngine] Missing hidden_states for request_id={request_id}: "
                    f"indices={missing[:16]}"
                )
        return hidden_states_by_index, token_ids_list

    def sleep(self, tags: Optional[str] = "all", return_response: bool = False):
        """Release GPU memory so student rollout can use it."""
        if not self._started:
            return None
        request_id = self._new_request_id("sleep")
        self.request_queue.put({"type": "sleep", "request_id": request_id, "tags": tags})
        response = self._get_response(req_type="sleep", request_id=request_id, timeout=300)
        if not response.get("success"):
            raise RuntimeError(f"Sleep failed: {response.get('error')}")
        if return_response:
            return response
        return response.get("tags")

    def wakeup(self, tags: Optional[str] = "all", return_response: bool = False):
        """Restore GPU memory for teacher inference."""
        if not self._started:
            return None
        request_id = self._new_request_id("wakeup")
        self.request_queue.put({"type": "wakeup", "request_id": request_id, "tags": tags})
        response = self._get_response(req_type="wakeup", request_id=request_id, timeout=300)
        if not response.get("success"):
            raise RuntimeError(f"Wakeup failed: {response.get('error')}")
        if return_response:
            return response
        return response.get("tags")

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update teacher weights (for self-distillation)."""
        if not self._started:
            raise RuntimeError("Service not started")
        kwargs = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        request_id = self._new_request_id("update_weights_from_tensor")
        self.request_queue.put(
            {"type": "update_weights_from_tensor", "request_id": request_id, "kwargs": kwargs}
        )
        response = self._get_response(
            req_type="update_weights_from_tensor", request_id=request_id, timeout=300
        )
        if not response.get("success"):
            raise RuntimeError(f"Update weights failed: {response.get('error')}")

    def _get_response(
        self,
        req_type: str = "unknown",
        request_id: Optional[str] = None,
        timeout: Optional[float] = 600,
        check_interval: float = 10,
    ):
        """Wait for the matching response, with liveness checks. Responses are tagged with
        request_id so a timed-out generate cannot poison a later call.
        """
        deadline_enabled = timeout is not None and timeout > 0
        deadline = time.monotonic() + float(timeout) if deadline_enabled else None
        check_interval = max(float(check_interval), 0.1)
        while True:
            wait_timeout = check_interval
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait_timeout = min(wait_timeout, remaining)
            try:
                response = self.response_queue.get(timeout=wait_timeout)
            except _queue_mod.Empty:
                if self.process is not None and not self.process.is_alive():
                    raise RuntimeError(
                        f"[TeacherEngine] Subprocess (PID={self.process.pid}) died during '{req_type}'! "
                        f"exitcode={self.process.exitcode}"
                    )
                continue
            if response.get("type") != req_type or response.get("request_id") != request_id:
                logger.warning(
                    "[TeacherEngine] Dropping stale/mismatched response: "
                    f"got type={response.get('type')!r} request_id={response.get('request_id')!r}, "
                    f"expected type={req_type!r} request_id={request_id!r}"
                )
                continue
            return response
        raise RuntimeError(f"Response timeout after {timeout}s during '{req_type}'")

    def shutdown(self):
        """Shut down the subprocess gracefully."""
        if not self._started:
            return
        self._started = False
        self._cleanup()

    def _cleanup(self):
        """Clean up subprocess, queues, and shared memory. hidden_queue is drained so
        share_memory_() tensors do not leak on an unexpected exit.
        """
        if self.request_queue is not None:
            try:
                self.request_queue.put(None)
            except Exception:
                pass

        if self.process is not None:
            self.process.join(timeout=30)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()

        if self.hidden_queue is not None:
            while not self.hidden_queue.empty():
                try:
                    self.hidden_queue.get_nowait()
                except Exception:
                    break

        self.hidden_queue = None
        self.process = None
        self.request_queue = None
        self.response_queue = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
