"""Subprocess request handlers for the BPM SGLang teacher engine: request dispatch and
shared-memory marshalling. The public service API stays in bpm_teacher_engine.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

import numpy as np
import torch
from torch.multiprocessing import Queue as MPQueue

logger = logging.getLogger(__name__)


def _normalize_tags(tags):
    """Convert teacher offload tags to SGLang's representation. The public config accepts
    `all`, but SGLang wants None; a literal ['all'] is a silent no-op.
    """
    if tags is None or tags == "all":
        return None
    if isinstance(tags, str):
        normalized = [tag.strip() for tag in tags.split(",") if tag.strip()]
    else:
        normalized = [str(tag).strip() for tag in tags if str(tag).strip()]
    if not normalized or any(tag.lower() == "all" for tag in normalized):
        return None
    return normalized


def _handle_generate(engine, request, hidden_queue, response_queue):
    """Run prefill-only inference and send hidden_states via shared memory.

    All hidden_states are processed before a success response, so a mid-way failure
    never leaves the caller blocked on missing tensors. return_token_ids needs
    return_logprob=True as a top-level generate() kwarg plus logprob_start_len=0.
    """
    kwargs = request["kwargs"]
    request_id = request.get("request_id")

    return_hidden_states = kwargs.get("return_hidden_states", True)
    return_token_ids = kwargs.get("return_token_ids", False)
    sampling_params = kwargs.get("sampling_params", {"max_new_tokens": 0})

    generate_kwargs: Dict[str, Any] = {
        "prompt": kwargs["prompt"],
        "sampling_params": sampling_params,
        "return_hidden_states": return_hidden_states,
    }
    if kwargs.get("input_ids") is not None:
        # pre-tokenized ids skip a second tokenization in the prefill path
        generate_kwargs["input_ids"] = kwargs["input_ids"]
    if kwargs.get("image_data") is not None:
        generate_kwargs["image_data"] = kwargs["image_data"]

    if return_token_ids:
        # return_logprob is an Engine.generate() kwarg, not a SamplingParams field
        generate_kwargs["return_logprob"] = True
        # logprob_start_len=0 covers all input tokens; the default -1 gives only the last
        generate_kwargs["logprob_start_len"] = 0

    try:
        outputs = engine.generate(**generate_kwargs)
    except Exception:
        import traceback

        response_queue.put(
            {"type": "generate", "request_id": request_id, "success": False, "error": traceback.format_exc()}
        )
        return

    num_samples = len(outputs)

    hidden_tensors = []
    token_ids_list = []

    try:
        for output, mask in zip(outputs, kwargs["loss_masks"]):
            meta_info = output.get("meta_info", {})
            hs_np = None
            if return_hidden_states:
                if "hidden_states" not in meta_info:
                    raise ValueError("Teacher engine output missing 'hidden_states' in meta_info")
                hs_list = meta_info["hidden_states"]
                if not hs_list or len(hs_list) == 0:
                    raise ValueError("Teacher engine returned empty hidden_states list")
                hs_np: np.ndarray = hs_list[0]

            # hidden_states rows and loss_mask rows must refer to the same sequence;
            # truncation would hide a template mismatch and train on the common prefix
            min_len = min(hs_np.shape[0], mask.shape[0]) if hs_np is not None else int(mask.shape[0])
            if hs_np is not None and hs_np.shape[0] != mask.shape[0]:
                msg = (
                    f"[OPD] _handle_generate: hidden_states shape {hs_np.shape[0]} != "
                    f"mask shape {mask.shape[0]}; this indicates a teacher prompt/response/EOS "
                    "alignment bug."
                )
                if not getattr(engine, "_opd_allow_prefix_truncation", False):
                    raise ValueError(
                        msg
                        + " Refusing to silently train on a prefix. Pass "
                        "--allow-opd-prefix-truncation only for legacy/debug replay."
                    )
                logging.getLogger(__name__).warning(msg + f" Legacy fallback enabled: truncating both to {min_len}.")
                hs_np = hs_np[:min_len]
                mask = mask[:min_len]

            # extract ids before masking so both use the same mask; dropping None
            # entries first would shift positions
            token_ids_masked = None
            if return_token_ids:
                token_logprobs = meta_info.get("input_token_logprobs", [])
                if not token_logprobs:
                    raise ValueError("return_logprob=True but input_token_logprobs not in meta_info")
                if len(token_logprobs) < min_len:
                    raise ValueError(
                        "input_token_logprobs shorter than hidden/loss-mask sequence: "
                        f"logprobs={len(token_logprobs)} min_len={min_len}"
                    )
                token_ids_full = []
                for pos, item in enumerate(token_logprobs[:min_len]):
                    # Format: [[logprob, token_id, decoded_token], ...].
                    if item is None or item[1] is None:
                        raise ValueError(
                            "input_token_logprobs has None teacher token_id at "
                            f"position {pos}; refusing to compact positions because alignment "
                            "requires one token id per selected hidden-state row."
                        )
                    token_ids_full.append(int(item[1]))
                token_ids_masked = [tid for tid, m in zip(token_ids_full, mask.tolist()) if m]
            masked_len = int(np.count_nonzero(mask[:min_len]))
            if hs_np is not None:
                hs_np = hs_np[mask]  # Only keep response positions (loss_mask == True)
                masked_len = int(hs_np.shape[0])

            if token_ids_masked is not None:
                if len(token_ids_masked) != masked_len:
                    raise ValueError(
                        "teacher token_ids/hidden_states length mismatch after mask: "
                        f"token_ids={len(token_ids_masked)} masked_rows={masked_len}"
                    )
                token_ids_list.append(token_ids_masked)

            hs_tensor = None
            if hs_np is not None:
                if not hs_np.flags["C_CONTIGUOUS"]:
                    hs_np = np.ascontiguousarray(hs_np)
                hs_tensor = torch.from_numpy(hs_np).share_memory_()

            hidden_tensors.append(hs_tensor)

    except Exception:
        import traceback

        response_queue.put(
            {"type": "generate", "request_id": request_id, "success": False, "error": traceback.format_exc()}
        )
        return

    # all hidden_states processed; send the response, then the tensors
    response_data = {
        "type": "generate",
        "request_id": request_id,
        "success": True,
        "num_samples": num_samples,
    }
    if return_token_ids:
        response_data["token_ids_list"] = token_ids_list

    response_queue.put(response_data)
    for sample_idx, hs_tensor in enumerate(hidden_tensors):
        hidden_queue.put({"request_id": request_id, "sample_idx": sample_idx, "tensor": hs_tensor})


def _handle_sleep(engine, request, config, response_queue):
    """Offload GPU memory for sharing with the student; report split timing."""
    tags = request.get("tags", config.offload_tags)
    t0 = time.time()
    # empty the cache before the memory-saver release: after it can add a sync
    torch.cuda.empty_cache()
    t_empty = time.time()
    engine.release_memory_occupation(tags=_normalize_tags(tags))
    t_release = time.time()
    response_queue.put(
        {
            "type": "sleep",
            "request_id": request.get("request_id"),
            "success": True,
            "tags": tags,
            "elapsed": t_release - t0,
            "empty_cache_time": t_empty - t0,
            "release_time": t_release - t_empty,
        }
    )


def _handle_wakeup(engine, request, config, response_queue):
    """Restore GPU memory for teacher inference; report split timing."""
    tags = request.get("tags", config.offload_tags)
    t0 = time.time()
    torch.cuda.empty_cache()
    t_empty = time.time()
    engine.resume_memory_occupation(tags=_normalize_tags(tags))
    t_resume = time.time()
    response_queue.put(
        {
            "type": "wakeup",
            "request_id": request.get("request_id"),
            "success": True,
            "tags": tags,
            "elapsed": t_resume - t0,
            "empty_cache_time": t_empty - t0,
            "resume_time": t_resume - t_empty,
        }
    )


def _handle_update_weights_from_tensor(engine, request, response_queue):
    """Update teacher weights from student tensors (for self-distillation)."""
    serialized_named_tensors = request["kwargs"]["serialized_named_tensors"]
    load_format = request["kwargs"]["load_format"]
    flush_cache = request["kwargs"]["flush_cache"]
    engine.update_weights_from_tensor(
        named_tensors=serialized_named_tensors,
        load_format=load_format,
        flush_cache=flush_cache,
    )
    response_queue.put(
        {"type": "update_weights_from_tensor", "request_id": request.get("request_id"), "success": True}
    )


def engine_worker(config, request_queue: MPQueue, response_queue: MPQueue, hidden_queue: MPQueue):
    """Worker process that runs PatchedEngine and handles requests."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if config.nnodes > 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    # Ray sets CUDA_VISIBLE_DEVICES='' for num_gpus=0 actors; the subprocess
    # inherits it and SGLang's get_device() would raise
    if config.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        logger.info(f"[TeacherEngine] Worker subprocess CUDA_VISIBLE_DEVICES set to: {config.cuda_visible_devices}")

    from .bpm_teacher_engine import PatchedEngine

    engine = None

    try:
        engine_kwargs = dict(
            model_path=config.model_path,
            tp_size=config.tp_size,
            ep_size=config.ep_size,
            pp_size=config.pp_size,
            chunked_prefill_size=config.chunked_prefill_size,
            disable_radix_cache=config.disable_radix_cache,
            enable_return_hidden_states=config.enable_return_hidden_states,
            enable_memory_saver=config.enable_memory_saver,
            enable_weights_cpu_backup=config.enable_weights_cpu_backup,
            quantization=config.quantization,
            mem_fraction_static=config.mem_fraction_static,
            base_gpu_id=config.base_gpu_id,
            nnodes=config.nnodes,
            node_rank=config.node_rank,
            dist_init_addr=config.dist_init_addr,
            # trust_remote_code must pass through: a custom architecture (partial-RoPE)
            # otherwise loads with wrong parameters and corrupts the tail hidden rows
            trust_remote_code=True,
            disable_custom_all_reduce=True,
            # overlap scheduling runs the forward on a separate stream whose copy_done
            # does not cover hidden_states, so reading it can race the forward tail
            disable_overlap_schedule=os.environ.get("OPD_TEACHER_DISABLE_OVERLAP", "0") == "1",
        )
        # some attention backends project the tail rows wrong while logits look fine
        _ab = os.environ.get("OPD_TEACHER_ATTENTION_BACKEND")
        if _ab:
            engine_kwargs["attention_backend"] = _ab
        if config.tokenizer_path is not None:
            # must match the tokenizer used to build the prompts and loss masks
            engine_kwargs["tokenizer_path"] = config.tokenizer_path
        try:
            engine: PatchedEngine = PatchedEngine(**engine_kwargs)
        except TypeError:
            if config.tokenizer_path is not None and config.tokenizer_path != config.model_path:
                raise
            # older SGLang builds do not accept tokenizer_path
            engine_kwargs.pop("tokenizer_path", None)
            engine = PatchedEngine(**engine_kwargs)

        # read by _handle_generate; keeps the default strict
        engine._opd_allow_prefix_truncation = bool(config.allow_prefix_truncation)

        response_queue.put({"type": "init_done", "success": True})

        while True:
            request = request_queue.get()
            if request is None:
                break

            req_type = request.get("type")

            try:
                if req_type == "generate":
                    _handle_generate(engine, request, hidden_queue, response_queue)
                elif req_type == "sleep":
                    _handle_sleep(engine, request, config, response_queue)
                elif req_type == "wakeup":
                    _handle_wakeup(engine, request, config, response_queue)
                elif req_type == "update_weights_from_tensor":
                    _handle_update_weights_from_tensor(engine, request, response_queue)
                else:
                    response_queue.put(
                        {
                            "type": req_type,
                            "request_id": request.get("request_id"),
                            "success": False,
                            "error": f"Unknown request type: {req_type}",
                        }
                    )
            except Exception:
                import traceback

                response_queue.put(
                    {
                        "type": req_type,
                        "request_id": request.get("request_id"),
                        "success": False,
                        "error": traceback.format_exc(),
                    }
                )

    except Exception:
        import traceback

        response_queue.put({"type": "init_done", "success": False, "error": traceback.format_exc()})
    finally:
        if engine:
            try:
                engine.shutdown()
            except Exception:
                pass
