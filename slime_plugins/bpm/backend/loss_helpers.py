"""Teacher lm_head loading and CP-aligned response slicing for the BPM backend."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from megatron.core import mpu

from slime.backends.megatron_utils.cp_utils import get_logits_and_tokens_offset_with_cp


def load_teacher_lm_head(
    model_path: str,
    device: str | torch.device | int = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    padded_vocab_size: int | None = None,
) -> torch.nn.Linear:
    """Load only lm_head weights from a teacher HF checkpoint as a frozen nn.Linear.

    Each TP rank loads its own vocab shard, using the same padded boundary as the
    student so rank>0 gathers the same global ids. Megatron padding rows are appended
    and masked downstream. Returns nn.Linear(hidden_size, vocab_shard_size), frozen.
    """
    import json
    import os
    from transformers import AutoConfig
    from safetensors import safe_open
    from megatron.core import mpu

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    is_local = os.path.exists(model_path)

    if hasattr(config, "text_config"):
        hidden_size = config.text_config.hidden_size
        vocab_size = config.text_config.vocab_size
    else:
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    if padded_vocab_size is None:
        padded_global_vocab_size = (
            ((vocab_size + tp_size - 1) // tp_size) * tp_size
            if tp_size > 1
            else vocab_size
        )
    else:
        padded_global_vocab_size = int(padded_vocab_size)
    if padded_global_vocab_size < vocab_size:
        raise ValueError(
            f"padded_vocab_size={padded_global_vocab_size} is smaller than teacher vocab_size={vocab_size}"
        )
    if tp_size > 1 and padded_global_vocab_size % tp_size != 0:
        raise ValueError(
            f"padded_vocab_size={padded_global_vocab_size} must be divisible by TP size {tp_size} "
            "for OPD teacher lm_head sharding."
        )
    shard_vocab_size = padded_global_vocab_size // tp_size
    vocab_start = tp_rank * shard_vocab_size
    vocab_end = vocab_start + shard_vocab_size

    def resolve_file(filename):
        if is_local:
            return os.path.join(model_path, filename)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=model_path, filename=filename)

    weight_map = None
    use_safetensors = True
    for index_name, is_safe in [
        ("model.safetensors.index.json", True),
        ("pytorch_model.bin.index.json", False),
    ]:
        try:
            with open(resolve_file(index_name)) as f:
                weight_map = json.load(f)["weight_map"]
                use_safetensors = is_safe
                break
        except Exception:
            continue

    EMBED_KEYS = ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight")

    if weight_map and "lm_head.weight" in weight_map:
        target_key = "lm_head.weight"
    elif getattr(config, "tie_word_embeddings", True):
        for cand in EMBED_KEYS:
            if weight_map is None or cand in weight_map:
                target_key = cand
                break
        else:
            raise ValueError(f"Could not find lm_head.weight or any of {EMBED_KEYS} in checkpoint.")
    else:
        raise ValueError("Could not find lm_head.weight in checkpoint and tie_word_embeddings is False.")

    if weight_map:
        checkpoint_file = resolve_file(weight_map[target_key])
    else:
        for name, is_safe in [("model.safetensors", True), ("pytorch_model.bin", False)]:
            try:
                checkpoint_file = resolve_file(name)
                use_safetensors = is_safe
                break
            except Exception:
                continue
        else:
            raise FileNotFoundError(f"No checkpoint file found in {model_path}")

    # read only the resolved target_key: a hardcoded priority could pick wrong
    if use_safetensors:
        with safe_open(checkpoint_file, framework="pt", device="cpu") as f:
            present = set(f.keys())
            state_dict = {}
            if target_key in present:
                state_dict[target_key] = f.get_tensor(target_key)
            if "lm_head.bias" in present:
                state_dict["lm_head.bias"] = f.get_tensor("lm_head.bias")
    else:
        state_dict = torch.load(checkpoint_file, map_location="cpu")

    if "lm_head.weight" in state_dict:
        weight_key = "lm_head.weight"
    else:
        weight_key = next((k for k in EMBED_KEYS if k in state_dict), None)
    if weight_key is None:
        raise ValueError(f"None of 'lm_head.weight' or {EMBED_KEYS} found. Available: {list(state_dict.keys())[:10]}...")
    weight = state_dict[weight_key]

    if int(weight.shape[1]) != int(hidden_size) or int(weight.shape[0]) < int(vocab_size):
        raise ValueError(
            f"Teacher lm_head shape mismatch: expected at least ({vocab_size}, {hidden_size}), "
            f"got {tuple(weight.shape)}"
        )
    if int(weight.shape[0]) > int(vocab_size):
        weight = weight[:vocab_size]

    real_vocab_end = min(vocab_end, vocab_size)
    if vocab_start < vocab_size:
        weight = weight[vocab_start:real_vocab_end, :].contiguous()
    else:
        weight = weight[:0, :].contiguous()
    pad_rows = shard_vocab_size - weight.shape[0]
    if pad_rows > 0:
        weight = F.pad(weight, (0, 0, 0, pad_rows), value=0.0).contiguous()

    has_bias = "lm_head.bias" in state_dict
    lm_head = torch.nn.Linear(hidden_size, shard_vocab_size, bias=has_bias)
    lm_head.weight = torch.nn.Parameter(weight.to(dtype=dtype))
    if has_bias:
        bias = state_dict["lm_head.bias"]
        if vocab_start < vocab_size:
            bias = bias[vocab_start:real_vocab_end].contiguous()
        else:
            bias = bias[:0].contiguous()
        pad_rows = shard_vocab_size - bias.shape[0]
        if pad_rows > 0:
            bias = F.pad(bias, (0, pad_rows), value=0.0).contiguous()
        lm_head.bias = torch.nn.Parameter(bias.to(dtype=dtype))

    lm_head = lm_head.to(device=device).eval()
    lm_head.requires_grad_(False)
    # consumed by the BPM loss for full-vocab / padded-row masking
    lm_head._opd_vocab_size = vocab_size
    lm_head._opd_padded_vocab_size = padded_global_vocab_size
    lm_head._opd_shard_vocab_size = shard_vocab_size
    lm_head._opd_vocab_start = vocab_start
    lm_head._opd_vocab_end = vocab_end
    return lm_head


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned (logits_chunk, tokens_chunk) pairs per sample.

    logits_chunk is [R, V] with next-token pairing (row t predicts token t+1). Under
    context parallelism each rank owns two sliced chunks. logits must be float32.
    """
    if logits.size(0) != 1:
        raise ValueError(f"expected batch dim 1, got logits shape {tuple(logits.shape)}")
    if logits.dtype != torch.float32:
        raise ValueError(f"expected float32 logits, got {logits.dtype}")

    logits = logits.squeeze(0)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    for tokens, total_length, response_length in zip(unconcat_tokens, total_lengths, response_lengths, strict=False):
        if cp_size == 1:
            end += total_length
            start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        else:
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            if logits_0.size(0) != tokens_0.size(0):
                raise RuntimeError(f"CP chunk 0 logits/tokens length mismatch: {logits_0.size(0)} vs {tokens_0.size(0)}")
            if logits_1.size(0) != tokens_1.size(0):
                raise RuntimeError(f"CP chunk 1 logits/tokens length mismatch: {logits_1.size(0)} vs {tokens_1.size(0)}")

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        yield logits_chunk, tokens_chunk
