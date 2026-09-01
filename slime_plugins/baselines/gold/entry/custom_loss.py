"""GOLD custom_loss adapter for upstream slime HEAD.

HEAD's callback is 4-arg func(args, batch, logits, sum_of_sample_mean); the GOLD backend
also takes a per-sample token-sum reducer, which this shim rebuilds from the batch.

Register with --loss-type custom_loss and --custom-loss-function-path pointing here.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import torch

from slime.utils.types import RolloutBatch
from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

from ..gold_megatron import gold_core_loss_function


def gold_custom_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """4-arg HEAD custom-loss entry that reconstructs ``sum_of_sample`` and calls the GOLD core."""
    sum_of_sample = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch.get("rollout_mask_sums"),
        True,
    )
    return gold_core_loss_function(args, batch, logits, sum_of_sample_mean, sum_of_sample)
