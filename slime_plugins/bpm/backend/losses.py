"""OPD backend entry point for the BPM release.

BPM is the only alignment backend, so the loss entry is the method core directly; no
per-backend dispatch layer.
"""

from __future__ import annotations

from ..loss.bpm_megatron import bpm_core_loss_function as bpm_opd_loss_function

__all__ = ["bpm_opd_loss_function"]
