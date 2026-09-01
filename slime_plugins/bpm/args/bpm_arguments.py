"""BPM command-line registration provider for upstream slime HEAD.

add_bpm_arguments is the add_custom_arguments callback parse_args expects.
get_bpm_args_provider bakes it into the full slime extra-args provider.
"""

from __future__ import annotations

import argparse

from .bpm_argument_groups import add_bpm_argument_groups
from .bpm_argument_validation import validate_bpm_args

__all__ = ["add_bpm_arguments", "get_bpm_args_provider", "validate_bpm_args"]


def add_bpm_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register all BPM-specific CLI flags and return the parser (add_custom_arguments contract)."""
    add_bpm_argument_groups(parser)
    return parser


def get_bpm_args_provider():
    """Return the full slime extra-args provider with the BPM flags registered up front."""
    from slime.utils.arguments import get_slime_extra_args_provider

    return get_slime_extra_args_provider(add_bpm_arguments)
