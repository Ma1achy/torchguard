"""Global configuration for the error-flag layout.

A single mutable ``ErrorConfig`` controls how many error slots each sample
carries and the integer storage dtype. There is no separate "experimental"
backend: flags are always integers (``int64`` by default, ``int32`` optional).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import torch

from .constants import SLOT_BITS

__all__ = [
    "ErrorConfig",
    "AccumulationPolicy",
    "Order",
    "Dedupe",
    "CONFIG",
    "get_config",
    "set_config",
]

_SUPPORTED_DTYPES = (torch.int64, torch.int32)


class Order(enum.Enum):
    """Eviction order when a sample's slots are full."""

    LIFO = "lifo"  # keep newest; insert at front, evict oldest
    FIFO = "fifo"  # keep oldest (root cause); append, drop new when full


class Dedupe(enum.Enum):
    """How a new error is collapsed against existing slots."""

    NONE = "none"  # every error gets its own slot
    CODE = "code"  # one slot per error code
    LOCATION = "location"  # one slot per location
    PAIR = "pair"  # one slot per (location, code)


@dataclass
class AccumulationPolicy:
    """How errors accumulate into the fixed set of per-sample slots.

    Args:
        order: Eviction order when slots are full (``LIFO`` default).
        dedupe: Collapse rule for matching errors (``NONE`` default).
        evict_by_severity: When full under ``FIFO``, replace the lowest-severity
            slot if the incoming error is more severe.
    """

    order: Order = Order.LIFO
    dedupe: Dedupe = Dedupe.NONE
    evict_by_severity: bool = False


@dataclass
class ErrorConfig:
    """Configuration for packed error flags.

    Args:
        num_slots: Number of error slots per sample (each slot is one error).
        flag_dtype: Integer storage dtype (``torch.int64`` or ``torch.int32``).
        accumulation: How errors accumulate into the slots.
    """

    num_slots: int = 16
    flag_dtype: torch.dtype = torch.int64
    accumulation: AccumulationPolicy = field(default_factory=AccumulationPolicy)

    def __post_init__(self) -> None:
        if self.num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {self.num_slots}")
        if self.flag_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"flag_dtype must be one of {_SUPPORTED_DTYPES}, got {self.flag_dtype}"
            )

    @property
    def slots_per_word(self) -> int:
        """Number of 16-bit slots packed into one storage word."""
        bits = torch.iinfo(self.flag_dtype).bits
        return bits // SLOT_BITS

    @property
    def num_words(self) -> int:
        """Number of storage words needed to hold ``num_slots`` slots."""
        spw = self.slots_per_word
        return (self.num_slots + spw - 1) // spw


CONFIG = ErrorConfig()


def get_config() -> ErrorConfig:
    """Return the active global configuration."""
    return CONFIG


def set_config(config: ErrorConfig) -> None:
    """Replace the active global configuration."""
    global CONFIG
    CONFIG = config
