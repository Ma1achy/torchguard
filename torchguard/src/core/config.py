"""
Configuration for compile-time error handling.

Three orthogonal axes for accumulation:
- Priority: What determines importance (CHRONO, SEVERITY, LOCATION)
- Order: Keep FIRST or LAST on priority axis
- Dedupe: How to group duplicates (NONE, CODE, LOCATION, UNIQUE)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .constants import SLOTS_PER_WORD
from .severity import Severity


class Priority(IntEnum):
    """
    What dimension determines error importance when slots are full.
    
    Attributes:
        CHRONO (int): Time of write (older vs newer errors)
        SEVERITY (int): Error importance level (OK < WARN < ERROR < CRITICAL)
        LOCATION (int): Where in the model the error occurred
    """
    CHRONO = 0
    SEVERITY = 1
    LOCATION = 2


class Order(IntEnum):
    """
    Which end of the priority axis to keep when slots are full.
    
    With Priority.CHRONO:
        FIRST = keep oldest errors (root cause preservation)
        LAST = keep newest errors (most recent state)
    
    With Priority.SEVERITY:
        FIRST = keep lowest severity (unusual)
        LAST = keep highest severity (typical - favor critical errors)
    
    Attributes:
        FIRST (int): Keep minimum on priority axis
        LAST (int): Keep maximum on priority axis
    """
    FIRST = 0
    LAST = 1


class Dedupe(IntEnum):
    """
    How to group/deduplicate errors in slots.
    
    Attributes:
        NONE (int): No deduplication - multiple entries per (location, code) allowed
        CODE (int): One entry per error code (collapse locations)
        LOCATION (int): One entry per location (collapse codes)
        UNIQUE (int): One entry per (location, code) pair
    """
    NONE = 0
    CODE = 1
    LOCATION = 2
    UNIQUE = 3


@dataclass(frozen=True)
class AccumulationConfig:
    """
    How to accumulate errors when pushing to flags.
    
    Three orthogonal axes:
        priority (Priority): What determines importance (time, severity, or location)
        order (Order): Keep FIRST or LAST on the priority axis
        dedupe (Dedupe): How to group duplicates
    
    Common configurations:
        LIFO (default): priority=CHRONO, order=LAST, dedupe=UNIQUE
        FIFO (root cause): priority=CHRONO, order=FIRST, dedupe=UNIQUE
        Severity: priority=SEVERITY, order=LAST, dedupe=UNIQUE
    """
    priority: Priority = Priority.CHRONO
    order: Order = Order.LAST
    dedupe: Dedupe = Dedupe.UNIQUE


@dataclass(frozen=True)
class ErrorConfig:
    """
    Configuration for error flag storage and behavior.
    
    Attributes:
        num_slots (int): Number of error slots (default 16, max 32768)
                         All operations are fully vectorized and torch.compile friendly.
        accumulation (AccumulationConfig): How to accumulate errors
        default_severity (int): Default severity for push operations (default ERROR)
        strict_validation (bool): If True, raise on error_t validation failure.
                                  If False (default), warn only.
    """
    num_slots: int = 16  # Vectorized ops - safe for torch.compile with any size
    accumulation: AccumulationConfig = field(default_factory=AccumulationConfig)
    default_severity: int = Severity.ERROR
    strict_validation: bool = False
    
    @property
    def num_words(self) -> int:
        """
        Number of int64 words needed (SLOTS_PER_WORD slots per word).
        
        Returns:
            (int): Number of int64 words for storage
        """
        return (self.num_slots + SLOTS_PER_WORD - 1) // SLOTS_PER_WORD
    
    def __post_init__(self) -> None:
        """Validate configuration on creation."""
        if not (1 <= self.num_slots <= 32768):
            raise ValueError(f"num_slots must be 1-32768, got {self.num_slots}")


# Default configuration: 16 slots (4 words), LIFO with (location, code) deduplication
# All operations are fully vectorized - safe for torch.compile with any num_slots.
DEFAULT_CONFIG: ErrorConfig = ErrorConfig()
