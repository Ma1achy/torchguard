"""
Configuration for compile-time error handling.

Three orthogonal axes for accumulation:
- Priority: What determines importance (CHRONO, SEVERITY, LOCATION)
- Order: Keep FIRST or LAST on priority axis
- Dedupe: How to group duplicates (NONE, CODE, LOCATION, UNIQUE)

Global Config:
    CONFIG - Mutable global configuration object
    get_config() - Get the global config

Usage:
    import torchguard as tg
    tg.CONFIG.flag_dtype = torch.float32
    tg.CONFIG.num_slots = 32
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import torch

from .constants import SLOTS_PER_WORD
from .severity import Severity


# Mapping from carrier dtype to int dtype for bitwise operations
# Float dtypes are used as carriers and viewed as int for bitwise ops
# Int dtypes are used directly (for stable backend compatibility)
_BITWISE_DTYPE_MAP = {
    torch.float32: torch.int32,
    torch.float64: torch.int64,
    torch.int32: torch.int32,
    torch.int64: torch.int64,
}

# Slots per word based on dtype (16-bit slots)
_SLOTS_PER_DTYPE = {
    torch.float32: 2,   # 32 bits / 16 bits = 2 slots
    torch.float64: 4,   # 64 bits / 16 bits = 4 slots
    torch.int32: 2,     # 32 bits / 16 bits = 2 slots
    torch.int64: 4,     # 64 bits / 16 bits = 4 slots
}

# Supported dtypes for error flags
_SUPPORTED_DTYPES = set(_BITWISE_DTYPE_MAP.keys())


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
        FIFO (default): priority=CHRONO, order=FIRST, dedupe=UNIQUE - preserves root causes
        LIFO: priority=CHRONO, order=LAST, dedupe=UNIQUE - tracks most recent errors
        Severity: priority=SEVERITY, order=LAST, dedupe=UNIQUE - prioritizes critical errors
    
    Default uses FIFO to preserve root causes when debugging NaN/Inf propagation.
    """
    priority: Priority = Priority.CHRONO
    order: Order = Order.FIRST  # FIFO: keep oldest errors (root cause preservation)
    dedupe: Dedupe = Dedupe.UNIQUE


@dataclass
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
        flag_dtype (torch.dtype): Dtype for flag tensors.
                                  Supports torch.float32, torch.float64, torch.int32, torch.int64.
                                  float64/int64 use 4 slots/word, float32/int32 use 2 slots/word.
    
    Example:
        >>> import torch
        >>> import torchguard as tg
        >>> # Use torch.dtype directly
        >>> tg.CONFIG.flag_dtype = torch.float32
        >>> # Or use dtype alias from typing
        >>> from torchguard.typing import float32_t
        >>> tg.CONFIG.flag_dtype = float32_t._dtype
    """
    num_slots: int = 16  # Vectorized ops - safe for torch.compile with any size
    accumulation: AccumulationConfig = field(default_factory=AccumulationConfig)
    default_severity: int = Severity.ERROR
    strict_validation: bool = False
    flag_dtype: torch.dtype = torch.int64  # Default to int64 (stable backend) for compatibility
    # Memory layout optimization for large batches
    use_transposed_layout: bool = False  # If True, use (num_words, N) instead of (N, num_words)
    transpose_threshold: int = 10000  # Auto-transpose for N > this when use_transposed_layout=True
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """
        Get the torch dtype for flag tensors (same as flag_dtype).
        
        This is an alias for backward compatibility with code that uses config.torch_dtype.
        """
        return self.flag_dtype
    
    @property
    def torch_int_dtype(self) -> torch.dtype:
        """
        Get the corresponding integer dtype for bitwise operations.
        
        For float carriers (float32/float64), returns the int type for view operations.
        For int dtypes (int32/int64), returns the same dtype.
        """
        return _BITWISE_DTYPE_MAP.get(self.flag_dtype, torch.int64)
    
    @property
    def slots_per_word(self) -> int:
        """Number of 16-bit slots per word based on dtype."""
        return _SLOTS_PER_DTYPE.get(self.flag_dtype, 4)
    
    @property
    def num_words(self) -> int:
        """
        Number of words needed based on dtype.
        
        float32/int32: 2 slots per word (32 bits)
        float64/int64: 4 slots per word (64 bits)
        
        Returns:
            (int): Number of words for storage
        """
        slots_per = self.slots_per_word
        return (self.num_slots + slots_per - 1) // slots_per
    
    @property
    def word_bits(self) -> int:
        """Number of bits per word based on dtype."""
        if self.flag_dtype in (torch.float32, torch.int32):
            return 32
        return 64
    
    def should_transpose(self, batch_size: int) -> bool:
        """
        Determine if transposed layout should be used for this batch size.
        
        Transposed layout (num_words, N) can be more cache-friendly for
        very large batches where N >> num_words.
        
        Args:
            batch_size (int): Number of samples in the batch
        
        Returns:
            (bool): True if transposed layout should be used
        """
        return self.use_transposed_layout and batch_size > self.transpose_threshold
    
    def __post_init__(self) -> None:
        """Validate configuration on creation."""
        if not (1 <= self.num_slots <= 32768):
            raise ValueError(f"num_slots must be 1-32768, got {self.num_slots}")
        if self.flag_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"flag_dtype must be one of {_SUPPORTED_DTYPES}, "
                f"got {self.flag_dtype}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Global config for torchguard (mutable, user-configurable)
# All operations use this config unless an explicit config is passed.
#
# Default to torch.int64 (stable backend) for maximum compatibility.
# Users can set to torch.float32/torch.float64 for experimental backend.
#
# Usage:
#     import torchguard as tg
#     tg.CONFIG.flag_dtype = torch.float32  # Use float32 for experimental backend
#     tg.CONFIG.num_slots = 32              # Increase slot count
CONFIG: ErrorConfig = ErrorConfig(flag_dtype=torch.int64)


def get_config() -> ErrorConfig:
    """
    Get the global torchguard configuration.
    
    Returns:
        (ErrorConfig): The global config object
    
    Example:
        >>> from torchguard import get_config
        >>> config = get_config()
        >>> config.flag_dtype = torch.float32
    """
    return CONFIG


def set_config(config: ErrorConfig) -> None:
    """
    Replace the global torchguard configuration.
    
    This replaces the entire global CONFIG object. For modifying
    individual attributes, prefer modifying CONFIG directly:
    
        import torchguard as tg
        tg.CONFIG.flag_dtype = torch.float32
    
    This function is useful for temporarily replacing the config:
    
        original = get_config()
        set_config(ErrorConfig(flag_dtype=torch.float32, num_slots=32))
        try:
            # ... use float32 config ...
        finally:
            set_config(original)
    
    Args:
        config (ErrorConfig): The new config to use globally
    
    Example:
        >>> from torchguard import set_config, ErrorConfig
        >>> import torch
        >>> set_config(ErrorConfig(flag_dtype=torch.float32))
    """
    global CONFIG
    CONFIG = config
