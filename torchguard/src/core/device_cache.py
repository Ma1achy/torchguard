"""Device-aware tensor cache for performance optimization.

This module provides caching for frequently-used tensors like:
- Shift tensors for bit operations (torch.arange patterns)
- Constant scalars (-inf, 0, 1, etc.)
- Slot index tensors

These tensors are created once per (device, dtype, size) combination
and reused, avoiding repeated allocations in hot paths.

Thread-safety: Lock-free design for torch.compile compatibility.
The cache uses simple dict operations which are thread-safe in CPython
for reads and single-item assignments. Duplicate creation during warm-up
is harmless since values are identical.

Memory overhead: ~1-2KB per device (negligible).

Usage:
    from torchguard.src.core.device_cache import get_device_cache
    
    cache = get_device_cache()
    shifts = cache.get_slot_shifts(device, dtype, slots_per_word, slot_bits)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


class DeviceCache:
    """
    Thread-safe cache for frequently used tensors per device.
    
    Caches:
    - Shift tensors for bit operations
    - Constant scalars (neg_inf, zero, etc.)
    - Slot indices for position tracking
    - Position arrays for scoring
    
    Memory overhead: ~1-2KB per device (negligible)
    
    Example:
        >>> cache = get_device_cache()
        >>> shifts = cache.get_slot_shifts(torch.device('cuda'), torch.int64, 4, 16)
        >>> # Same tensor returned on subsequent calls
        >>> shifts2 = cache.get_slot_shifts(torch.device('cuda'), torch.int64, 4, 16)
        >>> assert shifts is shifts2
    """
    
    def __init__(self):
        self._cache: Dict[Tuple, Tensor] = {}
    
    def get_slot_shifts(
        self,
        device: torch.device,
        dtype: torch.dtype,
        slots_per_word: int,
        slot_bits: int
    ) -> Tensor:
        """
        Get cached shift tensor for slot operations.
        
        This replaces repeated `torch.arange(slots_per_word) * slot_bits` calls.
        
        Args:
            device: Target device
            dtype: Target dtype (typically int64 or int32)
            slots_per_word: Number of slots per word
            slot_bits: Bits per slot
        
        Returns:
            Tensor of shape (slots_per_word,) with values [0, slot_bits, 2*slot_bits, ...]
        
        Example:
            >>> shifts = cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
            >>> shifts
            tensor([ 0, 16, 32, 48])
        """
        # Normalize device to avoid cuda:0 vs cuda mismatch
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device() if torch.cuda.is_available() else 0)
        
        key = ('slot_shifts', device.type, device.index, dtype, slots_per_word, slot_bits)
        
        # During torch.compile tracing, skip cache to avoid guard issues
        if torch.compiler.is_compiling():
            return torch.arange(slots_per_word, device=device, dtype=dtype) * slot_bits
        
        if key not in self._cache:
            # Create tensor if not cached (CPython dict assignment is atomic)
            shifts = torch.arange(slots_per_word, device=device, dtype=dtype) * slot_bits
            self._cache[key] = shifts
        
        return self._cache[key]
    
    def get_slot_indices(
        self,
        device: torch.device,
        num_slots: int
    ) -> Tensor:
        """
        Get cached slot indices for position tracking.
        
        This replaces repeated `torch.arange(num_slots)` calls.
        
        Args:
            device: Target device
            num_slots: Number of slots
        
        Returns:
            Tensor of shape (num_slots,) with values [0, 1, 2, ..., num_slots-1]
        
        Example:
            >>> indices = cache.get_slot_indices(torch.device('cpu'), 16)
            >>> indices
            tensor([ 0,  1,  2, ..., 15])
        """
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device() if torch.cuda.is_available() else 0)
        
        key = ('slot_indices', device.type, device.index, num_slots)
        
        # During torch.compile tracing, skip cache to avoid guard issues
        if torch.compiler.is_compiling():
            return torch.arange(num_slots, device=device)
        
        if key not in self._cache:
            # Create tensor if not cached (CPython dict assignment is atomic)
            indices = torch.arange(num_slots, device=device)
            self._cache[key] = indices
        
        return self._cache[key]
    
    def get_position_array(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_slots: int
    ) -> Tensor:
        """
        Get cached position array for scoring.
        
        Used in find operations for breaking ties (prefer earlier slots).
        
        Args:
            device: Target device
            dtype: Target dtype (typically float32)
            num_slots: Number of slots
        
        Returns:
            Tensor of shape (num_slots,) with float positions
        
        Example:
            >>> positions = cache.get_position_array(torch.device('cpu'), torch.float32, 16)
            >>> positions
            tensor([ 0.,  1.,  2., ..., 15.])
        """
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device() if torch.cuda.is_available() else 0)
        
        key = ('positions', device.type, device.index, dtype, num_slots)
        
        # During torch.compile tracing, skip cache to avoid guard issues
        if torch.compiler.is_compiling():
            return torch.arange(num_slots, device=device, dtype=dtype)
        
        if key not in self._cache:
            # Create tensor if not cached (CPython dict assignment is atomic)
            positions = torch.arange(num_slots, device=device, dtype=dtype)
            self._cache[key] = positions
        
        return self._cache[key]
    
    def get_constant(
        self,
        value: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tensor:
        """
        Get cached scalar constant tensor.
        
        This replaces repeated `torch.tensor(value, device=..., dtype=...)` calls.
        
        Args:
            value: Scalar value (e.g., -inf, 0.0, 1.0)
            device: Target device
            dtype: Target dtype
        
        Returns:
            Zero-dim tensor with the value
        
        Example:
            >>> neg_inf = cache.get_constant(-float('inf'), torch.device('cpu'), torch.float32)
            >>> neg_inf.item()
            -inf
        """
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device() if torch.cuda.is_available() else 0)
        
        # Use string representation for special values like -inf, nan
        value_key = str(value) if (value != value or value == float('inf') or value == float('-inf')) else value
        key = ('constant', value_key, device.type, device.index, dtype)
        
        # During torch.compile tracing, skip cache to avoid guard issues
        if torch.compiler.is_compiling():
            return torch.tensor(value, device=device, dtype=dtype)
        
        if key not in self._cache:
            # Create tensor if not cached (CPython dict assignment is atomic)
            tensor = torch.tensor(value, device=device, dtype=dtype)
            self._cache[key] = tensor
        
        return self._cache[key]
    
    def clear(self) -> None:
        """
        Clear all cached tensors.
        
        Useful for testing or when memory pressure is high.
        Note: Cached tensors will be recreated on next access.
        
        Example:
            >>> cache.clear()
            >>> cache.size()
            0
        """
        self._cache.clear()
    
    def size(self) -> int:
        """
        Return number of cached entries.
        
        Useful for monitoring cache usage.
        
        Returns:
            Number of cached tensors
        
        Example:
            >>> cache.size()
            5
        """
        return len(self._cache)
    
    def memory_usage_bytes(self) -> int:
        """
        Estimate total memory usage of cached tensors.
        
        Returns:
            Approximate memory usage in bytes
        
        Example:
            >>> cache.memory_usage_bytes()
            1024
        """
        total = 0
        for tensor in self._cache.values():
            total += tensor.element_size() * tensor.numel()
        return total


# Global singleton cache
_DEVICE_CACHE: DeviceCache = DeviceCache()


def get_device_cache() -> DeviceCache:
    """
    Get the global device cache instance.
    
    Returns:
        The singleton DeviceCache instance
    
    Example:
        >>> cache = get_device_cache()
        >>> shifts = cache.get_slot_shifts(device, dtype, slots_per_word, slot_bits)
    """
    return _DEVICE_CACHE
