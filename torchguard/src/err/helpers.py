"""
Error query and push functions with auto-location resolution.

Key API:
- has_err(flags) -> bool (scalar) - any error in batch?
- find(code, flags) -> Tensor[bool] (N,) - which samples have this error?
- push(flags, code, module, where=mask) -> record error where True
- fix(tensor, flags, module) -> replace bad values

Detection is done with standard PyTorch: torch.isnan(), torch.isinf(), etc.
This module only handles FLAGS, not raw tensor detection.
"""
from __future__ import annotations

import warnings
import weakref
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from ..core.codes import ErrorCode
from ..core.config import DEFAULT_CONFIG, ErrorConfig
from ..core.constants import CODE_SHIFT, SLOT_BITS, SLOT_MASK, SLOTS_PER_WORD
from ..core.severity import Severity
from .ops import ErrorOps
from ..core.location import ErrorLocation

if TYPE_CHECKING:
    import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level cache (WeakKeyDictionary for frozen module safety)
_LOCATION_CACHE: weakref.WeakKeyDictionary[nn.Module, int] = weakref.WeakKeyDictionary()

# Warn-once pattern to prevent log spam
_WARNED_KEYS: set[tuple] = set()


def __warn_once(key: tuple, msg: str) -> None:
    """
    Warn only once per unique key. Skips during torch.compile.
    
    Args:
        key (tuple): Unique key for deduplication
        msg (str): Warning message
    """
    if torch.compiler.is_compiling():
        return
    if key not in _WARNED_KEYS:
        _WARNED_KEYS.add(key)
        warnings.warn(msg, stacklevel=3)


# ═══════════════════════════════════════════════════════════════════════════════
# LOCATION RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_location(module: Union[nn.Module, int, str, None]) -> int:
    """
    Resolve location ID from module, int, string, or None.
    
    Called at trace time - the returned integer becomes a compile-time
    constant baked into the compiled graph.
    
    Resolution order:
    1. None -> ErrorLocation.UNKNOWN
    2. int -> passthrough (already a location ID)
    3. str -> ErrorLocation.get(str) if exists, else register (only outside compile)
    4. nn.Module:
       a. Check _LOCATION_CACHE (WeakKeyDictionary)
       b. Check _fx_path (injected by @tracked) -> lookup or register
       c. Fallback to UNKNOWN during compile, or class name outside compile
    
    Args:
        module (Union[nn.Module, int, str, None]): Module, int, str, or None
    
    Returns:
        (int): Location ID (0-1023)
    
    Note:
        During torch.compile, we cannot register new locations (uses threading lock).
        Use @tracked on your model class to auto-inject _fx_path before compilation.
    """
    if module is None:
        return ErrorLocation.UNKNOWN
    
    if isinstance(module, int):
        return module
    
    if isinstance(module, str):
        loc_id = ErrorLocation.get(module)
        if loc_id != ErrorLocation.UNKNOWN:
            return loc_id
        if torch.compiler.is_compiling():
            return ErrorLocation.UNKNOWN
        return ErrorLocation.register(module)
    
    # nn.Module - check cache first
    if module in _LOCATION_CACHE:
        return _LOCATION_CACHE[module]
    
    # Check for _fx_path (injected by @tracked)
    fx_path = getattr(module, '_fx_path', None)
    if fx_path is not None:
        loc_id = ErrorLocation.get(fx_path)
        if loc_id == ErrorLocation.UNKNOWN:
            if torch.compiler.is_compiling():
                return ErrorLocation.UNKNOWN
            loc_id = ErrorLocation.register(fx_path)
    else:
        if torch.compiler.is_compiling():
            return ErrorLocation.UNKNOWN
        
        name = module.__class__.__name__
        __warn_once(
            ('location_fallback', name),
            f"Module {name} has no _fx_path, using class name '{name}'. "
            f"Multiple modules with same class will share this location. "
            f"Use @tracked on parent class to get precise locations."
        )
        loc_id = ErrorLocation.register(name)
    
    if loc_id != ErrorLocation.UNKNOWN:
        _LOCATION_CACHE[module] = loc_id
    return loc_id


# ═══════════════════════════════════════════════════════════════════════════════
# CORE QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def has_err(flags: Tensor) -> bool:
    """
    Check if any error exists in the batch. PYTHON BOUNDARY ONLY.
    
    Returns Python bool - for logging, asserts, monitoring.
    For compiled code, use HAS(flags) from control.py instead.
    
    Args:
        flags (Tensor): Error flags tensor (N, num_words)
    
    Returns:
        (bool): True if any sample has any error
    """
    return bool(ErrorOps.any_err(flags))


def find(code: int, flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
    """
    Find which samples have a specific error code. Fully vectorized.
    
    Safe for hot path - all tensor ops, torch.compile friendly.
    
    Args:
        code (int): Error code to search for
        flags (Tensor): Error flags tensor (N, num_words)
        config (ErrorConfig): Error configuration
    
    Returns:
        (Tensor): Boolean mask (N,) - True where sample has this error
    """
    N, num_words = flags.shape
    device = flags.device
    dtype = flags.dtype
    
    slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=dtype) * SLOT_BITS
    words = flags.unsqueeze(-1)
    slots = (words >> slot_shifts) & SLOT_MASK
    slot_codes = (slots >> CODE_SHIFT) & 0xF
    
    total_slots = num_words * SLOTS_PER_WORD
    if config.num_slots < total_slots:
        valid = torch.arange(total_slots, device=device) < config.num_slots
        valid = valid.view(num_words, SLOTS_PER_WORD)
        slot_codes = torch.where(
            valid.unsqueeze(0),
            slot_codes,
            torch.zeros(1, dtype=dtype, device=device)
        )
    
    matches = (slot_codes == code)
    return matches.any(dim=(1, 2))


# ═══════════════════════════════════════════════════════════════════════════════
# CORE PUSH FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def push(flags: Tensor, code: int, module: Union[nn.Module, int, str, None], *, where: Optional[Tensor] = None, severity: Optional[int] = None, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
    """
    Push error code into flags where condition is True.
    
    Location auto-resolved from module at trace-time.
    Severity auto-resolved from code if not provided.
    
    Args:
        flags (Tensor): Existing error flags (N, num_words)
        code (int): Error code constant
        module (Union[nn.Module, int, str, None]): Module for auto-location
        where (Optional[Tensor]): Boolean mask (N,) - only push where True
        severity (Optional[int]): Severity (auto from code if None)
        config (ErrorConfig): Error configuration
    
    Returns:
        (Tensor): Updated error flags
    """
    loc = resolve_location(module)
    
    if severity is None:
        severity = ErrorCode.default_severity(code)
    
    n = flags.shape[0]
    
    if where is not None and where.shape[0] != n:
        raise ValueError(
            f"where mask shape mismatch: flags has {n} samples, "
            f"mask has {where.shape[0]} samples"
        )
    
    if where is None:
        code_tensor = torch.full((n,), code, dtype=torch.int64, device=flags.device)
    else:
        code_tensor = torch.where(where, code, ErrorCode.OK)
    
    return ErrorOps.push(flags, code_tensor, loc, severity, config)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def fix(tensor: Tensor, flags: Tensor, module: Union[nn.Module, int, str, None], fallback: Union[float, Tensor, Callable[[], Tensor]] = 0.0, config: ErrorConfig = DEFAULT_CONFIG) -> Tuple[Tensor, Tensor]:
    """
    Replace bad values (where flags have errors) with fallback.
    
    Records FALLBACK_VALUE error for samples that were fixed.
    
    Args:
        tensor (Tensor): Input tensor (N, ...)
        flags (Tensor): Error flags (N, num_words)
        module (Union[nn.Module, int, str, None]): Module for auto-location
        fallback (Union[float, Tensor, Callable[[], Tensor]]): Replacement value
        config (ErrorConfig): Error configuration
    
    Returns:
        (Tuple[Tensor, Tensor]): (cleaned_tensor, updated_flags)
    """
    loc = resolve_location(module)
    bad_mask = ErrorOps.is_err(flags)
    
    if callable(fallback):
        fallback_val = fallback()
    else:
        fallback_val = fallback
    
    bad_mask_exp = bad_mask
    while bad_mask_exp.dim() < tensor.dim():
        bad_mask_exp = bad_mask_exp.unsqueeze(-1)
    bad_mask_exp = bad_mask_exp.expand_as(tensor)
    
    cleaned = torch.where(bad_mask_exp, fallback_val, tensor)
    
    code = torch.where(bad_mask, ErrorCode.FALLBACK_VALUE, ErrorCode.OK)
    updated_flags = ErrorOps.push(flags, code, loc, Severity.WARN, config)
    
    return cleaned, updated_flags


# ═══════════════════════════════════════════════════════════════════════════════
# FLAG HELPERS (Error Recording)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These helpers combine detection + recording into a single call.
# They are NOT mask-producing functions for control flow.
#
# Pipeline stages:
#   1. Detection (pure tensor logic) - torch.isnan(), torch.isinf(), etc.
#   2. Error recording (these helpers) - flag_nan(), flag_inf(), flag_oob_indices()
#   3. Control flow/masking - find(), ErrorOps.is_err(), IF/HAS/IS/etc.
#
# Use these when you want the convenience of a one-liner.
# Use push() directly when you need more control over the detection logic.
# ═══════════════════════════════════════════════════════════════════════════════

def flag_nan(tensor: Tensor, module: Union[nn.Module, int, str, None], flags: Optional[Tensor] = None, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
    """
    Check tensor for NaN and write ErrorCode.NAN to flags. Traceable, hot-path safe.
    
    This is an error recording helper, not a mask-producing function.
    For control flow, use find(ErrorCode.NAN, flags) or ErrorOps.has_nan(flags).
    
    Equivalent to:
        nan_mask = torch.isnan(tensor).view(n, -1).any(dim=-1)
        flags = push(flags, ErrorCode.NAN, module, where=nan_mask)
    
    Args:
        tensor (Tensor): Input tensor to check (N, ...)
        module (Union[nn.Module, int, str, None]): Module for auto-location
        flags (Optional[Tensor]): Existing flags, or None to create new
        config (ErrorConfig): Error configuration
    
    Returns:
        (Tensor): Updated error flags with NAN code where detected
    """
    n = tensor.shape[0]
    if flags is None:
        flags = ErrorOps.new_t(n, tensor.device, config)
    nan_mask = torch.isnan(tensor).view(n, -1).any(dim=-1)
    return push(flags, ErrorCode.NAN, module, where=nan_mask, config=config)


def flag_inf(tensor: Tensor, module: Union[nn.Module, int, str, None], flags: Optional[Tensor] = None, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
    """
    Check tensor for Inf and write ErrorCode.INF to flags. Traceable, hot-path safe.
    
    This is an error recording helper, not a mask-producing function.
    For control flow, use find(ErrorCode.INF, flags) or ErrorOps.has_inf(flags).
    
    Equivalent to:
        inf_mask = torch.isinf(tensor).view(n, -1).any(dim=-1)
        flags = push(flags, ErrorCode.INF, module, where=inf_mask)
    
    Args:
        tensor (Tensor): Input tensor to check (N, ...)
        module (Union[nn.Module, int, str, None]): Module for auto-location
        flags (Optional[Tensor]): Existing flags, or None to create new
        config (ErrorConfig): Error configuration
    
    Returns:
        (Tensor): Updated error flags with INF code where detected
    """
    n = tensor.shape[0]
    if flags is None:
        flags = ErrorOps.new_t(n, tensor.device, config)
    inf_mask = torch.isinf(tensor).view(n, -1).any(dim=-1)
    return push(flags, ErrorCode.INF, module, where=inf_mask, config=config)


def flag_oob_indices(indices: Tensor, num_embeddings: int, module: Union[nn.Module, int, str, None], flags: Optional[Tensor] = None, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
    """
    Check indices for out-of-bounds and write ErrorCode.OUT_OF_BOUNDS to flags.
    
    This is an error recording helper, not a mask-producing function.
    For control flow, use find(ErrorCode.OUT_OF_BOUNDS, flags).
    
    Equivalent to:
        oob_mask = ((indices < 0) | (indices >= num_embeddings)).view(n, -1).any(dim=-1)
        flags = push(flags, ErrorCode.OUT_OF_BOUNDS, module, where=oob_mask)
    
    Args:
        indices (Tensor): Index tensor to check (N, ...)
        num_embeddings (int): Size of the embedding table (valid range: 0 to num_embeddings-1)
        module (Union[nn.Module, int, str, None]): Module for auto-location
        flags (Optional[Tensor]): Existing flags, or None to create new
        config (ErrorConfig): Error configuration
    
    Returns:
        (Tensor): Updated error flags with OUT_OF_BOUNDS code where detected
    """
    n = indices.shape[0]
    if flags is None:
        flags = ErrorOps.new_t(n, indices.device, config)
    flat = indices.view(n, -1)
    oob_mask = ((flat < 0) | (flat >= num_embeddings)).any(dim=-1)
    return push(flags, ErrorCode.OUT_OF_BOUNDS, module, where=oob_mask, config=config)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clear_location_cache() -> None:
    """Clear the location cache (for testing)."""
    _LOCATION_CACHE.clear()


def clear_warn_cache() -> None:
    """Clear the warning dedup cache (for testing)."""
    _WARNED_KEYS.clear()
