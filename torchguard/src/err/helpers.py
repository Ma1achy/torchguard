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
from ..core.config import ErrorConfig, get_config
from ..core.constants import CODE_SHIFT, SLOT_BITS, SLOT_MASK, SLOTS_PER_WORD
from ..core.device_cache import get_device_cache
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
    # Use experimental backend for float dtypes (float32 or float64)
    if flags.dtype in (torch.float32, torch.float64):
        from ..experimental.ops import Float64ErrorOps
        return bool(Float64ErrorOps.any_err(flags))
    else:
        return bool(ErrorOps.any_err(flags))


def find(code: int, flags: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
    """
    Find which samples have a specific error code. Fully vectorized.
    
    Safe for hot path - all tensor ops, torch.compile friendly.
    Automatically handles both int64 (stable) and float64 (experimental) backends.
    
    Args:
        code (int): Error code to search for
        flags (Tensor): Error flags tensor (N, num_words)
        config (ErrorConfig): Error configuration
    
    Returns:
        (Tensor): Boolean mask (N,) - True where sample has this error
    """
    cfg = config or get_config()
    
    # Use experimental backend for float dtypes (float32 or float64)
    if flags.dtype in (torch.float32, torch.float64):
        from ..experimental.ops import Float64ErrorOps
        return Float64ErrorOps.has_code(flags, code, cfg)
    
    # Stable int64 backend
    cache = get_device_cache()
    N, num_words = flags.shape
    device = flags.device
    dtype = flags.dtype
    
    # Use cached slot shifts
    slot_shifts = cache.get_slot_shifts(device, dtype, SLOTS_PER_WORD, SLOT_BITS)
    words = flags.unsqueeze(-1)
    slots = (words >> slot_shifts) & SLOT_MASK
    slot_codes = (slots >> CODE_SHIFT) & 0xF
    
    total_slots = num_words * SLOTS_PER_WORD
    if cfg.num_slots < total_slots:
        # Use cached slot indices
        valid = cache.get_slot_indices(device, total_slots) < cfg.num_slots
        valid = valid.view(num_words, SLOTS_PER_WORD)
        # Use scalar broadcast instead of torch.zeros(1, ...)
        slot_codes = torch.where(
            valid.unsqueeze(0),
            slot_codes,
            torch.zeros((), dtype=dtype, device=device)
        )
    
    matches = (slot_codes == code)
    return matches.any(dim=(1, 2))


# ═══════════════════════════════════════════════════════════════════════════════
# CORE PUSH FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def push(flags: Tensor, code: int, module: Union[nn.Module, int, str, None], *, where: Optional[Tensor] = None, severity: Optional[int] = None, config: Optional[ErrorConfig] = None) -> Tensor:
    """
    Push error code into flags where condition is True.
    
    Location auto-resolved from module at trace-time.
    Severity auto-resolved from code if not provided.
    Config auto-resolved from flags dtype if not provided.
    
    Automatically detects experimental (float64/float32) vs stable (int64) backend
    based on flags dtype.
    
    Args:
        flags (Tensor): Existing error flags (N, num_words)
        code (int): Error code constant
        module (Union[nn.Module, int, str, None]): Module for auto-location
        where (Optional[Tensor]): Boolean mask (N,) - only push where True
        severity (Optional[int]): Severity (auto from code if None)
        config (Optional[ErrorConfig]): Error configuration (auto from flags dtype if None)
    
    Returns:
        (Tensor): Updated error flags
    """
    loc = resolve_location(module)
    
    if severity is None:
        severity = ErrorCode.default_severity(code)
    
    # Use compile-safe tensor creation (avoid shape[0] extraction)
    template = flags[:, 0]  # Shape (N,) - preserves symbolic shapes
    
    # Use experimental backend for float dtypes (float32 or float64)
    if flags.dtype in (torch.float32, torch.float64):
        from ..experimental.ops import Float64ErrorOps
        
        # Use provided config or fall back to global config
        cfg = config or get_config()
        config = cfg
        
        int_dtype = torch.int32 if flags.dtype == torch.float32 else torch.int64
        
        if where is None:
            code_tensor = torch.full_like(template, code, dtype=int_dtype)
        else:
            code_tensor = torch.where(
                where, 
                torch.full_like(template, code, dtype=int_dtype),
                torch.full_like(template, ErrorCode.OK, dtype=int_dtype)
            )
        
        return Float64ErrorOps.push(flags, code_tensor, loc, severity, config)
    else:
        # Stable backend for int64/int32 flags
        cfg = config or get_config()
        config = cfg
        
        if where is None:
            code_tensor = torch.full_like(template, code, dtype=torch.int64)
        else:
            code_tensor = torch.where(
                where, 
                torch.full_like(template, code, dtype=torch.int64),
                torch.full_like(template, ErrorCode.OK, dtype=torch.int64)
            )
        
        return ErrorOps.push(flags, code_tensor, loc, severity, config)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def fix(tensor: Tensor, flags: Tensor, module: Union[nn.Module, int, str, None], fallback: Union[float, Tensor, Callable[[], Tensor]] = 0.0, config: Optional[ErrorConfig] = None) -> Tuple[Tensor, Tensor]:
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
    cfg = config or get_config()
    loc = resolve_location(module)
    
    # Use experimental backend for float dtypes (float32 or float64)
    if flags.dtype in (torch.float32, torch.float64):
        from ..experimental.ops import Float64ErrorOps
        bad_mask = Float64ErrorOps.is_err(flags)
    else:
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
    
    cache = get_device_cache()
    int_dtype = torch.int32 if flags.dtype == torch.float32 else torch.int64
    code = torch.where(
        bad_mask, 
        cache.get_constant(float(ErrorCode.FALLBACK_VALUE), flags.device, int_dtype),
        cache.get_constant(float(ErrorCode.OK), flags.device, int_dtype)
    )
    
    # Use experimental backend for float dtypes
    if flags.dtype in (torch.float32, torch.float64):
        from ..experimental.ops import Float64ErrorOps
        updated_flags = Float64ErrorOps.push(flags, code, loc, Severity.WARN, cfg)
    else:
        updated_flags = ErrorOps.push(flags, code, loc, Severity.WARN, cfg)
    
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

def flag_nan(tensor: Tensor, module: Union[nn.Module, int, str, None], flags: Optional[Tensor] = None, config: Optional[ErrorConfig] = None) -> Tensor:
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
        config (Optional[ErrorConfig]): Error configuration (auto from flags dtype if None)
    
    Returns:
        (Tensor): Updated error flags with NAN code where detected
    """
    n = tensor.shape[0]
    if flags is None:
        if config is None:
            config = get_config()
        flags = ErrorOps.new_t(n, tensor.device, config)
    nan_mask = torch.isnan(tensor).view(n, -1).any(dim=-1)
    return push(flags, ErrorCode.NAN, module, where=nan_mask, config=config)


def flag_inf(tensor: Tensor, module: Union[nn.Module, int, str, None], flags: Optional[Tensor] = None, config: Optional[ErrorConfig] = None) -> Tensor:
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
        config (Optional[ErrorConfig]): Error configuration (auto from flags dtype if None)
    
    Returns:
        (Tensor): Updated error flags with INF code where detected
    """
    n = tensor.shape[0]
    if flags is None:
        if config is None:
            config = get_config()
        flags = ErrorOps.new_t(n, tensor.device, config)
    inf_mask = torch.isinf(tensor).view(n, -1).any(dim=-1)
    return push(flags, ErrorCode.INF, module, where=inf_mask, config=config)


def flag_nan_and_inf(tensor: Tensor, module: Union[nn.Module, int, str, None], flags: Optional[Tensor] = None, config: Optional[ErrorConfig] = None) -> Tensor:
    """
    Check tensor for both NaN and Inf in a single fused pass. Traceable, hot-path safe.
    
    This is more efficient than calling flag_nan() and flag_inf() separately when
    you want to detect both issues, as it only iterates through the tensor once.
    
    Records ErrorCode.NAN for samples with NaN, ErrorCode.INF for samples with Inf.
    If a sample has both, both error codes are pushed (NaN first, then Inf).
    
    Equivalent to:
        nan_mask = torch.isnan(tensor).view(n, -1).any(dim=-1)
        flags = push(flags, ErrorCode.NAN, module, where=nan_mask)
        inf_mask = torch.isinf(tensor).view(n, -1).any(dim=-1)
        flags = push(flags, ErrorCode.INF, module, where=inf_mask)
    
    Args:
        tensor (Tensor): Input tensor to check (N, ...)
        module (Union[nn.Module, int, str, None]): Module for auto-location
        flags (Optional[Tensor]): Existing flags, or None to create new
        config (Optional[ErrorConfig]): Error configuration (auto from flags dtype if None)
    
    Returns:
        (Tensor): Updated error flags with NAN and/or INF codes where detected
    """
    n = tensor.shape[0]
    if flags is None:
        if config is None:
            config = get_config()
        flags = ErrorOps.new_t(n, tensor.device, config)
    
    # Single reshape for both checks
    flat = tensor.view(n, -1)
    
    # Check for NaN and Inf in a single iteration (fused)
    nan_mask = torch.isnan(flat).any(dim=-1)
    inf_mask = torch.isinf(flat).any(dim=-1)
    
    # Push both error codes
    flags = push(flags, ErrorCode.NAN, module, where=nan_mask, config=config)
    flags = push(flags, ErrorCode.INF, module, where=inf_mask, config=config)
    
    return flags


def flag_oob_indices(indices: Tensor, num_embeddings: int, module: Union[nn.Module, int, str, None], flags: Optional[Tensor] = None, config: Optional[ErrorConfig] = None) -> Tensor:
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
        config (Optional[ErrorConfig]): Error configuration (auto from flags dtype if None)
    
    Returns:
        (Tensor): Updated error flags with OUT_OF_BOUNDS code where detected
    """
    n = indices.shape[0]
    if flags is None:
        if config is None:
            config = get_config()
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
