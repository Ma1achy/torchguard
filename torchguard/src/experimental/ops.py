"""
FlexibleErrorOps: Experimental backend with configurable float dtype storage.

Bit layout:
    - 16-bit slots: [location:10][code:4][severity:2]
    - float32: 32-bit words, 2 slots per word, viewed as int32
    - float64: 64-bit words, 4 slots per word, viewed as int64

Default is float32 for better torch.compile compatibility.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from ..core.codes import ErrorCode
from ..core.config import ErrorConfig, Dedupe, Priority, Order, get_config
from ..core.constants import CODE_SHIFT, LOCATION_SHIFT, SEVERITY_MASK, SLOT_BITS, SLOT_MASK, SLOTS_PER_WORD
from ..core.device_cache import get_device_cache
from ..core.severity import Severity


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW HELPERS — Dtype-aware conversion between float and int views
# ═══════════════════════════════════════════════════════════════════════════════

def _as_int(flags: Tensor) -> Tensor:
    """
    View float flags as corresponding int type for bitwise operations.
    
    Zero-copy operation — same memory, different dtype interpretation.
    All bitwise operations (&, |, ^, <<, >>) should use this view.
    
    Args:
        flags: (N, num_words) float32 or float64 tensor
    
    Returns:
        (N, num_words) int32 or int64 tensor (same storage)
    """
    if flags.dtype == torch.float32:
        return flags.view(torch.int32)
    else:
        return flags.view(torch.int64)


def _as_float(flags_i: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
    """
    View int flags as corresponding float type for return.
    
    Zero-copy operation — same memory, different dtype interpretation.
    Use this when returning flags from operations.
    
    Args:
        flags_i: (N, num_words) int32 or int64 tensor
        config: ErrorConfig with flag_dtype
    
    Returns:
        (N, num_words) float32 or float64 tensor (same storage)
    """
    if flags_i.dtype == torch.int32:
        return flags_i.view(torch.float32)
    else:
        return flags_i.view(torch.float64)


def _get_word_bits(config: Optional[ErrorConfig] = None) -> int:
    """Get number of bits per word based on config."""
    return config.word_bits


def _get_slots_per_word(config: Optional[ErrorConfig] = None) -> int:
    """Get number of slots per word based on config."""
    return config.slots_per_word


class Float64ErrorOps:
    """
    Flexible error operations with configurable float storage (float32 or float64).
    
    API is identical to ErrorOps from the stable backend.
    Uses float storage for AOTAutograd/torch.compile compatibility.
    
    Bit Layout:
        Storage: float32 or float64 tensor, shape (N, num_words)
        View: int32 or int64 for all bitwise operations.
        
        float32: 32-bit words, 2 slots per word (default)
        float64: 64-bit words, 4 slots per word
        
        Slot (16 bits): [location:10][code:4][severity:2]
        - severity: 0=OK, 1=WARN, 2=ERROR, 3=CRITICAL
        - code: Error type (NaN=1, Inf=2, OutOfBounds=5, etc.)
        - location: Registered location ID (0-1023)
    
    Note: Class is named Float64ErrorOps for backward compatibility but supports both dtypes.
    """
    
    # Default dtype (can be overridden via config or get_config())
    _dtype = torch.float32  # Changed default from float64 to float32
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS - Bit Packing (identical to stable, but uses views)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def __pack_slot(code: int, location: int, severity: int) -> int:
        """Pack code, location, severity into 16-bit slot value."""
        return (severity & 0x3) | ((code & 0xF) << CODE_SHIFT) | ((location & 0x3FF) << LOCATION_SHIFT)
    
    @staticmethod
    def __pack_slot_tensor(code: Tensor, location: int, severity: int, config: Optional[ErrorConfig] = None) -> Tensor:
        """Pack code tensor with location and severity. Compilable. Dtype-aware."""
        int_dtype = config.torch_int_dtype
        return severity | ((code.to(int_dtype) & 0xF) << CODE_SHIFT) | (location << LOCATION_SHIFT)
    
    @staticmethod
    def __broadcast_mask(mask: Tensor, z: Tensor) -> Tensor:
        """Broadcast (N,) bool mask to match z's shape (N, d1, d2, ...)."""
        if z.ndim == 1:
            return mask
        shape = (mask.shape[0],) + (1,) * (z.ndim - 1)
        return mask.view(shape).expand_as(z)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS - Vectorized Slot Operations (dtype-aware)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def __extract_all_slots(flags: Tensor, config: ErrorConfig) -> Tensor:
        """
        Extract all slots into (N, num_slots) tensor. Fully vectorized.
        Uses int view (int32 or int64) for bitwise operations based on dtype.
        """
        flags_i = _as_int(flags)
        n, num_words = flags_i.shape
        device = flags_i.device
        int_dtype = config.torch_int_dtype
        slots_per_word = config.slots_per_word
        
        cache = get_device_cache()
        shifts = cache.get_slot_shifts(device, int_dtype, slots_per_word, SLOT_BITS)
        expanded = flags_i.unsqueeze(-1) >> shifts
        masked = expanded & SLOT_MASK
        
        return masked.reshape(n, -1)[:, :config.num_slots]
    
    @staticmethod
    def __pack_all_slots(slots: Tensor, config: ErrorConfig) -> Tensor:
        """
        Pack (N, num_slots) slots back into (N, num_words) float flags.
        Dtype-aware: works with both float32/int32 and float64/int64.
        """
        n = slots.shape[0]
        num_words = config.num_words
        device = slots.device
        int_dtype = config.torch_int_dtype
        slots_per_word = config.slots_per_word
        
        total_slot_capacity = num_words * slots_per_word
        if slots.shape[1] < total_slot_capacity:
            padded = torch.zeros(n, total_slot_capacity, dtype=int_dtype, device=device)
            padded[:, :slots.shape[1]] = slots
        else:
            padded = slots[:, :total_slot_capacity]
        
        reshaped = padded.reshape(n, num_words, slots_per_word)
        cache = get_device_cache()
        shifts = cache.get_slot_shifts(device, int_dtype, slots_per_word, SLOT_BITS)
        shifted = reshaped << shifts
        # Note: sum() promotes int32 to int64, need to cast back
        flags_i = shifted.sum(dim=-1).to(int_dtype)
        
        return _as_float(flags_i, config or get_config())
    
    @staticmethod
    def __lifo_shift_one(flags: Tensor, new_slot: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
        """
        LIFO shift: shift all slots right by one, insert new_slot at position 0.
        Dtype-aware: works with both 32-bit and 64-bit word sizes.
        """
        flags_i = _as_int(flags)
        n, num_words = flags_i.shape
        word_bits = config.word_bits
        KEEP_MASK = (1 << (word_bits - SLOT_BITS)) - 1
        
        carry = (flags_i >> (word_bits - SLOT_BITS)) & SLOT_MASK
        shifted = (flags_i & KEEP_MASK) << SLOT_BITS
        
        result = shifted.clone()
        result[:, 0] = result[:, 0] | new_slot
        if num_words > 1:
            result[:, 1:] = result[:, 1:] | carry[:, :-1]
        
        return _as_float(result, config or get_config())
    
    @staticmethod
    def __find_slot_matching(flags: Tensor, pred_fn, config: ErrorConfig) -> Tuple[Tensor, Tensor]:
        """Find first slot matching a predicate. Uses int64 view."""
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config or get_config())
        matches = pred_fn(all_slots)
        exists = matches.any(dim=1)
        
        match_scores = matches.float()
        cache = get_device_cache()
        positions = cache.get_position_array(flags.device, torch.float32, config.num_slots)
        scores = match_scores - positions * 1e-6
        scores = torch.where(matches, scores, float('-inf'))
        slot_idx = scores.argmax(dim=1)
        
        return exists, slot_idx
    
    @staticmethod
    def __replace_slot_at(flags: Tensor, slot_idx: Tensor, new_slot: Tensor, config: ErrorConfig) -> Tensor:
        """Replace slot at given index with new value. Returns same dtype as input."""
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config or get_config())
        n, num_slots = all_slots.shape
        
        cache = get_device_cache()
        indices = cache.get_slot_indices(flags.device, num_slots).unsqueeze(0)
        is_target = (indices == slot_idx.unsqueeze(1))
        updated = torch.where(is_target, new_slot.unsqueeze(1), all_slots)
        
        return Float64ErrorOps.__pack_all_slots(updated, config or get_config())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CREATION — Return float tensors (float32 by default, configurable)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def new(x: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
        """Create empty error flags from a reference tensor."""
        return torch.zeros(x.shape[0], config.num_words, dtype=config.torch_dtype, device=x.device)
    
    @staticmethod
    def new_t(n: int, device: Optional[torch.device] = None, config: Optional[ErrorConfig] = None) -> Tensor:
        """Create empty error flags with explicit arguments."""
        return torch.zeros(n, config.num_words, dtype=config.torch_dtype, device=device)
    
    @staticmethod
    def from_code(code: int, location: int, n: int, device: Optional[torch.device] = None, severity: int = Severity.ERROR, config: Optional[ErrorConfig] = None) -> Tensor:
        """Create flags with a single error for all samples."""
        flags = Float64ErrorOps.new_t(n, device, config or get_config())
        packed = Float64ErrorOps.__pack_slot(code, location, severity)
        flags_i = _as_int(flags)
        flags_i[:, 0] = packed
        return _as_float(flags_i, config or get_config())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RECORDING - Push Methods (use int64 view for bitwise ops)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def push(flags: Tensor, code: Tensor, location: int, severity: int = Severity.ERROR, config: Optional[ErrorConfig] = None) -> Tensor:
        """Push new error into flags. Returns same dtype as input."""
        acc = config.accumulation
        
        if acc.dedupe == Dedupe.LOCATION:
            return Float64ErrorOps.__push_dedupe_location(flags, code, location, severity, config or get_config())
        elif acc.dedupe == Dedupe.CODE:
            return Float64ErrorOps.__push_dedupe_code(flags, code, location, severity, config or get_config())
        elif acc.dedupe == Dedupe.UNIQUE:
            return Float64ErrorOps.__push_dedupe_unique(flags, code, location, severity, config or get_config())
        else:
            return Float64ErrorOps.__push_no_dedupe(flags, code, location, severity, config or get_config())
    
    @staticmethod
    def __push_no_dedupe(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Push without deduplication."""
        acc = config.accumulation
        if acc.priority == Priority.CHRONO:
            if acc.order == Order.LAST:
                return Float64ErrorOps.__push_chrono_last(flags, code, location, severity, config or get_config())
            else:
                return Float64ErrorOps.__push_chrono_first(flags, code, location, severity, config or get_config())
        elif acc.priority == Priority.SEVERITY:
            return Float64ErrorOps.__push_severity_based(flags, code, location, severity, config or get_config())
        else:
            return Float64ErrorOps.__push_chrono_last(flags, code, location, severity, config or get_config())
    
    @staticmethod
    def __push_chrono_last(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """LIFO push. Dtype-aware."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity, config or get_config())
        new_flags = Float64ErrorOps.__lifo_shift_one(flags, new_slot, config or get_config())
        return torch.where(should_push.unsqueeze(-1), new_flags, flags)
    
    @staticmethod
    def __push_chrono_first(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """FIFO push. Dtype-aware."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity, config or get_config())
        int_dtype = config.torch_int_dtype
        error_count = Float64ErrorOps.count_errors(flags, config or get_config()).to(int_dtype)
        has_space = error_count < config.num_slots
        should_push = should_push & has_space
        result = Float64ErrorOps.__replace_slot_at(flags, error_count, new_slot, config or get_config())
        return torch.where(should_push.unsqueeze(-1), result, flags)
    
    @staticmethod
    def __push_severity_based(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Severity-priority push. Dtype-aware."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity, config or get_config())
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config or get_config())
        severities = all_slots & SEVERITY_MASK
        min_slot_idx = severities.argmin(dim=1)
        min_sev = severities.gather(1, min_slot_idx.unsqueeze(1)).squeeze(1)
        should_replace = should_push & (severity > min_sev)
        result = Float64ErrorOps.__replace_slot_at(flags, min_slot_idx, new_slot, config or get_config())
        return torch.where(
            should_replace.unsqueeze(-1), result,
            torch.where(should_push.unsqueeze(-1), Float64ErrorOps.__push_chrono_last(flags, code, location, severity, config or get_config()), flags)
        )
    
    @staticmethod
    def __push_dedupe_location(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Location-dedupe push. Dtype-aware."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity, config or get_config())
        
        def match_location(slots):
            slot_loc = (slots >> LOCATION_SHIFT) & 0x3FF
            return (slot_loc == location) & (slots != 0)
        
        loc_exists, existing_slot_idx = Float64ErrorOps.__find_slot_matching(flags, match_location, config or get_config())
        
        return torch.where(
            (loc_exists & should_push).unsqueeze(-1),
            Float64ErrorOps.__update_slot_if_worse(flags, existing_slot_idx, new_slot, config or get_config()),
            torch.where((~loc_exists & should_push).unsqueeze(-1), Float64ErrorOps.__push_no_dedupe(flags, code, location, severity, config or get_config()), flags)
        )
    
    @staticmethod
    def __push_dedupe_code(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Code-dedupe push. Dtype-aware."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity, config or get_config())
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config or get_config())
        slot_codes = (all_slots >> CODE_SHIFT) & 0xF
        matches = (slot_codes == code.unsqueeze(1)) & (all_slots != 0)
        code_exists = matches.any(dim=1)
        
        match_scores = matches.float()
        cache = get_device_cache()
        positions = cache.get_position_array(flags.device, torch.float32, config.num_slots)
        scores = match_scores - positions * 1e-6
        scores = torch.where(matches, scores, float('-inf'))
        existing_slot_idx = scores.argmax(dim=1)
        
        return torch.where(
            (code_exists & should_push).unsqueeze(-1),
            Float64ErrorOps.__update_slot_if_worse(flags, existing_slot_idx, new_slot, config or get_config()),
            torch.where((~code_exists & should_push).unsqueeze(-1), Float64ErrorOps.__push_no_dedupe(flags, code, location, severity, config or get_config()), flags)
        )
    
    @staticmethod
    def __push_dedupe_unique(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Unique-dedupe push. Dtype-aware."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity, config or get_config())
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config or get_config())
        slot_loc = (all_slots >> LOCATION_SHIFT) & 0x3FF
        slot_codes = (all_slots >> CODE_SHIFT) & 0xF
        matches = (slot_loc == location) & (slot_codes == code.unsqueeze(1)) & (all_slots != 0)
        pair_exists = matches.any(dim=1)
        
        match_scores = matches.float()
        cache = get_device_cache()
        positions = cache.get_position_array(flags.device, torch.float32, config.num_slots)
        scores = match_scores - positions * 1e-6
        scores = torch.where(matches, scores, float('-inf'))
        existing_slot_idx = scores.argmax(dim=1)
        
        return torch.where(
            (pair_exists & should_push).unsqueeze(-1),
            Float64ErrorOps.__update_slot_if_worse(flags, existing_slot_idx, new_slot, config or get_config()),
            torch.where((~pair_exists & should_push).unsqueeze(-1), Float64ErrorOps.__push_no_dedupe(flags, code, location, severity, config or get_config()), flags)
        )
    
    @staticmethod
    def __update_slot_if_worse(flags: Tensor, slot_idx: Tensor, new_slot: Tensor, config: ErrorConfig) -> Tensor:
        """Update slot only if new error is worse. Returns same dtype as input."""
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config or get_config())
        n, num_slots = all_slots.shape
        cache = get_device_cache()
        indices = cache.get_slot_indices(flags.device, num_slots).unsqueeze(0)
        is_target = (indices == slot_idx.unsqueeze(1))
        new_sev = (new_slot & SEVERITY_MASK).unsqueeze(1)
        existing_sev = all_slots & SEVERITY_MASK
        should_update = is_target & (new_sev > existing_sev)
        updated = torch.where(should_update, new_slot.unsqueeze(1), all_slots)
        return Float64ErrorOps.__pack_all_slots(updated, config or get_config())
    
    @staticmethod
    def push_scalar(flags: Tensor, code: int, location: int, severity: int = Severity.ERROR, config: Optional[ErrorConfig] = None) -> Tensor:
        """Push same error to all samples. Returns same dtype as input."""
        if code == ErrorCode.OK or severity == Severity.OK:
            return flags
        n = flags.shape[0]
        code_tensor = torch.full((n,), code, dtype=torch.int64, device=flags.device)
        return Float64ErrorOps.push(flags, code_tensor, location, severity, config or get_config())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RECORDING - Merge Methods
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def merge(*flag_tensors: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
        """Merge multiple flag tensors into one. Returns same dtype as input."""
        if not flag_tensors:
            raise ValueError("Need at least one flag tensor")
        if len(flag_tensors) == 1:
            return flag_tensors[0]
        
        result = flag_tensors[0]
        for other in flag_tensors[1:]:
            result = Float64ErrorOps.__merge_two(result, other, config or get_config())
        return result
    
    @staticmethod
    def __merge_two(flags: Tensor, other: Tensor, config: ErrorConfig) -> Tensor:
        """Merge errors from other into flags. Returns same dtype as input."""
        flags_slots = Float64ErrorOps.__extract_all_slots(flags, config or get_config())
        other_slots = Float64ErrorOps.__extract_all_slots(other, config or get_config())
        combined = torch.cat([other_slots, flags_slots], dim=1)
        non_empty = (combined & SEVERITY_MASK) != 0
        sort_key = (~non_empty).long()
        _, indices = torch.sort(sort_key, dim=1, stable=True)
        compacted = torch.gather(combined, 1, indices)
        return Float64ErrorOps.__pack_all_slots(compacted[:, :config.num_slots], config or get_config())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHECKING (bool masks) — Can work on float64 directly for simple checks
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def is_ok(flags: Tensor) -> Tensor:
        """Return bool mask where True indicates sample has NO errors."""
        return (flags == 0.0).all(dim=-1)
    
    @staticmethod
    def is_err(flags: Tensor) -> Tensor:
        """Return bool mask where True indicates sample HAS errors."""
        return (flags != 0.0).any(dim=-1)
    
    @staticmethod
    def has_code(flags: Tensor, code: int, config: Optional[ErrorConfig] = None) -> Tensor:
        """Check if any slot contains specific error code."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        int_dtype = config.torch_int_dtype
        slots_per_word = config.slots_per_word
        
        cache = get_device_cache()
        slot_shifts = cache.get_slot_shifts(device, int_dtype, slots_per_word, SLOT_BITS)
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        slot_codes = (slots >> CODE_SHIFT) & 0xF
        slot_sev = slots & 0x3
        matches = (slot_codes == code) & (slot_sev != 0)
        
        return matches.any(dim=(1, 2))
    
    @staticmethod
    def has_nan(flags: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
        """Check if any slot has NaN error."""
        return Float64ErrorOps.has_code(flags, ErrorCode.NAN, config or get_config())
    
    @staticmethod
    def has_inf(flags: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
        """Check if any slot has Inf error."""
        return Float64ErrorOps.has_code(flags, ErrorCode.INF, config or get_config())
    
    @staticmethod
    def has_critical(flags: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
        """Check if any slot has CRITICAL severity."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        int_dtype = config.torch_int_dtype
        slots_per_word = config.slots_per_word
        
        cache = get_device_cache()
        slot_shifts = cache.get_slot_shifts(device, int_dtype, slots_per_word, SLOT_BITS)
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        severities = slots & 0x3
        
        return (severities == Severity.CRITICAL).any(dim=(1, 2))
    
    @staticmethod
    def has_fallback(flags: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
        """Check if any slot indicates fallback value was used."""
        return Float64ErrorOps.has_code(flags, ErrorCode.FALLBACK_VALUE, config or get_config())
    
    @staticmethod
    def has_domain(flags: Tensor, domain: int, config: Optional[ErrorConfig] = None) -> Tensor:
        """Check if any slot has error in given domain."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        int_dtype = config.torch_int_dtype
        slots_per_word = config.slots_per_word
        
        cache = get_device_cache()
        slot_shifts = cache.get_slot_shifts(device, int_dtype, slots_per_word, SLOT_BITS)
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        
        domain_bits = (domain >> 2) & 0x3
        slot_domains = (slots >> (CODE_SHIFT + 2)) & 0x3
        slot_sev = slots & 0x3
        
        matches = (slot_domains == domain_bits) & (slot_sev != 0)
        return matches.any(dim=(1, 2))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FILTERING — Use float64 comparison for masks
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def get_ok(flags: Tensor) -> Tensor:
        """Return only the flags for samples WITHOUT errors."""
        return flags[Float64ErrorOps.is_ok(flags)]
    
    @staticmethod
    def get_err(flags: Tensor) -> Tensor:
        """Return only the flags for samples WITH errors."""
        return flags[Float64ErrorOps.is_err(flags)]
    
    @staticmethod
    def take_ok(flags: Tensor, z: Tensor) -> Tensor:
        """Filter tensor z to only include samples WITHOUT errors."""
        return z[Float64ErrorOps.is_ok(flags)]
    
    @staticmethod
    def take_err(flags: Tensor, z: Tensor) -> Tensor:
        """Filter tensor z to only include samples WITH errors."""
        return z[Float64ErrorOps.is_err(flags)]
    
    @staticmethod
    def partition(flags: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Split tensor z into (ok_z, err_z) based on flags.
        
    NOTE: Uses dynamic shapes. For torch.compile(fullgraph=True),
    use take_ok_p()/take_err_p() instead.
        """
        mask_ok = Float64ErrorOps.is_ok(flags)
        return z[mask_ok], z[~mask_ok]
    
    @staticmethod
    def take_ok_p(flags: Tensor, z: Tensor, fill: float = 0.0) -> Tensor:
        """
        Return z with error samples replaced by fill value. STATIC SHAPE.
        
    Unlike take_ok() which filters to a smaller tensor, take_ok_p() returns the
    same shape with error samples replaced by the fill value. This is
        compatible with torch.compile(fullgraph=True).
        
        Args:
            flags: Error flags tensor (float64)
            z: Any tensor with N as first dimension
            fill: Value to use for error samples (default: 0.0)
        
        Returns:
            Same shape as z, with error samples replaced by fill
        """
        mask_ok = Float64ErrorOps.is_ok(flags)
        mask_exp = Float64ErrorOps.__broadcast_mask(mask_ok, z)
        return torch.where(mask_exp, z, fill)
    
    @staticmethod
    def take_err_p(flags: Tensor, z: Tensor, fill: float = 0.0) -> Tensor:
        """
        Return z with OK samples replaced by fill value. STATIC SHAPE.
        
    Unlike take_err() which filters to a smaller tensor, take_err_p() returns the
    same shape with OK samples replaced by the fill value. This is
        compatible with torch.compile(fullgraph=True).
        
        Args:
            flags: Error flags tensor (float64)
            z: Any tensor with N as first dimension
            fill: Value to use for OK samples (default: 0.0)
        
        Returns:
            Same shape as z, with OK samples replaced by fill
        """
        mask_err = Float64ErrorOps.is_err(flags)
        mask_exp = Float64ErrorOps.__broadcast_mask(mask_err, z)
        return torch.where(mask_exp, z, fill)
    
    @staticmethod
    def partition_many(flags: Tensor, *tensors: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Partition multiple tensors in lockstep based on flags."""
        mask_ok = Float64ErrorOps.is_ok(flags)
        ok = tuple(t[mask_ok] for t in tensors)
        err = tuple(t[~mask_ok] for t in tensors)
        return ok, err
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def all_ok(flags: Tensor) -> Tensor:
        """Single bool: True if every sample is OK."""
        return Float64ErrorOps.is_ok(flags).all()
    
    @staticmethod
    def any_err(flags: Tensor) -> Tensor:
        """Single bool: True if any sample has an error."""
        return Float64ErrorOps.is_err(flags).any()
    
    @staticmethod
    def map_ok(flags: Tensor, z: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
        """Apply fn to z, commit results only for samples WITHOUT errors."""
        mask_ok = Float64ErrorOps.is_ok(flags)
        z_new = fn(z)
        mask_expanded = Float64ErrorOps.__broadcast_mask(mask_ok, z)
        return torch.where(mask_expanded, z_new, z)
    
    @staticmethod
    def map_err(flags: Tensor, z: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
        """Apply fn to z, commit results only for samples WITH errors."""
        mask_err = Float64ErrorOps.is_err(flags)
        z_new = fn(z)
        mask_expanded = Float64ErrorOps.__broadcast_mask(mask_err, z)
        return torch.where(mask_expanded, z_new, z)
    
    @staticmethod
    def map_err_flags(flags: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
        """Transform flags only for samples that currently have errors."""
        mask_err = Float64ErrorOps.is_err(flags)
        flags_new = fn(flags)
        mask_expanded = mask_err.unsqueeze(-1)
        return torch.where(mask_expanded, flags_new, flags)
    
    @staticmethod
    def and_then(flags: Tensor, z: Tensor, fn: Callable[[Tensor], Tuple[Tensor, Tensor]], config: Optional[ErrorConfig] = None) -> Tuple[Tensor, Tensor]:
        """Strict Result-style chaining: only OK samples participate in fn."""
        mask_ok = Float64ErrorOps.is_ok(flags)
        z_new, flags_new = fn(z)
        
        mask_ok_flags = mask_ok.unsqueeze(-1)
        flags_new_masked = torch.where(mask_ok_flags, flags_new, torch.zeros_like(flags_new))
        flags_out = Float64ErrorOps.merge(flags, flags_new_masked, config=config or get_config())
        
        mask_expanded = Float64ErrorOps.__broadcast_mask(mask_ok, z)
        z_out = torch.where(mask_expanded, z_new, z)
        
        return z_out, flags_out
    
    @staticmethod
    def bind(flags: Tensor, z: Tensor, fn: Callable[[Tensor], Tuple[Tensor, Tensor]], config: Optional[ErrorConfig] = None) -> Tuple[Tensor, Tensor]:
        """Monadic bind: apply fn, merge ALL errors, update values only for OK samples."""
        mask_ok = Float64ErrorOps.is_ok(flags)
        z_new, flags_new = fn(z)
        
        flags_out = Float64ErrorOps.merge(flags, flags_new, config=config or get_config())
        
        mask_expanded = Float64ErrorOps.__broadcast_mask(mask_ok, z)
        z_out = torch.where(mask_expanded, z_new, z)
        
        return z_out, flags_out
    
    @staticmethod
    def ensure_mask(flags: Tensor, ok_mask: Tensor, code: int, location: int, severity: int = Severity.ERROR, config: Optional[ErrorConfig] = None) -> Tensor:
        """Push error for samples where ok_mask is False."""
        err_mask = ~ok_mask
        # Use scalar values in torch.where (PyTorch broadcasts automatically)
        code_tensor = torch.where(err_mask, code, ErrorCode.OK).to(torch.int64)
        return Float64ErrorOps.push(flags, code_tensor, location, severity, config or get_config())
    
    @staticmethod
    def guard(flags: Tensor, z: Tensor, pred: Callable[[Tensor], Tensor], code: int, location: int, severity: int = Severity.ERROR, config: Optional[ErrorConfig] = None) -> Tensor:
        """Evaluate pred(z) and push errors where it returns False."""
        ok_mask = pred(z).to(torch.bool)
        return Float64ErrorOps.ensure_mask(flags, ok_mask, code, location, severity, config or get_config())
    
    @staticmethod
    def recover_with_fallback(flags: Tensor, z: Tensor, fallback: Tensor, location: int, severity: int = Severity.WARN, config: Optional[ErrorConfig] = None) -> Tuple[Tensor, Tensor]:
        """Replace error samples with fallback value and mark with FALLBACK_VALUE."""
        mask_err = Float64ErrorOps.is_err(flags)
        fallback_full = fallback.expand_as(z)
        
        mask_expanded = Float64ErrorOps.__broadcast_mask(mask_err, z)
        z_out = torch.where(mask_expanded, fallback_full, z)
        
        # Use scalar values in torch.where (PyTorch broadcasts automatically)
        code_tensor = torch.where(mask_err, ErrorCode.FALLBACK_VALUE, ErrorCode.OK).to(torch.int64)
        flags_out = Float64ErrorOps.push(flags, code_tensor, location, severity, config or get_config())
        
        return z_out, flags_out
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERYING — Use int64 view for bit extraction
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def count_errors(flags: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
        """Count number of non-empty error slots per sample."""
        from ..core.config import get_config
        cfg = config or get_config()
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        int_dtype = cfg.torch_int_dtype
        slots_per_word = cfg.slots_per_word
        
        cache = get_device_cache()
        slot_shifts = cache.get_slot_shifts(device, int_dtype, slots_per_word, SLOT_BITS)
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        non_empty = (slots != 0)
        
        return non_empty.sum(dim=(1, 2)).to(torch.int32)
    
    @staticmethod
    def max_severity(flags: Tensor, config: Optional[ErrorConfig] = None) -> Tensor:
        """Get maximum severity across all slots per sample."""
        from ..core.config import get_config
        cfg = config or get_config()
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        int_dtype = cfg.torch_int_dtype
        slots_per_word = cfg.slots_per_word
        
        cache = get_device_cache()
        slot_shifts = cache.get_slot_shifts(device, int_dtype, slots_per_word, SLOT_BITS)
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        severities = slots & 0x3
        
        return severities.amax(dim=(1, 2))
    
    @staticmethod
    def get_slot(flags: Tensor, slot_idx: int, config: Optional[ErrorConfig] = None) -> Tensor:
        """Get raw slot value at index."""
        flags_i = _as_int(flags)
        slots_per_word = config.slots_per_word
        word_idx = slot_idx // slots_per_word
        bit_offset = (slot_idx % slots_per_word) * SLOT_BITS
        return (flags_i[:, word_idx] >> bit_offset) & SLOT_MASK
    
    @staticmethod
    def get_first_severity(flags: Tensor) -> Tensor:
        """Get severity from slot 0."""
        flags_i = _as_int(flags)
        return (flags_i[:, 0] & 0x3).to(torch.int32)
    
    @staticmethod
    def get_first_code(flags: Tensor) -> Tensor:
        """Get error code from slot 0."""
        flags_i = _as_int(flags)
        return ((flags_i[:, 0] >> CODE_SHIFT) & 0xF).to(torch.int32)
    
    @staticmethod
    def get_first_location(flags: Tensor) -> Tensor:
        """Get location from slot 0."""
        flags_i = _as_int(flags)
        return ((flags_i[:, 0] >> LOCATION_SHIFT) & 0x3FF).to(torch.int32)
    
    @staticmethod
    def clear(flags: Tensor, code: int, config: Optional[ErrorConfig] = None) -> Tensor:
        """Clear (remove) all occurrences of a specific error code from flags."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        int_dtype = config.torch_int_dtype
        slots_per_word = config.slots_per_word
        
        cache = get_device_cache()
        slot_shifts = cache.get_slot_shifts(device, int_dtype, slots_per_word, SLOT_BITS)
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        slot_codes = (slots >> CODE_SHIFT) & 0xF
        should_clear = (slot_codes == code)
        cleared_slots = torch.where(should_clear, torch.zeros(1, dtype=int_dtype, device=device), slots)
        shifted_slots = cleared_slots << slot_shifts
        # Note: sum() promotes int32 to int64, need to cast back
        new_words = shifted_slots.sum(dim=-1).to(int_dtype)
        
        return _as_float(new_words, config or get_config())

