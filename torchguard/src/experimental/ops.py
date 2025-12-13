"""
Float64ErrorOps: Experimental backend using float64 storage + int64 view.

Identical bit layout to stable backend:
    - 64-bit words, 4 slots per word
    - 16-bit slots: [location:10][code:4][severity:2]

Only difference: storage dtype is float64, operations use view(int64).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from ..core.codes import ErrorCode
from ..core.config import DEFAULT_CONFIG, ErrorConfig, Dedupe, Priority, Order
from ..core.constants import CODE_SHIFT, LOCATION_SHIFT, SEVERITY_MASK, SLOT_BITS, SLOT_MASK, SLOTS_PER_WORD
from ..core.severity import Severity


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW HELPERS — The core of the experimental backend
# ═══════════════════════════════════════════════════════════════════════════════

def _as_int(flags: Tensor) -> Tensor:
    """
    View float64 flags as int64 for bitwise operations.
    
    Zero-copy operation — same memory, different dtype interpretation.
    All bitwise operations (&, |, ^, <<, >>) should use this view.
    
    Args:
        flags: (N, num_words) float64 tensor
    
    Returns:
        (N, num_words) int64 tensor (same storage)
    """
    return flags.view(torch.int64)


def _as_float(flags_i: Tensor) -> Tensor:
    """
    View int64 flags as float64 for return.
    
    Zero-copy operation — same memory, different dtype interpretation.
    Use this when returning flags from operations.
    
    Args:
        flags_i: (N, num_words) int64 tensor
    
    Returns:
        (N, num_words) float64 tensor (same storage)
    """
    return flags_i.view(torch.float64)


class Float64ErrorOps:
    """
    Experimental error operations with float64 storage + int64 view.
    
    API is identical to ErrorOps from the stable backend.
    The only difference is internal: float64 storage for AOTAutograd compatibility.
    
    Bit Layout (UNCHANGED from stable backend):
        Storage: float64 tensor, shape (N, num_words), default 4 words = 16 slots.
        View: int64 for all bitwise operations.
        Each 64-bit word holds 4 × 16-bit slots.
        
        Slot (16 bits): [location:10][code:4][severity:2]
        - severity: 0=OK, 1=WARN, 2=ERROR, 3=CRITICAL
        - code: Error type (NaN=1, Inf=2, OutOfBounds=5, etc.)
        - location: Registered location ID (0-1023)
    """
    
    # Underlying dtype for error flags tensor (CHANGED from int64)
    _dtype = torch.float64
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS - Bit Packing (identical to stable, but uses views)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def __pack_slot(code: int, location: int, severity: int) -> int:
        """Pack code, location, severity into 16-bit slot value."""
        return (severity & 0x3) | ((code & 0xF) << CODE_SHIFT) | ((location & 0x3FF) << LOCATION_SHIFT)
    
    @staticmethod
    def __pack_slot_tensor(code: Tensor, location: int, severity: int) -> Tensor:
        """Pack code tensor with location and severity. Compilable."""
        return severity | ((code.to(torch.int64) & 0xF) << CODE_SHIFT) | (location << LOCATION_SHIFT)
    
    @staticmethod
    def __broadcast_mask(mask: Tensor, z: Tensor) -> Tensor:
        """Broadcast (N,) bool mask to match z's shape (N, d1, d2, ...)."""
        if z.ndim == 1:
            return mask
        shape = (mask.shape[0],) + (1,) * (z.ndim - 1)
        return mask.view(shape).expand_as(z)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS - Vectorized Slot Operations (use int64 view)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def __extract_all_slots(flags: Tensor, config: ErrorConfig) -> Tensor:
        """
        Extract all slots into (N, num_slots) tensor. Fully vectorized.
        Uses int64 view for bitwise operations.
        """
        flags_i = _as_int(flags)
        n, num_words = flags_i.shape
        device = flags_i.device
        
        shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
        expanded = flags_i.unsqueeze(-1) >> shifts
        masked = expanded & SLOT_MASK
        
        return masked.reshape(n, -1)[:, :config.num_slots]
    
    @staticmethod
    def __pack_all_slots(slots: Tensor, config: ErrorConfig) -> Tensor:
        """
        Pack (N, num_slots) slots back into (N, num_words) float64 flags.
        """
        n = slots.shape[0]
        num_words = config.num_words
        device = slots.device
        
        total_slot_capacity = num_words * SLOTS_PER_WORD
        if slots.shape[1] < total_slot_capacity:
            padded = torch.zeros(n, total_slot_capacity, dtype=torch.int64, device=device)
            padded[:, :slots.shape[1]] = slots
        else:
            padded = slots[:, :total_slot_capacity]
        
        reshaped = padded.reshape(n, num_words, SLOTS_PER_WORD)
        shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
        shifted = reshaped << shifts
        flags_i = shifted.sum(dim=-1)
        
        return _as_float(flags_i)
    
    @staticmethod
    def __lifo_shift_one(flags: Tensor, new_slot: Tensor) -> Tensor:
        """
        LIFO shift: shift all slots right by one, insert new_slot at position 0.
        Returns float64.
        """
        flags_i = _as_int(flags)
        n, num_words = flags_i.shape
        KEEP_MASK = (1 << (64 - SLOT_BITS)) - 1
        
        carry = (flags_i >> (64 - SLOT_BITS)) & SLOT_MASK
        shifted = (flags_i & KEEP_MASK) << SLOT_BITS
        
        result = shifted.clone()
        result[:, 0] = result[:, 0] | new_slot
        if num_words > 1:
            result[:, 1:] = result[:, 1:] | carry[:, :-1]
        
        return _as_float(result)
    
    @staticmethod
    def __find_slot_matching(flags: Tensor, pred_fn, config: ErrorConfig) -> Tuple[Tensor, Tensor]:
        """Find first slot matching a predicate. Uses int64 view."""
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config)
        matches = pred_fn(all_slots)
        exists = matches.any(dim=1)
        
        match_scores = matches.float()
        positions = torch.arange(config.num_slots, device=flags.device, dtype=torch.float32)
        scores = match_scores - positions * 1e-6
        scores = torch.where(matches, scores, torch.tensor(-float('inf'), device=flags.device))
        slot_idx = scores.argmax(dim=1)
        
        return exists, slot_idx
    
    @staticmethod
    def __replace_slot_at(flags: Tensor, slot_idx: Tensor, new_slot: Tensor, config: ErrorConfig) -> Tensor:
        """Replace slot at given index with new value. Returns float64."""
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config)
        n, num_slots = all_slots.shape
        
        indices = torch.arange(num_slots, device=flags.device).unsqueeze(0)
        is_target = (indices == slot_idx.unsqueeze(1))
        updated = torch.where(is_target, new_slot.unsqueeze(1), all_slots)
        
        return Float64ErrorOps.__pack_all_slots(updated, config)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CREATION — Return float64 tensors
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def new(x: Tensor) -> Tensor:
        """Create empty error flags from a reference tensor. Returns float64."""
        return torch.zeros(x.shape[0], DEFAULT_CONFIG.num_words, dtype=torch.float64, device=x.device)
    
    @staticmethod
    def new_t(n: int, device: Optional[torch.device] = None, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Create empty error flags with explicit arguments. Returns float64."""
        return torch.zeros(n, config.num_words, dtype=torch.float64, device=device)
    
    @staticmethod
    def from_code(code: int, location: int, n: int, device: Optional[torch.device] = None, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Create flags with a single error for all samples. Returns float64."""
        flags = Float64ErrorOps.new_t(n, device, config)
        packed = Float64ErrorOps.__pack_slot(code, location, severity)
        flags_i = _as_int(flags)
        flags_i[:, 0] = packed
        return _as_float(flags_i)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RECORDING - Push Methods (use int64 view for bitwise ops)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def push(flags: Tensor, code: Tensor, location: int, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Push new error into flags. Returns float64."""
        acc = config.accumulation
        
        if acc.dedupe == Dedupe.LOCATION:
            return Float64ErrorOps.__push_dedupe_location(flags, code, location, severity, config)
        elif acc.dedupe == Dedupe.CODE:
            return Float64ErrorOps.__push_dedupe_code(flags, code, location, severity, config)
        elif acc.dedupe == Dedupe.UNIQUE:
            return Float64ErrorOps.__push_dedupe_unique(flags, code, location, severity, config)
        else:
            return Float64ErrorOps.__push_no_dedupe(flags, code, location, severity, config)
    
    @staticmethod
    def __push_no_dedupe(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Push without deduplication."""
        acc = config.accumulation
        if acc.priority == Priority.CHRONO:
            if acc.order == Order.LAST:
                return Float64ErrorOps.__push_chrono_last(flags, code, location, severity, config)
            else:
                return Float64ErrorOps.__push_chrono_first(flags, code, location, severity, config)
        elif acc.priority == Priority.SEVERITY:
            return Float64ErrorOps.__push_severity_based(flags, code, location, severity, config)
        else:
            return Float64ErrorOps.__push_chrono_last(flags, code, location, severity, config)
    
    @staticmethod
    def __push_chrono_last(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """LIFO push. Returns float64."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity)
        new_flags = Float64ErrorOps.__lifo_shift_one(flags, new_slot)
        return torch.where(should_push.unsqueeze(-1), new_flags, flags)
    
    @staticmethod
    def __push_chrono_first(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """FIFO push. Returns float64."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity)
        error_count = Float64ErrorOps.count_errors(flags, config).to(torch.int64)
        has_space = error_count < config.num_slots
        should_push = should_push & has_space
        result = Float64ErrorOps.__replace_slot_at(flags, error_count, new_slot, config)
        return torch.where(should_push.unsqueeze(-1), result, flags)
    
    @staticmethod
    def __push_severity_based(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Severity-priority push. Returns float64."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity)
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config)
        severities = all_slots & SEVERITY_MASK
        min_slot_idx = severities.argmin(dim=1)
        min_sev = severities.gather(1, min_slot_idx.unsqueeze(1)).squeeze(1)
        should_replace = should_push & (severity > min_sev)
        result = Float64ErrorOps.__replace_slot_at(flags, min_slot_idx, new_slot, config)
        return torch.where(
            should_replace.unsqueeze(-1), result,
            torch.where(should_push.unsqueeze(-1), Float64ErrorOps.__push_chrono_last(flags, code, location, severity, config), flags)
        )
    
    @staticmethod
    def __push_dedupe_location(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Location-dedupe push. Returns float64."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity)
        
        def match_location(slots):
            slot_loc = (slots >> LOCATION_SHIFT) & 0x3FF
            return (slot_loc == location) & (slots != 0)
        
        loc_exists, existing_slot_idx = Float64ErrorOps.__find_slot_matching(flags, match_location, config)
        
        return torch.where(
            (loc_exists & should_push).unsqueeze(-1),
            Float64ErrorOps.__update_slot_if_worse(flags, existing_slot_idx, new_slot, config),
            torch.where((~loc_exists & should_push).unsqueeze(-1), Float64ErrorOps.__push_no_dedupe(flags, code, location, severity, config), flags)
        )
    
    @staticmethod
    def __push_dedupe_code(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Code-dedupe push. Returns float64."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity)
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config)
        slot_codes = (all_slots >> CODE_SHIFT) & 0xF
        matches = (slot_codes == code.unsqueeze(1)) & (all_slots != 0)
        code_exists = matches.any(dim=1)
        
        match_scores = matches.float()
        positions = torch.arange(config.num_slots, device=flags.device, dtype=torch.float32)
        scores = match_scores - positions * 1e-6
        scores = torch.where(matches, scores, torch.tensor(-float('inf'), device=flags.device))
        existing_slot_idx = scores.argmax(dim=1)
        
        return torch.where(
            (code_exists & should_push).unsqueeze(-1),
            Float64ErrorOps.__update_slot_if_worse(flags, existing_slot_idx, new_slot, config),
            torch.where((~code_exists & should_push).unsqueeze(-1), Float64ErrorOps.__push_no_dedupe(flags, code, location, severity, config), flags)
        )
    
    @staticmethod
    def __push_dedupe_unique(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Unique-dedupe push. Returns float64."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = Float64ErrorOps.__pack_slot_tensor(code, location, severity)
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config)
        slot_loc = (all_slots >> LOCATION_SHIFT) & 0x3FF
        slot_codes = (all_slots >> CODE_SHIFT) & 0xF
        matches = (slot_loc == location) & (slot_codes == code.unsqueeze(1)) & (all_slots != 0)
        pair_exists = matches.any(dim=1)
        
        match_scores = matches.float()
        positions = torch.arange(config.num_slots, device=flags.device, dtype=torch.float32)
        scores = match_scores - positions * 1e-6
        scores = torch.where(matches, scores, torch.tensor(-float('inf'), device=flags.device))
        existing_slot_idx = scores.argmax(dim=1)
        
        return torch.where(
            (pair_exists & should_push).unsqueeze(-1),
            Float64ErrorOps.__update_slot_if_worse(flags, existing_slot_idx, new_slot, config),
            torch.where((~pair_exists & should_push).unsqueeze(-1), Float64ErrorOps.__push_no_dedupe(flags, code, location, severity, config), flags)
        )
    
    @staticmethod
    def __update_slot_if_worse(flags: Tensor, slot_idx: Tensor, new_slot: Tensor, config: ErrorConfig) -> Tensor:
        """Update slot only if new error is worse. Returns float64."""
        all_slots = Float64ErrorOps.__extract_all_slots(flags, config)
        n, num_slots = all_slots.shape
        indices = torch.arange(num_slots, device=flags.device).unsqueeze(0)
        is_target = (indices == slot_idx.unsqueeze(1))
        new_sev = (new_slot & SEVERITY_MASK).unsqueeze(1)
        existing_sev = all_slots & SEVERITY_MASK
        should_update = is_target & (new_sev > existing_sev)
        updated = torch.where(should_update, new_slot.unsqueeze(1), all_slots)
        return Float64ErrorOps.__pack_all_slots(updated, config)
    
    @staticmethod
    def push_scalar(flags: Tensor, code: int, location: int, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Push same error to all samples. Returns float64."""
        if code == ErrorCode.OK or severity == Severity.OK:
            return flags
        n = flags.shape[0]
        code_tensor = torch.full((n,), code, dtype=torch.int64, device=flags.device)
        return Float64ErrorOps.push(flags, code_tensor, location, severity, config)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RECORDING - Merge Methods
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def merge(*flag_tensors: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Merge multiple flag tensors into one. Returns float64."""
        if not flag_tensors:
            raise ValueError("Need at least one flag tensor")
        if len(flag_tensors) == 1:
            return flag_tensors[0]
        
        result = flag_tensors[0]
        for other in flag_tensors[1:]:
            result = Float64ErrorOps.__merge_two(result, other, config)
        return result
    
    @staticmethod
    def __merge_two(flags: Tensor, other: Tensor, config: ErrorConfig) -> Tensor:
        """Merge errors from other into flags. Returns float64."""
        flags_slots = Float64ErrorOps.__extract_all_slots(flags, config)
        other_slots = Float64ErrorOps.__extract_all_slots(other, config)
        combined = torch.cat([other_slots, flags_slots], dim=1)
        non_empty = (combined & SEVERITY_MASK) != 0
        sort_key = (~non_empty).long()
        _, indices = torch.sort(sort_key, dim=1, stable=True)
        compacted = torch.gather(combined, 1, indices)
        return Float64ErrorOps.__pack_all_slots(compacted[:, :config.num_slots], config)
    
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
    def has_code(flags: Tensor, code: int, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Check if any slot contains specific error code."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        slot_codes = (slots >> CODE_SHIFT) & 0xF
        slot_sev = slots & 0x3
        matches = (slot_codes == code) & (slot_sev != 0)
        
        return matches.any(dim=(1, 2))
    
    @staticmethod
    def has_nan(flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Check if any slot has NaN error."""
        return Float64ErrorOps.has_code(flags, ErrorCode.NAN, config)
    
    @staticmethod
    def has_inf(flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Check if any slot has Inf error."""
        return Float64ErrorOps.has_code(flags, ErrorCode.INF, config)
    
    @staticmethod
    def has_critical(flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Check if any slot has CRITICAL severity."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        severities = slots & 0x3
        
        return (severities == Severity.CRITICAL).any(dim=(1, 2))
    
    @staticmethod
    def has_fallback(flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Check if any slot indicates fallback value was used."""
        return Float64ErrorOps.has_code(flags, ErrorCode.FALLBACK_VALUE, config)
    
    @staticmethod
    def has_domain(flags: Tensor, domain: int, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Check if any slot has error in given domain."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
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
    def and_then(flags: Tensor, z: Tensor, fn: Callable[[Tensor], Tuple[Tensor, Tensor]], config: ErrorConfig = DEFAULT_CONFIG) -> Tuple[Tensor, Tensor]:
        """Strict Result-style chaining: only OK samples participate in fn."""
        mask_ok = Float64ErrorOps.is_ok(flags)
        z_new, flags_new = fn(z)
        
        mask_ok_flags = mask_ok.unsqueeze(-1)
        flags_new_masked = torch.where(mask_ok_flags, flags_new, torch.zeros_like(flags_new))
        flags_out = Float64ErrorOps.merge(flags, flags_new_masked, config=config)
        
        mask_expanded = Float64ErrorOps.__broadcast_mask(mask_ok, z)
        z_out = torch.where(mask_expanded, z_new, z)
        
        return z_out, flags_out
    
    @staticmethod
    def bind(flags: Tensor, z: Tensor, fn: Callable[[Tensor], Tuple[Tensor, Tensor]], config: ErrorConfig = DEFAULT_CONFIG) -> Tuple[Tensor, Tensor]:
        """Monadic bind: apply fn, merge ALL errors, update values only for OK samples."""
        mask_ok = Float64ErrorOps.is_ok(flags)
        z_new, flags_new = fn(z)
        
        flags_out = Float64ErrorOps.merge(flags, flags_new, config=config)
        
        mask_expanded = Float64ErrorOps.__broadcast_mask(mask_ok, z)
        z_out = torch.where(mask_expanded, z_new, z)
        
        return z_out, flags_out
    
    @staticmethod
    def ensure_mask(flags: Tensor, ok_mask: Tensor, code: int, location: int, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Push error for samples where ok_mask is False."""
        err_mask = ~ok_mask
        code_tensor = torch.where(
            err_mask,
            torch.full((flags.shape[0],), code, dtype=torch.int64, device=flags.device),
            torch.full((flags.shape[0],), ErrorCode.OK, dtype=torch.int64, device=flags.device),
        )
        return Float64ErrorOps.push(flags, code_tensor, location, severity, config)
    
    @staticmethod
    def guard(flags: Tensor, z: Tensor, pred: Callable[[Tensor], Tensor], code: int, location: int, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Evaluate pred(z) and push errors where it returns False."""
        ok_mask = pred(z).to(torch.bool)
        return Float64ErrorOps.ensure_mask(flags, ok_mask, code, location, severity, config)
    
    @staticmethod
    def recover_with_fallback(flags: Tensor, z: Tensor, fallback: Tensor, location: int, severity: int = Severity.WARN, config: ErrorConfig = DEFAULT_CONFIG) -> Tuple[Tensor, Tensor]:
        """Replace error samples with fallback value and mark with FALLBACK_VALUE."""
        mask_err = Float64ErrorOps.is_err(flags)
        fallback_full = fallback.expand_as(z)
        
        mask_expanded = Float64ErrorOps.__broadcast_mask(mask_err, z)
        z_out = torch.where(mask_expanded, fallback_full, z)
        
        code_tensor = torch.where(
            mask_err,
            torch.full((flags.shape[0],), ErrorCode.FALLBACK_VALUE, dtype=torch.int64, device=flags.device),
            torch.full((flags.shape[0],), ErrorCode.OK, dtype=torch.int64, device=flags.device),
        )
        flags_out = Float64ErrorOps.push(flags, code_tensor, location, severity, config)
        
        return z_out, flags_out
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERYING — Use int64 view for bit extraction
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def count_errors(flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Count number of non-empty error slots per sample."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        non_empty = (slots != 0)
        
        return non_empty.sum(dim=(1, 2)).to(torch.int32)
    
    @staticmethod
    def max_severity(flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Get maximum severity across all slots per sample."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        severities = slots & 0x3
        
        return severities.amax(dim=(1, 2))
    
    @staticmethod
    def get_slot(flags: Tensor, slot_idx: int) -> Tensor:
        """Get raw slot value at index."""
        flags_i = _as_int(flags)
        word_idx = slot_idx // SLOTS_PER_WORD
        bit_offset = (slot_idx % SLOTS_PER_WORD) * SLOT_BITS
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
    def clear(flags: Tensor, code: int, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
        """Clear (remove) all occurrences of a specific error code from flags. Returns float64."""
        flags_i = _as_int(flags)
        N, num_words = flags_i.shape
        device = flags_i.device
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
        words = flags_i.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        slot_codes = (slots >> CODE_SHIFT) & 0xF
        should_clear = (slot_codes == code)
        cleared_slots = torch.where(should_clear, torch.zeros(1, dtype=torch.int64, device=device), slots)
        shifted_slots = cleared_slots << slot_shifts
        new_words = shifted_slots.sum(dim=-1)
        
        return _as_float(new_words)

