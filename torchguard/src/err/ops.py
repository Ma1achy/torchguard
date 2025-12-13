"""
error_t: Primary API for compiled error flags.

This is the main user-facing API for error handling within torch.compile() regions.
All methods are static and work with torch.compile(fullgraph=True).

For Python boundary operations (unpacking, debugging), use ErrorFlags directly.

Bit Layout:
    Error flags are stored as int64 tensors with shape (N, num_words).
    Each 64-bit word contains 4 slots of 16 bits each.
    
    Slot layout (16 bits, LSB to MSB):
        +------------+----------+----------+
        | bits 15-6  | bits 5-2 | bits 1-0 |
        | location   | code     | severity |
        | (10 bits)  | (4 bits) | (2 bits) |
        +------------+----------+----------+
    
    - severity: 0=OK, 1=WARN, 2=ERROR, 3=CRITICAL
    - code: 4-bit error code (NaN, Inf, OutOfBounds, etc.)
    - location: 10-bit location ID (0-1023)
    
    Default config: 64 words × 4 slots/word = 256 error slots per sample.

    Usage:
    from . import error_t, ErrorCode, Severity
    
    @torch.compile
    def forward(x: Tensor) -> tuple[Tensor, error_t]:
        flags = ErrorOps.new(x)
        nan_mask = torch.isnan(x).any(dim=-1)
        codes = torch.where(nan_mask, ErrorCode.NAN, ErrorCode.OK)
        flags = ErrorOps.push(flags, codes, location=42)
        ok_output = ErrorOps.take_ok(flags, output)
        return output, flags
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
from torch import Tensor

from ..core.codes import ErrorCode
from ..core.config import DEFAULT_CONFIG, Dedupe, ErrorConfig, Order, Priority
from ..core.constants import CODE_SHIFT, LOCATION_SHIFT, SEVERITY_MASK, SLOT_BITS, SLOT_MASK, SLOTS_PER_WORD
from ..core.severity import Severity

if TYPE_CHECKING:
    from ..typing import int64_t, bool_t, int32_t


class ErrorOps:
    """
    Primary API for compiled error flags.
    
    Use this class for all error handling within torch.compile() regions.
    All methods are static and work with torch.compile(fullgraph=True).
    
    This class also serves as a dtype alias for type annotations:
        Tensor[error_t, ("N",)]
    
    Bit Layout:
        Storage: int64 tensor, shape (N, num_words), default 64 words.
        Each 64-bit word holds 4 × 16-bit slots.
        
        Slot (16 bits): [location:10][code:4][severity:2]
        - severity: 0=OK, 1=WARN, 2=ERROR, 3=CRITICAL
        - code: Error type (NaN=1, Inf=2, OutOfBounds=4, etc.)
        - location: Registered location ID (0-1023)
        
        Fast check: (flags != 0).any(dim=-1) detects any error.
    
    Categories:
        Creation:     new(), new_t(), from_code()
        Recording:    push(), push_scalar(), merge()
        Checking:     is_ok(), is_err(), has_nan(), has_inf(), has_critical(), ...
        Filtering:    get_ok(), get_err(), take_ok(), take_err(), partition(), partition_many()
        Combinators:  map_ok(), map_err(), and_then(), bind(), guard(), recover_with_fallback()
        Querying:     count_errors(), max_severity(), get_first_*()
    
    For Python boundary operations (unpacking, converting to Error objects,
    debugging), use ErrorFlags directly:
        from . import ErrorFlags
        errors = ErrorFlags.to_errors(flags)
        ErrorFlags.pretty_print(flags)
    
    Type hint usage:
        def forward(self, x: Tensor) -> tuple[Tensor, error_t]:
            flags = ErrorOps.new(x)
            return output, flags
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DTYPE ALIAS - For type annotations like Tensor[error_t, ("N",)]
    # ═══════════════════════════════════════════════════════════════════════════
    
    _dtype = torch.int64  # Underlying dtype for error flags tensor
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS - Bit Packing/Extraction
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
    def __extract_severity(slot: Tensor) -> Tensor:
        """Extract severity from slot (bits 0-1)."""
        return slot & 0x3
    
    @staticmethod
    def __extract_code(slot: Tensor) -> Tensor:
        """Extract code from slot (bits 2-5)."""
        return (slot >> CODE_SHIFT) & 0xF
    
    @staticmethod
    def __extract_location(slot: Tensor) -> Tensor:
        """Extract location from slot (bits 6-15)."""
        return (slot >> LOCATION_SHIFT) & 0x3FF
    
    @staticmethod
    def __broadcast_mask(mask: Tensor, z: Tensor) -> Tensor:
        """Broadcast (N,) bool mask to match z's shape (N, d1, d2, ...)."""
        if z.ndim == 1:
            return mask
        shape = (mask.shape[0],) + (1,) * (z.ndim - 1)
        return mask.view(shape).expand_as(z)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS - Vectorized Slot Operations (torch.compile friendly)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def __extract_all_slots(flags: Tensor, config: ErrorConfig) -> Tensor:
        """
        Extract all slots into (N, num_slots) tensor. Fully vectorized.
        
        Uses broadcasting to extract all slots simultaneously without Python loops.
        """
        n, num_words = flags.shape
        device = flags.device
        
        # Shift amounts for each slot position within a word: [0, 16, 32, 48]
        shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
        
        # Broadcast: (n, num_words, 1) >> (4,) -> (n, num_words, 4)
        expanded = flags.unsqueeze(-1) >> shifts
        masked = expanded & SLOT_MASK
        
        # Reshape to (n, num_words * 4) and truncate to num_slots
        return masked.reshape(n, -1)[:, :config.num_slots]
    
    @staticmethod
    def __pack_all_slots(slots: Tensor, config: ErrorConfig) -> Tensor:
        """
        Pack (N, num_slots) slots back into (N, num_words) flags. Fully vectorized.
        
        Uses broadcasting to pack all slots simultaneously without Python loops.
        """
        n = slots.shape[0]
        num_words = config.num_words
        device = slots.device
        
        # Pad to full words if needed: (n, num_words * SLOTS_PER_WORD)
        total_slot_capacity = num_words * SLOTS_PER_WORD
        if slots.shape[1] < total_slot_capacity:
            padded = torch.zeros(n, total_slot_capacity, dtype=torch.int64, device=device)
            padded[:, :slots.shape[1]] = slots
        else:
            padded = slots[:, :total_slot_capacity]
        
        # Reshape to (n, num_words, SLOTS_PER_WORD)
        reshaped = padded.reshape(n, num_words, SLOTS_PER_WORD)
        
        # Shift each slot to its bit position within the word
        shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=torch.int64) * SLOT_BITS
        shifted = reshaped << shifts  # (n, num_words, 4)
        
        # Combine slots within each word using bitwise OR via sum (slots don't overlap)
        return shifted.sum(dim=-1)
    
    @staticmethod
    def __lifo_shift_one(flags: Tensor, new_slot: Tensor) -> Tensor:
        """
        LIFO shift: shift all slots right by one, insert new_slot at position 0.
        Fully vectorized - no Python loops.
        
        Args:
            flags: (N, num_words) current flags
            new_slot: (N,) new slot value to insert at position 0
        
        Returns:
            (N, num_words) with all slots shifted right, new_slot at slot 0
        """
        n, num_words = flags.shape
        device = flags.device
        KEEP_MASK = (1 << (64 - SLOT_BITS)) - 1
        
        # Extract carry bits (top SLOT_BITS of each word)
        carry = (flags >> (64 - SLOT_BITS)) & SLOT_MASK  # (n, num_words)
        
        # Shift all words left by SLOT_BITS (makes room at the bottom)
        shifted = (flags & KEEP_MASK) << SLOT_BITS  # (n, num_words)
        
        # Build result: 
        # - word[0] gets new_slot in lowest SLOT_BITS
        # - word[i>0] gets carry from word[i-1] in lowest SLOT_BITS
        result = shifted.clone()
        result[:, 0] = result[:, 0] | new_slot
        if num_words > 1:
            result[:, 1:] = result[:, 1:] | carry[:, :-1]
        
        return result
    
    @staticmethod
    def __find_slot_matching(flags: Tensor, pred_fn, config: ErrorConfig) -> Tuple[Tensor, Tensor]:
        """
        Find first slot matching a predicate. Fully vectorized.
        
        Args:
            flags: (N, num_words) flags tensor
            pred_fn: Function(slots) -> (N, num_slots) bool mask
            config: ErrorConfig
        
        Returns:
            (exists, slot_idx): both (N,) tensors
            - exists: True if any slot matches
            - slot_idx: index of first matching slot (0 if none)
        """
        # Extract all slots: (N, num_slots)
        all_slots = ErrorOps.__extract_all_slots(flags, config)
        
        # Apply predicate: (N, num_slots) bool
        matches = pred_fn(all_slots)
        
        # Find first match per sample
        # Use argmax on float tensor - first True becomes 1.0, argmax finds it
        # If no match, argmax returns 0, so we also need exists mask
        exists = matches.any(dim=1)  # (N,)
        
        # For finding first True: convert to float, argmax
        # But we need to handle "no match" case - use large negative for non-matches
        match_scores = matches.float()
        # Add position penalty to prefer earlier slots when multiple match
        positions = torch.arange(config.num_slots, device=flags.device, dtype=torch.float32)
        # Score: 1.0 for match, 0.0 for non-match, minus tiny position penalty
        scores = match_scores - positions * 1e-6
        scores = torch.where(matches, scores, torch.tensor(-float('inf'), device=flags.device))
        slot_idx = scores.argmax(dim=1)  # (N,)
        
        return exists, slot_idx
    
    @staticmethod  
    def __replace_slot_at(flags: Tensor, slot_idx: Tensor, new_slot: Tensor, config: ErrorConfig) -> Tensor:
        """
        Replace slot at given index with new value. Fully vectorized.
        
        Args:
            flags: (N, num_words) flags tensor
            slot_idx: (N,) index of slot to replace per sample
            new_slot: (N,) new slot value
            config: ErrorConfig
        
        Returns:
            (N, num_words) with slot replaced
        """
        # Extract all slots
        all_slots = ErrorOps.__extract_all_slots(flags, config)  # (N, num_slots)
        n, num_slots = all_slots.shape
        
        # Create index mask for the target slot
        indices = torch.arange(num_slots, device=flags.device).unsqueeze(0)  # (1, num_slots)
        is_target = (indices == slot_idx.unsqueeze(1))  # (N, num_slots)
        
        # Replace target slot
        updated = torch.where(is_target, new_slot.unsqueeze(1), all_slots)
        
        # Pack back
        return ErrorOps.__pack_all_slots(updated, config)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CREATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def new(x: Tensor) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Create empty error flags from a reference tensor.
        
        Extracts batch size and device from x automatically.
        All flags are initialized to zero (no errors).
        This is the preferred way to create flags in compiled code.
        
        Bit Layout:
            Returns int64 tensor of zeros, shape (x.shape[0], 64).
            Zero value = no errors (all slots empty).
        
        Args:
            x (Tensor): Reference tensor to get batch size (x.shape[0]) and device from
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Empty error flags tensor
        
        Example:
            >>> flags = ErrorOps.new(x)  # Shape: (batch_size, 64)
        """
        return torch.zeros(x.shape[0], DEFAULT_CONFIG.num_words, dtype=torch.int64, device=x.device)
    
    @staticmethod
    def new_t(n: int, device: Optional[torch.device] = None, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Create empty error flags with explicit arguments.
        
        Use when you need explicit control over batch size, device, or config.
        For most cases, prefer ErrorOps.new(x) instead.
        
        Bit Layout:
            Returns int64 tensor of zeros, shape (n, config.num_words).
        
        Args:
            n (int): Number of samples (batch size)
            device (Optional[torch.device]): Device for tensor (None for CPU)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Empty error flags tensor
        
        Example:
            >>> flags = ErrorOps.new_t(batch_size, device=x.device)
        """
        return torch.zeros(n, config.num_words, dtype=torch.int64, device=device)
    
    @staticmethod
    def from_code(code: int, location: int, n: int, device: Optional[torch.device] = None, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Create flags with a single error for all samples.
        
        Useful for creating test fixtures or initial error states.
        
        Bit Layout:
            Creates tensor with packed slot in word 0, bits 0-15:
            slot = severity | (code << 2) | (location << 6)
        
        Args:
            code (int): Error code value (from ErrorCode, 4 bits)
            location (int): Error location ID (10 bits, 0-1023)
            n (int): Number of samples
            device (Optional[torch.device]): Device for tensor
            severity (int): Severity level (2 bits: OK=0, WARN=1, ERROR=2, CRITICAL=3)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Error flags with error in slot 0
        
        Example:
            >>> flags = ErrorOps.from_code(ErrorCode.NAN, loc_id, batch_size)
        """
        flags = ErrorOps.new_t(n, device, config)
        packed = ErrorOps.__pack_slot(code, location, severity)
        flags[:, 0] = packed
        return flags
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RECORDING - Push Methods
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def push(flags: Tensor[int64_t, ("N", "num_words")], code: Tensor[int64_t, ("N",)], location: int, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Push new error into flags. Fully compilable.
        
        Only pushes where code != 0 and severity != OK.
        Accumulation behavior is controlled by config.accumulation.
        
        Bit Layout:
            Packs new slot: severity | (code << 2) | (location << 6)
            Default behavior (LIFO): shifts existing slots right, inserts at slot 0.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Existing error flags tensor
            code (Tensor[int64_t, (N,)]): Error code tensor (per-sample)
            location (int): Error location ID (same for all samples, 10 bits)
            severity (int): Severity level (2 bits, default: ERROR=2)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Updated error flags tensor
        
        Example:
            >>> nan_mask = torch.isnan(x).any(dim=-1)
            >>> codes = torch.where(nan_mask, ErrorCode.NAN, ErrorCode.OK)
            >>> flags = ErrorOps.push(flags, codes, location=loc_id)
        """
        acc = config.accumulation
        
        if acc.dedupe == Dedupe.LOCATION:
            return ErrorOps.__push_dedupe_location(flags, code, location, severity, config)
        elif acc.dedupe == Dedupe.CODE:
            return ErrorOps.__push_dedupe_code(flags, code, location, severity, config)
        elif acc.dedupe == Dedupe.UNIQUE:
            return ErrorOps.__push_dedupe_unique(flags, code, location, severity, config)
        else:  # Dedupe.NONE
            return ErrorOps.__push_no_dedupe(flags, code, location, severity, config)
    
    @staticmethod
    def __push_no_dedupe(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Push without deduplication. Uses Priority+Order to determine slot placement."""
        acc = config.accumulation
        if acc.priority == Priority.CHRONO:
            if acc.order == Order.LAST:
                return ErrorOps.__push_chrono_last(flags, code, location, severity, config)
            else:
                return ErrorOps.__push_chrono_first(flags, code, location, severity, config)
        elif acc.priority == Priority.SEVERITY:
            return ErrorOps.__push_severity_based(flags, code, location, severity, config)
        else:
            return ErrorOps.__push_chrono_last(flags, code, location, severity, config)
    
    @staticmethod
    def __push_chrono_last(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """LIFO push: Shift all slots right, insert new error at slot 0. Fully vectorized."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = ErrorOps.__pack_slot_tensor(code, location, severity)
        
        # Use vectorized LIFO shift
        new_flags = ErrorOps.__lifo_shift_one(flags, new_slot)
        
        return torch.where(should_push.unsqueeze(-1), new_flags, flags)
    
    @staticmethod
    def __push_chrono_first(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """FIFO push: Keep first N errors (root cause preservation). Fully vectorized."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = ErrorOps.__pack_slot_tensor(code, location, severity)
        
        # Count existing errors to find insertion point
        error_count = ErrorOps.count_errors(flags, config).to(torch.int64)
        has_space = error_count < config.num_slots
        should_push = should_push & has_space
        
        # Use vectorized slot replacement at the error_count position
        result = ErrorOps.__replace_slot_at(flags, error_count, new_slot, config)
        
        return torch.where(should_push.unsqueeze(-1), result, flags)
    
    @staticmethod
    def __push_severity_based(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Severity-priority push. Replace lowest severity slot if new is worse. Fully vectorized."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = ErrorOps.__pack_slot_tensor(code, location, severity)
        
        # Extract all slots and find minimum severity
        all_slots = ErrorOps.__extract_all_slots(flags, config)  # (N, num_slots)
        severities = all_slots & SEVERITY_MASK  # (N, num_slots)
        
        # Find slot with minimum severity (argmin)
        min_slot_idx = severities.argmin(dim=1)  # (N,)
        min_sev = severities.gather(1, min_slot_idx.unsqueeze(1)).squeeze(1)  # (N,)
        
        should_replace = should_push & (severity > min_sev)
        result = ErrorOps.__replace_slot_at(flags, min_slot_idx, new_slot, config)
        
        return torch.where(
            should_replace.unsqueeze(-1), result,
            torch.where(should_push.unsqueeze(-1), ErrorOps.__push_chrono_last(flags, code, location, severity, config), flags)
        )
    
    @staticmethod
    def __push_dedupe_location(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Location-dedupe push. One entry per location. Fully vectorized."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = ErrorOps.__pack_slot_tensor(code, location, severity)
        
        # Find existing slot with same location using vectorized helper
        def match_location(slots):
            slot_loc = (slots >> LOCATION_SHIFT) & 0x3FF
            return (slot_loc == location) & (slots != 0)
        
        loc_exists, existing_slot_idx = ErrorOps.__find_slot_matching(flags, match_location, config)
        
        return torch.where(
            (loc_exists & should_push).unsqueeze(-1),
            ErrorOps.__update_slot_if_worse(flags, existing_slot_idx, new_slot, config),
            torch.where((~loc_exists & should_push).unsqueeze(-1), ErrorOps.__push_no_dedupe(flags, code, location, severity, config), flags)
        )
    
    @staticmethod
    def __push_dedupe_code(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Code-dedupe push. One entry per error code. Fully vectorized."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = ErrorOps.__pack_slot_tensor(code, location, severity)
        
        # Extract all slots and check for matching code (per-sample)
        all_slots = ErrorOps.__extract_all_slots(flags, config)  # (N, num_slots)
        slot_codes = (all_slots >> CODE_SHIFT) & 0xF  # (N, num_slots)
        
        # code is (N,), need to compare per-sample
        matches = (slot_codes == code.unsqueeze(1)) & (all_slots != 0)  # (N, num_slots)
        
        # Find first match
        code_exists = matches.any(dim=1)  # (N,)
        # Use argmax with masking for first match index
        match_scores = matches.float()
        positions = torch.arange(config.num_slots, device=flags.device, dtype=torch.float32)
        scores = match_scores - positions * 1e-6
        scores = torch.where(matches, scores, torch.tensor(-float('inf'), device=flags.device))
        existing_slot_idx = scores.argmax(dim=1)  # (N,)
        
        return torch.where(
            (code_exists & should_push).unsqueeze(-1),
            ErrorOps.__update_slot_if_worse(flags, existing_slot_idx, new_slot, config),
            torch.where((~code_exists & should_push).unsqueeze(-1), ErrorOps.__push_no_dedupe(flags, code, location, severity, config), flags)
        )
    
    @staticmethod
    def __push_dedupe_unique(flags: Tensor, code: Tensor, location: int, severity: int, config: ErrorConfig) -> Tensor:
        """Unique-dedupe push. One entry per (location, code) pair. Fully vectorized."""
        should_push = (code != ErrorCode.OK) & (severity != Severity.OK)
        new_slot = ErrorOps.__pack_slot_tensor(code, location, severity)
        
        # Extract all slots and check for matching (location, code) pair
        all_slots = ErrorOps.__extract_all_slots(flags, config)  # (N, num_slots)
        slot_loc = (all_slots >> LOCATION_SHIFT) & 0x3FF  # (N, num_slots)
        slot_codes = (all_slots >> CODE_SHIFT) & 0xF  # (N, num_slots)
        
        # code is (N,), location is scalar
        matches = (slot_loc == location) & (slot_codes == code.unsqueeze(1)) & (all_slots != 0)  # (N, num_slots)
        
        # Find first match
        pair_exists = matches.any(dim=1)  # (N,)
        # Use argmax with masking for first match index
        match_scores = matches.float()
        positions = torch.arange(config.num_slots, device=flags.device, dtype=torch.float32)
        scores = match_scores - positions * 1e-6
        scores = torch.where(matches, scores, torch.tensor(-float('inf'), device=flags.device))
        existing_slot_idx = scores.argmax(dim=1)  # (N,)
        
        return torch.where(
            (pair_exists & should_push).unsqueeze(-1),
            ErrorOps.__update_slot_if_worse(flags, existing_slot_idx, new_slot, config),
            torch.where((~pair_exists & should_push).unsqueeze(-1), ErrorOps.__push_no_dedupe(flags, code, location, severity, config), flags)
        )
    
    @staticmethod
    def __replace_slot(flags: Tensor, slot_idx: Tensor, new_slot: Tensor, config: ErrorConfig) -> Tensor:
        """Replace slot at given index with new value. Fully vectorized."""
        return ErrorOps.__replace_slot_at(flags, slot_idx, new_slot, config)
    
    @staticmethod
    def __update_slot_if_worse(flags: Tensor, slot_idx: Tensor, new_slot: Tensor, config: ErrorConfig) -> Tensor:
        """Update slot only if new error is worse (higher severity). Fully vectorized."""
        # Extract all slots
        all_slots = ErrorOps.__extract_all_slots(flags, config)  # (N, num_slots)
        n, num_slots = all_slots.shape
        
        # Get existing slot at target index
        indices = torch.arange(num_slots, device=flags.device).unsqueeze(0)  # (1, num_slots)
        is_target = (indices == slot_idx.unsqueeze(1))  # (N, num_slots)
        
        # Get severity comparison
        new_sev = (new_slot & SEVERITY_MASK).unsqueeze(1)  # (N, 1)
        existing_sev = all_slots & SEVERITY_MASK  # (N, num_slots)
        
        # Only update if target AND new is worse
        should_update = is_target & (new_sev > existing_sev)  # (N, num_slots)
        
        # Update slots
        updated = torch.where(should_update, new_slot.unsqueeze(1), all_slots)
        
        # Pack back
        return ErrorOps.__pack_all_slots(updated, config)
    
    @staticmethod
    def push_scalar(flags: Tensor[int64_t, ("N", "num_words")], code: int, location: int, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Push same error to all samples. Fully compilable.
        
        Convenience method when the same error applies to all samples.
        
        Bit Layout:
            Same as push() - packs slot and inserts at position 0.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Existing error flags tensor
            code (int): Error code value (applied to all samples)
            location (int): Error location ID
            severity (int): Severity level (default: ERROR=2)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Updated error flags tensor
        
        Example:
            >>> flags = ErrorOps.push_scalar(flags, ErrorCode.RUNTIME, loc_id)
        """
        if code == ErrorCode.OK or severity == Severity.OK:
            return flags
        n = flags.shape[0]
        code_tensor = torch.full((n,), code, dtype=torch.int64, device=flags.device)
        return ErrorOps.push(flags, code_tensor, location, severity, config)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RECORDING - Merge Methods
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def merge(*flag_tensors: Tensor[int64_t, ("N", "num_words")], config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Merge multiple flag tensors into one. Compilable.
        
        Merges errors from all input tensors using the accumulation logic
        defined in config (same as push()).
        
        Bit Layout:
            Extracts each non-empty slot from source tensors and pushes
            into result using configured accumulation strategy.
        
        Args:
            *flag_tensors (Tensor[int64_t, (N, num_words)]): Flag tensors to merge (must have same shape)
            config (ErrorConfig): Error configuration for accumulation
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Merged error flags tensor
        
        Example:
            >>> flags = ErrorOps.merge(encoder_flags, decoder_flags, aux_flags)
            >>> flags = ErrorOps.merge(flags_a, flags_b)
        """
        if not flag_tensors:
            raise ValueError("Need at least one flag tensor")
        if len(flag_tensors) == 1:
            return flag_tensors[0]
        
        result = flag_tensors[0]
        for other in flag_tensors[1:]:
            result = ErrorOps.__merge_two(result, other, config)
        return result
    
    @staticmethod
    def __merge_two(flags: Tensor, other: Tensor, config: ErrorConfig) -> Tensor:
        """Merge errors from other into flags. Fully vectorized using stable sort."""
        # Extract all slots from both tensors
        flags_slots = ErrorOps.__extract_all_slots(flags, config)  # (N, num_slots)
        other_slots = ErrorOps.__extract_all_slots(other, config)  # (N, num_slots)
        
        # Concatenate: other first (newer in LIFO terms), then flags (older)
        combined = torch.cat([other_slots, flags_slots], dim=1)  # (N, 2*num_slots)
        
        # Identify non-empty slots (severity != 0)
        non_empty = (combined & SEVERITY_MASK) != 0  # (N, 2*num_slots)
        
        # Stable sort to move non-empty slots to front while preserving relative order
        # Key: 0 for non-empty (sort first), 1 for empty (sort last)
        sort_key = (~non_empty).long()
        _, indices = torch.sort(sort_key, dim=1, stable=True)
        
        # Gather reordered slots
        compacted = torch.gather(combined, 1, indices)
        
        # Take first num_slots and pack back
        return ErrorOps.__pack_all_slots(compacted[:, :config.num_slots], config)
    
    @staticmethod
    def __push_slot(flags: Tensor, slot: Tensor, config: ErrorConfig) -> Tensor:
        """Push a pre-packed slot into flags using LIFO logic. Fully vectorized."""
        sev = ErrorOps.__extract_severity(slot)
        should_push = (sev != Severity.OK)
        
        # Use vectorized LIFO shift
        new_flags = ErrorOps.__lifo_shift_one(flags, slot)
        
        return torch.where(should_push.unsqueeze(-1), new_flags, flags)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHECKING (bool masks)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def is_ok(flags: Tensor[int64_t, ("N", "num_words")]) -> Tensor[bool_t, ("N",)]:
        """
        Return bool mask where True indicates sample has NO errors.
        
        Bit Layout:
            Fast path: (flags == 0).all(dim=-1)
            All-zero tensor means no slots contain errors.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
        
        Returns:
            (Tensor[bool_t, (N,)]): True where all flags are zero
        
        Example:
            >>> mask = ErrorOps.is_ok(flags)
            >>> if mask.all(): print("All samples OK")
        """
        return (flags == 0).all(dim=-1)
    
    @staticmethod
    def is_err(flags: Tensor[int64_t, ("N", "num_words")]) -> Tensor[bool_t, ("N",)]:
        """
        Return bool mask where True indicates sample HAS errors.
        
        Bit Layout:
            Fast path: (flags != 0).any(dim=-1)
            Any non-zero word means at least one slot has an error.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
        
        Returns:
            (Tensor[bool_t, (N,)]): True where any flag is non-zero
        
        Example:
            >>> if ErrorOps.is_err(flags).any(): print("Some samples have errors")
        """
        return (flags != 0).any(dim=-1)
    
    @staticmethod
    def has_code(flags: Tensor[int64_t, ("N", "num_words")], code: int, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[bool_t, ("N",)]:
        """
        Check if any slot contains specific error code. Compilable.
        
        Bit Layout:
            Extracts code bits (bits 2-5) from each slot and compares.
            Only counts slots where severity != 0.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            code (int): Error code to check for (4-bit value)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[bool_t, (N,)]): True where code is found
        
        Example:
            >>> has_oob = ErrorOps.has_code(flags, ErrorCode.OUT_OF_BOUNDS)
        """
        N, num_words = flags.shape
        device = flags.device
        dtype = flags.dtype
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=dtype) * SLOT_BITS
        words = flags.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        
        slot_codes = (slots >> CODE_SHIFT) & 0xF
        slot_sev = slots & 0x3
        matches = (slot_codes == code) & (slot_sev != 0)
        
        return matches.any(dim=(1, 2))
    
    @staticmethod
    def has_nan(flags: Tensor[int64_t, ("N", "num_words")], config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[bool_t, ("N",)]:
        """
        Check if any slot has NaN error. Compilable.
        
        Bit Layout:
            Checks for code=1 (ErrorCode.NAN) in any slot.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[bool_t, (N,)]): True where NaN error found
        
        Example:
            >>> if ErrorOps.has_nan(flags).any(): print("NaN detected")
        """
        return ErrorOps.has_code(flags, ErrorCode.NAN, config)
    
    @staticmethod
    def has_inf(flags: Tensor[int64_t, ("N", "num_words")], config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[bool_t, ("N",)]:
        """
        Check if any slot has Inf error. Compilable.
        
        Bit Layout:
            Checks for code=2 (ErrorCode.INF) in any slot.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[bool_t, (N,)]): True where Inf error found
        """
        return ErrorOps.has_code(flags, ErrorCode.INF, config)
    
    @staticmethod
    def has_critical(flags: Tensor[int64_t, ("N", "num_words")], config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[bool_t, ("N",)]:
        """
        Check if any slot has CRITICAL severity. Compilable.
        
        Bit Layout:
            Checks for severity=3 (bits 0-1) in any slot.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[bool_t, (N,)]): True where critical error found
        """
        N, num_words = flags.shape
        device = flags.device
        dtype = flags.dtype
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=dtype) * SLOT_BITS
        words = flags.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        severities = slots & 0x3
        
        return (severities == Severity.CRITICAL).any(dim=(1, 2))
    
    @staticmethod
    def has_fallback(flags: Tensor[int64_t, ("N", "num_words")], config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[bool_t, ("N",)]:
        """
        Check if any slot indicates fallback value was used. Compilable.
        
        Bit Layout:
            Checks for code=ErrorCode.FALLBACK_VALUE in any slot.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[bool_t, (N,)]): True where fallback was used
        """
        return ErrorOps.has_code(flags, ErrorCode.FALLBACK_VALUE, config)
    
    @staticmethod
    def has_domain(flags: Tensor[int64_t, ("N", "num_words")], domain: int, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[bool_t, ("N",)]:
        """
        Check if any slot has error in given domain. Compilable.
        
        Bit Layout:
            Domain is encoded in top 2 bits of the 4-bit code (bits 4-5 of slot).
            Domains: NUMERIC=0, INDEX=1, QUALITY=2, RUNTIME=3.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            domain (int): Domain constant from ErrorDomain
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[bool_t, (N,)]): True where domain error found
        
        Example:
            >>> from . import ErrorDomain
            >>> has_numeric = ErrorOps.has_domain(flags, ErrorDomain.NUMERIC)
        """
        N, num_words = flags.shape
        device = flags.device
        dtype = flags.dtype
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=dtype) * SLOT_BITS
        words = flags.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        
        domain_bits = (domain >> 2) & 0x3
        slot_domains = (slots >> (CODE_SHIFT + 2)) & 0x3
        slot_sev = slots & 0x3
        
        matches = (slot_domains == domain_bits) & (slot_sev != 0)
        return matches.any(dim=(1, 2))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FILTERING
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def get_ok(flags: Tensor[int64_t, ("N", "num_words")]) -> Tensor[int64_t, ("M", "num_words")]:
        """
        Return only the flags for samples WITHOUT errors.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
        
        Returns:
            (Tensor[int64_t, (M, num_words)]): Filtered flags (M <= N ok samples)
        
        Example:
            >>> ok_flags = ErrorOps.get_ok(flags)
        """
        return flags[ErrorOps.is_ok(flags)]
    
    @staticmethod
    def get_err(flags: Tensor[int64_t, ("N", "num_words")]) -> Tensor[int64_t, ("M", "num_words")]:
        """
        Return only the flags for samples WITH errors.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
        
        Returns:
            (Tensor[int64_t, (M, num_words)]): Filtered flags (M <= N error samples)
        
        Example:
            >>> err_flags = ErrorOps.get_err(flags)
        """
        return flags[ErrorOps.is_err(flags)]
    
    @staticmethod
    def take_ok(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor) -> Tensor:
        """
        Filter tensor z to only include samples WITHOUT errors.
        
        NOTE: Uses dynamic shapes (boolean indexing). For torch.compile(fullgraph=True),
        use take_ok_p() / take_ok_p() (static-shape) instead, or enable:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            z (Tensor): Any tensor with N as first dimension
        
        Returns:
            (Tensor): Filtered tensor z, shape (M, ...) where M <= N
        
        Example:
            >>> output, flags = model(x)
            >>> ok_output = ErrorOps.take_ok(flags, output)
        """
        return z[ErrorOps.is_ok(flags)]
    
    @staticmethod
    def take_err(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor) -> Tensor:
        """
        Filter tensor z to only include samples WITH errors.
        
        NOTE: Uses dynamic shapes (boolean indexing). For torch.compile(fullgraph=True),
        use take_err_p() / take_err_p() (static-shape) instead, or enable:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            z (Tensor): Any tensor with N as first dimension
        
        Returns:
            (Tensor): Filtered tensor z, shape (M, ...) where M <= N
        
        Example:
            >>> output, flags = model(x)
            >>> err_output = ErrorOps.take_err(flags, output)
        """
        return z[ErrorOps.is_err(flags)]
    
    @staticmethod
    def partition(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Split tensor z into (ok_z, err_z) based on flags.
        
    Convenience method combining take_ok() and take_err() in one call.
        
        NOTE: Uses dynamic shapes (boolean indexing). For torch.compile(fullgraph=True),
        use take_ok_p()/take_err_p() instead, or enable:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            z (Tensor): Any tensor with N as first dimension
        
        Returns:
            (tuple[Tensor, Tensor]): (ok_samples, err_samples) where shapes are (M, ...) and (K, ...) with M + K = N
        
        Example:
            >>> output, flags = model(x)
            >>> ok_output, err_output = ErrorOps.partition(flags, output)
        """
        mask_ok = ErrorOps.is_ok(flags)
        return z[mask_ok], z[~mask_ok]
    
    @staticmethod
    def take_ok_p(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor, fill: float = 0.0) -> Tensor:
        """
        Return z with error samples replaced by fill value. STATIC SHAPE.

        Unlike take_ok() which filters to a smaller tensor, take_ok_p() returns
        the same shape with error samples replaced by the fill value. This is
        compatible with torch.compile(fullgraph=True).

        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            z (Tensor): Any tensor with N as first dimension
            fill (float): Value to use for error samples (default: 0.0)

        Returns:
            (Tensor): Same shape as z, with error samples replaced by fill

        Example:
            >>> output, flags = model(x)
            >>> # Inside torch.compile - use padded variant
            >>> ok_output = ErrorOps.take_ok_p(flags, output, fill=0.0)
        """
        mask_ok = ErrorOps.is_ok(flags)
        mask_exp = ErrorOps.__broadcast_mask(mask_ok, z)
        return torch.where(mask_exp, z, fill)
    
    @staticmethod
    def take_err_p(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor, fill: float = 0.0) -> Tensor:
        """
        Return z with OK samples replaced by fill value. STATIC SHAPE.

        Unlike take_err() which filters to a smaller tensor, take_err_p() returns
        the same shape with OK samples replaced by the fill value. This is
        compatible with torch.compile(fullgraph=True).

        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            z (Tensor): Any tensor with N as first dimension
            fill (float): Value to use for OK samples (default: 0.0)

        Returns:
            (Tensor): Same shape as z, with OK samples replaced by fill

        Example:
            >>> output, flags = model(x)
            >>> # Inside torch.compile - use padded variant
            >>> err_output = ErrorOps.take_err_p(flags, output, fill=0.0)
        """
        mask_err = ErrorOps.is_err(flags)
        mask_exp = ErrorOps.__broadcast_mask(mask_err, z)
        return torch.where(mask_exp, z, fill)
    
    @staticmethod
    def partition_many(flags: Tensor[int64_t, ("N", "num_words")], *tensors: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """
        Partition multiple tensors in lockstep based on flags.
        
        Splits all input tensors into ok/err groups using the same mask.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            *tensors (Tensor): Tensors to partition (all must have N as first dimension)
        
        Returns:
            (tuple[tuple[Tensor, ...], tuple[Tensor, ...]]): ((ok_t1, ok_t2, ...), (err_t1, err_t2, ...))
        
        Example:
            >>> output, hidden, flags = model(x)
            >>> (ok_out, ok_hid), (err_out, err_hid) = ErrorOps.partition_many(flags, output, hidden)
        """
        mask_ok = ErrorOps.is_ok(flags)
        ok = tuple(t[mask_ok] for t in tensors)
        err = tuple(t[~mask_ok] for t in tensors)
        return ok, err
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINATORS - Applicative / Monadic-style helpers
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def all_ok(flags: Tensor[int64_t, ("N", "num_words")]) -> Tensor:
        """
        Single bool: True if every sample is OK.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
        
        Returns:
            (Tensor): Scalar bool tensor
        
        Example:
            >>> if ErrorOps.all_ok(flags): print("Batch clean")
        """
        return ErrorOps.is_ok(flags).all()
    
    @staticmethod
    def any_err(flags: Tensor[int64_t, ("N", "num_words")]) -> Tensor:
        """
        Single bool: True if any sample has an error.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
        
        Returns:
            (Tensor): Scalar bool tensor
        
        Example:
            >>> if ErrorOps.any_err(flags): log_errors(flags)
        """
        return ErrorOps.is_err(flags).any()
    
    @staticmethod
    def map_ok(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Apply fn to z, commit results only for samples WITHOUT errors.
        
        fn is batch-level and runs on all samples (compile-friendly).
        Results are only used where flags indicate OK; error samples keep original z.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            z (Tensor): Input tensor (N, ...)
            fn (Callable[[Tensor], Tensor]): Batch-level transform z -> z_new
        
        Returns:
            (Tensor): z_out where z_out[i] = fn(z)[i] if OK, else z[i]
        
        Example:
            >>> z = ErrorOps.map_ok(flags, z, lambda x: x / x.norm(dim=-1, keepdim=True))
        """
        mask_ok = ErrorOps.is_ok(flags)
        z_new = fn(z)
        mask_expanded = ErrorOps.__broadcast_mask(mask_ok, z)
        return torch.where(mask_expanded, z_new, z)
    
    @staticmethod
    def map_err(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Apply fn to z, commit results only for samples WITH errors.
        
        fn is batch-level and runs on all samples (compile-friendly).
        Results are only used where flags indicate error; OK samples keep original z.
        
        This is the error-side analogue of map_ok().
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            z (Tensor): Input tensor (N, ...)
            fn (Callable[[Tensor], Tensor]): Batch-level transform z -> z_new
        
        Returns:
            (Tensor): z_out where z_out[i] = fn(z)[i] if ERR, else z[i]
        
        Example:
            >>> z = ErrorOps.map_err(flags, z, lambda x: torch.zeros_like(x))
        """
        mask_err = ErrorOps.is_err(flags)
        z_new = fn(z)
        mask_expanded = ErrorOps.__broadcast_mask(mask_err, z)
        return torch.where(mask_expanded, z_new, z)
    
    @staticmethod
    def map_err_flags(flags: Tensor[int64_t, ("N", "num_words")], fn: Callable[[Tensor], Tensor]) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Transform flags only for samples that currently have errors.
        
        fn is batch-level: fn(flags) -> flags_new (same shape).
        Useful for upgrading severities, remapping codes, etc.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            fn (Callable[[Tensor], Tensor]): Batch-level transform flags -> flags_new
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Transformed flags for error samples only
        
        Example:
            >>> flags = ErrorOps.map_err_flags(flags, lambda f: torch.zeros_like(f))
        """
        mask_err = ErrorOps.is_err(flags)
        flags_new = fn(flags)
        mask_expanded = mask_err.unsqueeze(-1)
        return torch.where(mask_expanded, flags_new, flags)
    
    @staticmethod
    def and_then(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor, fn: Callable[[Tensor], Tuple[Tensor, Tensor]], config: ErrorConfig = DEFAULT_CONFIG) -> Tuple[Tensor, Tensor[int64_t, ("N", "num_words")]]:
        """
        Strict Result-style chaining: only OK samples participate in fn.
        
        fn is batch-level: fn(z) -> (z_new, flags_new)
        
        Semantics per sample i:
            if is_ok(flags)[i]:
                z_out[i]     = z_new[i]
                flags_out[i] = merge(flags[i], flags_new[i])
            else:
                z_out[i]     = z[i]           # value frozen
                flags_out[i] = flags[i]       # flags_new[i] ignored
        
        This is true Result.and_then semantics: once errored, the sample
        is frozen and ignores all further operations.
        
        Note: fn still runs on all samples (batch-level, no dynamic shapes);
        masking is applied when committing results.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Current error flags
            z (Tensor): Input tensor (N, ...)
            fn (Callable): Batch-level fn(z) -> (z_new, flags_new)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tuple[Tensor, Tensor]): (z_out, flags_out)
        
        Example:
            >>> z, flags = ErrorOps.and_then(flags, z, layer_fn)
        """
        mask_ok = ErrorOps.is_ok(flags)
        z_new, flags_new = fn(z)
        
        # Only keep new flags where previously OK
        mask_ok_flags = mask_ok.unsqueeze(-1)
        flags_new_masked = torch.where(mask_ok_flags, flags_new, torch.zeros_like(flags_new))
        flags_out = ErrorOps.merge(flags, flags_new_masked, config=config)
        
        # Only update z where previously OK
        mask_expanded = ErrorOps.__broadcast_mask(mask_ok, z)
        z_out = torch.where(mask_expanded, z_new, z)
        
        return z_out, flags_out
    
    @staticmethod
    def bind(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor, fn: Callable[[Tensor], Tuple[Tensor, Tensor]], config: ErrorConfig = DEFAULT_CONFIG) -> Tuple[Tensor, Tensor[int64_t, ("N", "num_words")]]:
        """
        Monadic bind: apply fn, merge ALL errors, update values only for OK samples.
        
        fn is batch-level: fn(z) -> (z_new, flags_new)
        
        Semantics per sample i:
            Values:  z_out[i] = z_new[i] if is_ok(flags)[i], else z[i]
            Flags:   flags_out[i] = merge(flags[i], flags_new[i]) ALWAYS
        
        Values are short-circuited for errored samples, but errors keep
        accumulating so you see the full failure chain for debugging/telemetry.
        
        Compare to and_then() which does true short-circuit for both.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Current error flags
            z (Tensor): Input tensor (N, ...)
            fn (Callable): Batch-level fn(z) -> (z_new, flags_new)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tuple[Tensor, Tensor]): (z_out, flags_out)
        
        Example:
            >>> z, flags = ErrorOps.bind(flags, z, layer_fn)
        """
        mask_ok = ErrorOps.is_ok(flags)
        z_new, flags_new = fn(z)
        
        # Always accumulate flags (full error chain)
        flags_out = ErrorOps.merge(flags, flags_new, config=config)
        
        # Only update z where previously OK
        mask_expanded = ErrorOps.__broadcast_mask(mask_ok, z)
        z_out = torch.where(mask_expanded, z_new, z)
        
        return z_out, flags_out
    
    @staticmethod
    def ensure_mask(flags: Tensor[int64_t, ("N", "num_words")], ok_mask: Tensor, code: int, location: int, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Push error for samples where ok_mask is False.
        
        Turns a boolean predicate result into error flags.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Current error flags
            ok_mask (Tensor[bool, (N,)]): True = OK, False = push error
            code (int): Error code to push where mask is False
            location (int): Location ID for the error
            severity (int): Severity level (default: ERROR)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Updated flags
        
        Example:
            >>> ok_mask = (x.abs().amax(dim=-1) < 1e6)
            >>> flags = ErrorOps.ensure_mask(flags, ok_mask, ErrorCode.OUT_OF_BOUNDS, loc)
        """
        err_mask = ~ok_mask
        code_tensor = torch.where(
            err_mask,
            torch.full((flags.shape[0],), code, dtype=torch.int64, device=flags.device),
            torch.full((flags.shape[0],), ErrorCode.OK, dtype=torch.int64, device=flags.device),
        )
        return ErrorOps.push(flags, code_tensor, location, severity, config)
    
    @staticmethod
    def guard(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor, pred: Callable[[Tensor], Tensor], code: int, location: int, severity: int = Severity.ERROR, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Evaluate pred(z) and push errors where it returns False.
        
        Convenience wrapper around ensure_mask for predicate-based guards.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Current error flags
            z (Tensor): Input tensor to evaluate predicate on
            pred (Callable[[Tensor], Tensor]): Predicate z -> (N,) bool, True = OK
            code (int): Error code to push where pred is False
            location (int): Location ID
            severity (int): Severity level (default: ERROR)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Updated flags
        
        Example:
            >>> flags = ErrorOps.guard(flags, z, 
            ...     lambda x: (x >= 0).all(dim=-1),
            ...     ErrorCode.OUT_OF_BOUNDS, loc)
        """
        ok_mask = pred(z).to(torch.bool)
        return ErrorOps.ensure_mask(flags, ok_mask, code, location, severity, config)
    
    @staticmethod
    def recover_with_fallback(flags: Tensor[int64_t, ("N", "num_words")], z: Tensor, fallback: Tensor, location: int, severity: int = Severity.WARN, config: ErrorConfig = DEFAULT_CONFIG) -> Tuple[Tensor, Tensor[int64_t, ("N", "num_words")]]:
        """
        Replace error samples with fallback value and mark with FALLBACK_VALUE.
        
        OK samples keep their original value. Error samples get the fallback
        and receive an additional FALLBACK_VALUE error to track recovery.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Current error flags
            z (Tensor): Input tensor (N, ...)
            fallback (Tensor): Fallback value (scalar or matching shape)
            location (int): Location ID for the fallback marker
            severity (int): Severity for fallback marker (default: WARN)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tuple[Tensor, Tensor]): (z_recovered, flags_updated)
        
        Example:
            >>> z, flags = ErrorOps.recover_with_fallback(flags, z, torch.tensor(0.0), loc)
        """
        mask_err = ErrorOps.is_err(flags)
        
        # Broadcast fallback to match z shape
        fallback_full = fallback.expand_as(z)
        
        mask_expanded = ErrorOps.__broadcast_mask(mask_err, z)
        z_out = torch.where(mask_expanded, fallback_full, z)
        
        # Push FALLBACK_VALUE where we used fallback
        code_tensor = torch.where(
            mask_err,
            torch.full((flags.shape[0],), ErrorCode.FALLBACK_VALUE, dtype=torch.int64, device=flags.device),
            torch.full((flags.shape[0],), ErrorCode.OK, dtype=torch.int64, device=flags.device),
        )
        flags_out = ErrorOps.push(flags, code_tensor, location, severity, config)
        
        return z_out, flags_out
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERYING
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def count_errors(flags: Tensor[int64_t, ("N", "num_words")], config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int32_t, ("N",)]:
        """
        Count number of non-empty error slots per sample. Compilable.
        
        Bit Layout:
            Counts slots where severity != 0 (bits 0-1 of each 16-bit slot).
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[int32_t, (N,)]): Error count per sample
        
        Example:
            >>> counts = ErrorOps.count_errors(flags)
            >>> max_errors = counts.max().item()
        """
        N, num_words = flags.shape
        device = flags.device
        dtype = flags.dtype
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=dtype) * SLOT_BITS
        words = flags.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        non_empty = (slots != 0)
        
        return non_empty.sum(dim=(1, 2)).to(torch.int32)
    
    @staticmethod
    def max_severity(flags: Tensor[int64_t, ("N", "num_words")], config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int64_t, ("N",)]:
        """
        Get maximum severity across all slots per sample. Compilable.
        
        Bit Layout:
            Extracts severity (bits 0-1) from each slot and returns max.
            Values: OK=0, WARN=1, ERROR=2, CRITICAL=3.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[int64_t, (N,)]): Maximum severity per sample
        
        Example:
            >>> severities = ErrorOps.max_severity(flags)
            >>> has_critical = (severities == Severity.CRITICAL).any()
        """
        N, num_words = flags.shape
        device = flags.device
        dtype = flags.dtype
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=dtype) * SLOT_BITS
        words = flags.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        severities = slots & 0x3
        
        return severities.amax(dim=(1, 2))
    
    @staticmethod
    def get_slot(flags: Tensor[int64_t, ("N", "num_words")], slot_idx: int) -> Tensor[int64_t, ("N",)]:
        """
        Get raw slot value at index. Compilable.
        
        Bit Layout:
            Extracts 16-bit slot from position slot_idx.
            word_idx = slot_idx // 4, bit_offset = (slot_idx % 4) * 16.
            Returns (flags[:, word_idx] >> bit_offset) & 0xFFFF.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            slot_idx (int): Slot index to extract (0 to num_slots-1)
        
        Returns:
            (Tensor[int64_t, (N,)]): Raw 16-bit slot values
        """
        word_idx = slot_idx // SLOTS_PER_WORD
        bit_offset = (slot_idx % SLOTS_PER_WORD) * SLOT_BITS
        return (flags[:, word_idx] >> bit_offset) & SLOT_MASK
    
    @staticmethod
    def get_first_severity(flags: Tensor[int64_t, ("N", "num_words")]) -> Tensor[int32_t, ("N",)]:
        """
        Get severity from slot 0. Compilable.
        
        Bit Layout:
            Extracts bits 0-1 from first slot: flags[:, 0] & 0x3.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
        
        Returns:
            (Tensor[int32_t, (N,)]): Severity values (0-3)
        """
        return (flags[:, 0] & 0x3).to(torch.int32)
    
    @staticmethod
    def get_first_code(flags: Tensor[int64_t, ("N", "num_words")]) -> Tensor[int32_t, ("N",)]:
        """
        Get error code from slot 0. Compilable.
        
        Bit Layout:
            Extracts bits 2-5 from first slot: (flags[:, 0] >> 2) & 0xF.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
        
        Returns:
            (Tensor[int32_t, (N,)]): Code values (0-15)
        """
        return ((flags[:, 0] >> CODE_SHIFT) & 0xF).to(torch.int32)
    
    @staticmethod
    def get_first_location(flags: Tensor[int64_t, ("N", "num_words")]) -> Tensor[int32_t, ("N",)]:
        """
        Get location from slot 0. Compilable.
        
        Bit Layout:
            Extracts bits 6-15 from first slot: (flags[:, 0] >> 6) & 0x3FF.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
        
        Returns:
            (Tensor[int32_t, (N,)]): Location values (0-1023)
        """
        return ((flags[:, 0] >> LOCATION_SHIFT) & 0x3FF).to(torch.int32)
    
    @staticmethod
    def clear(flags: Tensor[int64_t, ("N", "num_words")], code: int, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor[int64_t, ("N", "num_words")]:
        """
        Clear (remove) all occurrences of a specific error code from flags.
        
        Vectorized - torch.compile friendly.
        
        Bit Layout:
            Zeros out any slot where (slot >> 2) & 0xF == code.
        
        Args:
            flags (Tensor[int64_t, (N, num_words)]): Error flags tensor
            code (int): Error code to clear (4-bit value)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Tensor[int64_t, (N, num_words)]): Flags with specified error code removed
        
        Example:
            >>> flags = ErrorOps.clear(flags, ErrorCode.NAN)
        """
        N, num_words = flags.shape
        device = flags.device
        dtype = flags.dtype
        
        slot_shifts = torch.arange(SLOTS_PER_WORD, device=device, dtype=dtype) * SLOT_BITS
        words = flags.unsqueeze(-1)
        slots = (words >> slot_shifts) & SLOT_MASK
        slot_codes = (slots >> CODE_SHIFT) & 0xF
        should_clear = (slot_codes == code)
        cleared_slots = torch.where(should_clear, torch.zeros(1, dtype=dtype, device=device), slots)
        shifted_slots = cleared_slots << slot_shifts
        new_words = shifted_slots.sum(dim=-1)
        
        return new_words
