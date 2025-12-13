"""
ErrorFlags: Python boundary operations for error flags.

This module provides operations for inspecting and debugging error flags
at the Python boundary (outside torch.compile regions).

For all compilable operations (creation, recording, checking, filtering),
use the error_t class instead:

    from ..err import err
    
    # Inside torch.compile
    flags = error_t.new(x)
    flags = error_t.push(flags, codes, location)
    
    # At Python boundary - use ErrorFlags for inspection
    from torchguard import ErrorFlags
    ErrorFlags.pretty_print(flags)

Bit Layout:
    Error flags are stored as int64 tensors with shape (N, num_words).
    Each 64-bit word contains 4 slots of 16 bits each.
    
    Slot layout (16 bits, LSB to MSB):
        +------------+----------+----------+
        | bits 15-6  | bits 5-2 | bits 1-0 |
        | location   | code     | severity |
        | (10 bits)  | (4 bits) | (2 bits) |
        +------------+----------+----------+
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

import torch
from torch import Tensor

from ..core.codes import ErrorCode
from ..core.config import DEFAULT_CONFIG, ErrorConfig
from ..core.constants import CODE_SHIFT, LOCATION_SHIFT, SLOT_BITS, SLOT_MASK, SLOTS_PER_WORD
from ..core.severity import Severity

__all__ = ['ErrorFlags', 'UnpackedError']


class UnpackedError(NamedTuple):
    """
    Single unpacked error from flags.
    
    Attributes:
        severity (int): Severity level (0-3)
        code (int): Error code (0-15)
        location (int): Location ID (0-1023)
        severity_name (str): Human-readable severity name
        code_name (str): Human-readable code name
        location_name (str): Human-readable location name
    """
    severity: int
    code: int
    location: int
    severity_name: str
    code_name: str
    location_name: str
    
    def __repr__(self) -> str:
        """String representation of the unpacked error."""
        if self.severity == Severity.OK:
            return "UnpackedError(OK)"
        return f"UnpackedError({self.severity_name}:{self.code_name}@{self.location_name})"


class ErrorFlags:
    """
    Python boundary operations for error flags.
    
    Use this class ONLY at Python boundaries for inspecting/debugging flags.
    For all compilable operations, use error_t instead.
    
    Available methods (all Python boundary only):
        unpack(flags, sample_idx)     - Extract errors for one sample
        unpack_all(flags)             - Extract errors for all samples
        to_errors(flags)              - Convert to Error objects
        interpret(flags, component)   - Convert with component context
        summary(flags)                - Aggregate counts by location/code
        repr(flags)                   - Human-readable summary string
        pretty_print(flags)           - Formatted debug output
        reduce_or(flags_list)         - Fast bitwise OR reduction
    
    Example:
        output, flags = compiled_model(x)
        
        # Inspect errors at boundary
        if err.is_err(flags).any():
            errors = ErrorFlags.to_errors(flags)
            for err in errors:
                print(f"Error: {err}")
            
            # Or debug print
            print(ErrorFlags.pretty_print(flags, sample_idx=0))
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UNPACKING (Python boundary only)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    @torch.compiler.disable
    def unpack(cls, flags: Tensor, sample_idx: int = 0, config: ErrorConfig = DEFAULT_CONFIG) -> List[UnpackedError]:
        """
        Unpack all errors for a single sample. Python boundary only.
        
        Bit Layout:
            Iterates through each slot, extracts severity/code/location
            from the 16-bit packed format.
        
        Args:
            flags (Tensor): Error flags tensor (N, num_words)
            sample_idx (int): Sample index to unpack
            config (ErrorConfig): Error configuration
        
        Returns:
            (List[UnpackedError]): List of unpacked errors (non-OK only)
        
        Example:
            >>> errors = ErrorFlags.unpack(flags, sample_idx=0)
            >>> for err in errors:
            ...     print(f"{err.code_name} @ {err.location_name}")
        """
        from ..core.location import ErrorLocation
        
        errors: List[UnpackedError] = []
        for w in range(config.num_words):
            word_val = flags[sample_idx, w].item()
            for s in range(SLOTS_PER_WORD):
                shift = s * SLOT_BITS
                slot_val = (word_val >> shift) & SLOT_MASK
                
                severity = slot_val & 0x3
                code = (slot_val >> CODE_SHIFT) & 0xF
                location = (slot_val >> LOCATION_SHIFT) & 0x3FF
                
                if severity != Severity.OK:
                    errors.append(UnpackedError(
                        severity=severity,
                        code=code,
                        location=location,
                        severity_name=Severity.name(severity),
                        code_name=ErrorCode.name(code),
                        location_name=ErrorLocation.name(location),
                    ))
        return errors
    
    @classmethod
    @torch.compiler.disable
    def unpack_all(cls, flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> List[List[UnpackedError]]:
        """
        Unpack errors for all samples. Python boundary only.
        
        Args:
            flags (Tensor): Error flags tensor (N, num_words)
            config (ErrorConfig): Error configuration
        
        Returns:
            (List[List[UnpackedError]]): List of unpacked errors per sample
        
        Example:
            >>> all_errors = ErrorFlags.unpack_all(flags)
            >>> for i, sample_errors in enumerate(all_errors):
            ...     if sample_errors:
            ...         print(f"Sample {i}: {len(sample_errors)} errors")
        """
        return [cls.unpack(flags, i, config) for i in range(flags.shape[0])]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY/DEBUG (Python boundary only)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    @torch.compiler.disable
    def summary(cls, flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> Dict[str, Dict[str, int]]:
        """
        Aggregate error counts by location and code.
        
        Python boundary only (logging, monitoring). O(N * num_slots).
        
        Args:
            flags (Tensor): Error flags tensor (N, num_words)
            config (ErrorConfig): Error configuration
        
        Returns:
            (Dict[str, Dict[str, int]]): {location_name: {code_name: count}}
        
        Example:
            >>> summary = ErrorFlags.summary(flags)
            >>> # {'customer_encoder': {'NAN': 5, 'INF': 2}, ...}
        """
        result: Dict[str, Dict[str, int]] = {}
        n = flags.shape[0]
        
        for i in range(n):
            for err in cls.unpack(flags, i, config):
                if err.code == ErrorCode.OK:
                    continue
                
                loc = err.location_name
                code = err.code_name
                
                if loc not in result:
                    result[loc] = {}
                result[loc][code] = result[loc].get(code, 0) + 1
        
        return result
    
    @classmethod
    @torch.compiler.disable
    def repr(cls, flags: Tensor, config: ErrorConfig = DEFAULT_CONFIG) -> str:
        """
        Pretty string representation of flags tensor.
        
        Python boundary only (debugging, logging).
        
        Args:
            flags (Tensor): Error flags tensor (N, num_words)
            config (ErrorConfig): Error configuration
        
        Returns:
            (str): Human-readable summary string
        
        Example:
            >>> print(ErrorFlags.repr(flags))
            >>> # ErrorFlags(32 samples, 5 errors: 3xNAN, 2xINF @ customer_encoder, group_encoder)
        """
        from ..err import err
        
        n = flags.shape[0]
        total_errors = int(err.count_errors(flags, config).sum().item())
        
        if total_errors == 0:
            return f"ErrorFlags({n} samples, no errors)"
        
        summary = cls.summary(flags, config)
        
        code_counts: Dict[str, int] = {}
        loc_set: set = set()
        for loc, codes in summary.items():
            loc_set.add(loc)
            for code, count in codes.items():
                code_counts[code] = code_counts.get(code, 0) + count
        
        code_str = ", ".join(f"{v}x{k}" for k, v in sorted(code_counts.items(), key=lambda x: -x[1]))
        
        loc_list = sorted(loc_set)[:3]
        loc_str = ", ".join(loc_list)
        if len(loc_set) > 3:
            loc_str += f" +{len(loc_set)-3} more"
        
        return f"ErrorFlags({n} samples, {total_errors} errors: {code_str} @ {loc_str})"
    
    @classmethod
    @torch.compiler.disable
    def pretty_print(cls, flags: Tensor, sample_idx: int = 0, config: ErrorConfig = DEFAULT_CONFIG) -> str:
        """
        Pretty print errors for debugging. Python boundary only.
        
        Args:
            flags (Tensor): Error flags tensor (N, num_words)
            sample_idx (int): Sample index to print
            config (ErrorConfig): Error configuration
        
        Returns:
            (str): Formatted string representation of errors
        
        Example:
            >>> print(ErrorFlags.pretty_print(flags, sample_idx=0))
            >>> # Found 2 error(s):
            >>> #   [0] ERROR | NUMERIC.NAN | @ customer_encoder
            >>> #   [1] WARN | INDEX.OUT_OF_BOUNDS | @ hash_helpers
        """
        errors = cls.unpack(flags, sample_idx, config)
        if not errors:
            return "[OK] No errors"
        
        lines = [f"Found {len(errors)} error(s):"]
        for i, err in enumerate(errors):
            domain = ErrorCode.domain(err.code)
            domain_names = {0: "NUMERIC", 1: "INDEX", 2: "QUALITY", 3: "RUNTIME"}
            lines.append(
                f"  [{i}] {err.severity_name} | "
                f"{domain_names.get(domain, '?')}.{err.code_name} | "
                f"@ {err.location_name}"
            )
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REDUCTION (Special operations)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    def reduce_or(cls, flags_list: List[Tensor]) -> Tensor:
        """
        Reduce multiple flag tensors using bitwise OR.
        
        WARNING: Bitwise OR can corrupt slot data! Use ONLY for fast
        "has any error?" checks, not for preserving individual errors.
        
        For lossless merging, use error_t.merge() instead.
        
        Args:
            flags_list (List[Tensor]): Flag tensors to reduce
        
        Returns:
            (Tensor): Reduced flags (bitwise OR of all inputs)
        
        Example:
            >>> # Fast check across multiple batches
            >>> combined = ErrorFlags.reduce_or([flags1, flags2, flags3])
            >>> if err.is_err(combined).any():
            ...     # At least one batch has errors
        """
        if not flags_list:
            raise ValueError("Need at least one flag tensor")
        device = flags_list[0].device
        gathered = [f.to(device) for f in flags_list]
        result = gathered[0]
        for f in gathered[1:]:
            result = result | f
        return result
