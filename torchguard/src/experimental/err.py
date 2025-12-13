"""
Experimental err namespace wrapper.

Mirrors the stable err namespace but uses Float64ErrorOps.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .ops import Float64ErrorOps
from ..core import ErrorCode, ErrorDomain, Severity
from ..core.config import DEFAULT_CONFIG, ErrorConfig


class _XErrNamespace:
    """
    Experimental compiled-safe error operations namespace.
    
    All methods return float64 tensors for AOTAutograd compatibility.
    API is identical to stable err namespace.
    """
    # Error codes as attributes
    OK = ErrorCode.OK
    NAN = ErrorCode.NAN
    INF = ErrorCode.INF
    OVERFLOW = ErrorCode.OVERFLOW
    OUT_OF_BOUNDS = ErrorCode.OUT_OF_BOUNDS
    NEGATIVE_IDX = ErrorCode.NEGATIVE_IDX
    EMPTY_INPUT = ErrorCode.EMPTY_INPUT
    ZERO_OUTPUT = ErrorCode.ZERO_OUTPUT
    CONSTANT_OUTPUT = ErrorCode.CONSTANT_OUTPUT
    SATURATED = ErrorCode.SATURATED
    FALLBACK_VALUE = ErrorCode.FALLBACK_VALUE
    VALUE_CLAMPED = ErrorCode.VALUE_CLAMPED
    UNKNOWN = ErrorCode.UNKNOWN
    
    # Severity levels
    WARN = Severity.WARN
    ERROR = Severity.ERROR
    CRITICAL = Severity.CRITICAL
    
    # Classes for advanced usage
    ErrorCode = ErrorCode
    ErrorDomain = ErrorDomain
    Severity = Severity
    
    # === Creation ===
    @staticmethod
    def new(reference):
        """Create empty error flags tensor from reference. Returns float64."""
        return Float64ErrorOps.new(reference)
    
    @staticmethod
    def new_t(batch_size, device=None, config=DEFAULT_CONFIG):
        """Create empty error flags tensor from batch size. Returns float64."""
        return Float64ErrorOps.new_t(batch_size, device, config)
    
    @staticmethod
    def from_code(code, location, batch_size, device=None, severity=None, config=DEFAULT_CONFIG):
        """Create flags with a single error code. Returns float64."""
        if severity is None:
            severity = Severity.ERROR
        return Float64ErrorOps.from_code(code, location, batch_size, device, severity, config)
    
    # === Recording ===
    @staticmethod
    def push(flags, code, location, severity=None, config=DEFAULT_CONFIG, *, where=None):
        """Push error code to flags. Returns float64."""
        if isinstance(code, torch.Tensor):
            from ..core.location import ErrorLocation
            if isinstance(location, str):
                loc = ErrorLocation.register(location)
            elif isinstance(location, int):
                loc = location
            else:
                loc = ErrorLocation.UNKNOWN
            if severity is None:
                severity = Severity.ERROR
            return Float64ErrorOps.push(flags, code, loc, severity, config)
        else:
            # Use high-level push with where support
            from ..core.location import ErrorLocation
            if isinstance(location, str):
                loc = ErrorLocation.register(location)
            elif isinstance(location, int):
                loc = location
            else:
                loc = ErrorLocation.UNKNOWN
            if severity is None:
                severity = ErrorCode.default_severity(code)
            
            n = flags.shape[0]
            if where is None:
                code_tensor = torch.full((n,), code, dtype=torch.int64, device=flags.device)
            else:
                code_tensor = torch.where(where, code, ErrorCode.OK)
            
            return Float64ErrorOps.push(flags, code_tensor, loc, severity, config)
    
    @staticmethod
    def merge(*flag_tensors, config=DEFAULT_CONFIG):
        """Merge multiple flag tensors. Returns float64."""
        return Float64ErrorOps.merge(*flag_tensors, config=config)
    
    # === Checking ===
    @staticmethod
    def is_ok(flags):
        """Per-sample bool mask: True where NO errors."""
        return Float64ErrorOps.is_ok(flags)
    
    @staticmethod
    def is_err(flags):
        """Per-sample bool mask: True where HAS errors."""
        return Float64ErrorOps.is_err(flags)
    
    @staticmethod
    def has_any(flags):
        """Scalar bool: True if ANY sample has errors."""
        return Float64ErrorOps.any_err(flags)
    
    @staticmethod
    def has_nan(flags, config=DEFAULT_CONFIG):
        """Per-sample bool mask for NaN errors."""
        return Float64ErrorOps.has_nan(flags, config)
    
    @staticmethod
    def has_inf(flags, config=DEFAULT_CONFIG):
        """Per-sample bool mask for Inf errors."""
        return Float64ErrorOps.has_inf(flags, config)
    
    @staticmethod
    def has_code(flags, code, config=DEFAULT_CONFIG):
        """Per-sample bool mask for specific error code."""
        return Float64ErrorOps.has_code(flags, code, config)
    
    @staticmethod
    def has_critical(flags, config=DEFAULT_CONFIG):
        """Per-sample bool mask for critical errors."""
        return Float64ErrorOps.has_critical(flags, config)
    
    @staticmethod
    def count_errors(flags, config=DEFAULT_CONFIG):
        """Count errors per sample."""
        return Float64ErrorOps.count_errors(flags, config)
    
    @staticmethod
    def max_severity(flags, config=DEFAULT_CONFIG):
        """Get max severity per sample."""
        return Float64ErrorOps.max_severity(flags, config)
    
    # === Filtering ===
    @staticmethod
    def get_ok(flags):
        """Filter flags to OK samples only."""
        return Float64ErrorOps.get_ok(flags)
    
    @staticmethod
    def get_err(flags):
        """Filter flags to error samples only."""
        return Float64ErrorOps.get_err(flags)
    
    @staticmethod
    def take_ok(flags, tensor):
        """Filter tensor to OK samples (alias for Float64ErrorOps.take_ok)."""
        return Float64ErrorOps.take_ok(flags, tensor)
    
    @staticmethod
    def take_err(flags, tensor):
        """Filter tensor to error samples (alias for Float64ErrorOps.take_err)."""
        return Float64ErrorOps.take_err(flags, tensor)
    
    @staticmethod
    def partition(flags, tensor):
        """Split tensor into (ok_tensor, err_tensor). Uses dynamic shapes."""
        return Float64ErrorOps.partition(flags, tensor)
    
    @staticmethod
    def take_ok_p(flags, tensor, fill=0.0):
        """Return tensor with error samples replaced by fill. STATIC SHAPE - torch.compile safe.

        Alias for Float64ErrorOps.take_ok_p.
        """
        return Float64ErrorOps.take_ok_p(flags, tensor, fill)
    
    @staticmethod
    def take_err_p(flags, tensor, fill=0.0):
        """Return tensor with OK samples replaced by fill. STATIC SHAPE - torch.compile safe.

        Alias for Float64ErrorOps.take_err_p.
        """
        return Float64ErrorOps.take_err_p(flags, tensor, fill)
    
    # === Combinators ===
    @staticmethod
    def map_ok(flags, tensor, fn):
        """Apply fn only to OK samples."""
        return Float64ErrorOps.map_ok(flags, tensor, fn)
    
    @staticmethod
    def map_err(flags, tensor, fn):
        """Apply fn only to error samples."""
        return Float64ErrorOps.map_err(flags, tensor, fn)
    
    @staticmethod
    def and_then(flags, tensor, fn):
        """Chain: skip fn for error samples (short-circuit)."""
        return Float64ErrorOps.and_then(flags, tensor, fn)
    
    @staticmethod
    def bind(flags, tensor, fn):
        """Chain: run fn for all, accumulate errors."""
        return Float64ErrorOps.bind(flags, tensor, fn)
    
    @staticmethod
    def guard(flags, tensor, predicate, code, location, severity=None, config=DEFAULT_CONFIG):
        """Push error where predicate is False."""
        if severity is None:
            severity = Severity.ERROR
        return Float64ErrorOps.guard(flags, tensor, predicate, code, location, severity, config)
    
    @staticmethod
    def recover_with_fallback(flags, tensor, fallback, location, *, config=DEFAULT_CONFIG):
        """Replace error samples with fallback value."""
        return Float64ErrorOps.recover_with_fallback(flags, tensor, fallback, location, config=config)
    
    @staticmethod
    def all_ok(flags):
        """Scalar bool: True if ALL samples are OK."""
        return Float64ErrorOps.all_ok(flags)
    
    @staticmethod
    def any_err(flags):
        """Scalar bool: True if ANY sample has errors."""
        return Float64ErrorOps.any_err(flags)
    
    @staticmethod
    def ensure_mask(flags, predicate, code, location, severity=None, config=DEFAULT_CONFIG):
        """Push error where predicate is False."""
        if severity is None:
            severity = Severity.ERROR
        return Float64ErrorOps.ensure_mask(flags, predicate, code, location, severity, config)
    
    @staticmethod
    def map_err_flags(flags, fn):
        """Apply fn to flags of error samples."""
        return Float64ErrorOps.map_err_flags(flags, fn)
    
    @staticmethod
    def partition_many(flags, *tensors):
        """Partition multiple tensors by error status."""
        return Float64ErrorOps.partition_many(flags, *tensors)
    
    @staticmethod
    def clear(flags, code, config=DEFAULT_CONFIG):
        """Clear specific error code from flags. Returns float64."""
        return Float64ErrorOps.clear(flags, code, config)
    
    @staticmethod
    def push_scalar(flags, code, location, severity=None, config=DEFAULT_CONFIG):
        """Push scalar code to all samples. Returns float64."""
        if severity is None:
            severity = Severity.ERROR
        return Float64ErrorOps.push_scalar(flags, code, location, severity, config)
    
    @staticmethod
    def has_domain(flags, domain, config=DEFAULT_CONFIG):
        """Per-sample bool mask for error domain."""
        return Float64ErrorOps.has_domain(flags, domain, config)
    
    @staticmethod
    def has_fallback(flags, config=DEFAULT_CONFIG):
        """Per-sample bool mask for fallback values."""
        return Float64ErrorOps.has_fallback(flags, config)
    
    @staticmethod
    def get_first_code(flags):
        """Get first error code per sample."""
        return Float64ErrorOps.get_first_code(flags)
    
    @staticmethod
    def get_first_location(flags):
        """Get first error location per sample."""
        return Float64ErrorOps.get_first_location(flags)
    
    @staticmethod
    def get_first_severity(flags):
        """Get first error severity per sample."""
        return Float64ErrorOps.get_first_severity(flags)
    
    @staticmethod
    def find(code, flags, config=DEFAULT_CONFIG):
        """Find which samples have a specific error code. Works with float64 flags."""
        return Float64ErrorOps.has_code(flags, code, config)


# Singleton instance
err = _XErrNamespace()

