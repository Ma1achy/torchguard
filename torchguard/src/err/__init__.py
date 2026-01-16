"""
Error handling namespace for torchguard.

Two namespaces:
- err: Compiled-safe tensor operations (use inside torch.compile)
- flags: Python boundary operations (use for debugging/inspection)

Usage:
    from torchguard import err, flags, error_t
    
    # Inside compiled regions - use err namespace
    f = err.new(x)
    f = err.push(f, err.NAN, location)
    mask = err.is_ok(f)
    
    # At Python boundary - use flags namespace
    if err.has_any(f):
        print(flags.repr(f))
        errors = flags.unpack(f)
"""
from __future__ import annotations

# Standard library
from typing import List, Optional, Union

# Third-party
import torch
from torch import Tensor

# Internal
from ..core import ErrorCode, ErrorDomain, Severity
from ..core.config import ErrorConfig, get_config
from .inspect import ErrorFlags, UnpackedError
from .ops import ErrorOps

# Type alias for replace() targets parameter
ReplaceTarget = Union[int, str, float]


class _ErrNamespace:
    """
    Compiled-safe error operations namespace.
    
    All methods work with torch.compile(fullgraph=True).
    """
    # Error codes as attributes (err.NAN, err.INF, etc.)
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
    
    # ErrorCode class for advanced usage
    ErrorCode = ErrorCode
    ErrorDomain = ErrorDomain
    Severity = Severity
    
    # === Creation ===
    @staticmethod
    def new(reference):
        """Create empty error flags tensor from reference."""
        config = get_config()
        # Route to appropriate backend based on global config dtype
        if config.flag_dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.new(reference, config)
        else:
            return ErrorOps.new(reference)
    
    @staticmethod
    def new_t(batch_size, device=None, config: Optional[ErrorConfig] = None):
        """Create empty error flags tensor from batch size."""
        cfg = config or get_config()
        # Route to appropriate backend based on config dtype
        if cfg.flag_dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.new_t(batch_size, device, cfg)
        else:
            return ErrorOps.new_t(batch_size, device, cfg)
    
    @staticmethod
    def from_code(code, location, batch_size, device=None, severity=None, config: Optional[ErrorConfig] = None):
        """Create flags with a single error code."""
        if severity is None:
            severity = Severity.ERROR
        cfg = config or get_config()
        # Route to appropriate backend based on config dtype
        if cfg.flag_dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.from_code(code, location, batch_size, device, severity, cfg)
        else:
            return ErrorOps.from_code(code, location, batch_size, device, severity, cfg)
    
    # === Recording ===
    @staticmethod
    def push(flags, code, location, severity=None, config: Optional[ErrorConfig] = None, *, where=None):
        """Push error code to flags (conditional with where mask).
        
        Args:
            flags: Error flags tensor
            code: Error code (int) or tensor of codes (for vectorized push)
            location: Location (int, str, or Module)
            severity: Optional severity override
            config: Error config
            where: Optional bool mask - only push where True
        """
        import torch
        cfg = config or get_config()
        
        # If code is a tensor, use low-level ErrorOps.push directly
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
            return ErrorOps.push(flags, code, loc, severity, cfg)
        else:
            # Use helpers.push for high-level API with where support
            from .helpers import push as helpers_push
            return helpers_push(flags, code, location, where=where, severity=severity, config=cfg)
    
    @staticmethod
    def merge(*flag_tensors, config: Optional[ErrorConfig] = None):
        """Merge multiple flag tensors."""
        return ErrorOps.merge(*flag_tensors, config=config or get_config())
    
    # === Checking ===
    @staticmethod
    def is_ok(flags):
        """Per-sample bool mask: True where NO errors."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.is_ok(flags)
        return ErrorOps.is_ok(flags)
    
    @staticmethod
    def is_err(flags):
        """Per-sample bool mask: True where HAS errors."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.is_err(flags)
        return ErrorOps.is_err(flags)
    
    @staticmethod
    def has_any(flags):
        """Scalar bool: True if ANY sample has errors."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.any_err(flags)
        return ErrorOps.any_err(flags)
    
    @staticmethod
    def has_nan(flags, config=None):
        """Per-sample bool mask for NaN errors."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.has_nan(flags, config)
        return ErrorOps.has_nan(flags, config)
    
    @staticmethod
    def has_inf(flags, config=None):
        """Per-sample bool mask for Inf errors."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.has_inf(flags, config)
        return ErrorOps.has_inf(flags, config)
    
    @staticmethod
    def has_code(flags, code, config=None):
        """Per-sample bool mask for specific error code."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.has_code(flags, code, config)
        return ErrorOps.has_code(flags, code, config)
    
    @staticmethod
    def has_critical(flags, config=None):
        """Per-sample bool mask for critical errors."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.has_critical(flags, config)
        return ErrorOps.has_critical(flags, config)
    
    @staticmethod
    def count_errors(flags, config=None):
        """Count errors per sample."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.count_errors(flags, config)
        return ErrorOps.count_errors(flags, config)
    
    @staticmethod
    def max_severity(flags, config=None):
        """Get max severity per sample."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.max_severity(flags, config)
        return ErrorOps.max_severity(flags, config)
    
    # === Filtering ===
    @staticmethod
    def get_ok(flags):
        """Filter flags to OK samples only."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.get_ok(flags)
        return ErrorOps.get_ok(flags)
    
    @staticmethod
    def get_err(flags):
        """Filter flags to error samples only."""
        if flags.dtype in (torch.float32, torch.float64):
            from ..experimental.ops import Float64ErrorOps
            return Float64ErrorOps.get_err(flags)
        return ErrorOps.get_err(flags)
    
    @staticmethod
    def take_ok(flags, tensor):
        """Filter tensor to OK samples (alias for ErrorOps.take_ok)."""
        return ErrorOps.take_ok(flags, tensor)
    
    @staticmethod
    def take_err(flags, tensor):
        """Filter tensor to error samples (alias for ErrorOps.take_err)."""
        return ErrorOps.take_err(flags, tensor)
    
    @staticmethod
    def partition(flags, tensor):
        """Split tensor into (ok_tensor, err_tensor). Uses dynamic shapes."""
        return ErrorOps.partition(flags, tensor)
    
    @staticmethod
    def take_ok_p(flags, tensor, fill=0.0):
        """Return tensor with error samples replaced by fill. STATIC SHAPE - torch.compile safe.

        Alias for ErrorOps.take_ok_p.
        """
        return ErrorOps.take_ok_p(flags, tensor, fill)
    
    @staticmethod
    def take_err_p(flags, tensor, fill=0.0):
        """Return tensor with OK samples replaced by fill. STATIC SHAPE - torch.compile safe.

        Alias for ErrorOps.take_err_p.
        """
        return ErrorOps.take_err_p(flags, tensor, fill)
    
    # === Combinators ===
    @staticmethod
    def map_ok(flags, tensor, fn):
        """Apply fn only to OK samples."""
        return ErrorOps.map_ok(flags, tensor, fn)
    
    @staticmethod
    def map_err(flags, tensor, fn):
        """Apply fn only to error samples."""
        return ErrorOps.map_err(flags, tensor, fn)
    
    @staticmethod
    def and_then(flags, tensor, fn):
        """Chain: skip fn for error samples (short-circuit)."""
        return ErrorOps.and_then(flags, tensor, fn)
    
    @staticmethod
    def bind(flags, tensor, fn):
        """Chain: run fn for all, accumulate errors."""
        return ErrorOps.bind(flags, tensor, fn)
    
    @staticmethod
    def guard(flags, tensor, predicate, code, location, severity=None, config=None):
        """Push error where predicate is False."""
        if severity is None:
            severity = Severity.ERROR
        return ErrorOps.guard(flags, tensor, predicate, code, location, severity, config)
    
    @staticmethod
    def recover_with_fallback(flags, tensor, fallback, location, *, config=None):
        """Replace error samples with fallback value."""
        return ErrorOps.recover_with_fallback(flags, tensor, fallback, location, config=config)
    
    @staticmethod
    def all_ok(flags):
        """Scalar bool: True if ALL samples are OK."""
        return ErrorOps.all_ok(flags)
    
    @staticmethod
    def any_err(flags):
        """Scalar bool: True if ANY sample has errors (alias for has_any)."""
        return ErrorOps.any_err(flags)
    
    @staticmethod
    def ensure_mask(flags, predicate, code, location, severity=None, config=None):
        """Push error where predicate is False."""
        if severity is None:
            severity = Severity.ERROR
        return ErrorOps.ensure_mask(flags, predicate, code, location, severity, config)
    
    @staticmethod
    def map_err_flags(flags, fn):
        """Apply fn to flags of error samples."""
        return ErrorOps.map_err_flags(flags, fn)
    
    @staticmethod
    def partition_many(flags, *tensors):
        """Partition multiple tensors by error status."""
        return ErrorOps.partition_many(flags, *tensors)
    
    @staticmethod
    def clear(flags, code, config=None):
        """Clear specific error code from flags."""
        return ErrorOps.clear(flags, code, config)
    
    @staticmethod
    def push_scalar(flags, code, location, severity=None, config=None):
        """Push scalar code to all samples."""
        if severity is None:
            severity = Severity.ERROR
        return ErrorOps.push_scalar(flags, code, location, severity, config)
    
    @staticmethod
    def has_domain(flags, domain, config=None):
        """Per-sample bool mask for error domain."""
        return ErrorOps.has_domain(flags, domain, config)
    
    @staticmethod
    def has_fallback(flags, config=None):
        """Per-sample bool mask for fallback values."""
        return ErrorOps.has_fallback(flags, config)
    
    @staticmethod
    def get_first_code(flags):
        """Get first error code per sample."""
        return ErrorOps.get_first_code(flags)
    
    @staticmethod
    def get_first_location(flags):
        """Get first error location per sample."""
        return ErrorOps.get_first_location(flags)
    
    @staticmethod
    def get_first_severity(flags):
        """Get first error severity per sample."""
        return ErrorOps.get_first_severity(flags)
    
    # === Recovery ===
    @staticmethod
    def replace(t: Tensor, value: float = 0.0, *, targets: List[ReplaceTarget]) -> Tensor:
        """
        Replace target values in a tensor with a replacement value.
        
        Gradient-safe: gradients flow through non-replaced values,
        replaced positions receive zero gradient. Unlike torch.zeros_like(),
        this maintains gradient connections needed for torch.compile backward pass.
        
        Args:
            t: Input tensor
            value: Replacement value (default 0.0)
            targets: Required list of values to replace. Can include:
                - err.NAN or 'nan': Replace NaN values
                - err.INF or 'inf': Replace all Inf values (+/-)
                - 'posinf': Replace only +Inf
                - 'neginf': Replace only -Inf
                - Any float/int: Replace that exact numerical value
            
        Returns:
            Tensor with target values replaced
            
        Examples:
            # Replace NaN with 0:
            z = err.replace(z, value=0.0, targets=[err.NAN])
            
            # Replace both NaN and Inf with 0:
            z = err.replace(z, value=0.0, targets=[err.NAN, err.INF])
            
            # Replace specific numerical values:
            z = err.replace(z, value=-1.0, targets=[0.0, -999.0])
        
        Raises:
            ValueError: If targets is empty
        """
        if not targets:
            raise ValueError("targets cannot be empty - specify what values to replace (e.g., [err.NAN, err.INF])")
        
        mask = torch.zeros_like(t, dtype=torch.bool)
        
        for target in targets:
            if target in (ErrorCode.NAN, 'nan'):
                mask = mask | torch.isnan(t)
            elif target in (ErrorCode.INF, 'inf'):
                mask = mask | torch.isinf(t)
            elif target == 'posinf':
                mask = mask | (t == float('inf'))
            elif target == 'neginf':
                mask = mask | (t == float('-inf'))
            elif isinstance(target, (int, float)):
                mask = mask | (t == target)
        
        return torch.where(mask, value, t)


class _FlagsNamespace:
    """
    Python boundary operations namespace.
    
    Use for debugging, inspection, and pretty-printing.
    NOT for use inside torch.compile regions.
    """
    @staticmethod
    def unpack(flags_tensor, sample_idx=0, config=None):
        """Unpack errors from flags tensor."""
        return ErrorFlags.unpack(flags_tensor, sample_idx, config)
    
    @staticmethod
    def repr(flags_tensor, config=None):
        """Get string representation of errors."""
        return ErrorFlags.repr(flags_tensor, config)
    
    @staticmethod
    def summary(flags_tensor, config=None):
        """Get batch summary of errors."""
        return ErrorFlags.summary(flags_tensor, config)


# Singleton instances
err = _ErrNamespace()
flags = _FlagsNamespace()

__all__ = [
    'err',
    'flags',
    'ErrorOps',
    'ErrorFlags',
    'UnpackedError',
]
