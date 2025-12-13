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
from .ops import ErrorOps
from ..core import ErrorCode, ErrorDomain, Severity
from ..core.config import DEFAULT_CONFIG, ErrorConfig

# Re-export UnpackedError for type hints
from .inspect import ErrorFlags, UnpackedError


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
        return ErrorOps.new(reference)
    
    @staticmethod
    def new_t(batch_size, device=None, config=DEFAULT_CONFIG):
        """Create empty error flags tensor from batch size."""
        return ErrorOps.new_t(batch_size, device, config)
    
    @staticmethod
    def from_code(code, location, batch_size, device=None, severity=None, config=DEFAULT_CONFIG):
        """Create flags with a single error code."""
        if severity is None:
            severity = Severity.ERROR
        return ErrorOps.from_code(code, location, batch_size, device, severity, config)
    
    # === Recording ===
    @staticmethod
    def push(flags, code, location, severity=None, config=DEFAULT_CONFIG, *, where=None):
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
            return ErrorOps.push(flags, code, loc, severity, config)
        else:
            # Use helpers.push for high-level API with where support
            from .helpers import push as helpers_push
            return helpers_push(flags, code, location, where=where, severity=severity, config=config)
    
    @staticmethod
    def merge(*flag_tensors, config=DEFAULT_CONFIG):
        """Merge multiple flag tensors."""
        return ErrorOps.merge(*flag_tensors, config=config)
    
    # === Checking ===
    @staticmethod
    def is_ok(flags):
        """Per-sample bool mask: True where NO errors."""
        return ErrorOps.is_ok(flags)
    
    @staticmethod
    def is_err(flags):
        """Per-sample bool mask: True where HAS errors."""
        return ErrorOps.is_err(flags)
    
    @staticmethod
    def has_any(flags):
        """Scalar bool: True if ANY sample has errors."""
        return ErrorOps.any_err(flags)
    
    @staticmethod
    def has_nan(flags, config=DEFAULT_CONFIG):
        """Per-sample bool mask for NaN errors."""
        return ErrorOps.has_nan(flags, config)
    
    @staticmethod
    def has_inf(flags, config=DEFAULT_CONFIG):
        """Per-sample bool mask for Inf errors."""
        return ErrorOps.has_inf(flags, config)
    
    @staticmethod
    def has_code(flags, code, config=DEFAULT_CONFIG):
        """Per-sample bool mask for specific error code."""
        return ErrorOps.has_code(flags, code, config)
    
    @staticmethod
    def has_critical(flags, config=DEFAULT_CONFIG):
        """Per-sample bool mask for critical errors."""
        return ErrorOps.has_critical(flags, config)
    
    @staticmethod
    def count_errors(flags, config=DEFAULT_CONFIG):
        """Count errors per sample."""
        return ErrorOps.count_errors(flags, config)
    
    @staticmethod
    def max_severity(flags, config=DEFAULT_CONFIG):
        """Get max severity per sample."""
        return ErrorOps.max_severity(flags, config)
    
    # === Filtering ===
    @staticmethod
    def get_ok(flags):
        """Filter flags to OK samples only."""
        return ErrorOps.get_ok(flags)
    
    @staticmethod
    def get_err(flags):
        """Filter flags to error samples only."""
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
    def guard(flags, tensor, predicate, code, location, severity=None, config=DEFAULT_CONFIG):
        """Push error where predicate is False."""
        if severity is None:
            severity = Severity.ERROR
        return ErrorOps.guard(flags, tensor, predicate, code, location, severity, config)
    
    @staticmethod
    def recover_with_fallback(flags, tensor, fallback, location, *, config=DEFAULT_CONFIG):
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
    def ensure_mask(flags, predicate, code, location, severity=None, config=DEFAULT_CONFIG):
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
    def clear(flags, code, config=DEFAULT_CONFIG):
        """Clear specific error code from flags."""
        return ErrorOps.clear(flags, code, config)
    
    @staticmethod
    def push_scalar(flags, code, location, severity=None, config=DEFAULT_CONFIG):
        """Push scalar code to all samples."""
        if severity is None:
            severity = Severity.ERROR
        return ErrorOps.push_scalar(flags, code, location, severity, config)
    
    @staticmethod
    def has_domain(flags, domain, config=DEFAULT_CONFIG):
        """Per-sample bool mask for error domain."""
        return ErrorOps.has_domain(flags, domain, config)
    
    @staticmethod
    def has_fallback(flags, config=DEFAULT_CONFIG):
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


class _FlagsNamespace:
    """
    Python boundary operations namespace.
    
    Use for debugging, inspection, and pretty-printing.
    NOT for use inside torch.compile regions.
    """
    @staticmethod
    def unpack(flags_tensor, sample_idx=0, config=DEFAULT_CONFIG):
        """Unpack errors from flags tensor."""
        return ErrorFlags.unpack(flags_tensor, sample_idx, config)
    
    @staticmethod
    def repr(flags_tensor, config=DEFAULT_CONFIG):
        """Get string representation of errors."""
        return ErrorFlags.repr(flags_tensor, config)
    
    @staticmethod
    def summary(flags_tensor, config=DEFAULT_CONFIG):
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
