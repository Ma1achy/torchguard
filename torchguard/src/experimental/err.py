"""
Experimental err namespace wrapper.

Mirrors the stable err namespace but uses Float64ErrorOps.
"""
from __future__ import annotations

from typing import Optional, List, Union

from typing import Optional
import torch
from torch import Tensor

from .ops import Float64ErrorOps
from ..core import ErrorCode, ErrorDomain, Severity
from ..core.config import ErrorConfig, get_config

# Type alias for replace() targets parameter
ReplaceTarget = Union[int, str, float]

# Experimental backend uses float32 by default for better torch.compile compatibility
_EXPERIMENTAL_DEFAULT = None  # Use global config


class _XErrNamespace:
    """
    Experimental compiled-safe error operations namespace.
    
    Returns float32 tensors by default (configurable via ErrorConfig.flag_dtype).
    float32 default provides better torch.compile compatibility.
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
    def new(reference, config: Optional[ErrorConfig] = None):
        """Create empty error flags tensor from reference. Returns float32 (default) or float64."""
        return Float64ErrorOps.new(reference, config or get_config())
    
    @staticmethod
    def new_t(batch_size, device=None, config: Optional[ErrorConfig] = None):
        """Create empty error flags tensor from batch size. Returns float32 (default) or float64."""
        return Float64ErrorOps.new_t(batch_size, device, config or get_config())
    
    @staticmethod
    def from_code(code, location, batch_size, device=None, severity=None, config: Optional[ErrorConfig] = None):
        """Create flags with a single error code. Returns float32 (default) or float64."""
        if severity is None:
            severity = Severity.ERROR
        return Float64ErrorOps.from_code(code, location, batch_size, device, severity, config or get_config())
    
    # === Recording ===
    @staticmethod
    def push(flags, code, location, severity=None, config: Optional[ErrorConfig] = None, *, where=None):
        """Push error code to flags. Preserves flag dtype (float32 or float64)."""
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
            return Float64ErrorOps.push(flags, code, loc, severity, config or get_config())
        else:
            # Use high-level push with where support
            cfg = config or get_config()
            from ..core.location import ErrorLocation
            if isinstance(location, str):
                loc = ErrorLocation.register(location)
            elif isinstance(location, int):
                loc = location
            else:
                loc = ErrorLocation.UNKNOWN
            if severity is None:
                severity = ErrorCode.default_severity(code)
            
            int_dtype = cfg.torch_int_dtype
            # Use compile-safe tensor creation (avoid shape[0] extraction)
            # Create a template tensor from flags' first column for broadcasting
            template = flags[:, 0]  # Shape (N,)
            if where is None:
                # All samples get the code
                code_tensor = torch.full_like(template, code, dtype=int_dtype)
            else:
                # Only samples where 'where' is True get the code
                code_tensor = torch.where(
                    where, 
                    torch.full_like(template, code, dtype=int_dtype),
                    torch.full_like(template, ErrorCode.OK, dtype=int_dtype)
                )
            
            return Float64ErrorOps.push(flags, code_tensor, loc, severity, cfg)
    
    @staticmethod
    def merge(*flag_tensors, config: Optional[ErrorConfig] = None):
        """Merge multiple flag tensors. Returns same dtype as input (float32 default)."""
        return Float64ErrorOps.merge(*flag_tensors, config=config or get_config())
    
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
    def has_nan(flags, config: Optional[ErrorConfig] = None):
        """Per-sample bool mask for NaN errors."""
        return Float64ErrorOps.has_nan(flags, config or get_config())
    
    @staticmethod
    def has_inf(flags, config: Optional[ErrorConfig] = None):
        """Per-sample bool mask for Inf errors."""
        return Float64ErrorOps.has_inf(flags, config or get_config())
    
    @staticmethod
    def has_code(flags, code, config: Optional[ErrorConfig] = None):
        """Per-sample bool mask for specific error code."""
        return Float64ErrorOps.has_code(flags, code, config or get_config())
    
    @staticmethod
    def has_critical(flags, config: Optional[ErrorConfig] = None):
        """Per-sample bool mask for critical errors."""
        return Float64ErrorOps.has_critical(flags, config or get_config())
    
    @staticmethod
    def count_errors(flags, config: Optional[ErrorConfig] = None):
        """Count errors per sample."""
        return Float64ErrorOps.count_errors(flags, config or get_config())
    
    @staticmethod
    def max_severity(flags, config: Optional[ErrorConfig] = None):
        """Get max severity per sample."""
        return Float64ErrorOps.max_severity(flags, config or get_config())
    
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
    def guard(flags, tensor, predicate, code, location, severity=None, config: Optional[ErrorConfig] = None):
        """Push error where predicate is False."""
        if severity is None:
            severity = Severity.ERROR
        return Float64ErrorOps.guard(flags, tensor, predicate, code, location, severity, config or get_config())
    
    @staticmethod
    def recover_with_fallback(flags, tensor, fallback, location, *, config: Optional[ErrorConfig] = None):
        """Replace error samples with fallback value."""
        return Float64ErrorOps.recover_with_fallback(flags, tensor, fallback, location, config=config or get_config())
    
    @staticmethod
    def all_ok(flags):
        """Scalar bool: True if ALL samples are OK."""
        return Float64ErrorOps.all_ok(flags)
    
    @staticmethod
    def any_err(flags):
        """Scalar bool: True if ANY sample has errors."""
        return Float64ErrorOps.any_err(flags)
    
    @staticmethod
    def ensure_mask(flags, predicate, code, location, severity=None, config: Optional[ErrorConfig] = None):
        """Push error where predicate is False."""
        if severity is None:
            severity = Severity.ERROR
        return Float64ErrorOps.ensure_mask(flags, predicate, code, location, severity, config or get_config())
    
    @staticmethod
    def map_err_flags(flags, fn):
        """Apply fn to flags of error samples."""
        return Float64ErrorOps.map_err_flags(flags, fn)
    
    @staticmethod
    def partition_many(flags, *tensors):
        """Partition multiple tensors by error status."""
        return Float64ErrorOps.partition_many(flags, *tensors)
    
    @staticmethod
    def clear(flags, code, config: Optional[ErrorConfig] = None):
        """Clear specific error code from flags. Returns same dtype as input (float32 default)."""
        return Float64ErrorOps.clear(flags, code, config or get_config())
    
    @staticmethod
    def push_scalar(flags, code, location, severity=None, config: Optional[ErrorConfig] = None):
        """Push scalar code to all samples. Returns same dtype as input (float32 default)."""
        if severity is None:
            severity = Severity.ERROR
        return Float64ErrorOps.push_scalar(flags, code, location, severity, config or get_config())
    
    @staticmethod
    def has_domain(flags, domain, config: Optional[ErrorConfig] = None):
        """Per-sample bool mask for error domain."""
        return Float64ErrorOps.has_domain(flags, domain, config or get_config())
    
    @staticmethod
    def has_fallback(flags, config: Optional[ErrorConfig] = None):
        """Per-sample bool mask for fallback values."""
        return Float64ErrorOps.has_fallback(flags, config or get_config())
    
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
    def find(code, flags, config: Optional[ErrorConfig] = None):
        """Find which samples have a specific error code. Works with float64 flags."""
        return Float64ErrorOps.has_code(flags, code, config or get_config())
    
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
            
            # Use in IF/ELIF/ELSE DSL:
            z, flags = (
                IF(IS(err.NAN, flags), lambda: (err.replace(z, value=0.0, targets=[err.NAN]), flags))
                .ELIF(IS(err.INF, flags), lambda: (torch.clamp(z, min=-10.0, max=10.0), flags))
                .ELSE(lambda: (z, flags))
            )
        
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


# Singleton instance
err = _XErrNamespace()

