"""
TorchGuard: Compile-time error handling for torch.compile() regions.

Zero external dependencies (only PyTorch).

Primary API:
    from torchguard import err, flags, error_t, tracked, tensorcheck
    
    @tracked
    class MyModel(nn.Module):
        @tensorcheck
        def forward(self, x: Tensor) -> tuple[Tensor, Tensor[error_t, "batch num_words"]]:
            f = err.new(x)
            f = err.push(f, err.NAN, location, where=torch.isnan(z).any(-1))
            return output, f
    
    # At Python boundary
    output, f = model(x)
    if err.has_any(f):
        print(flags.repr(f))
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# PRIMARY API - the only imports most users need
# ═══════════════════════════════════════════════════════════════════════════════

# Compiled-safe operations namespace
from .src.err import err

# Python boundary operations namespace  
from .src.err import flags, UnpackedError

# Type alias (for annotations)
from .src.typing import error_t

# Decorators
from .src.decorators import tracked, tensorcheck, as_result, as_exception, unwrap

# ═══════════════════════════════════════════════════════════════════════════════
# SECONDARY EXPORTS - for advanced usage
# ═══════════════════════════════════════════════════════════════════════════════

# Core types
from .src.core import (
    # Codes
    ErrorCode,
    ErrorDomain,
    # Severity
    Severity,
    # Config
    AccumulationConfig,
    DEFAULT_CONFIG,
    Dedupe,
    ErrorConfig,
    Order,
    Priority,
    # Location (for testing/internal use - prefer passing modules to push())
    ErrorLocation,
    # Constants
    CODE_BITS,
    CODE_MASK,
    CODE_SHIFT,
    LOCATION_BITS,
    LOCATION_MASK,
    LOCATION_SHIFT,
    SEVERITY_BITS,
    SEVERITY_MASK,
    SEVERITY_SHIFT,
    SLOT_BITS,
    SLOT_MASK,
    SLOTS_PER_WORD,
)

# Control Flow DSL
from .src.control import AND, HAS, IF, IS, NOT, OR

# Result type
from .src.err.result import Ok, Err, Result

# Validation errors (for catching)
from .src.typing import (
    ValidationError,
    DimensionMismatchError,
    DTypeMismatchError,
    DeviceMismatchError,
    InvalidParameterError,
    TypeMismatchError,
    InvalidReturnTypeError,
)

# Helper functions (convenience wrappers with auto-location)
from .src.err.helpers import (
    find,
    fix,
    flag_inf,
    flag_nan,
    flag_oob_indices,
    has_err,
    push,
)

# Experimental backend (float64 storage + int64 view)
from .src import experimental


__all__ = [
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIMARY API
    # ═══════════════════════════════════════════════════════════════════════════
    'err',           # Compiled-safe operations + error codes
    'flags',         # Python boundary operations
    'error_t',       # Type alias (annotations only)
    'tracked',       # @tracked decorator for classes
    'tensorcheck',   # @tensorcheck decorator for methods
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECONDARY EXPORTS
    # ═══════════════════════════════════════════════════════════════════════════
    # Unpacked error type
    'UnpackedError',
    # Core types
    'ErrorCode',
    'ErrorDomain',
    'ErrorLocation',
    'Severity',
    # Configuration
    'AccumulationConfig',
    'Priority',
    'Order',
    'Dedupe',
    'ErrorConfig',
    'DEFAULT_CONFIG',
    # Constants
    'SLOT_BITS',
    'SLOTS_PER_WORD',
    'SEVERITY_SHIFT',
    'SEVERITY_BITS',
    'SEVERITY_MASK',
    'CODE_SHIFT',
    'CODE_BITS',
    'CODE_MASK',
    'LOCATION_SHIFT',
    'LOCATION_BITS',
    'LOCATION_MASK',
    'SLOT_MASK',
    # Helper functions
    'has_err',
    'find',
    'push',
    'fix',
    'flag_nan',
    'flag_inf',
    'flag_oob_indices',
    # Control Flow DSL
    'IF',
    'HAS',
    'IS',
    'OR',
    'AND',
    'NOT',
    # Decorators (secondary)
    'as_result',
    'as_exception',
    'unwrap',
    # Result type
    'Ok',
    'Err',
    'Result',
    # Validation errors
    'ValidationError',
    'DimensionMismatchError',
    'DTypeMismatchError',
    'DeviceMismatchError',
    'InvalidParameterError',
    'TypeMismatchError',
    'InvalidReturnTypeError',
    # Experimental backend
    'experimental',
]

__version__ = '0.1.0'
