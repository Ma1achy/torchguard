"""
Local validation error types for torchguard.

These replace the external Error class dependency for validation purposes.
All error types inherit from ValidationError for easy catching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

__all__ = [
    'ValidationError',
    'DimensionMismatchError',
    'DTypeMismatchError',
    'DeviceMismatchError',
    'InvalidParameterError',
    'TypeMismatchError',
    'InvalidReturnTypeError',
]


@dataclass
class ValidationError(Exception):
    """
    Base validation error for tensor typing.
    
    All torchguard validation errors inherit from this class.
    
    Attributes:
        message: Human-readable error description
        context: Additional context as key-value pairs
    """
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({ctx})"
        return self.message
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, context={self.context!r})"


@dataclass
class DimensionMismatchError(ValidationError):
    """
    Tensor dimension mismatch error.
    
    Raised when a tensor's shape doesn't match the expected annotation.
    
    Attributes:
        expected: Expected dimension value or pattern
        actual: Actual dimension value
        dim_name: Name of the dimension (if named)
        parameter: Parameter name where mismatch occurred
        function: Function name where validation failed
    """
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    dim_name: Optional[str] = None
    parameter: Optional[str] = None
    function: Optional[str] = None


@dataclass
class DTypeMismatchError(ValidationError):
    """
    Tensor dtype mismatch error.
    
    Raised when a tensor's dtype doesn't match the expected type.
    
    Attributes:
        expected: Expected dtype
        actual: Actual dtype
        parameter: Parameter name where mismatch occurred
        function: Function name where validation failed
    """
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    parameter: Optional[str] = None
    function: Optional[str] = None


@dataclass
class DeviceMismatchError(ValidationError):
    """
    Tensor device mismatch error.
    
    Raised when a tensor's device doesn't match expected device.
    
    Attributes:
        expected: Expected device string
        actual: Actual device string
        parameter: Parameter name where mismatch occurred
        function: Function name where validation failed
    """
    expected: Optional[str] = None
    actual: Optional[str] = None
    parameter: Optional[str] = None
    function: Optional[str] = None


@dataclass
class InvalidParameterError(ValidationError):
    """
    Invalid function parameter error.
    
    Raised when a function parameter fails validation.
    
    Attributes:
        parameter: Name of the invalid parameter
        function: Function name where validation failed
        num_errors: Number of validation errors (for aggregated errors)
    """
    parameter: Optional[str] = None
    function: Optional[str] = None
    num_errors: Optional[int] = None


@dataclass
class TypeMismatchError(ValidationError):
    """
    Type mismatch in return value error.
    
    Raised when a function's return value doesn't match annotation.
    
    Attributes:
        expected: Expected return type
        actual: Actual return type
        function: Function name where validation failed
        num_errors: Number of validation errors (for tuple returns)
    """
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    function: Optional[str] = None
    num_errors: Optional[int] = None


@dataclass
class InvalidReturnTypeError(ValidationError):
    """
    Invalid return type error.
    
    Raised when a function returns an unexpected type.
    
    Attributes:
        expected: Expected return type
        actual: Actual return type  
        function: Function name where validation failed
    """
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    function: Optional[str] = None

