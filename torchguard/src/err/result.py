"""
Minimal Result type for torchguard.

Zero external dependencies - pure Python implementation of Ok/Err pattern.

Usage:
    from torchguard import Ok, Err, Result, as_result, as_exception, unwrap
    
    # Wrap exception-throwing code
    @as_result
    def divide(a, b):
        return a / b
    
    result = divide(10, 0)  # Returns Err(ZeroDivisionError)
    
    # Convert Result back to exceptions
    @as_exception
    def safe_divide(a, b) -> Result[float, Exception]:
        if b == 0:
            return Err(ValueError("division by zero"))
        return Ok(a / b)
    
    value = safe_divide(10, 2)  # Returns 5.0 or raises
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Union

__all__ = ['Ok', 'Err', 'Result', 'as_result', 'as_exception', 'unwrap']

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E', bound=Exception)


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Success value wrapper."""
    value: T
    
    def is_ok(self) -> bool:
        """Check if this is an Ok value."""
        return True
    
    def is_err(self) -> bool:
        """Check if this is an Err value."""
        return False
    
    def unwrap(self) -> T:
        """Get the success value."""
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Get success value or return default."""
        return self.value
    
    def unwrap_err(self) -> Exception:
        """Get error value. Raises ValueError for Ok."""
        raise ValueError("Called unwrap_err on Ok")
    
    def map(self, fn: Callable[[T], U]) -> Result[U, Exception]:
        """Apply function to success value."""
        return Ok(fn(self.value))
    
    def map_err(self, fn: Callable[[Exception], Any]) -> Result[T, Exception]:
        """Apply function to error value (no-op for Ok)."""
        return self
    
    def and_then(self, fn: Callable[[T], Result[U, Exception]]) -> Result[U, Exception]:
        """Chain operations on success value."""
        return fn(self.value)
    
    @property
    def ok_value(self) -> T:
        """Alias for value (compatibility)."""
        return self.value
    
    @property
    def err_value(self) -> None:
        """Error value (None for Ok)."""
        return None


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Error value wrapper."""
    error: E
    
    def is_ok(self) -> bool:
        """Check if this is an Ok value."""
        return False
    
    def is_err(self) -> bool:
        """Check if this is an Err value."""
        return True
    
    def unwrap(self) -> T:
        """Get success value. Raises the wrapped error for Err."""
        raise self.error
    
    def unwrap_or(self, default: T) -> T:
        """Get success value or return default."""
        return default
    
    def unwrap_err(self) -> E:
        """Get the error value."""
        return self.error
    
    def map(self, fn: Callable[[Any], Any]) -> Result[Any, E]:
        """Apply function to success value (no-op for Err)."""
        return self
    
    def map_err(self, fn: Callable[[E], Any]) -> Result[Any, Any]:
        """Apply function to error value."""
        return Err(fn(self.error))
    
    def and_then(self, fn: Callable[[Any], Result]) -> Result[Any, E]:
        """Chain operations (no-op for Err)."""
        return self
    
    @property
    def ok_value(self) -> None:
        """Success value (None for Err)."""
        return None
    
    @property
    def err_value(self) -> E:
        """Alias for error (compatibility)."""
        return self.error


# Type alias
Result = Union[Ok[T], Err[E]]


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSION DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def as_result(func: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """
    Wrap a function that may raise into one that returns Result.
    
    Converts exceptions to Err, successful returns to Ok.
    
    Example:
        @as_result
        def divide(a, b):
            return a / b
        
        result = divide(10, 0)  # Returns Err(ZeroDivisionError(...))
        if result.is_ok():
            print(result.unwrap())
        else:
            print(f"Error: {result.unwrap_err()}")
    
    Args:
        func: Function that may raise exceptions
    
    Returns:
        Function that returns Result[T, Exception]
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Result[T, Exception]:
        try:
            return Ok(func(*args, **kwargs))
        except Exception as e:
            return Err(e)
    return wrapper


def as_exception(func: Callable[..., Result[T, E]]) -> Callable[..., T]:
    """
    Unwrap a Result-returning function to raise on Err.
    
    Converts Err to raised exceptions, Ok to returned values.
    
    Example:
        def safe_divide(a, b) -> Result[float, ValueError]:
            if b == 0:
                return Err(ValueError("division by zero"))
            return Ok(a / b)
        
        @as_exception
        def divide(a, b):
            return safe_divide(a, b)
        
        divide(10, 2)  # Returns 5.0
        divide(10, 0)  # Raises ValueError
    
    Args:
        func: Function returning Result[T, E]
    
    Returns:
        Function that returns T or raises E
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        result = func(*args, **kwargs)
        if result.is_err():
            raise result.unwrap_err()
        return result.unwrap()
    return wrapper


def unwrap(result: Result[T, E]) -> T:
    """
    Unwrap a Result, raising if Err.
    
    Convenience function for inline unwrapping.
    
    Example:
        result = some_operation()
        value = unwrap(result)  # Raises if Err
    
    Args:
        result: Result to unwrap
    
    Returns:
        The success value
    
    Raises:
        The wrapped exception if result is Err
    """
    if result.is_err():
        raise result.unwrap_err()
    return result.unwrap()

