"""
Tensor-based control-flow DSL for error handling in compiled models.

This module provides:
- Predicates: HAS, IS, OR, AND, NOT
- Control-flow: IF(...).ELIF(...).ELSE(...)

IMPORTANT:
- Inside compiled code, use HAS/IS/OR/AND/NOT + IF/ELIF/ELSE.
- At the Python boundary, use has_err(flags) from checks.py for Python bool.

Example:
    from src.utils.errors.compiled.control import IF, HAS, IS, OR
    
    z, flags = (
        IF(IS(ErrorCode.NAN, flags), lambda: fix(z, flags, self))
          .ELIF(IS(ErrorCode.OUT_OF_BOUNDS, flags), lambda: handle_oob(z, flags))
          .ELSE(lambda: (z, flags))
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, TypeVar

import torch
from torch import Tensor

# IMPORTANT: Import directly from helpers.py, NOT from __init__.py
# This avoids circular imports since __init__.py exports from control.py
from .err.helpers import find
from .core.config import DEFAULT_CONFIG, ErrorConfig

T = TypeVar("T")


def _clone_outputs(result: T) -> T:
    """
    Clone tensor outputs to avoid aliasing issues with torch.cond.
    
    torch.cond doesn't support input-to-output aliasing in higher-order ops.
    This helper ensures returned tensors are new objects.
    
    Args:
        result: Any value, possibly containing tensors
    
    Returns:
        The same structure with all tensors cloned
    """
    if isinstance(result, Tensor):
        return result.clone()
    elif isinstance(result, tuple):
        return tuple(_clone_outputs(x) for x in result)
    elif isinstance(result, list):
        return [_clone_outputs(x) for x in result]
    elif isinstance(result, dict):
        return {k: _clone_outputs(v) for k, v in result.items()}
    else:
        return result


__all__ = [
    'HAS',
    'IS',
    'OR',
    'AND',
    'NOT',
    'IF',
]


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_scalar(cond: Tensor, name: str = "condition") -> Tensor:
    """
    Ensure that cond is a 0-D bool Tensor.
    
    Provides clear error messages instead of cryptic torch.cond failures.
    
    Args:
        cond (Tensor): Tensor candidate
        name (str): Human-readable name for error messages
    
    Returns:
        (Tensor): cond unchanged if valid
    
    Raises:
        TypeError: if cond is not a Tensor or is non-bool
        ValueError: if cond is not 0-D (scalar)
    """
    if not isinstance(cond, Tensor):
        raise TypeError(f"{name} must be a Tensor, got {type(cond).__name__}")
    
    if cond.ndim != 0:
        raise ValueError(
            f"{name} must be 0-D (scalar), got shape {tuple(cond.shape)}. "
            f"Did you forget `.any()` / `.all()` / use a per-sample mask?"
        )
    
    if cond.dtype is not torch.bool:
        raise TypeError(
            f"{name} must be a bool Tensor, got dtype {cond.dtype}. "
            f"Use HAS(), IS(), OR(), AND(), NOT(), or comparison ops."
        )
    
    return cond


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICATES
# ═══════════════════════════════════════════════════════════════════════════════

def HAS(flags: Tensor) -> Tensor:
    """
    Tensor predicate: any error in the entire batch?
    
    For compiled code: returns 0-D bool Tensor for torch.cond/torch.where.
    At Python boundary: use `has_err(flags)` from checks.py instead.
    
    Args:
        flags (Tensor): error_t flags tensor, shape (N, num_words), dtype int64
    
    Returns:
        (Tensor): 0-D bool Tensor, True if any slot is non-zero
    
    Example:
        # Inside compiled model
        z = torch.where(HAS(flags), fallback_tensor, z)
        
        # Or with IF DSL
        z, flags = IF(HAS(flags), lambda: fix(...)).ELSE(lambda: (z, flags))
    """
    cond = (flags != 0).any()
    return _ensure_scalar(cond, "HAS(flags)")


def IS(code: int, flags: Tensor, *, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
    """
    Tensor predicate: does any sample have this specific error code?
    
    Essentially: find(code, flags, config).any() with validation.
    
    Args:
        code (int): Error code integer (e.g. ErrorCode.NAN)
        flags (Tensor): error_t flags tensor, shape (N, num_words)
        config (ErrorConfig): ErrorConfig used for bit layout
    
    Returns:
        (Tensor): 0-D bool Tensor, True if any sample has this error code
    
    Example:
        cond_nan = IS(ErrorCode.NAN, flags)
        cond_oob = IS(ErrorCode.OUT_OF_BOUNDS, flags)
        
        z, flags = (
            IF(cond_nan, lambda: fix(z, flags, self))
              .ELIF(cond_oob, lambda: self.handle_oob(z, flags))
              .ELSE(lambda: (z, flags))
        )
    """
    cond = find(code, flags, config=config).any()
    return _ensure_scalar(cond, f"IS({code}, flags)")


def OR(*conds: Tensor) -> Tensor:
    """
    Tensor OR for 0-D bool Tensors.
    
    NOTE: Does NOT short-circuit. All conditions are evaluated.
    Uses stack + any() for a single reduction kernel.
    
    Args:
        *conds (Tensor): One or more 0-D bool Tensors
    
    Returns:
        (Tensor): 0-D bool Tensor, logical OR of all inputs
    
    Raises:
        ValueError: if no conditions provided
    
    Example:
        cond_bad_numeric = OR(
            IS(ErrorCode.NAN, flags),
            IS(ErrorCode.INF, flags),
        )
    """
    if not conds:
        raise ValueError("OR() requires at least one condition")
    
    validated = [_ensure_scalar(c, "OR condition") for c in conds]
    stacked = torch.stack(validated)
    out = stacked.any()
    return _ensure_scalar(out, "OR(...) result")


def AND(*conds: Tensor) -> Tensor:
    """
    Tensor AND for 0-D bool Tensors.
    
    NOTE: Does NOT short-circuit. All conditions are evaluated.
    Uses stack + all() for a single reduction kernel.
    
    Args:
        *conds (Tensor): One or more 0-D bool Tensors
    
    Returns:
        (Tensor): 0-D bool Tensor, logical AND of all inputs
    
    Raises:
        ValueError: if no conditions provided
    
    Example:
        cond_severe = AND(
            IS(ErrorCode.NAN, flags),
            NOT(IS(ErrorCode.FALLBACK_VALUE, flags)),
        )
    """
    if not conds:
        raise ValueError("AND() requires at least one condition")
    
    validated = [_ensure_scalar(c, "AND condition") for c in conds]
    stacked = torch.stack(validated)
    out = stacked.all()
    return _ensure_scalar(out, "AND(...) result")


def NOT(cond: Tensor) -> Tensor:
    """
    Tensor logical negation for a 0-D bool Tensor.
    
    Args:
        cond (Tensor): 0-D bool Tensor
    
    Returns:
        (Tensor): 0-D bool Tensor, ~cond
    
    Example:
        cond_not_fixed = NOT(IS(ErrorCode.FALLBACK_VALUE, flags))
    """
    cond = _ensure_scalar(cond, "NOT condition")
    out = ~cond
    return _ensure_scalar(out, "NOT(...) result")


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL FLOW
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Branch(Generic[T]):
    """Single conditional branch."""
    cond: Tensor
    fn: Callable[[], T]


class _IfChain(Generic[T]):
    """
    Internal IF/ELIF/ELSE chain builder.
    
    Use via IF(cond, fn).ELIF(cond2, fn2).ELSE(else_fn).
    
    Eager mode: Uses Python if/elif/else for debugging.
    Compiled mode: Uses nested torch.cond calls.
    """
    
    def __init__(self, first_branch: _Branch[T]) -> None:
        """
        Initialize with first branch.
        
        Args:
            first_branch (_Branch[T]): Initial IF branch
        """
        self.__branches: List[_Branch[T]] = [first_branch]
    
    def ELIF(self, cond: Tensor, fn: Callable[[], T]) -> _IfChain[T]:
        """
        Add an elif branch.
        
        Args:
            cond (Tensor): 0-D bool tensor condition
            fn (Callable[[], T]): Branch body returning T
        
        Returns:
            (_IfChain[T]): Self for chaining
        """
        cond = _ensure_scalar(cond, "ELIF condition")
        self.__branches.append(_Branch(cond, fn))
        return self
    
    def ELSE(self, else_fn: Callable[[], T]) -> T:
        """
        Finalize the chain with an else branch.
        
        Performance Note:
            Eager fallback uses Python control flow (faster for debugging).
            Compiled mode uses nested torch.cond (fully traceable).
        
        Constraints (for torch.compile):
            - All branches must return the same type/structure T
            - Branch functions must be side-effect-free
        
        Args:
            else_fn (Callable[[], T]): Else branch body
        
        Returns:
            (T): Result from the taken branch
        """
        if not self.__branches:
            return else_fn()
        
        # Eager mode: Python control flow
        if not torch.compiler.is_compiling():
            for br in self.__branches:
                if br.cond.item():
                    return br.fn()
            return else_fn()
        
        # Compiled mode: nested torch.cond
        # Wrap functions to clone outputs, avoiding aliasing issues with torch.cond
        def wrap_fn(fn: Callable[[], T]) -> Callable[[], T]:
            def wrapped() -> T:
                return _clone_outputs(fn())
            return wrapped
        
        wrapped_else = wrap_fn(else_fn)
        wrapped_branches = [(br.cond, wrap_fn(br.fn)) for br in self.__branches]
        
        def build(idx: int) -> T:
            if idx == len(wrapped_branches) - 1:
                cond, fn = wrapped_branches[idx]
                return torch.cond(cond, fn, wrapped_else)
            
            cond, fn = wrapped_branches[idx]
            return torch.cond(cond, fn, lambda: build(idx + 1))
        
        return build(0)


def IF(cond: Tensor, then_fn: Callable[[], T]) -> _IfChain[T]:
    """
    Start an IF/ELIF/ELSE chain.
    
    Args:
        cond (Tensor): 0-D bool tensor (use HAS/IS/OR/AND/NOT or comparisons)
        then_fn (Callable[[], T]): Callable with no args, returns T
    
    Returns:
        (_IfChain[T]): Chain object for .ELIF(...).ELSE(...)
    
    Example:
        z, flags = (
            IF(IS(ErrorCode.NAN, flags), lambda: fix(z, flags, self))
              .ELIF(IS(ErrorCode.OUT_OF_BOUNDS, flags), lambda: self.handle_oob(z, flags))
              .ELSE(lambda: (z, flags))
        )
    
    Notes:
        - All branches must return same type for torch.compile
        - Eager mode uses Python control flow
        - Compiled mode uses nested torch.cond
    """
    cond = _ensure_scalar(cond, "IF condition")
    return _IfChain(_Branch(cond, then_fn))
