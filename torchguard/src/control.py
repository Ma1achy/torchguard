"""
Tensor-based control-flow DSL for error handling in compiled models.

This module provides:
- Predicates: HAS, IS, OR, AND, NOT
- Control-flow: IF(...).ELIF(...).ELSE(...)

IMPORTANT:
- Inside compiled code, use HAS/IS/OR/AND/NOT + IF/ELIF/ELSE.
- At the Python boundary, use has_err(flags) from checks.py for Python bool.

Implementation Note:
    Uses torch.where instead of torch.cond for inductor backward pass compatibility.
    Both branches are always evaluated (eager), then results are selected.
    This is slightly less efficient but ensures gradients flow correctly.

Example:
    from src.utils.errors.compiled.control import IF, HAS, IS, OR
    
    z, flags = (
        IF(IS(ErrorCode.NAN, flags), lambda: fix(z, flags, self))
          .ELIF(IS(ErrorCode.OUT_OF_BOUNDS, flags), lambda: handle_oob(z, flags))
          .ELSE(lambda: (z, flags))
    )
"""
from __future__ import annotations

# Standard library
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar, Union

# Third-party
import torch
from torch import Tensor

# Internal - Import directly from helpers.py, NOT from __init__.py
# This avoids circular imports since __init__.py exports from control.py
from .core.config import ErrorConfig, get_config
from .err.helpers import find

T = TypeVar("T")


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


def IS(code: int, flags: Tensor, *, config: Optional[ErrorConfig] = None) -> Tensor:
    """
    Tensor predicate: does any sample have this specific error code?
    
    Essentially: find(code, flags, config or get_config()).any() with validation.
    
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
    cond = find(code, flags, config=config or get_config()).any()
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


def _where_select(cond: Tensor, true_val: T, false_val: T) -> T:
    """
    Apply torch.where element-wise to potentially nested outputs.
    
    Handles tuples, lists, and single tensors. Non-tensor values pass through
    from the true branch (assuming structure matches).
    
    Args:
        cond: 0-D bool tensor condition
        true_val: Result from the "true" branch
        false_val: Result from the "false" branch
    
    Returns:
        Selected values based on condition
    """
    if isinstance(true_val, Tensor):
        # Broadcast scalar condition to match tensor shape
        return torch.where(cond, true_val, false_val)
    elif isinstance(true_val, tuple):
        return tuple(
            _where_select(cond, t, f)
            for t, f in zip(true_val, false_val)
        )
    elif isinstance(true_val, list):
        return [
            _where_select(cond, t, f)
            for t, f in zip(true_val, false_val)
        ]
    elif isinstance(true_val, dict):
        return {
            k: _where_select(cond, true_val[k], false_val[k])
            for k in true_val
        }
    else:
        # Non-tensor: return true_val (conditions should have same structure)
        return true_val


class _IfChain(Generic[T]):
    """
    Internal IF/ELIF/ELSE chain builder using torch.where.
    
    Use via IF(cond, fn).ELIF(cond2, fn2).ELSE(else_fn).
    
    Implementation:
        - Eager mode: Uses Python if/elif/else for debugging
        - Compiled mode: Evaluates ALL branches, uses torch.where to select
        
    Note:
        Unlike torch.cond, this evaluates all branches (slightly less efficient)
        but ensures proper gradient flow for inductor backward pass.
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
        
        Implementation:
            - Eager mode: Python control flow (short-circuit, efficient)
            - Compiled mode: torch.where selection (all branches evaluated)
        
        Constraints (for torch.compile):
            - All branches must return the same type/structure T
            - Branch functions should be cheap (all are evaluated in compiled mode)
        
        Args:
            else_fn (Callable[[], T]): Else branch body
        
        Returns:
            (T): Result from the taken branch (eager) or selected result (compiled)
        """
        if not self.__branches:
            return else_fn()
        
        # Eager mode: Python control flow (short-circuit evaluation)
        if not torch.compiler.is_compiling():
            for br in self.__branches:
                if br.cond.item():
                    return br.fn()
            return else_fn()
        
        # Compiled mode: evaluate all branches, select with torch.where
        # This ensures gradient flow works correctly with inductor
        
        # Start with else result
        result = else_fn()
        
        # Apply branches in reverse order (last ELIF first, then IF)
        # This gives correct priority: IF > ELIF1 > ELIF2 > ... > ELSE
        for br in reversed(self.__branches):
            branch_result = br.fn()
            result = _where_select(br.cond, branch_result, result)
        
        return result


def IF(cond: Tensor, then_fn: Callable[[], T]) -> _IfChain[T]:
    """
    Start an IF/ELIF/ELSE chain using torch.where for selection.
    
    Args:
        cond (Tensor): 0-D bool tensor (use HAS/IS/OR/AND/NOT or comparisons)
        then_fn (Callable[[], T]): Callable with no args, returns T
    
    Returns:
        (_IfChain[T]): Chain object for .ELIF(...).ELSE(...)
    
    Example:
        z, flags = (
            IF(IS(ErrorCode.NAN, flags), lambda: (err.replace(z, 0.0, [err.NAN]), flags))
              .ELIF(IS(ErrorCode.INF, flags), lambda: (torch.clamp(z, -10, 10), flags))
              .ELSE(lambda: (z, flags))
        )
    
    Notes:
        - All branches must return same type/structure for torch.compile
        - Eager mode uses Python control flow (efficient, short-circuit)
        - Compiled mode uses torch.where (all branches evaluated, gradient-safe)
    """
    cond = _ensure_scalar(cond, "IF condition")
    return _IfChain(_Branch(cond, then_fn))
