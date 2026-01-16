"""
Tests for the Control Flow DSL (control.py).

Tests cover:
- Predicates: HAS, IS, OR, AND, NOT
- Control Flow: IF/ELIF/ELSE chain
- Validation: _ensure_scalar
- torch.compile compatibility

Run with:
    pytest torchguard/tests/test_control_flow.py -v
"""
from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torchguard import ErrorCode, ErrorLocation, err, push, AND, HAS, IF, IS, NOT, OR
from torchguard.typing import bool_t, error_t
# Internal helper for validation testing
from ..src.control import _ensure_scalar


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsureScalar:
    """Tests for _ensure_scalar() validation."""
    
    def test_accepts_scalar_bool(self) -> None:
        """
        Verify valid 0-D bool tensor passes unchanged.
        
        Returns:
            (Tensor[bool_t, ()]): Same tensor unchanged
        
        Expected:
            result is t (same object reference)
        """
        t: Tensor[bool_t, ()] = torch.tensor(True)
        result: Tensor[bool_t, ()] = _ensure_scalar(t, "test")
        assert result is t
    
    def test_rejects_non_tensor(self) -> None:
        """
        Verify non-tensor input raises TypeError.
        
        Expected:
            TypeError with "must be a Tensor" message
        """
        with pytest.raises(TypeError, match="must be a Tensor"):
            _ensure_scalar(True, "test")  # type: ignore
    
    def test_rejects_non_scalar(self) -> None:
        """
        Verify non-0D tensor raises ValueError.
        
        Expected:
            ValueError with "must be 0-D" message
        """
        with pytest.raises(ValueError, match="must be 0-D"):
            _ensure_scalar(torch.tensor([True, False]), "test")
    
    def test_rejects_non_bool(self) -> None:
        """
        Verify non-bool dtype raises TypeError.
        
        Expected:
            TypeError with "must be a bool Tensor" message
        """
        with pytest.raises(TypeError, match="must be a bool Tensor"):
            _ensure_scalar(torch.tensor(1), "test")


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHASPredicate:
    """Tests for HAS() predicate."""
    
    def __setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_has_returns_false_when_no_errors(self) -> None:
        """
        Verify HAS returns False for clean flags.
        
        Expected:
            result.dtype == torch.bool
            result.ndim == 0
            bool(result) is False
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        result: Tensor[bool_t, ()] = HAS(flags)
        
        assert result.dtype == torch.bool
        assert result.ndim == 0
        assert bool(result) is False
    
    def test_has_returns_true_when_errors(self) -> None:
        """
        Verify HAS returns True when any error exists.
        
        Expected:
            bool(result) is True
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        nan_mask: Tensor[bool_t, ("N",)] = torch.tensor([True, False, False, False])
        flags = push(flags, ErrorCode.NAN, "test", where=nan_mask)
        
        result: Tensor[bool_t, ()] = HAS(flags)
        assert bool(result) is True


class TestISPredicate:
    """Tests for IS() predicate."""
    
    def __setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_is_finds_specific_code(self) -> None:
        """
        Verify IS finds specific error code.
        
        Expected:
            bool(IS(ErrorCode.NAN, flags)) is True
            bool(IS(ErrorCode.INF, flags)) is False
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        nan_mask: Tensor[bool_t, ("N",)] = torch.tensor([True, False, False, False])
        flags = push(flags, ErrorCode.NAN, "test", where=nan_mask)
        
        assert bool(IS(ErrorCode.NAN, flags)) is True
        assert bool(IS(ErrorCode.INF, flags)) is False
    
    def test_is_returns_scalar(self) -> None:
        """
        Verify IS returns 0-D bool tensor.
        
        Expected:
            result.dtype == torch.bool
            result.ndim == 0
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        result: Tensor[bool_t, ()] = IS(ErrorCode.NAN, flags)
        
        assert result.dtype == torch.bool
        assert result.ndim == 0


class TestORPredicate:
    """Tests for OR() predicate."""
    
    def __setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_or_any_true(self) -> None:
        """
        Verify OR returns True if any condition is True.
        
        Expected:
            bool(result) is True
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        nan_mask: Tensor[bool_t, ("N",)] = torch.tensor([True, False, False, False])
        flags = push(flags, ErrorCode.NAN, "test", where=nan_mask)
        
        result: Tensor[bool_t, ()] = OR(IS(ErrorCode.NAN, flags), IS(ErrorCode.INF, flags))
        assert bool(result) is True
    
    def test_or_all_false(self) -> None:
        """
        Verify OR returns False if all conditions are False.
        
        Expected:
            bool(result) is False
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        
        result: Tensor[bool_t, ()] = OR(IS(ErrorCode.NAN, flags), IS(ErrorCode.INF, flags))
        assert bool(result) is False
    
    def test_or_empty_raises(self) -> None:
        """
        Verify OR with no conditions raises ValueError.
        
        Expected:
            ValueError with "requires at least one" message
        """
        with pytest.raises(ValueError, match="requires at least one"):
            OR()
    
    def test_or_single_condition(self) -> None:
        """
        Verify OR with single condition works.
        
        Expected:
            bool(result) is True
        """
        result: Tensor[bool_t, ()] = OR(torch.tensor(True))
        assert bool(result) is True


class TestANDPredicate:
    """Tests for AND() predicate."""
    
    def __setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_and_all_true(self) -> None:
        """
        Verify AND returns True if all conditions are True.
        
        Expected:
            bool(result) is True
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        nan_mask: Tensor[bool_t, ("N",)] = torch.tensor([True, False, False, False])
        flags = push(flags, ErrorCode.NAN, "test", where=nan_mask)
        inf_mask: Tensor[bool_t, ("N",)] = torch.tensor([True, False, False, False])
        flags = push(flags, ErrorCode.INF, "test", where=inf_mask)
        
        result: Tensor[bool_t, ()] = AND(IS(ErrorCode.NAN, flags), IS(ErrorCode.INF, flags))
        assert bool(result) is True
    
    def test_and_one_false(self) -> None:
        """
        Verify AND returns False if any condition is False.
        
        Expected:
            bool(result) is False
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        nan_mask: Tensor[bool_t, ("N",)] = torch.tensor([True, False, False, False])
        flags = push(flags, ErrorCode.NAN, "test", where=nan_mask)
        
        result: Tensor[bool_t, ()] = AND(IS(ErrorCode.NAN, flags), IS(ErrorCode.INF, flags))
        assert bool(result) is False
    
    def test_and_empty_raises(self) -> None:
        """
        Verify AND with no conditions raises ValueError.
        
        Expected:
            ValueError with "requires at least one" message
        """
        with pytest.raises(ValueError, match="requires at least one"):
            AND()


class TestNOTPredicate:
    """Tests for NOT() predicate."""
    
    def __setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_not_true(self) -> None:
        """
        Verify NOT(True) returns False.
        
        Expected:
            bool(result) is False
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        nan_mask: Tensor[bool_t, ("N",)] = torch.tensor([True, False, False, False])
        flags = push(flags, ErrorCode.NAN, "test", where=nan_mask)
        
        result: Tensor[bool_t, ()] = NOT(IS(ErrorCode.NAN, flags))
        assert bool(result) is False
    
    def test_not_false(self) -> None:
        """
        Verify NOT(False) returns True.
        
        Expected:
            bool(result) is True
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        
        result: Tensor[bool_t, ()] = NOT(IS(ErrorCode.NAN, flags))
        assert bool(result) is True


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL FLOW TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIfElseChain:
    """Tests for IF/ELSE chain."""
    
    def test_if_true_branch(self) -> None:
        """
        Verify IF with True condition executes then branch.
        
        Expected:
            out.item() == 2.0
        """
        x: Tensor = torch.tensor(1.0)
        out: Tensor = IF(torch.tensor(True), lambda: x + 1).ELSE(lambda: x + 2)
        assert out.item() == 2.0
    
    def test_if_false_branch(self) -> None:
        """
        Verify IF with False condition executes else branch.
        
        Expected:
            out.item() == 3.0
        """
        x: Tensor = torch.tensor(1.0)
        out: Tensor = IF(torch.tensor(False), lambda: x + 1).ELSE(lambda: x + 2)
        assert out.item() == 3.0


class TestIfElifElseChain:
    """Tests for IF/ELIF/ELSE chain."""
    
    def test_elif_taken(self) -> None:
        """
        Verify ELIF is taken when IF is False.
        
        Expected:
            out.item() == 3.0
        """
        x: Tensor = torch.tensor(1.0)
        out: Tensor = IF(torch.tensor(False), lambda: x * 2).ELIF(torch.tensor(True), lambda: x * 3).ELSE(lambda: x * 4)
        assert out.item() == 3.0
    
    def test_else_taken(self) -> None:
        """
        Verify ELSE is taken when all conditions are False.
        
        Expected:
            out.item() == 4.0
        """
        x: Tensor = torch.tensor(1.0)
        out: Tensor = IF(torch.tensor(False), lambda: x * 2).ELIF(torch.tensor(False), lambda: x * 3).ELSE(lambda: x * 4)
        assert out.item() == 4.0
    
    def test_if_taken_first(self) -> None:
        """
        Verify IF is taken when True, ignoring ELIF.
        
        Expected:
            out.item() == 2.0
        """
        x: Tensor = torch.tensor(1.0)
        out: Tensor = IF(torch.tensor(True), lambda: x * 2).ELIF(torch.tensor(True), lambda: x * 3).ELSE(lambda: x * 4)
        assert out.item() == 2.0
    
    def test_multiple_elifs(self) -> None:
        """
        Verify multiple ELIFs work correctly, taking the first True.
        
        Expected:
            out.item() == 3.0
        """
        x: Tensor = torch.tensor(1.0)
        out: Tensor = IF(torch.tensor(False), lambda: x * 1).ELIF(torch.tensor(False), lambda: x * 2).ELIF(torch.tensor(True), lambda: x * 3).ELIF(torch.tensor(False), lambda: x * 4).ELSE(lambda: x * 5)
        assert out.item() == 3.0


class TestIfRejectsInvalid:
    """Tests for IF input validation."""
    
    def test_if_rejects_non_scalar(self) -> None:
        """
        Verify IF rejects non-scalar conditions.
        
        Expected:
            ValueError with "must be 0-D" message
        """
        with pytest.raises(ValueError, match="must be 0-D"):
            IF(torch.tensor([True, False]), lambda: 1).ELSE(lambda: 2)
    
    def test_elif_rejects_non_scalar(self) -> None:
        """
        Verify ELIF rejects non-scalar conditions.
        
        Expected:
            ValueError with "must be 0-D" message
        """
        with pytest.raises(ValueError, match="must be 0-D"):
            IF(torch.tensor(False), lambda: 1).ELIF(torch.tensor([True]), lambda: 2).ELSE(lambda: 3)


class TestIfWithTupleReturn:
    """Tests for IF/ELSE with tuple returns."""
    
    def __setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_tuple_return(self) -> None:
        """
        Verify IF/ELSE works with tuple returns.
        
        Expected:
            (out == x * 2).all() is True
        """
        ErrorLocation.reset()
        x: Tensor = torch.tensor([1.0, 2.0])
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(2)
        
        out, out_flags = IF(torch.tensor(True), lambda: (x * 2, flags)).ELSE(lambda: (x * 3, flags))
        
        assert (out == x * 2).all()


# ═══════════════════════════════════════════════════════════════════════════════
# TORCH.COMPILE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestControlFlowCompiled:
    """Tests for Control Flow DSL with torch.compile."""
    
    def __setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_has_compiles(self) -> None:
        """
        Verify HAS works in compiled model without graph breaks.
        
        Expected:
            result.dtype == torch.bool
        """
        ErrorLocation.reset()
        
        @torch.compile(backend="eager", fullgraph=True)
        def fn(flags: Tensor) -> Tensor:
            return HAS(flags)
        
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(3)
        result: Tensor[bool_t, ()] = fn(flags)
        assert result.dtype == torch.bool
    
    def test_predicates_compile(self) -> None:
        """
        Verify OR/AND/NOT work in compiled code.
        
        Expected:
            bool(or_result) is True
            bool(and_result) is False
            bool(not_result) is False
        """
        @torch.compile(backend="eager", fullgraph=True)
        def fn() -> Tuple[Tensor, Tensor, Tensor]:
            a: Tensor[bool_t, ()] = torch.tensor(True)
            b: Tensor[bool_t, ()] = torch.tensor(False)
            return OR(a, b), AND(a, b), NOT(a)
        
        or_result, and_result, not_result = fn()
        assert bool(or_result) is True
        assert bool(and_result) is False
        assert bool(not_result) is False
    
    def test_if_else_compiles(self) -> None:
        """
        Verify IF/ELSE compiles with fullgraph=True.
        
        Expected:
            Positive input: (result == x * 2).all()
            Negative input: (result == x * 3).all()
        """
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x: Tensor) -> Tensor:
            cond: Tensor[bool_t, ()] = (x.sum() > 0)
            return IF(cond, lambda: x * 2).ELSE(lambda: x * 3)
        
        x_pos: Tensor = torch.ones(2, 4)
        result: Tensor = fn(x_pos)
        assert (result == x_pos * 2).all()
        
        x_neg: Tensor = -torch.ones(2, 4)
        result = fn(x_neg)
        assert (result == x_neg * 3).all()
    
    def test_full_model_compiles(self) -> None:
        """
        Verify full model with control flow compiles with fullgraph=True.
        
        Expected:
            (y == x * 2).all() is True
        """
        ErrorLocation.reset()
        
        class Model(nn.Module):
            def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
                n: int = x.shape[0]
                flags: Tensor[error_t, ("N", "num_words")] = err.new_t(n, x.device)
                cond: Tensor[bool_t, ()] = (x.sum() > 0)
                y: Tensor = IF(cond, lambda: x * 2).ELSE(lambda: x * 3)
                return y, flags
        
        model: nn.Module = Model()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        x: Tensor = torch.ones(2, 4)
        y, flags = compiled(x)
        assert (y == x * 2).all()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestControlFlowIntegration:
    """Integration tests combining predicates and control flow."""
    
    def __setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_predicates_combined(self) -> None:
        """
        Verify HAS, IS, OR, AND, NOT work together correctly.
        
        Expected:
            Empty flags: bool(HAS(flags)) is False
            With NAN: bool(HAS(flags)) is True
            bool(IS(ErrorCode.NAN, flags)) is True
            bool(IS(ErrorCode.INF, flags)) is False
            bool(OR(IS(NAN), IS(INF))) is True
            bool(AND(IS(NAN), IS(INF))) is False
            bool(NOT(IS(NAN))) is False
            bool(NOT(IS(INF))) is True
        """
        ErrorLocation.reset()
        flags: Tensor[error_t, ("N", "num_words")] = err.new_t(4)
        
        assert bool(HAS(flags)) is False
        
        nan_mask: Tensor[bool_t, ("N",)] = torch.tensor([True, False, False, False])
        flags = push(flags, ErrorCode.NAN, "test", where=nan_mask)
        
        assert bool(HAS(flags)) is True
        assert bool(IS(ErrorCode.NAN, flags)) is True
        assert bool(IS(ErrorCode.INF, flags)) is False
        
        cond: Tensor[bool_t, ()] = OR(IS(ErrorCode.NAN, flags), IS(ErrorCode.INF, flags))
        assert bool(cond) is True
        
        cond = AND(IS(ErrorCode.NAN, flags), IS(ErrorCode.INF, flags))
        assert bool(cond) is False
        
        assert bool(NOT(IS(ErrorCode.NAN, flags))) is False
        assert bool(NOT(IS(ErrorCode.INF, flags))) is True
    
    def test_if_with_has_predicate(self) -> None:
        """
        Verify IF with HAS predicate behaves correctly.
        
        Expected:
            Clean flags: (result == x).all()
            Dirty flags: (result == x * 0).all()
        """
        ErrorLocation.reset()
        flags_clean: Tensor[error_t, ("N", "num_words")] = err.new_t(2)
        flags_dirty: Tensor[error_t, ("N", "num_words")] = err.new_t(2)
        nan_mask: Tensor[bool_t, ("N",)] = torch.tensor([True, False])
        flags_dirty = push(flags_dirty, ErrorCode.NAN, "test", where=nan_mask)
        
        x: Tensor = torch.tensor([1.0, 2.0])
        
        result: Tensor = IF(HAS(flags_clean), lambda: x * 0).ELSE(lambda: x)
        assert (result == x).all()
        
        result = IF(HAS(flags_dirty), lambda: x * 0).ELSE(lambda: x)
        assert (result == x * 0).all()
