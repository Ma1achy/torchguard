"""
Tests for error_t combinators (monadic/applicative-style helpers).

Tests cover:
- all_ok / any_err: Scalar predicates
- map_ok / map_err: Conditional value transforms
- map_err_flags: Flag transforms for error samples
- and_then: Strict Result-style chaining
- bind: Accumulate-all chaining
- ensure_mask / guard: Predicate to error conversion
- recover_with_fallback: Error recovery with marking

Run with:
    pytest tests/utils/errors/compiled/test_combinators.py -v
"""
import pytest
import torch
from torch import Tensor

from torchguard import (
    error_t,
    ErrorCode,
    ErrorLocation,
    Severity,
    DEFAULT_CONFIG,
    err,
    flags as flags_ns,
)


class TestAllOkAnyErr:
    """Tests for all_ok and any_err scalar predicates."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_all_ok_clean_batch(self) -> None:
        """all_ok returns True when all samples are clean."""
        x = torch.randn(8, 10)
        flags = err.new(x)
        assert err.all_ok(flags).item() is True
    
    def test_all_ok_with_errors(self) -> None:
        """all_ok returns False when any sample has error."""
        flags = err.from_code(ErrorCode.NAN, 42, 8)
        assert err.all_ok(flags).item() is False
    
    def test_any_err_clean_batch(self) -> None:
        """any_err returns False when all samples are clean."""
        x = torch.randn(8, 10)
        flags = err.new(x)
        assert err.any_err(flags).item() is False
    
    def test_any_err_with_errors(self) -> None:
        """any_err returns True when any sample has error."""
        flags = err.from_code(ErrorCode.NAN, 42, 8)
        assert err.any_err(flags).item() is True
    
    def test_all_ok_any_err_complementary(self) -> None:
        """all_ok and any_err are complementary for non-mixed batches."""
        # All clean
        clean_flags = err.new_t(8, None)
        assert err.all_ok(clean_flags).item() is True
        assert err.any_err(clean_flags).item() is False
        
        # All error
        err_flags = err.from_code(ErrorCode.NAN, 42, 8)
        assert err.all_ok(err_flags).item() is False
        assert err.any_err(err_flags).item() is True


class TestMapOk:
    """Tests for map_ok value transform."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_map_ok_all_clean(self) -> None:
        """map_ok applies fn to all samples when all are clean."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        
        result = err.map_ok(flags, x, lambda t: t * 2)
        
        assert torch.allclose(result, torch.ones(4, 3) * 2)
    
    def test_map_ok_all_error(self) -> None:
        """map_ok keeps original values when all samples have errors."""
        x = torch.ones(4, 3)
        flags = err.from_code(ErrorCode.NAN, 42, 4)
        
        result = err.map_ok(flags, x, lambda t: t * 2)
        
        assert torch.allclose(result, x)  # Unchanged
    
    def test_map_ok_mixed(self) -> None:
        """map_ok only transforms OK samples in mixed batch."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        # Add error to samples 1 and 3
        codes = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        flags = err.push(flags, codes, location=42)
        
        result = err.map_ok(flags, x, lambda t: t * 10)
        
        # OK samples (0, 2) should be transformed
        assert torch.allclose(result[0], torch.ones(3) * 10)
        assert torch.allclose(result[2], torch.ones(3) * 10)
        # Error samples (1, 3) should be unchanged
        assert torch.allclose(result[1], torch.ones(3))
        assert torch.allclose(result[3], torch.ones(3))
    
    def test_map_ok_1d_tensor(self) -> None:
        """map_ok works with 1D tensors."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 0], dtype=torch.int64)
        flags = err.push(flags, codes, location=42)
        
        result = err.map_ok(flags, x, lambda t: t * 2)
        
        expected = torch.tensor([2.0, 2.0, 6.0, 8.0])  # index 1 unchanged
        assert torch.allclose(result, expected)


class TestMapErr:
    """Tests for map_err value transform."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_map_err_all_clean(self) -> None:
        """map_err keeps original values when all samples are clean."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        
        result = err.map_err(flags, x, lambda t: t * 0)
        
        assert torch.allclose(result, x)  # Unchanged
    
    def test_map_err_all_error(self) -> None:
        """map_err applies fn to all samples when all have errors."""
        x = torch.ones(4, 3)
        flags = err.from_code(ErrorCode.NAN, 42, 4)
        
        result = err.map_err(flags, x, lambda t: t * 0)
        
        assert torch.allclose(result, torch.zeros(4, 3))
    
    def test_map_err_mixed(self) -> None:
        """map_err only transforms error samples in mixed batch."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        flags = err.push(flags, codes, location=42)
        
        result = err.map_err(flags, x, lambda t: t * 0)
        
        # OK samples (0, 2) should be unchanged
        assert torch.allclose(result[0], torch.ones(3))
        assert torch.allclose(result[2], torch.ones(3))
        # Error samples (1, 3) should be zeroed
        assert torch.allclose(result[1], torch.zeros(3))
        assert torch.allclose(result[3], torch.zeros(3))


class TestMapErrFlags:
    """Tests for map_err_flags flag transform."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_map_err_flags_clears_errors(self) -> None:
        """map_err_flags can clear errors for error samples."""
        flags = err.from_code(ErrorCode.NAN, 42, 4)
        
        # Clear flags for error samples
        result = err.map_err_flags(flags, lambda f: torch.zeros_like(f))
        
        # All samples should now be OK
        assert err.all_ok(result).item() is True
    
    def test_map_err_flags_preserves_ok_samples(self) -> None:
        """map_err_flags doesn't touch OK samples."""
        x = torch.randn(4, 10)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 0], dtype=torch.int64)
        flags = err.push(flags, codes, location=42)
        
        # Apply a transform that would add errors if applied
        result = err.map_err_flags(flags, lambda f: f | 0xFF)
        
        # OK samples (0, 2, 3) should still be OK
        assert err.is_ok(result)[0].item() is True
        assert err.is_ok(result)[2].item() is True
        assert err.is_ok(result)[3].item() is True


class TestAndThen:
    """Tests for and_then strict Result-style chaining."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        self.loc = ErrorLocation.register("test_and_then")
    
    def test_and_then_all_clean(self) -> None:
        """and_then applies fn and merges flags when all clean."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        
        def layer(z: Tensor):
            new_flags = err.new(z)
            return z * 2, new_flags
        
        z_out, flags_out = err.and_then(flags, x, layer)
        
        assert torch.allclose(z_out, torch.ones(4, 3) * 2)
        assert err.all_ok(flags_out).item() is True
    
    def test_and_then_strict_shortcircuit_values(self) -> None:
        """and_then freezes values for errored samples."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        flags = err.push(flags, codes, location=self.loc)
        
        def layer(z: Tensor):
            new_flags = err.new(z)
            return z * 10, new_flags
        
        z_out, _ = err.and_then(flags, x, layer)
        
        # OK samples transformed
        assert torch.allclose(z_out[0], torch.ones(3) * 10)
        assert torch.allclose(z_out[2], torch.ones(3) * 10)
        # Error samples frozen
        assert torch.allclose(z_out[1], torch.ones(3))
        assert torch.allclose(z_out[3], torch.ones(3))
    
    def test_and_then_strict_shortcircuit_flags(self) -> None:
        """and_then ignores new flags for already-errored samples."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        flags = err.push(flags, codes, location=self.loc)
        
        def layer(z: Tensor):
            # Add INF error to all samples
            new_flags = err.from_code(ErrorCode.INF, self.loc, z.shape[0], z.device)
            return z * 2, new_flags
        
        _, flags_out = err.and_then(flags, x, layer)
        
        # OK samples (0, 2) should have INF error
        assert err.has_inf(flags_out)[0].item() is True
        assert err.has_inf(flags_out)[2].item() is True
        # Error samples (1, 3) should NOT have INF (short-circuited)
        assert err.has_inf(flags_out)[1].item() is False
        assert err.has_inf(flags_out)[3].item() is False
        # But they should still have NAN
        assert err.has_nan(flags_out)[1].item() is True
        assert err.has_nan(flags_out)[3].item() is True


class TestBind:
    """Tests for bind accumulate-all chaining."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        self.loc = ErrorLocation.register("test_bind")
    
    def test_bind_all_clean(self) -> None:
        """bind applies fn and merges flags when all clean."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        
        def layer(z: Tensor):
            new_flags = err.new(z)
            return z * 2, new_flags
        
        z_out, flags_out = err.bind(flags, x, layer)
        
        assert torch.allclose(z_out, torch.ones(4, 3) * 2)
        assert err.all_ok(flags_out).item() is True
    
    def test_bind_shortcircuits_values(self) -> None:
        """bind freezes values for errored samples."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        flags = err.push(flags, codes, location=self.loc)
        
        def layer(z: Tensor):
            new_flags = err.new(z)
            return z * 10, new_flags
        
        z_out, _ = err.bind(flags, x, layer)
        
        # OK samples transformed
        assert torch.allclose(z_out[0], torch.ones(3) * 10)
        assert torch.allclose(z_out[2], torch.ones(3) * 10)
        # Error samples frozen
        assert torch.allclose(z_out[1], torch.ones(3))
        assert torch.allclose(z_out[3], torch.ones(3))
    
    def test_bind_accumulates_all_flags(self) -> None:
        """bind accumulates flags even for already-errored samples."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 1], dtype=torch.int64)  # NAN on 1, 3
        flags = err.push(flags, codes, location=self.loc)
        
        def layer(z: Tensor):
            # Add INF error to all samples
            new_flags = err.from_code(ErrorCode.INF, self.loc, z.shape[0], z.device)
            return z * 2, new_flags
        
        _, flags_out = err.bind(flags, x, layer)
        
        # ALL samples should have INF error (accumulated)
        assert err.has_inf(flags_out).all().item() is True
        # Error samples should still have NAN
        assert err.has_nan(flags_out)[1].item() is True
        assert err.has_nan(flags_out)[3].item() is True
    
    def test_bind_vs_and_then_difference(self) -> None:
        """bind and and_then differ in flag accumulation for errored samples."""
        x = torch.ones(4, 3)
        
        # Create flags with error on sample 1
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 0], dtype=torch.int64)
        flags = err.push(flags, codes, location=self.loc)
        
        def add_inf(z: Tensor):
            new_flags = err.from_code(ErrorCode.INF, self.loc, z.shape[0], z.device)
            return z, new_flags
        
        _, flags_bind = err.bind(flags.clone(), x, add_inf)
        _, flags_and_then = err.and_then(flags.clone(), x, add_inf)
        
        # bind: sample 1 should have BOTH NAN and INF
        assert err.has_nan(flags_bind)[1].item() is True
        assert err.has_inf(flags_bind)[1].item() is True
        
        # and_then: sample 1 should have ONLY NAN (INF ignored)
        assert err.has_nan(flags_and_then)[1].item() is True
        assert err.has_inf(flags_and_then)[1].item() is False


class TestEnsureMask:
    """Tests for ensure_mask predicate-to-error conversion."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        self.loc = ErrorLocation.register("test_ensure")
    
    def test_ensure_mask_all_pass(self) -> None:
        """ensure_mask adds no errors when all pass."""
        flags = err.new_t(4, None)
        ok_mask = torch.tensor([True, True, True, True])
        
        result = err.ensure_mask(flags, ok_mask, ErrorCode.OUT_OF_BOUNDS, self.loc)
        
        assert err.all_ok(result).item() is True
    
    def test_ensure_mask_all_fail(self) -> None:
        """ensure_mask adds errors to all when all fail."""
        flags = err.new_t(4, None)
        ok_mask = torch.tensor([False, False, False, False])
        
        result = err.ensure_mask(flags, ok_mask, ErrorCode.OUT_OF_BOUNDS, self.loc)
        
        assert err.is_err(result).all().item() is True
        assert err.has_code(result, ErrorCode.OUT_OF_BOUNDS).all().item() is True
    
    def test_ensure_mask_mixed(self) -> None:
        """ensure_mask adds errors only where mask is False."""
        flags = err.new_t(4, None)
        ok_mask = torch.tensor([True, False, True, False])
        
        result = err.ensure_mask(flags, ok_mask, ErrorCode.OUT_OF_BOUNDS, self.loc)
        
        assert err.is_ok(result)[0].item() is True
        assert err.is_err(result)[1].item() is True
        assert err.is_ok(result)[2].item() is True
        assert err.is_err(result)[3].item() is True


class TestGuard:
    """Tests for guard predicate-based error insertion."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        self.loc = ErrorLocation.register("test_guard")
    
    def test_guard_all_pass(self) -> None:
        """guard adds no errors when predicate passes for all."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        flags = err.new(x)
        
        # Predicate: all values positive
        result = err.guard(flags, x, lambda t: (t > 0).all(dim=-1), ErrorCode.OUT_OF_BOUNDS, self.loc)
        
        assert err.all_ok(result).item() is True
    
    def test_guard_some_fail(self) -> None:
        """guard adds errors where predicate fails."""
        x = torch.tensor([[1.0, 2.0], [-1.0, 4.0], [3.0, 4.0]])
        flags = err.new(x)
        
        # Predicate: all values positive
        result = err.guard(flags, x, lambda t: (t > 0).all(dim=-1), ErrorCode.OUT_OF_BOUNDS, self.loc)
        
        assert err.is_ok(result)[0].item() is True
        assert err.is_err(result)[1].item() is True  # Has negative value
        assert err.is_ok(result)[2].item() is True
    
    def test_guard_with_existing_errors(self) -> None:
        """guard accumulates with existing errors."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        flags = err.new(x)
        # Add NAN to sample 0
        codes = torch.tensor([1, 0, 0], dtype=torch.int64)
        flags = err.push(flags, codes, location=self.loc)
        
        # Guard: all values < 5 (sample 2 fails)
        result = err.guard(flags, x, lambda t: (t < 5).all(dim=-1), ErrorCode.OUT_OF_BOUNDS, self.loc)
        
        # Sample 0: has NAN, no OOB (passes guard)
        assert err.has_nan(result)[0].item() is True
        assert err.has_code(result, ErrorCode.OUT_OF_BOUNDS)[0].item() is False
        # Sample 1: OK
        assert err.is_ok(result)[1].item() is True
        # Sample 2: fails guard
        assert err.has_code(result, ErrorCode.OUT_OF_BOUNDS)[2].item() is True


class TestRecoverWithFallback:
    """Tests for recover_with_fallback error recovery."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        self.loc = ErrorLocation.register("test_recover")
    
    def test_recover_all_clean(self) -> None:
        """recover_with_fallback does nothing when all clean."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        
        z_out, flags_out = err.recover_with_fallback(flags, x, torch.tensor(0.0), self.loc)
        
        assert torch.allclose(z_out, x)  # Unchanged
        assert err.all_ok(flags_out).item() is True  # No fallback marker
    
    def test_recover_replaces_error_values(self) -> None:
        """recover_with_fallback replaces error sample values."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        flags = err.push(flags, codes, location=self.loc)
        
        z_out, _ = err.recover_with_fallback(flags, x, torch.tensor(-1.0), self.loc)
        
        # OK samples unchanged
        assert torch.allclose(z_out[0], torch.ones(3))
        assert torch.allclose(z_out[2], torch.ones(3))
        # Error samples replaced
        assert torch.allclose(z_out[1], torch.ones(3) * -1)
        assert torch.allclose(z_out[3], torch.ones(3) * -1)
    
    def test_recover_adds_fallback_marker(self) -> None:
        """recover_with_fallback adds FALLBACK_VALUE marker."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        flags = err.push(flags, codes, location=self.loc)
        
        _, flags_out = err.recover_with_fallback(flags, x, torch.tensor(0.0), self.loc)
        
        # Error samples should have FALLBACK_VALUE marker
        assert err.has_fallback(flags_out)[1].item() is True
        assert err.has_fallback(flags_out)[3].item() is True
        # OK samples should NOT have fallback marker
        assert err.has_fallback(flags_out)[0].item() is False
        assert err.has_fallback(flags_out)[2].item() is False
    
    def test_recover_preserves_original_errors(self) -> None:
        """recover_with_fallback keeps original errors plus adds fallback."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 0], dtype=torch.int64)  # NAN on sample 1
        flags = err.push(flags, codes, location=self.loc)
        
        _, flags_out = err.recover_with_fallback(flags, x, torch.tensor(0.0), self.loc)
        
        # Sample 1 should have both NAN and FALLBACK_VALUE
        assert err.has_nan(flags_out)[1].item() is True
        assert err.has_fallback(flags_out)[1].item() is True
    
    def test_recover_with_tensor_fallback(self) -> None:
        """recover_with_fallback works with tensor fallback values."""
        x = torch.ones(4, 3)
        flags = err.new(x)
        codes = torch.tensor([0, 1, 0, 0], dtype=torch.int64)
        flags = err.push(flags, codes, location=self.loc)
        
        fallback = torch.tensor([10.0, 20.0, 30.0])
        z_out, _ = err.recover_with_fallback(flags, x, fallback, self.loc)
        
        # Error sample should have fallback values
        assert torch.allclose(z_out[1], fallback)
        # OK samples unchanged
        assert torch.allclose(z_out[0], torch.ones(3))


class TestCombinatorsCompile:
    """Tests that combinators work with torch.compile."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        self.loc = ErrorLocation.register("test_compile")
    
    def test_map_ok_compiles(self) -> None:
        """map_ok works inside torch.compile."""
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x: Tensor, flags: Tensor) -> Tensor:
            return err.map_ok(flags, x, lambda t: t * 2)
        
        x = torch.ones(4, 3)
        flags = err.new(x)
        result = fn(x, flags)
        assert torch.allclose(result, x * 2)
    
    def test_map_err_compiles(self) -> None:
        """map_err works inside torch.compile."""
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x: Tensor, flags: Tensor) -> Tensor:
            return err.map_err(flags, x, lambda t: t * 0)
        
        x = torch.ones(4, 3)
        flags = err.from_code(ErrorCode.NAN, self.loc, 4)
        result = fn(x, flags)
        assert torch.allclose(result, torch.zeros(4, 3))
    
    def test_ensure_mask_compiles(self) -> None:
        """ensure_mask works inside torch.compile."""
        loc = self.loc
        
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x: Tensor) -> Tensor:
            flags = err.new(x)
            ok_mask = (x.sum(dim=-1) > 0)
            return err.ensure_mask(flags, ok_mask, ErrorCode.OUT_OF_BOUNDS, loc)
        
        x = torch.tensor([[1.0, 2.0], [-5.0, 1.0]])
        flags = fn(x)
        assert err.is_ok(flags)[0].item() is True
        assert err.is_err(flags)[1].item() is True
    
    def test_recover_with_fallback_compiles(self) -> None:
        """recover_with_fallback works inside torch.compile."""
        loc = self.loc
        
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x: Tensor, flags: Tensor) -> tuple[Tensor, Tensor]:
            return err.recover_with_fallback(flags, x, torch.tensor(0.0), loc)
        
        x = torch.ones(4, 3)
        flags = err.from_code(ErrorCode.NAN, self.loc, 4)
        z_out, flags_out = fn(x, flags)
        assert torch.allclose(z_out, torch.zeros(4, 3))
        assert err.has_fallback(flags_out).all().item() is True
