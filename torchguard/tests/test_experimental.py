"""
Tests for the experimental float64+view backend.

This test suite verifies that the experimental backend:
1. Returns float64 tensors
2. Preserves bit-level correctness via view(int64)
3. Works with torch.compile(fullgraph=True)
4. Works with torch.cond (IF/ELSE DSL)
5. Supports training (forward + backward)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchguard.src.experimental import err, IF, IS, HAS, AND, OR, NOT
from torchguard.src.core.codes import ErrorCode
from torchguard.src.core.severity import Severity


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC FUNCTIONALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFloat64Storage:
    """Test that experimental backend returns float64 tensors."""
    
    def test_new_returns_float64(self):
        """err.new() should return float64 tensor."""
        x = torch.randn(8, 32)
        f = err.new(x)
        assert f.dtype == torch.float64, f"Expected float64, got {f.dtype}"
        assert f.shape == (8, 4)  # Default config: 4 words
    
    def test_new_t_returns_float64(self):
        """err.new_t() should return float64 tensor."""
        f = err.new_t(16)
        assert f.dtype == torch.float64
        assert f.shape == (16, 4)
    
    def test_from_code_returns_float64(self):
        """err.from_code() should return float64 tensor."""
        f = err.from_code(ErrorCode.NAN, location=42, batch_size=8)
        assert f.dtype == torch.float64
        assert err.has_nan(f).all()
    
    def test_push_returns_float64(self):
        """err.push() should return float64 tensor."""
        x = torch.randn(8, 32)
        f = err.new(x)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True] * 4 + [False] * 4))
        assert f.dtype == torch.float64
    
    def test_merge_returns_float64(self):
        """err.merge() should return float64 tensor."""
        f1 = err.from_code(ErrorCode.NAN, location=1, batch_size=8)
        f2 = err.from_code(ErrorCode.INF, location=2, batch_size=8)
        merged = err.merge(f1, f2)
        assert merged.dtype == torch.float64


class TestViewRoundtrip:
    """Test that view(int64) <-> view(float64) preserves bits."""
    
    def test_empty_flags_roundtrip(self):
        """Empty flags should survive roundtrip."""
        f = err.new_t(8)
        f_int = f.view(torch.int64)
        f_back = f_int.view(torch.float64)
        assert (f == f_back).all()
    
    def test_flags_with_errors_roundtrip(self):
        """Flags with errors should survive roundtrip."""
        f = err.from_code(ErrorCode.NAN, location=42, batch_size=8)
        f_int = f.view(torch.int64)
        f_back = f_int.view(torch.float64)
        assert (f == f_back).all()
        # Verify data is still accessible
        assert err.has_nan(f_back).all()


class TestBasicOperations:
    """Test basic err operations work correctly."""
    
    def test_is_ok_empty(self):
        """is_ok should return True for empty flags."""
        f = err.new_t(8)
        assert err.is_ok(f).all()
        assert not err.is_err(f).any()
    
    def test_is_err_with_error(self):
        """is_err should return True when flags have errors."""
        f = err.from_code(ErrorCode.NAN, location=1, batch_size=8)
        assert err.is_err(f).all()
        assert not err.is_ok(f).any()
    
    def test_push_where(self):
        """push with where mask should only affect masked samples."""
        f = err.new_t(8)
        mask = torch.tensor([True, True, False, False, True, False, True, False])
        f = err.push(f, err.NAN, location=1, where=mask)
        
        has_nan = err.has_nan(f)
        assert (has_nan == mask).all()
    
    def test_count_errors(self):
        """count_errors should count correctly."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, True, False, False]))
        f = err.push(f, err.INF, location=2, where=torch.tensor([True, False, True, False]))
        
        counts = err.count_errors(f)
        expected = torch.tensor([2, 1, 1, 0], dtype=torch.int32)
        assert (counts == expected).all()
    
    def test_has_code(self):
        """has_code should detect specific error codes."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, False, False]))
        f = err.push(f, err.INF, location=2, where=torch.tensor([False, True, False, False]))
        
        assert err.has_code(f, ErrorCode.NAN)[0]
        assert not err.has_code(f, ErrorCode.NAN)[1]
        assert err.has_code(f, ErrorCode.INF)[1]
        assert not err.has_code(f, ErrorCode.INF)[0]
    
    def test_get_first_code(self):
        """get_first_code should extract code from slot 0."""
        f = err.from_code(ErrorCode.NAN, location=42, batch_size=4)
        codes = err.get_first_code(f)
        assert (codes == ErrorCode.NAN).all()
    
    def test_get_first_location(self):
        """get_first_location should extract location from slot 0."""
        f = err.from_code(ErrorCode.NAN, location=42, batch_size=4)
        locs = err.get_first_location(f)
        assert (locs == 42).all()
    
    def test_clear(self):
        """clear should remove specific error codes."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, True, True, True]))
        f = err.push(f, err.INF, location=2, where=torch.tensor([True, False, True, False]))
        
        f = err.clear(f, ErrorCode.NAN)
        assert f.dtype == torch.float64
        assert not err.has_nan(f).any()
        assert err.has_inf(f)[0]
        assert err.has_inf(f)[2]


# ═══════════════════════════════════════════════════════════════════════════════
# TORCH.COMPILE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTorchCompile:
    """Test that experimental backend works with torch.compile."""
    
    def test_basic_compile(self):
        """Basic operations should compile with fullgraph=True."""
        @torch.compile(backend="inductor", fullgraph=True)
        def forward(x):
            f = err.new(x)
            f = err.push(f, err.NAN, location=1, where=torch.isnan(x).any(dim=-1))
            return x, f
        
        x = torch.randn(8, 32)
        out, f = forward(x)
        assert out.shape == (8, 32)
        assert f.dtype == torch.float64
    
    def test_compile_with_view_operations(self):
        """View-based operations should compile."""
        @torch.compile(backend="inductor", fullgraph=True)
        def forward(x):
            f = err.new(x)
            nan_mask = torch.isnan(x).any(dim=-1)
            f = err.push(f, err.NAN, location=1, where=nan_mask)
            
            has_nan = err.has_nan(f)
            count = err.count_errors(f)
            
            return x, f, has_nan, count
        
        x = torch.randn(8, 32)
        out, f, has_nan, count = forward(x)
        assert f.dtype == torch.float64
        assert has_nan.dtype == torch.bool
        assert count.dtype == torch.int32


# ═══════════════════════════════════════════════════════════════════════════════
# DSL (IF/ELSE) TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDSL:
    """Test IF/ELSE/IS/HAS DSL with experimental backend."""
    
    def test_if_else_eager(self):
        """IF/ELSE should work in eager mode."""
        x = torch.randn(8, 32)
        f = err.new(x)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True] * 4 + [False] * 4))
        
        out = x * 2
        out, f = (
            IF(HAS(f), lambda: (torch.zeros_like(out), f.clone()))
            .ELSE(lambda: (out.clone(), f.clone()))
        )
        
        # Since HAS(f) is True, out should be zeros
        assert (out == 0).all()
    
    def test_if_else_compiled(self):
        """IF/ELSE should compile with fullgraph=True."""
        @torch.compile(backend="inductor", fullgraph=True)
        def forward(x):
            f = err.new(x)
            f = err.push(f, err.NAN, location=1, where=torch.isnan(x).any(dim=-1))
            
            out = x * 2
            out, f = (
                IF(HAS(f), lambda: (torch.zeros_like(out), f.clone()))
                .ELSE(lambda: (out.clone(), f.clone()))
            )
            return out, f
        
        x = torch.randn(8, 32)
        out, f = forward(x)
        assert out.shape == (8, 32)
        assert f.dtype == torch.float64
    
    def test_is_predicate(self):
        """IS predicate should detect specific error codes."""
        x = torch.randn(8, 32)
        f = err.new(x)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True] + [False] * 7))
        
        cond = IS(err.NAN, f)
        assert cond.item() == True
        
        cond_inf = IS(err.INF, f)
        assert cond_inf.item() == False
    
    def test_compound_predicates(self):
        """AND/OR/NOT should work with predicates."""
        x = torch.randn(8, 32)
        f = err.new(x)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True] + [False] * 7))
        
        # Test OR
        cond = OR(IS(err.NAN, f), IS(err.INF, f))
        assert cond.item() == True
        
        # Test AND
        cond = AND(IS(err.NAN, f), IS(err.INF, f))
        assert cond.item() == False
        
        # Test NOT
        cond = NOT(IS(err.INF, f))
        assert cond.item() == True
    
    def test_if_elif_else(self):
        """IF/ELIF/ELSE chain should work."""
        x = torch.randn(8, 32)
        f = err.new(x)
        f = err.push(f, err.INF, location=1, where=torch.tensor([True] + [False] * 7))
        
        out = x * 2
        out, f = (
            IF(IS(err.NAN, f), lambda: (torch.ones_like(out), f.clone()))
            .ELIF(IS(err.INF, f), lambda: (torch.zeros_like(out), f.clone()))
            .ELSE(lambda: (out.clone(), f.clone()))
        )
        
        # Should hit ELIF branch (INF, not NAN)
        assert (out == 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraining:
    """Test that experimental backend supports training (backward pass)."""
    
    def test_basic_backward(self):
        """Basic backward pass should work."""
        x = torch.randn(8, 32, requires_grad=True)
        f = err.new(x)
        
        out = x * 2
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == (8, 32)
    
    def test_compiled_backward(self):
        """Compiled model backward pass should work."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                out = self.linear(x)
                f = err.push(f, err.NAN, location=1, where=torch.isnan(out).any(dim=-1))
                return out, f
        
        model = Model()
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        x = torch.randn(8, 32, requires_grad=True)
        out, f = compiled(x)
        
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == (8, 32)
    
    def test_dsl_backward(self):
        """IF/ELSE DSL backward pass should work."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                out = self.linear(x)
                f = err.push(f, err.NAN, location=1, where=torch.isnan(out).any(dim=-1))
                
                out, f = (
                    IF(IS(err.NAN, f), lambda: (torch.zeros_like(out), f.clone()))
                    .ELSE(lambda: (out.clone(), f.clone()))
                )
                return out, f
        
        model = Model()
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        x = torch.randn(8, 32, requires_grad=True)
        out, f = compiled(x)
        
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert f.dtype == torch.float64


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINATORS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCombinators:
    """Test combinator functions."""
    
    def test_map_ok(self):
        """map_ok should apply fn only to OK samples."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, True, False]))
        
        z = torch.ones(4, 8)
        z_out = err.map_ok(f, z, lambda x: x * 2)
        
        # Samples 0, 2 have errors - should keep original
        # Samples 1, 3 are OK - should be doubled
        assert (z_out[0] == 1).all()
        assert (z_out[1] == 2).all()
        assert (z_out[2] == 1).all()
        assert (z_out[3] == 2).all()
    
    def test_map_err(self):
        """map_err should apply fn only to error samples."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, True, False]))
        
        z = torch.ones(4, 8)
        z_out = err.map_err(f, z, lambda x: x * 0)
        
        # Samples 0, 2 have errors - should be zeroed
        # Samples 1, 3 are OK - should keep original
        assert (z_out[0] == 0).all()
        assert (z_out[1] == 1).all()
        assert (z_out[2] == 0).all()
        assert (z_out[3] == 1).all()
    
    def test_all_ok(self):
        """all_ok should return True only if all samples are OK."""
        f = err.new_t(4)
        assert err.all_ok(f).item() == True
        
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, False, False]))
        assert err.all_ok(f).item() == False
    
    def test_any_err(self):
        """any_err should return True if any sample has errors."""
        f = err.new_t(4)
        assert err.any_err(f).item() == False
        
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, False, False]))
        assert err.any_err(f).item() == True


# ═══════════════════════════════════════════════════════════════════════════════
# FILTERING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFiltering:
    """Test filtering functions."""
    
    def test_Ok_filter(self):
        """Ok() should filter to samples without errors."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, True, False]))
        
        z = torch.arange(4).float().unsqueeze(-1)
        ok_z = err.take_ok(f, z)
        
        assert ok_z.shape[0] == 2
        assert (ok_z[:, 0] == torch.tensor([1.0, 3.0])).all()
    
    def test_Err_filter(self):
        """Err() should filter to samples with errors."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, True, False]))
        
        z = torch.arange(4).float().unsqueeze(-1)
        err_z = err.take_err(f, z)
        
        assert err_z.shape[0] == 2
        assert (err_z[:, 0] == torch.tensor([0.0, 2.0])).all()
    
    def test_partition(self):
        """partition() should split into (ok, err)."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, True, False]))
        
        z = torch.arange(4).float().unsqueeze(-1)
        ok_z, err_z = err.partition(f, z)
        
        assert ok_z.shape[0] == 2
        assert err_z.shape[0] == 2
        assert (ok_z[:, 0] == torch.tensor([1.0, 3.0])).all()
        assert (err_z[:, 0] == torch.tensor([0.0, 2.0])).all()


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC-SHAPE PADDED VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaddedVariants:
    """Test take_ok_p and take_err_p static-shape methods."""
    
    def test_take_ok_p_basic(self):
        """take_ok_p should replace error samples with fill value."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, True, False]))
        
        z = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        ok_z = err.take_ok_p(f, z, fill=0.0)
        
        # Same shape as input
        assert ok_z.shape == z.shape
        # Error samples (0, 2) should be 0.0, OK samples (1, 3) should be original
        expected = torch.tensor([[0.0], [2.0], [0.0], [4.0]])
        assert torch.allclose(ok_z, expected)
    
    def test_take_err_p_basic(self):
        """take_err_p should replace OK samples with fill value."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, True, False]))
        
        z = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        err_z = err.take_err_p(f, z, fill=0.0)
        
        # Same shape as input
        assert err_z.shape == z.shape
        # OK samples (1, 3) should be 0.0, error samples (0, 2) should be original
        expected = torch.tensor([[1.0], [0.0], [3.0], [0.0]])
        assert torch.allclose(err_z, expected)
    
    def test_take_ok_p_custom_fill(self):
        """take_ok_p should use custom fill value."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, True, False, False]))
        
        z = torch.ones(4, 8)
        ok_z = err.take_ok_p(f, z, fill=-1.0)
        
        # Error samples should be -1.0
        assert (ok_z[0] == -1.0).all()
        assert (ok_z[1] == -1.0).all()
        # OK samples should be 1.0
        assert (ok_z[2] == 1.0).all()
        assert (ok_z[3] == 1.0).all()
    
    def test_padded_compiles(self):
        """take_ok_p and take_err_p should compile with fullgraph=True."""
        @torch.compile(backend="inductor", fullgraph=True)
        def forward(x):
            f = err.new(x)
            f = err.push(f, err.NAN, location=1, where=torch.isnan(x).any(dim=-1))
            
            out = x * 2
            ok_out = err.take_ok_p(f, out, fill=0.0)
            err_out = err.take_err_p(f, out, fill=0.0)
            
            return ok_out, err_out, f
        
        x = torch.randn(8, 32)
        ok_out, err_out, f = forward(x)
        
        assert ok_out.shape == (8, 32)
        assert err_out.shape == (8, 32)
        assert f.dtype == torch.float64
    
    def test_padded_multidim(self):
        """Padded variants should work with multi-dimensional tensors."""
        f = err.new_t(4)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True, False, False, True]))
        
        z = torch.randn(4, 8, 16)
        ok_z = err.take_ok_p(f, z, fill=-999.0)
        
        assert ok_z.shape == z.shape
        # Error samples (0, 3) should be all -999.0
        assert (ok_z[0] == -999.0).all()
        assert (ok_z[3] == -999.0).all()
        # OK samples (1, 2) should be original
        assert torch.allclose(ok_z[1], z[1])
        assert torch.allclose(ok_z[2], z[2])


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_batch(self):
        """Operations should handle empty batch (N=0)."""
        x = torch.randn(0, 32)
        f = err.new(x)
        assert f.shape == (0, 4)
        assert f.dtype == torch.float64
    
    def test_single_sample(self):
        """Operations should handle single sample."""
        x = torch.randn(1, 32)
        f = err.new(x)
        f = err.push(f, err.NAN, location=1, where=torch.tensor([True]))
        assert err.has_nan(f).all()
    
    def test_large_batch(self):
        """Operations should handle large batch."""
        x = torch.randn(1024, 32)
        f = err.new(x)
        mask = torch.rand(1024) > 0.5
        f = err.push(f, err.NAN, location=1, where=mask)
        
        has_nan = err.has_nan(f)
        assert (has_nan == mask).all()
    
    def test_max_location(self):
        """Should handle max location value (1023)."""
        f = err.from_code(ErrorCode.NAN, location=1023, batch_size=4)
        locs = err.get_first_location(f)
        assert (locs == 1023).all()
    
    def test_all_severity_levels(self):
        """Should handle all severity levels."""
        from torchguard.src.experimental.ops import Float64ErrorOps
        
        for sev in [Severity.WARN, Severity.ERROR, Severity.CRITICAL]:
            f = Float64ErrorOps.from_code(ErrorCode.NAN, location=1, n=4, severity=sev)
            sevs = Float64ErrorOps.get_first_severity(f)
            assert (sevs == sev).all(), f"Expected {sev}, got {sevs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

