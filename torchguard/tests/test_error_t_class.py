"""
Tests for the error_t class API.

This tests the error_t class with its static methods for creating and
querying error flags. All methods should work with torch.compile(fullgraph=True).

API:
    err.new(x)          - Create empty flags from reference tensor
    err.is_ok(flags)    - Bool mask: True where sample has NO errors
    err.is_err(flags)   - Bool mask: True where sample HAS errors
    err.get_ok(flags)   - Filtered flags (OK samples only)
    err.get_err(flags)  - Filtered flags (error samples only)
    err.take_ok(flags, z)    - Filter any tensor z to OK samples
    err.take_err(flags, z)   - Filter any tensor z to error samples

Run with:
    pytest tests/utils/errors/compiled/test_error_t_class.py -v
"""
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torchguard import (
    error_t,
    ErrorCode,
    ErrorLocation,
    push,
    has_err,
    err,
    flags as flags_ns,
    CONFIG,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CREATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorTNew:
    """Tests for err.new() factory method."""
    
    def test_new_creates_correct_shape(self) -> None:
        """err.new(x) creates tensor with correct shape."""
        x = torch.randn(32, 16)
        flags = err.new(x)
        
        assert flags.shape == (32, CONFIG.num_words)
        assert flags.dtype == torch.int64
    
    def test_new_creates_zeros(self) -> None:
        """err.new(x) creates all-zero tensor (no errors)."""
        x = torch.randn(16, 8)
        flags = err.new(x)
        
        assert (flags == 0).all()
    
    def test_new_inherits_device(self) -> None:
        """err.new(x) creates tensor on same device as x."""
        x = torch.randn(16, 8)
        flags = err.new(x)
        
        assert flags.device == x.device
    
    def test_new_compiles(self) -> None:
        """err.new(x) works inside torch.compile(fullgraph=True)."""
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                flags = err.new(x)
                out = self.linear(x)
                return out, flags
        
        model = Model()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        x = torch.randn(16, 8)
        out, flags = compiled(x)
        
        assert out.shape == (16, 4)
        assert flags.shape == (16, CONFIG.num_words)
        assert (flags == 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# MASK TESTS (is_ok, is_err)
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorTMasks:
    """Tests for err.is_ok() and err.is_err() mask methods."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_is_ok_all_clean(self) -> None:
        """is_ok returns all True when no errors."""
        x = torch.randn(8, 4)
        flags = err.new(x)
        
        mask = err.is_ok(flags)
        
        assert mask.shape == (8,)
        assert mask.dtype == torch.bool
        assert mask.all()
    
    def test_is_err_all_clean(self) -> None:
        """is_err returns all False when no errors."""
        x = torch.randn(8, 4)
        flags = err.new(x)
        
        mask = err.is_err(flags)
        
        assert mask.shape == (8,)
        assert mask.dtype == torch.bool
        assert not mask.any()
    
    def test_is_ok_with_errors(self) -> None:
        """is_ok returns False where errors exist."""
        x = torch.randn(8, 4)
        flags = err.new(x)
        
        # Push error on samples 2 and 5
        error_mask = torch.zeros(8, dtype=torch.bool)
        error_mask[2] = True
        error_mask[5] = True
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        ok_mask = err.is_ok(flags)
        
        assert ok_mask[0] == True
        assert ok_mask[1] == True
        assert ok_mask[2] == False  # Has error
        assert ok_mask[3] == True
        assert ok_mask[4] == True
        assert ok_mask[5] == False  # Has error
        assert ok_mask[6] == True
        assert ok_mask[7] == True
    
    def test_is_err_with_errors(self) -> None:
        """is_err returns True where errors exist."""
        x = torch.randn(8, 4)
        flags = err.new(x)
        
        # Push error on samples 2 and 5
        error_mask = torch.zeros(8, dtype=torch.bool)
        error_mask[2] = True
        error_mask[5] = True
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        err_mask = err.is_err(flags)
        
        assert err_mask[0] == False
        assert err_mask[1] == False
        assert err_mask[2] == True  # Has error
        assert err_mask[3] == False
        assert err_mask[4] == False
        assert err_mask[5] == True  # Has error
        assert err_mask[6] == False
        assert err_mask[7] == False
    
    def test_masks_are_complementary(self) -> None:
        """is_ok and is_err are logical complements."""
        x = torch.randn(16, 4)
        flags = err.new(x)
        
        # Add some errors
        error_mask = torch.rand(16) > 0.5
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        ok_mask = err.is_ok(flags)
        err_mask = err.is_err(flags)
        
        # They should be logical complements
        assert (ok_mask == ~err_mask).all()
    
    def test_masks_compile(self) -> None:
        """is_ok and is_err work inside torch.compile."""
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                flags = err.new(x)
                out = self.linear(x)
                
                # Detect NaN
                nan_mask = torch.isnan(out).any(dim=-1)
                flags = push(flags, ErrorCode.NAN, self, where=nan_mask)
                
                ok_mask = err.is_ok(flags)
                err_mask = err.is_err(flags)
                
                return ok_mask, err_mask, flags
        
        model = Model()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        x = torch.randn(16, 8)
        ok_mask, err_mask, flags = compiled(x)
        
        assert ok_mask.shape == (16,)
        assert err_mask.shape == (16,)
        assert (ok_mask == ~err_mask).all()


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER FLAGS TESTS (get_ok, get_err)
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorTGetFilters:
    """Tests for err.get_ok() and err.get_err() filter methods."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_get_ok_all_clean(self) -> None:
        """get_ok returns all flags when no errors."""
        x = torch.randn(8, 4)
        flags = err.new(x)
        
        ok_flags = err.get_ok(flags)
        
        assert ok_flags.shape == (8, CONFIG.num_words)
    
    def test_get_err_all_clean(self) -> None:
        """get_err returns empty when no errors."""
        x = torch.randn(8, 4)
        flags = err.new(x)
        
        err_flags = err.get_err(flags)
        
        assert err_flags.shape == (0, CONFIG.num_words)
    
    def test_get_ok_filters_correctly(self) -> None:
        """get_ok returns only flags for samples without errors."""
        x = torch.randn(8, 4)
        flags = err.new(x)
        
        # Push error on samples 2, 5, 7
        error_mask = torch.zeros(8, dtype=torch.bool)
        error_mask[2] = True
        error_mask[5] = True
        error_mask[7] = True
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        ok_flags = err.get_ok(flags)
        
        # 8 samples - 3 errors = 5 OK samples
        assert ok_flags.shape == (5, CONFIG.num_words)
        # All OK flags should be zero
        assert (ok_flags == 0).all()
    
    def test_get_err_filters_correctly(self) -> None:
        """get_err returns only flags for samples with errors."""
        x = torch.randn(8, 4)
        flags = err.new(x)
        
        # Push error on samples 2, 5, 7
        error_mask = torch.zeros(8, dtype=torch.bool)
        error_mask[2] = True
        error_mask[5] = True
        error_mask[7] = True
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        err_flags = err.get_err(flags)
        
        # 3 samples with errors
        assert err_flags.shape == (3, CONFIG.num_words)
        # All error flags should be non-zero
        assert (err_flags != 0).any(dim=-1).all()


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER ANY TENSOR TESTS (Ok, Err)
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorTOkErr:
    """Tests for err.take_ok() and err.take_err() tensor filter methods."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_Ok_filters_tensor(self) -> None:
        """take_ok(flags, z) returns z filtered to only OK samples."""
        batch_size = 8
        x = torch.randn(batch_size, 4)
        z = torch.arange(batch_size).float()  # [0, 1, 2, 3, 4, 5, 6, 7]
        flags = err.new(x)
        
        # Push error on samples 2, 5
        error_mask = torch.zeros(batch_size, dtype=torch.bool)
        error_mask[2] = True
        error_mask[5] = True
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        ok_z = err.take_ok(flags, z)
        
        # Should contain [0, 1, 3, 4, 6, 7] (samples without errors)
        assert ok_z.shape == (6,)
        assert torch.allclose(ok_z, torch.tensor([0., 1., 3., 4., 6., 7.]))
    
    def test_Err_filters_tensor(self) -> None:
        """take_err(flags, z) returns z filtered to only error samples."""
        batch_size = 8
        x = torch.randn(batch_size, 4)
        z = torch.arange(batch_size).float()  # [0, 1, 2, 3, 4, 5, 6, 7]
        flags = err.new(x)
        
        # Push error on samples 2, 5
        error_mask = torch.zeros(batch_size, dtype=torch.bool)
        error_mask[2] = True
        error_mask[5] = True
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        err_z = err.take_err(flags, z)
        
        # Should contain [2, 5] (samples with errors)
        assert err_z.shape == (2,)
        assert torch.allclose(err_z, torch.tensor([2., 5.]))
    
    def test_Ok_Err_are_complementary(self) -> None:
        """Ok and Err together cover all samples."""
        batch_size = 16
        x = torch.randn(batch_size, 4)
        z = torch.arange(batch_size)
        flags = err.new(x)
        
        # Random errors
        error_mask = torch.rand(batch_size) > 0.5
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        ok_z = err.take_ok(flags, z)
        err_z = err.take_err(flags, z)
        
        # Combined should have all samples
        combined = torch.cat([ok_z, err_z])
        assert combined.shape[0] == batch_size
        assert set(combined.tolist()) == set(range(batch_size))
    
    def test_Ok_with_multidim_tensor(self) -> None:
        """Ok works with multi-dimensional tensors."""
        batch_size = 8
        x = torch.randn(batch_size, 4)
        z = torch.randn(batch_size, 3, 5)  # (batch, channels, features)
        flags = err.new(x)
        
        # Push error on samples 1, 4
        error_mask = torch.zeros(batch_size, dtype=torch.bool)
        error_mask[1] = True
        error_mask[4] = True
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        ok_z = err.take_ok(flags, z)
        
        # 8 - 2 = 6 OK samples
        assert ok_z.shape == (6, 3, 5)
    
    def test_Err_with_multidim_tensor(self) -> None:
        """Err works with multi-dimensional tensors."""
        batch_size = 8
        x = torch.randn(batch_size, 4)
        z = torch.randn(batch_size, 3, 5)  # (batch, channels, features)
        flags = err.new(x)
        
        # Push error on samples 1, 4
        error_mask = torch.zeros(batch_size, dtype=torch.bool)
        error_mask[1] = True
        error_mask[4] = True
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        err_z = err.take_err(flags, z)
        
        # 2 error samples
        assert err_z.shape == (2, 3, 5)
    
    def test_Ok_all_clean(self) -> None:
        """Ok returns all when no errors."""
        batch_size = 8
        x = torch.randn(batch_size, 4)
        z = torch.arange(batch_size)
        flags = err.new(x)
        
        ok_z = err.take_ok(flags, z)
        
        assert ok_z.shape == (batch_size,)
        assert torch.allclose(ok_z, z)
    
    def test_Err_all_clean(self) -> None:
        """Err returns empty when no errors."""
        batch_size = 8
        x = torch.randn(batch_size, 4)
        z = torch.arange(batch_size)
        flags = err.new(x)
        
        err_z = err.take_err(flags, z)
        
        assert err_z.shape == (0,)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorTCompiles:
    """Tests that all error_t methods work with torch.compile."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_full_workflow_compiles(self) -> None:
        """Complete error_t workflow compiles with fullgraph=True."""
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                flags = err.new(x)
                out = self.linear(x)
                
                # Detect NaN
                nan_mask = torch.isnan(out).any(dim=-1)
                flags = push(flags, ErrorCode.NAN, self, where=nan_mask)
                
                return out, flags
        
        model = Model()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        x = torch.randn(16, 8)
        out, flags = compiled(x)
        
        assert out.shape == (16, 4)
        assert flags.shape == (16, CONFIG.num_words)
    
    def test_dynamic_batch_size(self) -> None:
        """error_t.new handles dynamic batch sizes."""
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                flags = err.new(x)
                out = self.linear(x)
                return out, flags
        
        model = Model()
        compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=True)
        
        for batch_size in [8, 16, 32, 64]:
            x = torch.randn(batch_size, 8)
            out, flags = compiled(x)
            assert out.shape == (batch_size, 4)
            assert flags.shape == (batch_size, CONFIG.num_words)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH EXISTING API
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorTIntegration:
    """Tests that error_t works with existing ErrorFlags API."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_new_and_new_t_produce_same_result(self) -> None:
        """err.new() and err.new_t() produce equivalent results."""
        x = torch.randn(16, 8)
        
        flags_new = err.new(x)
        flags_new_t = err.new_t(x.shape[0], x.device)
        
        assert flags_new.shape == flags_new_t.shape
        assert flags_new.dtype == flags_new_t.dtype
        assert (flags_new == flags_new_t).all()
    
    def test_is_ok_is_err_are_consistent(self) -> None:
        """err.is_ok() and err.is_err() are logical opposites."""
        x = torch.randn(16, 8)
        flags = err.new(x)
        
        # Add some errors
        error_mask = torch.rand(16) > 0.5
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        is_ok = err.is_ok(flags)
        is_err = err.is_err(flags)
        
        # They should be logical opposites
        assert (is_ok ^ is_err).all()
        # is_ok should match where we DIDN'T add errors
        assert (is_ok == ~error_mask).all()
    
    def test_error_t_works_with_ErrorFlags_boundary(self) -> None:
        """error_t flags work with ErrorFlags Python boundary methods."""
        x = torch.randn(16, 8)
        flags = err.new(x)
        
        # Add some errors
        error_mask = torch.tensor([True, False] * 8)
        flags = push(flags, ErrorCode.NAN, "test", where=error_mask)
        
        # ErrorFlags.unpack should work with flags from error_t
        errors = flags_ns.unpack(flags, sample_idx=0)
        assert len(errors) == 1
        assert errors[0].code_name == "NAN"
    
    def test_has_err_works_with_error_t(self) -> None:
        """has_err() works with flags created by err.new()."""
        x = torch.randn(16, 8)
        flags = err.new(x)
        
        assert not has_err(flags)
        
        # Add error
        flags = push(flags, ErrorCode.NAN, "test", where=torch.tensor([True] + [False] * 15))
        
        assert has_err(flags)
