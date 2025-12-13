"""
Tests for tensorcheck and tracked decorators.

Tests cover:
- @tensorcheck on methods: compile skip, validation, NaN/Inf auto-detection
- @tracked on classes: _fx_path injection for location tracking

Run with:
    pytest torchguard/tests/test_tensorcheck.py -v
"""
import pytest
import torch
import torch.nn as nn

from torchguard import tensorcheck, tracked
from torchguard import ErrorCode, ErrorLocation, error_t, err, flags as flags_ns, push
from torchguard.src.core.severity import Severity


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILE SKIP TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompileSkip:
    """Tests for torch.compile compatibility."""
    
    def test_tensorcheck_skips_during_compile(self) -> None:
        """Verify tensorcheck skips validation during torch.compile."""
        @tensorcheck
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2
        
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4, 8)
        
        # Should not raise or cause graph breaks
        result = compiled_fn(x)
        assert result.shape == x.shape
    
    def test_tensorcheck_function_compiles_fullgraph(self) -> None:
        """Verify decorated function can compile with fullgraph=True."""
        @tensorcheck
        def compute(x: torch.Tensor) -> torch.Tensor:
            return x.relu()
        
        # This should not raise
        compiled = torch.compile(compute, backend="eager", fullgraph=True)
        
        x = torch.randn(2, 4)
        result = compiled(x)
        assert result.shape == (2, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# NO RESULT WRAPPING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoResultWrapping:
    """Tests that tensorcheck doesn't wrap returns in Result."""
    
    def test_tensorcheck_returns_raw_value(self) -> None:
        """Verify tensorcheck returns raw value, not Result."""
        @tensorcheck
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2
        
        x = torch.randn(2, 4)
        result = fn(x)
        
        # Should be a tensor, not a Result
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
    
    def test_tensorcheck_returns_raw_tuple(self) -> None:
        """Verify tensorcheck returns raw tuple, not Result."""
        @tensorcheck
        def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x * 2
        
        x = torch.randn(2, 4)
        result = fn(x)
        
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)
    
    def test_tensorcheck_no_result_wrapping_with_error_t(self) -> None:
        """Verify tensorcheck returns raw tuple with error_t."""
        @tensorcheck
        def fn(x: torch.Tensor):
            flags = err.new_t(x.shape[0], x.device)
            return x, flags
        
        x = torch.randn(2, 4)
        result = fn(x)
        
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert result[1].dtype == torch.int64  # error_t


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS DECORATION TESTS (@tracked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassDecoration:
    """Tests for @tracked on classes."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_tracked_class_injects_fx_path(self) -> None:
        """Verify @tracked on class injects _fx_path into submodules."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(8, 4)
                self.decoder = nn.Linear(4, 8)
            
            def forward(self, x):
                return self.encoder(x)
        
        model = Model()
        
        assert hasattr(model.encoder, '_fx_path')
        assert model.encoder._fx_path == "encoder"
        assert model.decoder._fx_path == "decoder"
    
    def test_tracked_class_nested_modules(self) -> None:
        """Verify @tracked injects _fx_path for nested modules."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(8, 4),
                    nn.ReLU(),
                )
            
            def forward(self, x):
                return self.encoder(x)
        
        model = Model()
        
        assert hasattr(model.encoder, '_fx_path')
        assert model.encoder._fx_path == "encoder"
        # Sequential children
        assert hasattr(model.encoder[0], '_fx_path')
        assert model.encoder[0]._fx_path == "encoder.0"
    
    def test_tracked_class_registers_locations(self) -> None:
        """Verify @tracked registers all submodule paths with ErrorLocation."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(8, 4)
                self.layer2 = nn.Linear(4, 2)
            
            def forward(self, x):
                return self.layer2(self.layer1(x))
        
        model = Model()
        
        # Paths should be registered
        assert ErrorLocation.is_registered("layer1")
        assert ErrorLocation.is_registered("layer2")
    
    def test_tracked_class_returns_tuple(self) -> None:
        """Verify @tracked class returns raw tuple, not Result."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                return self.linear(x), flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        result = model(x)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR REQUIREMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassDecorationRequiresModule:
    """Tests that @tracked on class requires nn.Module."""
    
    def test_tracked_class_requires_nn_module(self) -> None:
        """Verify @tracked on non-Module class raises TypeError."""
        with pytest.raises(TypeError, match="requires nn.Module subclass"):
            @tracked
            class NotAModule:
                pass
    
    def test_tensorcheck_rejects_class(self) -> None:
        """Verify @tensorcheck on class raises TypeError."""
        with pytest.raises(TypeError, match="cannot be applied to classes"):
            @tensorcheck
            class NotAllowed(nn.Module):
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# FROZEN MODULE WARNING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrozenModuleWarning:
    """Tests that @tracked warns when _fx_path injection fails."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_tracked_warns_on_frozen_submodule(self) -> None:
        """Verify @tracked warns when submodule can't receive _fx_path."""
        import warnings
        
        # Create a module that blocks attribute setting
        class FrozenModule(nn.Module):
            def __setattr__(self, name: str, value) -> None:
                if name == '_fx_path':
                    raise AttributeError("Cannot set _fx_path on frozen module")
                super().__setattr__(name, value)
            
            def forward(self, x):
                return x
        
        # Create model with frozen submodule
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            @tracked
            class ModelWithFrozen(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.normal = nn.Linear(8, 4)
                    self.frozen = FrozenModule()
                
                def forward(self, x):
                    return self.frozen(self.normal(x))
            
            model = ModelWithFrozen()
        
        # Should have warned about the frozen module
        fx_path_warnings = [
            warning for warning in w 
            if "Could not inject _fx_path" in str(warning.message)
        ]
        assert len(fx_path_warnings) == 1
        assert "frozen (FrozenModule)" in str(fx_path_warnings[0].message)
    
    def test_tracked_no_warning_on_normal_modules(self) -> None:
        """Verify @tracked doesn't warn when all modules accept _fx_path."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            @tracked
            class NormalModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(8, 4)
                    self.linear2 = nn.Linear(4, 2)
                
                def forward(self, x):
                    return self.linear2(self.linear1(x))
            
            model = NormalModel()
        
        # Should NOT have warned
        fx_path_warnings = [
            warning for warning in w 
            if "Could not inject _fx_path" in str(warning.message)
        ]
        assert len(fx_path_warnings) == 0
        
        # But _fx_path should be injected
        assert model.linear1._fx_path == "linear1"
        assert model.linear2._fx_path == "linear2"


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECT NaN/Inf TESTS (@tensorcheck on methods)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoDetect:
    """Tests for auto_detect=True (default) NaN/Inf detection on methods."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_auto_detect_nan_in_output(self) -> None:
        """Verify auto_detect finds NaN in output tensor."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck  # auto_detect=True by default
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                # Intentionally introduce NaN
                output = x.clone()
                output[0, 0] = float('nan')
                return output, flags
        
        model = Model()
        x = torch.randn(3, 8)
        
        output, flags = model(x)
        
        # Should have detected NaN in sample 0
        assert err.is_err(flags)[0].item()  # Sample 0 has error
        assert not err.is_err(flags)[1].item()  # Sample 1 OK
        assert not err.is_err(flags)[2].item()  # Sample 2 OK
        
        # Should specifically be NAN code
        assert err.has_nan(flags)[0].item()
    
    def test_auto_detect_inf_in_output(self) -> None:
        """Verify auto_detect finds Inf in output tensor."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck  # auto_detect=True by default
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                # Intentionally introduce Inf
                output = x.clone()
                output[1, 0] = float('inf')
                return output, flags
        
        model = Model()
        x = torch.randn(3, 8)
        
        output, flags = model(x)
        
        # Should have detected Inf in sample 1
        assert not err.is_err(flags)[0].item()  # Sample 0 OK
        assert err.is_err(flags)[1].item()  # Sample 1 has error
        assert not err.is_err(flags)[2].item()  # Sample 2 OK
        
        # Should specifically be INF code
        assert err.has_inf(flags)[1].item()
    
    def test_auto_detect_multiple_outputs(self) -> None:
        """Verify auto_detect checks all output tensors."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                out1 = x.clone()
                out2 = x.clone()
                out2[0, 0] = float('nan')  # NaN in second output
                return out1, out2, flags
        
        model = Model()
        x = torch.randn(2, 4)
        
        out1, out2, flags = model(x)
        
        # Should have detected NaN from out2 in sample 0
        assert err.is_err(flags)[0].item()
    
    def test_auto_detect_no_error_when_clean(self) -> None:
        """Verify auto_detect doesn't flag clean tensors."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            @tensorcheck
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                return self.linear(x), flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        output, flags = model(x)
        
        # Should be all OK
        assert err.is_ok(flags).all().item()
    
    def test_auto_detect_disabled(self) -> None:
        """Verify auto_detect=False disables detection."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck(auto_detect=False)
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                # Intentionally introduce NaN
                output = x.clone()
                output[0, 0] = float('nan')
                return output, flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        output, flags = model(x)
        
        # NaN should NOT be detected (auto_detect=False)
        assert err.is_ok(flags).all().item()
    
    def test_auto_detect_skips_int_tensors(self) -> None:
        """Verify auto_detect skips non-floating-point tensors."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                # Int tensor output
                int_output = torch.randint(0, 10, (x.shape[0], 4))
                return int_output, flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        output, flags = model(x)
        
        # Should be OK (int tensors skipped)
        assert err.is_ok(flags).all().item()
    
    def test_auto_detect_compiles(self) -> None:
        """Verify auto_detect works with torch.compile (fullgraph=True)."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            @tensorcheck
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                output = self.linear(x)
                # NaN/Inf auto-detection happens here - should compile!
                return output, flags
        
        model = Model()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        # Clean input
        x = torch.randn(3, 8)
        output, flags = compiled(x)
        assert err.is_ok(flags).all().item()
        
        # Input that causes NaN in output
        x_bad = torch.randn(3, 8) * 1e38  # Large values may cause NaN after linear
        x_bad[0, :] = float('nan')  # Explicit NaN in input will propagate
        output, flags = compiled(x_bad)
        # Sample 0 input has NaN, so output should have NaN detected
        assert err.has_nan(flags)[0].item()
    
    def test_auto_detect_accumulates_with_existing_errors(self) -> None:
        """Verify auto_detect accumulates with errors already in flags."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                # Manually push an OOB error
                oob_mask = torch.tensor([False, True])
                flags = push(flags, ErrorCode.OUT_OF_BOUNDS, self, where=oob_mask)
                
                # Output with NaN in sample 0
                output = x.clone()
                output[0, 0] = float('nan')
                return output, flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        output, flags = model(x)
        
        # Sample 0 should have NaN (from auto_detect)
        assert err.has_nan(flags)[0].item()
        
        # Sample 1 should have OOB (from manual push)
        assert err.is_err(flags)[1].item()


class TestAutoDetectEdgeCases:
    """Edge case tests for auto_detect feature on methods."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_auto_detect_non_tuple_return(self) -> None:
        """Verify auto_detect handles non-tuple returns gracefully."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            @tensorcheck
            def forward(self, x):
                # Returns just tensor, not tuple
                return self.linear(x)
        
        model = Model()
        x = torch.randn(2, 8)
        
        # Should not crash
        output = model(x)
        assert isinstance(output, torch.Tensor)
    
    def test_auto_detect_single_element_tuple(self) -> None:
        """Verify auto_detect handles single-element tuples."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            @tensorcheck
            def forward(self, x):
                # Returns single-element tuple
                return (self.linear(x),)
        
        model = Model()
        x = torch.randn(2, 8)
        
        # Should not crash
        result = model(x)
        assert isinstance(result, tuple)
        assert len(result) == 1
    
    def test_auto_detect_non_error_t_last_element(self) -> None:
        """Verify auto_detect skips when last element isn't error_t."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            @tensorcheck
            def forward(self, x):
                # Returns (tensor, tensor) not (tensor, error_t)
                output = self.linear(x)
                other = torch.zeros(x.shape[0])  # float tensor, not int64 error_t
                return output, other
        
        model = Model()
        x = torch.randn(2, 8)
        
        # Should not crash (auto_detect skipped)
        output, other = model(x)
        assert isinstance(output, torch.Tensor)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECT SPECIFIC CODES TESTS (@tensorcheck on methods)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoDetectSpecificCodes:
    """Tests for auto_detect with specific ErrorCode sets on methods."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_auto_detect_nan_only(self) -> None:
        """Verify auto_detect={ErrorCode.NAN} only detects NaN."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck(auto_detect={ErrorCode.NAN})
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                output = x.clone()
                output[0, 0] = float('nan')  # NaN in sample 0
                output[1, 0] = float('inf')  # Inf in sample 1
                return output, flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        output, flags = model(x)
        
        # Sample 0 should have NaN detected
        assert err.has_nan(flags)[0].item()
        
        # Sample 1 should NOT have Inf detected (only NaN is configured)
        assert not err.has_inf(flags)[1].item()
        assert err.is_ok(flags)[1].item()
    
    def test_auto_detect_inf_only(self) -> None:
        """Verify auto_detect={ErrorCode.INF} only detects Inf."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck(auto_detect={ErrorCode.INF})
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                output = x.clone()
                output[0, 0] = float('nan')  # NaN in sample 0
                output[1, 0] = float('inf')  # Inf in sample 1
                return output, flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        output, flags = model(x)
        
        # Sample 0 should NOT have NaN detected (only Inf is configured)
        assert not err.has_nan(flags)[0].item()
        assert err.is_ok(flags)[0].item()
        
        # Sample 1 should have Inf detected
        assert err.has_inf(flags)[1].item()
    
    def test_auto_detect_both_explicit(self) -> None:
        """Verify auto_detect={ErrorCode.NAN, ErrorCode.INF} detects both."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck(auto_detect={ErrorCode.NAN, ErrorCode.INF})
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                output = x.clone()
                output[0, 0] = float('nan')
                output[1, 0] = float('inf')
                return output, flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        output, flags = model(x)
        
        # Both should be detected
        assert err.has_nan(flags)[0].item()
        assert err.has_inf(flags)[1].item()
    
    def test_auto_detect_none_same_as_false(self) -> None:
        """Verify auto_detect=None behaves same as False."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck(auto_detect=None)
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                output = x.clone()
                output[0, 0] = float('nan')
                return output, flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        output, flags = model(x)
        
        # NaN should NOT be detected
        assert err.is_ok(flags).all().item()
    
    def test_auto_detect_empty_set_same_as_false(self) -> None:
        """Verify auto_detect=set() behaves same as False."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            
            @tensorcheck(auto_detect=set())
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                output = x.clone()
                output[0, 0] = float('nan')
                return output, flags
        
        model = Model()
        x = torch.randn(2, 8)
        
        output, flags = model(x)
        
        # NaN should NOT be detected
        assert err.is_ok(flags).all().item()
    
    def test_auto_detect_specific_codes_compiles(self) -> None:
        """Verify auto_detect with specific codes works with torch.compile."""
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            @tensorcheck(auto_detect={ErrorCode.NAN})
            def forward(self, x):
                flags = err.new_t(x.shape[0], x.device)
                output = self.linear(x)
                return output, flags
        
        model = Model()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        # Clean input
        x = torch.randn(3, 8)
        output, flags = compiled(x)
        assert err.is_ok(flags).all().item()


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""
    
    def test_tensorcheck_method_decorator(self) -> None:
        """Verify @tensorcheck works on methods."""
        class MyClass:
            @tensorcheck
            def compute(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2
        
        obj = MyClass()
        x = torch.randn(2, 4)
        result = obj.compute(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
    
    def test_tensorcheck_with_optional_param(self) -> None:
        """Verify @tensorcheck handles Optional parameters."""
        from typing import Optional
        
        @tensorcheck
        def fn(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
            if y is None:
                return x
            return x + y
        
        x = torch.randn(2, 4)
        
        # Without optional
        result1 = fn(x)
        assert isinstance(result1, torch.Tensor)
        
        # With optional
        y = torch.randn(2, 4)
        result2 = fn(x, y)
        assert isinstance(result2, torch.Tensor)

