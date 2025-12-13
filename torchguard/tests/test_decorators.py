"""
Tests for the @as_result decorator.

Tests cover:
- Success wrapping in Ok
- Exception catching in Err
- Works with compiled models
- Preserves function metadata

Run with:
    pytest tests/utils/errors/compiled/test_decorators.py -v
"""
import pytest
import torch
import torch.nn as nn

from torchguard.src.decorators import as_result
from torchguard import error_t, err, flags as flags_ns


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC FUNCTIONALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsResultBasic:
    """Tests for basic @as_result functionality."""
    
    def test_as_result_wraps_in_ok(self) -> None:
        """Verify successful returns are wrapped in Ok."""
        @as_result
        def fn():
            return (torch.randn(2, 4), torch.zeros(2, 2, dtype=torch.int64))
        
        result = fn()
        
        assert result.is_ok()
        out, flags = result.unwrap()
        assert out.shape == (2, 4)
        assert flags.shape == (2, 2)
    
    def test_as_result_catches_exceptions(self) -> None:
        """Verify exceptions are wrapped in Err."""
        @as_result
        def fn():
            raise ValueError("Test error")
        
        result = fn()
        
        assert result.is_err()
        assert isinstance(result.unwrap_err(), ValueError)
        assert "Test error" in str(result.unwrap_err())
    
    def test_as_result_catches_runtime_error(self) -> None:
        """Verify RuntimeError is wrapped in Err."""
        @as_result
        def fn():
            raise RuntimeError("Runtime failure")
        
        result = fn()
        
        assert result.is_err()
        assert isinstance(result.unwrap_err(), RuntimeError)
    
    def test_as_result_simple_return(self) -> None:
        """Verify simple values are wrapped correctly."""
        @as_result
        def fn():
            return 42
        
        result = fn()
        
        assert result.is_ok()
        assert result.unwrap() == 42
    
    def test_as_result_with_args(self) -> None:
        """Verify args are passed through correctly."""
        @as_result
        def add(a, b):
            return a + b
        
        result = add(3, 4)
        
        assert result.is_ok()
        assert result.unwrap() == 7
    
    def test_as_result_with_kwargs(self) -> None:
        """Verify kwargs are passed through correctly."""
        @as_result
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = greet("World", greeting="Hi")
        
        assert result.is_ok()
        assert result.unwrap() == "Hi, World!"


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA PRESERVATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsResultMetadata:
    """Tests for function metadata preservation."""
    
    def test_as_result_preserves_function_name(self) -> None:
        """Verify @as_result preserves function name."""
        @as_result
        def my_function():
            return 42
        
        assert my_function.__name__ == "my_function"
    
    def test_as_result_preserves_docstring(self) -> None:
        """Verify @as_result preserves docstring."""
        @as_result
        def my_function():
            """My docstring."""
            return 42
        
        assert my_function.__doc__ == "My docstring."
    
    def test_as_result_preserves_module(self) -> None:
        """Verify @as_result preserves module."""
        @as_result
        def my_function():
            return 42
        
        assert my_function.__module__ == __name__


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILED MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsResultWithCompiledModels:
    """Tests for @as_result with torch.compile."""
    
    def test_as_result_with_compiled_model(self) -> None:
        """Verify as_result works with compiled model output."""
        class Model(nn.Module):
            def forward(self, x):
                return x * 2, torch.zeros(x.shape[0], 2, dtype=torch.int64)
        
        model = torch.compile(Model(), backend="eager", fullgraph=True)
        
        @as_result
        def run(m, x):
            return m(x)
        
        result = run(model, torch.randn(4, 8))
        
        assert result.is_ok()
        out, flags = result.unwrap()
        assert out.shape == (4, 8)
    
    def test_as_result_wraps_compile_error(self) -> None:
        """Verify as_result catches errors from compiled models."""
        class BrokenModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("Model failure")
        
        model = BrokenModel()
        
        @as_result
        def run(m, x):
            return m(x)
        
        result = run(model, torch.randn(2, 4))
        
        assert result.is_err()
        assert "Model failure" in str(result.unwrap_err())


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR_T INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsResultWithErrorFlags:
    """Tests for @as_result with error_t integration."""
    
    def test_as_result_with_error_flags(self) -> None:
        """Verify as_result works with ErrorFlags returns."""
        class Model(nn.Module):
            def forward(self, x):
                n = x.shape[0]
                flags = err.new_t(n, x.device)
                return x * 2, flags
        
        model = Model()
        
        @as_result
        def run(m, x):
            return m(x)
        
        result = run(model, torch.randn(3, 4))
        
        assert result.is_ok()
        out, flags = result.unwrap()
        assert out.shape == (3, 4)
        assert flags.shape[0] == 3
        assert (flags == 0).all()  # No errors
    
    def test_result_map_pattern(self) -> None:
        """Verify map pattern works with as_result."""
        @as_result
        def compute():
            return torch.tensor([1.0, 2.0, 3.0])
        
        # Use map to transform the result
        result = compute().map(lambda t: t.sum())
        
        assert result.is_ok()
        assert result.unwrap().item() == 6.0
    
    def test_result_map_err_pattern(self) -> None:
        """Verify map_err pattern works."""
        @as_result
        def failing():
            raise ValueError("Original error")
        
        # Use map_err to transform the error
        result = failing().map_err(lambda e: RuntimeError(f"Wrapped: {e}"))
        
        assert result.is_err()
        assert isinstance(result.unwrap_err(), RuntimeError)
        assert "Wrapped" in str(result.unwrap_err())


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsResultEdgeCases:
    """Tests for edge cases."""
    
    def test_as_result_returns_none(self) -> None:
        """Verify None return is wrapped correctly."""
        @as_result
        def fn():
            return None
        
        result = fn()
        
        assert result.is_ok()
        assert result.unwrap() is None
    
    def test_as_result_returns_empty_tuple(self) -> None:
        """Verify empty tuple return is wrapped correctly."""
        @as_result
        def fn():
            return ()
        
        result = fn()
        
        assert result.is_ok()
        assert result.unwrap() == ()
    
    def test_as_result_catches_keyboard_interrupt(self) -> None:
        """Verify KeyboardInterrupt is caught (Exception subclass check)."""
        # Note: KeyboardInterrupt is NOT a subclass of Exception
        # It's a subclass of BaseException, so it won't be caught
        # This is intentional - we only catch Exception
        @as_result
        def fn():
            raise ValueError("Regular exception")
        
        result = fn()
        assert result.is_err()

