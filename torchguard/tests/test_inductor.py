"""
Tests for TorchGuard with the inductor backend.

These tests verify that all core operations work correctly when compiled
with torch.compile(backend="inductor", fullgraph=True).

Run with:
    pytest torchguard/tests/test_inductor.py -v

Note: Some tests require CUDA. They will be skipped if CUDA is not available.
"""
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torchguard import (
    ErrorCode,
    ErrorLocation,
    Severity,
    err,
    flags as flags_ns,
    has_err,
    find,
    push,
    fix,
    flag_nan,
    flag_inf,
    flag_oob_indices,
    tracked,
    tensorcheck,
    IF,
    IS,
    OR,
)


# Skip all tests in this module if inductor is not available
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "_dynamo"),
    reason="torch._dynamo not available (PyTorch < 2.0)"
)


def get_device():
    """Get the best available device for inductor tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_inductor_error = None

def inductor_available() -> bool:
    """Check if inductor backend is available."""
    global _inductor_error
    try:
        # Try a simple compile to see if inductor works
        @torch.compile(backend="inductor")
        def _test_fn(x):
            return x + 1
        _test_fn(torch.tensor([1.0]))
        return True
    except Exception as e:
        _inductor_error = str(e)
        return False


def _get_skip_reason() -> str:
    if not inductor_available():
        if _inductor_error and "C++ compiler" in _inductor_error:
            return "inductor requires C++ compiler (g++/clang++) - install build-essential"
        return f"inductor backend not available: {_inductor_error or 'unknown error'}"
    return ""


skip_if_no_inductor = pytest.mark.skipif(
    not inductor_available(),
    reason=_get_skip_reason() or "inductor backend not available"
)


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC OPERATIONS WITH INDUCTOR
# ═══════════════════════════════════════════════════════════════════════════════

@skip_if_no_inductor
class TestInductorBasicOps:
    """Test basic error flag operations with inductor backend."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_err_new_compiles(self) -> None:
        """err.new(x) works with inductor backend."""
        device = get_device()
        
        @torch.compile(backend="inductor", fullgraph=True)
        def create_flags(x):
            return err.new(x)
        
        x = torch.randn(8, 32, device=device)
        f = create_flags(x)
        
        assert f.shape[0] == 8
        assert f.dtype == torch.int64
        assert not has_err(f)
    
    def test_err_new_t_compiles(self) -> None:
        """err.new_t(n, device) works with inductor backend."""
        device = get_device()
        
        @torch.compile(backend="inductor", fullgraph=True)
        def create_flags_explicit(n: int):
            return err.new_t(n, device)
        
        f = create_flags_explicit(16)
        
        assert f.shape[0] == 16
        assert f.dtype == torch.int64
        assert not has_err(f)
    
    def test_push_compiles(self) -> None:
        """err.push works with inductor backend."""
        device = get_device()
        loc_id = ErrorLocation.register("test_layer")
        
        @torch.compile(backend="inductor", fullgraph=True)
        def push_error(x):
            f = err.new(x)
            mask = x[:, 0] > 0  # Some samples flagged
            f = err.push(f, ErrorCode.NAN, loc_id, where=mask)
            return f
        
        x = torch.randn(8, 32, device=device)
        f = push_error(x)
        
        assert f.shape[0] == 8
        # Some samples should have errors (where x[:, 0] > 0)
    
    def test_is_ok_is_err_compile(self) -> None:
        """err.is_ok and err.is_err work with inductor backend."""
        device = get_device()
        loc_id = ErrorLocation.register("test_layer")
        
        @torch.compile(backend="inductor", fullgraph=True)
        def check_errors(x):
            f = err.new(x)
            mask = x[:, 0] > 0
            f = err.push(f, ErrorCode.NAN, loc_id, where=mask)
            ok = err.is_ok(f)
            bad = err.is_err(f)
            return f, ok, bad
        
        x = torch.randn(8, 32, device=device)
        f, ok, bad = check_errors(x)
        
        assert ok.shape == (8,)
        assert bad.shape == (8,)
        assert ok.dtype == torch.bool
        assert bad.dtype == torch.bool
        # ok and bad should be complementary
        assert (ok == ~bad).all()
    
    def test_partition_padded_compiles(self) -> None:
        """err.take_ok_p/take_err_p works with inductor backend.

        Note: partition()/take_ok()/take_err() use dynamic shapes which don't work with
        fullgraph=True. Use take_ok_p()/take_err_p() for static shapes.
        """
        device = get_device()
        loc_id = ErrorLocation.register("test_layer")
        
        @torch.compile(backend="inductor", fullgraph=True)
        def partition_by_error(x):
            f = err.new(x)
            mask = x[:, 0] > 0
            f = err.push(f, ErrorCode.NAN, loc_id, where=mask)
            # Use padded variants for static shapes
            ok_x = err.take_ok_p(f, x, fill=0.0)
            err_x = err.take_err_p(f, x, fill=0.0)
            return f, ok_x, err_x
        
        x = torch.randn(8, 32, device=device)
        f, ok_x, err_x = partition_by_error(x)
        
        # Both outputs have same shape as input (static)
        assert ok_x.shape == (8, 32)
        assert err_x.shape == (8, 32)
    
    def test_merge_compiles(self) -> None:
        """err.merge works with inductor backend."""
        device = get_device()
        loc1 = ErrorLocation.register("layer1")
        loc2 = ErrorLocation.register("layer2")
        
        @torch.compile(backend="inductor", fullgraph=True)
        def merge_flags(x):
            f1 = err.new(x)
            f2 = err.new(x)
            mask1 = x[:, 0] > 0
            mask2 = x[:, 1] > 0
            f1 = err.push(f1, ErrorCode.NAN, loc1, where=mask1)
            f2 = err.push(f2, ErrorCode.INF, loc2, where=mask2)
            return err.merge(f1, f2)
        
        x = torch.randn(8, 32, device=device)
        f = merge_flags(x)
        
        assert f.shape[0] == 8


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS WITH INDUCTOR
# ═══════════════════════════════════════════════════════════════════════════════

@skip_if_no_inductor
class TestInductorHelpers:
    """Test helper functions with inductor backend."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_flag_nan_compiles(self) -> None:
        """flag_nan works with inductor backend."""
        device = get_device()
        
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                out = self.linear(x)
                f = flag_nan(out, self.linear, f)
                return out, f
        
        model = Model().to(device)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        # Clean input
        x = torch.randn(8, 32, device=device)
        out, f = compiled(x)
        
        assert out.shape == (8, 16)
        assert not has_err(f)
        
        # Input that causes NaN (extreme values)
        x_bad = torch.randn(8, 32, device=device)
        x_bad[0] = float('nan')
        out, f = compiled(x_bad)
        
        assert has_err(f)
    
    def test_flag_inf_compiles(self) -> None:
        """flag_inf works with inductor backend."""
        device = get_device()
        
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                out = self.linear(x)
                f = flag_inf(out, self.linear, f)
                return out, f
        
        model = Model().to(device)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        x = torch.randn(8, 32, device=device)
        out, f = compiled(x)
        
        assert out.shape == (8, 16)
        assert not has_err(f)
    
    def test_fix_compiles(self) -> None:
        """fix() works with inductor backend."""
        device = get_device()
        
        @tracked
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                out = self.linear(x)
                f = flag_nan(out, self.linear, f)
                out, f = fix(out, f, self.linear, fallback=0.0)
                return out, f
        
        model = Model().to(device)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        # Input with NaN
        x = torch.randn(8, 32, device=device)
        x[0] = float('nan')
        out, f = compiled(x)
        
        # Output should not contain NaN after fix
        assert not torch.isnan(out).any()
        assert has_err(f)  # Error was recorded


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODEL WITH INDUCTOR
# ═══════════════════════════════════════════════════════════════════════════════

@skip_if_no_inductor
class TestInductorFullModel:
    """Test complete models with inductor backend."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_encoder_decoder_compiles(self) -> None:
        """Full encoder-decoder model compiles with inductor."""
        device = get_device()
        
        @tracked
        class EncoderDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(64, 32)
                self.decoder = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                
                h = self.encoder(x)
                f = flag_nan(h, self.encoder, f)
                f = flag_inf(h, self.encoder, f)
                
                out = self.decoder(h)
                f = flag_nan(out, self.decoder, f)
                f = flag_inf(out, self.decoder, f)
                
                return out, f
        
        model = EncoderDecoder().to(device)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        x = torch.randn(16, 64, device=device)
        out, f = compiled(x)
        
        assert out.shape == (16, 16)
        assert not has_err(f)
    
    def test_model_with_partition_padded(self) -> None:
        """Model using err.take_ok_p/err.take_err_p compiles with inductor.

        Note: take_ok()/take_err()/partition() use dynamic shapes (boolean indexing) which
        don't work with fullgraph=True. Use take_ok_p()/take_err_p() instead
        for static shapes inside compiled code.
        """
        device = get_device()
        
        @tracked
        class ModelWithPartition(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                
                out = self.layer(x)
                f = flag_nan(out, self.layer, f)
                
                # Use padded variants for static shapes (torch.compile safe)
                ok_out = err.take_ok_p(f, out, fill=0.0)
                err_out = err.take_err_p(f, out, fill=0.0)
                
                return out, f, ok_out, err_out
        
        model = ModelWithPartition().to(device)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        x = torch.randn(8, 32, device=device)
        out, f, ok_out, err_out = compiled(x)
        
        assert out.shape == (8, 16)
        assert ok_out.shape == (8, 16)  # Same shape as input (static)
        assert err_out.shape == (8, 16)  # Same shape as input (static)
    
    def test_model_with_recovery(self) -> None:
        """Model with error recovery compiles with inductor."""
        device = get_device()
        
        @tracked
        class ModelWithRecovery(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                
                out = self.layer(x)
                f = flag_nan(out, self.layer, f)
                f = flag_inf(out, self.layer, f)
                
                # Replace bad samples with zeros
                bad = err.is_err(f)
                out = torch.where(bad.unsqueeze(-1), torch.zeros_like(out), out)
                
                return out, f
        
        model = ModelWithRecovery().to(device)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        # Input with NaN
        x = torch.randn(8, 32, device=device)
        x[0] = float('nan')
        out, f = compiled(x)
        
        assert has_err(f)
        # Bad samples should be zeroed
        assert not torch.isnan(out).any()


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINATORS WITH INDUCTOR
# ═══════════════════════════════════════════════════════════════════════════════

@skip_if_no_inductor
class TestInductorCombinators:
    """Test combinators with inductor backend."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_map_ok_compiles(self) -> None:
        """err.map_ok works with inductor backend."""
        device = get_device()
        loc_id = ErrorLocation.register("layer")
        
        @torch.compile(backend="inductor", fullgraph=True)
        def apply_map_ok(x):
            f = err.new(x)
            mask = x[:, 0] > 0
            f = err.push(f, ErrorCode.NAN, loc_id, where=mask)
            
            # Only normalize OK samples
            out = err.map_ok(f, x, lambda z: z / z.norm(dim=-1, keepdim=True))
            return out, f
        
        x = torch.randn(8, 32, device=device)
        out, f = apply_map_ok(x)
        
        assert out.shape == x.shape
    
    def test_map_err_compiles(self) -> None:
        """err.map_err works with inductor backend."""
        device = get_device()
        loc_id = ErrorLocation.register("layer")
        
        @torch.compile(backend="inductor", fullgraph=True)
        def apply_map_err(x):
            f = err.new(x)
            mask = x[:, 0] > 0
            f = err.push(f, ErrorCode.NAN, loc_id, where=mask)
            
            # Zero out error samples
            out = err.map_err(f, x, lambda z: torch.zeros_like(z))
            return out, f
        
        x = torch.randn(8, 32, device=device)
        out, f = apply_map_err(x)
        
        assert out.shape == x.shape


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL FLOW DSL WITH INDUCTOR
# ═══════════════════════════════════════════════════════════════════════════════

@skip_if_no_inductor
class TestInductorControlFlow:
    """Test control flow DSL with inductor backend."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_if_else_compiles(self) -> None:
        """IF/ELSE DSL works with inductor backend."""
        device = get_device()
        
        @tracked
        class ModelWithIF(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                
                out = self.layer(x)
                f = flag_nan(out, self.layer, f)
                
                # Use IF/ELSE for recovery
                out, f = (
                    IF(IS(ErrorCode.NAN, f), lambda: fix(out, f, self.layer))
                    .ELSE(lambda: (out, f))
                )
                
                return out, f
        
        model = ModelWithIF().to(device).eval()
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        x = torch.randn(8, 32, device=device)
        # Use inference_mode to avoid AOTAutograd backward tracing issues
        # with torch.cond + mixed float/int outputs
        with torch.inference_mode():
            out, f = compiled(x)
        
        assert out.shape == (8, 16)
    
    def test_compound_predicates_compile(self) -> None:
        """OR/AND predicates work with inductor backend."""
        device = get_device()
        
        @tracked
        class ModelWithOR(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                
                out = self.layer(x)
                f = flag_nan(out, self.layer, f)
                f = flag_inf(out, self.layer, f)
                
                # Recovery if NAN or INF
                bad_numeric = OR(IS(ErrorCode.NAN, f), IS(ErrorCode.INF, f))
                out, f = (
                    IF(bad_numeric, lambda: fix(out, f, self.layer))
                    .ELSE(lambda: (out, f))
                )
                
                return out, f
        
        model = ModelWithOR().to(device).eval()
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        x = torch.randn(8, 32, device=device)
        # Use inference_mode to avoid AOTAutograd backward tracing issues
        # with torch.cond + mixed float/int outputs
        with torch.inference_mode():
            out, f = compiled(x)
        
        assert out.shape == (8, 16)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@skip_if_no_inductor
class TestInductorPerformance:
    """Basic performance sanity checks with inductor."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_large_batch_cuda(self) -> None:
        """Large batch compiles and runs with inductor on CUDA."""
        device = torch.device("cuda")
        
        @tracked
        class LargeBatchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(256, 128)
                self.layer2 = nn.Linear(128, 64)
            
            def forward(self, x):
                f = err.new(x)
                
                h = self.layer1(x)
                f = flag_nan(h, self.layer1, f)
                
                out = self.layer2(h)
                f = flag_nan(out, self.layer2, f)
                
                return out, f
        
        model = LargeBatchModel().to(device)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        # Warm up
        x = torch.randn(1024, 256, device=device)
        _ = compiled(x)
        
        # Run with large batch
        x = torch.randn(8192, 256, device=device)
        out, f = compiled(x)
        
        assert out.shape == (8192, 64)
        assert f.shape[0] == 8192
        assert not has_err(f)
    
    def test_repeated_calls_stable(self) -> None:
        """Multiple calls to compiled model are stable."""
        device = get_device()
        
        @tracked
        class StableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 16)
            
            def forward(self, x):
                f = err.new(x)
                out = self.layer(x)
                f = flag_nan(out, self.layer, f)
                return out, f
        
        model = StableModel().to(device)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        
        # Multiple calls should all work
        for _ in range(10):
            x = torch.randn(8, 32, device=device)
            out, f = compiled(x)
            assert out.shape == (8, 16)
            assert not has_err(f)

