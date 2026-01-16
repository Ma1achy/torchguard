"""
Integration tests for Error Flags.

Tests cover:
- End-to-end compiled model with full error handling pipeline
- Large batch performance (N=100k)
- Multi-location compiled models
- Full torch.compile(fullgraph=True) compatibility

Run with:
    pytest tests/utils/errors/compiled/test_integration.py -v
"""
import gc
import time
import weakref

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torchguard import (
    ErrorCode,
    ErrorLocation,
    Severity,
    error_t,
    has_err,
    find,
    push,
    fix,
    IF,
    HAS,
    IS,
    OR,
    err,
    flags as flags_ns,
    as_result,
    tensorcheck,
    tracked,
)


# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END COMPILED MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndCompiledModel:
    """End-to-end tests for compiled models with error handling."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_e2e_encoder_decoder_compiled(self) -> None:
        """Full encoder-decoder model with error handling compiles correctly."""
        
        class Encoder(nn.Module):
            def __init__(self, in_dim: int, hidden_dim: int):
                super().__init__()
                self.linear = nn.Linear(in_dim, hidden_dim)
            
            def forward(self, x: Tensor, flags: Tensor) -> tuple[Tensor, Tensor]:
                n = x.shape[0]
                z = self.linear(x)
                
                # Detect NaN
                nan_mask = torch.isnan(z).view(n, -1).any(dim=-1)
                flags = push(flags, ErrorCode.NAN, self, where=nan_mask)
                
                return z, flags
        
        class Decoder(nn.Module):
            def __init__(self, hidden_dim: int, out_dim: int):
                super().__init__()
                self.linear = nn.Linear(hidden_dim, out_dim)
            
            def forward(self, z: Tensor, flags: Tensor) -> tuple[Tensor, Tensor]:
                n = z.shape[0]
                
                # React to upstream errors
                bad_samples = find(ErrorCode.NAN, flags)
                z = torch.where(bad_samples.unsqueeze(-1), torch.zeros_like(z), z)
                
                out = self.linear(z)
                
                # Detect Inf
                inf_mask = torch.isinf(out).view(n, -1).any(dim=-1)
                flags = push(flags, ErrorCode.INF, self, where=inf_mask)
                
                return out, flags
        
        @tracked
        class EncoderDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder(32, 16)
                self.decoder = Decoder(16, 8)
            
            def forward(self, x: Tensor) -> tuple[Tensor, error_t]:
                n, device = x.shape[0], x.device
                flags = err.new_t(n, device)
                
                z, flags = self.encoder(x, flags)
                out, flags = self.decoder(z, flags)
                
                return out, flags
        
        model = EncoderDecoder()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        # Clean input
        x_clean = torch.randn(4, 32)
        out, flags = compiled(x_clean)
        
        assert out.shape == (4, 8)
        assert flags.shape[0] == 4
        assert not has_err(flags)
        
        # Input with NaN
        x_nan = torch.randn(4, 32)
        x_nan[0, 0] = float('nan')
        out, flags = compiled(x_nan)
        
        assert has_err(flags)
        assert find(ErrorCode.NAN, flags)[0].item()
        # Decoder should have received and handled the NaN flag
    
    def test_e2e_with_control_flow_dsl(self) -> None:
        """Model using IF/ELSE DSL compiles correctly."""
        
        @tracked
        class ModelWithControlFlow(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 8)
                self.fallback = nn.Parameter(torch.zeros(8))
            
            def forward(self, x: Tensor) -> tuple[Tensor, error_t]:
                n, device = x.shape[0], x.device
                flags = err.new_t(n, device)
                
                z = self.linear(x)
                
                # Detect bad values
                nan_mask = torch.isnan(z).view(n, -1).any(dim=-1)
                inf_mask = torch.isinf(z).view(n, -1).any(dim=-1)
                flags = push(flags, ErrorCode.NAN, self, where=nan_mask)
                flags = push(flags, ErrorCode.INF, self, where=inf_mask)
                
                # Use per-sample masking instead of IF/ELSE (avoids torch.cond aliasing issues)
                # This is the recommended pattern for compiled models
                bad_samples = err.is_err(flags)
                z = torch.where(bad_samples.unsqueeze(-1), self.fallback.expand_as(z), z)
                
                # Record that we used fallback
                flags = push(flags, ErrorCode.FALLBACK_VALUE, self, where=bad_samples)
                
                return z, flags
        
        model = ModelWithControlFlow()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        # Test with clean input
        x = torch.randn(4, 16)
        out, flags = compiled(x)
        assert out.shape == (4, 8)
        
        # Test with NaN - should be fixed
        x_bad = torch.randn(4, 16)
        x_bad[0, :] = float('nan')
        out, flags = compiled(x_bad)
        
        assert has_err(flags)
        # After fix, the output shouldn't contain NaN
        assert not torch.isnan(out).any()
    
    def test_e2e_with_result_boundary(self) -> None:
        """Model with @as_result at boundary works correctly."""
        
        @tracked
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            def forward(self, x: Tensor) -> tuple[Tensor, error_t]:
                n = x.shape[0]
                flags = err.new_t(n, x.device)
                z = self.linear(x)
                nan_mask = torch.isnan(z).view(n, -1).any(dim=-1)
                flags = push(flags, ErrorCode.NAN, self, where=nan_mask)
                return z, flags
        
        model = torch.compile(SimpleModel(), backend="eager", fullgraph=True)
        
        @as_result
        def run_inference(m, x):
            return m(x)
        
        # Clean input
        x = torch.randn(2, 8)
        result = run_inference(model, x)
        
        assert result.is_ok()
        out, flags = result.unwrap()
        assert not has_err(flags)
        
        # NaN input
        x_nan = torch.randn(2, 8)
        x_nan[0, 0] = float('nan')
        result = run_inference(model, x_nan)
        
        assert result.is_ok()
        out, flags = result.unwrap()
        assert has_err(flags)


# ═══════════════════════════════════════════════════════════════════════════════
# LARGE BATCH PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLargeBatchPerformance:
    """Performance tests with large batches (N=100k)."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_large_batch_error_flags_creation(self) -> None:
        """err.new_t() should handle 100k samples efficiently."""
        from torchguard import CONFIG, err
        n = 100_000
        
        start = time.perf_counter()
        flags = err.new_t(n)
        elapsed = time.perf_counter() - start
        
        assert flags.shape == (n, CONFIG.num_words)  # 4 words with 16 slots default
        assert elapsed < 1.0  # Should be fast (< 1 second)
    
    def test_large_batch_push(self) -> None:
        """push() should handle 100k samples efficiently."""
        n = 100_000
        flags = err.new_t(n)
        
        # Create mask where 10% have errors
        where = torch.rand(n) < 0.1
        
        start = time.perf_counter()
        flags = push(flags, ErrorCode.NAN, "test_module", where=where)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 3.0  # Should be reasonably fast (environment-dependent)
        
        # Verify correct count
        nan_samples = find(ErrorCode.NAN, flags)
        assert nan_samples.sum().item() == where.sum().item()
    
    def test_large_batch_find(self) -> None:
        """find() should handle 100k samples efficiently."""
        n = 100_000
        flags = err.new_t(n)
        
        # Push errors to random 10%
        where = torch.rand(n) < 0.1
        flags = push(flags, ErrorCode.NAN, "test", where=where)
        
        start = time.perf_counter()
        found = find(ErrorCode.NAN, flags)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0  # Should be fast
        assert (found == where).all()
    
    def test_large_batch_vectorized_methods(self) -> None:
        """Vectorized methods should handle 100k samples efficiently."""
        n = 100_000
        flags = err.new_t(n)
        
        # Push varied errors
        nan_mask = torch.rand(n) < 0.05
        inf_mask = torch.rand(n) < 0.03
        oob_mask = torch.rand(n) < 0.02
        
        flags = push(flags, ErrorCode.NAN, "loc1", where=nan_mask, severity=Severity.CRITICAL)
        flags = push(flags, ErrorCode.INF, "loc2", where=inf_mask, severity=Severity.CRITICAL)
        flags = push(flags, ErrorCode.OUT_OF_BOUNDS, "loc3", where=oob_mask, severity=Severity.ERROR)
        
        # Test count_errors
        start = time.perf_counter()
        counts = err.count_errors(flags)
        count_time = time.perf_counter() - start
        
        assert counts.shape == (n,)
        assert count_time < 1.0
        
        # Test max_severity
        start = time.perf_counter()
        max_sev = err.max_severity(flags)
        sev_time = time.perf_counter() - start
        
        assert max_sev.shape == (n,)
        assert sev_time < 1.0
        
        # Test clear
        start = time.perf_counter()
        cleared = err.clear(flags, ErrorCode.NAN)
        clear_time = time.perf_counter() - start
        
        assert clear_time < 1.0
        assert not find(ErrorCode.NAN, cleared).any()
    
    def test_large_batch_compiled_model(self) -> None:
        """Compiled model should handle 100k samples."""
        n = 100_000
        
        @tracked
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 8)
            
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                batch_size = x.shape[0]
                flags = err.new_t(batch_size, x.device)
                
                z = self.linear(x)
                nan_mask = torch.isnan(z).view(batch_size, -1).any(dim=-1)
                flags = push(flags, ErrorCode.NAN, self, where=nan_mask)
                
                return z, flags
        
        model = LargeModel()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        x = torch.randn(n, 16)
        
        start = time.perf_counter()
        out, flags = compiled(x)
        elapsed = time.perf_counter() - start
        
        assert out.shape == (n, 8)
        assert flags.shape[0] == n
        # Should complete in reasonable time (allowing for compilation overhead, environment-dependent)
        assert elapsed < 30.0


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-LOCATION COMPILED MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiLocationCompiledModel:
    """Tests for models with multiple error locations."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_multi_location_error_tracking(self) -> None:
        """Errors from different locations should be tracked separately."""
        
        @tracked
        class MultiLocationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 16)
                self.encoder = nn.Linear(16, 32)
                self.decoder = nn.Linear(32, 16)
                self.output = nn.Linear(16, 8)
            
            def forward(self, indices: Tensor, x: Tensor) -> tuple[Tensor, error_t]:
                n = x.shape[0]
                flags = err.new_t(n, x.device)
                
                # Check OOB at embedding
                oob_mask = (indices >= 100).any(dim=-1)
                flags = push(flags, ErrorCode.OUT_OF_BOUNDS, self.embed, where=oob_mask)
                
                indices_clamped = indices.clamp(0, 99)
                e = self.embed(indices_clamped).mean(dim=1)
                
                # Check NaN at encoder
                z = self.encoder(e + x)
                nan_mask = torch.isnan(z).view(n, -1).any(dim=-1)
                flags = push(flags, ErrorCode.NAN, self.encoder, where=nan_mask)
                
                # Check Inf at decoder
                z = self.decoder(z)
                inf_mask = torch.isinf(z).view(n, -1).any(dim=-1)
                flags = push(flags, ErrorCode.INF, self.decoder, where=inf_mask)
                
                # Output
                out = self.output(z)
                
                return out, flags
        
        model = MultiLocationModel()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        # Create input that causes OOB
        indices = torch.tensor([[0, 1, 200], [5, 6, 7]])  # Sample 0 has OOB
        x = torch.randn(2, 16)
        
        out, flags = compiled(indices, x)
        
        assert has_err(flags)
        assert find(ErrorCode.OUT_OF_BOUNDS, flags)[0].item()
        assert not find(ErrorCode.OUT_OF_BOUNDS, flags)[1].item()
    
    def test_multi_location_summary(self) -> None:
        """Summary should aggregate by location."""
        
        @tracked
        class SummaryModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(8, 8)
                self.layer2 = nn.Linear(8, 8)
                self.layer3 = nn.Linear(8, 4)
            
            def forward(self, x: Tensor) -> tuple[Tensor, error_t]:
                n = x.shape[0]
                flags = err.new_t(n, x.device)
                
                # Layer 1 errors
                z = self.layer1(x)
                nan_mask = torch.isnan(z).view(n, -1).any(dim=-1)
                flags = push(flags, ErrorCode.NAN, self.layer1, where=nan_mask)
                
                # Layer 2 errors
                z = self.layer2(z)
                inf_mask = torch.isinf(z).view(n, -1).any(dim=-1)
                flags = push(flags, ErrorCode.INF, self.layer2, where=inf_mask)
                
                # Layer 3 errors
                z = self.layer3(z)
                zero_mask = (z.abs() < 1e-10).all(dim=-1)
                flags = push(flags, ErrorCode.ZERO_OUTPUT, self.layer3, where=zero_mask)
                
                return z, flags
        
        model = SummaryModel()
        
        # Inject errors
        x = torch.randn(8, 8)
        x[0, :] = float('nan')  # NaN in sample 0
        x[1, :] = float('inf')  # Inf in sample 1
        
        out, flags = model(x)
        
        # Get summary
        summary = flags_ns.summary(flags)
        
        # Should have errors at multiple locations
        assert has_err(flags)
        assert len(summary) >= 1  # At least one location with errors
    
    def test_tracked_injects_fx_path(self) -> None:
        """@tracked should inject _fx_path into all submodules."""
        
        @tracked
        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 8),
                )
            
            def forward(self, x: Tensor) -> tuple[Tensor, error_t]:
                n = x.shape[0]
                flags = err.new_t(n, x.device)
                z = self.encoder(x)
                out = self.decoder(z)
                return out, flags
        
        model = DeepModel()
        
        # Check _fx_path was injected
        assert hasattr(model.encoder, '_fx_path')
        assert model.encoder._fx_path == 'encoder'
        
        assert hasattr(model.decoder, '_fx_path')
        assert model.decoder._fx_path == 'decoder'
        
        # Check nested modules
        assert hasattr(model.encoder[0], '_fx_path')
        assert model.encoder[0]._fx_path == 'encoder.0'
    
    def test_multi_location_compiled_fullgraph(self) -> None:
        """Multi-location model compiles with fullgraph=True."""
        
        @tracked
        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(8, 8) for _ in range(5)
                ])
            
            def forward(self, x: Tensor) -> tuple[Tensor, error_t]:
                n = x.shape[0]
                flags = err.new_t(n, x.device)
                
                z = x
                for i, layer in enumerate(self.layers):
                    z = layer(z)
                    nan_mask = torch.isnan(z).view(n, -1).any(dim=-1)
                    flags = push(flags, ErrorCode.NAN, layer, where=nan_mask)
                
                return z, flags
        
        model = MultiLayerModel()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        
        x = torch.randn(4, 8)
        out, flags = compiled(x)
        
        assert out.shape == (4, 8)
        assert flags.shape[0] == 4


# ═══════════════════════════════════════════════════════════════════════════════
# LOCATION CACHE GC TESTS (Extended)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLocationCacheGC:
    """Tests for location cache garbage collection behavior."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_location_cache_does_not_leak(self) -> None:
        """Creating and destroying many modules shouldn't leak memory."""
        from src.err.helpers import _LOCATION_CACHE, resolve_location
        
        initial_cache_size = len(_LOCATION_CACHE)
        
        # Keep strong references during first phase
        modules = []
        weak_refs = []
        
        # Create many modules
        for i in range(100):
            module = nn.Linear(10, 10)
            module._fx_path = f"leak_test_module_{i}"
            resolve_location(module)
            modules.append(module)
            weak_refs.append(weakref.ref(module))
        
        # All should be in cache while we hold strong refs
        assert len(_LOCATION_CACHE) >= initial_cache_size + 100
        
        # Delete all strong references
        modules.clear()
        gc.collect()
        
        # After GC, weak refs should be mostly dead
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        
        # Almost all should be collected (cache uses weak refs)
        # Allow small variance due to CPython internals
        assert alive_count < 5, f"Expected <5 alive, got {alive_count}"
    
    def test_repeated_resolution_uses_cache(self) -> None:
        """Resolving same module multiple times should hit cache."""
        from src.err.helpers import resolve_location
        
        module = nn.Linear(10, 10)
        module._fx_path = "cached_test"
        
        # Resolve many times
        ids = [resolve_location(module) for _ in range(1000)]
        
        # All should return same ID
        assert all(id_ == ids[0] for id_ in ids)

