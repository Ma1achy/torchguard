"""
Tests for Phase 3 optimizations.

Tests cover:
- Vectorized unpacking (3.1)
- Memory layout optimization (3.2)
- Performance benchmarks

Run with:
    pytest torchguard/tests/test_phase3_optimizations.py -v
"""
import time

import pytest
import torch

import torchguard as tg
from torchguard import (
    ErrorCode,
    ErrorConfig,
    ErrorLocation,
    ErrorFlags,
    UnpackedError,
    Severity,
    err,
    push,
)


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED UNPACKING TESTS (Phase 3.1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorizedUnpacking:
    """Tests for vectorized unpacking optimization."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_unpack_all_matches_sequential(self) -> None:
        """Verify vectorized unpack produces same results as sequential."""
        # Create flags with errors
        x = torch.randn(10, 5)
        flags = err.new_t(10, x.device)
        
        # Add some errors
        flags = push(flags, ErrorCode.NAN, "loc1", where=torch.tensor([True, False, True, False, False, True, False, False, False, True]))
        flags = push(flags, ErrorCode.INF, "loc2", where=torch.tensor([False, True, False, True, False, False, True, False, False, False]))
        
        # Compare vectorized vs sequential
        vectorized_result = ErrorFlags.unpack_all_vectorized(flags)
        sequential_result = [ErrorFlags.unpack(flags, i) for i in range(flags.shape[0])]
        
        assert len(vectorized_result) == len(sequential_result)
        
        for i, (vec_errors, seq_errors) in enumerate(zip(vectorized_result, sequential_result)):
            assert len(vec_errors) == len(seq_errors), f"Sample {i}: different error count"
            
            # Sort by code for stable comparison
            vec_sorted = sorted(vec_errors, key=lambda e: (e.code, e.location))
            seq_sorted = sorted(seq_errors, key=lambda e: (e.code, e.location))
            
            for v, s in zip(vec_sorted, seq_sorted):
                assert v.severity == s.severity
                assert v.code == s.code
                assert v.location == s.location
    
    def test_unpack_all_empty(self) -> None:
        """Test vectorized unpack with no errors."""
        flags = err.new_t(5, torch.device('cpu'))
        
        result = ErrorFlags.unpack_all(flags)
        
        assert len(result) == 5
        assert all(len(errors) == 0 for errors in result)
    
    def test_unpack_all_single_sample(self) -> None:
        """Test vectorized unpack with single sample."""
        flags = err.new_t(1, torch.device('cpu'))
        flags = push(flags, ErrorCode.NAN, "test_loc", where=torch.tensor([True]))
        
        result = ErrorFlags.unpack_all(flags)
        
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0].code == ErrorCode.NAN
    
    def test_unpack_all_float32_dtype(self) -> None:
        """Test vectorized unpack with float32 flags."""
        tg.CONFIG.flag_dtype = torch.float32
        try:
            flags = err.new_t(5, torch.device('cpu'))
            flags = push(flags, ErrorCode.NAN, "loc1", where=torch.tensor([True, False, True, False, False]))
            
            result = ErrorFlags.unpack_all(flags)
            
            assert len(result) == 5
            assert len(result[0]) == 1  # Sample 0 has NAN
            assert len(result[1]) == 0  # Sample 1 has no errors
            assert len(result[2]) == 1  # Sample 2 has NAN
        finally:
            tg.CONFIG.flag_dtype = torch.int64
    
    def test_unpack_all_large_batch(self) -> None:
        """Test vectorized unpack with large batch."""
        n = 1000
        flags = err.new_t(n, torch.device('cpu'))
        
        # Add errors to every 10th sample
        where = torch.zeros(n, dtype=torch.bool)
        where[::10] = True
        flags = push(flags, ErrorCode.NAN, "loc", where=where)
        
        result = ErrorFlags.unpack_all(flags)
        
        assert len(result) == n
        error_count = sum(len(errors) for errors in result)
        assert error_count == 100  # Every 10th sample has an error


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY LAYOUT OPTIMIZATION TESTS (Phase 3.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryLayoutOptimization:
    """Tests for memory layout optimization (transposed layout)."""
    
    def test_should_transpose_default(self) -> None:
        """Test default configuration doesn't transpose."""
        cfg = ErrorConfig()
        
        # Default: use_transposed_layout = False
        assert not cfg.should_transpose(100)
        assert not cfg.should_transpose(10000)
        assert not cfg.should_transpose(100000)
    
    def test_should_transpose_enabled(self) -> None:
        """Test transposed layout with explicit enable."""
        cfg = ErrorConfig(use_transposed_layout=True, transpose_threshold=10000)
        
        assert not cfg.should_transpose(100)
        assert not cfg.should_transpose(9999)
        assert cfg.should_transpose(10001)
        assert cfg.should_transpose(100000)
    
    def test_should_transpose_custom_threshold(self) -> None:
        """Test transposed layout with custom threshold."""
        cfg = ErrorConfig(use_transposed_layout=True, transpose_threshold=500)
        
        assert not cfg.should_transpose(100)
        assert not cfg.should_transpose(500)
        assert cfg.should_transpose(501)
        assert cfg.should_transpose(1000)
    
    def test_config_properties_with_transpose(self) -> None:
        """Test that other config properties still work with transpose options."""
        cfg = ErrorConfig(
            num_slots=32,
            flag_dtype=torch.float32,
            use_transposed_layout=True,
            transpose_threshold=5000
        )
        
        assert cfg.num_slots == 32
        assert cfg.flag_dtype == torch.float32
        assert cfg.slots_per_word == 2  # float32 has 2 slots per word
        assert cfg.num_words == 16  # 32 slots / 2 slots per word
        assert cfg.should_transpose(10000)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase3Performance:
    """Performance benchmarks for Phase 3 optimizations."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    @pytest.mark.parametrize("batch_size", [10, 100, 500])
    def test_vectorized_unpack_correctness(self, batch_size: int) -> None:
        """Verify vectorized unpack produces correct results at various batch sizes."""
        flags = err.new_t(batch_size, torch.device('cpu'))
        
        # Add errors at random positions
        where = torch.rand(batch_size) > 0.8
        flags = push(flags, ErrorCode.NAN, "bench_loc", where=where)
        
        # Get results from both methods
        vec_result = ErrorFlags.unpack_all_vectorized(flags)
        seq_result = [ErrorFlags.unpack(flags, i) for i in range(batch_size)]
        
        # Verify they match
        assert len(vec_result) == len(seq_result)
        for i in range(batch_size):
            assert len(vec_result[i]) == len(seq_result[i])
    
    def test_vectorized_unpack_speedup(self) -> None:
        """Benchmark: vectorized should be faster for large batches."""
        batch_size = 500
        flags = err.new_t(batch_size, torch.device('cpu'))
        
        # Add some errors
        where = torch.rand(batch_size) > 0.9
        flags = push(flags, ErrorCode.NAN, "bench", where=where)
        
        # Warmup
        for _ in range(3):
            _ = ErrorFlags.unpack_all_vectorized(flags)
            _ = [ErrorFlags.unpack(flags, i) for i in range(batch_size)]
        
        # Benchmark sequential
        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            _ = [ErrorFlags.unpack(flags, i) for i in range(batch_size)]
        sequential_time = time.perf_counter() - start
        
        # Benchmark vectorized
        start = time.perf_counter()
        for _ in range(iterations):
            _ = ErrorFlags.unpack_all_vectorized(flags)
        vectorized_time = time.perf_counter() - start
        
        print(f"\nBatch size {batch_size}:")
        print(f"  Sequential: {sequential_time:.3f}s")
        print(f"  Vectorized: {vectorized_time:.3f}s")
        print(f"  Speedup: {sequential_time/vectorized_time:.2f}x")
        
        # Vectorized should be at least comparable (not slower)
        # Note: for small batches, overhead may make it similar or slightly slower
        # but for large batches it should be faster
        assert vectorized_time < sequential_time * 2.0  # At least not 2x slower


# ═══════════════════════════════════════════════════════════════════════════════
# UNPACKED ERROR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnpackedError:
    """Tests for UnpackedError dataclass."""
    
    def test_unpacked_error_repr_ok(self) -> None:
        """Test UnpackedError repr for OK."""
        err = UnpackedError(
            severity=Severity.OK,
            code=ErrorCode.OK,
            location=0,
            severity_name="OK",
            code_name="OK",
            location_name="Unknown"
        )
        assert "OK" in repr(err)
    
    def test_unpacked_error_repr_error(self) -> None:
        """Test UnpackedError repr for error."""
        err = UnpackedError(
            severity=Severity.ERROR,
            code=ErrorCode.NAN,
            location=1,
            severity_name="ERROR",
            code_name="NAN",
            location_name="test_loc"
        )
        result = repr(err)
        assert "ERROR" in result
        assert "NAN" in result
        assert "test_loc" in result
