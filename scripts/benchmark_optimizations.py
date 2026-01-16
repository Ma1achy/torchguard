#!/usr/bin/env python3
"""
Benchmark script for TorchGuard optimizations.

Measures performance of key operations with and without caching.
Run from the torchguard directory:
    python scripts/benchmark_optimizations.py

Requirements:
    - torch
    - torchguard (local)
"""

import sys
import time
from pathlib import Path

# Add torchguard to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


def benchmark_operation(name: str, fn, warmup: int = 5, iterations: int = 100):
    """Run a benchmark and return mean time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1e6  # Convert to microseconds
        times.append(elapsed)
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    print(f"  {name}: {mean_time:.2f} ± {std_time:.2f} μs")
    return mean_time


def benchmark_error_ops():
    """Benchmark basic error operations."""
    from torchguard import err, push, find, ErrorCode
    from torchguard.src.core.device_cache import get_device_cache
    
    print("\n" + "=" * 60)
    print("Error Operations Benchmark")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = [32, 128, 512, 2048]
    
    for N in batch_sizes:
        print(f"\nBatch size: {N}")
        
        # Benchmark new_t
        def bench_new_t():
            return err.new_t(N, device)
        benchmark_operation("err.new_t", bench_new_t)
        
        # Benchmark push
        flags = err.new_t(N, device)
        mask = torch.rand(N, device=device) > 0.5
        
        def bench_push():
            return push(flags, ErrorCode.NAN, None, where=mask)
        benchmark_operation("push (with mask)", bench_push)
        
        # Benchmark find
        flags_with_errors = push(flags, ErrorCode.NAN, None, where=mask)
        
        def bench_find():
            return find(ErrorCode.NAN, flags_with_errors)
        benchmark_operation("find", bench_find)
    
    # Print cache stats
    cache = get_device_cache()
    print(f"\nDeviceCache entries: {cache.size()}")


def benchmark_compiled_model():
    """Benchmark a compiled model with error handling."""
    from torchguard import tracked, err, push, ErrorCode
    
    print("\n" + "=" * 60)
    print("Compiled Model Benchmark")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @tracked
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 32)
        
        def forward(self, x):
            flags = err.new_t(x.shape[0], x.device)
            y = self.linear(x)
            nan_mask = torch.isnan(y).any(dim=-1)
            flags = push(flags, ErrorCode.NAN, self, where=nan_mask)
            return y, flags
    
    model = SimpleModel().to(device)
    x = torch.randn(128, 64, device=device)
    
    # Eager mode
    print("\nEager mode:")
    def bench_eager():
        return model(x)
    benchmark_operation("forward pass", bench_eager)
    
    # Compiled mode
    print("\nCompiled mode (inductor):")
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
    
    # Extra warmup for compilation
    for _ in range(3):
        compiled_model(x)
    
    def bench_compiled():
        return compiled_model(x)
    benchmark_operation("forward pass", bench_compiled)


def benchmark_flag_detection():
    """Benchmark NaN/Inf detection functions."""
    from torchguard import flag_nan, flag_inf, flag_nan_and_inf, err
    
    print("\n" + "=" * 60)
    print("Flag Detection Benchmark")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = [128, 512, 2048]
    
    for N in batch_sizes:
        print(f"\nBatch size: {N}, tensor shape: ({N}, 256)")
        
        # Create test tensor with some NaN/Inf values
        x = torch.randn(N, 256, device=device)
        x[::10, ::10] = float('nan')  # Add some NaNs
        x[::15, ::15] = float('inf')  # Add some Infs
        
        # Benchmark separate calls
        def bench_separate():
            flags = flag_nan(x, None)
            flags = flag_inf(x, None, flags=flags)
            return flags
        benchmark_operation("flag_nan + flag_inf (separate)", bench_separate)
        
        # Benchmark fused call
        def bench_fused():
            return flag_nan_and_inf(x, None)
        benchmark_operation("flag_nan_and_inf (fused)", bench_fused)


def benchmark_vectorized_unpack():
    """Benchmark vectorized unpacking (Phase 3)."""
    from torchguard import err, push, ErrorCode, ErrorFlags
    
    print("\n" + "=" * 60)
    print("Vectorized Unpacking Benchmark (Phase 3)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = [100, 500, 1000]
    
    for N in batch_sizes:
        print(f"\nBatch size: {N}")
        
        # Create flags with some errors
        flags = err.new_t(N, device)
        where = torch.rand(N, device=device) > 0.9
        flags = push(flags, ErrorCode.NAN, None, where=where)
        
        # Move to CPU for unpacking
        flags_cpu = flags.cpu()
        
        # Benchmark sequential
        def bench_sequential():
            return [ErrorFlags.unpack(flags_cpu, i) for i in range(N)]
        benchmark_operation("sequential unpack_all", bench_sequential, warmup=2, iterations=10)
        
        # Benchmark vectorized
        def bench_vectorized():
            return ErrorFlags.unpack_all_vectorized(flags_cpu)
        benchmark_operation("vectorized unpack_all", bench_vectorized, warmup=2, iterations=10)


def main():
    print("TorchGuard Optimization Benchmark")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        benchmark_error_ops()
    except Exception as e:
        print(f"Error in error_ops benchmark: {e}")
    
    try:
        benchmark_flag_detection()
    except Exception as e:
        print(f"Error in flag detection benchmark: {e}")
    
    try:
        benchmark_compiled_model()
    except Exception as e:
        print(f"Error in compiled model benchmark: {e}")
    
    try:
        benchmark_vectorized_unpack()
    except Exception as e:
        print(f"Error in vectorized unpack benchmark: {e}")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
