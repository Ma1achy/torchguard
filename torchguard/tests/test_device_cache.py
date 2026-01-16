"""Tests for device cache optimization."""

import threading
import sys
from pathlib import Path

import torch
import pytest

# Add parent to path for test imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from src.core.device_cache import DeviceCache, get_device_cache


class TestDeviceCacheSingleton:
    """Test singleton behavior."""
    
    def test_cache_singleton(self):
        """Test that get_device_cache returns same instance."""
        cache1 = get_device_cache()
        cache2 = get_device_cache()
        assert cache1 is cache2
    
    def test_cache_is_device_cache(self):
        """Test that singleton is DeviceCache instance."""
        cache = get_device_cache()
        assert isinstance(cache, DeviceCache)


class TestSlotShiftsCaching:
    """Test slot shifts caching."""
    
    def test_slot_shifts_cached(self):
        """Test slot shifts are cached and reused."""
        cache = DeviceCache()  # Use fresh cache for testing
        
        device = torch.device('cpu')
        shifts1 = cache.get_slot_shifts(device, torch.int64, 4, 16)
        initial_size = cache.size()
        
        shifts2 = cache.get_slot_shifts(device, torch.int64, 4, 16)
        
        # Same object returned
        assert shifts1 is shifts2
        # No new cache entry
        assert cache.size() == initial_size
    
    def test_slot_shifts_correct_values(self):
        """Test slot shifts have correct values."""
        cache = DeviceCache()
        
        shifts = cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
        
        expected = torch.tensor([0, 16, 32, 48], dtype=torch.int64)
        assert torch.equal(shifts, expected)
    
    def test_slot_shifts_different_params(self):
        """Test different params create different entries."""
        cache = DeviceCache()
        
        shifts1 = cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
        shifts2 = cache.get_slot_shifts(torch.device('cpu'), torch.int64, 2, 32)
        
        assert shifts1 is not shifts2
        assert cache.size() == 2
    
    def test_slot_shifts_different_dtype(self):
        """Test different dtypes create different entries."""
        cache = DeviceCache()
        
        shifts_i64 = cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
        shifts_i32 = cache.get_slot_shifts(torch.device('cpu'), torch.int32, 4, 16)
        
        assert shifts_i64 is not shifts_i32
        assert shifts_i64.dtype == torch.int64
        assert shifts_i32.dtype == torch.int32


class TestSlotIndicesCaching:
    """Test slot indices caching."""
    
    def test_slot_indices_cached(self):
        """Test slot indices are cached."""
        cache = DeviceCache()
        
        indices1 = cache.get_slot_indices(torch.device('cpu'), 16)
        indices2 = cache.get_slot_indices(torch.device('cpu'), 16)
        
        assert indices1 is indices2
    
    def test_slot_indices_correct_values(self):
        """Test slot indices have correct values."""
        cache = DeviceCache()
        
        indices = cache.get_slot_indices(torch.device('cpu'), 8)
        
        expected = torch.arange(8)
        assert torch.equal(indices, expected)


class TestPositionArrayCaching:
    """Test position array caching."""
    
    def test_position_array_cached(self):
        """Test position array is cached."""
        cache = DeviceCache()
        
        pos1 = cache.get_position_array(torch.device('cpu'), torch.float32, 64)
        pos2 = cache.get_position_array(torch.device('cpu'), torch.float32, 64)
        
        assert pos1 is pos2
    
    def test_position_array_correct_values(self):
        """Test position array has correct values."""
        cache = DeviceCache()
        
        positions = cache.get_position_array(torch.device('cpu'), torch.float32, 4)
        
        expected = torch.tensor([0., 1., 2., 3.], dtype=torch.float32)
        assert torch.equal(positions, expected)


class TestConstantCaching:
    """Test constant tensor caching."""
    
    def test_neg_inf_cached(self):
        """Test negative infinity is cached."""
        cache = DeviceCache()
        
        neg_inf1 = cache.get_constant(-float('inf'), torch.device('cpu'), torch.float32)
        neg_inf2 = cache.get_constant(-float('inf'), torch.device('cpu'), torch.float32)
        
        assert neg_inf1 is neg_inf2
        assert neg_inf1.item() == float('-inf')
    
    def test_zero_cached(self):
        """Test zero constant is cached."""
        cache = DeviceCache()
        
        zero1 = cache.get_constant(0.0, torch.device('cpu'), torch.float32)
        zero2 = cache.get_constant(0.0, torch.device('cpu'), torch.float32)
        
        assert zero1 is zero2
        assert zero1.item() == 0.0
    
    def test_different_values_different_entries(self):
        """Test different values create different entries."""
        cache = DeviceCache()
        
        neg_inf = cache.get_constant(-float('inf'), torch.device('cpu'), torch.float32)
        zero = cache.get_constant(0.0, torch.device('cpu'), torch.float32)
        one = cache.get_constant(1.0, torch.device('cpu'), torch.float32)
        
        assert neg_inf is not zero
        assert zero is not one
        assert cache.size() == 3


class TestCacheManagement:
    """Test cache management functions."""
    
    def test_clear_cache(self):
        """Test cache clearing."""
        cache = DeviceCache()
        
        # Add some entries
        cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
        cache.get_constant(0.0, torch.device('cpu'), torch.float32)
        assert cache.size() > 0
        
        # Clear
        cache.clear()
        assert cache.size() == 0
    
    def test_size(self):
        """Test size reporting."""
        cache = DeviceCache()
        
        assert cache.size() == 0
        
        cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
        assert cache.size() == 1
        
        cache.get_constant(0.0, torch.device('cpu'), torch.float32)
        assert cache.size() == 2
    
    def test_memory_usage(self):
        """Test memory usage estimation."""
        cache = DeviceCache()
        
        assert cache.memory_usage_bytes() == 0
        
        # Add a tensor: 4 int64 = 32 bytes
        cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
        assert cache.memory_usage_bytes() == 32
        
        # Add another: 8 float32 = 32 bytes
        cache.get_position_array(torch.device('cpu'), torch.float32, 8)
        assert cache.memory_usage_bytes() == 64


class TestThreadSafety:
    """Test thread-safe behavior."""
    
    @pytest.mark.parametrize("num_threads", [2, 4, 8])
    def test_concurrent_access(self, num_threads):
        """Test concurrent access returns correct values."""
        cache = DeviceCache()
        results = []
        
        def worker():
            shifts = cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
            results.append(shifts)
        
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All threads should get tensors with identical values
        # (may be different objects due to lock-free design, but values are the same)
        expected = torch.tensor([0, 16, 32, 48], dtype=torch.int64)
        assert all(torch.equal(r, expected) for r in results)
    
    def test_concurrent_different_keys(self):
        """Test concurrent access with different keys."""
        cache = DeviceCache()
        results = {}
        
        def worker(key_id):
            shifts = cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4 + key_id, 16)
            results[key_id] = shifts
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have 4 different entries
        assert cache.size() == 4
        # Each should have different shape
        assert results[0].shape[0] == 4
        assert results[1].shape[0] == 5
        assert results[2].shape[0] == 6
        assert results[3].shape[0] == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDADevice:
    """Test CUDA device handling."""
    
    def test_cuda_caching(self):
        """Test caching works on CUDA."""
        cache = DeviceCache()
        
        shifts1 = cache.get_slot_shifts(torch.device('cuda'), torch.int64, 4, 16)
        shifts2 = cache.get_slot_shifts(torch.device('cuda'), torch.int64, 4, 16)
        
        assert shifts1 is shifts2
        assert shifts1.device.type == 'cuda'
    
    def test_cpu_cuda_separate(self):
        """Test CPU and CUDA have separate cache entries."""
        cache = DeviceCache()
        
        cpu_shifts = cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
        cuda_shifts = cache.get_slot_shifts(torch.device('cuda'), torch.int64, 4, 16)
        
        assert cpu_shifts.device.type == 'cpu'
        assert cuda_shifts.device.type == 'cuda'
        assert cpu_shifts is not cuda_shifts
        assert cache.size() == 2


class TestGlobalCacheUsage:
    """Test global cache is used correctly."""
    
    def test_global_cache_persistence(self):
        """Test global cache persists across calls."""
        cache = get_device_cache()
        initial_size = cache.size()
        
        # Add entry
        cache.get_slot_shifts(torch.device('cpu'), torch.int64, 4, 16)
        
        # Get cache again
        cache2 = get_device_cache()
        
        # Should be same cache with entry
        assert cache2.size() >= initial_size
