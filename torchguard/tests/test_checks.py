"""
Tests for src/utils/errors/compiled/checks.py.

Tests cover:
- has_err() - Python bool return
- find() - vectorized error code search
- push() - conditional error push with where parameter
- fix() - bad value replacement
- resolve_location() - module to location ID resolution
- flag_nan/flag_inf/flag_oob_indices() - flag helpers (error recording)

Run with:
    pytest tests/utils/errors/compiled/test_checks.py -v
"""
import gc
import warnings
import weakref

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torchguard import (
    CONFIG,
    ErrorCode,
    ErrorConfig,
    ErrorLocation,
    Severity,
    error_t,
    err,
    flags as flags_ns,
    # Helper functions
    find,
    fix,
    flag_inf,
    flag_nan,
    flag_oob_indices,
    has_err,
    push,
)
# Internal helpers for test cleanup (relative import since we're inside the package)
from ..src.err.helpers import (
    clear_location_cache,
    clear_warn_cache,
    resolve_location,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HAS_ERR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHasErr:
    """Tests for has_err() function."""
    
    def test_has_err_returns_bool(self) -> None:
        """
        Verify has_err returns Python bool.
        
        Expected: Returns False (bool) for clean flags.
        """
        flags = err.new_t(5)
        result = has_err(flags)
        assert isinstance(result, bool)
        assert result is False
    
    def test_has_err_true_when_error(self) -> None:
        """
        Verify has_err detects errors in batch.
        
        Expected: Returns True when any sample has error.
        """
        flags = err.new_t(5)
        code = torch.tensor([ErrorCode.OK, ErrorCode.NAN, ErrorCode.OK, ErrorCode.OK, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        result = has_err(flags)
        assert result is True
    
    def test_has_err_false_when_all_ok(self) -> None:
        """
        Verify has_err returns False for clean batch.
        
        Expected: Returns False when all samples are OK.
        """
        flags = err.new_t(10)
        result = has_err(flags)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# FIND TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFind:
    """Tests for find() function."""
    
    def test_find_returns_mask(self) -> None:
        """
        Verify find returns boolean mask.
        
        Expected: Returns (N,) bool tensor, all False for clean flags.
        """
        flags = err.new_t(5)
        result = find(ErrorCode.NAN, flags)
        
        assert result.dtype == torch.bool
        assert result.shape == (5,)
        assert result.all() == False
    
    def test_find_finds_specific_code(self) -> None:
        """
        Verify find identifies samples with specific error code.
        
        Expected: Mask is True only for samples with target code.
        """
        flags = err.new_t(5)
        
        code = torch.tensor([ErrorCode.OK, ErrorCode.NAN, ErrorCode.OK, ErrorCode.NAN, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        result = find(ErrorCode.NAN, flags)
        
        expected = torch.tensor([False, True, False, True, False])
        assert result.tolist() == expected.tolist()
    
    def test_find_distinguishes_codes(self) -> None:
        """
        Verify find distinguishes between different error codes.
        
        Expected: Different masks for NAN vs INF.
        """
        flags = err.new_t(4)
        
        code1 = torch.tensor([ErrorCode.NAN, ErrorCode.OK, ErrorCode.NAN, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.CRITICAL)
        
        code2 = torch.tensor([ErrorCode.OK, ErrorCode.INF, ErrorCode.INF, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code2, location=2, severity=Severity.CRITICAL)
        
        nan_mask = find(ErrorCode.NAN, flags)
        inf_mask = find(ErrorCode.INF, flags)
        
        assert nan_mask.tolist() == [True, False, True, False]
        assert inf_mask.tolist() == [False, True, True, False]
    
    def test_find_vectorized_partial_slots(self) -> None:
        """
        Verify find handles configs with partial slot usage.
        
        Expected: All pushed codes found even with partial slot config.
        """
        config = ErrorConfig(num_slots=3)
        flags = err.new_t(2, config=config)
        
        code1 = torch.tensor([ErrorCode.NAN, ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.CRITICAL, config=config)
        
        code2 = torch.tensor([ErrorCode.INF, ErrorCode.INF], dtype=torch.int64)
        flags = err.push(flags, code2, location=2, severity=Severity.CRITICAL, config=config)
        
        code3 = torch.tensor([ErrorCode.OVERFLOW, ErrorCode.OVERFLOW], dtype=torch.int64)
        flags = err.push(flags, code3, location=3, severity=Severity.ERROR, config=config)
        
        assert find(ErrorCode.NAN, flags, config).all()
        assert find(ErrorCode.INF, flags, config).all()
        assert find(ErrorCode.OVERFLOW, flags, config).all()


# ═══════════════════════════════════════════════════════════════════════════════
# PUSH TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPush:
    """Tests for push() function."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        clear_location_cache()
    
    def test_push_with_where_parameter(self) -> None:
        """
        Verify push only adds errors where mask is True.
        
        Expected: Errors only at masked positions [0, 2, 4].
        """
        flags = err.new_t(5)
        where = torch.tensor([True, False, True, False, True])
        
        flags = push(flags, ErrorCode.NAN, "test_module", where=where)
        
        has_error = err.is_err(flags)
        assert has_error.tolist() == [True, False, True, False, True]
    
    def test_push_without_where(self) -> None:
        """
        Verify push without where pushes to all samples.
        
        Expected: All samples have error.
        """
        flags = err.new_t(3)
        flags = push(flags, ErrorCode.NAN, "test_module")
        
        has_error = err.is_err(flags)
        assert has_error.all()
    
    def test_push_batch_size_validation(self) -> None:
        """
        Verify push raises error on batch size mismatch.
        
        Expected: RuntimeError from torch.where due to shape mismatch.
        """
        flags = err.new_t(5)
        where = torch.tensor([True, False, True])
        
        # torch.where raises RuntimeError on shape mismatch
        with pytest.raises(RuntimeError, match="size of tensor"):
            push(flags, ErrorCode.NAN, "test", where=where)
    
    def test_push_auto_severity(self) -> None:
        """
        Verify push auto-infers severity from code.
        
        Expected: NAN gets CRITICAL severity.
        """
        flags = err.new_t(1)
        flags = push(flags, ErrorCode.NAN, "test")
        
        errors = flags_ns.unpack(flags, 0)
        assert len(errors) == 1
        assert errors[0].severity == Severity.CRITICAL
    
    def test_push_explicit_severity(self) -> None:
        """
        Verify push uses explicit severity when provided.
        
        Expected: Uses WARN instead of default CRITICAL.
        """
        flags = err.new_t(1)
        flags = push(flags, ErrorCode.NAN, "test", severity=Severity.WARN)
        
        errors = flags_ns.unpack(flags, 0)
        assert len(errors) == 1
        assert errors[0].severity == Severity.WARN
    
    def test_push_with_string_location(self) -> None:
        """
        Verify push accepts string for location.
        
        Expected: Location name is registered string.
        """
        flags = err.new_t(1)
        flags = push(flags, ErrorCode.NAN, "encoder.layer0.attn")
        
        errors = flags_ns.unpack(flags, 0)
        assert len(errors) == 1
        assert errors[0].location_name == "encoder.layer0.attn"
    
    def test_push_with_int_location(self) -> None:
        """
        Verify push accepts int for location passthrough.
        
        Expected: Location is exactly 42.
        """
        flags = err.new_t(1)
        flags = push(flags, ErrorCode.NAN, 42)
        
        errors = flags_ns.unpack(flags, 0)
        assert len(errors) == 1
        assert errors[0].location == 42
    
    def test_push_with_none_location(self) -> None:
        """
        Verify push with None uses UNKNOWN location.
        
        Expected: Location is ErrorLocation.UNKNOWN (0).
        """
        flags = err.new_t(1)
        flags = push(flags, ErrorCode.NAN, None)
        
        errors = flags_ns.unpack(flags, 0)
        assert len(errors) == 1
        assert errors[0].location == ErrorLocation.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════════
# FIX TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix:
    """Tests for fix() function."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        clear_location_cache()
    
    def test_fix_multidimensional(self) -> None:
        """
        Verify fix handles multidimensional tensors.
        
        Expected: Only sample 1 replaced with fallback.
        """
        tensor = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ])
        
        flags = err.new_t(3)
        code = torch.tensor([ErrorCode.OK, ErrorCode.NAN, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        cleaned, updated_flags = fix(tensor, flags, "test", fallback=0.0)
        
        assert cleaned[0].tolist() == [1.0, 2.0, 3.0, 4.0]
        assert cleaned[1].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert cleaned[2].tolist() == [9.0, 10.0, 11.0, 12.0]
    
    def test_fix_preserves_original_errors(self) -> None:
        """
        Verify fix preserves original errors and adds FALLBACK_VALUE.
        
        Expected: Sample 0 has both NAN and FALLBACK_VALUE.
        """
        flags = err.new_t(2)
        code = torch.tensor([ErrorCode.NAN, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        tensor = torch.tensor([[1.0], [2.0]])
        cleaned, updated_flags = fix(tensor, flags, "test", fallback=-1.0)
        
        errors_0 = flags_ns.unpack(updated_flags, 0)
        codes = [e.code for e in errors_0]
        assert ErrorCode.NAN in codes
        assert ErrorCode.FALLBACK_VALUE in codes
        
        errors_1 = flags_ns.unpack(updated_flags, 1)
        assert len(errors_1) == 0
    
    def test_fix_with_tensor_fallback(self) -> None:
        """
        Verify fix accepts tensor as fallback.
        
        Expected: Sample 0 replaced with fallback tensor values.
        """
        tensor = torch.ones(2, 3)
        flags = err.new_t(2)
        code = torch.tensor([ErrorCode.NAN, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        fallback_tensor = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]])
        cleaned, _ = fix(tensor, flags, "test", fallback=fallback_tensor)
        
        assert cleaned[0].tolist() == [-1.0, -2.0, -3.0]
        assert cleaned[1].tolist() == [1.0, 1.0, 1.0]
    
    def test_fix_with_callable_fallback(self) -> None:
        """
        Verify fix accepts callable that returns fallback.
        
        Expected: Sample 0 replaced with callable result (zeros).
        """
        tensor = torch.ones(2, 3)
        flags = err.new_t(2)
        code = torch.tensor([ErrorCode.NAN, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        cleaned, _ = fix(tensor, flags, "test", fallback=lambda: torch.zeros(2, 3))
        
        assert cleaned[0].tolist() == [0.0, 0.0, 0.0]
        assert cleaned[1].tolist() == [1.0, 1.0, 1.0]


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLVE_LOCATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolveLocation:
    """Tests for resolve_location() function."""
    
    def setup_method(self) -> None:
        """Reset caches before each test."""
        ErrorLocation.reset()
        clear_location_cache()
        clear_warn_cache()
    
    def test_resolve_none_returns_unknown(self) -> None:
        """
        Verify None resolves to UNKNOWN.
        
        Expected: Returns ErrorLocation.UNKNOWN (0).
        """
        assert resolve_location(None) == ErrorLocation.UNKNOWN
    
    def test_resolve_int_passthrough(self) -> None:
        """
        Verify int is passed through unchanged.
        
        Expected: Returns exact same int value.
        """
        assert resolve_location(42) == 42
        assert resolve_location(0) == 0
        assert resolve_location(1023) == 1023
    
    def test_resolve_string_registers(self) -> None:
        """
        Verify string is registered and returns ID.
        
        Expected: Returns ID >= 1, name lookup returns original string.
        """
        loc_id = resolve_location("encoder.layer0.attn")
        assert loc_id >= 1
        assert ErrorLocation.name(loc_id) == "encoder.layer0.attn"
    
    def test_resolve_string_idempotent(self) -> None:
        """
        Verify same string returns same ID.
        
        Expected: Both calls return identical ID.
        """
        id1 = resolve_location("encoder.layer0")
        id2 = resolve_location("encoder.layer0")
        assert id1 == id2
    
    def test_resolve_module_with_fx_path(self) -> None:
        """
        Verify module with _fx_path uses that path.
        
        Expected: Location name is the _fx_path value.
        """
        module = nn.Linear(10, 10)
        module._fx_path = "encoder.layer0.linear"
        
        loc_id = resolve_location(module)
        assert ErrorLocation.name(loc_id) == "encoder.layer0.linear"
    
    def test_resolve_module_fallback_warning(self) -> None:
        """
        Verify module without _fx_path warns and uses class name.
        
        Expected: Warning emitted, location is class name "Linear".
        """
        module = nn.Linear(10, 10)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loc_id = resolve_location(module)
            
            assert len(w) == 1
            assert "Linear" in str(w[0].message)
            assert "no _fx_path" in str(w[0].message)
        
        assert ErrorLocation.name(loc_id) == "Linear"
    
    def test_resolve_location_cache_weak_ref(self) -> None:
        """
        Verify cache doesn't prevent module garbage collection.
        
        Expected: Module is GC'd after deleting strong reference.
        """
        module = nn.Linear(10, 10)
        module._fx_path = "test_module"
        loc_id = resolve_location(module)
        
        ref = weakref.ref(module)
        
        del module
        gc.collect()
        
        assert ref() is None
    
    def test_resolve_module_cached(self) -> None:
        """
        Verify same module returns cached result.
        
        Expected: Both calls return identical ID.
        """
        module = nn.Linear(10, 10)
        module._fx_path = "cached_module"
        
        id1 = resolve_location(module)
        id2 = resolve_location(module)
        
        assert id1 == id2


# ═══════════════════════════════════════════════════════════════════════════════
# FLAG HELPER TESTS (Error Recording)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These helpers combine detection + recording in one call.
# They are NOT mask-producing functions - for control flow use find().
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlagNan:
    """Tests for flag_nan() function (error recording helper)."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        clear_location_cache()
    
    def test_flag_nan_writes_error(self) -> None:
        """
        Verify flag_nan writes NAN code to flags for affected samples.
        
        Expected: Error only in sample 1 which contains NaN.
        """
        tensor = torch.tensor([
            [1.0, 2.0],
            [float('nan'), 1.0],
            [3.0, 4.0],
        ])
        
        flags = flag_nan(tensor, "test")
        
        has_error = err.is_err(flags)
        assert has_error.tolist() == [False, True, False]
    
    def test_flag_nan_no_errors(self) -> None:
        """
        Verify flag_nan returns clean flags when no NaN present.
        
        Expected: has_err returns False.
        """
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        flags = flag_nan(tensor, "test")
        
        assert not has_err(flags)
    
    def test_flag_nan_accumulates(self) -> None:
        """
        Verify flag_nan accumulates into existing flags.
        
        Expected: Sample 0 has both INF and NAN codes.
        """
        existing_flags = err.new_t(2)
        code = torch.tensor([ErrorCode.INF, ErrorCode.OK], dtype=torch.int64)
        existing_flags = err.push(existing_flags, code, location=1, severity=Severity.CRITICAL)
        
        tensor = torch.tensor([[float('nan')], [float('nan')]])
        
        flags = flag_nan(tensor, "test", flags=existing_flags)
        
        errors_0 = flags_ns.unpack(flags, 0)
        codes = [e.code for e in errors_0]
        assert ErrorCode.INF in codes
        assert ErrorCode.NAN in codes


class TestFlagInf:
    """Tests for flag_inf() function (error recording helper)."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        clear_location_cache()
    
    def test_flag_inf_writes_error(self) -> None:
        """
        Verify flag_inf writes INF code to flags for affected samples.
        
        Expected: Errors in samples 1 and 2 which contain Inf.
        """
        tensor = torch.tensor([
            [1.0, 2.0],
            [float('inf'), 1.0],
            [3.0, float('-inf')],
        ])
        
        flags = flag_inf(tensor, "test")
        
        has_error = err.is_err(flags)
        assert has_error.tolist() == [False, True, True]
    
    def test_flag_inf_no_errors(self) -> None:
        """
        Verify flag_inf returns clean flags when no Inf present.
        
        Expected: has_err returns False.
        """
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        flags = flag_inf(tensor, "test")
        
        assert not has_err(flags)


class TestFlagOobIndices:
    """Tests for flag_oob_indices() function (error recording helper)."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        clear_location_cache()
    
    def test_flag_oob_indices_writes_error(self) -> None:
        """
        Verify flag_oob_indices writes OUT_OF_BOUNDS code for affected samples.
        
        Expected: Errors in samples 1 (10 >= 10) and 2 (-1 < 0).
        """
        indices = torch.tensor([
            [0, 1, 2],
            [5, 10, 3],
            [-1, 5, 6],
        ])
        
        flags = flag_oob_indices(indices, num_embeddings=10, module="test")
        
        has_error = err.is_err(flags)
        assert has_error.tolist() == [False, True, True]
    
    def test_flag_oob_indices_no_errors(self) -> None:
        """
        Verify flag_oob_indices returns clean flags when all indices valid.
        
        Expected: has_err returns False.
        """
        indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
        flags = flag_oob_indices(indices, num_embeddings=10, module="test")
        
        assert not has_err(flags)


# ═══════════════════════════════════════════════════════════════════════════════
# TORCH.COMPILE COMPATIBILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompileCompatibility:
    """Tests that checks functions work with torch.compile."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
        clear_location_cache()
    
    def test_find_compiles(self) -> None:
        """
        Verify find is compilable.
        
        Expected: Compiled function returns correct mask.
        """
        @torch.compile(backend="eager", fullgraph=True)
        def find_nan(flags: Tensor) -> Tensor:
            return find(ErrorCode.NAN, flags)
        
        flags = err.new_t(5)
        code = torch.tensor([ErrorCode.OK, ErrorCode.NAN, ErrorCode.OK, ErrorCode.NAN, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        result = find_nan(flags)
        assert result.tolist() == [False, True, False, True, False]
    
    def test_push_compiles(self) -> None:
        """
        Verify push is compilable with pre-resolved location.
        
        Expected: Compiled function pushes errors correctly.
        """
        loc = resolve_location("test_compile")
        
        @torch.compile(backend="eager", fullgraph=True)
        def push_nan(flags: Tensor, where: Tensor) -> Tensor:
            code_tensor = torch.where(where, ErrorCode.NAN, ErrorCode.OK)
            return err.push(flags, code_tensor, loc, Severity.CRITICAL)
        
        flags = err.new_t(3)
        where = torch.tensor([True, False, True])
        
        result = push_nan(flags, where)
        has_error = err.is_err(result)
        assert has_error.tolist() == [True, False, True]


# ═══════════════════════════════════════════════════════════════════════════════
# err.replace() Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrReplace:
    """Tests for stable err.replace() function."""
    
    def test_replace_nan(self) -> None:
        """Should replace NaN values with specified value."""
        t = torch.tensor([1.0, float('nan'), 3.0])
        result = err.replace(t, value=0.0, targets=[err.NAN])
        expected = torch.tensor([1.0, 0.0, 3.0])
        assert torch.allclose(result, expected, equal_nan=False)
        assert not torch.isnan(result).any()
    
    def test_replace_inf(self) -> None:
        """Should replace all Inf values (+/-) with specified value."""
        t = torch.tensor([1.0, float('inf'), float('-inf'), 4.0])
        result = err.replace(t, value=0.0, targets=[err.INF])
        expected = torch.tensor([1.0, 0.0, 0.0, 4.0])
        assert torch.allclose(result, expected)
    
    def test_replace_posinf_only(self) -> None:
        """Should replace only +Inf when using 'posinf'."""
        t = torch.tensor([1.0, float('inf'), float('-inf'), 4.0])
        result = err.replace(t, value=0.0, targets=['posinf'])
        expected = torch.tensor([1.0, 0.0, float('-inf'), 4.0])
        assert torch.allclose(result, expected)
    
    def test_replace_neginf_only(self) -> None:
        """Should replace only -Inf when using 'neginf'."""
        t = torch.tensor([1.0, float('inf'), float('-inf'), 4.0])
        result = err.replace(t, value=0.0, targets=['neginf'])
        expected = torch.tensor([1.0, float('inf'), 0.0, 4.0])
        assert torch.allclose(result, expected)
    
    def test_replace_specific_value(self) -> None:
        """Should replace specific numerical values."""
        t = torch.tensor([1.0, 999.0, 3.0, 999.0])
        result = err.replace(t, value=-1.0, targets=[999.0])
        expected = torch.tensor([1.0, -1.0, 3.0, -1.0])
        assert torch.allclose(result, expected)
    
    def test_replace_multiple_targets(self) -> None:
        """Should replace multiple target types at once."""
        t = torch.tensor([1.0, float('nan'), float('inf'), 999.0, 5.0])
        result = err.replace(t, value=0.0, targets=[err.NAN, err.INF, 999])
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 5.0])
        assert torch.allclose(result, expected)
    
    def test_replace_empty_targets_raises(self) -> None:
        """Should raise ValueError when targets is empty."""
        t = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="targets cannot be empty"):
            err.replace(t, value=0.0, targets=[])
    
    def test_replace_preserves_requires_grad(self) -> None:
        """Should preserve requires_grad on output tensor."""
        t = torch.tensor([1.0, float('nan'), 3.0], requires_grad=True)
        result = err.replace(t, value=0.0, targets=[err.NAN])
        assert result.requires_grad
    
    def test_replace_gradient_flows(self) -> None:
        """Gradients should flow through non-replaced values."""
        t = torch.tensor([1.0, float('nan'), 3.0], requires_grad=True)
        result = err.replace(t, value=0.0, targets=[err.NAN])
        loss = result.sum()
        loss.backward()
        # Gradient should be 1 for non-replaced values, 0 for replaced
        expected_grad = torch.tensor([1.0, 0.0, 1.0])
        assert torch.allclose(t.grad, expected_grad)
    
    def test_replace_multidim_tensor(self) -> None:
        """Should work with multi-dimensional tensors."""
        t = torch.tensor([[1.0, float('nan')], [float('inf'), 4.0]])
        result = err.replace(t, value=0.0, targets=[err.NAN, err.INF])
        expected = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        assert torch.allclose(result, expected)
