"""
Tests for compiled module exports.

Tests cover:
- err and flags namespaces
- error_t type alias
- Check functions exports
- Control Flow DSL exports
- as_result export
- No circular imports

Run with:
    pytest tests/utils/errors/compiled/test_exports.py -v
"""
import pytest
import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════
# NAMESPACE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNamespaces:
    """Tests for err and flags namespaces."""
    
    def test_err_namespace_exported(self) -> None:
        """Verify err namespace is exported with expected methods."""
        from torchguard import err
        
        # Core methods
        assert hasattr(err, 'new')
        assert hasattr(err, 'new_t')
        assert hasattr(err, 'push')
        assert hasattr(err, 'is_ok')
        assert hasattr(err, 'is_err')
        assert hasattr(err, 'has_any')
        
        # Error codes as attributes
        assert hasattr(err, 'NAN')
        assert hasattr(err, 'INF')
        assert hasattr(err, 'CRITICAL')
    
    def test_flags_namespace_exported(self) -> None:
        """Verify flags namespace is exported with expected methods."""
        from torchguard import flags
        
        assert hasattr(flags, 'unpack')
        assert hasattr(flags, 'repr')
        assert hasattr(flags, 'summary')
    
    def test_error_t_exported(self) -> None:
        """Verify error_t is exported as type alias."""
        from torchguard import error_t
        
        # error_t should be a class (dtype alias)
        assert isinstance(error_t, type)


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK FUNCTIONS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckFunctionsExported:
    """Tests for check functions exports."""
    
    def test_check_functions_exported(self) -> None:
        """Verify all check functions are exported."""
        from torchguard import (
            has_err, find, push, fix,
            flag_nan, flag_inf, flag_oob_indices,
        )
        
        assert callable(has_err)
        assert callable(find)
        assert callable(push)
        assert callable(fix)
        assert callable(flag_nan)
        assert callable(flag_inf)
        assert callable(flag_oob_indices)
    
    def test_check_functions_in_all(self) -> None:
        """Verify check functions are in __all__."""
        from torchguard import __all__
        
        expected = [
            'has_err', 'find', 'push', 'fix',
            'flag_nan', 'flag_inf', 'flag_oob_indices',
        ]
        for name in expected:
            assert name in __all__, f"{name} not in __all__"


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL FLOW DSL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestControlFlowDSLExported:
    """Tests for Control Flow DSL exports."""
    
    def test_control_flow_dsl_exported(self) -> None:
        """Verify Control Flow DSL is exported."""
        from torchguard import IF, HAS, IS, OR, AND, NOT
        
        assert callable(IF)
        assert callable(HAS)
        assert callable(IS)
        assert callable(OR)
        assert callable(AND)
        assert callable(NOT)
    
    def test_control_flow_dsl_in_all(self) -> None:
        """Verify Control Flow DSL is in __all__."""
        from torchguard import __all__
        
        expected = ['IF', 'HAS', 'IS', 'OR', 'AND', 'NOT']
        for name in expected:
            assert name in __all__, f"{name} not in __all__"


# ═══════════════════════════════════════════════════════════════════════════════
# AS_RESULT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsResultExported:
    """Tests for as_result export."""
    
    def test_as_result_exported(self) -> None:
        """Verify as_result is exported."""
        from torchguard import as_result
        
        assert callable(as_result)
    
    def test_as_result_in_all(self) -> None:
        """Verify as_result is in __all__."""
        from torchguard import __all__
        
        assert 'as_result' in __all__


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCULAR IMPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoCircularImports:
    """Tests to verify no circular imports."""
    
    def test_no_circular_imports_full(self) -> None:
        """Verify module imports without circular import errors."""
        # This test passes if the import succeeds
        from torchguard import (
            err, flags, error_t, ErrorCode,
            has_err, find, push, fix,
            IF, HAS, IS, OR, AND, NOT,
            as_result,
        )
        
        # Basic sanity checks
        assert hasattr(err, 'new')
        assert hasattr(flags, 'unpack')
        assert callable(has_err)
        assert callable(IF)
        assert callable(as_result)
    
    def test_import_order_safe(self) -> None:
        """Verify import order doesn't cause issues."""
        # Import in different order
        from torchguard import as_result
        from torchguard import IF, HAS
        from torchguard import has_err, push
        from torchguard import err, flags, error_t
        
        assert callable(as_result)
        assert callable(IF)
        assert callable(has_err)
        assert hasattr(err, 'new')


# ═══════════════════════════════════════════════════════════════════════════════
# CORE EXPORTS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoreExportsPreserved:
    """Tests that core exports are preserved."""
    
    def test_core_types_preserved(self) -> None:
        """Verify core types are still exported."""
        from torchguard import (
            ErrorCode,
            ErrorDomain,
            ErrorLocation,
            UnpackedError,
            Severity,
            AccumulationConfig,
            Priority,
            Order,
            Dedupe,
            ErrorConfig,
            DEFAULT_CONFIG,
        )
        
        assert ErrorCode is not None
        assert Severity is not None
        assert AccumulationConfig is not None
        assert Priority is not None
        assert Order is not None
        assert Dedupe is not None
    
    def test_constants_preserved(self) -> None:
        """Verify constants are still exported."""
        from torchguard import (
            SLOT_BITS,
            SLOTS_PER_WORD,
            SEVERITY_MASK,
            CODE_SHIFT,
            LOCATION_SHIFT,
            SLOT_MASK,
        )
        
        assert SLOT_BITS == 16
        assert SLOTS_PER_WORD == 4
