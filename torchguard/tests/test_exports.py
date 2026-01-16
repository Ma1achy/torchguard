"""Test that all public API is properly exported.

This test module validates that:
1. All intended exports are present in torchguard
2. Class methods are accessible
3. Various import patterns work
4. No circular imports exist
"""

import pytest
import torch


class TestPrimaryExports:
    """Test primary API exports."""
    
    def test_import_top_level(self):
        """Test top-level import works."""
        import torchguard
        assert torchguard is not None
    
    def test_err_namespace(self):
        """Test err namespace is exported."""
        from torchguard import err
        assert hasattr(err, 'new')
        assert hasattr(err, 'push')
        assert hasattr(err, 'is_ok')
        assert hasattr(err, 'is_err')
        assert hasattr(err, 'has_any')
        assert hasattr(err, 'NAN')
        assert hasattr(err, 'INF')
    
    def test_flags_namespace(self):
        """Test flags namespace is exported."""
        from torchguard import flags
        assert hasattr(flags, 'unpack')
        assert hasattr(flags, 'repr')
    
    def test_decorators(self):
        """Test decorators are exported."""
        from torchguard import tracked, tensorcheck
        assert callable(tracked)
        assert callable(tensorcheck)
    
    def test_error_t_alias(self):
        """Test error_t type alias is exported."""
        from torchguard import error_t
        assert error_t is not None


class TestSecondaryExports:
    """Test secondary/advanced API exports."""
    
    def test_error_code_class(self):
        """Test ErrorCode class and attributes."""
        from torchguard import ErrorCode
        
        # Check constants
        assert ErrorCode.OK == 0
        assert ErrorCode.NAN == 1
        assert ErrorCode.INF == 2
        assert hasattr(ErrorCode, 'OVERFLOW')
        assert hasattr(ErrorCode, 'OUT_OF_BOUNDS')
    
    def test_error_domain_class(self):
        """Test ErrorDomain class and attributes."""
        from torchguard import ErrorDomain
        
        # Check constants
        assert ErrorDomain.NUMERIC == 0
        assert hasattr(ErrorDomain, 'INDEX')
        assert hasattr(ErrorDomain, 'QUALITY')
        assert hasattr(ErrorDomain, 'RUNTIME')
    
    def test_severity_class(self):
        """Test Severity class and attributes."""
        from torchguard import Severity
        
        # Check constants
        assert Severity.OK == 0
        assert Severity.WARN == 1
        assert Severity.ERROR == 2
        assert Severity.CRITICAL == 3
    
    def test_config_exports(self):
        """Test configuration exports."""
        from torchguard import ErrorConfig, CONFIG, get_config, set_config
        
        assert isinstance(CONFIG, ErrorConfig)
        assert callable(get_config)
        assert callable(set_config)
        
        # Test they work
        cfg = get_config()
        assert isinstance(cfg, ErrorConfig)
    
    def test_accumulation_config(self):
        """Test AccumulationConfig and related enums."""
        from torchguard import AccumulationConfig, Priority, Order, Dedupe
        
        # Check enums have expected values
        assert hasattr(Priority, 'CHRONO')
        assert hasattr(Priority, 'SEVERITY')
        assert hasattr(Order, 'FIRST')
        assert hasattr(Order, 'LAST')
        assert hasattr(Dedupe, 'UNIQUE')
        assert hasattr(Dedupe, 'NONE')
        assert hasattr(Dedupe, 'CODE')
        assert hasattr(Dedupe, 'LOCATION')
    
    def test_constants(self):
        """Test bit layout constants are exported."""
        from torchguard import (
            SLOT_BITS, SLOTS_PER_WORD,
            CODE_SHIFT, CODE_BITS, CODE_MASK,
            LOCATION_SHIFT, LOCATION_BITS, LOCATION_MASK,
            SEVERITY_SHIFT, SEVERITY_BITS, SEVERITY_MASK,
            SLOT_MASK,
        )
        
        assert SLOT_BITS == 16
        assert SLOTS_PER_WORD == 4
        assert isinstance(CODE_SHIFT, int)
        assert isinstance(CODE_BITS, int)
        assert isinstance(CODE_MASK, int)
    
    def test_control_flow_dsl(self):
        """Test control flow DSL exports."""
        from torchguard import IF, HAS, IS, OR, AND, NOT
        
        assert callable(IF)
        assert callable(HAS)
        assert callable(IS)
        assert callable(OR)
        assert callable(AND)
        assert callable(NOT)
    
    def test_result_types(self):
        """Test Result type exports."""
        from torchguard import Ok, Err, Result
        
        # Test constructors work
        ok = Ok(42)
        err = Err("error")
        
        assert ok.is_ok()
        assert not ok.is_err()
        assert err.is_err()
        assert not err.is_ok()
    
    def test_helper_functions(self):
        """Test helper function exports."""
        from torchguard import (
            push, find, fix,
            flag_nan, flag_inf, flag_oob_indices,
            has_err,
        )
        
        assert callable(push)
        assert callable(find)
        assert callable(fix)
        assert callable(flag_nan)
        assert callable(flag_inf)
        assert callable(flag_oob_indices)
        assert callable(has_err)
    
    def test_validation_errors(self):
        """Test validation error exports."""
        from torchguard import (
            ValidationError,
            DimensionMismatchError,
            DTypeMismatchError,
            DeviceMismatchError,
            InvalidParameterError,
            TypeMismatchError,
            InvalidReturnTypeError,
        )
        
        # Check they're exceptions
        assert issubclass(ValidationError, Exception)
        assert issubclass(DimensionMismatchError, ValidationError)
        assert issubclass(DTypeMismatchError, ValidationError)
        assert issubclass(DeviceMismatchError, ValidationError)
    
    def test_experimental_namespace(self):
        """Test experimental namespace is exported."""
        from torchguard import experimental
        
        assert hasattr(experimental, 'err')
        assert hasattr(experimental, 'IF')
    
    def test_unpacked_error(self):
        """Test UnpackedError and ErrorFlags exports."""
        from torchguard import UnpackedError, ErrorFlags
        
        assert UnpackedError is not None
        assert ErrorFlags is not None
    
    def test_error_location(self):
        """Test ErrorLocation export."""
        from torchguard import ErrorLocation
        
        assert ErrorLocation is not None
    
    def test_version(self):
        """Test __version__ is exported."""
        from torchguard import __version__
        
        assert isinstance(__version__, str)
        # Should be in format X.Y.Z
        parts = __version__.split('.')
        assert len(parts) >= 2


class TestClassMethods:
    """Test that class methods are accessible."""
    
    def test_error_code_name(self):
        """Test ErrorCode.name() method."""
        import torchguard as tg
        
        assert tg.ErrorCode.name(0) == "OK"
        assert tg.ErrorCode.name(1) == "NAN"
        assert tg.ErrorCode.name(2) == "INF"
        assert tg.ErrorCode.name(3) == "OVERFLOW"
    
    def test_error_code_is_critical(self):
        """Test ErrorCode.is_critical() method."""
        import torchguard as tg
        
        assert tg.ErrorCode.is_critical(tg.ErrorCode.NAN) == True
        assert tg.ErrorCode.is_critical(tg.ErrorCode.INF) == True
        assert tg.ErrorCode.is_critical(tg.ErrorCode.OVERFLOW) == False
        assert tg.ErrorCode.is_critical(tg.ErrorCode.OK) == False
    
    def test_error_code_domain(self):
        """Test ErrorCode.domain() method."""
        import torchguard as tg
        
        # NAN is in NUMERIC domain
        assert tg.ErrorCode.domain(tg.ErrorCode.NAN) == 0
        # OUT_OF_BOUNDS is in INDEX domain
        assert tg.ErrorCode.domain(tg.ErrorCode.OUT_OF_BOUNDS) == 1
    
    def test_error_code_in_domain(self):
        """Test ErrorCode.in_domain() method."""
        import torchguard as tg
        
        assert tg.ErrorCode.in_domain(tg.ErrorCode.NAN, tg.ErrorDomain.NUMERIC) == True
        assert tg.ErrorCode.in_domain(tg.ErrorCode.NAN, tg.ErrorDomain.INDEX) == False
        assert tg.ErrorCode.in_domain(tg.ErrorCode.OUT_OF_BOUNDS, tg.ErrorDomain.INDEX) == True
    
    def test_severity_name(self):
        """Test Severity.name() method."""
        import torchguard as tg
        
        assert tg.Severity.name(0) == "OK"
        assert tg.Severity.name(1) == "WARN"
        assert tg.Severity.name(2) == "ERROR"
        assert tg.Severity.name(3) == "CRITICAL"
    
    def test_severity_is_critical(self):
        """Test Severity.is_critical() method."""
        import torchguard as tg
        
        assert tg.Severity.is_critical(tg.Severity.CRITICAL) == True
        assert tg.Severity.is_critical(tg.Severity.ERROR) == False
    
    def test_severity_is_error_or_worse(self):
        """Test Severity.is_error_or_worse() method."""
        import torchguard as tg
        
        assert tg.Severity.is_error_or_worse(tg.Severity.CRITICAL) == True
        assert tg.Severity.is_error_or_worse(tg.Severity.ERROR) == True
        assert tg.Severity.is_error_or_worse(tg.Severity.WARN) == False
        assert tg.Severity.is_error_or_worse(tg.Severity.OK) == False
    
    def test_error_domain_name(self):
        """Test ErrorDomain.name() method."""
        import torchguard as tg
        
        assert tg.ErrorDomain.name(0) == "NUMERIC"
        assert tg.ErrorDomain.name(1) == "INDEX"
        assert tg.ErrorDomain.name(2) == "QUALITY"
        assert tg.ErrorDomain.name(3) == "RUNTIME"


class TestConfigFunctions:
    """Test configuration management functions."""
    
    def test_get_config(self):
        """Test get_config returns ErrorConfig."""
        import torchguard as tg
        
        cfg = tg.get_config()
        assert isinstance(cfg, tg.ErrorConfig)
    
    def test_set_config(self):
        """Test set_config replaces global config."""
        import torchguard as tg
        
        # Save original
        original = tg.get_config()
        original_dtype = original.flag_dtype
        
        try:
            # Create and set new config
            new_config = tg.ErrorConfig(flag_dtype=torch.float32)
            tg.set_config(new_config)
            
            # Verify it was set
            assert tg.get_config().flag_dtype == torch.float32
        finally:
            # Restore original
            tg.set_config(original)
    
    def test_config_is_global(self):
        """Test CONFIG is the global instance."""
        import torchguard as tg
        
        assert tg.CONFIG is tg.get_config()


class TestImportPatterns:
    """Test various import patterns work."""
    
    def test_star_import_has_all(self):
        """Test that __all__ is defined."""
        import torchguard
        
        assert hasattr(torchguard, '__all__')
        assert isinstance(torchguard.__all__, (list, tuple))
    
    def test_selective_import(self):
        """Test selective imports work."""
        from torchguard import err, ErrorCode, CONFIG
        
        assert err is not None
        assert ErrorCode is not None
        assert CONFIG is not None
    
    def test_aliased_import(self):
        """Test aliased import works."""
        import torchguard as tg
        
        assert hasattr(tg, 'err')
        assert hasattr(tg, 'ErrorCode')
        assert hasattr(tg, 'CONFIG')
    
    def test_import_err_directly(self):
        """Test importing err works."""
        from torchguard import err
        
        # Test err has expected attributes
        assert hasattr(err, 'new')
        assert hasattr(err, 'new_t')
        assert hasattr(err, 'push')
        assert hasattr(err, 'NAN')


class TestNoCircularImports:
    """Test that no circular imports exist by importing in different orders."""
    
    def test_import_order_core_first(self):
        """Test import order 1: core types first, then err."""
        import sys
        # Clear cached modules to test fresh import
        keys_to_remove = [k for k in sys.modules if k.startswith('torchguard')]
        for k in keys_to_remove:
            del sys.modules[k]
        
        # Import core types first
        from torchguard import ErrorCode, ErrorConfig
        # Then err namespace
        from torchguard import err
        
        assert ErrorCode is not None
        assert err is not None
    
    def test_import_order_err_first(self):
        """Test import order 2: err first, then core."""
        import sys
        keys_to_remove = [k for k in sys.modules if k.startswith('torchguard')]
        for k in keys_to_remove:
            del sys.modules[k]
        
        # Import err namespace first
        from torchguard import err
        # Then core types
        from torchguard import ErrorCode, ErrorConfig
        
        assert err is not None
        assert ErrorCode is not None
    
    def test_import_order_experimental(self):
        """Test import order 3: experimental namespace."""
        import sys
        keys_to_remove = [k for k in sys.modules if k.startswith('torchguard')]
        for k in keys_to_remove:
            del sys.modules[k]
        
        # Import all namespaces
        from torchguard import ErrorConfig, experimental, err
        
        assert ErrorConfig is not None
        assert experimental is not None
        assert err is not None
    
    def test_import_decorators(self):
        """Test decorators can be imported."""
        from torchguard import tracked, tensorcheck
        
        assert callable(tracked)
        assert callable(tensorcheck)
    
    def test_import_control(self):
        """Test control flow can be imported."""
        from torchguard import IF, HAS
        
        assert callable(IF)
        assert callable(HAS)


class TestErrNamespaceMethods:
    """Test err namespace methods work."""
    
    def test_err_new(self):
        """Test err.new creates flags."""
        import torchguard as tg
        
        x = torch.randn(5, 10)
        flags = tg.err.new(x)
        
        assert flags.shape[0] == 5  # Batch size
        assert flags.dtype == tg.CONFIG.torch_dtype
    
    def test_err_new_t(self):
        """Test err.new_t creates flags."""
        import torchguard as tg
        
        flags = tg.err.new_t(10)
        
        assert flags.shape[0] == 10
        assert flags.dtype == tg.CONFIG.torch_dtype
    
    def test_err_is_ok(self):
        """Test err.is_ok works."""
        import torchguard as tg
        
        flags = tg.err.new_t(5)
        ok = tg.err.is_ok(flags)
        
        assert ok.shape == (5,)
        assert ok.all()  # Fresh flags should be OK
    
    def test_err_push(self):
        """Test err.push adds error."""
        import torchguard as tg
        
        flags = tg.err.new_t(5)
        flags = tg.err.push(flags, code=tg.ErrorCode.NAN, location=1)
        
        assert tg.err.has_code(flags, tg.ErrorCode.NAN).any()
    
    def test_err_codes_as_attributes(self):
        """Test error codes are accessible as err attributes."""
        import torchguard as tg
        
        assert tg.err.OK == tg.ErrorCode.OK
        assert tg.err.NAN == tg.ErrorCode.NAN
        assert tg.err.INF == tg.ErrorCode.INF


class TestAllDeclaration:
    """Test __all__ is comprehensive."""
    
    def test_all_contains_primary(self):
        """Test __all__ contains primary exports."""
        import torchguard as tg
        
        primary = ['err', 'flags', 'error_t', 'tracked', 'tensorcheck']
        for name in primary:
            assert name in tg.__all__, f"'{name}' not in __all__"
    
    def test_all_contains_config(self):
        """Test __all__ contains config exports."""
        import torchguard as tg
        
        config_exports = ['ErrorConfig', 'CONFIG', 'get_config', 'set_config']
        for name in config_exports:
            assert name in tg.__all__, f"'{name}' not in __all__"
    
    def test_all_contains_dsl(self):
        """Test __all__ contains DSL exports."""
        import torchguard as tg
        
        dsl_exports = ['IF', 'HAS', 'IS', 'OR', 'AND', 'NOT']
        for name in dsl_exports:
            assert name in tg.__all__, f"'{name}' not in __all__"
