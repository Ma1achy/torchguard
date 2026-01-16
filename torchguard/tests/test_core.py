"""
Tests for the V2 bit-packed error flags system.

Tests cover:
- ErrorCode values, names, and domain helpers
- ErrorLocation values, names, and auto-registration
- Severity levels and core mapping
- ErrorConfig and AccumulationConfig (Priority, Order, Dedupe)
- ErrorFlags creation, push (all configs), extend, combine
- Check methods (is_ok, is_err, has_code, has_nan, has_critical, max_severity, etc.)
- Unpacking and interpretation
- torch.compile() compatibility

Run with:
    pytest tests/utils/errors/compiled/test_core.py -v
"""
import pytest
import torch
from torch import Tensor

from torchguard import (
    # Core types
    ErrorCode,
    ErrorDomain,
    ErrorLocation,
    UnpackedError,
    error_t,
    err,
    flags as flags_ns,
    # Configuration
    Severity,
    AccumulationConfig,
    Priority,
    Order,
    Dedupe,
    ErrorConfig,
    CONFIG,
    # Constants
    SLOT_BITS,
    SLOTS_PER_WORD,
    SEVERITY_MASK,
    CODE_SHIFT,
    LOCATION_SHIFT,
    SLOT_MASK,
)


class TestErrorCode:
    """Tests for ErrorCode constants and methods."""

    def test_ok_is_zero(self) -> None:
        """OK should be 0 for easy bitwise checks."""
        assert ErrorCode.OK == 0

    def test_all_codes_fit_in_4_bits(self) -> None:
        """All codes should be 0-15 (4 bits)."""
        for name in dir(ErrorCode):
            if name.isupper() and not name.startswith('_'):
                val = getattr(ErrorCode, name)
                if isinstance(val, int):
                    assert 0 <= val <= 15, f"{name}={val} doesn't fit in 4 bits"

    def test_codes_are_unique(self) -> None:
        """All error codes should have unique values (except aliases)."""
        # Aliases that map to same values
        aliases = {'UNDERFLOW', 'INVALID_VALUE', 'UNSTABLE', 'CLAMPED', 'SPARSE'}
        codes = {}
        for name in dir(ErrorCode):
            if name.isupper() and not name.startswith('_') and name not in aliases:
                val = getattr(ErrorCode, name)
                if isinstance(val, int):
                    assert val not in codes, f"{name} and {codes[val]} have same value {val}"
                    codes[val] = name

    def test_critical_codes(self) -> None:
        """NAN and INF should be marked as critical."""
        assert ErrorCode.is_critical(ErrorCode.NAN)
        assert ErrorCode.is_critical(ErrorCode.INF)
        assert not ErrorCode.is_critical(ErrorCode.OK)
        assert not ErrorCode.is_critical(ErrorCode.OUT_OF_BOUNDS)

    def test_name_lookup(self) -> None:
        """name() should return correct names."""
        assert ErrorCode.name(0) == "OK"
        assert ErrorCode.name(1) == "NAN"
        assert ErrorCode.name(2) == "INF"
        assert ErrorCode.name(15) == "UNKNOWN"
        assert "CODE_" in ErrorCode.name(99)

    def test_domain_extraction(self) -> None:
        """domain() should extract high 2 bits."""
        assert ErrorCode.domain(ErrorCode.NAN) == 0  # NUMERIC
        assert ErrorCode.domain(ErrorCode.OUT_OF_BOUNDS) == 1  # INDEX
        assert ErrorCode.domain(ErrorCode.ZERO_OUTPUT) == 2  # QUALITY
        assert ErrorCode.domain(ErrorCode.FALLBACK_VALUE) == 3  # RUNTIME

    def test_in_domain(self) -> None:
        """in_domain() should correctly identify domains."""
        assert ErrorCode.in_domain(ErrorCode.NAN, ErrorDomain.NUMERIC)
        assert ErrorCode.in_domain(ErrorCode.INF, ErrorDomain.NUMERIC)
        assert ErrorCode.in_domain(ErrorCode.OUT_OF_BOUNDS, ErrorDomain.INDEX)
        assert not ErrorCode.in_domain(ErrorCode.NAN, ErrorDomain.INDEX)


class TestErrorLocation:
    """Tests for ErrorLocation constants and registry."""

    def test_unknown_is_zero(self) -> None:
        """UNKNOWN should be 0."""
        assert ErrorLocation.UNKNOWN == 0

    def test_unknown_is_only_builtin(self) -> None:
        """UNKNOWN is the only built-in location constant."""
        assert ErrorLocation.UNKNOWN == 0
        assert ErrorLocation.name(0) == "Unknown"

    def test_name_lookup(self) -> None:
        """name() should return registered names or fallback."""
        ErrorLocation.reset()
        assert ErrorLocation.name(0) == "Unknown"
        loc_id = ErrorLocation.register("test.module")
        assert ErrorLocation.name(loc_id) == "test.module"
        assert "Location_" in ErrorLocation.name(999)

    def test_register_and_lookup(self) -> None:
        """register() should return consistent IDs."""
        ErrorLocation.reset()  # Clean slate
        
        id1 = ErrorLocation.register("test.module.a")
        id2 = ErrorLocation.register("test.module.b")
        id3 = ErrorLocation.register("test.module.a")  # Same as first
        
        assert id1 >= 1  # After UNKNOWN (0)
        assert id2 == id1 + 1
        assert id3 == id1  # Idempotent
        
        assert ErrorLocation.name(id1) == "test.module.a"
        
        ErrorLocation.reset()  # Cleanup


class TestSeverity:
    """Tests for Severity levels and core mapping."""

    def test_severity_values(self) -> None:
        """Severity should have correct values."""
        assert Severity.OK == 0
        assert Severity.WARN == 1
        assert Severity.ERROR == 2
        assert Severity.CRITICAL == 3

    def test_severity_fits_in_2_bits(self) -> None:
        """All severities should fit in 2 bits."""
        for val in [Severity.OK, Severity.WARN, Severity.ERROR, Severity.CRITICAL]:
            assert 0 <= val <= 3

    def test_severity_helpers(self) -> None:
        """Test severity helper methods."""
        assert Severity.is_critical(Severity.CRITICAL)
        assert not Severity.is_critical(Severity.ERROR)
        
        assert Severity.is_error_or_worse(Severity.CRITICAL)
        assert Severity.is_error_or_worse(Severity.ERROR)
        assert not Severity.is_error_or_worse(Severity.WARN)
        
        assert Severity.is_warn_or_worse(Severity.CRITICAL)
        assert Severity.is_warn_or_worse(Severity.WARN)
        assert not Severity.is_warn_or_worse(Severity.OK)


class TestErrorConfig:
    """Tests for ErrorConfig configuration."""

    def test_default_config(self) -> None:
        """
        Verify CONFIG has expected values.
        
        Expected: 16 slots, FIFO accumulation (root cause preservation), ERROR severity.
        """
        assert CONFIG.num_slots == 16
        assert CONFIG.accumulation.priority == Priority.CHRONO
        assert CONFIG.accumulation.order == Order.FIRST  # FIFO: preserve root causes
        assert CONFIG.accumulation.dedupe == Dedupe.UNIQUE
        assert CONFIG.default_severity == Severity.ERROR
        assert CONFIG.num_words == 4  # 16 slots / 4 slots per word

    def test_custom_config(self) -> None:
        """Custom config should work."""
        config = ErrorConfig(
            num_slots=16,
            accumulation=AccumulationConfig(priority=Priority.SEVERITY, order=Order.LAST)
        )
        assert config.num_slots == 16
        assert config.num_words == 4
        assert config.accumulation.priority == Priority.SEVERITY

    def test_invalid_num_slots(self) -> None:
        """
        Verify invalid num_slots values are rejected.
        
        Expected: ValueError for num_slots=0 and num_slots>32768.
        """
        with pytest.raises(ValueError):
            ErrorConfig(num_slots=0)
        with pytest.raises(ValueError):
            ErrorConfig(num_slots=32769)  # Max is 32768


class TestErrorFlagsCreation:
    """Tests for ErrorFlags creation methods."""

    def test_ok_creates_zeros(self) -> None:
        """ok() should create all-zero tensor with correct shape."""
        flags = err.new_t(10, torch.device('cpu'))
        assert flags.shape == (10, CONFIG.num_words)
        assert flags.dtype == CONFIG.flag_dtype
        assert (flags == 0).all()

    def test_ok_custom_config(self) -> None:
        """ok() should respect custom config."""
        config = ErrorConfig(num_slots=4, flag_dtype=torch.float64)
        flags = err.new_t(5, config=config)
        assert flags.shape == (5, 1)  # 4 slots with float64 = 1 word (4 slots/word)

    def test_from_code(self) -> None:
        """from_code() should create tensor with packed error."""
        flags = err.from_code(ErrorCode.NAN, ErrorLocation.register("customer_encoder"), 5,
                                      severity=Severity.CRITICAL)
        assert flags.shape == (5, CONFIG.num_words)
        assert (flags[:, 0] != 0).all()  # Error in slot 0

        code = err.get_first_code(flags)
        loc = err.get_first_location(flags)
        sev = err.get_first_severity(flags)
        
        assert (code == ErrorCode.NAN).all()
        assert (loc == ErrorLocation.register("customer_encoder")).all()
        assert (sev == Severity.CRITICAL).all()


class TestErrorFlagsPushModes:
    """Tests for error accumulation with orthogonal axes."""

    def test_push_chrono_last(self) -> None:
        """CHRONO + LAST (LIFO) should keep newest errors."""
        config = ErrorConfig(
            num_slots=4,
            accumulation=AccumulationConfig(priority=Priority.CHRONO, order=Order.LAST)
        )
        flags = err.new_t(1, config=config)
        
        # Push 3 errors
        for code, loc in [(ErrorCode.NAN, 1), (ErrorCode.INF, 2), (ErrorCode.OVERFLOW, 3)]:
            code_t = torch.tensor([code], dtype=torch.int64)
            flags = err.push(flags, code_t, loc, Severity.ERROR, config)
        
        errors = flags_ns.unpack(flags, 0, config)
        assert len(errors) == 3
        # Newest should be in slot 0
        assert errors[0].code == ErrorCode.OVERFLOW

    def test_push_chrono_first(self) -> None:
        """CHRONO + FIRST (FIFO) should keep oldest errors."""
        config = ErrorConfig(
            num_slots=2,
            accumulation=AccumulationConfig(priority=Priority.CHRONO, order=Order.FIRST)
        )
        flags = err.new_t(1, config=config)
        
        # Push 3 errors (only 2 slots)
        for code in [ErrorCode.NAN, ErrorCode.INF, ErrorCode.OVERFLOW]:
            code_t = torch.tensor([code], dtype=torch.int64)
            flags = err.push(flags, code_t, 1, Severity.ERROR, config)
        
        errors = flags_ns.unpack(flags, 0, config)
        assert len(errors) == 2
        # Oldest should be in slot 0
        assert errors[0].code == ErrorCode.NAN
        assert errors[1].code == ErrorCode.INF

    def test_push_severity_priority(self) -> None:
        """SEVERITY + LAST should keep highest severity errors."""
        config = ErrorConfig(
            num_slots=2,
            accumulation=AccumulationConfig(priority=Priority.SEVERITY, order=Order.LAST)
        )
        flags = err.new_t(1, config=config)
        
        # Push WARN, then CRITICAL - CRITICAL should be kept
        warn_code = torch.tensor([ErrorCode.ZERO_OUTPUT], dtype=torch.int64)
        flags = err.push(flags, warn_code, 1, Severity.WARN, config)
        
        critical_code = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, critical_code, 2, Severity.CRITICAL, config)
        
        assert err.max_severity(flags, config)[0] == Severity.CRITICAL

    def test_push_dedupe_location(self) -> None:
        """Dedupe.LOCATION should dedupe by location."""
        config = ErrorConfig(
            num_slots=4,
            accumulation=AccumulationConfig(dedupe=Dedupe.LOCATION)
        )
        flags = err.new_t(1, config=config)
        
        # Push 3 errors at same location - should only keep one
        for code in [ErrorCode.NAN, ErrorCode.INF, ErrorCode.OVERFLOW]:
            code_t = torch.tensor([code], dtype=torch.int64)
            flags = err.push(flags, code_t, location=1, severity=Severity.ERROR, config=config)
        
        # Should have errors, and location 1 should exist
        assert err.is_err(flags)[0]

    def test_push_with_error_condition(self) -> None:
        """push() should only add error where code != OK."""
        config = CONFIG
        flags = err.new_t(3, config=config)
        codes = torch.tensor([ErrorCode.NAN, ErrorCode.OK, ErrorCode.INF], dtype=torch.int64)
        flags = err.push(flags, codes, ErrorLocation.register("customer_encoder"), 
                                severity=Severity.CRITICAL, config=config)

        assert err.is_err(flags)[0]
        assert err.is_ok(flags)[1]
        assert err.is_err(flags)[2]

    def test_push_scalar(self) -> None:
        """push_scalar() should add same error to all samples."""
        flags = err.new_t(5)
        flags = err.push_scalar(flags, ErrorCode.NAN, ErrorLocation.register("hash_helpers"),
                                        severity=Severity.CRITICAL)

        assert err.is_err(flags).all()
        assert (err.get_first_code(flags) == ErrorCode.NAN).all()


class TestErrorFlagsMerge:
    """Tests for error merging operations."""

    def test_merge_two(self) -> None:
        """merge() should merge errors from two tensors."""
        flags1 = err.from_code(ErrorCode.NAN, ErrorLocation.register("customer_encoder"), 3,
                                       severity=Severity.CRITICAL)
        flags2 = err.from_code(ErrorCode.INF, ErrorLocation.register("group_encoder"), 3,
                                       severity=Severity.CRITICAL)

        merged = err.merge(flags1, flags2)
        assert err.count_errors(merged)[0] >= 1

    def test_merge_multiple(self) -> None:
        """merge() should merge multiple flag tensors."""
        f1 = err.from_code(ErrorCode.NAN, ErrorLocation.register("customer_encoder"), 2,
                                   severity=Severity.CRITICAL)
        f2 = err.from_code(ErrorCode.INF, ErrorLocation.register("group_encoder"), 2,
                                   severity=Severity.CRITICAL)
        f3 = err.from_code(ErrorCode.OUT_OF_BOUNDS, ErrorLocation.register("hash_helpers"), 2,
                                   severity=Severity.ERROR)

        merged = err.merge(f1, f2, f3)
        assert err.is_err(merged).all()


class TestErrorFlagsChecks:
    """Tests for error check methods."""

    def test_is_ok_is_err(self) -> None:
        """is_ok and is_err should be opposites."""
        flags = err.new_t(4)
        codes = torch.tensor([ErrorCode.OK, ErrorCode.NAN, ErrorCode.OK, ErrorCode.INF], 
                            dtype=torch.int64)
        flags = err.push(flags, codes, ErrorLocation.UNKNOWN, Severity.CRITICAL)

        is_ok = err.is_ok(flags)
        is_err = err.is_err(flags)

        assert is_ok[0] and not is_err[0]
        assert not is_ok[1] and is_err[1]
        assert is_ok[2] and not is_err[2]
        assert not is_ok[3] and is_err[3]

    def test_has_code(self) -> None:
        """has_code() should find specific error codes."""
        flags = err.new_t(1)
        flags = err.push_scalar(flags, ErrorCode.NAN, ErrorLocation.UNKNOWN, Severity.CRITICAL)
        flags = err.push_scalar(flags, ErrorCode.OUT_OF_BOUNDS, ErrorLocation.UNKNOWN, Severity.ERROR)

        assert err.has_code(flags, ErrorCode.NAN)[0]
        assert err.has_code(flags, ErrorCode.OUT_OF_BOUNDS)[0]
        assert not err.has_code(flags, ErrorCode.INF)[0]

    def test_has_nan(self) -> None:
        """has_nan() should detect NAN errors."""
        flags = err.from_code(ErrorCode.NAN, ErrorLocation.UNKNOWN, 2, 
                                      severity=Severity.CRITICAL)
        assert err.has_nan(flags).all()

        flags_ok = err.new_t(2)
        assert not err.has_nan(flags_ok).any()

    def test_has_inf(self) -> None:
        """has_inf() should detect INF errors."""
        flags = err.from_code(ErrorCode.INF, ErrorLocation.UNKNOWN, 2,
                                      severity=Severity.CRITICAL)
        assert err.has_inf(flags).all()

    def test_has_critical(self) -> None:
        """has_critical() should detect CRITICAL severity."""
        flags_crit = err.from_code(ErrorCode.NAN, ErrorLocation.UNKNOWN, 1,
                                           severity=Severity.CRITICAL)
        flags_warn = err.from_code(ErrorCode.ZERO_OUTPUT, ErrorLocation.UNKNOWN, 1,
                                           severity=Severity.WARN)

        assert err.has_critical(flags_crit)[0]
        assert not err.has_critical(flags_warn)[0]

    def test_max_severity(self) -> None:
        """max_severity() should return highest severity."""
        config = ErrorConfig(num_slots=4)
        flags = err.new_t(1, config=config)
        
        # Add WARN
        code = torch.tensor([ErrorCode.ZERO_OUTPUT], dtype=torch.int64)
        flags = err.push(flags, code, 1, Severity.WARN, config)
        assert err.max_severity(flags, config)[0] == Severity.WARN
        
        # Add ERROR
        code = torch.tensor([ErrorCode.OUT_OF_BOUNDS], dtype=torch.int64)
        flags = err.push(flags, code, 2, Severity.ERROR, config)
        assert err.max_severity(flags, config)[0] == Severity.ERROR
        
        # Add CRITICAL
        code = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code, 3, Severity.CRITICAL, config)
        assert err.max_severity(flags, config)[0] == Severity.CRITICAL

    def test_count_errors(self) -> None:
        """count_errors() should count non-empty slots."""
        config = ErrorConfig(num_slots=8)
        flags = err.new_t(1, config=config)
        assert err.count_errors(flags, config)[0] == 0

        flags = err.push_scalar(flags, ErrorCode.NAN, ErrorLocation.UNKNOWN, 
                                        Severity.CRITICAL, config)
        assert err.count_errors(flags, config)[0] == 1

        flags = err.push_scalar(flags, ErrorCode.INF, ErrorLocation.UNKNOWN,
                                        Severity.CRITICAL, config)
        assert err.count_errors(flags, config)[0] == 2

    def test_has_domain(self) -> None:
        """has_domain() should check error domain."""
        flags = err.from_code(ErrorCode.NAN, ErrorLocation.UNKNOWN, 1,
                                      severity=Severity.CRITICAL)
        
        assert err.has_domain(flags, ErrorDomain.NUMERIC)[0]
        assert not err.has_domain(flags, ErrorDomain.INDEX)[0]


class TestErrorFlagsUnpacking:
    """Tests for error unpacking at Python boundary."""

    def test_unpack_empty(self) -> None:
        """Unpacking OK flags should return empty list."""
        flags = err.new_t(1)
        errors = flags_ns.unpack(flags, 0)
        assert errors == []

    def test_unpack_single_error(self) -> None:
        """Unpacking single error should return one UnpackedError."""
        flags = err.from_code(ErrorCode.NAN, ErrorLocation.register("customer_encoder"), 1,
                                      severity=Severity.CRITICAL)
        errors = flags_ns.unpack(flags, 0)

        assert len(errors) == 1
        assert errors[0].code == ErrorCode.NAN
        assert errors[0].location == ErrorLocation.register("customer_encoder")
        assert errors[0].severity == Severity.CRITICAL
        assert errors[0].code_name == "NAN"
        assert errors[0].location_name == "customer_encoder"
        assert errors[0].severity_name == "CRITICAL"

    def test_unpack_multiple_errors(self) -> None:
        """Unpacking multiple errors should return all of them."""
        config = ErrorConfig(num_slots=8)
        flags = err.new_t(1, config=config)
        flags = err.push_scalar(flags, ErrorCode.NAN, ErrorLocation.register("customer_encoder"),
                                        Severity.CRITICAL, config)
        flags = err.push_scalar(flags, ErrorCode.INF, ErrorLocation.register("group_encoder"),
                                        Severity.CRITICAL, config)
        flags = err.push_scalar(flags, ErrorCode.OUT_OF_BOUNDS, ErrorLocation.register("hash_helpers"),
                                        Severity.ERROR, config)

        errors = flags_ns.unpack(flags, 0, config)
        assert len(errors) == 3

        codes = {e.code for e in errors}
        assert codes == {ErrorCode.NAN, ErrorCode.INF, ErrorCode.OUT_OF_BOUNDS}

    def test_pretty_print(self) -> None:
        """pretty_print() should return readable output."""
        f = err.from_code(ErrorCode.NAN, ErrorLocation.register("customer_encoder"), 1,
                          severity=Severity.CRITICAL)
        output = flags_ns.repr(f)
        
        assert "error" in output.lower() or "Error" in output
        assert "NAN" in output
        # Summary format doesn't include severity, just counts


class TestCompileCompatibility:
    """Tests for torch.compile() compatibility."""

    def test_ok_compiles(self) -> None:
        """err.new_t() should compile."""
        config = CONFIG
        
        @torch.compile(backend="eager")
        def fn(n: int) -> Tensor:
            return err.new_t(n, torch.device('cpu'), config)

        flags = fn(10)
        assert flags.shape == (10, config.num_words)

    def test_push_compiles(self) -> None:
        """err.push() should compile."""
        config = CONFIG
        
        @torch.compile(backend="eager")
        def fn(flags: Tensor, codes: Tensor) -> Tensor:
            return err.push(flags, codes, ErrorLocation.register("customer_encoder"),
                                   Severity.ERROR, config)

        flags = err.new_t(5, config=config)
        codes = torch.tensor([1, 0, 2, 0, 1], dtype=torch.int64)
        result = fn(flags, codes)
        assert result.shape == (5, config.num_words)

    def test_merge_compiles(self) -> None:
        """
        Verify err.merge() compiles.
        
        Expected: No graph breaks, result has errors from both inputs.
        
        Note: Uses small config (4 slots) to avoid slow compile with
        default 256 slots which causes 16k loop iterations.
        """
        # Use small config to avoid slow compile (merge has nested loops)
        config = ErrorConfig(num_slots=4)
        
        @torch.compile(backend="eager")
        def fn(f1: Tensor, f2: Tensor) -> Tensor:
            return err.merge(f1, f2, config=config)

        f1 = err.from_code(ErrorCode.NAN, ErrorLocation.UNKNOWN, 3,
                                   severity=Severity.CRITICAL, config=config)
        f2 = err.from_code(ErrorCode.INF, ErrorLocation.UNKNOWN, 3,
                                   severity=Severity.CRITICAL, config=config)
        result = fn(f1, f2)
        assert err.is_err(result).all()

    def test_checks_compile(self) -> None:
        """Check methods should compile."""
        config = CONFIG
        
        @torch.compile(backend="eager")
        def fn(flags: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            is_err = err.is_err(flags)
            has_nan = err.has_nan(flags, config)
            has_critical = err.has_critical(flags, config)
            count = err.count_errors(flags, config)
            return is_err, has_nan, has_critical, count

        flags = err.from_code(ErrorCode.NAN, ErrorLocation.UNKNOWN, 5,
                                      severity=Severity.CRITICAL)
        is_err, has_nan, has_critical, count = fn(flags)

        assert is_err.all()
        assert has_nan.all()
        assert has_critical.all()
        assert (count == 1).all()

class TestBitLayout:
    """Tests for correct V2 bit packing/unpacking (16-bit slots)."""

    def test_slot_layout(self) -> None:
        """V2 slot should have severity(2) + code(4) + location(10) = 16 bits."""
        assert SLOT_BITS == 16
        assert SLOTS_PER_WORD == 4
        
        # Masks should be correct
        assert SEVERITY_MASK == 0x3
        assert (CODE_SHIFT == 2)
        assert (LOCATION_SHIFT == 6)
        assert SLOT_MASK == 0xFFFF

    def test_slot_0_packing(self) -> None:
        """Slot 0 should have correct V2 bit layout."""
        flags = err.from_code(ErrorCode.NAN, ErrorLocation.register("customer_encoder"), 1,
                                      severity=Severity.CRITICAL)

        val = flags[0, 0].item()
        slot = val & SLOT_MASK
        
        severity = slot & 0x3
        code = (slot >> CODE_SHIFT) & 0xF
        loc = (slot >> LOCATION_SHIFT) & 0x3FF

        assert severity == Severity.CRITICAL
        assert code == ErrorCode.NAN
        assert loc == ErrorLocation.register("customer_encoder")

    def test_multiple_slots(self) -> None:
        """Errors should shift to higher slots within word."""
        config = ErrorConfig(num_slots=4)
        flags = err.new_t(1, config=config)

        flags = err.push_scalar(flags, ErrorCode.NAN, ErrorLocation.register("customer_encoder"),
                                        Severity.CRITICAL, config)
        flags = err.push_scalar(flags, ErrorCode.INF, ErrorLocation.register("group_encoder"),
                                        Severity.CRITICAL, config)
        flags = err.push_scalar(flags, ErrorCode.OUT_OF_BOUNDS, ErrorLocation.register("hash_helpers"),
                                        Severity.ERROR, config)

        errors = flags_ns.unpack(flags, 0, config)
        assert len(errors) == 3

    def test_multi_word_storage(self) -> None:
        """Should be able to use multiple words for more slots."""
        config = ErrorConfig(num_slots=8)  # 8 slots = 2 words
        flags = err.new_t(1, config=config)
        
        assert flags.shape == (1, 2)  # 2 words
        
        # Fill all 8 slots
        for i in range(8):
            code = (i % 15) + 1  # Codes 1-15
            flags = err.push_scalar(flags, code, ErrorLocation.UNKNOWN,
                                            Severity.ERROR, config)

        assert err.count_errors(flags, config)[0] == 8

    def test_all_zeros_is_ok(self) -> None:
        """All zeros (all words = 0) should mean completely OK."""
        flags = torch.zeros(10, 2, dtype=torch.int64)

        assert err.is_ok(flags).all()
        assert not err.is_err(flags).any()
        assert err.count_errors(flags).sum() == 0
