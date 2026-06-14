"""Unit tests for error codes, domains, severity, and config validation."""
from __future__ import annotations

import pytest
import torch

from torchguard import ErrorCode, ErrorConfig, ErrorDomain, Severity


def test_code_names_and_critical():
    assert ErrorCode.name(ErrorCode.NAN) == "NAN"
    assert ErrorCode.name(0) == "OK"
    assert ErrorCode.is_critical(ErrorCode.NAN)
    assert ErrorCode.is_critical(ErrorCode.INF)
    assert not ErrorCode.is_critical(ErrorCode.ZERO_OUTPUT)


def test_domains():
    assert ErrorCode.in_domain(ErrorCode.NAN, ErrorDomain.NUMERIC)
    assert ErrorCode.in_domain(ErrorCode.OUT_OF_BOUNDS, ErrorDomain.INDEX)
    assert not ErrorCode.in_domain(ErrorCode.NAN, ErrorDomain.INDEX)
    assert ErrorCode.domain_name(ErrorCode.SATURATED) == "QUALITY"
    assert ErrorDomain.name(3) == "RUNTIME"


def test_default_severity():
    assert ErrorCode.default_severity(ErrorCode.OK) == Severity.OK
    assert ErrorCode.default_severity(ErrorCode.NAN) == Severity.CRITICAL
    assert ErrorCode.default_severity(ErrorCode.OUT_OF_BOUNDS) == Severity.ERROR
    assert ErrorCode.default_severity(ErrorCode.ZERO_OUTPUT) == Severity.WARN


def test_all_codes_fit_4_bits():
    for name in dir(ErrorCode):
        val = getattr(ErrorCode, name)
        if isinstance(val, int):
            assert 0 <= val <= 15


def test_severity_helpers():
    assert Severity.name(Severity.CRITICAL) == "CRITICAL"
    assert Severity.is_critical(Severity.CRITICAL)
    assert Severity.is_error_or_worse(Severity.ERROR)
    assert not Severity.is_error_or_worse(Severity.WARN)


def test_config_validation():
    with pytest.raises(ValueError):
        ErrorConfig(num_slots=0)
    with pytest.raises(ValueError):
        ErrorConfig(flag_dtype=torch.float32)  # only int64/int32 supported


def test_config_num_words():
    assert ErrorConfig(num_slots=16, flag_dtype=torch.int64).num_words == 4
    assert ErrorConfig(num_slots=16, flag_dtype=torch.int32).num_words == 8
    assert ErrorConfig(num_slots=1).num_words == 1
