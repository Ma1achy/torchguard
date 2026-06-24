"""Unit tests for the bit-packed flag engine."""
from __future__ import annotations

import pytest
import torch

from torchguard import ErrorCode, ErrorConfig, Severity
from torchguard import flags as F


@pytest.fixture(params=[torch.int64, torch.int32])
def config(request):
    return ErrorConfig(num_slots=16, flag_dtype=request.param)


def test_new_is_empty(config):
    f = F.new(8, config)
    assert f.shape == (8, config.num_words)
    assert f.dtype == config.flag_dtype
    assert not bool(F.any_err(f))
    assert torch.equal(F.is_ok(f), torch.ones(8, dtype=torch.bool))


def test_pack_unpack_roundtrip(config):
    f = F.new(1, config)
    code = torch.tensor([ErrorCode.NAN], dtype=config.flag_dtype)
    f = F.push(f, code, location=42, severity=Severity.CRITICAL, config=config)
    decoded = F.unpack(f[0], config)
    assert decoded == [(ErrorCode.NAN, 42, Severity.CRITICAL)]


def test_push_only_where_code_nonzero(config):
    f = F.new(3, config)
    code = torch.tensor([ErrorCode.NAN, 0, ErrorCode.INF], dtype=config.flag_dtype)
    f = F.push(f, code, location=1, severity=Severity.CRITICAL, config=config)
    assert torch.equal(F.is_err(f), torch.tensor([True, False, True]))


def test_push_is_lifo(config):
    f = F.new(1, config)
    one = torch.tensor([ErrorCode.NAN], dtype=config.flag_dtype)
    two = torch.tensor([ErrorCode.INF], dtype=config.flag_dtype)
    f = F.push(f, one, location=1, severity=Severity.CRITICAL, config=config)
    f = F.push(f, two, location=2, severity=Severity.CRITICAL, config=config)
    decoded = F.unpack(f[0], config)
    # newest first
    assert decoded[0] == (ErrorCode.INF, 2, Severity.CRITICAL)
    assert decoded[1] == (ErrorCode.NAN, 1, Severity.CRITICAL)


def test_find_respects_num_slots():
    # num_slots smaller than physical capacity: a slot beyond num_slots must be ignored.
    config = ErrorConfig(num_slots=2, flag_dtype=torch.int64)  # 1 word holds 4 slots
    f = F.new(1, config)
    # Manually place an error in physical slot 3 (beyond num_slots=2) of word 0.
    slot = F.pack_slot_scalar(ErrorCode.NAN, 7, Severity.CRITICAL)
    f[0, 0] = slot << (3 * 16)
    assert not bool(F.find(ErrorCode.NAN, f, config).any())


def test_find_and_has_critical(config):
    f = F.new(2, config)
    code = torch.tensor([ErrorCode.OUT_OF_BOUNDS, ErrorCode.NAN], dtype=config.flag_dtype)
    f = F.push(f, code, location=3, severity=Severity.ERROR, config=config)
    assert torch.equal(F.find(ErrorCode.OUT_OF_BOUNDS, f, config), torch.tensor([True, False]))
    # severity passed was ERROR for both; only sample with explicit CRITICAL would trip has_critical
    crit = torch.tensor([0, ErrorCode.NAN], dtype=config.flag_dtype)
    f2 = F.push(F.new(2, config), crit, location=0, severity=Severity.CRITICAL, config=config)
    assert torch.equal(F.has_critical(f2, config), torch.tensor([False, True]))


def test_count(config):
    f = F.new(1, config)
    f = F.push(f, torch.tensor([ErrorCode.NAN], dtype=config.flag_dtype), 1, Severity.CRITICAL, config)
    f = F.push(f, torch.tensor([ErrorCode.INF], dtype=config.flag_dtype), 2, Severity.CRITICAL, config)
    assert int(F.count(f, config)[0]) == 2


def test_merge_combines_and_compacts(config):
    a = F.push(F.new(1, config), torch.tensor([ErrorCode.NAN], dtype=config.flag_dtype), 1, Severity.CRITICAL, config)
    b = F.push(F.new(1, config), torch.tensor([ErrorCode.INF], dtype=config.flag_dtype), 2, Severity.CRITICAL, config)
    merged = F.merge(a, b, config)
    codes = {c for c, _, _ in F.unpack(merged[0], config)}
    assert codes == {ErrorCode.NAN, ErrorCode.INF}
