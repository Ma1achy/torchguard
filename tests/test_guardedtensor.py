"""Eager behaviour of GuardedTensor: wrapping, propagation, detection."""
from __future__ import annotations

import torch

from torchguard import ErrorCode, GuardedTensor, flag_nan, flag_nan_inf, guard
from torchguard import flags as F


def test_guard_wraps_and_is_idempotent():
    x = torch.randn(4, 8)
    gx = guard(x)
    assert isinstance(gx, GuardedTensor)
    assert not gx.has_err()
    assert guard(gx) is gx


def test_unwrap_returns_plain_tensor():
    gx = guard(torch.randn(4, 8))
    assert not isinstance(gx.unwrap(), GuardedTensor)
    assert torch.equal(gx.unwrap(), gx._data)


def test_flags_propagate_through_ops():
    # mark sample 2 with a NaN error, then run more ops; the flag must survive.
    data = torch.randn(4, 8)
    data[2] = float("nan")
    gx = flag_nan(data, location=1)
    out = torch.relu(gx + 1.0) * 2.0
    assert isinstance(out, GuardedTensor)
    assert torch.equal(out.is_err(), torch.tensor([False, False, True, False]))


def test_detection_locates_per_sample():
    data = torch.randn(3, 5)
    data[0] = float("nan")
    data[1] = float("inf")
    gx = flag_nan_inf(data, location=7)
    assert torch.equal(gx.is_err(), torch.tensor([True, True, False]))
    assert torch.equal(F.find(ErrorCode.NAN, gx.flags), torch.tensor([True, False, False]))
    assert torch.equal(F.find(ErrorCode.INF, gx.flags), torch.tensor([False, True, False]))


def test_binary_op_merges_two_channels():
    a = torch.randn(4, 8)
    a[0] = float("nan")
    b = torch.randn(4, 8)
    b[3] = float("inf")
    ga = flag_nan(a, location=1)
    gb = flag_nan_inf(b, location=2)
    out = ga + gb
    assert isinstance(out, GuardedTensor)
    assert torch.equal(out.is_err(), torch.tensor([True, False, False, True]))


def test_no_false_positives():
    gx = guard(torch.randn(16, 4))
    out = (gx @ torch.randn(4, 4)).relu()
    assert not out.has_err()
