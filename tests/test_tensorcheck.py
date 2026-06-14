"""@tensorcheck: validation (raises eagerly) + NaN/Inf auto-flagging.

No `from __future__ import annotations` here on purpose: @tensorcheck needs the
Tensor[...] annotations to be real runtime objects, not strings.
"""
import pytest
import torch
import torch.nn as nn

from torchguard import ErrorCode, GuardedTensor, flags, guard, tensorcheck, track
from torchguard.typing import DTypeMismatchError, Tensor, float32_t


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    @tensorcheck
    def forward(self, x: Tensor[float32_t, ("N", 8)]) -> Tensor[float32_t, ("N", 8)]:
        return self.lin(x)


def test_validation_passes_on_correct_shapes():
    net = track(Net())
    out = net(guard(torch.randn(4, 8)))
    assert isinstance(out, GuardedTensor)
    assert not out.has_err()


def test_validation_raises_on_dtype_mismatch():
    net = track(Net())
    with pytest.raises(DTypeMismatchError):
        net(torch.zeros(4, 8, dtype=torch.float64))


def test_validation_raises_on_shape_mismatch():
    net = track(Net())
    with pytest.raises(Exception):
        net(torch.randn(4, 9))  # 9 != 8


def test_auto_flag_records_nan_at_module_location():
    net = track(Net())
    x = torch.randn(4, 8)
    x[1] = float("nan")
    out = net(guard(x))
    assert torch.equal(out.is_err(), torch.tensor([False, True, False, False]))
    assert torch.equal(flags.find(ErrorCode.NAN, out.flags), torch.tensor([False, True, False, False]))


def test_auto_detect_false_is_noop():
    class Clean(nn.Module):
        @tensorcheck(auto_detect=False)
        def forward(self, x: Tensor[float32_t, ("N", 8)]):
            return guard(x)

    net = Clean()
    x = torch.randn(4, 8)
    x[0] = float("nan")
    out = net(x)
    assert not out.has_err()  # detection disabled


def test_tensorcheck_on_plain_function():
    @tensorcheck
    def f(x: Tensor[float32_t, ("N", 4)]) -> Tensor[float32_t, ("N", 4)]:
        return guard(x) * 2.0

    out = f(torch.randn(3, 4))
    assert isinstance(out, GuardedTensor)


def test_compiles_fullgraph():
    @tensorcheck
    def run(x: Tensor[float32_t, ("N", 8)]) -> Tensor[float32_t, ("N", 8)]:
        return guard(x) + 1.0

    compiled = torch.compile(run, fullgraph=True, backend="inductor", dynamic=False)
    out = compiled(torch.randn(4, 8))
    assert isinstance(out, GuardedTensor)


def test_tensorcheck_rejects_class():
    with pytest.raises(TypeError):
        tensorcheck(int)
