"""Edge cases for @tensorcheck and location resolution.

No `from __future__ import annotations` (the Tensor[...] annotations must stay
real runtime objects for @tensorcheck to read them).
"""
import pytest
import torch
import torch.nn as nn

from torchguard import (
    ErrorCode,
    ErrorLocation,
    GuardedTensor,
    flag_nan,
    guard,
    resolve_location,
    tensorcheck,
    tracked,
)
from torchguard import flags as F
from torchguard.typing import DimensionMismatchError, Tensor, bool_t, float32_t


def test_optional_param_none_is_skipped():
    @tensorcheck
    def f(x: Tensor[float32_t, ("N", 4)], mask: Tensor[bool_t, ("N",)] | None = None):
        return guard(x)

    assert not f(torch.randn(3, 4)).has_err()  # mask omitted -> fine
    f(torch.randn(3, 4), torch.ones(3, dtype=torch.bool))  # mask provided & valid


def test_cross_input_named_dim_consistency():
    @tensorcheck
    def g(x: Tensor[float32_t, ("N", 4)], y: Tensor[float32_t, ("N", 2)]):
        return guard(x)

    g(torch.randn(3, 4), torch.randn(3, 2))  # N=3 on both -> ok
    with pytest.raises(DimensionMismatchError):
        g(torch.randn(3, 4), torch.randn(5, 2))  # N mismatch across inputs


def test_explicit_auto_detect_codes():
    @tensorcheck(auto_detect={ErrorCode.NAN})
    def h(x: Tensor[float32_t, ("N", 4)]):
        return guard(x)

    d = torch.randn(2, 4)
    d[0, 0] = float("nan")
    d[1, 0] = float("inf")
    out = h(d)
    assert F.find(ErrorCode.NAN, out.flags).tolist() == [True, False]
    assert F.find(ErrorCode.INF, out.flags).tolist() == [False, False]  # INF not requested


def test_non_tensor_return_is_passthrough():
    @tensorcheck
    def k(x: Tensor[float32_t, ("N", 4)]) -> int:
        return 42

    assert k(torch.randn(3, 4)) == 42


def test_tracked_and_tensorcheck_stack():
    @tracked
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        @tensorcheck
        def forward(self, x: Tensor[float32_t, ("N", 4)]) -> Tensor[float32_t, ("N", 4)]:
            return self.lin(x)

    m = M()
    d = torch.randn(3, 4)
    d[1] = float("nan")
    out = m(guard(d))
    assert isinstance(out, GuardedTensor)
    assert out.is_err().tolist() == [False, True, False]


def test_resolve_unregistered_under_compile_is_unknown():
    class Plain(nn.Module):
        def forward(self, x):
            return flag_nan(x, location=self)  # self never tracked

    cf = torch.compile(Plain(), fullgraph=True, backend="eager", dynamic=False)
    d = torch.full((4, 4), float("nan"))
    out = cf(guard(d))
    locs = {loc for _c, loc, _s in F.unpack(out.flags[0])}
    assert locs == {0}  # UNKNOWN under compile (no registration mid-trace)


def test_class_name_fallback_eager():
    class Widget(nn.Module):
        pass

    loc = resolve_location(Widget())
    assert loc != 0
    assert ErrorLocation.name(loc) == "Widget"
