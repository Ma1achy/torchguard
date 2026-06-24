"""Compile/autograd regression tests promoted from the de-risking spike.

These encode the load-bearing guarantees of the rewrite:
  * a GuardedTensor (int flags) survives torch.compile(fullgraph=True) on both
    the eager and inductor backends,
  * gradients flow through the compiled graph to weights and the wrapped input,
  * variable batch sizes work via automatic dynamic shapes,
  * explicit mark_dynamic on a subclass dim is a known, documented xfail.
"""
from __future__ import annotations

import pytest
import torch

from torchguard import GuardedTensor, flag_nan, guard


def _model(gx, w1, w2):
    h = torch.relu(gx @ w1)
    return (h @ w2 * 2.0 + 1.0).sum()


@pytest.fixture
def weights():
    return torch.randn(8, 16, requires_grad=True), torch.randn(16, 4, requires_grad=True)


def test_eager_backward_to_weights_and_input(weights):
    w1, w2 = weights
    x = torch.randn(4, 8, requires_grad=True)
    gx = guard(x)
    loss = _model(gx, w1, w2)
    assert isinstance(loss, GuardedTensor)
    loss.backward()
    assert w1.grad is not None and w2.grad is not None
    assert gx.grad is not None
    assert torch.isfinite(w1.grad).all()


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_compile_forward(backend, weights):
    w1, w2 = weights
    m = torch.compile(_model, fullgraph=True, backend=backend, dynamic=False)
    gx = guard(torch.randn(4, 8))
    loss = m(gx, w1, w2)
    assert isinstance(loss, GuardedTensor)


def test_compile_inductor_backward(weights):
    w1, w2 = weights
    m = torch.compile(_model, fullgraph=True, backend="inductor", dynamic=False)
    x = torch.randn(4, 8, requires_grad=True)
    gx = guard(x)
    loss = m(gx, w1, w2)
    loss.backward()
    assert w1.grad is not None and w2.grad is not None
    assert torch.isfinite(w1.grad).all()


def test_variable_batch_automatic_dynamic(weights):
    w1, w2 = weights
    m = torch.compile(_model, fullgraph=True, backend="inductor")
    for n in (4, 4, 7, 16):
        w1.grad = None
        w2.grad = None
        gx = guard(torch.randn(n, 8, requires_grad=True))
        loss = m(gx, w1, w2)
        loss.backward()
        assert w1.grad is not None


def test_detection_under_compile():
    def f(gx):
        return flag_nan(gx + 0.0, location=1)

    cf = torch.compile(f, fullgraph=True, backend="inductor", dynamic=False)
    data = torch.randn(4, 8)
    data[1] = float("nan")
    out = cf(guard(data))
    assert isinstance(out, GuardedTensor)
    assert torch.equal(out.is_err(), torch.tensor([False, True, False, False]))


@pytest.mark.xfail(reason="mark_dynamic on a subclass dim is a PyTorch-internal limitation", strict=False)
def test_mark_dynamic_on_subclass_is_unsupported(weights):
    w1, w2 = weights
    m = torch.compile(_model, fullgraph=True, backend="inductor")
    x = torch.randn(4, 8, requires_grad=True)
    torch._dynamo.mark_dynamic(x, 0)
    gx = guard(x)
    m(gx, w1, w2)
