"""End-to-end / integration tests across the whole stack."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchguard import GuardedTensor, flag_nan_inf, guard, track
from torchguard import flags as F


class Encoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        return flag_nan_inf(torch.relu(self.proj(x)), location=self)


class Decoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.out = nn.Linear(d, d)

    def forward(self, x):
        return flag_nan_inf(self.out(x), location=self)


class Net(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.encoder = Encoder(d)
        self.decoder = Decoder(d)

    def forward(self, x):
        return self.decoder(self.encoder(x))


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_compiled_training_loop_updates_weights(backend):
    net = track(Net())
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    fwd = torch.compile(net, fullgraph=True, backend=backend)
    before = net.decoder.out.weight.detach().clone()
    target = torch.zeros(16, 8)
    for _ in range(3):
        opt.zero_grad()
        out = fwd(guard(torch.randn(16, 8)))
        loss = ((out - target) ** 2).mean()      # stays a GuardedTensor
        assert isinstance(loss, GuardedTensor)
        loss.backward()                            # backward on the GuardedTensor
        assert torch.isfinite(net.decoder.out.weight.grad).all()
        opt.step()
    assert not torch.equal(before, net.decoder.out.weight)  # training progressed


def test_error_attribution_to_submodule():
    net = track(Net())
    from torchguard import resolve_location
    loc_enc = resolve_location(net.encoder)
    loc_dec = resolve_location(net.decoder)
    x = torch.randn(4, 8)
    x[2] = float("nan")  # NaN enters at the encoder and propagates to the decoder
    out = net(guard(x))
    recorded = {loc for _c, loc, _s in F.unpack(out.flags[2])}
    assert recorded == {loc_enc, loc_dec}
    assert out.is_err().tolist() == [False, False, True, False]


def test_inference_drop_bad_samples():
    net = track(Net())
    x = torch.randn(10, 8)
    x[3] = x[7] = float("nan")
    with torch.no_grad():
        out = net(guard(x))
    clean = out.take_ok()
    assert clean.unwrap().shape[0] == 8
    assert not clean.has_err()
    assert torch.isfinite(clean.unwrap()).all()


def test_fallback_replacement_is_finite():
    net = track(Net())
    x = torch.randn(6, 8)
    x[1] = float("inf")  # propagates to a non-finite output for sample 1
    out = net(guard(x))
    assert out.is_err().tolist() == [False, True, False, False, False, False]
    # replace error samples with a finite fallback for downstream use
    safe = torch.where(out.is_ok().unsqueeze(1), out, torch.zeros_like(out))
    assert torch.isfinite(safe.unwrap()).all()


def test_backward_through_flagged_output():
    # Regression: flag_* must not break the autograd graph (clean data -> finite grad).
    net = track(Net())
    out = net(guard(torch.randn(8, 8)))  # clean inputs, but flagged inside forward
    loss = (out ** 2).mean()
    loss.backward()
    assert net.encoder.proj.weight.grad is not None
    assert torch.isfinite(net.encoder.proj.weight.grad).all()


def test_model_state_dict_roundtrip(tmp_path):
    net = track(Net())
    x = guard(torch.randn(4, 8))
    out_before = net(x).unwrap()
    path = tmp_path / "net.pt"
    torch.save(net.state_dict(), path)
    net2 = track(Net())
    net2.load_state_dict(torch.load(path))
    out_after = net2(guard(x.unwrap())).unwrap()
    assert torch.allclose(out_before, out_after)


def test_large_batch_attribution():
    net = track(Net())
    x = torch.randn(1000, 8)
    bad = torch.tensor([0, 500, 999])
    x[bad] = float("nan")
    out = net(guard(x))
    assert out.is_err().sum().item() == 3
    assert out.is_err()[bad].all()
