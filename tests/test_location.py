"""Location tracking: registry, module-tree resolution, compile correctness."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchguard as tg
from torchguard import ErrorLocation, flag_nan, guard, resolve_location, track, tracked
from torchguard import flags as F


def test_registry_register_is_idempotent():
    a = ErrorLocation.register("encoder.ffn")
    b = ErrorLocation.register("encoder.ffn")
    assert a == b != 0
    assert ErrorLocation.get("encoder.ffn") == a
    assert ErrorLocation.name(a) == "encoder.ffn"


def test_registry_unknown():
    assert ErrorLocation.get("nope") == 0
    assert ErrorLocation.name(0) == "UNKNOWN"
    assert ErrorLocation.name(999) == "999"  # unregistered -> id as string


def test_resolve_variants():
    assert resolve_location(None) == 0
    assert resolve_location(7) == 7
    loc = resolve_location("layer.norm")  # registers outside compile
    assert loc != 0 and ErrorLocation.get("layer.norm") == loc


def test_track_injects_paths_and_ids():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

    net = track(Net())
    assert net.a._tg_path == "a"
    assert net.b._tg_path == "b"
    assert resolve_location(net.a) == ErrorLocation.get("a")
    assert resolve_location(net.a) != resolve_location(net.b)


def test_tracked_decorator():
    @tracked
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(4, 4)

    net = Net()
    assert net.enc._tg_path == "enc"
    assert resolve_location(net.enc) == ErrorLocation.get("enc")


def test_flag_nan_with_module_location():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return flag_nan(self.lin(x), location=self.lin)

    net = track(Net())
    data = torch.randn(4, 8)
    data[1] = float("nan")
    out = net(guard(data))
    loc = ErrorLocation.get("lin")
    recorded = {locid for _c, locid, _s in F.unpack(out.flags[1])}
    assert recorded == {loc}


class _Sub(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return flag_nan(self.lin(x), location=self)


class _Net(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.a = _Sub(d)
        self.b = _Sub(d)

    def forward(self, x):
        return self.b(self.a(x))


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_locations_bake_correctly_under_compile(backend):
    net = track(_Net())
    loc_a, loc_b = resolve_location(net.a), resolve_location(net.b)
    fn = torch.compile(net, fullgraph=True, backend=backend, dynamic=False)
    data = torch.randn(4, 8)
    data[1] = float("nan")  # propagates through both subs
    out = fn(guard(data))
    recorded = {locid for _c, locid, _s in F.unpack(out.flags[1])}
    assert recorded == {loc_a, loc_b}


def test_report_uses_location_names():
    net = track(_Net())
    data = torch.randn(3, 8)
    data[0] = float("nan")
    out = net(guard(data))
    text = tg.inspect.report(out.flags)
    assert "@a" in text and "@b" in text
