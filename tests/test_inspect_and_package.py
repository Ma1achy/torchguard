"""Boundary inspection and public-API surface."""
from __future__ import annotations

import torch

import torchguard as tg
from torchguard import flag_nan_inf, inspect


def test_public_api_surface():
    for name in [
        "GuardedTensor",
        "guard",
        "flag_nan",
        "flag_inf",
        "flag_nan_inf",
        "ErrorCode",
        "Severity",
        "ErrorConfig",
        "get_config",
        "set_config",
    ]:
        assert hasattr(tg, name), name
    assert tg.__version__ == "0.2.0"


def test_summary_and_report():
    data = torch.randn(3, 4)
    data[0] = float("nan")
    data[2] = float("inf")
    gx = flag_nan_inf(data, location=5)
    s = inspect.summary(gx.flags)
    assert s.get("NAN") == 1
    assert s.get("INF") == 1
    text = inspect.report(gx.flags)
    assert "NAN@5" in text and "INF@5" in text
    assert "3 samples" in text
