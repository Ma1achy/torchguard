"""Shared fixtures.

Resets global config and the Dynamo cache around every test, pinning the
in-process state-hygiene issue (stale grads + repeated compiles) noted as risk
R3 in the rewrite plan.
"""
from __future__ import annotations

import pytest
import torch

import torchguard as tg


@pytest.fixture(autouse=True)
def _reset_state():
    tg.set_config(tg.ErrorConfig())
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
