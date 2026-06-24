"""TorchGuard: per-sample, ``torch.compile``-safe error tracking for PyTorch.

A ``GuardedTensor`` carries a bit-packed, per-sample error-flag channel
alongside its data and propagates it automatically through every operation, so
one bad sample no longer poisons a whole batch inside a compiled graph.

Quick start::

    import torch
    from torchguard import guard, flag_nan_inf

    x = guard(torch.randn(32, 512))   # wrap once at the boundary
    h = layer1(x)                     # flags ride along automatically
    h = flag_nan_inf(h, location=1)   # record NaN/Inf where they appear
    if h.has_err():
        from torchguard import inspect
        print(inspect.report(h.flags))
"""
from __future__ import annotations

from . import flags, inspect, typing
from .codes import ErrorCode, ErrorDomain
from .config import (
    CONFIG,
    AccumulationPolicy,
    Dedupe,
    ErrorConfig,
    Order,
    get_config,
    set_config,
)
from .decorators import track, tracked
from .detect import flag_inf, flag_nan, flag_nan_inf
from .location import ErrorLocation, resolve_location
from .severity import Severity
from .tensor import GuardedTensor, guard
from .tensorcheck import tensorcheck
from .typing import Broadcast, Dim, Tensor

__version__ = "0.2.0"

__all__ = [
    "GuardedTensor",
    "guard",
    "flag_nan",
    "flag_inf",
    "flag_nan_inf",
    "track",
    "tracked",
    "tensorcheck",
    "Tensor",
    "Dim",
    "Broadcast",
    "ErrorLocation",
    "resolve_location",
    "ErrorCode",
    "ErrorDomain",
    "Severity",
    "ErrorConfig",
    "AccumulationPolicy",
    "Order",
    "Dedupe",
    "CONFIG",
    "get_config",
    "set_config",
    "flags",
    "inspect",
    "typing",
    "__version__",
]
