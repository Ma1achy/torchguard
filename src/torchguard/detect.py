"""Detection helpers: check a tensor for NaN/Inf and record it into the flags.

These return a ``GuardedTensor`` and are safe inside compiled regions (pure
tensor ops). Detection is explicit rather than per-op-automatic because the
per-sample reduction has a real cost; call it at the points you care about.

``location`` is an integer id for now; automatic module-tree location tracking
is added in a later phase.
"""
from __future__ import annotations

import torch
from torch import Tensor

from . import flags as F
from .codes import ErrorCode
from .config import ErrorConfig, get_config
from .tensor import GuardedTensor, guard

__all__ = ["flag_nan", "flag_inf", "flag_nan_inf"]


def _split(x: Tensor | GuardedTensor, config: ErrorConfig) -> tuple[Tensor, Tensor]:
    if isinstance(x, GuardedTensor):
        return x._data, x._flags
    g = guard(x, config)
    return g._data, g._flags


def _detect(data: Tensor, predicate: Tensor, code: int, flags: Tensor, location: int,
            config: ErrorConfig) -> Tensor:
    n = data.shape[0]
    mask = predicate.reshape(n, -1).any(dim=1)
    code_t = mask.to(flags.dtype) * code
    return F.push(flags, code_t, location, ErrorCode.default_severity(code), config)


def flag_nan(x: Tensor | GuardedTensor, location: int = 0,
             config: ErrorConfig | None = None) -> GuardedTensor:
    """Record ``ErrorCode.NAN`` for any sample of ``x`` containing a NaN."""
    config = config or get_config()
    data, flags = _split(x, config)
    new_flags = _detect(data, torch.isnan(data), ErrorCode.NAN, flags, location, config)
    return GuardedTensor(data, new_flags)


def flag_inf(x: Tensor | GuardedTensor, location: int = 0,
             config: ErrorConfig | None = None) -> GuardedTensor:
    """Record ``ErrorCode.INF`` for any sample of ``x`` containing an Inf."""
    config = config or get_config()
    data, flags = _split(x, config)
    new_flags = _detect(data, torch.isinf(data), ErrorCode.INF, flags, location, config)
    return GuardedTensor(data, new_flags)


def flag_nan_inf(x: Tensor | GuardedTensor, location: int = 0,
                 config: ErrorConfig | None = None) -> GuardedTensor:
    """Record NaN and Inf errors for ``x`` in a single call (NaN slot first)."""
    config = config or get_config()
    data, flags = _split(x, config)
    flags = _detect(data, torch.isnan(data), ErrorCode.NAN, flags, location, config)
    flags = _detect(data, torch.isinf(data), ErrorCode.INF, flags, location, config)
    return GuardedTensor(data, flags)
