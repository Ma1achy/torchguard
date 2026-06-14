"""Detection helpers: check a tensor for NaN/Inf and record it into the flags.

These return a ``GuardedTensor`` and are safe inside compiled regions (pure
tensor ops). Detection is explicit rather than per-op-automatic because the
per-sample reduction has a real cost; call it at the points you care about.

Detection adds the error by *adding a marker* (a GuardedTensor with zero data
and the new error in its flags) to the input. Going through a real dispatch op
keeps the data's autograd graph intact — gradients flow through a flagged
output. The error is combined via :func:`torchguard.flags.merge`, which honors
the configured ``dedupe`` policy.

``location`` accepts an ``nn.Module`` (resolved to its tracked path id), an int
id, a path string, or ``None``.
"""
from __future__ import annotations

import torch
from torch import Tensor

from . import flags as F
from .codes import ErrorCode
from .config import ErrorConfig, get_config
from .location import Where, resolve_location
from .tensor import GuardedTensor, guard

__all__ = ["flag_nan", "flag_inf", "flag_nan_inf"]


def _flag(x: Tensor | GuardedTensor, predicate: Tensor, code: int, location: Where,
          config: ErrorConfig | None) -> GuardedTensor:
    config = config or get_config()
    loc = resolve_location(location)
    gx = guard(x, config)
    data = gx._data
    # one bool per sample (flatten handles >1 feature dims and an empty batch)
    mask = predicate if predicate.ndim <= 1 else predicate.flatten(1).any(dim=1)
    code_t = mask.to(gx._flags.dtype) * code
    err = F.push(F.new(data.shape[0], config, data.device), code_t, loc,
                 ErrorCode.default_severity(code), config)
    # Add an all-zero-data marker carrying the new error: a dispatch op, so the
    # data's autograd graph is preserved while flags merge in.
    marker = GuardedTensor(torch.zeros_like(data), err)
    return gx + marker  # type: ignore[return-value]  # dispatch returns a GuardedTensor


def flag_nan(x: Tensor | GuardedTensor, location: Where = None,
             config: ErrorConfig | None = None) -> GuardedTensor:
    """Record ``ErrorCode.NAN`` for any sample of ``x`` containing a NaN."""
    return _flag(x, torch.isnan(guard(x, config)._data), ErrorCode.NAN, location, config)


def flag_inf(x: Tensor | GuardedTensor, location: Where = None,
             config: ErrorConfig | None = None) -> GuardedTensor:
    """Record ``ErrorCode.INF`` for any sample of ``x`` containing an Inf."""
    return _flag(x, torch.isinf(guard(x, config)._data), ErrorCode.INF, location, config)


def flag_nan_inf(x: Tensor | GuardedTensor, location: Where = None,
                 config: ErrorConfig | None = None) -> GuardedTensor:
    """Record NaN and Inf errors for ``x`` in a single call."""
    return flag_inf(flag_nan(x, location, config), location, config)
