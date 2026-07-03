"""Detection and recovery helpers.

All of these return a ``GuardedTensor`` and are safe inside compiled regions.
They record errors by *adding a marker* (a GuardedTensor with zero data and the
new error in its flags) to the input. Going through a real dispatch op keeps the
data's autograd graph intact — gradients flow through a flagged/fixed output.

* ``flag_nan`` / ``flag_inf`` / ``flag_nan_inf`` — detect NaN/Inf.
* ``flag_oob_indices`` — detect out-of-bounds index/embedding lookups.
* ``fix`` — replace bad samples' values with a fallback and record it.

``location`` accepts an ``nn.Module`` (resolved to its tracked path id), an int
id, a path string, or ``None``.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from . import flags as F
from .codes import ErrorCode
from .config import ErrorConfig, get_config
from .location import Where, resolve_location
from .tensor import GuardedTensor, guard

__all__ = ["flag_nan", "flag_inf", "flag_nan_inf", "flag_oob_indices", "fix"]


def _flag_from_mask(gx: GuardedTensor, sample_mask: Tensor, code: int, location: Where,
                    config: ErrorConfig) -> GuardedTensor:
    """Record ``code`` for samples where ``sample_mask`` (shape ``(N,)``) is True.

    Adds an all-zero-data marker carrying the error so the data's autograd graph
    survives; the error merges in via the configured ``dedupe`` policy.
    """
    loc = resolve_location(location)
    code_t = sample_mask.to(gx._flags.dtype) * code
    err = F.push(F.new(gx._data.shape[0], config, gx._data.device), code_t, loc,
                 ErrorCode.default_severity(code), config)
    marker = GuardedTensor(torch.zeros_like(gx._data), err)
    return gx + marker  # type: ignore[return-value]  # dispatch returns a GuardedTensor


def _per_sample(predicate: Tensor) -> Tensor:
    """Reduce a per-element predicate to one bool per sample (handles empty batch)."""
    return predicate if predicate.ndim <= 1 else predicate.flatten(1).any(dim=1)


def flag_nan(x: Tensor | GuardedTensor, location: Where = None,
             config: ErrorConfig | None = None) -> GuardedTensor:
    """Record ``ErrorCode.NAN`` for any sample of ``x`` containing a NaN."""
    config = config or get_config()
    gx = guard(x, config)
    return _flag_from_mask(gx, _per_sample(torch.isnan(gx._data)), ErrorCode.NAN, location, config)


def flag_inf(x: Tensor | GuardedTensor, location: Where = None,
             config: ErrorConfig | None = None) -> GuardedTensor:
    """Record ``ErrorCode.INF`` for any sample of ``x`` containing an Inf."""
    config = config or get_config()
    gx = guard(x, config)
    return _flag_from_mask(gx, _per_sample(torch.isinf(gx._data)), ErrorCode.INF, location, config)


def flag_nan_inf(x: Tensor | GuardedTensor, location: Where = None,
                 config: ErrorConfig | None = None) -> GuardedTensor:
    """Record NaN and Inf errors for ``x`` in a single call."""
    return flag_inf(flag_nan(x, location, config), location, config)


def flag_oob_indices(indices: Tensor | GuardedTensor, num_embeddings: int,
                     location: Where = None,
                     config: ErrorConfig | None = None) -> GuardedTensor:
    """Record ``ErrorCode.OUT_OF_BOUNDS`` for samples with an index outside
    ``[0, num_embeddings)`` (e.g. before an embedding lookup)."""
    config = config or get_config()
    gx = guard(indices, config)
    oob = (gx._data < 0) | (gx._data >= num_embeddings)
    return _flag_from_mask(gx, _per_sample(oob), ErrorCode.OUT_OF_BOUNDS, location, config)


def fix(x: Tensor | GuardedTensor, fallback: float | Tensor = 0.0, location: Where = None,
        codes: Iterable[int] | None = None,
        config: ErrorConfig | None = None) -> GuardedTensor:
    """Replace error samples' values with ``fallback`` and record ``FALLBACK_VALUE``.

    By default every sample that currently carries an error is replaced. Pass
    ``codes`` to restrict replacement to samples carrying one of those error
    codes. Gradients flow through the kept samples (the replacement is a
    ``torch.where``, so it composes with autograd and ``torch.compile``).
    """
    config = config or get_config()
    gx = guard(x, config)
    data = gx._data
    if codes is None:
        bad = F.is_err(gx._flags)
    else:
        bad = torch.zeros(gx._flags.shape[0], dtype=torch.bool, device=gx._flags.device)
        for c in codes:
            bad = bad | F.find(c, gx._flags, config)
    cond = bad if data.ndim <= 1 else bad.reshape(bad.shape[0], *([1] * (data.ndim - 1)))
    fb = torch.as_tensor(fallback, dtype=data.dtype, device=data.device)
    fixed = torch.where(cond, fb, gx)  # GuardedTensor; grad flows to kept samples
    return _flag_from_mask(guard(fixed, config), bad, ErrorCode.FALLBACK_VALUE, location, config)
