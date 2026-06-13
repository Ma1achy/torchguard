"""Python-boundary inspection of error flags: human-readable summaries.

Do not call these inside a compiled region (they return Python values and would
force a graph break).
"""
from __future__ import annotations

from torch import Tensor

from . import flags as F
from .codes import ErrorCode
from .config import ErrorConfig, get_config

__all__ = ["summary", "report"]


def summary(flags: Tensor, config: ErrorConfig | None = None) -> dict[str, int]:
    """Count errors by code name across the whole batch."""
    config = config or get_config()
    out: dict[str, int] = {}
    for i in range(flags.shape[0]):
        for code, _loc, _sev in F.unpack(flags[i], config):
            name = ErrorCode.name(code)
            out[name] = out.get(name, 0) + 1
    return out


def report(flags: Tensor, config: ErrorConfig | None = None) -> str:
    """Render a one-line human-readable report of all errors in the batch."""
    config = config or get_config()
    n = flags.shape[0]
    parts = []
    total = 0
    for i in range(n):
        errors = F.unpack(flags[i], config)
        total += len(errors)
        for code, loc, _sev in errors:
            parts.append(f"s{i}:{ErrorCode.name(code)}@{loc}")
    head = f"GuardedTensor({n} samples, {total} errors)"
    return head if not parts else f"{head}: " + ", ".join(parts)
