"""Decorators/helpers that register a module tree for location tracking.

``track(model)`` (or the ``@tracked`` class decorator) walks the module tree
and records each submodule's dotted path so that ``flag_*`` calls passing
``location=self`` resolve to a precise, compile-time-constant location id.
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, TypeVar

from .location import ErrorLocation

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = ["track", "tracked"]

M = TypeVar("M", bound="nn.Module")


def track(model: M) -> M:
    """Register every submodule's dotted path and stamp it with ``_tg_path``.

    Idempotent and safe to call after construction. Returns ``model`` for
    convenience. Frozen modules that reject attribute assignment are skipped
    (they resolve to their class name instead).
    """
    for path, sub in model.named_modules():
        if path == "":
            continue
        try:
            sub._tg_path = path
        except Exception:  # pragma: no cover - frozen/locked modules
            pass
        ErrorLocation.register(path)
    return model


def tracked(cls: type) -> type:
    """Class decorator: ``track(self)`` after ``__init__`` for an ``nn.Module``."""
    orig_init = cls.__init__  # type: ignore[misc]

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        orig_init(self, *args, **kwargs)
        track(self)

    cls.__init__ = __init__  # type: ignore[method-assign, misc]
    return cls
