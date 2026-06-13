"""Location tracking: map a module's dotted path to a small integer id that
fits the 10-bit location field of an error slot.

Resolution is designed to be ``torch.compile``-safe: ids are registered at
construction time (via :func:`track`/``@tracked``), and inside a compiled
region :func:`resolve_location` performs only attribute + dict lookups that
Dynamo constant-folds into the graph (validated against the contextvar approach,
which Dynamo cannot trace).
"""
from __future__ import annotations

import threading
import weakref
from typing import Union

import torch

__all__ = ["ErrorLocation", "resolve_location", "UNKNOWN"]

UNKNOWN = 0
_MAX_LOCATIONS = 1023  # 10-bit field, id 0 reserved for UNKNOWN


class ErrorLocation:
    """Thread-safe registry mapping dotted module paths to location ids."""

    _lock = threading.Lock()
    _paths: dict[str, int] = {}
    _names: dict[int, str] = {UNKNOWN: "UNKNOWN"}

    @classmethod
    def register(cls, path: str) -> int:
        """Register ``path`` (idempotent); return its id, or ``UNKNOWN`` if full."""
        existing = cls._paths.get(path)
        if existing is not None:
            return existing
        with cls._lock:
            existing = cls._paths.get(path)
            if existing is not None:
                return existing
            if len(cls._paths) >= _MAX_LOCATIONS:
                return UNKNOWN
            loc_id = len(cls._paths) + 1
            cls._paths[path] = loc_id
            cls._names[loc_id] = path
            return loc_id

    @classmethod
    def get(cls, path: str) -> int:
        """Return the id for ``path`` without registering (``UNKNOWN`` if absent)."""
        return cls._paths.get(path, UNKNOWN)

    @classmethod
    def name(cls, loc_id: int) -> str:
        """Return the path for an id, or the id as a string if unregistered."""
        known = cls._names.get(loc_id)
        return known if known is not None else str(loc_id)

    @classmethod
    def reset(cls) -> None:
        """Clear the registry (test helper)."""
        with cls._lock:
            cls._paths.clear()
            cls._names = {UNKNOWN: "UNKNOWN"}


# Cache module -> id (weak so it never holds modules alive). Only used eagerly.
_cache: weakref.WeakKeyDictionary[object, int] = weakref.WeakKeyDictionary()

Where = Union["torch.nn.Module", int, str, None]


def resolve_location(where: Where) -> int:
    """Resolve a location id from a module, int, path string, or ``None``.

    Returns a plain int so the value becomes a compile-time constant. During
    ``torch.compile`` no new ids are registered (registration uses a lock and
    mutates Python state); unknown locations resolve to ``UNKNOWN``.
    """
    if where is None:
        return UNKNOWN
    if isinstance(where, int):
        return where

    compiling = torch.compiler.is_compiling()

    if isinstance(where, str):
        loc = ErrorLocation.get(where)
        if loc == UNKNOWN and not compiling:
            loc = ErrorLocation.register(where)
        return loc

    # nn.Module
    if not compiling:
        cached = _cache.get(where)
        if cached is not None:
            return cached

    path = getattr(where, "_tg_path", None)
    if path is not None:
        loc = ErrorLocation.get(path)
        if loc == UNKNOWN and not compiling:
            loc = ErrorLocation.register(path)
    elif compiling:
        return UNKNOWN
    else:
        loc = ErrorLocation.register(type(where).__name__)

    if not compiling and loc != UNKNOWN:
        _cache[where] = loc
    return loc
