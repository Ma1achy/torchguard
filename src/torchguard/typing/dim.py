"""Dimension helpers for tensor annotations.

* ``Dim.attr`` -> an :class:`_AttributeRef` resolved against the instance at
  validation time (e.g. ``Tensor[float32_t, ("N", Dim.hidden_size)]``).
* ``Broadcast`` -> a dimension that may be any size.
"""
from __future__ import annotations

from typing import Any

__all__ = ["Dim", "Broadcast"]


class _AttributeRef:
    """A reference to an instance attribute, resolved to an int dimension."""

    def __init__(self, obj_name: str, attr: str) -> None:
        self.obj_name = obj_name
        self.attr = attr

    def __getattr__(self, name: str) -> _AttributeRef:
        # Support chained access: Dim.config.hidden_size
        return _AttributeRef(f"{self.obj_name}.{self.attr}", name)

    def _path(self) -> list[str]:
        if "." in self.obj_name:
            parts = self.obj_name.split(".")[1:]  # drop leading 'self'
            parts.append(self.attr)
        else:
            parts = [self.attr]
        return parts

    def resolve(self, instance: Any) -> int:
        """Resolve to an int by walking the attribute path on ``instance``.

        Raises ``ValueError`` if ``instance`` is None, ``AttributeError`` if a
        path component is missing, ``TypeError`` if the result is not an int.
        """
        if instance is None:
            raise ValueError(
                f"Cannot resolve '{self}' without an instance "
                f"(use @tensorcheck on an instance method)."
            )
        value: Any = instance
        for part in self._path():
            if not hasattr(value, part):
                raise AttributeError(
                    f"{type(instance).__name__} has no attribute '{part}' "
                    f"while resolving '{self}'"
                )
            value = getattr(value, part)
        if not isinstance(value, int):
            raise TypeError(
                f"Dimension '{self}' resolved to {type(value).__name__}, expected int"
            )
        return value

    def __repr__(self) -> str:
        return f"{self.obj_name}.{self.attr}"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _AttributeRef)
            and self.obj_name == other.obj_name
            and self.attr == other.attr
        )

    def __hash__(self) -> int:
        return hash((self.obj_name, self.attr))


class _DimProxy:
    """``Dim.foo`` captures an attribute reference ``self.foo``."""

    def __getattr__(self, name: str) -> _AttributeRef:
        return _AttributeRef("self", name)

    def __repr__(self) -> str:
        return "Dim"


class _BroadcastMarker:
    """Marker for a broadcast-compatible (any-size) dimension."""

    def __repr__(self) -> str:
        return "*"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _BroadcastMarker)

    def __hash__(self) -> int:
        return hash("*")


Dim = _DimProxy()
Broadcast = _BroadcastMarker()
