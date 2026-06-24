"""Validation error types for tensor typing. All inherit ``ValidationError``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ValidationError",
    "DimensionMismatchError",
    "DTypeMismatchError",
    "DeviceMismatchError",
    "InvalidParameterError",
    "TypeMismatchError",
    "InvalidReturnTypeError",
]


@dataclass
class ValidationError(Exception):
    """Base validation error; carries a message and free-form context."""

    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({ctx})"
        return self.message


@dataclass
class DimensionMismatchError(ValidationError):
    """A tensor's shape did not match the annotation."""

    expected: Any | None = None
    actual: Any | None = None
    dim_name: str | None = None
    parameter: str | None = None
    function: str | None = None


@dataclass
class DTypeMismatchError(ValidationError):
    """A tensor's dtype did not match the annotation."""

    expected: Any | None = None
    actual: Any | None = None
    parameter: str | None = None
    function: str | None = None


@dataclass
class DeviceMismatchError(ValidationError):
    """A tensor's device did not match the annotation."""

    expected: str | None = None
    actual: str | None = None
    parameter: str | None = None
    function: str | None = None


@dataclass
class InvalidParameterError(ValidationError):
    """A function parameter failed validation."""

    parameter: str | None = None
    function: str | None = None


@dataclass
class TypeMismatchError(ValidationError):
    """A return value did not match its annotation."""

    expected: Any | None = None
    actual: Any | None = None
    function: str | None = None


@dataclass
class InvalidReturnTypeError(ValidationError):
    """A function returned an unexpected type."""

    expected: Any | None = None
    actual: Any | None = None
    function: str | None = None
