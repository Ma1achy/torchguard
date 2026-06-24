"""Minimal ``Result`` type (``Ok``/``Err``) used by validation.

A lightweight, dependency-free Result for returning validation outcomes without
raising. ``unwrap``/``unwrap_err`` are typed ``NoReturn`` on the failing side.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, NoReturn, TypeVar

__all__ = ["Ok", "Err", "Result"]

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Success branch carrying a value."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    @property
    def ok_value(self) -> T:
        return self.value

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def unwrap_err(self) -> NoReturn:
        raise ValueError(f"called unwrap_err on Ok({self.value!r})")

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[object], object]) -> Ok[T]:
        return self

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return fn(self.value)


@dataclass(frozen=True)
class Err(Generic[E]):
    """Failure branch carrying an error."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    @property
    def err_value(self) -> E:
        return self.error

    def unwrap(self) -> NoReturn:
        if isinstance(self.error, BaseException):
            raise self.error
        raise ValueError(f"called unwrap on Err({self.error!r})")

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_err(self) -> E:
        return self.error

    def map(self, fn: Callable[[object], object]) -> Err[E]:
        return self

    def map_err(self, fn: Callable[[E], U]) -> Err[U]:
        return Err(fn(self.error))

    def and_then(self, fn: Callable[[object], object]) -> Err[E]:
        return self


Result = Ok[T] | Err[E]
