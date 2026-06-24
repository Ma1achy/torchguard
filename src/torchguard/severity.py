"""Severity levels for packed error flags (2 bits = 4 levels)."""
from __future__ import annotations

__all__ = ["Severity"]


class Severity:
    """Compile-time severity levels for bit packing.

    Bit values: ``OK`` (empty slot), ``WARN``, ``ERROR``, ``CRITICAL``.
    These are plain integers (not an ``enum``) so they can be used directly
    inside compiled tensor operations without boxing overhead.
    """

    OK: int = 0
    WARN: int = 1
    ERROR: int = 2
    CRITICAL: int = 3

    _NAMES = {0: "OK", 1: "WARN", 2: "ERROR", 3: "CRITICAL"}

    @classmethod
    def name(cls, sev: int) -> str:
        """Return the human-readable name for a severity value."""
        return cls._NAMES.get(sev, f"SEV_{sev}")

    @classmethod
    def is_critical(cls, sev: int) -> bool:
        """True if ``sev`` is ``CRITICAL``."""
        return sev == cls.CRITICAL

    @classmethod
    def is_error_or_worse(cls, sev: int) -> bool:
        """True if ``sev`` is ``ERROR`` or ``CRITICAL``."""
        return sev >= cls.ERROR
