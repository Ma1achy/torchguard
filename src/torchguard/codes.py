"""Error codes and domains for packed error flags.

A code is 4 bits: a 2-bit domain (high) plus a 2-bit subcode (low). ``OK`` is 0;
any non-zero code is an error.
"""
from __future__ import annotations

from .severity import Severity

__all__ = ["ErrorCode", "ErrorDomain"]


class ErrorDomain:
    """Error domains (high 2 bits of a 4-bit code), for coarse filtering."""

    NUMERIC: int = 0b00_00
    INDEX: int = 0b01_00
    QUALITY: int = 0b10_00
    RUNTIME: int = 0b11_00

    _NAMES = {0: "NUMERIC", 1: "INDEX", 2: "QUALITY", 3: "RUNTIME"}

    @classmethod
    def name(cls, domain_bits: int) -> str:
        """Return the name for a 2-bit domain value (0-3)."""
        return cls._NAMES.get(domain_bits, f"DOMAIN_{domain_bits}")


class ErrorCode:
    """Error codes (4 bits, 16 values). ``code == 0`` means OK."""

    OK: int = 0

    # NUMERIC domain (0b00_xx)
    NAN: int = 0b00_01
    INF: int = 0b00_10
    OVERFLOW: int = 0b00_11

    # INDEX domain (0b01_xx)
    OUT_OF_BOUNDS: int = 0b01_01
    NEGATIVE_IDX: int = 0b01_10
    EMPTY_INPUT: int = 0b01_11

    # QUALITY domain (0b10_xx)
    ZERO_OUTPUT: int = 0b10_01
    CONSTANT_OUTPUT: int = 0b10_10
    SATURATED: int = 0b10_11

    # RUNTIME domain (0b11_xx)
    FALLBACK_VALUE: int = 0b11_01
    VALUE_CLAMPED: int = 0b11_10
    UNKNOWN: int = 0b11_11

    _CRITICAL = frozenset({NAN, INF})

    _NAMES = {
        0: "OK",
        1: "NAN",
        2: "INF",
        3: "OVERFLOW",
        5: "OUT_OF_BOUNDS",
        6: "NEGATIVE_IDX",
        7: "EMPTY_INPUT",
        9: "ZERO_OUTPUT",
        10: "CONSTANT_OUTPUT",
        11: "SATURATED",
        13: "FALLBACK_VALUE",
        14: "VALUE_CLAMPED",
        15: "UNKNOWN",
    }

    @classmethod
    def name(cls, code: int) -> str:
        """Return the name for an error code (0-15)."""
        return cls._NAMES.get(code, f"CODE_{code}")

    @classmethod
    def is_critical(cls, code: int) -> bool:
        """True if ``code`` is NaN or Inf."""
        return code in cls._CRITICAL

    @classmethod
    def domain(cls, code: int) -> int:
        """Extract the 2-bit domain from a code."""
        return (code >> 2) & 0x3

    @classmethod
    def in_domain(cls, code: int, domain: int) -> bool:
        """True if ``code`` belongs to ``domain`` (an ``ErrorDomain`` value)."""
        return cls.domain(code) == (domain >> 2)

    @classmethod
    def domain_name(cls, code: int) -> str:
        """Return the domain name for a code."""
        return ErrorDomain.name(cls.domain(code))

    @classmethod
    def default_severity(cls, code: int) -> int:
        """Infer a default severity for a code."""
        if code == cls.OK:
            return Severity.OK
        if code in (cls.NAN, cls.INF):
            return Severity.CRITICAL
        if code in (cls.OUT_OF_BOUNDS, cls.NEGATIVE_IDX, cls.OVERFLOW, cls.EMPTY_INPUT):
            return Severity.ERROR
        if code in (
            cls.ZERO_OUTPUT,
            cls.CONSTANT_OUTPUT,
            cls.SATURATED,
            cls.FALLBACK_VALUE,
            cls.VALUE_CLAMPED,
        ):
            return Severity.WARN
        return Severity.ERROR
