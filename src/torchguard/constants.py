"""Bit-layout constants for a 16-bit error slot.

Slot layout (LSB-first), so the smallest fields need the least shifting::

    +------------+----------+----------+
    | bits 15-6  | bits 5-2 | bits 1-0 |
    | location   | code     | severity |
    | (10 bits)  | (4 bits) | (2 bits) |
    +------------+----------+----------+

Several 16-bit slots are packed into each storage word: 4 per ``int64`` word,
2 per ``int32`` word.
"""
from __future__ import annotations

__all__ = [
    "SLOT_BITS",
    "SLOT_MASK",
    "SEVERITY_SHIFT",
    "SEVERITY_MASK",
    "CODE_SHIFT",
    "CODE_MASK",
    "LOCATION_SHIFT",
    "LOCATION_MASK",
]

SLOT_BITS: int = 16
SLOT_MASK: int = 0xFFFF

SEVERITY_SHIFT: int = 0
SEVERITY_MASK: int = 0x3

CODE_SHIFT: int = 2
CODE_MASK: int = 0xF  # value mask after shifting right by CODE_SHIFT

LOCATION_SHIFT: int = 6
LOCATION_MASK: int = 0x3FF  # value mask after shifting right by LOCATION_SHIFT
