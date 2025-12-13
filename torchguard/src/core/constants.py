"""
Bit layout constants for compile-time error flags.

Slot Layout (16 bits, organized smallest->largest):
+------------+----------+----------+
| 15-6       | 5-2      | 1-0      |
| location   | code     | severity |
| (10)       | (4)      | (2)      |
+------------+----------+----------+

Extraction (efficient - smallest fields need no/less shifting):
  severity = slot & 0x3           # No shift!
  code = (slot >> 2) & 0xF        # Shift by 2
  location = (slot >> 6) & 0x3FF  # Shift by 6
"""

# Word and slot sizes
SLOT_BITS: int = 16
SLOTS_PER_WORD: int = 4  # 64 / 16

# Severity field (bits 1-0)
SEVERITY_SHIFT: int = 0
SEVERITY_BITS: int = 2
SEVERITY_MASK: int = 0x3

# Code field (bits 5-2)
CODE_SHIFT: int = 2
CODE_BITS: int = 4
CODE_MASK: int = 0xF << CODE_SHIFT  # 0x3C

# Location field (bits 15-6)
LOCATION_SHIFT: int = 6
LOCATION_BITS: int = 10
LOCATION_MASK: int = 0x3FF << LOCATION_SHIFT  # 0xFFC0

# Full slot mask
SLOT_MASK: int = 0xFFFF
