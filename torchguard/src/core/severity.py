"""
Compile-time severity levels for bit packing.

These are integer constants (not enums) for use in compiled tensor
operations where enum overhead is not acceptable.
"""
from __future__ import annotations

from typing import Dict

__all__ = ['Severity']


class Severity:
    """
    Compile-time severity levels for bit packing (2 bits = 4 levels).
    
    Used in compiled tensor operations where enum overhead is not acceptable.
    
    Bit values:
        0 = OK (empty slot)
        1 = WARN (warning - may indicate issues)
        2 = ERROR (error - something went wrong)
        3 = CRITICAL (critical - computation invalid, NaN/Inf)
    """
    OK: int = 0
    WARN: int = 1
    ERROR: int = 2
    CRITICAL: int = 3
    
    _NAMES: Dict[int, str] = {0: "OK", 1: "WARN", 2: "ERROR", 3: "CRITICAL"}
    
    @classmethod
    def name(cls, sev: int) -> str:
        """
        Get name for severity value.
        
        Args:
            sev (int): Severity value (0-3)
            
        Returns:
            (str): Severity name
        """
        return cls._NAMES.get(sev, f"SEV_{sev}")
    
    @classmethod
    def is_critical(cls, sev: int) -> bool:
        """Check if severity is critical."""
        return sev == cls.CRITICAL
    
    @classmethod
    def is_error_or_worse(cls, sev: int) -> bool:
        """Check if severity is error or critical."""
        return sev >= cls.ERROR
    
    @classmethod
    def is_warn_or_worse(cls, sev: int) -> bool:
        """Check if severity is warning or worse."""
        return sev >= cls.WARN
