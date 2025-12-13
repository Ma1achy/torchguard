"""
Error codes for torch.compile() compatible error handling.

Codes organized into 4 domains (2 bits) + subcodes (2 bits).

The XOR-hierarchical structure allows:
- Quick domain checks: (code >> 2) & 0x3
- Subcode extraction: code & 0x3
- Domain membership: ErrorCode.in_domain(code, ErrorDomain.NUMERIC)
"""
from __future__ import annotations

from typing import Dict, Optional

__all__ = ['ErrorCode', 'ErrorDomain']


class ErrorDomain:
    """
    Error domains (high 2 bits of 4-bit code).
    
    Domains group related errors for quick filtering.
    Use with ErrorCode.in_domain() or ErrorCode.domain().
    
    Attributes:
        NUMERIC (int): Numerical issues (NaN, Inf, overflow)
        INDEX (int): Indexing issues (OOB, negative)
        QUALITY (int): Output quality (zero, constant, saturated)
        RUNTIME (int): Runtime issues (fallback, clamped)
    """
    NUMERIC: int = 0b00_00
    INDEX: int = 0b01_00
    QUALITY: int = 0b10_00
    RUNTIME: int = 0b11_00
    
    _NAMES: Dict[int, str] = {
        0: "NUMERIC",
        1: "INDEX",
        2: "QUALITY",
        3: "RUNTIME",
    }
    
    @classmethod
    def name(cls, domain_bits: int) -> str:
        """
        Get name for domain bits.
        
        Args:
            domain_bits (int): Domain bits (0-3)
            
        Returns:
            (str): Domain name
        """
        return cls._NAMES.get(domain_bits, f"DOMAIN_{domain_bits}")


class ErrorCode:
    """
    Error codes (4 bits = 16 values max).
    
    code=0 means OK. Any non-zero code is an error.
    
    Layout: 4 bits = 2-bit domain + 2-bit subcode
        Domain: (code >> 2) & 0x3
        Subcode: code & 0x3
    """
    # ─────────────────────────────────────────────────────────────────────────
    # SUCCESS (0)
    # ─────────────────────────────────────────────────────────────────────────
    OK: int = 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # NUMERIC DOMAIN (0b00_xx) - codes 0-3
    # ─────────────────────────────────────────────────────────────────────────
    NAN: int = 0b00_01
    INF: int = 0b00_10
    OVERFLOW: int = 0b00_11
    
    # ─────────────────────────────────────────────────────────────────────────
    # INDEX DOMAIN (0b01_xx) - codes 4-7
    # ─────────────────────────────────────────────────────────────────────────
    OUT_OF_BOUNDS: int = 0b01_01
    NEGATIVE_IDX: int = 0b01_10
    EMPTY_INPUT: int = 0b01_11
    
    # ─────────────────────────────────────────────────────────────────────────
    # QUALITY DOMAIN (0b10_xx) - codes 8-11
    # ─────────────────────────────────────────────────────────────────────────
    ZERO_OUTPUT: int = 0b10_01
    CONSTANT_OUTPUT: int = 0b10_10
    SATURATED: int = 0b10_11
    
    # ─────────────────────────────────────────────────────────────────────────
    # RUNTIME DOMAIN (0b11_xx) - codes 12-15
    # ─────────────────────────────────────────────────────────────────────────
    FALLBACK_VALUE: int = 0b11_01
    VALUE_CLAMPED: int = 0b11_10
    UNKNOWN: int = 0b11_11
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMPATIBILITY ALIASES
    # ─────────────────────────────────────────────────────────────────────────
    UNDERFLOW: int = OVERFLOW
    INVALID_VALUE: int = NEGATIVE_IDX
    UNSTABLE: int = SATURATED
    CLAMPED: int = VALUE_CLAMPED
    SPARSE: int = ZERO_OUTPUT
    
    # Critical codes for quick check
    _CRITICAL: frozenset = frozenset({NAN, INF})
    
    # Names for all codes
    _NAMES: Dict[int, str] = {
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # BASIC METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def name(cls, code: int) -> str:
        """
        Get name for error code.
        
        Args:
            code (int): Error code value (0-15)
            
        Returns:
            (str): Error code name
        """
        return cls._NAMES.get(code, f"CODE_{code}")
    
    @classmethod
    def is_critical(cls, code: int) -> bool:
        """
        Check if error code is critical (NaN or Inf).
        
        Args:
            code (int): Error code value
            
        Returns:
            (bool): True if critical error
        """
        return code in cls._CRITICAL
    
    # ─────────────────────────────────────────────────────────────────────────
    # DOMAIN METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def domain(cls, code: int) -> int:
        """
        Extract domain bits from code (high 2 bits).
        
        Args:
            code (int): Error code value
            
        Returns:
            (int): Domain bits (0-3)
        """
        return (code >> 2) & 0x3
    
    @classmethod
    def subcode(cls, code: int) -> int:
        """
        Extract subcode from code (low 2 bits).
        
        Args:
            code (int): Error code value
            
        Returns:
            (int): Subcode bits (0-3)
        """
        return code & 0x3
    
    @classmethod
    def in_domain(cls, code: int, domain: int) -> bool:
        """
        Check if code belongs to a domain.
        
        Args:
            code (int): Error code value
            domain (int): Domain constant from ErrorDomain
            
        Returns:
            (bool): True if code is in domain
        """
        return cls.domain(code) == (domain >> 2)
    
    @classmethod
    def domain_name(cls, code: int) -> str:
        """
        Get domain name for a code.
        
        Args:
            code (int): Error code value
            
        Returns:
            (str): Domain name
        """
        return ErrorDomain.name(cls.domain(code))
    
    @classmethod
    def default_severity(cls, code: int) -> int:
        """
        Get default severity for an error code.
        
        Inference rules:
        - CRITICAL: NaN, Inf (always fatal for numerics)
        - ERROR: Index issues, overflow, empty input
        - WARN: Quality issues, runtime adjustments
        - OK: Only for code=0
        
        Args:
            code (int): Error code value (0-15)
            
        Returns:
            (int): Default severity level (Severity.OK/WARN/ERROR/CRITICAL)
        """
        from .severity import Severity
        
        if code == cls.OK:
            return Severity.OK
        
        if code in (cls.NAN, cls.INF):
            return Severity.CRITICAL
        
        if code in (cls.OUT_OF_BOUNDS, cls.NEGATIVE_IDX, cls.OVERFLOW, cls.EMPTY_INPUT):
            return Severity.ERROR
        
        if code in (cls.ZERO_OUTPUT, cls.CONSTANT_OUTPUT, cls.SATURATED,
                    cls.FALLBACK_VALUE, cls.VALUE_CLAMPED):
            return Severity.WARN
        
        return Severity.ERROR
