"""
Error location registry for compile-time error handling.

10-bit location codes (0-1023), with auto-registration from module paths.

Thread-safe: Uses lock for multi-worker dataloader compatibility.
"""
from __future__ import annotations

import threading
import warnings
from typing import Dict


class ErrorLocation:
    """
    Component location codes for error tracking.
    
    Layout: 10 bits = 1024 locations max
    
    Reserved:
        - 0: UNKNOWN (reserved for unknown/unregistered locations)
        - 1-1023: Auto-registered from module paths or manual registration
    
    Thread-safe: Uses lock for multi-worker dataloader compatibility.
    
    Usage:
        # Register a location (idempotent)
        loc_id = ErrorLocation.register("encoder.layer0.attention")
        
        # Use in error tracking
        flags = ErrorFlags.push(flags, code, location=loc_id, ...)
        
        # Get name for display
        name = ErrorLocation.name(loc_id)  # "encoder.layer0.attention"
    """
    # ─────────────────────────────────────────────────────────────────────────
    # RESERVED LOCATION
    # ─────────────────────────────────────────────────────────────────────────
    
    UNKNOWN: int = 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # AUTO-REGISTRATION REGISTRY (thread-safe)
    # ─────────────────────────────────────────────────────────────────────────
    
    _lock: threading.Lock = threading.Lock()
    _next_id: int = 1
    _path_to_id: Dict[str, int] = {}
    _id_to_path: Dict[int, str] = {0: "Unknown"}
    
    # Maximum location ID (10 bits = 1024 values)
    MAX_LOCATION_ID: int = 1023
    
    # ─────────────────────────────────────────────────────────────────────────
    # REGISTRATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def register(cls, path: str) -> int:
        """
        Register a module path and return its location ID.
        
        Idempotent - returns existing ID if already registered.
        Thread-safe for multi-worker dataloaders.
        
        Args:
            path (str): Module path string (e.g., "encoder.layers.0.conv")
            
        Returns:
            (int): Location ID. Returns UNKNOWN if registry is full.
        """
        if path in cls._path_to_id:
            return cls._path_to_id[path]
        
        with cls._lock:
            if path in cls._path_to_id:
                return cls._path_to_id[path]
            
            if cls._next_id > cls.MAX_LOCATION_ID:
                warnings.warn(f"Location registry full (>{cls.MAX_LOCATION_ID}), using UNKNOWN for: {path}")
                return cls.UNKNOWN
            
            loc_id = cls._next_id
            cls._next_id += 1
            cls._path_to_id[path] = loc_id
            cls._id_to_path[loc_id] = path
            return loc_id
    
    @classmethod
    def get(cls, path: str) -> int:
        """
        Get location ID for path without registering.
        
        Args:
            path (str): Module path string
            
        Returns:
            (int): Location ID if registered, UNKNOWN otherwise.
        """
        return cls._path_to_id.get(path, cls.UNKNOWN)
    
    @classmethod
    def is_registered(cls, path: str) -> bool:
        """
        Check if path is registered.
        
        Args:
            path (str): Module path string
            
        Returns:
            (bool): True if path is registered
        """
        return path in cls._path_to_id
    
    @classmethod
    def count(cls) -> int:
        """
        Return number of registered locations.
        
        Returns:
            (int): Total count of registered locations
        """
        return cls._next_id
    
    # ─────────────────────────────────────────────────────────────────────────
    # NAME RESOLUTION
    # ─────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def name(cls, loc: int) -> str:
        """
        Get human-readable name for location ID.
        
        Args:
            loc (int): Location code value
            
        Returns:
            (str): Human-readable name string
        """
        if loc in cls._id_to_path:
            return cls._id_to_path[loc]
        return f"Location_{loc}"
    
    @classmethod
    def path(cls, loc: int) -> str:
        """
        Get the module path for a registered location.
        
        Args:
            loc (int): Location code value
            
        Returns:
            (str): Module path string, or empty string if not registered.
        """
        return cls._id_to_path.get(loc, "")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TESTING / RESET
    # ─────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def reset(cls) -> None:
        """Reset all registered locations (for testing). Thread-safe."""
        with cls._lock:
            cls._next_id = 1
            cls._path_to_id.clear()
            cls._id_to_path.clear()
            cls._id_to_path[0] = "Unknown"
    
    @classmethod
    def get_all_paths(cls) -> Dict[str, int]:
        """
        Get all registered paths and their IDs.
        
        Returns:
            (Dict[str, int]): Dict mapping path -> location ID
        """
        return dict(cls._path_to_id)
    
    @classmethod
    def get_all_ids(cls) -> Dict[int, str]:
        """
        Get all registered IDs and their paths.
        
        Returns:
            (Dict[int, str]): Dict mapping location ID -> path/name
        """
        return dict(cls._id_to_path)
