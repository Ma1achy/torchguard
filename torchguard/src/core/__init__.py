"""
Core types and constants for torchguard.

Exports:
- ErrorCode, ErrorDomain: Error classification
- Severity: Error severity levels
- ErrorConfig, AccumulationConfig: Configuration
- CONFIG: Global mutable config object
- get_config: Get the global config
- ErrorLocation: Location registry (from location/)
- Bit layout constants
"""
from .codes import ErrorCode, ErrorDomain
from .severity import Severity
from .config import (
    ErrorConfig,
    AccumulationConfig,
    Priority,
    Order,
    Dedupe,
    CONFIG,
    get_config,
    set_config,
)
from .constants import (
    SLOT_BITS,
    SLOTS_PER_WORD,
    SEVERITY_SHIFT,
    SEVERITY_BITS,
    SEVERITY_MASK,
    CODE_SHIFT,
    CODE_BITS,
    CODE_MASK,
    LOCATION_SHIFT,
    LOCATION_BITS,
    LOCATION_MASK,
    SLOT_MASK,
)
from .location import (
    ErrorLocation,
    LocationTree,
    WeightedLocationTree,
    LocationExtractor,
    extract_locations,
    location_from_stack,
)
from .device_cache import DeviceCache, get_device_cache

__all__ = [
    # Codes
    'ErrorCode',
    'ErrorDomain',
    # Severity
    'Severity',
    # Config
    'ErrorConfig',
    'AccumulationConfig',
    'Priority',
    'Order',
    'Dedupe',
    'CONFIG',
    'get_config',
    'set_config',
    # Constants
    'SLOT_BITS',
    'SLOTS_PER_WORD',
    'SEVERITY_SHIFT',
    'SEVERITY_BITS',
    'SEVERITY_MASK',
    'CODE_SHIFT',
    'CODE_BITS',
    'CODE_MASK',
    'LOCATION_SHIFT',
    'LOCATION_BITS',
    'LOCATION_MASK',
    'SLOT_MASK',
    # Location
    'ErrorLocation',
    'LocationTree',
    'WeightedLocationTree',
    'LocationExtractor',
    'extract_locations',
    'location_from_stack',
    # Device cache
    'DeviceCache',
    'get_device_cache',
]
