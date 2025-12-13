"""
Location tracking for error flags.

Provides:
- ErrorLocation: Registry for mapping module paths to location IDs
- LocationTree: Hierarchical location management with pruning
- WeightedLocationTree: Location tree with error weights
- FX extraction utilities
"""
from .registry import ErrorLocation
from .tree import LocationTree, WeightedLocationTree
from .fx import LocationExtractor, extract_locations, location_from_stack

__all__ = [
    'ErrorLocation',
    'LocationTree',
    'WeightedLocationTree',
    'LocationExtractor',
    'extract_locations',
    'location_from_stack',
]
