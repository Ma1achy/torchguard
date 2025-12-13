"""
FX-based auto-location extraction for compile-time error tracking.

Provides tools to automatically register all module paths in a model
as error locations using torch.fx tracing.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict

from .registry import ErrorLocation

if TYPE_CHECKING:
    import torch.nn as nn

# Check for torch.fx availability
try:
    import torch
    import torch.fx as fx
    _HAS_FX = True
except ImportError:
    _HAS_FX = False


if _HAS_FX:
    class LocationExtractor(fx.Interpreter):
        """
        FX Interpreter that extracts module paths from traced graphs.
        
        Run this during tracing to auto-register locations for all
        submodules in a model.
        
        Usage:
            extractor = LocationExtractor(model)
            extractor.run(example_input)
            locations = extractor.locations  # Dict[path, location_id]
        
        Attributes:
            locations (Dict[str, int]): Extracted module paths and their location IDs
        """
        
        def __init__(self, module: nn.Module) -> None:
            """
            Initialize the extractor.
            
            Args:
                module (nn.Module): PyTorch module to trace
            """
            gm = fx.symbolic_trace(module)
            super().__init__(gm)
            self.locations: Dict[str, int] = {}
        
        def call_module(self, target: str, args: tuple, kwargs: Dict[str, Any]) -> Any:
            """
            Called for each module call in the graph.
            
            Registers the module path as a location.
            
            Args:
                target (str): Module path (e.g., "encoder.layers.0.conv")
                args (tuple): Positional arguments
                kwargs (Dict[str, Any]): Keyword arguments
            
            Returns:
                (Any): Result of the module call
            """
            loc_id = ErrorLocation.register(target)
            self.locations[target] = loc_id
            return super().call_module(target, args, kwargs)


def extract_locations(module: nn.Module, example_input: Any = None) -> Dict[str, int]:
    """
    Extract and register all submodule paths as locations.
    Also injects _fx_path attribute into each submodule.
    
    If example_input is provided, uses FX tracing for more precise locations.
    Otherwise, falls back to named_modules() enumeration.
    
    Args:
        module (nn.Module): PyTorch module to extract from
        example_input (Any): Example input tensor(s) for FX tracing (optional)
        
    Returns:
        (Dict[str, int]): Dict mapping module path -> location ID
        
    Example:
        # Without FX tracing (simpler, always works)
        locations = extract_locations(model)
        
        # With FX tracing (more precise, may fail on dynamic control flow)
        locations = extract_locations(model, dummy_input)
    """
    locations: Dict[str, int] = {}
    
    for name, child in module.named_modules():
        if name:
            try:
                child._fx_path = name
            except AttributeError:
                pass
            loc_id = ErrorLocation.register(name)
            locations[name] = loc_id
    
    if example_input is not None and _HAS_FX:
        try:
            extractor = LocationExtractor(module)
            extractor.run(example_input)
            locations.update(extractor.locations)
        except Exception as e:
            warnings.warn(f"FX tracing failed: {e}. Using named_modules() only.")
    
    return locations


def location_from_stack(nn_module_stack: Dict[str, tuple]) -> int:
    """
    Extract location ID from nn_module_stack provided by torch.compile.
    
    During torch.compile, operations have access to nn_module_stack which
    maps module paths to (class_name, class) tuples. This function extracts
    the deepest (most specific) module path and registers it.
    
    Args:
        nn_module_stack (Dict[str, tuple]): Dict like {
            'model': ('Model', <class>),
            'model.encoder': ('Encoder', <class>),
            'model.encoder.conv': ('Conv2d', <class>),
        }
        
    Returns:
        (int): Location ID for the deepest module path
    """
    if not nn_module_stack:
        return ErrorLocation.UNKNOWN
    
    path = list(nn_module_stack.keys())[-1]
    return ErrorLocation.register(path)
