"""
TorchGuard development package wrapper.

Re-exports everything from torchguard.torchguard for local development use.
When installed via pip, this wrapper is not needed.

This module also provides:
- torchguard.typing: Re-exports from torchguard.torchguard.src.typing
- torchguard.experimental: Re-exports from torchguard.torchguard.src.experimental
"""
import sys

# First, import the inner package
from .torchguard import *
from .torchguard import __all__, __version__

# Import and alias the submodules (renamed to _typing_bridge to avoid shadowing stdlib typing)
from .torchguard import _typing_bridge as _typing
from .torchguard import _experimental_bridge as _experimental

# Register aliases in sys.modules for proper import resolution
sys.modules['torchguard.typing'] = _typing
sys.modules['torchguard.experimental'] = _experimental

# Re-export as module attributes
typing = _typing
experimental = _experimental
