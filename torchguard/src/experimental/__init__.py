"""
TorchGuard Experimental Backend: float64 storage + int64 view.

This backend stores flags as float64 tensors but performs all bit
operations via view(torch.int64). This satisfies AOTAutograd's
assumption that all tensor outputs are differentiable floats.

Requirements:
    - PyTorch >= 2.0 (for view(dtype) support)
    - Recommended: PyTorch >= 2.7 for best torch.compile compatibility

Status:
    EXPERIMENTAL - API mirrors stable backend, no stability guarantees.

Usage:
    from torchguard.src.experimental import err, IF, IS, HAS
    
    @torch.compile(fullgraph=True)
    def forward(x):
        f = err.new(x)
        f = err.push(f, err.NAN, location=42, where=torch.isnan(x).any(-1))
        out, f = IF(IS(err.NAN, f), lambda: fix(out, f)).ELSE(lambda: (out.clone(), f.clone()))
        return out, f
"""
from __future__ import annotations

import torch
from torch import Tensor
from packaging import version

# Version check
_MIN_VERSION = "2.0.0"
if version.parse(torch.__version__) < version.parse(_MIN_VERSION):
    raise RuntimeError(
        f"torchguard.experimental requires torch >= {_MIN_VERSION} "
        f"for view(dtype) support. You have {torch.__version__}"
    )

# Import experimental err namespace
from .err import err

# Re-export control flow primitives (IF, HAS, AND, OR, NOT work with any dtype)
from ..control import IF, HAS, AND, OR, NOT, _ensure_scalar

# Import config
from ..core.config import DEFAULT_CONFIG, ErrorConfig


def IS(code: int, flags: Tensor, *, config: ErrorConfig = DEFAULT_CONFIG) -> Tensor:
    """
    Experimental IS predicate: does any sample have this specific error code?
    
    Uses experimental err.find() which properly handles float64 flags.
    
    Args:
        code (int): Error code integer (e.g. ErrorCode.NAN)
        flags (Tensor): error_t flags tensor (float64), shape (N, num_words)
        config (ErrorConfig): ErrorConfig used for bit layout
    
    Returns:
        (Tensor): 0-D bool Tensor, True if any sample has this error code
    """
    cond = err.find(code, flags, config).any()
    return _ensure_scalar(cond, f"IS({code}, flags)")


__all__ = [
    'err',
    'IF', 'IS', 'HAS', 'AND', 'OR', 'NOT',
]

