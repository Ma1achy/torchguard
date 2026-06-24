"""``Tensor[...]`` annotation syntax.

Examples::

    x: Tensor[float32_t, ("N", "D")]
    x: Tensor[float32_t, ("N", Dim.hidden_size)]
    x: Tensor[float32_t, (Ellipsis, "seq", "hidden")]
    bias: Tensor[float32_t, (Broadcast, Dim.features)]
"""
from __future__ import annotations

from typing import Any

from .annotation import TensorAnnotation

__all__ = ["Tensor", "TensorType"]


def _norm_dtype(dtype: Any) -> Any:
    return dtype._dtype if hasattr(dtype, "_dtype") else dtype


class TensorType(type):
    """Metaclass providing ``Tensor[dtype, shape, device, requires_grad]``."""

    def __getitem__(cls, params: Any) -> TensorAnnotation:
        if not isinstance(params, tuple):
            if isinstance(params, list):
                return TensorAnnotation(dtype=None, shape=tuple(params))
            return TensorAnnotation(dtype=_norm_dtype(params), shape=None)

        if len(params) > 4:
            raise ValueError(
                "Tensor[...] expects 1-4 parameters "
                f"(dtype, shape, device, requires_grad), got {len(params)}"
            )
        dtype = _norm_dtype(params[0]) if len(params) >= 1 else None
        shape: tuple | None = params[1] if len(params) >= 2 else None
        device = params[2] if len(params) >= 3 else None
        requires_grad = params[3] if len(params) >= 4 else None
        return TensorAnnotation(dtype=dtype, shape=shape, device=device,
                                requires_grad=requires_grad)


class Tensor(metaclass=TensorType):
    """Subscript to build a :class:`TensorAnnotation` for runtime validation."""
