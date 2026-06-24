"""dtype aliases for tensor annotations, e.g. ``Tensor[float32_t, ("N", "D")]``.

Each alias is a small object exposing ``._dtype`` (the underlying ``torch.dtype``)
which :class:`~torchguard.typing.tensor.TensorType` reads when building an
annotation.
"""
from __future__ import annotations

import torch

__all__ = [
    "float16_t",
    "float32_t",
    "float64_t",
    "bfloat16_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "bool_t",
    "complex64_t",
    "complex128_t",
    "error_t",
    "PYTHON_TYPE_TO_TORCH_DTYPE",
]


class _DTypeAlias:
    """A named handle for a ``torch.dtype`` usable inside ``Tensor[...]``."""

    __slots__ = ("_dtype", "_name")

    def __init__(self, dtype: torch.dtype, name: str) -> None:
        self._dtype = dtype
        self._name = name

    def __repr__(self) -> str:
        return self._name


float16_t = _DTypeAlias(torch.float16, "float16_t")
float32_t = _DTypeAlias(torch.float32, "float32_t")
float64_t = _DTypeAlias(torch.float64, "float64_t")
bfloat16_t = _DTypeAlias(torch.bfloat16, "bfloat16_t")
int8_t = _DTypeAlias(torch.int8, "int8_t")
int16_t = _DTypeAlias(torch.int16, "int16_t")
int32_t = _DTypeAlias(torch.int32, "int32_t")
int64_t = _DTypeAlias(torch.int64, "int64_t")
uint8_t = _DTypeAlias(torch.uint8, "uint8_t")
bool_t = _DTypeAlias(torch.bool, "bool_t")
complex64_t = _DTypeAlias(torch.complex64, "complex64_t")
complex128_t = _DTypeAlias(torch.complex128, "complex128_t")

# Error flags are stored as int64 by default; error_t names that for annotations.
error_t = _DTypeAlias(torch.int64, "error_t")

# Python builtins map to PyTorch's default dtypes (not Python's notion).
PYTHON_TYPE_TO_TORCH_DTYPE: dict[type, torch.dtype] = {
    float: torch.float32,
    int: torch.int64,
    bool: torch.bool,
    complex: torch.complex64,
}
