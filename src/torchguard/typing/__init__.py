"""Tensor typing: ``Tensor[...]`` annotations, ``Dim``/``Broadcast``, dtype
aliases, and the validation engine."""
from __future__ import annotations

from .annotation import TensorAnnotation
from .dim import Broadcast, Dim
from .dtypes import (
    PYTHON_TYPE_TO_TORCH_DTYPE,
    bfloat16_t,
    bool_t,
    complex64_t,
    complex128_t,
    error_t,
    float16_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
)
from .errors import (
    DeviceMismatchError,
    DimensionMismatchError,
    DTypeMismatchError,
    InvalidParameterError,
    InvalidReturnTypeError,
    TypeMismatchError,
    ValidationError,
)
from .tensor import Tensor, TensorType

__all__ = [
    "Tensor",
    "TensorType",
    "TensorAnnotation",
    "Dim",
    "Broadcast",
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
    "ValidationError",
    "DimensionMismatchError",
    "DTypeMismatchError",
    "DeviceMismatchError",
    "InvalidParameterError",
    "TypeMismatchError",
    "InvalidReturnTypeError",
]
