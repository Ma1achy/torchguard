"""
Tensor typing system for torchguard.

Provides:
- error_t: Type alias for error flag tensors
- Other dtype aliases (float32_t, int64_t, etc.)
- TensorAnnotation: Internal annotation parsing
- Validation errors
"""
from .dtypes import (
    # Error type
    error_t,
    # Float types
    float8_e5m2_t, float8_e4m3fn_t, float8_e5m2fnuz_t, float8_e4m3fnuz_t,
    float16_t, bfloat16_t, float32_t, float64_t,
    # Int types
    int8_t, int16_t, int32_t, int64_t,
    uint8_t, uint16_t, uint32_t, uint64_t,
    # Other types
    complex32_t, complex64_t, complex128_t,
    bool_t,
    # Quantized types
    qint8_t, qint32_t, quint8_t, quint4x2_t, quint2x4_t,
    # Mapping
    PYTHON_TYPE_TO_TORCH_DTYPE,
)
from .annotation import TensorAnnotation
from .errors import (
    ValidationError,
    DimensionMismatchError,
    DTypeMismatchError,
    DeviceMismatchError,
    InvalidParameterError,
    TypeMismatchError,
    InvalidReturnTypeError,
)

__all__ = [
    # Primary export
    'error_t',
    # Float types
    'float8_e5m2_t', 'float8_e4m3fn_t', 'float8_e5m2fnuz_t', 'float8_e4m3fnuz_t',
    'float16_t', 'bfloat16_t', 'float32_t', 'float64_t',
    # Int types
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    # Other types
    'complex32_t', 'complex64_t', 'complex128_t',
    'bool_t',
    # Quantized
    'qint8_t', 'qint32_t', 'quint8_t', 'quint4x2_t', 'quint2x4_t',
    # Mapping
    'PYTHON_TYPE_TO_TORCH_DTYPE',
    # Annotation
    'TensorAnnotation',
    # Validation errors
    'ValidationError',
    'DimensionMismatchError',
    'DTypeMismatchError',
    'DeviceMismatchError',
    'InvalidParameterError',
    'TypeMismatchError',
    'InvalidReturnTypeError',
]
