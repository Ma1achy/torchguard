import torch
from ..err.result import Result, Ok, Err
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .errors import ValidationError

__all__ = [
    # Floating point types
    'float8_e5m2_t', 'float8_e4m3fn_t', 'float8_e5m2fnuz_t', 'float8_e4m3fnuz_t',
    'float16_t', 'bfloat16_t', 'float32_t', 'float64_t',
    # Signed integer types
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    # Unsigned integer types
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    # Complex types
    'complex32_t', 'complex64_t', 'complex128_t',
    # Boolean type
    'bool_t',
    # Semantic types
    'error_t',
    # Quantized types
    'qint8_t', 'qint32_t', 'quint8_t', 'quint4x2_t', 'quint2x4_t',
    # Casting utility
    'type_cast',
    # Configuration
    'PYTHON_TYPE_TO_TORCH_DTYPE',
]

# ============================================================================
# Configuration: Python Type to PyTorch Dtype Mapping
# ============================================================================

# Mapping of Python built-in types to PyTorch dtypes (matching PyTorch defaults).
# This mapping is used throughout the typing system to convert Python types
# (float, int, bool, complex) to their corresponding torch.dtype equivalents.
#
# Note:
#     - float → torch.float32 (PyTorch's default, NOT Python's float64)
#     - int → torch.int64 (PyTorch's default, equivalent to torch.long)
#     - bool → torch.bool
#     - complex → torch.complex128 (Python's native precision)
#
#     This matches PyTorch's tensor creation behavior:
#     >>> torch.tensor([1.0, 2.0]).dtype  # torch.float32
#     >>> torch.tensor([1, 2]).dtype      # torch.int64
PYTHON_TYPE_TO_TORCH_DTYPE = {
    float: torch.float32,      # PyTorch default (not Python's float64!)
    int: torch.int64,          # PyTorch default (equivalent to torch.long)
    bool: torch.bool,          # Boolean type
    complex: torch.complex128, # Python complex is 128-bit
}

class __DTypeAliasMeta(type):
    """Metaclass for dtype aliases that provides syntax highlighting and blocks subscripting."""
    
    def __getitem__(cls, item):
        """
        Block subscripting of dtype aliases.
        
        Args:
            item: Subscript item (not used)
        
        Raises:
            (TypeError): Always raised to prevent subscripting
        """
        raise TypeError(
            f"Type alias '{cls.__name__}' cannot be subscripted. "
            f"Use 'Tensor[{cls.__name__}, shape]' instead of '{cls.__name__}[shape]'."
        )
    
    def __repr__(cls) -> str:
        """
        Get string representation of the dtype alias.
        
        Returns:
            (str): Name of the dtype alias class
        """
        return f"{cls.__name__}"
    
    def __eq__(cls, other) -> bool:
        """
        Check equality with another dtype alias or torch.dtype.
        
        Args:
            other: Object to compare against
        
        Returns:
            (bool): True if equal, False otherwise
        """
        if isinstance(other, torch.dtype):
            return cls._dtype == other
        return super().__eq__(other)
    
    def __hash__(cls) -> int:
        """
        Get hash of the dtype alias.
        
        Returns:
            (int): Hash of the underlying torch.dtype
        """
        return hash(cls._dtype)

class __DTypeAlias(metaclass=__DTypeAliasMeta):
    """Base class for all dtype aliases."""
    _dtype: torch.dtype = None

# ============================================================================
# Floating Point Types
# ============================================================================

class float8_e5m2_t(__DTypeAlias):
    """
    8-bit floating point data type with 5 exponent bits and 2 mantissa bits.
    Equivalent to: `torch.float8_e5m2`
    Format: 1 sign bit + 5 exponent bits + 2 mantissa bits
    
    Usage:
        >>> ultra_low_precision: Tensor[float8_e5m2_t, ("N", "D")] = model(x).to(torch.float8_e5m2)
        >>> compressed_weights: Tensor[float8_e5m2_t, ("out", "in")] = quantize_to_fp8(weights)
    
    Note:
        Extreme low precision format for specialized hardware (H100, MI300).
        Larger dynamic range but less precision than float8_e4m3fn.
        Primarily used for gradients and activations in ultra-low precision training.
    """
    _dtype = torch.float8_e5m2

class float8_e4m3fn_t(__DTypeAlias):
    """
    8-bit floating point data type with 4 exponent bits and 3 mantissa bits (finite, with NaN).
    Equivalent to: `torch.float8_e4m3fn`
    Format: 1 sign bit + 4 exponent bits + 3 mantissa bits
    
    Usage:
        >>> weights_fp8: Tensor[float8_e4m3fn_t, ("N", "D")] = weights.to(torch.float8_e4m3fn)
        >>> activations: Tensor[float8_e4m3fn_t, ("batch", "features")] = forward_fp8(x)
    
    Note:
        Better precision but smaller dynamic range than float8_e5m2.
        Supports NaN but not infinity. Commonly used for weights in FP8 training.
        'fn' = finite with NaN support.
    """
    _dtype = torch.float8_e4m3fn

class float8_e5m2fnuz_t(__DTypeAlias):
    """
    8-bit floating point with 5 exponent bits and 2 mantissa bits (finite, no NaN, unsigned zero).
    Equivalent to: `torch.float8_e5m2fnuz`
    Format: 1 sign bit + 5 exponent bits + 2 mantissa bits
    
    Usage:
        >>> compressed: Tensor[float8_e5m2fnuz_t, ("N", "D")] = data.to(torch.float8_e5m2fnuz)
    
    Note:
        Variant without NaN/Inf support, with unsigned zero.
        'fnuz' = finite, no NaN, unsigned zero.
        Used in some AMD hardware implementations.
    """
    _dtype = torch.float8_e5m2fnuz

class float8_e4m3fnuz_t(__DTypeAlias):
    """
    8-bit floating point with 4 exponent bits and 3 mantissa bits (finite, no NaN, unsigned zero).
    Equivalent to: `torch.float8_e4m3fnuz`
    Format: 1 sign bit + 4 exponent bits + 3 mantissa bits
    
    Usage:
        >>> weights: Tensor[float8_e4m3fnuz_t, ("N", "D")] = weights.to(torch.float8_e4m3fnuz)
    
    Note:
        Variant without NaN/Inf support, with unsigned zero.
        'fnuz' = finite, no NaN, unsigned zero.
        Used in some AMD hardware implementations.
    """
    _dtype = torch.float8_e4m3fnuz

class float16_t(__DTypeAlias):
    """
    Half precision (16-bit) floating point data type.
    Equivalent to: `torch.float16`, `torch.half`
    
    Usage:
        >>> embeddings: Tensor[float16_t, ("N", "D")] = model(x).half()
        >>> mixed_precision: Tensor[float16_t, ("batch", "seq", "hidden")] = encoder(ids)
    
    Note:
        float16 has limited range and precision. Consider bfloat16_t for better stability.
    """
    _dtype = torch.float16

class bfloat16_t(__DTypeAlias):
    """
    Brain floating point (16-bit) data type.
    Equivalent to: `torch.bfloat16`
    
    Usage:
        >>> embeddings: Tensor[bfloat16_t, ("N", "D")] = model(x).bfloat16()
        >>> activations: Tensor[bfloat16_t, ("batch", "features")] = layer(x)
    
    Note:
        bfloat16 has the same exponent range as fp32 but reduced mantissa precision.
        Better than fp16 for most deep learning tasks. Widely used in modern accelerators.
    """
    _dtype = torch.bfloat16

class float32_t(__DTypeAlias):
    """
    Single precision (32-bit) floating point data type. (default floating point type)
    Equivalent to: `torch.float32`, `torch.float`, Python `float`
    
    Usage:
        >>> embeddings: Tensor[float32_t, ("N", "D")] = encoder(ids)
        >>> weights: Tensor[float32_t, ("out_features", "in_features")] = layer.weight
        >>> logits: Tensor[float32_t, ("batch", "num_classes")] = classifier(x)
        >>> # Python float is automatically converted
        >>> data: Tensor[float, ("N",)] = torch.tensor([1.0, 2.0])  # → torch.float32
    
    Note:
        Standard precision for most neural network operations.
        Python float literals are automatically converted to torch.float32.
    """
    _dtype = torch.float32

class float64_t(__DTypeAlias):
    """
    Double precision (64-bit) floating point data type.
    Equivalent to: `torch.float64`, `torch.double`
    
    Usage:
        >>> high_precision: Tensor[float64_t, ("N", "D")] = compute_double()
        >>> numerical_result: Tensor[float64_t, ("batch",)] = solve_system(A, b)
    
    Note:
        Rarely needed for deep learning. Use for numerical stability in specific algorithms.
    """
    _dtype = torch.float64
    
# ============================================================================
# Signed Integer Types
# ============================================================================

class int8_t(__DTypeAlias):
    """
    Signed 8-bit integer data type.
    Equivalent to: `torch.int8`
    Range: -128 to 127
    
    Usage:
        >>> quantized: Tensor[int8_t, ("N", "D")] = quantize_weights(weights)
        >>> compressed: Tensor[int8_t, ("batch", "features")] = compress(data)
    
    Note:
        Common for quantized models and compression.
    """
    _dtype = torch.int8

class int16_t(__DTypeAlias):
    """
    Signed 16-bit integer data type.
    Equivalent to: `torch.int16`, `torch.short`
    Range: -32,768 to 32,767
    
    Usage:
        >>> indices: Tensor[int16_t, ("N",)] = torch.tensor([1, 2, 3], dtype=torch.int16)
        >>> medium_range: Tensor[int16_t, ("batch",)] = data.short()
    """
    _dtype = torch.int16

class int32_t(__DTypeAlias):
    """
    Signed 32-bit integer data type.
    Equivalent to: `torch.int32`, `torch.int`
    Range: -2,147,483,648 to 2,147,483,647
    
    Usage:
        >>> indices: Tensor[int32_t, ("N",)] = torch.tensor([1000, 2000], dtype=torch.int32)
        >>> ids: Tensor[int32_t, ("batch",)] = get_customer_ids()
    """
    _dtype = torch.int32

class int64_t(__DTypeAlias):
    """
    Signed 64-bit integer data type (default integer type).
    Equivalent to: `torch.int64`, `torch.long`, Python `int`
    Range: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
    
    Usage:
        >>> indices: Tensor[int64_t, ("N",)] = torch.arange(100)
        >>> edge_index: Tensor[int64_t, (2, "E")] = graph.edge_index
        >>> node_ids: Tensor[int64_t, ("batch",)] = batch.node_ids
        >>> # Python int is automatically converted
        >>> ids: Tensor[int, ("N",)] = torch.tensor([1, 2, 3])  # → torch.int64
    
    Note:
        Default integer type in PyTorch. Use for indices, IDs, and counts.
        Python int literals are automatically converted to torch.int64.
    """
    _dtype = torch.int64

# ============================================================================
# Unsigned Integer Types
# ============================================================================

class uint8_t(__DTypeAlias):
    """
    Unsigned 8-bit integer data type.
    Equivalent to: `torch.uint8`, `torch.byte`
    Range: 0 to 255
    
    Usage:
        >>> image: Tensor[uint8_t, ("H", "W", 3)] = load_image()
        >>> mask: Tensor[uint8_t, ("batch", "seq")] = (x > threshold).byte()
        >>> pixels: Tensor[uint8_t, (224, 224, 3)] = preprocess_image(img)
    
    Note:
        Common for images and binary masks.
    """
    _dtype = torch.uint8

class uint16_t(__DTypeAlias):
    """
    Unsigned 16-bit integer data type.
    Equivalent to: `torch.uint16`
    Range: 0 to 65,535
    
    Usage:
        >>> depth_map: Tensor[uint16_t, ("H", "W")] = capture_depth()
        >>> large_indices: Tensor[uint16_t, ("N",)] = torch.tensor([60000], dtype=torch.uint16)
    """
    _dtype = torch.uint16

class uint32_t(__DTypeAlias):
    """
    Unsigned 32-bit integer data type.
    Equivalent to: `torch.uint32`
    Range: 0 to 4,294,967,295
    
    Usage:
        >>> large_ids: Tensor[uint32_t, ("N",)] = get_global_ids()
        >>> hash_values: Tensor[uint32_t, ("batch",)] = hash_function(strings)
    """
    _dtype = torch.uint32

class uint64_t(__DTypeAlias):
    """
    Unsigned 64-bit integer data type.
    Equivalent to: `torch.uint64`
    Range: 0 to 18,446,744,073,709,551,615
    
    Usage:
        >>> very_large: Tensor[uint64_t, ("N",)] = torch.tensor([2**50], dtype=torch.uint64)
        >>> timestamps: Tensor[uint64_t, ("batch",)] = get_unix_timestamps_ns()
    
    Note:
        Use for very large unsigned integers or high-precision timestamps.
    """
    _dtype = torch.uint64

# ============================================================================
# Complex Number Types
# ============================================================================

class complex32_t(__DTypeAlias):
    """
    Complex number data type with 16-bit float real and imaginary components.
    Equivalent to: `torch.complex32`, `torch.chalf`
    Total size: 32 bits (16 real + 16 imaginary)
    
    Usage:
        >>> complex_tensor: Tensor[complex32_t, ("N", "D")] = torch.randn(100, 100, dtype=torch.complex32)
        >>> eigenvalues: Tensor[complex32_t, ("N",)] = torch.linalg.eigvals(matrix)
        >>> phase: Tensor[complex32_t, ("batch", "time")] = complex_signal
    
    Note:
        Use for signal processing, FFT, and complex-valued neural networks.
    """
    _dtype = torch.complex32

class complex64_t(__DTypeAlias):
    """
    Complex number data type with 32-bit float real and imaginary components.
    Equivalent to: `torch.complex64`, `torch.cfloat`
    Total size: 64 bits (32 real + 32 imaginary)
    
    Usage:
        >>> fft_result: Tensor[complex64_t, ("N", "freq")] = torch.fft.fft(signal)
        >>> eigenvalues: Tensor[complex64_t, ("N",)] = torch.linalg.eigvals(matrix)
        >>> phase: Tensor[complex64_t, ("batch", "time")] = complex_signal
    
    Note:
        Use for signal processing, FFT, and complex-valued neural networks.
    """
    _dtype = torch.complex64

class complex128_t(__DTypeAlias):
    """
    Complex number data type with 64-bit float real and imaginary components.
    Equivalent to: `torch.complex128`, `torch.cdouble`, Python `complex`
    Total size: 128 bits (64 real + 64 imaginary)
    
    Usage:
        >>> high_precision_fft: Tensor[complex128_t, ("N",)] = torch.fft.fft(signal.double())
        >>> eigenvalues: Tensor[complex128_t, ("N",)] = compute_eigenvalues(matrix)
        >>> # Python complex is automatically converted
        >>> values: Tensor[complex, ("N",)] = torch.tensor([1+2j, 3+4j])  # → torch.complex128
    
    Note:
        Higher precision for numerical algorithms requiring complex numbers.
        Python complex literals are automatically converted to torch.complex128.
    """
    _dtype = torch.complex128

# ============================================================================
# Boolean Type
# ============================================================================

class bool_t(__DTypeAlias):
    """
    Boolean data type.
    Equivalent to: `torch.bool`, Python `bool`
    
    Usage:
        >>> mask: Tensor[bool_t, ("batch", "seq")] = attention_mask
        >>> valid: Tensor[bool_t, ("N",)] = (scores > threshold)
        >>> is_padding: Tensor[bool_t, ("batch", "max_len")] = get_padding_mask()
        >>> # Python bool is automatically converted
        >>> flags: Tensor[bool, ("N",)] = torch.tensor([True, False])  # → torch.bool
    
    Note:
        Use for masks, boolean flags, and conditions. Memory efficient (1 bit per element).
        Python bool literals are automatically converted to torch.bool.
    """
    _dtype = torch.bool

# ============================================================================
# Semantic Types (special-purpose aliases)
# ============================================================================

class error_t(__DTypeAlias):
    """
    Error flags tensor dtype (int64).
    
    Pure type alias for annotations. For operations, use the `err` namespace:
    
        from torchguard import err, flags, error_t
        
        # Type hints use error_t
        def forward(x: Tensor) -> tuple[Tensor, Tensor[error_t, "batch num_words"]]:
            f = err.new(x)  # Operations use err namespace
            return output, f
    
    Equivalent to: `torch.int64`
    
    Storage:
        - int64 tensor, shape (N, num_words)
        - Each 64-bit word holds 4 × 16-bit error slots
        - Slot layout: [location:10][code:4][severity:2]
    """
    _dtype = torch.int64

# ============================================================================
# Quantized Types
# ============================================================================

class qint8_t(__DTypeAlias):
    """
    Quantized signed 8-bit integer data type.
    Equivalent to: `torch.qint8`
    Range: -128 to 127 (with scale and zero_point)
    
    Usage:
        >>> quantized_weights: Tensor[qint8_t, ("N", "D")] = torch.quantize_per_tensor(
        ...     weights, scale=0.1, zero_point=0, dtype=torch.qint8
        ... )
        >>> quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    Note:
        Used in quantization-aware training and post-training quantization.
        Requires scale and zero_point for proper quantization/dequantization.
    """
    _dtype = torch.qint8

class qint32_t(__DTypeAlias):
    """
    Quantized signed 32-bit integer data type.
    Equivalent to: `torch.qint32`
    
    Usage:
        >>> accumulator: Tensor[qint32_t, ("N",)] = quantized_matmul_output
    
    Note:
        Typically used internally for accumulation in quantized matrix operations.
        Provides higher precision for intermediate computations in quantized inference.
    """
    _dtype = torch.qint32

class quint8_t(__DTypeAlias):
    """
    Quantized unsigned 8-bit integer data type.
    Equivalent to: `torch.quint8`
    Range: 0 to 255 (with scale and zero_point)
    
    Usage:
        >>> quantized_activations: Tensor[quint8_t, ("batch", "features")] = torch.quantize_per_tensor(
        ...     activations, scale=0.1, zero_point=128, dtype=torch.quint8
        ... )
    
    Note:
        Common for activations in quantized neural networks.
        Unsigned representation is often more suitable for ReLU-based activations.
    """
    _dtype = torch.quint8

class quint4x2_t(__DTypeAlias):
    """
    Quantized 4-bit unsigned integer data type (packed, 2 values per byte).
    Equivalent to: `torch.quint4x2`
    Range: 0 to 15 per 4-bit value
    
    Usage:
        >>> ultra_compressed: Tensor[quint4x2_t, ("N", "D")] = quantize_4bit(weights)
    
    Note:
        Highly compressed format for extreme quantization scenarios.
        Two 4-bit values are packed into each byte for memory efficiency.
        Mainly used in edge deployment and mobile inference.
    """
    _dtype = torch.quint4x2

class quint2x4_t(__DTypeAlias):
    """
    Quantized 2-bit unsigned integer data type (packed, 4 values per byte).
    Equivalent to: `torch.quint2x4`
    Range: 0 to 3 per 2-bit value
    
    Usage:
        >>> extreme_compressed: Tensor[quint2x4_t, ("N", "D")] = quantize_2bit(weights)
    
    Note:
        Extreme compression for very low-precision quantization.
        Four 2-bit values are packed into each byte.
        Experimental and rarely used in practice.
    """
    _dtype = torch.quint2x4

# ============================================================================
# Casting Utility
# ============================================================================

class _CastMeta(type):
    """Metaclass that enables subscript syntax for type_cast: type_cast[dtype](tensor)"""
    
    def __getitem__(cls, target_dtype: type) -> '_CastCallable':
        """
        Enable subscript syntax: type_cast[float32_t](tensor) or type_cast[float](tensor)
        
        Args:
            target_dtype (type): Target dtype (custom alias, torch.dtype, or Python type)
        
        Returns:
            (_CastCallable): Callable that performs the cast
        """
        # Custom type alias (e.g., float32_t, int64_t)
        if hasattr(target_dtype, '_dtype'):
            torch_dtype = target_dtype._dtype
        # Native torch dtype (e.g., torch.float32)
        elif isinstance(target_dtype, torch.dtype):
            torch_dtype = target_dtype
        # Python built-in types (e.g., float, int, bool)
        elif target_dtype in PYTHON_TYPE_TO_TORCH_DTYPE:
            torch_dtype = PYTHON_TYPE_TO_TORCH_DTYPE[target_dtype]
        else:
            raise TypeError(f"type_cast[...] expects a dtype alias (e.g., float32_t), torch.dtype, or Python type (float, int, bool, complex), got {target_dtype!r}")
        
        return _CastCallable(torch_dtype, target_dtype)

class _CastCallable:
    """Callable returned by cast[dtype] that performs the actual cast."""
    
    def __init__(self, torch_dtype: torch.dtype, type_alias: type) -> None:
        """
        Initialize cast callable.
        
        Args:
            torch_dtype (torch.dtype): Target PyTorch dtype
            type_alias (type): Type alias used for casting
        """
        self.torch_dtype = torch_dtype
        self.type_alias = type_alias
    
    def __call__(self, tensor: 'torch.Tensor') -> 'Result[torch.Tensor, Error]':
        """
        Cast tensor to the target dtype (returns Result).
        
        Args:
            tensor (torch.Tensor): Tensor to cast
        
        Returns:
            (Result[torch.Tensor, Error]): Ok(casted_tensor) on success, Err on failure
        """
        from ..errors import ValidationError
        # from ..debug import Debugger  # Removed
        
        # Validate input is a tensor
        if not isinstance(tensor, torch.Tensor):
            return Err(Error.Tensor.TypeCast(
                f"Expected torch.Tensor, got {type(tensor).__name__}",
                target_dtype=str(self.torch_dtype),
                input_type=type(tensor).__name__
            ))
        
        # Validate tensor is not empty
        err = Debugger.validate_all(
            Debugger.non_negative("tensor.numel()", torch.tensor([tensor.numel()]))
        )
        if err.is_err():
            return Err(Error.Tensor.TypeCast(
                f"Cannot cast empty tensor to {self.torch_dtype}",
                target_dtype=str(self.torch_dtype),
                tensor_shape=str(tuple(tensor.shape))
            ))
        
        # Attempt the cast
        try:
            casted = tensor.to(self.torch_dtype)
            return Ok(casted)
        except Exception as e:
            return Err(Error.Tensor.TypeCast(
                f"Failed to cast tensor to {self.torch_dtype}: {e}",
                target_dtype=str(self.torch_dtype),
                source_dtype=str(tensor.dtype),
                error=str(e)
            ))
    
    def __repr__(self) -> str:
        """
        Get string representation of the cast callable.
        
        Returns:
            (str): String representation showing the type alias
        """
        return f"type_cast[{self.type_alias.__name__ if hasattr(self.type_alias, '__name__') else self.type_alias}]"

class type_cast(metaclass=_CastMeta):
    """
    Type-safe tensor casting with subscript syntax and Result-based error handling.
    
    Syntax:
        type_cast[target_dtype](tensor)  # Returns Result[Tensor, Error]
    
    Supports:
        - Custom dtype aliases: float32_t, int64_t, bool_t, etc.
        - PyTorch dtypes: torch.float32, torch.int64, etc.
        - Python types: float, int, bool, complex (uses PyTorch defaults)
    
    Usage:
        >>> # Returns Result for error handling
        >>> result = type_cast[float32_t](int_tensor)
        >>> if result.is_err():
        >>>     log_error(result)
        >>>     return fallback
        >>> casted = result.ok_value
        
        >>> # Python built-in types (PyTorch defaults)
        >>> result = type_cast[float](int_tensor)    # → Result[torch.float32, Error]
        >>> result = type_cast[int](float_tensor)    # → Result[torch.int64, Error]
        >>> result = type_cast[bool](binary_tensor)  # → Result[torch.bool, Error]
        
        >>> # PyTorch dtypes directly
        >>> result = type_cast[torch.float16](tensor)
        >>> result = type_cast[torch.bfloat16](tensor)
    
    Error Handling:
        - Always returns Result[Tensor, Error]
        - On success: Ok(casted_tensor)
        - On failure: Err(Error.Tensor.TypeCast(...))
        - Validates input is a tensor and not empty
    
    Note:
        Python type mappings follow PyTorch conventions:
        - float   → torch.float32 (PyTorch's default, NOT Python's float64)
        - int     → torch.int64 (PyTorch's default)
        - bool    → torch.bool
        - complex → torch.complex128 (Python's native precision)
        
        This matches PyTorch's tensor creation behavior:
        >>> torch.tensor([1.0, 2.0]).dtype  # torch.float32
        >>> torch.tensor([1, 2]).dtype      # torch.int64
    """
    pass
