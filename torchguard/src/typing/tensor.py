from typing import Any, Optional, Tuple, Union
from .annotation import TensorAnnotation
__all__ = ['TensorType', 'Tensor']

class TensorType(type):
    def __getitem__(cls, params: Any) -> TensorAnnotation:
        """
        Allow subscript syntax: Tensor[dtype, shape, device, requires_grad]

        Args:
            params (Any): Can be:
                - dtype only: Tensor[float32_t]
                - dtype + shape: Tensor[float32_t, ("N", "D")]
                - dtype + shape + device: Tensor[float32_t, ("N", "D"), "cuda"]
                - dtype + shape + device + requires_grad: Tensor[float32_t, ("N", "D"), "cuda", True]

        Returns:
            (TensorAnnotation): A TensorAnnotation instance
        """
        if not isinstance(params, tuple):
            return cls.__handle_single_parameter(params)
        
        cls.__validate_parameter_count(params)
        return cls.__create_annotation_from_tuple(params)
    
    def __handle_single_parameter(cls, param: Any) -> TensorAnnotation:
        """
        Handle single parameter case (non-tuple).

        Args:
            param (Any): Single parameter (dtype or shape)

        Returns:
            (TensorAnnotation): TensorAnnotation with single parameter set
        """
        if isinstance(param, (tuple, list)):
            return TensorAnnotation(dtype=None, shape=param)
        else:
            normalized_dtype = cls.__normalize_dtype(param)
            return TensorAnnotation(dtype=normalized_dtype, shape=None)
    
    def __normalize_dtype(cls, dtype: Any) -> Any:
        """
        Normalize dtype by extracting _dtype attribute if present.

        Args:
            dtype (Any): Dtype value (may be alias with _dtype attribute)

        Returns:
            (Any): Normalized dtype value
        """
        if hasattr(dtype, '_dtype'):
            return dtype._dtype
        return dtype
    
    def __validate_parameter_count(cls, params: Tuple) -> None:
        """
        Validate that parameter count is within acceptable range.

        Args:
            params (Tuple): Tuple of parameters

        Raises:
            ValueError: If parameter count exceeds maximum of 4
        """
        if len(params) > 4:
            raise ValueError(
                f"Tensor[...] expects 1-4 parameters (dtype, shape, device, requires_grad), "
                f"got {len(params)}"
            )
    
    def __extract_dtype(cls, params: Tuple) -> Optional[Any]:
        """
        Extract and normalize dtype from parameters tuple.

        Args:
            params (Tuple): Tuple of parameters

        Returns:
            (Optional[Any]): Extracted and normalized dtype, or None if not provided
        """
        if len(params) >= 1:
            return cls.__normalize_dtype(params[0])
        return None
    
    def __extract_shape(cls, params: Tuple) -> Optional[Any]:
        """
        Extract shape from parameters tuple.

        Args:
            params (Tuple): Tuple of parameters

        Returns:
            (Optional[Any]): Extracted shape, or None if not provided
        """
        if len(params) >= 2:
            return params[1]
        return None
    
    def __extract_device(cls, params: Tuple) -> Optional[Union[str, Any]]:
        """
        Extract device from parameters tuple.

        Args:
            params (Tuple): Tuple of parameters

        Returns:
            (Optional[Union[str, Any]]): Extracted device, or None if not provided
        """
        if len(params) >= 3:
            return params[2]
        return None
    
    def __extract_requires_grad(cls, params: Tuple) -> Optional[bool]:
        """
        Extract requires_grad from parameters tuple.

        Args:
            params (Tuple): Tuple of parameters

        Returns:
            (Optional[bool]): Extracted requires_grad, or None if not provided
        """
        if len(params) >= 4:
            return params[3]
        return None
    
    def __create_annotation_from_tuple(cls, params: Tuple) -> TensorAnnotation:
        """
        Create TensorAnnotation from tuple of parameters.

        Args:
            params (Tuple): Tuple of parameters (dtype, shape, device, requires_grad)

        Returns:
            (TensorAnnotation): TensorAnnotation instance with extracted parameters
        """
        dtype = cls.__extract_dtype(params)
        shape = cls.__extract_shape(params)
        device = cls.__extract_device(params)
        requires_grad = cls.__extract_requires_grad(params)
        
        return TensorAnnotation(
            dtype=dtype,
            shape=shape,
            device=device,
            requires_grad=requires_grad
        )

class Tensor(metaclass=TensorType):
    """
    Type annotation helper for torch.Tensor with comprehensive validation.

    Syntax:
        Tensor[dtype, shape, device, requires_grad]

    Examples:
        >>> # Basic usage
        >>> x: Tensor[float32_t, ("N", "D")]

        >>> # With instance attributes
        >>> x: Tensor[float32_t, ("N", Dim.in_channels)]

        >>> # With ellipsis for variable batch dims
        >>> x: Tensor[float32_t, (..., "seq", "hidden")]

        >>> # With broadcast markers
        >>> bias: Tensor[float32_t, (Broadcast, Dim.features)]

        >>> # With device
        >>> x: Tensor[float32_t, ("N", "D"), "cuda"]

        >>> # Optional tensors
        >>> mask: Optional[Tensor[bool_t, ("N",)]] = None

        >>> # Tuple returns
        >>> def forward(...) -> tuple[Tensor[float32_t, ("N", "D")], Tensor[int64_t, ("N",)]]:
        ...     return embeddings, indices

    Features:
    - Named dimensions with consistency tracking
    - Instance attribute dimensions (Dim.attr)
    - Ellipsis for variable batch dimensions
    - Broadcast markers for broadcast-compatible dims
    - Device validation
    - requires_grad validation
    - Optional tensor support
    - Enhanced error messages
    - Zero overhead with `python -O`

    Runtime Validation:
        Use the @tensorcheck decorator:

        >>> @tensorcheck
        ... def forward(
        ...     self,
        ...     x: Tensor[float32_t, ("N", Dim.in_channels)],
        ...     mask: Optional[Tensor[bool_t, ("N",)]] = None
        ... ) -> Tensor[float32_t, ("N", Dim.out_channels)]:
        ...     return self.layer(x)

        Validation only runs when assertions are enabled (normal Python execution).
        When running with 'python -O', validation is completely removed for zero overhead.
    """
    pass

