import torch
from typing import Any, Tuple, Optional, Union
from ..err.result import Result, Ok, Err
from .dim import _AttributeRef, _BroadcastMarker
from .validation import DimensionMismatchError, DTypeMismatchError, DeviceMismatchError
from .dtypes import PYTHON_TYPE_TO_TORCH_DTYPE
from .errors import ValidationError

__all__ = ['TensorAnnotation']

class TensorAnnotation:
    def __init__(self, dtype: Any = None, shape: Optional[Tuple] = None, device: Optional[Union[str, torch.device]] = None, requires_grad: Optional[bool] = None) -> None:
        """
        Internal class representing a typed tensor annotation with full validation support.

        Args:
            dtype (Any): Expected dtype (torch.dtype or Python type)
            shape (Optional[Tuple]): Expected shape tuple (ints, strings, Dim refs, Broadcast, or Ellipsis)
            device (Optional[Union[str, torch.device]]): Expected device ('cuda', 'cpu', torch.device, etc.)
            requires_grad (Optional[bool]): Expected gradient tracking state
        """
        self.dtype = dtype
        self.shape = shape
        self.device = device
        self.requires_grad = requires_grad
        
        # Add these for compatibility with PyTorch Geometric's type inspector
        self.__qualname__ = 'Tensor'
        self.__module__ = 'torch'
        
        self.__normalize_dtype()
    
    def __normalize_dtype(self) -> None:
        """
        Convert Python types to torch dtypes for consistency.

        Maps Python built-in types (float, int, bool, complex) to their corresponding torch.dtype equivalents using the shared mapping.
        """
        if self.dtype in PYTHON_TYPE_TO_TORCH_DTYPE:
            self.dtype = PYTHON_TYPE_TO_TORCH_DTYPE[self.dtype]
    
    def __resolve_dimension(self, dim_spec: Any, instance: Any, dim_registry: dict) -> Optional[int]:
        """
        Resolve a dimension specification to an actual integer value.

        Args:
            dim_spec (Any): Can be int, str (named), Dim.attr (attribute ref), Broadcast, or Ellipsis
            instance (Any): The instance to resolve attributes from
            dim_registry (dict): Registry of named dimensions

        Returns:
            (Optional[int]): Expected dimension size, or None if symbolic/broadcast
        """
        if isinstance(dim_spec, int):
            return dim_spec
        
        if isinstance(dim_spec, _AttributeRef):
            return dim_spec.resolve(instance)
        
        if isinstance(dim_spec, _BroadcastMarker):
            return None
        
        if dim_spec is Ellipsis:
            return None
        
        if isinstance(dim_spec, str):
            return dim_registry.get(dim_spec)
        
        return None
    
    def __validate_attribute_ref_dimension(self, name: str, expected_spec: _AttributeRef, expected: int, actual: int, tensor: torch.Tensor, function_name: Optional[str]) -> None:
        """
        Validate a dimension specified by an attribute reference (Dim.attribute).

        Args:
            name (str): Parameter name for error messages
            expected_spec (_AttributeRef): The attribute reference specification
            expected (int): Expected dimension size from resolved attribute
            actual (int): Actual dimension size from tensor
            tensor (torch.Tensor): Tensor being validated
            function_name (Optional[str]): Name of function being validated

        Raises:
            DimensionMismatchError: If expected and actual dimensions don't match
        """
        if expected != actual:
            raise DimensionMismatchError(
                name, str(expected_spec), expected, actual, 
                tensor, function_name
            )
    
    def __validate_named_dimension(self, name: str, expected_spec: str, actual: int, dim_registry: dict, tensor: torch.Tensor, function_name: Optional[str]) -> None:
        """
        Validate a named dimension and track it in the dimension registry.

        Args:
            name (str): Parameter name for error messages
            expected_spec (str): Named dimension string (e.g., "N", "D")
            actual (int): Actual dimension size from tensor
            dim_registry (dict): Registry of named dimensions across tensors
            tensor (torch.Tensor): Tensor being validated
            function_name (Optional[str]): Name of function being validated

        Raises:
            DimensionMismatchError: If dimension doesn't match previously registered value
        """
        if expected_spec in dim_registry:
            if dim_registry[expected_spec] != actual:
                raise DimensionMismatchError(
                    name, expected_spec, dim_registry[expected_spec], 
                    actual, tensor, function_name
                )
        else:
            dim_registry[expected_spec] = actual
    
    def __validate_literal_dimension(self, name: str, dim_idx: int, expected: int, actual: int, tensor: torch.Tensor, function_name: Optional[str]) -> None:
        """
        Validate a literal integer dimension specification.

        Args:
            name (str): Parameter name for error messages
            dim_idx (int): Index of the dimension being validated
            expected (int): Expected literal dimension size
            actual (int): Actual dimension size from tensor
            tensor (torch.Tensor): Tensor being validated
            function_name (Optional[str]): Name of function being validated

        Raises:
            DimensionMismatchError: If expected and actual dimensions don't match
        """
        if expected != actual:
            raise DimensionMismatchError(
                name, f"dim[{dim_idx}]", expected, actual, 
                tensor, function_name
            )
    
    def __validate_single_dimension(self, name: str, dim_idx: int, expected_spec: Any, actual: int, dim_registry: dict, instance: Any, tensor: torch.Tensor, function_name: Optional[str] = None) -> None:
        """
        Validate a single dimension against its specification.

        Args:
            name (str): Parameter name for error messages
            dim_idx (int): Index of the dimension being validated
            expected_spec (Any): Expected dimension specification (int, str, _AttributeRef, _BroadcastMarker)
            actual (int): Actual dimension size from tensor
            dim_registry (dict): Registry of named dimensions across tensors
            instance (Any): Instance object (self) for resolving Dim.attribute
            tensor (torch.Tensor): Tensor being validated
            function_name (Optional[str]): Name of function being validated

        Raises:
            DimensionMismatchError: If dimension validation fails
        """
        if isinstance(expected_spec, _BroadcastMarker):
            return
        
        expected = self.__resolve_dimension(expected_spec, instance, dim_registry)
        
        if isinstance(expected_spec, _AttributeRef):
            self.__validate_attribute_ref_dimension(
                name, expected_spec, expected, actual, tensor, function_name
            )
        elif isinstance(expected_spec, str):
            self.__validate_named_dimension(
                name, expected_spec, actual, dim_registry, tensor, function_name
            )
        elif isinstance(expected_spec, int):
            self.__validate_literal_dimension(
                name, dim_idx, expected, actual, tensor, function_name
            )
    
    def __validate_ellipsis_prefix(self, name: str, prefix: Tuple, tensor: torch.Tensor, dim_registry: dict, instance: Any, function_name: Optional[str]) -> None:
        """
        Validate prefix dimensions when Ellipsis is present in shape specification.

        Args:
            name (str): Parameter name for error messages
            prefix (Tuple): Prefix dimensions before Ellipsis
            tensor (torch.Tensor): Tensor being validated
            dim_registry (dict): Registry of named dimensions across tensors
            instance (Any): Instance object (self) for resolving Dim.attribute
            function_name (Optional[str]): Name of function being validated
        """
        for i, expected_spec in enumerate(prefix):
            actual = tensor.shape[i]
            self.__validate_single_dimension(
                name, i, expected_spec, actual, dim_registry, instance, 
                tensor, function_name
            )
    
    def __validate_ellipsis_suffix(self, name: str, suffix: Tuple, tensor: torch.Tensor, dim_registry: dict, instance: Any, function_name: Optional[str]) -> None:
        """
        Validate suffix dimensions when Ellipsis is present in shape specification.

        Args:
            name (str): Parameter name for error messages
            suffix (Tuple): Suffix dimensions after Ellipsis
            tensor (torch.Tensor): Tensor being validated
            dim_registry (dict): Registry of named dimensions across tensors
            instance (Any): Instance object (self) for resolving Dim.attribute
            function_name (Optional[str]): Name of function being validated
        """
        for i, expected_spec in enumerate(suffix):
            actual_idx = tensor.ndim - len(suffix) + i
            actual = tensor.shape[actual_idx]
            self.__validate_single_dimension(
                name, actual_idx, expected_spec, actual, dim_registry, 
                instance, tensor, function_name
            )
    
    def __validate_shape_with_ellipsis(self, name: str, tensor: torch.Tensor, dim_registry: dict, instance: Any, function_name: Optional[str] = None) -> None:
        """
        Handle shape validation when Ellipsis (...) is present.

        Args:
            name (str): Parameter name for error messages
            tensor (torch.Tensor): Tensor being validated
            dim_registry (dict): Registry of named dimensions across tensors
            instance (Any): Instance object (self) for resolving Dim.attribute
            function_name (Optional[str]): Name of function being validated

        Raises:
            ValueError: If tensor doesn't have enough dimensions
        """
        ellipsis_idx = self.shape.index(Ellipsis)
        prefix = self.shape[:ellipsis_idx]
        suffix = self.shape[ellipsis_idx + 1:]
        
        min_dims = len(prefix) + len(suffix)
        if tensor.ndim < min_dims:
            raise ValueError(
                f"Tensor '{name}' has {tensor.ndim} dimensions but annotation "
                f"requires at least {min_dims} (prefix: {prefix}, suffix: {suffix})"
            )
        
        self.__validate_ellipsis_prefix(name, prefix, tensor, dim_registry, instance, function_name)
        self.__validate_ellipsis_suffix(name, suffix, tensor, dim_registry, instance, function_name)
    
    def __validate_regular_shape(self, name: str, tensor: torch.Tensor, dim_registry: dict, instance: Any, function_name: Optional[str]) -> None:
        """
        Validate shape when no Ellipsis is present (regular shape validation).

        Args:
            name (str): Parameter name for error messages
            tensor (torch.Tensor): Tensor being validated
            dim_registry (dict): Registry of named dimensions across tensors
            instance (Any): Instance object (self) for resolving Dim.attribute
            function_name (Optional[str]): Name of function being validated

        Raises:
            ValueError: If dimension count doesn't match
        """
        if len(self.shape) != tensor.ndim:
            raise ValueError(
                f"Tensor '{name}' dimension count mismatch: "
                f"expected {len(self.shape)} dims, got {tensor.ndim} "
                f"with shape {tuple(tensor.shape)}"
            )
        
        for i, (expected_spec, actual) in enumerate(zip(self.shape, tensor.shape)):
            self.__validate_single_dimension(
                name, i, expected_spec, actual, dim_registry, 
                instance, tensor, function_name
            )
    
    def __validate_dtype(self, name: str, tensor: torch.Tensor, function_name: Optional[str]) -> None:
        """
        Validate tensor dtype matches annotation.

        Args:
            name (str): Parameter name for error messages
            tensor (torch.Tensor): Tensor being validated
            function_name (Optional[str]): Name of function being validated

        Raises:
            DTypeMismatchError: If dtype doesn't match
        """
        if self.dtype is not None and tensor.dtype != self.dtype:
            raise DTypeMismatchError(name, self.dtype, tensor, function_name)
    
    def __validate_device(self, name: str, tensor: torch.Tensor, function_name: Optional[str]) -> None:
        """
        Validate tensor device matches annotation.

        Args:
            name (str): Parameter name for error messages
            tensor (torch.Tensor): Tensor being validated
            function_name (Optional[str]): Name of function being validated

        Raises:
            DeviceMismatchError: If device doesn't match
        """
        if self.device is not None:
            expected_device = str(self.device)
            actual_device = str(tensor.device)
            if expected_device != actual_device:
                raise DeviceMismatchError(name, expected_device, tensor, function_name)
    
    def __validate_requires_grad(self, name: str, tensor: torch.Tensor) -> None:
        """
        Validate tensor requires_grad matches annotation.

        Args:
            name (str): Parameter name for error messages
            tensor (torch.Tensor): Tensor being validated

        Raises:
            ValueError: If requires_grad doesn't match
        """
        if self.requires_grad is not None and tensor.requires_grad != self.requires_grad:
            raise ValueError(
                f"Tensor '{name}' requires_grad mismatch: "
                f"expected {self.requires_grad}, got {tensor.requires_grad}"
            )
    
    def __validate_shape(self, name: str, tensor: torch.Tensor, dim_registry: dict, instance: Any, function_name: Optional[str]) -> None:
        """
        Validate tensor shape matches annotation.

        Args:
            name (str): Parameter name for error messages
            tensor (torch.Tensor): Tensor being validated
            dim_registry (dict): Registry of named dimensions across tensors
            instance (Any): Instance object (self) for resolving Dim.attribute
            function_name (Optional[str]): Name of function being validated
        """
        if Ellipsis in self.shape:
            self.__validate_shape_with_ellipsis(
                name, tensor, dim_registry, instance, function_name
            )
        else:
            self.__validate_regular_shape(
                name, tensor, dim_registry, instance, function_name
            )
    
    def validate(self, name: str, tensor: torch.Tensor, dim_registry: Optional[dict] = None, instance: Any = None, function_name: Optional[str] = None) -> bool:
        """
        Validate tensor against annotation with comprehensive checks.

        Args:
            name (str): Parameter name for error messages
            tensor (torch.Tensor): Tensor to validate
            dim_registry (Optional[dict]): Registry for named dimensions across tensors
            instance (Any): Instance object (self) for resolving Dim.attribute
            function_name (Optional[str]): Name of function being validated (for error messages)

        Returns:
            (bool): True if validation passes

        Raises:
            DTypeMismatchError: If dtype doesn't match
            DimensionMismatchError: If dimensions don't match
            DeviceMismatchError: If device doesn't match
            ValueError: For other validation failures
        """
        if dim_registry is None:
            dim_registry = {}
        
        self.__validate_dtype(name, tensor, function_name)
        self.__validate_device(name, tensor, function_name)
        self.__validate_requires_grad(name, tensor)
        
        if self.shape is not None:
            self.__validate_shape(name, tensor, dim_registry, instance, function_name)
        
        return True
    
    def validate_result(self, name: str, tensor: torch.Tensor, dim_registry: Optional[dict] = None, instance: Any = None, function_name: Optional[str] = None) -> Result[bool, ValidationError]:
        """
        Validate tensor against annotation, returning Result instead of raising.

        Args:
            name (str): Parameter name for error messages
            tensor (torch.Tensor): Tensor to validate
            dim_registry (Optional[dict]): Registry for named dimensions across tensors
            instance (Any): Instance object (self) for resolving Dim.attribute
            function_name (Optional[str]): Name of function being validated (for error messages)

        Returns:
            (Result[bool, ValidationError]): Ok(True) if validation passes, Err with error details otherwise
        """
        try:
            self.validate(name, tensor, dim_registry, instance, function_name)
            return Ok(True)
        except (DTypeMismatchError, DimensionMismatchError, DeviceMismatchError) as e:
            return Err(ValidationError(
                message=str(e),
                context={
                    "parameter": name,
                    "function": function_name,
                }
            ))
        except ValueError as e:
            return Err(ValidationError(
                message=str(e),
                context={
                    "parameter": name,
                    "function": function_name,
                }
            ))
    
    def __repr__(self) -> str:
        """
        Return string representation of the tensor annotation.

        Returns:
            (str): String representation in format Tensor[dtype, shape, device=..., requires_grad=...]
        """
        parts = []
        if self.dtype:
            dtype_str = str(self.dtype).replace('torch.', '')
            parts.append(dtype_str)
        if self.shape:
            parts.append(str(self.shape))
        if self.device:
            parts.append(f"device={self.device}")
        if self.requires_grad is not None:
            parts.append(f"requires_grad={self.requires_grad}")
        return f"Tensor[{', '.join(parts)}]"
    
    def __class_getitem__(cls, item: Any) -> 'TensorAnnotation':
        """
        Allow class to be used as a type annotation.

        Args:
            item (Any): Item passed to class subscript

        Returns:
            (TensorAnnotation): Returns the class itself for type checking
        """
        return cls
    
    def __or__(self, other: Any) -> Union['TensorAnnotation', Any]:
        """
        Support union operator for type hints: Tensor[...] | None.

        Args:
            other (Any): Right operand (typically None or another type)

        Returns:
            (Union[TensorAnnotation, Any]): Union type for type checking
        """
        return Union[self, other]
    
    def __ror__(self, other: Any) -> Union[Any, 'TensorAnnotation']:
        """
        Support reverse union operator for type hints: None | Tensor[...].

        Args:
            other (Any): Left operand (typically None or another type)

        Returns:
            (Union[Any, TensorAnnotation]): Union type for type checking
        """
        return Union[other, self]

