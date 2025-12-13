import torch
from typing import Any, Optional, Dict
__all__ = ['TensorValidationError', 'DimensionMismatchError', 'DTypeMismatchError', 'DeviceMismatchError']

class TensorValidationError(ValueError):
    """Base class for tensor validation errors with enhanced context."""
    
    def __init__(self, param_name: str, message: str, expected: Any = None, actual_tensor: Optional[torch.Tensor] = None, function_name: Optional[str] = None) -> None:
        """
        Initialize tensor validation error with enhanced context.

        Args:
            param_name (str): Name of the parameter that failed validation
            message (str): Error message describing the validation failure
            expected (Any): Expected value or type
            actual_tensor (Optional[torch.Tensor]): Actual tensor that failed validation
            function_name (Optional[str]): Name of the function where validation failed
        """
        self.param_name = param_name
        self.function_name = function_name
        self.expected = expected
        self.actual = self.__extract_tensor_info(actual_tensor)
        
        error_message = self.__build_error_message(message)
        super().__init__(error_message)
    
    def __extract_tensor_info(self, tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
        """
        Extract information from tensor for error reporting.

        Args:
            tensor (Optional[torch.Tensor]): Tensor to extract information from

        Returns:
            (Optional[Dict[str, Any]]): Dictionary with tensor info or None if tensor is None
        """
        if tensor is None:
            return None
        
        return {
            'dtype': tensor.dtype,
            'shape': tuple(tensor.shape),
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad,
        }
    
    def __build_header(self, message: str) -> str:
        """
        Build error message header with separator lines.

        Args:
            message (str): Main error message

        Returns:
            (str): Formatted header section of error message
        """
        header = f"\n{'='*70}\n"
        header += f"Tensor Validation Error: {message}\n"
        header += f"{'='*70}\n"
        return header
    
    def __build_function_info(self) -> str:
        """
        Build function information section of error message.

        Returns:
            (str): Function information section or empty string
        """
        if self.function_name:
            return f"Function: {self.function_name}\n"
        return ""
    
    def __build_parameter_info(self) -> str:
        """
        Build parameter information section of error message.

        Returns:
            (str): Parameter information section
        """
        return f"Parameter: '{self.param_name}'\n"
    
    def __build_expected_info(self) -> str:
        """
        Build expected value information section of error message.

        Returns:
            (str): Expected information section or empty string
        """
        if self.expected is not None:
            return f"Expected: {self.expected}\n"
        return ""
    
    def __build_actual_info(self) -> str:
        """
        Build actual tensor information section of error message.

        Returns:
            (str): Actual tensor information section or empty string
        """
        if self.actual is None:
            return ""
        
        info = "Got:\n"
        info += f"  dtype         : {self.actual['dtype']}\n"
        info += f"  shape         : {self.actual['shape']}\n"
        info += f"  device        : {self.actual['device']}\n"
        info += f"  requires_grad : {self.actual['requires_grad']}\n"
        return info
    
    def __build_footer(self) -> str:
        """
        Build error message footer with separator line.

        Returns:
            (str): Footer section of error message
        """
        return f"{'='*70}"
    
    def __build_error_message(self, message: str) -> str:
        """
        Build complete error message from all components.

        Args:
            message (str): Main error message

        Returns:
            (str): Complete formatted error message
        """
        error_msg = self.__build_header(message)
        error_msg += self.__build_function_info()
        error_msg += self.__build_parameter_info()
        error_msg += self.__build_expected_info()
        error_msg += self.__build_actual_info()
        error_msg += self.__build_footer()
        return error_msg

class DimensionMismatchError(TensorValidationError):
    """Error for dimension mismatches."""
    def __init__(self, param_name: str, dim_name: str, expected: int, actual: int, actual_tensor: torch.Tensor, function_name: Optional[str] = None) -> None:
        """
        Initialize dimension mismatch error.

        Args:
            param_name (str): Name of the parameter with dimension mismatch
            dim_name (str): Name of the dimension that mismatched
            expected (int): Expected dimension size
            actual (int): Actual dimension size
            actual_tensor (torch.Tensor): Tensor that failed validation
            function_name (Optional[str]): Name of the function where validation failed
        """
        message = f"Dimension '{dim_name}' mismatch: expected {expected}, got {actual}"
        super().__init__(param_name, message, expected=expected, actual_tensor=actual_tensor, function_name=function_name)

class DTypeMismatchError(TensorValidationError):
    """Error for dtype mismatches."""
    def __init__(self, param_name: str, expected_dtype: torch.dtype, actual_tensor: torch.Tensor, function_name: Optional[str] = None) -> None:
        """
        Initialize dtype mismatch error.

        Args:
            param_name (str): Name of the parameter with dtype mismatch
            expected_dtype (torch.dtype): Expected dtype
            actual_tensor (torch.Tensor): Tensor that failed validation
            function_name (Optional[str]): Name of the function where validation failed
        """
        message = f"dtype mismatch: expected {expected_dtype}, got {actual_tensor.dtype}"
        super().__init__(param_name, message, expected=expected_dtype, actual_tensor=actual_tensor, function_name=function_name)

class DeviceMismatchError(TensorValidationError):
    """Error for device mismatches."""
    
    def __init__(self, param_name: str, expected_device: str, actual_tensor: torch.Tensor, function_name: Optional[str] = None) -> None:
        """
        Initialize device mismatch error.

        Args:
            param_name (str): Name of the parameter with device mismatch
            expected_device (str): Expected device string
            actual_tensor (torch.Tensor): Tensor that failed validation
            function_name (Optional[str]): Name of the function where validation failed
        """
        message = f"device mismatch: expected {expected_device}, got {actual_tensor.device}"
        super().__init__(param_name, message, expected=expected_device, actual_tensor=actual_tensor, function_name=function_name)
