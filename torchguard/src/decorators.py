"""
Decorators for torchguard.

Provides:
- @tracked: Location tracking for nn.Module classes
- @tensorcheck: Tensor validation and NaN/Inf auto-detection
- @as_result: Wrap exceptions in Result (re-exported from result.py)
- @as_exception: Unwrap Result to exceptions (re-exported from result.py)
"""
from __future__ import annotations

import functools
import inspect
import typing
import warnings
from typing import Any, Callable, List, Optional, Set, Tuple, Union, get_args, get_origin

import torch
import torch.nn as nn

from .typing.errors import InvalidParameterError, TypeMismatchError, InvalidReturnTypeError, ValidationError
from .err.result import Ok, Err, Result, as_result, as_exception, unwrap
from .typing.annotation import TensorAnnotation

__all__ = ['tensorcheck', 'tracked', 'as_result', 'as_exception', 'unwrap']

# Type alias for auto_detect parameter
# - True: detect all default codes (NaN, Inf)
# - False/None: no detection
# - Set[int]: specific codes, e.g. {ErrorCode.NAN}
AutoDetectType = Union[bool, Set[int], None]


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _partition(results: List[Result]) -> Tuple[List, List]:
    """Partition results into (ok_values, err_values)."""
    oks = []
    errs = []
    for r in results:
        if r.is_ok():
            oks.append(r.unwrap())
        else:
            errs.append(r.unwrap_err())
    return oks, errs


# ═══════════════════════════════════════════════════════════════════════════════
# WARN-ONCE PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

_WARNED_KEYS: set[tuple] = set()


def __warn_once(key: tuple, msg: str) -> None:
    """Warn only once per unique key. Prevents spam."""
    if key not in _WARNED_KEYS:
        _WARNED_KEYS.add(key)
        warnings.warn(msg, stacklevel=3)


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR_T VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def __validate_error_t(maybe_flags: Any, strict: bool = False) -> None:
    """
    Validate tensor looks like a valid error_t.
    
    Args:
        maybe_flags: Value to check
        strict: If True, raise on invalid. If False, warn once.
    """
    if not hasattr(maybe_flags, 'dtype'):
        return  # Not a tensor
    
    if maybe_flags.dtype != torch.int64:
        msg = f"error_t should be int64, got {maybe_flags.dtype}"
        if strict:
            raise TypeError(msg)
        else:
            __warn_once(('error_t', 'dtype'), msg)
    
    if maybe_flags.ndim != 2:
        msg = f"error_t should be 2D (N, num_words), got {maybe_flags.ndim}D"
        if strict:
            raise ValueError(msg)
        else:
            __warn_once(('error_t', 'ndim'), msg)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS DECORATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def __extract_and_inject_locations(module: nn.Module) -> None:
    """
    Extract locations from nn.Module hierarchy and inject _fx_path.
    
    Warns if any submodules can't receive _fx_path (e.g., frozen modules).
    """
    from .core.location import ErrorLocation
    
    failed_injections: list[str] = []
    
    for name, child in module.named_modules():
        if name:
            try:
                child._fx_path = name
            except AttributeError:
                # Frozen module or custom __setattr__ - can't inject
                failed_injections.append(f"{name} ({child.__class__.__name__})")
            ErrorLocation.register(name)
    
    # Warn about modules that couldn't receive _fx_path
    if failed_injections:
        failed_list = ", ".join(failed_injections[:5])
        extra = f" (+{len(failed_injections) - 5} more)" if len(failed_injections) > 5 else ""
        __warn_once(
            ('fx_path_injection_failed', module.__class__.__name__),
            f"Could not inject _fx_path into {len(failed_injections)} submodule(s): "
            f"{failed_list}{extra}. "
            f"These modules will use class name fallback for error location tracking. "
            f"This may cause location collisions if multiple modules share the same class."
        )


def __is_optional(annotation: Any) -> bool:
    """Check if annotation is Optional[T] (Union[T, None])."""
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        return type(None) in args
    return False


def __unwrap_optional(annotation: Any) -> Any:
    """Extract T from Optional[T]."""
    args = get_args(annotation)
    return next(arg for arg in args if arg is not type(None))


def __should_skip_parameter(param_name: str, param: inspect.Parameter) -> bool:
    """Check if a parameter should be skipped during validation."""
    if param_name == 'self':
        return True
    if param.annotation == inspect.Parameter.empty:
        return True
    return False


def __process_optional_annotation(annotation: Any, param_value: Any) -> Optional[Any]:
    """Process Optional annotation and handle None values."""
    if __is_optional(annotation):
        if param_value is None:
            return None
        return __unwrap_optional(annotation)
    return annotation


def __validate_single_parameter(param_name: str, param_value: Any, annotation: Any, dim_registry: dict, instance: Any, function_name: Optional[str]) -> Result[bool, ValidationError]:
    """Validate a single parameter against its type annotation."""
    if isinstance(annotation, TensorAnnotation) and isinstance(param_value, torch.Tensor):
        return annotation.validate_result(param_name, param_value, dim_registry, instance, function_name)
    return Ok(True)


def __collect_parameter_validations(sig: inspect.Signature, bound_args: inspect.BoundArguments, dim_registry: dict, instance: Any, function_name: Optional[str]) -> list[Result[bool, ValidationError]]:
    """Collect validation results for all function parameters."""
    validations: list[Result[bool, ValidationError]] = []
    
    for param_name, param_value in bound_args.arguments.items():
        param: inspect.Parameter = sig.parameters[param_name]
        
        if __should_skip_parameter(param_name, param):
            continue
        
        annotation = param.annotation
        processed_annotation = __process_optional_annotation(annotation, param_value)
        
        if processed_annotation is None:
            continue
        
        result = __validate_single_parameter(param_name, param_value, processed_annotation, dim_registry, instance, function_name)
        validations.append(result)
    
    return validations


def __validate_input_arguments(sig: inspect.Signature, bound_args: inspect.BoundArguments, dim_registry: dict, instance: Any, function_name: Optional[str]) -> Result[bool, ValidationError]:
    """Validate all input arguments with instance context."""
    validations = __collect_parameter_validations(sig, bound_args, dim_registry, instance, function_name)
    
    if not validations:
        return Ok(True)
    
    # Check for validation errors
    _, errors = _partition(validations)
    
    if errors:
        # Create a single error with all validation failures
        error_details = "\n  - ".join(str(e) for e in errors)
        msg = f"Parameter validation failed for {function_name}: {len(errors)} error(s)\n  - {error_details}"
        return Err(InvalidParameterError(msg, function=function_name, num_errors=len(errors)))
    
    return Ok(True)


def __process_optional_return_type(return_annotation: Any, result: Any) -> Optional[Any]:
    """Process Optional return type annotation and handle None results."""
    if __is_optional(return_annotation):
        if result is None:
            return None
        return __unwrap_optional(return_annotation)
    return return_annotation


def __validate_single_tuple_element(ret_val: Any, ret_type: Any, index: int, dim_registry: dict, instance: Any, function_name: Optional[str]) -> Result[bool, ValidationError]:
    """Validate a single element from a tuple return value."""
    processed_type = __process_optional_return_type(ret_type, ret_val)
    if processed_type is None:
        return Ok(True)
    
    if isinstance(processed_type, TensorAnnotation) and isinstance(ret_val, torch.Tensor):
        return processed_type.validate_result(f"return[{index}]", ret_val, dim_registry, instance, function_name)
    return Ok(True)


def __collect_tuple_validations(result: tuple, return_types: tuple, dim_registry: dict, instance: Any, function_name: Optional[str]) -> list[Result[bool, ValidationError]]:
    """Collect validation results for tuple return elements."""
    validations: list[Result[bool, ValidationError]] = []
    for i, (ret_val, ret_type) in enumerate(zip(result, return_types)):
        validation = __validate_single_tuple_element(ret_val, ret_type, i, dim_registry, instance, function_name)
        validations.append(validation)
    return validations


def __validate_tuple_return(result: tuple, return_types: tuple, dim_registry: dict, instance: Any, function_name: Optional[str]) -> Result[bool, ValidationError]:
    """Validate all elements in tuple return values."""
    validations = __collect_tuple_validations(result, return_types, dim_registry, instance, function_name)
    
    if not validations:
        return Ok(True)
    
    # Check for validation errors
    _, errors = _partition(validations)
    
    if errors:
        # Create a single error with all validation failures
        error_details = "\n  - ".join(str(e) for e in errors)
        msg = f"Tuple return validation failed for {function_name}: {len(errors)} error(s)\n  - {error_details}"
        return Err(TypeMismatchError(msg, function=function_name, num_errors=len(errors)))
    
    return Ok(True)


def __validate_single_tensor_return(result: torch.Tensor, return_annotation: TensorAnnotation, dim_registry: dict, instance: Any, function_name: Optional[str]) -> Result[bool, ValidationError]:
    """Validate a single tensor return value against annotation."""
    return return_annotation.validate_result("return", result, dim_registry, instance, function_name)


def __validate_tuple_return_type(result: Any, return_annotation: Any, dim_registry: dict, instance: Any, function_name: Optional[str]) -> Result[bool, ValidationError]:
    """Validate return value when annotation specifies a tuple type."""
    args: tuple = get_args(return_annotation)
    if args and isinstance(result, tuple):
        return __validate_tuple_return(result, args, dim_registry, instance, function_name)
    return Ok(True)


def __validate_return_value(result: Any, return_annotation: Any, dim_registry: dict, instance: Any, function_name: Optional[str]) -> Result[bool, ValidationError]:
    """Validate return value against type annotation with instance context."""
    if return_annotation == inspect.Signature.empty:
        return Ok(True)
    
    processed_annotation = __process_optional_return_type(return_annotation, result)
    if processed_annotation is None:
        return Ok(True)
    
    origin = get_origin(processed_annotation)
    if origin is tuple:
        return __validate_tuple_return_type(result, processed_annotation, dim_registry, instance, function_name)
    elif isinstance(processed_annotation, TensorAnnotation) and isinstance(result, torch.Tensor):
        return __validate_single_tensor_return(result, processed_annotation, dim_registry, instance, function_name)
    
    return Ok(True)


def __get_function_name(func: Callable) -> str:
    """Generate fully qualified function name for error messages."""
    return f"{func.__module__}.{func.__qualname__}"


def __get_instance_from_args(bound_args: inspect.BoundArguments) -> Optional[Any]:
    """Extract instance object (self) from bound arguments if present."""
    return bound_args.arguments.get('self', None)


def __extract_from_union(args: tuple) -> Any:
    """Extract inner type T from Union[Ok[T], Err[E]]."""
    ok_args = get_args(args[0])
    return ok_args[0] if ok_args else None


def __unwrap_result_annotation(return_annotation: Any) -> Any:
    """Extract inner type T from Result[T, Error] annotation."""
    origin = get_origin(return_annotation)
    
    if origin is Union:
        args = get_args(return_annotation)
        if len(args) == 2:
            inner_type = __extract_from_union(args)
            if inner_type is not None:
                return inner_type
    
    return return_annotation


def __check_ok_err_types(args: tuple) -> bool:
    """Check if tuple contains Ok and Err type origins."""
    ok_type = get_origin(args[0])
    err_type = get_origin(args[1])
    if ok_type is None or err_type is None:
        return False
    return (ok_type.__name__ in ('Ok', 'Err')) and (err_type.__name__ in ('Ok', 'Err'))


def __is_result_type(return_annotation: Any) -> bool:
    """Check if return annotation is Result[T, E]."""
    origin = get_origin(return_annotation)
    if origin is Union:
        args = get_args(return_annotation)
        if len(args) == 2:
            return __check_ok_err_types(args)
    return False


def __is_result_value(value: Any) -> bool:
    """Check if value is a Result type (Ok or Err)."""
    return hasattr(value, 'is_ok') and hasattr(value, 'is_err')


def __validate_result_output(result: Any, return_annotation: Any, dim_registry: dict, instance: Any, func_name: str) -> Result[Any, ValidationError]:
    """Validate output from Result-returning function."""
    if not __is_result_value(result):
        raise InvalidReturnTypeError(
            message=f"Function {func_name} annotated to return Result but returned {type(result)}",
            function=func_name,
            expected="Result",
            actual=type(result).__name__
        )
    
    if result.is_err():
        return result  # Propagate function error
    
    # Validate the Ok value
    value = result.ok_value
    inner_type = __unwrap_result_annotation(return_annotation)
    output_validation = __validate_return_value(value, inner_type, dim_registry, instance, func_name)
    
    if output_validation.is_err():
        raise output_validation.err_value
    
    return Ok(value)


def __create_wrapper(func: Callable, sig: inspect.Signature, func_name: str, resolved_return_annotation: Any, detect_codes: frozenset[int] = frozenset()) -> Callable:
    """Create wrapper function with validation logic and optional NaN/Inf detection."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        # Get module for location tracking (args[0] is self for methods)
        module = args[0] if args and isinstance(args[0], nn.Module) else None
        
        # === SHAPE/DTYPE VALIDATION (runtime only, skipped during compile) ===
        if not torch.compiler.is_compiling():
            # Bind arguments and get context
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            instance = __get_instance_from_args(bound_args)
            dim_registry: dict = {}
            
            # Validate input arguments
            input_validation = __validate_input_arguments(sig, bound_args, dim_registry, instance, func_name)
            if input_validation.is_err():
                raise input_validation.err_value
        else:
            instance = module
            dim_registry = {}
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # === NaN/Inf AUTO-DETECTION (compiles! uses pure tensor ops) ===
        if detect_codes and isinstance(result, tuple) and len(result) >= 2:
            *outputs, flags = result
            
            # Only process if flags looks like error_t (int64, 2D)
            if (hasattr(flags, 'dtype') and flags.dtype == torch.int64 
                and hasattr(flags, 'ndim') and flags.ndim == 2):
                flags = __auto_detect_errors(tuple(outputs), flags, module, detect_codes)
                result = (*outputs, flags)
        
        # === OUTPUT VALIDATION (runtime only) ===
        if not torch.compiler.is_compiling():
            # Validate output based on return type
            returns_result = __is_result_type(resolved_return_annotation)
            
            if returns_result:
                # Handle Result-returning functions
                validation = __validate_result_output(result, resolved_return_annotation, dim_registry, instance, func_name)
                if validation.is_err():
                    raise validation.err_value
            else:
                # Plain return - validate and return as-is
                output_validation = __validate_return_value(result, resolved_return_annotation, dim_registry, instance, func_name)
                if output_validation.is_err():
                    raise output_validation.err_value
                
                # Optionally validate error_t if present
                if isinstance(result, tuple) and len(result) >= 2:
                    *_, maybe_flags = result
                    __validate_error_t(maybe_flags)
        
        return result
    
    return wrapper


def __resolve_return_annotation(func: Callable, sig: inspect.Signature) -> Any:
    """Resolve return annotation, handling PEP 563 string annotations."""
    try:
        hints = typing.get_type_hints(func)
        return hints.get('return', sig.return_annotation)
    except Exception:
        return sig.return_annotation


def __tensorcheck_function(func: Callable, *, auto_detect: AutoDetectType = True) -> Callable:
    """Handle @tensorcheck on a function/method."""
    sig: inspect.Signature = inspect.signature(func)
    func_name: str = __get_function_name(func)
    resolved_return_annotation = __resolve_return_annotation(func, sig)
    
    # Normalize auto_detect to frozenset
    if auto_detect is True:
        detect_codes = __get_default_codes()
    elif auto_detect is False or auto_detect is None:
        detect_codes = frozenset()
    else:
        detect_codes = frozenset(auto_detect)
    
    return __create_wrapper(func, sig, func_name, resolved_return_annotation, detect_codes)


def __auto_detect_errors(outputs: tuple, flags: torch.Tensor, module: nn.Module, codes: frozenset[int]) -> torch.Tensor:
    """Auto-detect specified error codes in output tensors and record to flags."""
    from .err.helpers import flag_nan, flag_inf
    from .core.codes import ErrorCode
    
    for out in outputs:
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            if ErrorCode.NAN in codes:
                flags = flag_nan(out, module, flags)
            if ErrorCode.INF in codes:
                flags = flag_inf(out, module, flags)
    
    return flags


def __get_default_codes() -> frozenset[int]:
    """Get default codes lazily to avoid import issues."""
    from .core.codes import ErrorCode
    return frozenset({ErrorCode.NAN, ErrorCode.INF})


def __tracked_class(cls: type) -> type:
    """Handle @tracked on a class."""
    if not issubclass(cls, nn.Module):
        raise TypeError(f"@tracked on class requires nn.Module subclass, got {cls}")
    
    original_init = cls.__init__
    
    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        __extract_and_inject_locations(self)
    
    cls.__init__ = new_init
    return cls


def tracked(cls_or_none=None):
    """
    Decorator for location tracking on nn.Module classes.
    
    Wraps __init__ to auto-extract and inject _fx_path for all submodules,
    enabling precise error location tracking throughout the module hierarchy.
    
    Args:
        cls_or_none: Class to decorate (must be nn.Module subclass)
    
    Returns:
        Decorated class with location injection
    
    Example:
        @tracked
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()  # Gets _fx_path = "encoder"
                self.decoder = Decoder()  # Gets _fx_path = "decoder"
    """
    def decorator(cls):
        return __tracked_class(cls)
    
    if cls_or_none is None:
        return decorator
    else:
        return decorator(cls_or_none)


def tensorcheck(func_or_none=None, *, auto_detect: AutoDetectType = True):
    """
    Decorator for tensor validation and NaN/Inf auto-detection on methods.
    
    Features:
        - Validates tensor shapes/dtypes from type hints (runtime only)
        - Auto-detects NaN/Inf in outputs (compiles with fullgraph=True!)
        - Returns function result as-is (no Result wrapping)
    
    Args:
        func_or_none: Function or method to decorate
        auto_detect: Error codes to auto-detect (True=NaN+Inf, False=none, or set of codes)
    
    Returns:
        Decorated function
    """
    def decorator(func):
        if isinstance(func, type):
            raise TypeError(
                f"@tensorcheck cannot be applied to classes. "
                f"Use @tracked for location injection on {func.__name__}."
            )
        return __tensorcheck_function(func, auto_detect=auto_detect)
    
    if func_or_none is None:
        return decorator
    else:
        return decorator(func_or_none)
