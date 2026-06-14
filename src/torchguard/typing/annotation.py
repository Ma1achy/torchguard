"""Runtime validation of tensors against ``Tensor[...]`` annotations."""
from __future__ import annotations

from typing import Any, Union

import torch

from ..result import Err, Ok, Result
from .dim import _AttributeRef, _BroadcastMarker
from .dtypes import PYTHON_TYPE_TO_TORCH_DTYPE
from .errors import (
    DeviceMismatchError,
    DimensionMismatchError,
    DTypeMismatchError,
    ValidationError,
)

__all__ = ["TensorAnnotation"]


class TensorAnnotation:
    """A parsed ``Tensor[dtype, shape, device, requires_grad]`` annotation."""

    def __init__(
        self,
        dtype: Any = None,
        shape: tuple | None = None,
        device: str | torch.device | None = None,
        requires_grad: bool | None = None,
    ) -> None:
        self.dtype = PYTHON_TYPE_TO_TORCH_DTYPE.get(dtype, dtype)
        self.shape = shape
        self.device = device
        self.requires_grad = requires_grad
        # PyG's type inspector pokes at these.
        self.__qualname__ = "Tensor"
        self.__module__ = "torch"

    # -- dimension resolution ------------------------------------------------
    def _resolve_dim(self, spec: Any, instance: Any, registry: dict) -> int | None:
        if isinstance(spec, int):
            return spec
        if isinstance(spec, _AttributeRef):
            return spec.resolve(instance)
        if isinstance(spec, str):
            return registry.get(spec)
        return None  # Broadcast / Ellipsis / unknown -> symbolic

    def _check_dim(self, name: str, idx: int, spec: Any, actual: int, registry: dict,
                   instance: Any, fn: str | None) -> None:
        if isinstance(spec, _BroadcastMarker):
            return
        if isinstance(spec, _AttributeRef):
            expected = spec.resolve(instance)
            if expected != actual:
                raise DimensionMismatchError(
                    message=f"Dimension '{spec}' mismatch for '{name}': "
                            f"expected {expected}, got {actual}",
                    expected=expected, actual=actual, dim_name=str(spec),
                    parameter=name, function=fn,
                )
        elif isinstance(spec, str):
            if spec in registry:
                if registry[spec] != actual:
                    raise DimensionMismatchError(
                        message=f"Dimension '{spec}' mismatch for '{name}': "
                                f"expected {registry[spec]}, got {actual}",
                        expected=registry[spec], actual=actual, dim_name=spec,
                        parameter=name, function=fn,
                    )
            else:
                registry[spec] = actual
        elif isinstance(spec, int):
            if spec != actual:
                raise DimensionMismatchError(
                    message=f"Dimension 'dim[{idx}]' mismatch for '{name}': "
                            f"expected {spec}, got {actual}",
                    expected=spec, actual=actual, dim_name=f"dim[{idx}]",
                    parameter=name, function=fn,
                )

    def _check_shape(self, name: str, tensor: torch.Tensor, registry: dict,
                     instance: Any, fn: str | None) -> None:
        shape = self.shape
        assert shape is not None
        if Ellipsis in shape:
            i = shape.index(Ellipsis)
            prefix, suffix = shape[:i], shape[i + 1:]
            if tensor.ndim < len(prefix) + len(suffix):
                raise DimensionMismatchError(
                    message=f"Tensor '{name}' has {tensor.ndim} dims but annotation "
                            f"requires at least {len(prefix) + len(suffix)}",
                    parameter=name, function=fn,
                )
            for idx, spec in enumerate(prefix):
                self._check_dim(name, idx, spec, tensor.shape[idx], registry, instance, fn)
            for j, spec in enumerate(suffix):
                idx = tensor.ndim - len(suffix) + j
                self._check_dim(name, idx, spec, tensor.shape[idx], registry, instance, fn)
        else:
            if len(shape) != tensor.ndim:
                raise DimensionMismatchError(
                    message=f"Tensor '{name}' dim count mismatch: expected {len(shape)}, "
                            f"got {tensor.ndim} with shape {tuple(tensor.shape)}",
                    expected=len(shape), actual=tensor.ndim, parameter=name, function=fn,
                )
            for idx, (spec, actual) in enumerate(zip(shape, tensor.shape, strict=False)):
                self._check_dim(name, idx, spec, actual, registry, instance, fn)

    # -- public API ----------------------------------------------------------
    def validate(self, name: str, tensor: torch.Tensor, dim_registry: dict | None = None,
                 instance: Any = None, function_name: str | None = None) -> bool:
        """Validate ``tensor`` against this annotation; raise on mismatch.

        Works on a ``GuardedTensor`` directly (its ``shape``/``dtype``/``device``
        reflect the underlying data).
        """
        registry = dim_registry if dim_registry is not None else {}
        if self.dtype is not None and tensor.dtype != self.dtype:
            raise DTypeMismatchError(
                message=f"dtype mismatch for '{name}': expected {self.dtype}, got {tensor.dtype}",
                expected=self.dtype, actual=tensor.dtype, parameter=name, function=function_name,
            )
        if self.device is not None and str(self.device) != str(tensor.device):
            raise DeviceMismatchError(
                message=f"device mismatch for '{name}': expected {self.device}, got {tensor.device}",
                expected=str(self.device), actual=str(tensor.device),
                parameter=name, function=function_name,
            )
        if self.requires_grad is not None and tensor.requires_grad != self.requires_grad:
            raise ValidationError(
                message=f"requires_grad mismatch for '{name}': "
                        f"expected {self.requires_grad}, got {tensor.requires_grad}",
                context={"parameter": name, "function": function_name},
            )
        if self.shape is not None:
            self._check_shape(name, tensor, registry, instance, function_name)
        return True

    def validate_result(self, name: str, tensor: torch.Tensor, dim_registry: dict | None = None,
                        instance: Any = None,
                        function_name: str | None = None) -> Result[bool, ValidationError]:
        """Like :meth:`validate` but return ``Ok``/``Err`` instead of raising.

        Catches ``AttributeError``/``TypeError`` from ``Dim`` resolution too, so a
        bad attribute reference becomes an ``Err`` rather than escaping.
        """
        try:
            self.validate(name, tensor, dim_registry, instance, function_name)
            return Ok(True)
        except ValidationError as e:
            return Err(e)
        except (ValueError, AttributeError, TypeError) as e:
            return Err(ValidationError(
                message=str(e), context={"parameter": name, "function": function_name},
            ))

    def __repr__(self) -> str:
        parts = []
        if self.dtype:
            parts.append(str(self.dtype).replace("torch.", ""))
        if self.shape:
            parts.append(str(self.shape))
        if self.device:
            parts.append(f"device={self.device}")
        if self.requires_grad is not None:
            parts.append(f"requires_grad={self.requires_grad}")
        return f"Tensor[{', '.join(parts)}]"

    def __or__(self, other: Any) -> Any:
        # Must use typing.Union, not ``self | other`` (that would recurse here).
        return Union[self, other]  # noqa: UP007

    def __ror__(self, other: Any) -> Any:
        return Union[other, self]  # noqa: UP007
