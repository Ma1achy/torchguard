"""``@tensorcheck``: validate shapes/dtypes from type hints and optionally
auto-flag NaN/Inf into a returned ``GuardedTensor``.

Behaviour:
* Shape/dtype/device validation runs **eagerly only** (skipped under
  ``torch.compile``) and **raises** on mismatch — a shape/dtype mismatch is a
  programming bug, not a data error.
* If ``auto_detect`` is set, NaN/Inf in a returned ``GuardedTensor`` are recorded
  into its flag channel at the decorated module's location. This path compiles.

Note: annotations must be real runtime objects (do not put
``from __future__ import annotations`` in the module defining the decorated
function, or the ``Tensor[...]`` annotations become strings this can't evaluate).
"""
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, get_args

import torch
import torch.nn as nn

from .codes import ErrorCode
from .detect import flag_inf, flag_nan
from .tensor import GuardedTensor
from .typing.annotation import TensorAnnotation

__all__ = ["tensorcheck"]

AutoDetect = bool | set | frozenset | None


def _codes(auto_detect: AutoDetect) -> frozenset[int]:
    if auto_detect is True:
        return frozenset({ErrorCode.NAN, ErrorCode.INF})
    if not auto_detect:  # False or None
        return frozenset()
    return frozenset(auto_detect)  # explicit set of codes


def _as_annotation(ann: Any) -> TensorAnnotation | None:
    """Return a ``TensorAnnotation`` from ``ann``, unwrapping ``Optional[...]``."""
    if isinstance(ann, TensorAnnotation):
        return ann
    for arg in get_args(ann):
        if isinstance(arg, TensorAnnotation):
            return arg
    return None


def _validate(value: Any, ann: Any, name: str, registry: dict, instance: Any,
              fname: str) -> None:
    annotation = _as_annotation(ann)
    if annotation is None or value is None:
        return
    if isinstance(value, torch.Tensor):  # GuardedTensor is a torch.Tensor too
        annotation.validate(name, value, registry, instance, fname)


def _autoflag(result: Any, codes: frozenset[int], module: nn.Module | None) -> Any:
    if isinstance(result, GuardedTensor):
        if ErrorCode.NAN in codes:
            result = flag_nan(result, location=module)
        if ErrorCode.INF in codes:
            result = flag_inf(result, location=module)
        return result
    if isinstance(result, (tuple, list)):
        return type(result)(_autoflag(r, codes, module) for r in result)
    return result


def _wrap(fn: Callable, auto_detect: AutoDetect) -> Callable:
    sig = inspect.signature(fn)
    fname = getattr(fn, "__qualname__", getattr(fn, "__name__", "function"))
    param_anns = {
        n: p.annotation
        for n, p in sig.parameters.items()
        if p.annotation is not inspect.Parameter.empty
    }
    return_ann = (
        sig.return_annotation
        if sig.return_annotation is not inspect.Signature.empty
        else None
    )
    codes = _codes(auto_detect)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        module = args[0] if args and isinstance(args[0], nn.Module) else None
        compiling = torch.compiler.is_compiling()

        if not compiling:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            registry: dict = {}
            for pname, value in bound.arguments.items():
                _validate(value, param_anns.get(pname), pname, registry, module, fname)
            result = fn(*args, **kwargs)
            if codes:
                result = _autoflag(result, codes, module)
            _validate(result, return_ann, "return", registry, module, fname)
            return result

        result = fn(*args, **kwargs)
        if codes:
            result = _autoflag(result, codes, module)
        return result

    return wrapper


def tensorcheck(func: Callable | None = None, *, auto_detect: AutoDetect = True) -> Callable:
    """Decorate a function/method to validate tensor annotations and auto-flag NaN/Inf.

    Args:
        func: The function being decorated (when used without parentheses).
        auto_detect: ``True`` for NaN+Inf, ``False``/``None`` for none, or an
            explicit set of ``ErrorCode`` values.
    """
    def decorator(fn: Callable) -> Callable:
        if isinstance(fn, type):
            raise TypeError("@tensorcheck cannot decorate a class; use @tracked instead.")
        return _wrap(fn, auto_detect)

    return decorator(func) if func is not None else decorator
