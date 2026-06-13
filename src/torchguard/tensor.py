"""``GuardedTensor``: a ``__torch_dispatch__`` subclass that carries a per-sample
error-flag channel alongside its data and propagates it through every operation.

The flags are an ordinary integer tensor stored as an inner tensor of the
subclass. Because the traceable-subclass protocol partitions differentiable
data (``_data``) from non-differentiable metadata (``_flags``), this composes
cleanly with autograd and ``torch.compile`` (validated: eager/inductor forward
and backward, plus variable batch via automatic dynamic shapes).

Note: explicitly calling ``torch._dynamo.mark_dynamic`` on a ``GuardedTensor``
dimension is unsupported (a PyTorch-internal limitation in symbolic-shape
propagation to subclass inner tensors). Automatic dynamic shapes handle
variable batch sizes without it.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils._pytree import tree_map

from . import flags as F
from .config import ErrorConfig, get_config

__all__ = ["GuardedTensor", "guard"]


class GuardedTensor(torch.Tensor):
    """A tensor that carries and auto-propagates a packed error-flag channel."""

    _data: Tensor
    _flags: Tensor

    @staticmethod
    def __new__(cls, data: Tensor, flags: Tensor) -> GuardedTensor:
        return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            data.shape,
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            device=data.device,
            requires_grad=data.requires_grad,
        )

    def __init__(self, data: Tensor, flags: Tensor) -> None:
        self._data = data
        self._flags = flags

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        # Must stay compile-safe: never force host values (no int()/.item()) here,
        # since torch may repr() a fake GuardedTensor while tracing.
        return f"GuardedTensor(shape={tuple(self.shape)}, dtype={self.dtype}, words={self._flags.shape[-1]})"

    # --- traceable-subclass protocol (so Dynamo/AOTAutograd see through us) ---
    def __tensor_flatten__(self) -> tuple[list[str], None]:
        return ["_data", "_flags"], None

    @staticmethod
    def __tensor_unflatten__(
        inner: dict, meta: Any, outer_size: Any, outer_stride: Any
    ) -> GuardedTensor:
        return GuardedTensor(inner["_data"], inner["_flags"])

    def _stable_hash_for_caching(self) -> str:
        return f"GuardedTensor(dtype={self._flags.dtype},words={self._flags.shape[-1]})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # noqa: ANN001
        kwargs = kwargs or {}
        collected: list[Tensor] = []

        def unwrap(x: Any) -> Any:
            if isinstance(x, GuardedTensor):
                collected.append(x._flags)
                return x._data
            return x

        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        if not collected:
            return out

        merged = collected[0]
        for extra in collected[1:]:
            merged = F.merge(merged, extra)

        def wrap(o: Any) -> Any:
            if isinstance(o, Tensor) and not isinstance(o, GuardedTensor):
                return GuardedTensor(o, merged)
            return o

        return tree_map(wrap, out)

    # --- Python-boundary helpers (return Python/host values) ---
    @property
    def flags(self) -> Tensor:
        """The raw packed flags tensor ``(N, num_words)``."""
        return self._flags

    def unwrap(self) -> Tensor:
        """Return the underlying plain data tensor."""
        return self._data

    def has_err(self) -> bool:
        """``True`` if any sample carries any error (Python boundary)."""
        return bool(F.any_err(self._flags))

    def is_err(self) -> Tensor:
        """Per-sample bool mask of samples with errors."""
        return F.is_err(self._flags)

    def is_ok(self) -> Tensor:
        """Per-sample bool mask of error-free samples."""
        return F.is_ok(self._flags)


def guard(x: Tensor, config: ErrorConfig | None = None) -> GuardedTensor:
    """Wrap ``x`` in a ``GuardedTensor`` with an empty error channel.

    Returns ``x`` unchanged if it is already a ``GuardedTensor``.
    """
    if isinstance(x, GuardedTensor):
        return x
    config = config or get_config()
    return GuardedTensor(x, F.new(x.shape[0], config, x.device))
