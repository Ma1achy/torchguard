"""Tensor typing: annotation parsing and the validation engine (incl. bugfix)."""
from __future__ import annotations

import pytest
import torch

from torchguard import guard
from torchguard.typing import (
    Broadcast,
    Dim,
    DimensionMismatchError,
    DTypeMismatchError,
    Tensor,
    TensorAnnotation,
    float32_t,
    int64_t,
)


def test_annotation_parse():
    ann = Tensor[float32_t, ("N", "D")]
    assert isinstance(ann, TensorAnnotation)
    assert ann.dtype == torch.float32
    assert ann.shape == ("N", "D")
    assert Tensor[float32_t].shape is None


def test_validate_ok_and_dtype_mismatch():
    ann = Tensor[float32_t, ("N", "D")]
    assert ann.validate("x", torch.randn(4, 8)) is True
    with pytest.raises(DTypeMismatchError):
        ann.validate("x", torch.zeros(4, 8, dtype=torch.int64))


def test_literal_and_named_dims():
    assert Tensor[float32_t, (4, 8)].validate("x", torch.randn(4, 8))
    with pytest.raises(DimensionMismatchError):
        Tensor[float32_t, (4, 8)].validate("x", torch.randn(4, 9))
    # named-dim consistency across two tensors sharing "N"
    ann_x = Tensor[float32_t, ("N", "D")]
    ann_y = Tensor[int64_t, ("N",)]
    reg: dict = {}
    ann_x.validate("x", torch.randn(4, 8), reg)
    ann_y.validate("y", torch.zeros(4, dtype=torch.int64), reg)  # N=4 consistent
    with pytest.raises(DimensionMismatchError):
        ann_y.validate("y", torch.zeros(5, dtype=torch.int64), reg)  # N=5 != 4


def test_broadcast_and_ellipsis():
    Tensor[float32_t, (Broadcast, 8)].validate("b", torch.randn(1, 8))
    Tensor[float32_t, (Broadcast, 8)].validate("b", torch.randn(4, 8))
    ann = Tensor[float32_t, (Ellipsis, "H")]
    ann.validate("x", torch.randn(2, 3, 8))  # variable prefix, suffix H=8
    with pytest.raises(DimensionMismatchError):
        Tensor[float32_t, (Ellipsis, 5, 6)].validate("x", torch.randn(5))  # too few dims


def test_dim_attribute_ref():
    class Cfg:
        hidden = 8

    ann = Tensor[float32_t, ("N", Dim.hidden)]
    ann.validate("x", torch.randn(4, 8), {}, instance=Cfg())
    with pytest.raises(DimensionMismatchError):
        ann.validate("x", torch.randn(4, 9), {}, instance=Cfg())


def test_device_mismatch():
    ann = Tensor[float32_t, ("N",), "meta"]
    with pytest.raises(Exception):
        ann.validate("x", torch.randn(4))  # cpu != meta


def test_validate_result_returns_err_not_raises():
    ann = Tensor[float32_t, ("N", "D")]
    assert ann.validate_result("x", torch.randn(4, 8)).is_ok()
    assert ann.validate_result("x", torch.zeros(4, 8, dtype=torch.int64)).is_err()


def test_validate_result_catches_dim_resolution_errors():
    # BUGFIX: AttributeError / TypeError from Dim resolution must become Err, not escape.
    class Empty:
        pass

    class BadType:
        hidden = "not-an-int"

    ann = Tensor[float32_t, ("N", Dim.hidden)]
    missing = ann.validate_result("x", torch.randn(4, 8), {}, instance=Empty())
    assert missing.is_err()  # AttributeError -> Err
    bad = ann.validate_result("x", torch.randn(4, 8), {}, instance=BadType())
    assert bad.is_err()  # TypeError -> Err


def test_validate_accepts_guardedtensor():
    ann = Tensor[float32_t, ("N", "D")]
    assert ann.validate("x", guard(torch.randn(4, 8))) is True
