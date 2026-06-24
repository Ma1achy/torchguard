"""Unit tests for the Result (Ok/Err) type."""
from __future__ import annotations

import pytest

from torchguard.result import Err, Ok


def test_ok_basics():
    r = Ok(5)
    assert r.is_ok() and not r.is_err()
    assert r.unwrap() == 5
    assert r.ok_value == 5
    assert r.unwrap_or(0) == 5
    assert r.map(lambda x: x + 1).unwrap() == 6
    assert r.and_then(lambda x: Ok(x * 2)).unwrap() == 10
    with pytest.raises(ValueError):
        r.unwrap_err()


def test_err_basics():
    r = Err("boom")
    assert r.is_err() and not r.is_ok()
    assert r.err_value == "boom"
    assert r.unwrap_or(42) == 42
    assert r.unwrap_err() == "boom"
    assert r.map(lambda x: x + 1).is_err()           # map is a no-op on Err
    assert r.map_err(str.upper).err_value == "BOOM"
    assert r.and_then(lambda x: Ok(x)).is_err()      # short-circuits


def test_err_unwrap_raises_wrapped_exception():
    r = Err(KeyError("k"))
    with pytest.raises(KeyError):
        r.unwrap()


def test_err_unwrap_non_exception():
    with pytest.raises(ValueError):
        Err("plain").unwrap()
