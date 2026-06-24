"""Edge cases for GuardedTensor and the flag engine."""
from __future__ import annotations

import pytest
import torch

from torchguard import ErrorConfig, GuardedTensor, Severity, flag_nan, guard
from torchguard import flags as F


# ---- batch shape edges -------------------------------------------------------
def test_guard_rejects_0d():
    with pytest.raises(ValueError, match="leading batch dimension"):
        guard(torch.tensor(3.0))


def test_empty_batch():
    g = flag_nan(torch.randn(0, 8), location=1)
    assert tuple(g.flags.shape) == (0, F.new(0).shape[1])
    assert g.has_err() is False
    assert tuple(g.is_err().shape) == (0,)
    assert g.take_ok().unwrap().shape[0] == 0


def test_single_sample():
    g = flag_nan(torch.full((1, 8), float("nan")), location=1)
    assert g.has_err() is True
    assert g.is_err().tolist() == [True]


# ---- tensor protocol edges ---------------------------------------------------
def test_clone_detach_preserve_flags():
    d = torch.randn(4, 8)
    d[1] = float("nan")
    g = flag_nan(d, location=1)
    for h in (g.clone(), g.detach()):
        assert isinstance(h, GuardedTensor)
        assert h.is_err().tolist() == [False, True, False, False]


def test_dtype_cast_keeps_int_flags():
    g = flag_nan(torch.randn(4, 8), location=1)
    h = g.to(torch.float64)
    assert h.unwrap().dtype == torch.float64
    assert h.flags.dtype == torch.int64  # flags stay integer regardless of data dtype


def test_flags_tensor_serializes(tmp_path):
    d = torch.randn(4, 8)
    d[2] = float("nan")
    g = flag_nan(d, location=3)
    path = tmp_path / "flags.pt"
    torch.save(g.flags, path)
    loaded = torch.load(path)
    assert torch.equal(loaded, g.flags)
    assert F.is_err(loaded).tolist() == [False, False, True, False]


# ---- batch-dim realignment ---------------------------------------------------
def test_slice_realigns_flags():
    d = torch.randn(4, 8)
    d[1] = float("nan")
    g = flag_nan(d, location=1)
    s = g[:2]
    assert s.unwrap().shape[0] == 2
    assert s.flags.shape[0] == 2
    assert s.is_err().tolist() == [False, True]


def test_index_realigns_flags():
    d = torch.randn(4, 8)
    d[1] = d[3] = float("nan")
    g = flag_nan(d, location=1)
    assert g[torch.tensor([1, 3])].is_err().tolist() == [True, True]
    assert g[~g.is_err()].has_err() is False  # boolean-mask drop


def test_non_batch_slice_leaves_flags():
    g = flag_nan(torch.randn(4, 8), location=1)
    assert g[:, :4].flags.shape[0] == 4  # slicing dim 1 must not subset flags


def test_cat_dim0_concatenates_flags():
    a = torch.randn(4, 8)
    a[1] = float("nan")
    ga = flag_nan(a, location=1)
    gb = flag_nan(torch.randn(2, 8), location=2)
    c = torch.cat([ga, gb], dim=0)
    assert c.unwrap().shape[0] == 6
    assert c.flags.shape[0] == 6
    assert c.is_err().tolist() == [False, True, False, False, False, False]


def test_subset_and_partition():
    d = torch.randn(5, 4)
    d[0] = d[3] = float("nan")
    g = flag_nan(d, location=1)
    ok, err = g.partition()
    assert ok.unwrap().shape[0] == 3 and not ok.has_err()
    assert err.unwrap().shape[0] == 2 and bool(err.is_err().all())


# ---- flag layout edges -------------------------------------------------------
def test_packing_boundaries():
    # max values for each field
    slot = F.pack_slot_scalar(code=0xF, location=0x3FF, severity=0x3)
    f = torch.tensor([[0]], dtype=torch.int64)
    f[0, 0] = slot
    assert F.unpack(f[0]) == [(0xF, 0x3FF, 0x3)]


def test_location_overflow_wraps():
    # location is 10 bits; 1024 wraps to 0
    assert F.pack_slot_scalar(1, 1024, 2) == F.pack_slot_scalar(1, 0, 2)


def test_multi_word_layout():
    cfg = ErrorConfig(num_slots=8)  # int64 -> 4 slots/word -> 2 words
    assert cfg.num_words == 2
    f = F.new(1, cfg)
    for i in range(6):
        f = F.push(f, torch.tensor([1 + (i % 3)], dtype=torch.int64), i + 1, Severity.ERROR, cfg)
    assert int(F.count(f, cfg)[0]) == 6  # spans both words


def test_num_slots_one():
    cfg = ErrorConfig(num_slots=1)
    f = F.push(F.new(1, cfg), torch.tensor([1], dtype=torch.int64), 1, Severity.CRITICAL, cfg)
    f = F.push(f, torch.tensor([2], dtype=torch.int64), 2, Severity.CRITICAL, cfg)
    assert int(F.count(f, cfg)[0]) == 1  # only newest kept (LIFO)
    assert F.unpack(f[0], cfg)[0][0] == 2


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_both_dtypes_roundtrip(dtype):
    cfg = ErrorConfig(num_slots=8, flag_dtype=dtype)
    f = F.push(F.new(2, cfg), torch.tensor([1, 0], dtype=dtype), 5, Severity.CRITICAL, cfg)
    assert F.is_err(f).tolist() == [True, False]
    assert F.unpack(f[0], cfg) == [(1, 5, Severity.CRITICAL)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_move_keeps_flags_aligned():
    g = flag_nan(torch.randn(4, 8).cuda(), location=1).cpu()
    assert g.unwrap().device.type == "cpu"
    assert g.flags.device.type == "cpu"
