"""Recovery helpers: fix() (fallback replacement) and flag_oob_indices()."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchguard import ErrorCode, GuardedTensor, fix, flag_nan, flag_oob_indices, guard
from torchguard import flags as F


def test_fix_replaces_error_samples():
    d = torch.randn(4, 8)
    d[1] = float("nan")
    g = flag_nan(d, location=1)
    out = fix(g, fallback=0.0, location=2)
    assert isinstance(out, GuardedTensor)
    assert torch.isfinite(out.unwrap()).all()                  # NaN replaced
    assert (out.unwrap()[1] == 0.0).all()                      # sample 1 -> fallback
    assert F.find(ErrorCode.FALLBACK_VALUE, out.flags).tolist() == [False, True, False, False]
    assert F.find(ErrorCode.NAN, out.flags).tolist() == [False, True, False, False]  # original kept


def test_fix_only_replaces_targeted_codes():
    from torchguard import flag_inf

    d = torch.randn(3, 4)
    d[0] = float("inf")
    g = flag_inf(guard(d), location=1)
    out = fix(g, fallback=1.0, codes=[ErrorCode.NAN])  # only NaN targeted -> Inf sample untouched
    assert not torch.isfinite(out.unwrap()[0]).all()   # inf NOT replaced
    assert F.find(ErrorCode.FALLBACK_VALUE, out.flags).tolist() == [False, False, False]


def test_fix_preserves_gradient():
    w = nn.Linear(8, 8)
    out = fix(flag_nan(w(guard(torch.randn(4, 8))), location=1), fallback=0.0, location=2)
    out.square().mean().backward()
    assert torch.isfinite(w.weight.grad).all()


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_fix_compiles(backend):
    def f(gx):
        return fix(flag_nan(gx, location=1), fallback=0.0, location=2)

    x = torch.randn(4, 8)
    x[1] = float("nan")
    out = torch.compile(f, fullgraph=True, backend=backend, dynamic=False)(guard(x))
    assert torch.isfinite(out.unwrap()).all()


def test_flag_oob_indices():
    ids = torch.tensor([[1, 2, 3], [5, 200, 0], [0, 0, 0], [-1, 4, 4]])
    g = flag_oob_indices(ids, num_embeddings=100, location="emb")
    assert F.find(ErrorCode.OUT_OF_BOUNDS, g.flags).tolist() == [False, True, False, True]


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_flag_oob_indices_compiles(backend):
    ids = torch.tensor([[1, 2], [200, 0], [0, 0], [4, 4]])
    fn = torch.compile(lambda i: flag_oob_indices(i, 100, location="emb"),
                       fullgraph=True, backend=backend, dynamic=False)
    out = fn(guard(ids))
    assert F.find(ErrorCode.OUT_OF_BOUNDS, out.flags).tolist() == [False, True, False, False]


def test_oob_then_fix_then_embed():
    # end-to-end recovery: flag OOB -> clamp bad indices -> embed safely
    ids = torch.tensor([[1, 2], [200, 0], [0, 0], [-1, 4]])
    g = flag_oob_indices(ids, 100, location="emb")
    clamped = fix(g, fallback=0, codes=[ErrorCode.OUT_OF_BOUNDS])
    emb = nn.Embedding(100, 16)
    out = emb(clamped)
    assert isinstance(out, GuardedTensor)
    assert out.is_err().tolist() == [False, True, False, True]  # flags survive the lookup
    assert torch.isfinite(out.unwrap()).all()
