"""Real-model smoke tests: transformer (attention/SDPA/layernorm/FFN),
embedding LM, and autocast — eager, compiled, and backward."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from torchguard import GuardedTensor, flag_nan_inf, guard, track


def _encoder(layers=2):
    layer = nn.TransformerEncoderLayer(16, 4, dim_feedforward=32, batch_first=True, dropout=0.0)
    return nn.TransformerEncoder(layer, num_layers=layers).eval()


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_transformer_compiles_and_attributes(backend):
    enc = _encoder()
    x = torch.randn(4, 6, 16)
    x[2, 0, 0] = float("nan")  # one position of sample 2 -> whole sample via attention
    cenc = torch.compile(enc, fullgraph=True, backend=backend, dynamic=False)
    out = flag_nan_inf(cenc(guard(x)), location=1)
    assert isinstance(out, GuardedTensor)
    assert out.is_err().tolist() == [False, False, True, False]


def test_transformer_backward_reaches_params():
    enc = _encoder()
    enc.zero_grad()
    out = flag_nan_inf(enc(guard(torch.randn(4, 6, 16))), location=1)
    out.square().mean().backward()
    grad = enc.layers[0].linear1.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_sdpa_propagates_flags():
    q = guard(torch.randn(4, 6, 16))
    k = torch.randn(4, 6, 16)
    v = torch.randn(4, 6, 16)
    out = Fn.scaled_dot_product_attention(q, k, v)
    assert isinstance(out, GuardedTensor)


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 16)
        self.enc = _encoder(1)
        self.head = nn.Linear(16, 100)

    def forward(self, ids):
        return flag_nan_inf(self.head(self.enc(self.emb(ids))), location=self.head)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_embedding_lm(backend):
    lm = track(_LM())
    ids = torch.randint(0, 100, (4, 6))
    fn = torch.compile(lm, fullgraph=True, backend=backend, dynamic=False)
    out = fn(guard(ids))
    assert isinstance(out, GuardedTensor)
    assert tuple(out.unwrap().shape) == (4, 6, 100)
    assert not out.has_err()  # clean ids


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_autocast_bf16(backend):
    w = nn.Linear(8, 8)
    x = torch.randn(4, 8)
    x[1] = float("nan")

    def f(gx):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return flag_nan_inf(w(gx), location=1)

    fn = torch.compile(f, fullgraph=True, backend=backend, dynamic=False)
    out = fn(guard(x))
    assert out.unwrap().dtype == torch.bfloat16
    assert out.flags.dtype == torch.int64  # flags stay int under AMP
    assert out.is_err().tolist() == [False, True, False, False]


def test_autocast_bf16_backward():
    w = nn.Linear(8, 8)
    w.zero_grad()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = flag_nan_inf(w(guard(torch.randn(4, 8))), location=1)
    out.float().square().mean().backward()
    assert torch.isfinite(w.weight.grad).all()
