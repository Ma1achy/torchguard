# TorchGuard

Per-sample error tracking for `torch.compile()` models. One bad sample no longer kills
the whole batch: TorchGuard carries a compact, per-sample **error channel** alongside
your data through compiled graphs, marks the bad samples, and lets you inspect, drop, or
repair them at the Python boundary — without graph breaks.

```bash
pip install torchguard
```

Requires PyTorch ≥ 2.7.

```python
import torch
from torchguard import guard, flag_nan_inf

x = guard(torch.randn(32, 512))   # wrap once at the boundary
h = layer(x)                      # the error channel rides along automatically
h = flag_nan_inf(h, location=1)   # record NaN/Inf where they appear
if h.has_err():
    from torchguard import inspect
    print(inspect.report(h.flags))
```

---

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Using `torch.compile`](#using-torchcompile)
- [Location tracking](#location-tracking)
- [`@tensorcheck` and tensor typing](#tensorcheck-and-tensor-typing)
- [Inspecting errors](#inspecting-errors)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Development](#development)

---

## How it works

`GuardedTensor` is a `torch.Tensor` subclass (built on `__torch_dispatch__`) that wraps a
data tensor plus a small **flags** tensor — one bit-packed error record per sample in the
leading batch dimension. Every operation you run on a `GuardedTensor` transparently
propagates (and, where two guarded tensors meet, merges) that error channel, so the
information survives all the way to the output of a compiled region.

Because the subclass cleanly separates the differentiable data from the
non-differentiable flags, it composes with autograd and `torch.compile` — validated for
eager and `inductor`, forward and backward, including variable batch sizes via automatic
dynamic shapes.

**Flags layout.** Each error is a 16-bit slot packing `[location:10][code:4][severity:2]`;
slots are packed into integer words (`int64` by default, 4 slots/word). A sample is
error-free iff all its words are zero.

| Field | Bits | Meaning |
|---|---|---|
| severity | 2 | `OK` / `WARN` / `ERROR` / `CRITICAL` |
| code | 4 | `NAN`, `INF`, `OUT_OF_BOUNDS`, … (grouped into domains) |
| location | 10 | a registered module path id (0 = `UNKNOWN`) |

---

## Quick start

```python
import torch
import torch.nn as nn
from torchguard import guard, flag_nan_inf, track, inspect

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 256)
        self.head = nn.Linear(256, 128)

    def forward(self, x):
        x = self.encoder(x)
        x = flag_nan_inf(x, location=self.encoder)   # record + keep going
        x = self.head(x)
        x = flag_nan_inf(x, location=self.head)
        return x

model = track(Model())                 # register module paths for precise locations
out = model(guard(torch.randn(32, 512)))

out.has_err()        # -> bool: any error in the batch?
out.is_err()         # -> (32,) bool mask of bad samples
out.is_ok()          # -> (32,) bool mask of clean samples
out.unwrap()         # -> the plain data tensor
out.flags            # -> the raw (32, num_words) flags tensor

if out.has_err():
    print(inspect.report(out.flags))   # "GuardedTensor(32 samples, 3 errors): s7:NAN@head, ..."
```

### Drop or keep samples at the boundary

```python
mask = out.is_ok()
clean = out.unwrap()[mask]            # dynamic-shape filtering (Python boundary)
```

---

## Using `torch.compile`

`GuardedTensor` works under `torch.compile(fullgraph=True)`, including the backward pass:

```python
cmodel = torch.compile(model, fullgraph=True)   # backend="inductor" by default
out = cmodel(guard(torch.randn(32, 512)))
loss = out.square().mean()                       # keep it a GuardedTensor through the loss
loss.backward()                                  # gradients flow to the model weights
```

* **Backward runs on the `GuardedTensor`.** Keep the tensor guarded through your loss and
  call `.backward()` on it. `unwrap()` returns the plain data tensor *detached from the
  subclass's autograd graph*, so use it for inspection — not before `.backward()`.
* **Variable batch size** works out of the box via PyTorch's automatic dynamic shapes —
  just call the model with different batch sizes.
* **Wrap at the boundary**, return the `GuardedTensor`, and do inspection
  (`has_err`, `inspect.report`) *outside* the compiled region.
* **Known limitation:** explicitly calling `torch._dynamo.mark_dynamic` on a
  `GuardedTensor` dimension is unsupported (a PyTorch-internal limitation). Automatic
  dynamic shapes cover variable batches without it.

---

## Location tracking

Locations turn a raw slot into a human-readable origin like `encoder.ffn`. Register a
module tree once and pass the module as the `location`:

```python
from torchguard import track, tracked, flag_nan

model = track(model)                 # stamp every submodule's dotted path

# or as a decorator on the class:
@tracked
class Model(nn.Module):
    ...

# then, inside forward, location=self / a submodule resolves to that path:
x = flag_nan(x, location=self.encoder)
```

`location` also accepts a plain int id, a path string, or `None` (→ `UNKNOWN`). Resolution
happens at trace time and becomes a compile-time constant, so it is `torch.compile`-safe.

---

## `@tensorcheck` and tensor typing

Annotate tensors with shapes/dtypes and validate them at runtime:

```python
from torchguard import tensorcheck, guard
from torchguard.typing import Tensor, Dim, float32_t

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = 256
        self.lin = nn.Linear(512, 256)

    @tensorcheck
    def forward(
        self,
        x: Tensor[float32_t, ("N", 512)],
    ) -> Tensor[float32_t, ("N", Dim.hidden)]:
        return self.lin(x)
```

`@tensorcheck`:

* **Validates** input and return shapes/dtypes against the annotations and **raises**
  (`DTypeMismatchError`, `DimensionMismatchError`, …) on a mismatch — a contract violation
  is a bug, not a data error. Validation runs eagerly only and is skipped under
  `torch.compile`.
* **Auto-flags** NaN/Inf into the returned `GuardedTensor` at the module's location
  (`auto_detect=True` by default; pass `auto_detect=False` or a set of `ErrorCode`s). This
  path compiles.

The typing DSL supports named dims (`"N"`, consistency-checked across tensors),
`Dim.attr` references to instance attributes, `Broadcast`, and `Ellipsis`:

```python
Tensor[float32_t, ("N", "D")]
Tensor[float32_t, ("N", Dim.hidden)]
Tensor[float32_t, (Ellipsis, "seq", "hidden")]
Tensor[float32_t, (Broadcast, "features")]
```

> Annotations must be real runtime objects, so do **not** add
> `from __future__ import annotations` to a module whose functions you decorate with
> `@tensorcheck` (it would turn the annotations into strings).

---

## Inspecting errors

Python-boundary helpers (do not call inside a compiled region):

```python
from torchguard import inspect, flags, ErrorCode

inspect.report(out.flags)              # one-line human-readable summary
inspect.summary(out.flags)             # {"NAN": 3, "INF": 1}

flags.find(ErrorCode.NAN, out.flags)   # (N,) bool mask of NaN samples
flags.count(out.flags)                 # (N,) errors per sample
flags.unpack(out.flags[i])             # [(code, location, severity), ...] for sample i
```

Error codes are grouped into domains (`NUMERIC`, `INDEX`, `QUALITY`, `RUNTIME`) — see
`ErrorCode` / `ErrorDomain` / `Severity`.

---

## Configuration

```python
import torch
from torchguard import ErrorConfig, AccumulationPolicy, Order, Dedupe, set_config

set_config(ErrorConfig(
    num_slots=32,
    flag_dtype=torch.int64,
    accumulation=AccumulationPolicy(
        order=Order.FIFO,        # keep oldest (root cause) when slots fill up
        dedupe=Dedupe.PAIR,      # one slot per (location, code)
        evict_by_severity=True,  # under FIFO, a more severe error can evict a milder one
    ),
))
```

* `num_slots` — error slots per sample (default 16).
* `flag_dtype` — `torch.int64` (default, 4 slots/word) or `torch.int32` (2 slots/word).
* `accumulation` — how errors fill the fixed slots:
  * `order` — `LIFO` (default, keep newest) or `FIFO` (keep oldest / root cause).
  * `dedupe` — `NONE` (default), `CODE`, `LOCATION`, or `PAIR` (one slot per
    location+code; the kept slot upgrades to the worst severity seen).
  * `evict_by_severity` — under `FIFO`, replace the lowest-severity slot when full if the
    incoming error is more severe.

All policies are vectorized and `torch.compile`-safe. When two guarded tensors meet,
their slots are merged and compacted (truncated to `num_slots`).

---

## Limitations

* `mark_dynamic` on a `GuardedTensor` dim is unsupported (see above).
* `@tensorcheck` reads runtime annotations, so stringized (`from __future__`) annotations
  on decorated functions are not validated.
* Detection (`flag_nan`/`flag_inf`) is explicit by design — per-op auto-detection would
  add a reduction to every operation.
* Deduplication applies to `flag_*`/`push`; merging two guarded tensors compacts slots but
  does not itself deduplicate across them.

---

## Development

```bash
pip install -e ".[dev]"
pytest                 # test suite
ruff check .           # lint
mypy                   # type-check
```

CI runs the suite across Python 3.10–3.12 and torch {floor 2.7, latest}.
