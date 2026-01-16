# TorchGuard

Per-sample error tracking for `torch.compile()` models, so one broken sample no longer kills the whole batch. TorchGuard keeps the batch, marks the bad samples, and lets you log or repair them instead of breaking the graph.

```bash
pip install torchguard
```

Requires PyTorch ≥ 2.0 (≥ 2.7 recommended for `torch.compile(fullgraph=True)`).

NaN/Inf values can propagate through compiled graphs, but most real training/inference pipelines react to them (exceptions, asserts, AMP overflow logic, logging hooks), causing graph breaks, recompiles, or step failures. TorchGuard keeps an in-graph, per-sample error channel so one bad sample doesn't poison the whole step.

* Per-sample error tracking inside compiled graphs (no graph breaks)
* One bad token does not mean drop 32 examples
* Exact location of every NaN/Inf/OOB, even in nested submodules
* Compiled-safe conditional recovery (`torch.where`-based DSL)

```python
output, f = model(x)          # works with torch.compile(fullgraph=True)
if has_err(f):
    print(flags.repr(f))
    # "3xNAN @ encoder.ffn.2, 1xINF @ classifier"
```

---

## Contents

* [Before vs After](#before-vs-after)
* [Quick Start](#quick-start)
* [Common Patterns](#common-patterns)
* [Core Concepts](#core-concepts)
* [API Reference](#api-reference)
* [Location Tracking](#location-tracking-tracked)
* [Tensor Typing System](#tensor-typing-system)
* [Configuration](#configuration)
* [Performance](#performance-benchmarks)
* [When NOT to Use](#when-not-to-use-torchguard)
* [Experimental Backend](#experimental-backend)

### At a glance

- **Works with `torch.compile(fullgraph=True)`**: per-sample error tracking without graph breaks
- **Bit-packed flags**: error slots stored in a compact `(batch, num_words)` flags tensor
- **Optional control-flow DSL**: `torch.where`-based `IF`/`ELIF`/`ELSE` for recovery inside compiled graphs
- **Configurable accumulation**: FIFO/LIFO, severity-based policies, and deduplication options

---

## Before vs After

### Without TorchGuard: typical error handling causes graph breaks

```python
@torch.compile(fullgraph=True)
def forward(self, ids):
    embed = self.embedding(ids)
    # NaN here will propagate through subsequent ops
    out = self.classifier(embed)
    # Boundary checks, AMP logic, or assertions react to NaN
    # → RuntimeError, graph breaks, batch lost, or recompile
    return out
```

### With TorchGuard: only bad samples are marked

```python
@torch.compile(fullgraph=True)
def forward(self, ids):
    f = err.new(ids)
    embed = self.embedding(ids)
    f = flag_nan(embed, self.embedding, f)   # records location, never raises
    out = self.classifier(embed)
    f = flag_nan(out, self.classifier, f)
    return out, f                            # all 32 samples returned

# downstream
ok_output = err.take_ok(f, out)    # shape (29, ...) - only clean samples
err_output = err.take_err(f, out)  # shape (3, ...) - samples with errors
```

---

### Conceptual Model

TorchGuard does **not** replace normal Python exceptions at the Python boundary. Instead, it gives you a **tensor-based error channel** you can carry through compiled regions.

TorchGuard does not prevent floating-point NaNs from existing; it prevents control-flow reactions to them from breaking compiled execution.

* **Inside compiled regions**: return `(output, f)` where `f` is the per-sample flags tensor; avoid `flags.*` inspection calls (they return Python values and will cause graph breaks).
* **At the Python boundary**: inspect `f` with `flags.*`, log/aggregate, and decide how to handle bad samples.

> **Backends at a glance**
>
> * **Stable (default):** `from torchguard import err, flags, ...`
>   — `int64` bitpacking, best for eager or light compile usage.
> * **Experimental (compile-focused):** `from torchguard.experimental import err, IF, IS, ...`
>   — `float32` storage (configurable), best for `torch.compile(fullgraph=True)` training with `inductor` backend.

---

## Quick Start

The main entrypoints are:

* `err` – tensor-only, `torch.compile`-safe operations
* `flags` – Python-side inspection utilities (**do not call inside compiled regions** - these return Python values and cause graph breaks)
* Helper functions – `flag_nan`, `flag_inf`, `fix`, etc. (tensor-returning helpers are safe inside compiled regions; `has_err` returns a Python bool and is intended for the boundary)

If you only read one section after Quick Start, read **Common Patterns**.

```python
import torch
import torch.nn as nn
from torchguard import err, flags, has_err, flag_nan, flag_nan_and_inf, tracked

@tracked                              # enables location names
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512, 256)
        self.layer2 = nn.Linear(256, 128)

    def forward(self, x):
        f = err.new(x)                # create per-sample error tracker
        x = self.layer1(x)
        f = flag_nan(x, self.layer1, f)
        x = self.layer2(x)
        f = flag_nan(x, self.layer2, f)
        return x, f

model = torch.compile(MyModel(), fullgraph=True)
out, f = model(torch.randn(32, 512))
# f: flags tensor, shape (batch, num_words), dtype int64 (default)
#     Change backend via tg.CONFIG.flag_dtype = torch.float32 for experimental backend

if has_err(f):
    print(flags.repr(f))
    # ErrorFlags(32 samples, 4 errors: 4xNAN @ layer2)
```

---

## Common Patterns

### Drop bad samples (training)

```python
# Only compute loss on clean samples (at Python boundary)
clean_logits = err.take_ok(f, logits)
clean_targets = err.take_ok(f, targets)
loss = criterion(clean_logits, clean_targets)

# Inside torch.compile(fullgraph=True), use static-shape variant:
clean_logits = err.take_ok_p(f, logits, fill=0.0)
```

### Replace bad samples with fallback (inference)

```python
# Replace NaN/Inf with zeros and continue
logits, f = fix(logits, f, self.head, fallback=0.0)
```

### Log and continue (monitoring)

```python
if has_err(f):
    logger.warning("Bad samples: %s", flags.summary(f))
    # {'encoder': {'NAN': 3}, 'classifier': {'INF': 1}}
```

### Partition for separate handling

```python
# At Python boundary (dynamic shapes OK):
ok_out, err_out = err.partition(f, output)
# Process ok_out normally, handle err_out separately

# Recover indices of bad samples if you want to log / trace them
bad_indices = torch.nonzero(err.is_err(f)).squeeze(-1)

# Inside torch.compile(fullgraph=True), use static-shape variants:
ok_out = err.take_ok_p(f, output, fill=0.0)
err_out = err.take_err_p(f, output, fill=0.0)
```

### Conditional recovery inside compiled graph

```python
from torchguard.experimental import err, IF, IS
from torchguard import fix

z, f = (
    IF(IS(err.NAN, f), lambda: fix(z, f, self.layer))
    .ELSE(lambda: (z.clone(), f.clone()))
)
```

---

## Core Concepts

### Terminology

* **Sample** – One row in the leading batch dimension
* **Flags tensor** – Shape `(batch, num_words)`, dtype `int64`. Each sample gets its own error slots (or float32/float64 when using the [experimental backend](#experimental-backend); the bit layout is identical)
* **`error_t`** – A type alias used in annotations for an `int64` tensor of shape `(batch, num_words)` (or float carrier with identical bit layout when using the experimental backend)

### Flags Tensor Layout

Each error is packed into a 16-bit slot. Four slots fit into one 64-bit word:

```text
                            64-bit Word (int64)
    +-------------------------------------------------------------------+
    |  +-------------+ +-------------+ +-------------+ +-------------+  |
    |  |   Slot 3    | |   Slot 2    | |   Slot 1    | |   Slot 0    |  |
    |  |  bits 63-48 | |  bits 47-32 | |  bits 31-16 | |  bits 15-0  |  |
    |  +-------------+ +-------------+ +-------------+ +-------------+  |
    +-------------------------------------------------------------------+
                                      |
                      +---------------+---------------+
                      |      16-bit Slot Detail       |
                      +-----------+--------+----------+
                      |  location |  code  | severity |
                      |  10 bits  | 4 bits |  2 bits  |
                      |  (0-1023) | (0-15) |  (0-3)   |
                      +-----------+--------+----------+
```

**Storage:**

* 4 slots per `int64` word (16 bits × 4 = 64 bits)
* Default: 16 slots = 4 words per sample
* Tensor shape: `(N, num_words)` where `N` = batch size

**Semantics:**

* All words zero for a sample = no errors
* Any word non-zero = at least one error
* Use `err.is_ok(f)` / `has_err(f)` rather than comparing directly

### Error Codes and Domains

Error codes are organized into **domains** for quick filtering. Each 4-bit code contains a 2-bit domain and 2-bit subcode.

**Error Domains:**

| Domain      | Value | Description                       | Query with              |
| ----------- | ----- | --------------------------------- | ----------------------- |
| `NUMERIC`   | 0     | Numerical issues (NaN, Inf)       | `err.has_domain(f, 0)`  |
| `INDEX`     | 1     | Indexing issues (OOB, negative)   | `err.has_domain(f, 1)`  |
| `QUALITY`   | 2     | Output quality (zero, constant)   | `err.has_domain(f, 2)`  |
| `RUNTIME`   | 3     | Runtime recovery (fallback, clamp)| `err.has_domain(f, 3)`  |

```python
from torchguard import ErrorDomain

# Check if any sample has numeric errors (NaN, Inf, or Overflow)
numeric_mask = err.has_domain(flags, ErrorDomain.NUMERIC)

# Check for any runtime recovery operations
runtime_mask = err.has_domain(flags, ErrorDomain.RUNTIME)
```

**Error Codes:**

| Domain      | Code                    | Value | Description           | Used by                    |
| ----------- | ----------------------- | ----- | --------------------- | -------------------------- |
| **NUMERIC** | `OK`                    | 0     | No error              |                            |
|             | `NAN`                   | 1     | Not-a-Number          | `flag_nan`, `@tensorcheck` |
|             | `INF`                   | 2     | Infinity              | `flag_inf`, `@tensorcheck` |
|             | `OVERFLOW`              | 3     | Numeric overflow      | manual `push`              |
| **INDEX**   | `OUT_OF_BOUNDS` / `OOB` | 5     | Index out of range    | `flag_oob_indices`         |
|             | `NEGATIVE_IDX`          | 6     | Negative index        |                            |
|             | `EMPTY_INPUT`           | 7     | Empty input tensor    |                            |
| **QUALITY** | `ZERO_OUTPUT`           | 9     | All-zero output       |                            |
|             | `CONSTANT_OUTPUT`       | 10    | Constant output       |                            |
|             | `SATURATED`             | 11    | Saturated activations |                            |
| **RUNTIME** | `FALLBACK_VALUE`        | 13    | Fallback value used   | `fix`                      |
|             | `VALUE_CLAMPED`         | 14    | Values were clamped   |                            |

### Severity Levels

| Level      | Value | Default For                                                     |
| ---------- | ----- | --------------------------------------------------------------- |
| `OK`       | 0     | No error                                                        |
| `WARN`     | 1     | `ZERO_OUTPUT`, `CONSTANT_OUTPUT`, `SATURATED`, `FALLBACK_VALUE` |
| `ERROR`    | 2     | `OUT_OF_BOUNDS`, `NEGATIVE_IDX`, `OVERFLOW`, `EMPTY_INPUT`      |
| `CRITICAL` | 3     | `NAN`, `INF`                                                    |

You can override severity when pushing manually: `err.push(..., severity=err.WARN)`.

---

## API Reference

Quick Start + Common Patterns cover most use cases. The sections below are reference/advanced.

<details>
<summary><strong>Core API – <code>err</code> Namespace (Compiled-Safe)</strong></summary>

The `err` namespace is the primary API for all operations inside compiled regions.

### Error Codes as Attributes

```python
from torchguard import err

err.NAN       # 1
err.INF       # 2
err.OVERFLOW  # 3
err.OOB       # 5 (alias for OUT_OF_BOUNDS)

err.CRITICAL  # 3
err.ERROR     # 2
err.WARN      # 1
err.OK        # 0
```

### Creation Methods

| Method                             | Description                              |
| ---------------------------------- | ---------------------------------------- |
| `err.new(x)`                       | Create empty flags from reference tensor |
| `err.new_t(n, device, config)`     | Create empty flags with explicit args    |
| `err.from_code(code, loc, n, ...)` | Create flags with single error           |

### Recording Methods

| Method                                         | Description                     |
| ---------------------------------------------- | ------------------------------- |
| `err.push(f, code, location, severity, where)` | Push error where mask is `True` |
| `err.push_scalar(f, code, location, severity)` | Push same error to all samples  |
| `err.merge(f1, f2, ...)`                       | Merge multiple flag tensors     |

### Querying Methods

| Method                    | Returns      | Description                              |
| ------------------------- | ------------ | ---------------------------------------- |
| `err.is_ok(f)`            | `(N,) bool`  | `True` where sample has **no** errors    |
| `err.is_err(f)`           | `(N,) bool`  | `True` where sample **has** errors       |
| `err.all_ok(f)`           | `() bool`    | Scalar: all samples OK?                  |
| `err.any_err(f)`          | `() bool`    | Scalar: any errors?                      |
| `err.has_nan(f)`          | `(N,) bool`  | Per-sample NaN check                     |
| `err.has_inf(f)`          | `(N,) bool`  | Per-sample Inf check                     |
| `err.has_code(f, code)`   | `(N,) bool`  | Per-sample code check                    |
| `err.has_critical(f)`     | `(N,) bool`  | Per-sample critical check                |
| `err.has_domain(f, dom)`  | `(N,) bool`  | Per-sample domain check (NUMERIC, etc.)  |
| `err.has_fallback(f)`     | `(N,) bool`  | Per-sample check for fallback values     |
| `err.count_errors(f)`     | `(N,) int32` | Error count per sample                   |
| `err.max_severity(f)`     | `(N,) int64` | Max severity per sample                  |

### Filtering Methods

| Method                        | Returns            | Description                        |
| ----------------------------- | ------------------ | ---------------------------------- |
| `err.take_ok(f, z)`           | `Tensor`           | Filter tensor `z` to OK samples    |
| `err.take_err(f, z)`          | `Tensor`           | Filter tensor `z` to error samples |
| `err.partition(f, z)`         | `(Tensor, Tensor)` | Split `z` into `(ok_z, err_z)`     |
| `err.partition_many(f, *zs)`  | tuple              | Split multiple tensors             |

Filtering is applied along the leading (batch) dimension of `z`. These methods use boolean indexing internally and are equivalent to older `err.Ok` / `err.Err` names (which are kept as aliases).

**Dynamic vs Static Shape Filtering:**

`take_ok`, `take_err`, and `partition` use boolean indexing (`z[mask]`) which produces **dynamic output shapes** — the output size depends on how many samples pass the filter. By default, `torch.compile(fullgraph=True)` cannot trace these operations:

```text
torch._dynamo.exc.Unsupported: Dynamic shape operator - aten.nonzero.default
```

**This is the most common source of graph breaks when using TorchGuard filtering inside compiled code.**

**Option 1: Use static-shape alternatives (recommended for compiled code)**

| Method                              | Returns  | Description                                         |
| ----------------------------------- | -------- | --------------------------------------------------- |
| `err.take_ok_p(f, z, fill=0)`  | `Tensor` | Same shape as `z`, error samples filled with `fill` |
| `err.take_err_p(f, z, fill=0)` | `Tensor` | Same shape as `z`, OK samples filled with `fill`    |
| `err.map_ok(f, z, fn)`              | `Tensor` | Apply `fn` only to OK samples, others unchanged     |
| `err.map_err(f, z, fn)`             | `Tensor` | Apply `fn` only to error samples, others unchanged  |

```python
# Static (inside torch.compile):
ok_out = err.take_ok_p(f, out, fill=0.0)    # Errors become 0.0
err_out = err.take_err_p(f, out, fill=0.0)  # OKs become 0.0
out = err.map_err(f, out, lambda z: torch.zeros_like(z))
```

**Option 2: Enable dynamic shape capture**

If you need the actual dynamic-shape behaviour inside compiled code, enable it globally:

```python
import torch._dynamo.config
torch._dynamo.config.capture_dynamic_output_shape_ops = True

import torch.nn as nn
model = nn.Identity()

@torch.compile(backend="inductor", fullgraph=True)
def forward(x):
    f = err.new(x)
    out = model(x)
    # Now take_ok(), take_err(), partition() work inside compiled code
    ok_out = err.take_ok(f, out)
    return ok_out
```

**Trade-off:** May reduce optimisation opportunities and increase compile time.

**Option 3: Use at Python boundary only**

Keep dynamic-shape operations outside compiled regions:

```python
import torch.nn as nn
model = nn.Identity()

@torch.compile(backend="inductor", fullgraph=True)
def compiled_forward(x):
    f = err.new(x)
    out = model(x)
    # Use static-shape ops inside compiled region
    return out, f

# Dynamic shapes fine outside compile
out, f = compiled_forward(x)
ok_out = err.take_ok(f, out)
```

| Method                        | Shape   | `torch.compile`     | Use Case             |
| ----------------------------- | ------- | ------------------- | -------------------- |
| `take_ok(f, z)`               | Dynamic | No (unless enabled) | Python boundary      |
| `take_err(f, z)`              | Dynamic | No (unless enabled) | Python boundary      |
| `partition(f, z)`             | Dynamic | No (unless enabled) | Python boundary      |
| `take_ok_p(f, z, fill)`  | Static  | Yes                 | Inside compiled code |
| `take_err_p(f, z, fill)` | Static  | Yes                 | Inside compiled code |
| `map_ok(f, z, fn)`            | Static  | Yes                 | Inside compiled code |
| `map_err(f, z, fn)`           | Static  | Yes                 | Inside compiled code |

### Slot Inspection

| Method                      | Description                |
| --------------------------- | -------------------------- |
| `err.get_first_code(f)`     | Get code from slot 0       |
| `err.get_first_location(f)` | Get location from slot 0   |
| `err.get_first_severity(f)` | Get severity from slot 0   |
| `err.clear(f, code)`        | Remove specific error code |

</details>

<details>
<summary><strong>Core API – <code>flags</code> Namespace (Python Boundary)</strong></summary>

The `flags` namespace provides inspection and debugging methods. **NOT** for use inside compiled regions.
These functions return Python values and will introduce graph breaks if used inside `torch.compile` regions.

| Method                        | Description                         |
| ----------------------------- | ----------------------------------- |
| `flags.unpack(f, sample_idx)` | Unpack errors for one sample        |
| `flags.unpack_all(f)`         | Unpack errors for all samples (vectorized) |
| `flags.repr(f)`               | Pretty string representation        |
| `flags.summary(f)`            | Dict of `{location: {code: count}}` |

**Classes:**

| Class | Description |
| ----- | ----------- |
| `ErrorFlags` | Static methods for flag inspection |
| `UnpackedError` | NamedTuple with `severity`, `code`, `location`, `*_name` fields |

```python
if has_err(f):
    print(flags.repr(f))
    # ErrorFlags(32 samples, 5 errors: 3xNAN @ encoder, 2xINF @ output)
    
    summary = flags.summary(f)
    # {'encoder': {'NAN': 3}, 'output': {'INF': 2}}
    
    errors = flags.unpack(f, sample_idx=0)
    for e in errors:
        print(f"{e.code_name} at {e.location_name}")
```

</details>

<details>
<summary><strong>Helper Functions</strong></summary>

Convenience functions with auto-location resolution.

All helper functions are implemented in terms of the `err` namespace. Detection/transformation helpers (`flag_*`, `push`, `fix`, `find`, etc.) return tensors and are safe inside `torch.compile(fullgraph=True)`. Helpers that return Python values (like `has_err(f)`) are intended for the Python boundary and will cause graph breaks if used inside compiled regions.

| Function                                           | Description                                       |
| -------------------------------------------------- | ------------------------------------------------- |
| `has_err(f)`                                       | Python bool: any errors in batch? (boundary only) |
| `find(code, f)`                                    | Per-sample mask for specific code                 |
| `push(f, code, module, where=...)`                 | Push with auto-location from module               |
| `fix(z, f, module, fallback=0.0)`                  | Replace bad values, record `FALLBACK_VALUE`       |
| `flag_nan(z, module, f)`                           | Detect NaN and record                             |
| `flag_inf(z, module, f)`                           | Detect Inf and record                             |
| `flag_nan_and_inf(z, module, f)`                   | **Fused** NaN+Inf detection (faster than separate calls) |
| `flag_oob_indices(ids, num_embeddings, module, f)` | Check index bounds                                |

```python
@tracked
class Model(nn.Module):
    def forward(self, x):
        f = err.new(x)
        
        out = self.layer(x)
        f = flag_nan(out, self.layer, f)
        f = flag_inf(out, self.layer, f)
        
        # Manual push with condition
        bad = (out.abs() > 1e6).any(dim=-1)
        f = push(f, err.OVERFLOW, self.layer, where=bad)
        
        # Fix and continue
        out, f = fix(out, f, self.layer, fallback=0.0)
        
        return out, f
```

</details>

<details>
<summary><strong>Combinators (Advanced)</strong></summary>

Applicative/monadic-style combinators, all `torch.compile(fullgraph=True)` compatible.

### Value Transforms

| Method                  | Description                              |
| ----------------------- | ---------------------------------------- |
| `err.map_ok(f, z, fn)`  | Apply `fn` to `z` for OK samples only    |
| `err.map_err(f, z, fn)` | Apply `fn` to `z` for error samples only |
| `err.replace(t, value, targets)` | Replace NaN/Inf/specific values in tensor |

```python
# Normalise only clean samples
h = err.map_ok(f, h, lambda z: z / z.norm(dim=-1, keepdim=True))

# Zero out error samples
h = err.map_err(f, h, lambda z: torch.zeros_like(z))

# Replace NaN and Inf with 0.0 (gradient-safe)
h = err.replace(h, value=0.0, targets=[err.NAN, err.INF])

# Replace only NaN
h = err.replace(h, value=0.0, targets=[err.NAN])

# Replace specific numerical values
h = err.replace(h, value=-1.0, targets=[999, float('inf')])
```

### Chaining

| Method                    | Description                                |
| ------------------------- | ------------------------------------------ |
| `err.and_then(f, z, fn)`  | Short-circuit: skip `fn` for error samples |
| `err.bind(f, z, fn)`      | Accumulate: run `fn`, collect ALL errors   |
| `err.map_err_flags(f, fn)`| Apply `fn` to flags of error samples       |

```python
# Apply transformation only to flags of error samples
def add_context(flags):
    # Add additional error context
    return err.push(flags, err.SATURATED, location=99, severity=err.WARN)

updated_flags = err.map_err_flags(flags, add_context)
```

### Guards and Recovery

| Method                                           | Description                           |
| ------------------------------------------------ | ------------------------------------- |
| `err.ensure_mask(f, ok_mask, code, loc)`         | Push error where mask is `False`      |
| `err.guard(f, z, pred, code, loc)`               | Push error where `pred(z)` is `False` |
| `err.recover_with_fallback(f, z, fallback, loc)` | Replace errors with fallback          |

</details>

<details>
<summary><strong>Control Flow DSL (Advanced)</strong></summary>

Conditional logic inside compiled code using `torch.where` for selection.

**Note:** For use with `torch.compile(fullgraph=True)`, import from the experimental backend:

```python
from torchguard.experimental import err, IF, IS, HAS, AND, OR, NOT
```

See [Experimental Backend](#experimental-backend) for details.

**Important constraints:**
* All branches are evaluated (no short-circuit) - this ensures proper gradient flow
* Keep branch bodies **side-effect free** (no logging, no mutation of Python objects, no graph breaks)
* Branch outputs must be **shape-consistent** across all paths
* In eager mode, uses Python control flow; in compiled mode, uses `torch.where` selection

### Predicates

| Function      | Description            |
| ------------- | ---------------------- |
| `HAS(f)`      | Any error in batch?    |
| `IS(code, f)` | Any sample has `code`? |
| `OR(*conds)`  | Logical OR             |
| `AND(*conds)` | Logical AND            |
| `NOT(cond)`   | Logical negation       |

### Usage

```python
from torchguard.experimental import err, IF, IS, OR

# Simple conditional
z, f = (
    IF(IS(err.NAN, f), lambda: (torch.zeros_like(z), f.clone()))
    .ELSE(lambda: (z.clone(), f.clone()))
)

# Multiple conditions
z, f = (
    IF(IS(err.NAN, f), lambda: handle_nan(z, f))
    .ELIF(IS(err.INF, f), lambda: handle_inf(z, f))
    .ELSE(lambda: (z.clone(), f.clone()))
)

# Compound predicates
z, f = (
    IF(OR(IS(err.NAN, f), IS(err.INF, f)), lambda: (torch.zeros_like(z), f.clone()))
    .ELSE(lambda: (z.clone(), f.clone()))
)
```

</details>

<details>
<summary><strong>Auto-Detection (<code>@tensorcheck</code>, Advanced)</strong></summary>

Automatic NaN/Inf detection on method return values. Works with both stable (`int64`) and experimental (`float32`/`float64`) backends:

```python
from torchguard import tracked, tensorcheck, err

@tracked
class SafeModel(nn.Module):
    @tensorcheck  # Auto-detects NaN and Inf in returned tensors
    def forward(self, x):
        f = err.new(x)
        out = self.layer(x)
        # TorchGuard adds flags for any NaN/Inf in `out` to `f` before returning
        return out, f
    
    @tensorcheck(auto_detect={err.NAN})  # Only detect NaN
    def forward_nan_only(self, x):
        ...
    
    @tensorcheck(auto_detect=False)  # Validation only, no detection
    def forward_no_detect(self, x):
        ...
```

```python
# Also works with experimental float32 backend
from torchguard.experimental import err as exp_err

@tracked
class ExperimentalModel(nn.Module):
    @tensorcheck
    def forward(self, x):
        f = exp_err.new(x)  # float32 flags
        out = self.layer(x)
        return out, f  # Auto-detection works!
```

</details>

---

## Location Tracking (`@tracked`)

`@tracked` injects an `_fx_path` attribute into submodules so you can pass `self.submodule` to TorchGuard helpers; the decorator handles wiring and registration.

```python
@tracked
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)  # _fx_path = "attention"
        self.ffn = nn.Sequential(                               # _fx_path = "ffn"
            nn.Linear(dim, dim * 4),                            # _fx_path = "ffn.0"
            nn.GELU(),                                          # stateless — no _fx_path
            nn.Linear(dim * 4, dim),                            # _fx_path = "ffn.2"
        )
```

Usage inside compiled code:
```python
def forward(self, x):
    f = err.new(x)
    x = self.attention(x)
    f = flag_nan(x, self.attention, f)   # _fx_path resolved automatically
    x = self.ffn[0](x)
    f = flag_nan(x, self.ffn[0], f)
    return x, f
```

### How location IDs work

The `@tracked` pass builds a module tree and prunes it to only parameter-containing modules (e.g., `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, `nn.Embedding`). Stateless ops (e.g., `GELU`, `ReLU`, `Dropout`) are skipped to save ID space.

>Rationale: learnable modules are the common sources of numerical instability (weight/grad issues); stateless ops just propagate inputs. By tracking only "unsafe" components, errors are attributed to root causes, not symptoms. For example, if an activation output is NaN, the error is attributed to the preceding parameter-containing layer that produced it.

The pruned modules are assigned sequential 10-bit IDs (0–1023) via deterministic depth-first traversal. A registry maps ID ↔ `_fx_path` so `flags.repr()` and `flags.summary()` can show human names at the Python boundary.

### What happens if you exceed 1024 modules

When a model has more parameter-containing modules than available IDs (1024 slots), TorchGuard **adaptively prunes the module tree** instead of using hash collisions. The pruning algorithm finds a depth cutoff where modules at or shallower than that depth get unique IDs; deeper modules are collapsed and share their ancestor's ID. Example: if pruning happens at depth 2, `encoder.layer.0.attn.q` and `encoder.layer.0.attn.k` both inherit the ID of `encoder.layer.0`. This preserves granularity for the most important (shallowest) parts of your model while collapsing detail in deeper subtrees.

### Workarounds for very deep models

You can influence pruning via `WeightedLocationTree`: set high weights on important paths (e.g., classifier head) to preserve them at deeper levels while collapsing other paths.

- Use `WeightedLocationTree` to manually specify which module subtrees should be preserved:
  ```python
  from torchguard.src.core.location.tree import WeightedLocationTree
  tree = WeightedLocationTree(max_locations=1024)
  tree.set_weight("head.*", 10.0)           # Preserve classifier granularity
  tree.set_weight("encoder.layer*", 1.0)   # Collapse encoder layers
  ```
- Use manual location IDs for critical checks:
  ```python
  f = push(f, err.NAN, location=42, where=torch.isnan(x).any(dim=-1))
  ```
- Restructure your model with blocks/containers to reduce the total number of tracked modules.

> Note:
> ID assignment is deterministic across runs for the same model structure.
> Registry reverse-lookup is used only at the Python boundary; all in-graph operations use numeric IDs (no Python lookups inside compiled regions).

---

## Tensor Typing System

TorchGuard includes a complete tensor typing system for type-safe annotations with runtime validation.

```python
from torchguard.typing import Tensor, Dim, Broadcast, float32_t, int64_t, error_t, type_cast
from torchguard import tracked, tensorcheck
```

### Basic Usage

```python
from torchguard.typing import Tensor, float32_t, int64_t, Dim

@tracked
class MyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
    
    @tensorcheck
    def forward(
        self,
        x: Tensor[float32_t, ("batch", Dim.in_features)],
    ) -> tuple[Tensor[float32_t, ("batch", Dim.out_features)], Tensor[error_t, ("batch", "num_words")]]:
        f = err.new(x)
        out = self.linear(x)
        f = flag_nan(out, self.linear, f)
        return out, f
```

### Tensor Annotation Syntax

```python
Tensor[dtype, shape]
Tensor[dtype, shape, device]
Tensor[dtype, shape, device, requires_grad]
```

**Shape specifications:**
- **Literal integers**: `Tensor[float32_t, (32, 512)]` - exact dimensions
- **Named dimensions**: `Tensor[float32_t, ("batch", "features")]` - tracked across tensors
- **Instance attributes**: `Tensor[float32_t, ("N", Dim.hidden_size)]` - resolved from `self`
- **Ellipsis**: `Tensor[float32_t, (..., "seq", "hidden")]` - variable batch dimensions
- **Broadcast marker**: `Tensor[float32_t, (Broadcast, Dim.features)]` - broadcast-compatible

### Dtype Aliases

| Alias | PyTorch dtype |
|-------|--------------|
| `float32_t` | `torch.float32` |
| `float64_t` | `torch.float64` |
| `float16_t` | `torch.float16` |
| `bfloat16_t` | `torch.bfloat16` |
| `int64_t` | `torch.int64` |
| `int32_t` | `torch.int32` |
| `int8_t` | `torch.int8` |
| `bool_t` | `torch.bool` |
| `error_t` | `torch.int64` (error flags) |

### Type-Safe Casting

```python
from torchguard.typing import type_cast, float32_t

# Returns Result[Tensor, Error]
result = type_cast[float32_t](int_tensor)
if result.is_ok():
    float_tensor = result.unwrap()
else:
    print(f"Cast failed: {result.unwrap_err()}")
```

### Dim and Broadcast

```python
from torchguard.typing import Tensor, Dim, Broadcast, float32_t

class Encoder(nn.Module):
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
    
    @tensorcheck
    def forward(
        self,
        x: Tensor[float32_t, ("batch", "seq", Dim.hidden_size)],  # Dim.hidden_size → self.hidden_size
        bias: Tensor[float32_t, (Broadcast, Dim.hidden_size)],    # Broadcast-compatible
    ) -> Tensor[float32_t, ("batch", "seq", Dim.hidden_size)]:
        return x + bias
```

### @tensorcheck Decorator

Validates tensor shapes and dtypes at runtime (skipped during `torch.compile`):

```python
@tensorcheck                          # Default: validates shapes/dtypes + auto-detects NaN/Inf
@tensorcheck(auto_detect=True)        # Same as default
@tensorcheck(auto_detect=False)       # Validation only, no NaN/Inf detection
@tensorcheck(auto_detect={err.NAN})   # Only detect NaN
```

**Backend support:** Auto-detection works with both stable (`int64`) and experimental (`float32`/`float64`) backends. The decorator automatically recognizes flag tensors based on their dtype and shape.

### Validation Errors

When `@tensorcheck` detects validation failures, it raises specific exception types:

```python
from torchguard import (
    ValidationError,          # Base class for all validation errors
    DimensionMismatchError,   # Shape mismatch
    DTypeMismatchError,       # Dtype mismatch
    DeviceMismatchError,      # Device mismatch (CPU vs GPU)
    InvalidParameterError,    # Parameter validation failed
    TypeMismatchError,        # Return type mismatch
    InvalidReturnTypeError,   # Invalid return value
)

try:
    output, flags = model(x)
except DimensionMismatchError as e:
    print(f"Shape validation failed: {e}")
except DTypeMismatchError as e:
    print(f"Dtype validation failed: {e}")
```

These exceptions are only raised at runtime (not during `torch.compile`).

</details>

<details>
<summary><strong>Result-Oriented Programming</strong></summary>

TorchGuard includes decorators for exception-free programming with `Result` types:

```python
from torchguard import as_result, as_exception, unwrap, Ok, Err, Result

@as_result
def might_fail(x):
    """Returns Result[T, Exception] instead of raising."""
    if x < 0:
        raise ValueError("Negative input")
    return x * 2

result = might_fail(-5)
if result.is_err():
    print(f"Failed: {result.unwrap_err()}")
else:
    print(f"Success: {result.unwrap()}")
```

**Decorators:**

| Decorator        | Behavior                                          |
| ---------------- | ------------------------------------------------- |
| `@as_result`     | Catches exceptions, returns `Result[T, Exception]`|
| `@as_exception`  | Unwraps `Result`, raises on `Err`                 |
| `@unwrap`        | Unwraps `Result`, raises on `Err` (alias)         |

```python
@as_result
def safe_divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Division by zero")
    return a / b

# Returns Result[float, ZeroDivisionError]
result = safe_divide(10, 0)
assert result.is_err()

# Chain results
def process(x):
    result = safe_divide(x, 2)
    if result.is_err():
        return result  # Propagate error
    return Ok(result.unwrap() + 1)
```

**Result API:**

| Method          | Description                       |
| --------------- | --------------------------------- |
| `is_ok()`       | Check if result is Ok             |
| `is_err()`      | Check if result is Err            |
| `unwrap()`      | Get value or raise                |
| `unwrap_err()`  | Get error or raise                |
| `unwrap_or(d)`  | Get value or default              |
| `map(fn)`       | Transform Ok value                |
| `map_err(fn)`   | Transform Err value               |

</details>

---

## Configuration

TorchGuard uses a **global mutable config** that controls behavior for all operations:

### Global Config

```python
import torch
import torchguard as tg

# Access the global config
print(tg.CONFIG.flag_dtype)  # torch.int64 (default)
print(tg.CONFIG.num_slots)   # 16 (default)

# Modify config directly
tg.CONFIG.flag_dtype = torch.float32  # Switch to experimental backend
tg.CONFIG.num_slots = 32              # More error slots per sample

# All operations automatically use the new config
x = torch.randn(4, 8)
flags = tg.err.new(x)  # Creates float32 flags with 32 slots

# Or replace the entire config at once
from torchguard import set_config, ErrorConfig, get_config

original = get_config()
set_config(ErrorConfig(flag_dtype=torch.float32, num_slots=64))
try:
    # ... use new config ...
finally:
    set_config(original)  # Restore
```

### Config Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `flag_dtype` | `torch.dtype` | `torch.int64` | Storage dtype: `float32`/`float64` (experimental) or `int32`/`int64` (stable) |
| `num_slots` | `int` | `16` | Max errors per sample (1-32768, fully vectorized) |
| `accumulation` | `AccumulationConfig` | FIFO | How to handle slot overflow (see below) |
| `default_severity` | `Severity` | `ERROR` | Default severity for `push` operations |
| `strict_validation` | `bool` | `False` | Raise vs warn on validation failures |
| `use_transposed_layout` | `bool` | `False` | Use transposed memory layout for large batches |
| `transpose_threshold` | `int` | `10000` | Batch size above which transposition is applied |

### Backend Selection via flag_dtype

```python
# Stable backend (int64) - default, best for inference/eager mode
tg.CONFIG.flag_dtype = torch.int64

# Experimental backend (float32) - best for torch.compile(fullgraph=True) training
tg.CONFIG.flag_dtype = torch.float32

# Experimental backend (float64) - more slots per word (4 vs 2 for float32)
tg.CONFIG.flag_dtype = torch.float64
```

### Per-Operation Config Override

```python
# Use global config by default
flags = tg.err.new(x)  # Uses tg.CONFIG

# Override for specific operations
custom = tg.ErrorConfig(flag_dtype=torch.float64, num_slots=8)
flags = tg.err.new_t(5, config=custom)  # Uses custom config
```

### Accumulation Policies

When error slots fill up (e.g., NaN propagates through 20 layers but you only have 16 slots), the accumulation policy decides which errors to keep:

<details>
<summary><strong>Policy Details</strong></summary>

```python
from torchguard import AccumulationConfig, Priority, Order, Dedupe

# Modify global config's accumulation
tg.CONFIG.accumulation = AccumulationConfig(
    priority=Priority.CHRONO,  # What determines importance
    order=Order.FIRST,         # Keep FIRST or LAST on priority axis
    dedupe=Dedupe.UNIQUE       # Deduplication strategy
)
```

**Three orthogonal axes:**

| Axis | Values | Description |
|------|--------|-------------|
| `Priority` | `CHRONO`, `SEVERITY`, `LOCATION` | What determines importance |
| `Order` | `FIRST`, `LAST` | Keep min or max on priority axis |
| `Dedupe` | `NONE`, `CODE`, `LOCATION`, `UNIQUE` | How to group duplicates |

**Common configurations:**

```python
# FIFO (default) – keep oldest errors for root cause debugging
AccumulationConfig(priority=Priority.CHRONO, order=Order.FIRST, dedupe=Dedupe.UNIQUE)

# LIFO – keep newest errors to track most recent state
AccumulationConfig(priority=Priority.CHRONO, order=Order.LAST, dedupe=Dedupe.UNIQUE)

# Severity-based – keep highest severity errors
AccumulationConfig(priority=Priority.SEVERITY, order=Order.LAST, dedupe=Dedupe.UNIQUE)
```

**Example: Why FIFO is default**

```python
# Scenario: NaN originates in layer 1, propagates through 20 layers

# With FIFO (default - Order.FIRST):
# Slots 1-16 record layers 1-16
# Layer 1's error is preserved ✅
# You can see where the NaN originated

# With LIFO (Order.LAST):
# Slots 1-16 record layers 5-20
# Layer 1's error gets dropped ❌
# You only see propagation, not the root cause
```

**When to change:**

- **Keep FIFO** for debugging NaN/Inf issues (find where it starts)
- **Use LIFO** for monitoring production systems (track recent state)
- **Use SEVERITY** for production error handling (prioritize critical issues)

</details>

---

## Advanced Topics

<details>
<summary><strong>Bit-Level Constants (Advanced)</strong></summary>

For advanced users who need to inspect or manipulate flag tensors directly, TorchGuard exports bit-level constants:

```python
from torchguard import (
    SLOT_BITS,        # 16 - bits per slot
    SLOTS_PER_WORD,   # 4 - slots per 64-bit word (int64/float64)
    SEVERITY_SHIFT,   # 0 - bit offset for severity
    SEVERITY_BITS,    # 2 - bits for severity
    SEVERITY_MASK,    # 0x3 - mask for severity
    CODE_SHIFT,       # 2 - bit offset for code
    CODE_BITS,        # 4 - bits for code
    CODE_MASK,        # 0xF - mask for code
    LOCATION_SHIFT,   # 6 - bit offset for location
    LOCATION_BITS,    # 10 - bits for location
    LOCATION_MASK,    # 0x3FF - mask for location
    SLOT_MASK,        # 0xFFFF - mask for entire slot
)

# Extract components from a packed slot manually
def unpack_slot(slot_value):
    severity = (slot_value >> SEVERITY_SHIFT) & SEVERITY_MASK
    code = (slot_value >> CODE_SHIFT) & CODE_MASK
    location = (slot_value >> LOCATION_SHIFT) & LOCATION_MASK
    return severity, code, location
```

**Warning:** Direct bit manipulation is rarely needed. Use `err.*` and `flags.*` APIs instead. These constants are primarily for:
- Debugging TorchGuard itself
- Implementing custom backends
- Performance-critical custom operations

**Slot layout:**

```
16-bit slot:
┌─────────────┬──────────┬──────────┐
│ Location    │ Code     │ Severity │
│ 10 bits     │ 4 bits   │ 2 bits   │
│ bits 15-6   │ bits 5-2 │ bits 1-0 │
└─────────────┴──────────┴──────────┘
```

</details>

---

## Performance Benchmarks

<details>
<summary><strong>Details</strong></summary>

### Benchmark Setup

* Device: CPU
* Batch size: 32
* Iterations: 50
* Tensor shape: `(32, 512)` for input, `(32, num_words)` for flags
* Mode: Eager execution
* Operations tested:
  * `new`: Create empty flags tensor
  * `push`: Record errors using `flag_nan` (computes `torch.isnan(x).any(dim=-1)` per sample, then updates flags)
  * `merge(2)`: Merge two flags tensors
  * `is_ok`: Check per-sample OK status

### Benchmark Results

| Slot Size | `new` (µs) | `push` (µs) | `merge(2)` (µs) | `is_ok` (µs) |
| --------- | ---------: | ----------: | --------------: | -----------: |
| 4 slots   |          6 |        ~700 |            ~165 |            7 |
| 16 slots  |          4 |        ~860 |            ~300 |            8 |
| 64 slots  |          5 |        ~680 |            ~330 |           16 |
| 256 slots |         11 |       ~2200 |           ~1100 |           16 |

**Notes:**
* `push` times include per-sample NaN detection (`torch.isnan(x).any(dim=-1)`) plus bitpacking overhead
* Times will vary with tensor size, device (GPU faster), and compile mode (compiled graphs amortize overhead)
* Slot count affects merge time linearly (more words to process)

**Recommendation:** 16 slots (default) for most use cases.

</details>

<details>
<summary><strong>Performance Optimizations</strong></summary>

TorchGuard includes several optimizations for high-performance workloads:

### Device-Aware Tensor Caching

TorchGuard automatically caches frequently-used tensors (shift arrays, constants) per device to avoid redundant allocations in hot paths:

```python
# These operations reuse cached tensors internally:
flags = err.new(x)  # Reuses cached slot shift tensors
flags = err.push(flags, code, loc)  # Reuses cached constants
```

The cache is:
- **Thread-safe**: Safe for multi-threaded usage
- **torch.compile-compatible**: Marked with `@torch._dynamo.disable` to prevent tracing issues
- **Device-aware**: Maintains separate caches for CPU, CUDA, MPS, etc.
- **Memory-efficient**: ~1-2KB per device

### Fused Operations

Use `flag_nan_and_inf()` instead of separate `flag_nan()` + `flag_inf()` calls:

```python
# Before: Two separate passes
f = flag_nan(out, self.layer, f)
f = flag_inf(out, self.layer, f)

# After: Single fused pass (~30% faster)
f = flag_nan_and_inf(out, self.layer, f)
```

### Vectorized Unpacking

For large batches, `flags.unpack_all(f)` uses vectorized extraction:

```python
# Automatically uses vectorized implementation
all_errors = flags.unpack_all(f)  # List[List[UnpackedError]] for all samples
```

Benefits:
- Batch-level tensor operations instead of per-sample Python loops
- Significant speedup for batches > 100 samples

### Memory Layout Optimization (Advanced)

For very large batches (10,000+ samples), TorchGuard supports an optional transposed memory layout for better cache locality:

```python
import torchguard as tg

# Enable transposed layout for large batches
tg.CONFIG.use_transposed_layout = True
tg.CONFIG.transpose_threshold = 10000  # Only transpose batches > 10k

# Check if transposition will be used
will_transpose = tg.CONFIG.should_transpose(batch_size=15000)  # True
```

**When to use:** Only for very large batches where profiling shows memory bandwidth is a bottleneck. For most workloads, the default layout is optimal.

</details>

---

## When NOT to Use TorchGuard

* **You never get NaN/Inf** (lucky you!) – if your training is stable, you do not need this
* **Inference with guaranteed-clean data** – no point tracking errors that cannot happen
* **Eager mode only** – use regular Python exceptions instead
* **Latency-critical paths** – TorchGuard adds overhead depending on slot count and number of checks (see Performance Benchmarks)
* **Simple debugging** – if you just need to find where NaN happens once, use `torch.autograd.set_detect_anomaly(True)`

### Non-Goals

TorchGuard does not aim to:
- Prevent NaNs from occurring
- Make error flags differentiable
- Replace Python exception handling entirely

---

## Experimental Backend

TorchGuard includes an **experimental backend** designed for full `torch.compile(fullgraph=True)` compatibility, including training with gradients.

### Why It Exists

The stable backend stores flags as `int64` tensors. Some `torch.compile` / AOTAutograd setups are sensitive to additional non-differentiable outputs, particularly:

1. **AOTAutograd wrapper constraints around aliasing/mutation/view reconstruction** – the runtime wrapper logic can be sensitive to extra outputs, especially when those outputs have complex view/alias relationships
2. **Inductor backward pass constraints** – certain configurations can cause issues during gradient computation

These issues don't affect all users, but they can manifest as:
* Errors about view operations during backward pass setup
* Issues with certain compiler backend configurations (especially `inductor`)

### The Solution

The experimental backend stores flags as **float tensors** (float32 by default, float64 optional) but uses `view()` to the corresponding int type for all bitwise operations:

```python
import torch
import torchguard as tg

# Switch to experimental backend (float32)
tg.CONFIG.flag_dtype = torch.float32

# Storage: float32 (better inductor compatibility)
flags = tg.err.new_t(N)  # Creates float32 flags

# Operations: internally views as int32 for bitwise ops (zero-copy)
flags = tg.err.push(flags, tg.ErrorCode.NAN, location=1)  # Works!
```

**Key properties:**

* float32: 2×16-bit slots per 32-bit word (8 words for 16 slots)
* float64: 4×16-bit slots per 64-bit word (4 words for 16 slots)
* Same API as stable backend
* Zero overhead (`view` is zero-copy)
* Works with `torch.compile(fullgraph=True)`
* Training (forward + backward) works

**Backend selection:**

```python
# float32 (recommended for inductor)
tg.CONFIG.flag_dtype = torch.float32

# float64 (more slots per word)
tg.CONFIG.flag_dtype = torch.float64

# Back to stable (int64)
tg.CONFIG.flag_dtype = torch.int64
```

### Critical Constraints

**You must NEVER run floating-point operations on flags tensors.** The float dtype is purely a storage carrier - the bit patterns are integer slots, not IEEE 754 floats. 

```python
# WRONG — corrupts flags
f = f + 1.0
loss = f.sum()
```

If you accidentally do `flags + 1.0` or include flags in differentiable computations, you will corrupt the error tracking state.

TorchGuard's API prevents this by always returning flags from operations, but if you manipulate flags tensors manually, keep them isolated from float ops.

### Why It Is Experimental

* **Relies on `view(dtype)` bit reinterpretation semantics** which may change across PyTorch versions
* **Compiler assumptions**: Current behaviour depends on how Inductor and AOTAutograd handle dtype views; future PyTorch versions could tighten dtype checking or change view semantics
* **Less battle-tested** than the stable `int64` backend
* **API may evolve** based on compiler stack changes
* Requires PyTorch ≥ 2.0 (recommended: ≥ 2.7 for best compatibility)

### Quick Start

```python
# Import from experimental instead of main package
from torchguard.experimental import err, IF, IS, HAS, AND, OR, NOT

@torch.compile(backend="inductor", fullgraph=True)
def forward(x):
    f = err.new(x)  # Creates float32 flags (default)
    
    # Same API as stable backend
    f = err.push(f, err.NAN, location=42, where=torch.isnan(x).any(dim=-1))
    
    # Control flow DSL works!
    out, f = (
        IF(IS(err.NAN, f), lambda: (torch.zeros_like(x), f.clone()))
        .ELSE(lambda: (x.clone(), f.clone()))
    )
    return out, f
```

### Training Example

```python
import torch
import torch.nn as nn
from torchguard.experimental import err, IF, IS

class SafeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 16)
    
    def forward(self, x):
        f = err.new(x)
        out = self.linear(x)
        
        # Flag NaN values
        f = err.push(f, err.NAN, location=1, where=torch.isnan(out).any(dim=-1))
        
        # Conditional recovery inside compiled graph
        out, f = (
            IF(IS(err.NAN, f), lambda: (torch.zeros_like(out), f.clone()))
            .ELSE(lambda: (out.clone(), f.clone()))
        )
        return out, f

model = SafeModel()
compiled = torch.compile(model, backend="inductor", fullgraph=True)

# Training works!
x = torch.randn(8, 32, requires_grad=True)
out, f = compiled(x)
loss = out.sum()
loss.backward()  # Works!
```

**Note:** Training (forward + backward) works as long as flags are not used in differentiable computations. Never include flags in loss calculations or gradient paths.

### When to Use Experimental vs Stable

| Use Case                          | Backend                                                      |
| --------------------------------- | ------------------------------------------------------------ |
| Inference without `torch.compile` | Stable (`from torchguard import err`)                        |
| Inference with `torch.compile`    | Either (experimental recommended)                            |
| Training with `torch.compile`     | **Experimental** (`from torchguard.experimental import err`) |
| Control flow DSL in compiled code | **Experimental**                                             |
| Production, maximum stability     | Stable                                                       |

---

## Python Boundary Utilities

<details>
<summary><strong><code>Result</code> Type (Additional Details)</strong></summary>

The `Result` type is fully documented in the **Result-Oriented Programming** section under **Tensor Typing System** above.

Quick reference:

```python
from torchguard import Ok, Err, Result, as_result

# Create Results directly
success = Ok(42)
failure = Err("Something went wrong")

# Use decorator for automatic exception catching
@as_result
def might_fail(x):
    if x < 0:
        raise ValueError("Negative")
    return x * 2

result = might_fail(10)  # Ok(20)
result = might_fail(-5)  # Err(ValueError("Negative"))
```

See the **Result-Oriented Programming** section for full API details, chaining, and advanced patterns.

</details>