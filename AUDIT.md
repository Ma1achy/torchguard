# TorchGuard Audit & Rewrite Foundation

This document records the audit of the original TorchGuard implementation, the
empirical de-risking that justified a rewrite, and what the rewrite delivers.
Findings are tagged **[CONFIRMED]** (verified by reading the original source) or
**[REPORT]** (surfaced in review, to be reproduced as a regression test).

## 1. Audit of the original implementation

### Packaging — blocking
- No `pyproject.toml`/`setup.py`; the package could not be built or installed
  despite the README's `pip install torchguard`. **[CONFIRMED]**
- `.gitignore` ignored `pyproject.toml`, actively preventing the fix. **[CONFIRMED]**
- Nested `torchguard/torchguard/` layout with a root `sys.modules` shim; fragile,
  broken once installed. **[CONFIRMED]**
- Undeclared deps (`torch`, `packaging`); tests imported via `sys.path` hacks. **[CONFIRMED]**

### Tooling — absent
No CI, devcontainer, pre-commit, ruff/mypy config, `conftest.py`, or coverage. **[CONFIRMED]**

### Code bugs
- `ErrorOps.new_t(n)` crashed without a `config` (body used `config.num_words` with
  no `get_config()` fallback) yet the README documented that exact call. **[CONFIRMED]**
- `validate_result()` caught only 3 validation errors + `ValueError`, but `validate()`
  could raise `AttributeError`/`TypeError` from `Dim` resolution → leaked. **[CONFIRMED]**
- `has_code`/`has_critical`/`has_domain` ignored the `num_slots` validity mask that
  `find()` applied → inconsistent results. **[CONFIRMED]**
- Experimental backend: class `Float64ErrorOps` defaulted to `float32`; docstring said
  "float64 storage"; `__all__`/import paths drifted from the README. **[CONFIRMED]**
- Redundant `has_any`/`any_err`; capacity doc-drift (docstrings said 256 slots, the
  default was 16). **[CONFIRMED]**
- `__merge_two` possibly losing cross-operand order; experimental `sum().to()` int
  promotion; storing bit patterns in float carriers risking NaN/Inf. **[REPORT]**

### Design
- ~70% duplication between the int64 ("stable") and float ("experimental") backends.
- Global mutable state (location registry, caches) with thread-safety questions.

## 2. Why the experimental float backend existed — and why it's gone

The float backend worked around an **older-PyTorch** issue: an `int64` flag tensor
entangled with the autograd graph forced a dtype match with the float data, so people
stored flags as floats to keep everything "differentiable."

De-risking spikes (torch 2.12 CPU) showed this is no longer necessary:

| Validated | Result |
|---|---|
| `__torch_dispatch__` subclass carrying **int64 flags** + float data; eager propagation + NaN auto-detect via int bitwise | PASS |
| Eager backward → grads to weights **and** wrapped input | PASS |
| `compile(fullgraph=True, backend="inductor")` forward (int bitwise + `isnan` in graph) | PASS |
| Backward **through** the compiled inductor graph | PASS |
| Variable batch (4→7→16) via **automatic dynamic shapes** | PASS |
| Naive int threading (even `int + float` promotion) under inductor + backward | PASS |
| Explicit `mark_dynamic(x, 0)` on a **subclass** dim | FAIL `ConstraintViolationError` (PyTorch-internal, unfixable) |

**Conclusion:** a single int backend is viable; the split is removed. The
`GuardedTensor` subclass additionally insulates against the original issue by
construction (differentiable `_data` vs non-differentiable `_flags`).

## 3. Architecture of the rewrite

- **One integer backend** (`int64` default, `int32` optional). No experimental split.
- **`GuardedTensor`** (`__torch_dispatch__` traceable subclass) is the single API: wrap
  once with `guard(x)`, and the bit-packed per-sample flag channel propagates through
  every op automatically; merge on multi-input ops.
- **Explicit detection** (`flag_nan`/`flag_inf`/`flag_nan_inf`) records errors where they
  occur (per-op auto-detection is opt-in by design — the reduction has a cost).
- **Boundary inspection** (`inspect.summary`/`report`) renders errors Python-side.
- **Documented limitation:** `mark_dynamic` on a `GuardedTensor` dim is unsupported;
  automatic dynamic shapes cover variable batch sizes.

The confirmed bugs above are fixed by construction and covered by tests (e.g. `find`
and friends now share one `num_slots`-respecting slot extractor).

## 4. Status / roadmap

This PR is the **foundation + core engine** (rewrite phases 1–2):
`src/torchguard/` layout, `pyproject.toml`, devcontainer, CI matrix (py3.10–3.12 ×
torch floor/latest), ruff + mypy + pre-commit, and a working `GuardedTensor` core with
the spike promoted into regression tests.

Remaining phases: automatic module-tree location tracking (forward-hooks + contextvar;
risk: contextvar under compile — spike first), full accumulation policies
(FIFO/severity/dedupe), the tensor typing system + `@tensorcheck`/`@tracked`, and a
README rewrite to the subclass API.
