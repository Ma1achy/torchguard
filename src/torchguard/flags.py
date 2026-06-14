"""Vectorized, ``torch.compile``-friendly operations on packed error flags.

A flags tensor has shape ``(N, num_words)`` with an integer dtype. Each word
holds ``slots_per_word`` 16-bit slots; each slot packs ``[location:10][code:4]
[severity:2]``. A sample is error-free iff all of its words are zero.

These functions operate on plain tensors so they can be reused by
``GuardedTensor`` and called at the Python boundary alike. They are written to
avoid integer ``sum`` reductions (which promote to ``int64``) by OR-reducing the
small, statically-sized ``slots_per_word`` dimension.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .config import Dedupe, ErrorConfig, Order, get_config
from .constants import (
    CODE_MASK,
    CODE_SHIFT,
    LOCATION_MASK,
    LOCATION_SHIFT,
    SEVERITY_MASK,
    SLOT_BITS,
    SLOT_MASK,
)
from .severity import Severity

__all__ = [
    "new",
    "pack_slot_scalar",
    "extract_slots",
    "pack_slots",
    "is_ok",
    "is_err",
    "any_err",
    "find",
    "count",
    "has_critical",
    "merge",
    "push",
    "unpack",
]


def new(n: int, config: ErrorConfig | None = None, device: torch.device | None = None) -> Tensor:
    """Create a zeroed flags tensor of shape ``(n, num_words)``."""
    config = config or get_config()
    return torch.zeros(n, config.num_words, dtype=config.flag_dtype, device=device)


def pack_slot_scalar(code: int, location: int, severity: int) -> int:
    """Pack a single slot into a Python int."""
    return (
        (severity & SLOT_MASK)
        | ((code & CODE_MASK) << CODE_SHIFT)
        | ((location & LOCATION_MASK) << LOCATION_SHIFT)
    )


def extract_slots(flags: Tensor, config: ErrorConfig | None = None) -> Tensor:
    """Unpack ``flags`` into a ``(N, num_slots)`` tensor of 16-bit slot values."""
    config = config or get_config()
    spw = config.slots_per_word
    shifts = torch.arange(spw, device=flags.device, dtype=flags.dtype) * SLOT_BITS
    # (N, num_words, 1) >> (spw,) -> (N, num_words, spw)
    slots = (flags.unsqueeze(-1) >> shifts) & SLOT_MASK
    slots = slots.reshape(flags.shape[0], -1)
    return slots[:, : config.num_slots]


def pack_slots(slots: Tensor, config: ErrorConfig | None = None) -> Tensor:
    """Pack a ``(N, num_slots)`` slot tensor back into ``(N, num_words)``."""
    config = config or get_config()
    spw = config.slots_per_word
    total = config.num_words * spw
    n = slots.shape[0]
    if slots.shape[1] < total:
        pad = torch.zeros(n, total - slots.shape[1], dtype=slots.dtype, device=slots.device)
        slots = torch.cat([slots, pad], dim=1)
    else:
        slots = slots[:, :total]
    reshaped = slots.reshape(n, config.num_words, spw)
    # OR-reduce the (statically-sized) slot dimension to avoid int64 promotion.
    words = reshaped[..., 0].clone()
    for i in range(1, spw):
        words = words | (reshaped[..., i] << (i * SLOT_BITS))
    return words


def is_ok(flags: Tensor) -> Tensor:
    """Per-sample bool mask: ``True`` where the sample has no errors."""
    return (flags == 0).all(dim=-1)


def is_err(flags: Tensor) -> Tensor:
    """Per-sample bool mask: ``True`` where the sample has any error."""
    return (flags != 0).any(dim=-1)


def any_err(flags: Tensor) -> Tensor:
    """Scalar bool tensor: ``True`` if any sample has any error."""
    return (flags != 0).any()


def find(code: int, flags: Tensor, config: ErrorConfig | None = None) -> Tensor:
    """Per-sample bool mask: ``True`` where a sample carries ``code``.

    Respects ``config.num_slots`` consistently (a slot only counts when its
    severity is non-zero), matching every other query in this module.
    """
    config = config or get_config()
    slots = extract_slots(flags, config)
    codes = (slots >> CODE_SHIFT) & CODE_MASK
    sev = slots & Severity.CRITICAL  # mask low 2 bits
    return ((codes == code) & (sev != 0)).any(dim=1)


def count(flags: Tensor, config: ErrorConfig | None = None) -> Tensor:
    """Per-sample count of non-empty error slots."""
    slots = extract_slots(flags, config or get_config())
    return (slots != 0).sum(dim=1)


def has_critical(flags: Tensor, config: ErrorConfig | None = None) -> Tensor:
    """Per-sample bool mask: ``True`` where a slot has ``CRITICAL`` severity."""
    slots = extract_slots(flags, config or get_config())
    sev = slots & Severity.CRITICAL
    return (sev == Severity.CRITICAL).any(dim=1)


def merge(a: Tensor, b: Tensor, config: ErrorConfig | None = None) -> Tensor:
    """Merge two flags tensors, keeping ``a``'s slots ahead of ``b``'s.

    Slots are compacted (non-empty first, order preserved) and truncated to
    ``num_slots``. Equivalent to error accumulation with the slots of ``a``
    treated as older/earlier.
    """
    config = config or get_config()
    sa = extract_slots(a, config)
    sb = extract_slots(b, config)
    combined = torch.cat([sa, sb], dim=1)
    empty = (combined == 0).to(torch.int64)  # 0 -> keep first, 1 -> push to back
    _, idx = torch.sort(empty, dim=1, stable=True)
    compacted = torch.gather(combined, 1, idx)
    return pack_slots(compacted[:, : config.num_slots], config)


def _slot_key(x: Tensor, dedupe: Dedupe) -> Tensor:
    """Extract the dedupe key from packed slot value(s)."""
    if dedupe is Dedupe.CODE:
        return (x >> CODE_SHIFT) & CODE_MASK
    if dedupe is Dedupe.LOCATION:
        return (x >> LOCATION_SHIFT) & LOCATION_MASK
    return x & (SLOT_MASK ^ SEVERITY_MASK)  # PAIR: location+code, ignore severity


def push(
    flags: Tensor,
    code: Tensor,
    location: int,
    severity: int,
    config: ErrorConfig | None = None,
) -> Tensor:
    """Record a new error slot per sample, honoring the accumulation policy.

    Samples with ``code == 0`` are left unchanged. Slots stay front-packed. The
    policy (``config.accumulation``) controls eviction order (LIFO/FIFO),
    deduplication (none/code/location/pair), and severity-based eviction. Fully
    vectorized and ``torch.compile``-safe.

    Args:
        flags: Current flags ``(N, num_words)``.
        code: Per-sample error code ``(N,)``.
        location: Location id (same for all samples).
        severity: Severity for the pushed slot.
        config: Layout + accumulation configuration.
    """
    config = config or get_config()
    policy = config.accumulation
    slots = extract_slots(flags, config)  # (N, S), front-packed
    n, s = slots.shape
    device = slots.device

    new_slot = (
        ((code.to(flags.dtype) & CODE_MASK) << CODE_SHIFT)
        | ((location & LOCATION_MASK) << LOCATION_SHIFT)
        | (severity & SLOT_MASK)
    )
    new_col = new_slot.unsqueeze(1)  # (N, 1)
    should = code != 0
    nonempty = slots != 0
    sev = slots & SEVERITY_MASK
    new_sev = new_col & SEVERITY_MASK

    # -- deduplication --
    if policy.dedupe is not Dedupe.NONE:
        match = nonempty & (_slot_key(slots, policy.dedupe) == _slot_key(new_col, policy.dedupe))
        has_match = match.any(dim=1)
        deduped = torch.where(match & (new_sev > sev), new_col, slots)
    else:
        has_match = torch.zeros(n, dtype=torch.bool, device=device)
        deduped = slots

    insert = should & ~has_match

    # -- insertion by order --
    if policy.order is Order.LIFO:
        inserted = torch.cat([new_col, slots[:, :-1]], dim=1)
    else:  # FIFO: append at first empty slot
        count = nonempty.sum(dim=1)
        idx = torch.arange(s, device=device).unsqueeze(0)  # (1, S)
        has_space = count < s
        pos = torch.minimum(count, torch.full_like(count, s - 1))
        inserted = torch.where((idx == pos.unsqueeze(1)) & has_space.unsqueeze(1), new_col, slots)
        if policy.evict_by_severity:
            min_pos = sev.argmin(dim=1)
            min_sev = sev.gather(1, min_pos.unsqueeze(1)).squeeze(1)
            evict = (~has_space) & (new_sev.squeeze(1) > min_sev)
            at_min = (idx == min_pos.unsqueeze(1)) & evict.unsqueeze(1)
            inserted = torch.where(at_min, new_col, inserted)

    out = torch.where((should & has_match).unsqueeze(1), deduped, slots)
    out = torch.where(insert.unsqueeze(1), inserted, out)
    return pack_slots(out, config)


def unpack(flags_row: Tensor, config: ErrorConfig | None = None) -> list[tuple[int, int, int]]:
    """Decode one sample's flags into ``(code, location, severity)`` tuples.

    Python-boundary only (returns Python values); do not call inside a compiled
    region.
    """
    config = config or get_config()
    slots = extract_slots(flags_row.unsqueeze(0), config)[0]
    out: list[tuple[int, int, int]] = []
    for slot in slots.tolist():
        if slot == 0:
            continue
        severity = slot & Severity.CRITICAL
        code = (slot >> CODE_SHIFT) & CODE_MASK
        location = (slot >> LOCATION_SHIFT) & LOCATION_MASK
        out.append((code, location, severity))
    return out
