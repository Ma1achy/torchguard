"""Accumulation policies: LIFO/FIFO order, dedupe modes, severity eviction."""
from __future__ import annotations

import torch

from torchguard import AccumulationPolicy, Dedupe, ErrorCode, ErrorConfig, Order, Severity
from torchguard import flags as F


def _cfg(num_slots=4, **policy):
    return ErrorConfig(num_slots=num_slots, accumulation=AccumulationPolicy(**policy))


def _push(f, code, loc, sev, cfg):
    return F.push(f, torch.tensor([code], dtype=cfg.flag_dtype), loc, sev, cfg)


def _codes(f, cfg):
    return [c for c, _loc, _sev in F.unpack(f[0], cfg)]


def test_lifo_keeps_newest_first():
    cfg = _cfg(order=Order.LIFO)
    f = _push(F.new(1, cfg), ErrorCode.NAN, 1, Severity.CRITICAL, cfg)
    f = _push(f, ErrorCode.INF, 2, Severity.CRITICAL, cfg)
    assert _codes(f, cfg) == [ErrorCode.INF, ErrorCode.NAN]  # newest first


def test_fifo_keeps_oldest_first():
    cfg = _cfg(order=Order.FIFO)
    f = _push(F.new(1, cfg), ErrorCode.NAN, 1, Severity.CRITICAL, cfg)
    f = _push(f, ErrorCode.INF, 2, Severity.CRITICAL, cfg)
    assert _codes(f, cfg) == [ErrorCode.NAN, ErrorCode.INF]  # oldest first


def test_fifo_drops_new_when_full():
    cfg = _cfg(num_slots=2, order=Order.FIFO)
    f = _push(F.new(1, cfg), ErrorCode.NAN, 1, Severity.ERROR, cfg)
    f = _push(f, ErrorCode.INF, 2, Severity.ERROR, cfg)
    f = _push(f, ErrorCode.OVERFLOW, 3, Severity.ERROR, cfg)  # full -> dropped
    assert _codes(f, cfg) == [ErrorCode.NAN, ErrorCode.INF]


def test_fifo_evicts_lowest_severity_when_configured():
    cfg = _cfg(num_slots=2, order=Order.FIFO, evict_by_severity=True)
    f = _push(F.new(1, cfg), ErrorCode.ZERO_OUTPUT, 1, Severity.WARN, cfg)
    f = _push(f, ErrorCode.INF, 2, Severity.WARN, cfg)  # full, both WARN
    f = _push(f, ErrorCode.NAN, 3, Severity.CRITICAL, cfg)  # more severe -> evicts a WARN
    codes = set(_codes(f, cfg))
    assert ErrorCode.NAN in codes
    assert len(codes) == 2


def test_dedupe_none_allows_duplicates():
    cfg = _cfg(dedupe=Dedupe.NONE)
    f = _push(F.new(1, cfg), ErrorCode.NAN, 1, Severity.CRITICAL, cfg)
    f = _push(f, ErrorCode.NAN, 1, Severity.CRITICAL, cfg)
    assert int(F.count(f, cfg)[0]) == 2


def test_dedupe_pair_collapses_and_upgrades_severity():
    cfg = _cfg(dedupe=Dedupe.PAIR)
    f = _push(F.new(1, cfg), ErrorCode.NAN, 1, Severity.WARN, cfg)
    f = _push(f, ErrorCode.NAN, 1, Severity.CRITICAL, cfg)  # same (loc, code) -> upgrade
    assert int(F.count(f, cfg)[0]) == 1
    assert F.unpack(f[0], cfg) == [(ErrorCode.NAN, 1, Severity.CRITICAL)]
    f = _push(f, ErrorCode.NAN, 1, Severity.WARN, cfg)  # not worse -> unchanged
    assert F.unpack(f[0], cfg) == [(ErrorCode.NAN, 1, Severity.CRITICAL)]


def test_dedupe_code_and_location():
    cfg_code = _cfg(dedupe=Dedupe.CODE)
    f = _push(F.new(1, cfg_code), ErrorCode.NAN, 1, Severity.CRITICAL, cfg_code)
    f = _push(f, ErrorCode.NAN, 2, Severity.CRITICAL, cfg_code)  # same code, diff loc
    assert int(F.count(f, cfg_code)[0]) == 1

    cfg_loc = _cfg(dedupe=Dedupe.LOCATION)
    g = _push(F.new(1, cfg_loc), ErrorCode.NAN, 5, Severity.CRITICAL, cfg_loc)
    g = _push(g, ErrorCode.INF, 5, Severity.CRITICAL, cfg_loc)  # same loc, diff code
    assert int(F.count(g, cfg_loc)[0]) == 1


def test_policy_default_is_lifo_none():
    p = ErrorConfig().accumulation
    assert p.order is Order.LIFO
    assert p.dedupe is Dedupe.NONE
    assert p.evict_by_severity is False
