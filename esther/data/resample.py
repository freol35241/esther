"""Masks and hold-aggregation onto a regular grid (spec §6.1).

A history is a list of samples ``(t_i, v_i)``. Each sample is taken to hold
its value from ``t_i`` until the next sample, but never longer than the
signal's *tolerance* and never across a *break* (the start of an exclusion
window). Past that, the signal is missing. Nothing is interpolated: a step's
value is the time-weighted mean of the held values inside it, and its
*coverage* is the fraction of the step that a held value covers. For a binary
signal the same mean is the on-fraction δ.

Everything here works on plain numpy arrays of float seconds; the dataset
builder converts timestamps once.
"""

from __future__ import annotations

import numpy as np

from .config import SignalSpec

__all__ = ["MASK_RULES", "hold_integrals", "mask_samples"]

#: Mask rules in the order they are applied; a sample masked by one is not
#: tested by the later ones (a sentinel value must not distort a spike median).
MASK_RULES = ("excluded", "invalid_value", "range", "stuck", "spike")


def mask_samples(
    t: np.ndarray, v: np.ndarray, spec: SignalSpec, windows: list[tuple[float, float]]
) -> np.ndarray:
    """Return, per sample, the index into :data:`MASK_RULES` of the rule that
    masked it, or ``-1`` for a kept sample. ``v`` is numeric (NaN counts as
    an invalid value); ``windows`` are ``(start, end)`` in the same seconds."""
    n = len(t)
    rule = np.full(n, -1, dtype=np.int8)
    if n == 0:
        return rule

    def apply(i: int, hit: np.ndarray) -> None:
        rule[(rule < 0) & hit] = i

    excl = np.zeros(n, dtype=bool)
    for a, b in windows:
        excl |= (t >= a) & (t < b)
    apply(0, excl)

    bad = ~np.isfinite(v)
    for x in spec.invalid_values:
        bad |= np.isclose(v, x, rtol=1e-6, atol=1e-4)
    apply(1, bad)

    if spec.range is not None:
        lo, hi = spec.range
        with np.errstate(invalid="ignore"):
            apply(2, (v < lo) | (v > hi))

    if spec.stuck is not None:
        apply(3, _stuck(t, v, rule < 0, spec.stuck))

    if spec.spike is not None:
        apply(4, _spikes(v, rule < 0, spec.spike.threshold, spec.spike.window))
    return rule


def _stuck(t: np.ndarray, v: np.ndarray, keep: np.ndarray, limit: float) -> np.ndarray:
    """Runs of identical consecutive (kept) values lasting ``limit`` or more."""
    out = np.zeros(len(t), dtype=bool)
    idx = np.flatnonzero(keep)
    if len(idx) < 2:
        return out
    tv, vv = t[idx], v[idx]
    new_run = np.r_[True, vv[1:] != vv[:-1]]
    run_id = np.cumsum(new_run) - 1
    first = np.flatnonzero(new_run)
    last = np.r_[first[1:] - 1, len(vv) - 1]
    long = (tv[last] - tv[first]) >= limit
    out[idx] = long[run_id]
    return out


def _spikes(v: np.ndarray, keep: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """Samples further than ``threshold`` from their centred running median."""
    out = np.zeros(len(v), dtype=bool)
    idx = np.flatnonzero(keep)
    if len(idx) < window:
        return out
    vv = v[idx]
    h = window // 2
    padded = np.pad(vv, h, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)
    out[idx] = np.abs(vv - med) > threshold
    return out


def hold_integrals(
    t: np.ndarray,
    v: np.ndarray,
    edges: np.ndarray,
    tolerance: float,
    breaks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the held signal over each bin ``[edges[j], edges[j+1])``.

    Returns ``(integral of v dt, covered time)`` per bin. ``t`` sorted, kept
    samples only; ``breaks`` sorted times at which every hold ends.
    """
    nb = len(edges) - 1
    if len(t) == 0 or nb <= 0:
        return np.zeros(max(nb, 0)), np.zeros(max(nb, 0))
    a = t
    b = np.minimum(np.r_[t[1:], np.inf], t + tolerance)
    if len(breaks):
        k = np.searchsorted(breaks, a, side="right")
        nxt = np.where(k < len(breaks), breaks[np.minimum(k, len(breaks) - 1)], np.inf)
        b = np.minimum(b, nxt)
    length = np.maximum(b - a, 0.0)

    def cumulative(weights: np.ndarray) -> np.ndarray:
        full = np.r_[0.0, np.cumsum(weights * length)]
        k = np.searchsorted(a, edges, side="right") - 1
        kc = np.maximum(k, 0)
        part = weights[kc] * np.clip(edges - a[kc], 0.0, length[kc])
        return np.where(k >= 0, full[kc] + part, 0.0)

    vi = cumulative(np.nan_to_num(v))
    ci = cumulative(np.ones_like(a))
    return np.diff(vi), np.diff(ci)
