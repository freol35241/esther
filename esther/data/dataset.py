"""The identification dataset (spec §6.1, F2).

:func:`build_dataset` turns per-signal sample histories into a regular grid
at the configured step (5 min by default): a time-weighted mean per step for
analog signals, the on-fraction for binary ones, the longest-held value for
categorical ones. Masked samples and exclusion windows are removed first.
A step where any required signal is not covered ends a **segment**; nothing is
interpolated across. :func:`read_signals` fetches the histories from any
:class:`HistorySource` given the integration layer's address map.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .config import DatasetConfig, SignalSpec
from .resample import MASK_RULES, hold_integrals, mask_samples

__all__ = ["Dataset", "HistorySource", "Segment", "build_dataset", "read_signals"]

TimeLike = dt.datetime | pd.Timestamp | str


class HistorySource(Protocol):
    """Anything that returns a signal's recorded samples.

    ``read`` returns a :class:`pandas.Series` indexed by a tz-aware UTC
    :class:`pandas.DatetimeIndex`, sorted, with numeric, boolean or string
    values, covering at least ``[start, end)``. ``address`` is opaque to esther.
    """

    def read(self, address: str, start: TimeLike | None = None, end: TimeLike | None = None) -> pd.Series: ...


@dataclass(frozen=True)
class Segment:
    """A run of usable steps ``[first, stop)`` in the dataset's index."""

    first: int
    stop: int
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive: the end of the last step

    @property
    def n_steps(self) -> int:
        return self.stop - self.first

    @property
    def duration(self) -> pd.Timedelta:
        return self.end - self.start


@dataclass
class Dataset:
    """A built dataset. Every frame shares ``index`` (step starts, UTC)."""

    config: DatasetConfig
    index: pd.DatetimeIndex
    values: pd.DataFrame  # NaN (None for categorical) where not valid
    coverage: pd.DataFrame  # fraction of each step covered by kept samples
    valid: pd.DataFrame  # bool: coverage ≥ the signal's min_coverage, not excluded
    excluded: pd.Series  # bool: the step overlaps a dataset exclusion window
    usable: pd.Series  # bool: every required signal valid and not excluded
    segments: list[Segment]
    masked: dict[str, pd.Series]  # per signal: masked samples' timestamps → rule
    report: pd.DataFrame  # per signal: counts

    def segment_frames(self, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        """The values of each segment, in order."""
        cols = list(self.values.columns) if columns is None else columns
        for s in self.segments:
            yield self.values.iloc[s.first : s.stop][cols]

    def summary(self) -> str:
        n = len(self.index)
        seg = sum(s.n_steps for s in self.segments)
        lines = [
            f"{n} steps of {self.config.step:g} s from {self.index[0] if n else '-'}",
            f"usable {int(self.usable.sum())}, excluded {int(self.excluded.sum())}, "
            f"in {len(self.segments)} segments: {seg} steps",
            self.report.to_string(),
        ]
        return "\n".join(lines)


def _utc(x: TimeLike) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        raise ValueError(f"{x!r} has no time zone; give a UTC offset")
    return ts.tz_convert("UTC")


def read_signals(
    source: HistorySource,
    addresses: Mapping[str, str],
    config: DatasetConfig,
    start: TimeLike,
    end: TimeLike,
) -> dict[str, pd.Series]:
    """Read every configured signal from ``source``. ``addresses`` maps each
    signal name to the source's address for it (the integration's mapping).
    Reads from one tolerance before ``start``, so the first step can be held."""
    missing = [n for n in config.signals if n not in addresses]
    if missing:
        raise KeyError(f"no address for signal(s): {', '.join(missing)}")
    s0, s1 = _utc(start), _utc(end)
    out = {}
    for name in config.signals:
        lead = pd.Timedelta(seconds=config.tolerance_of(name))
        out[name] = source.read(addresses[name], s0 - lead, s1)
    return out


_WORDS = {"true": 1.0, "on": 1.0, "yes": 1.0, "false": 0.0, "off": 0.0, "no": 0.0}


def _to_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, str):
        m = x.strip().lower()
        if m in _WORDS:
            return _WORDS[m]
        try:
            return float(m)
        except ValueError:
            return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _numeric(spec: SignalSpec, s: pd.Series) -> np.ndarray:
    """Values as float, NaN where not a number. Booleans and the words
    true/false, on/off, yes/no count as 1/0. Binary: any non-zero is on."""
    if pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype):
        out = s.to_numpy(dtype=float, na_value=np.nan)
    else:
        out = np.fromiter((_to_float(x) for x in s.to_numpy(dtype=object)), dtype=float, count=len(s))
    if spec.kind == "binary":
        out = np.where(np.isnan(out), np.nan, (out != 0).astype(float))
    return out


def build_dataset(
    config: DatasetConfig,
    samples: Mapping[str, pd.Series],
    start: TimeLike | None = None,
    end: TimeLike | None = None,
) -> Dataset:
    """Build the dataset from ``samples`` (signal name → raw history).

    The grid is aligned to multiples of the step since the Unix epoch; the
    first step starts at or before ``start``, the last ends at or after
    ``end``. Without ``start``/``end`` the span of the samples is used.
    """
    missing = [n for n in config.signals if n not in samples]
    if missing:
        raise KeyError(f"no samples for signal(s): {', '.join(missing)}")
    step = config.step
    step_ns = round(step * 1e9)

    def times(s: pd.Series) -> np.ndarray:
        idx = s.index
        if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
            raise ValueError("sample histories need a tz-aware DatetimeIndex")
        return idx.tz_convert("UTC").as_unit("ns").asi8

    spans = [times(samples[n]) for n in config.signals]
    t_lo = _utc(start).value if start is not None else min((x[0] for x in spans if len(x)), default=None)
    t_hi = _utc(end).value if end is not None else max((x[-1] for x in spans if len(x)), default=None)
    if t_lo is None or t_hi is None:
        raise ValueError("no samples and no start/end: the dataset would be empty")
    g0 = (t_lo // step_ns) * step_ns
    g1 = -((-t_hi) // step_ns) * step_ns
    if g1 <= g0:
        g1 = g0 + step_ns
    n = (g1 - g0) // step_ns
    index = pd.DatetimeIndex(pd.to_datetime(g0 + step_ns * np.arange(n), utc=True), name="time")
    edges = (np.arange(n + 1) * step).astype(float)  # seconds since g0

    def rel(ns: np.ndarray | int) -> np.ndarray:
        return (np.asarray(ns, dtype=np.int64) - g0) / 1e9

    def win(w) -> tuple[float, float]:
        return float(rel(pd.Timestamp(w.start).value)), float(rel(pd.Timestamp(w.end).value))

    global_windows = [win(w) for w in config.exclude]
    excluded = np.zeros(n, dtype=bool)
    for a, b in global_windows:
        excluded |= (edges[:-1] < b) & (edges[1:] > a)

    values: dict[str, Any] = {}
    coverage: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    masked: dict[str, pd.Series] = {}
    rows = []
    for name, spec in config.signals.items():
        s = samples[name]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        t = rel(times(s))
        windows = global_windows + [win(w) for w in spec.exclude]
        breaks = np.array(sorted(a for a, _ in windows), dtype=float)
        tol = config.tolerance_of(name)

        if spec.kind == "categorical":
            raw = s.to_numpy(dtype=object)
            num = np.where(pd.isna(raw), np.nan, 0.0)
        else:
            num = _numeric(spec, s)
        rule = mask_samples(t, num, spec, windows)
        keep = rule < 0
        masked[name] = pd.Series(
            np.asarray(MASK_RULES, dtype=object)[rule[~keep]], index=s.index[~keep], name=name
        )

        tk = t[keep]
        if spec.kind == "categorical":
            cats = pd.unique(raw[keep])
            _, cov_t = hold_integrals(tk, np.ones(len(tk)), edges, tol, breaks)
            best = np.full(n, None, dtype=object)
            best_t = np.zeros(n)
            for c in cats:
                ind = (raw[keep] == c).astype(float)
                it, _ = hold_integrals(tk, ind, edges, tol, breaks)
                better = it > best_t
                best[better], best_t[better] = c, it[better]
            vals = best
        else:
            integ, cov_t = hold_integrals(tk, num[keep], edges, tol, breaks)
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = integ / cov_t
            factor, offset = spec.to_si()
            vals = vals * factor + offset
        cov = cov_t / step
        ok = (cov >= config.coverage_of(name) - 1e-9) & ~excluded
        vals = np.where(ok, vals, None) if spec.kind == "categorical" else np.where(ok, vals, np.nan)
        values[name], coverage[name], valid[name] = vals, cov, ok

        counts = pd.Series(rule[~keep]).value_counts()
        rows.append(
            {
                "signal": name,
                "kind": spec.kind,
                "required": spec.required,
                "samples": len(t),
                **{f"masked_{r}": int(counts.get(i, 0)) for i, r in enumerate(MASK_RULES)},
                "steps_valid": int(ok.sum()),
                "steps_valid_pct": round(100.0 * ok.sum() / max(1, (~excluded).sum()), 2),
            }
        )

    values_df = pd.DataFrame(values, index=index)
    coverage_df = pd.DataFrame(coverage, index=index)
    valid_df = pd.DataFrame(valid, index=index)
    req = list(config.required)
    usable = valid_df[req].all(axis=1).to_numpy() & ~excluded if req else ~excluded

    segments = []
    min_steps = max(1, int(np.ceil(config.min_segment / step - 1e-9)))
    d = np.diff(np.r_[0, usable.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
        if b - a >= min_steps:
            segments.append(Segment(int(a), int(b), index[a], index[b - 1] + pd.Timedelta(seconds=step)))

    return Dataset(
        config=config,
        index=index,
        values=values_df,
        coverage=coverage_df,
        valid=valid_df,
        excluded=pd.Series(excluded, index=index, name="excluded"),
        usable=pd.Series(usable, index=index, name="usable"),
        segments=segments,
        masked=masked,
        report=pd.DataFrame(rows).set_index("signal"),
    )
