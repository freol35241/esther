"""The forecaster interface (spec §7) and what every forecaster shares.

Every forecaster implements ``forecast(now, horizon, step) → Forecast``: step
means of its signals on the planning grid (:func:`plan_grid`), with
**provenance** — which source, which issue, when it became available and the
newest datum used — so that a backtest can prove that only information
available at ``now`` went in (§7, §10.2).

Two rules keep a forecast honest, and both are checked:

- A forecaster reads its history only **as of** ``now``: :class:`Issues`
  hands out only issues available by then (issue time plus an availability
  lag, since a numerical forecast is stamped with its model run time, before
  it can be fetched), and :func:`until` cuts a series at ``now``.
- Provenance records ``data_until``, the newest timestamp the forecast read;
  :func:`check_provenance` refuses a forecast that read past ``now``. A
  forecaster that is *meant* to see the future (the perfect-foresight bound,
  :class:`Perfect`) says so, and is refused unless the caller allows it.

Signals are in SI units (spec N4), as the model reads them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "Event",
    "Forecast",
    "Forecaster",
    "Issues",
    "LeakError",
    "Perfect",
    "Provenance",
    "check_provenance",
    "combine",
    "plan_grid",
    "step_means",
    "until",
]

TimeLike = pd.Timestamp | str
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 has only trapz


class LeakError(RuntimeError):
    """A forecast used information from after the time it was made for."""


def _ts(x: TimeLike) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        raise ValueError(f"{x!r} has no time zone; times are UTC with an explicit offset")
    return t.tz_convert("UTC")


def plan_grid(now: TimeLike, horizon: float, step: float) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The planning grid (§8.1): step starts and lengths (s) from ``now``.

    Steps are aligned to multiples of ``step`` since the epoch; the first is
    shortened to the next boundary (a first step under a tenth of ``step`` is
    merged into the next). The grid ends at the boundary at or after
    ``now + horizon``.
    """
    t = _ts(now)
    if step <= 0 or horizon <= 0:
        raise ValueError("step and horizon must be positive")
    s_ns = round(step * 1e9)
    t_ns = t.value
    nxt = -(-t_ns // s_ns) * s_ns  # ceil to the boundary
    if nxt - t_ns < 0.1 * s_ns and nxt != t_ns:
        nxt += s_ns
    if nxt == t_ns:
        nxt += s_ns
    end = -(-(t_ns + round(horizon * 1e9)) // s_ns) * s_ns
    edges = np.r_[t_ns, np.arange(nxt, end + 1, s_ns, dtype=np.int64)]
    starts = pd.DatetimeIndex(pd.to_datetime(edges[:-1], utc=True))
    return starts, np.diff(edges) / 1e9


def until(series: pd.Series, now: TimeLike, inclusive: bool = True) -> pd.Series:
    """``series`` cut at ``now`` — what was known then."""
    t = _ts(now)
    return series.loc[:t] if inclusive else series.loc[series.index < t]


def step_means(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    starts: pd.DatetimeIndex,
    dt: np.ndarray,
    kind: str = "instant",
    ends: pd.DatetimeIndex | None = None,
) -> np.ndarray:
    """Means of a forecast over each step ``[start, start + dt)``, **using the
    forecast's own timestamps** (§7.1: resolution coarsens with lead time).

    ``kind="instant"``: values at instants, linear between them, held flat
    beyond the ends. ``kind="interval"``: each value holds over ``[time,
    end)`` (``ends`` given, or the next time); the step mean is the
    overlap-weighted mean, NaN where nothing covers the step.
    """
    v = np.asarray(values, dtype=float)
    a = starts.as_unit("ns").asi8.astype(float) / 1e9
    b = a + np.asarray(dt, dtype=float)
    if len(times) == 0:
        return np.full(len(starts), np.nan)
    t = times.as_unit("ns").asi8.astype(float) / 1e9
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    e_all = None if ends is None else ends.as_unit("ns").asi8.astype(float)[order] / 1e9
    # only the points that can touch the grid (and one either side, for interpolation)
    lo_i = max(0, int(np.searchsorted(t, a[0], side="right")) - 2)
    hi_i = min(len(t), int(np.searchsorted(t, b[-1], side="left")) + 2)
    t, v = t[lo_i:hi_i], v[lo_i:hi_i]
    if e_all is not None:
        e_all = e_all[lo_i:hi_i]
    if len(t) == 0:
        return np.full(len(a), np.nan)
    if kind == "instant":
        # exact mean of the piecewise-linear interpolant: integrate on a fine
        # merged grid of breakpoints
        out = np.empty(len(a))
        for i, (lo, hi) in enumerate(zip(a, b, strict=True)):
            inner = t[(t > lo) & (t < hi)]
            x = np.r_[lo, inner, hi]
            y = np.interp(x, t, v)
            out[i] = _trapezoid(y, x) / (hi - lo)
        return out
    if kind != "interval":
        raise ValueError(f"kind must be 'instant' or 'interval', got {kind!r}")
    if e_all is None:
        e = np.r_[t[1:], t[-1] + (t[-1] - t[-2] if len(t) > 1 else 3600.0)]
    else:
        e = np.where(np.isfinite(e_all), e_all, np.r_[t[1:], t[-1] + 3600.0])
    out = np.full(len(a), np.nan)
    for i, (lo, hi) in enumerate(zip(a, b, strict=True)):
        ov = np.clip(np.minimum(e, hi) - np.maximum(t, lo), 0.0, None)
        w = ov.sum()
        if w > 0:
            out[i] = (ov * v).sum() / w
    return out


@dataclass(frozen=True)
class Provenance:
    """Where one forecast signal came from.

    ``issued``: the issue it was taken from (None for a learned profile).
    ``available``: when that issue could first have been read.
    ``data_until``: the newest timestamp of any datum read. ``method``: how.
    ``leaky``: the forecaster is *meant* to see the future (perfect foresight).
    """

    signal: str
    source: str
    method: str
    data_until: pd.Timestamp | None
    issued: pd.Timestamp | None = None
    available: pd.Timestamp | None = None
    leaky: bool = False
    note: str = ""


@dataclass(frozen=True)
class Event:
    """A device-internal event (§7.6) forecast or detected: it occupies
    ``[start, end)`` and belongs to ``component``."""

    component: str
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    forecast: bool = True


@dataclass
class Forecast:
    """Step means of one or more signals on a planning grid.

    ``values``: one column per signal, indexed by step start, SI units.
    ``estimated``: per signal and step, true where the value is a model's
    guess rather than a published forecast (§7.2: estimated prices).
    """

    now: pd.Timestamp
    dt: np.ndarray
    values: pd.DataFrame
    estimated: pd.DataFrame
    provenance: list[Provenance] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.values.index

    def __getitem__(self, signal: str) -> np.ndarray:
        return self.values[signal].to_numpy(dtype=float)

    def __contains__(self, signal: str) -> bool:
        return signal in self.values.columns

    def data_until(self) -> pd.Timestamp | None:
        ts = [p.data_until for p in self.provenance if p.data_until is not None]
        return max(ts) if ts else None


@runtime_checkable
class Forecaster(Protocol):
    """``forecast(now, horizon, step)`` → a :class:`Forecast` on :func:`plan_grid`."""

    signals: tuple[str, ...]

    def forecast(self, now: TimeLike, horizon: float, step: float) -> Forecast: ...


def empty(now: TimeLike, horizon: float, step: float) -> tuple[pd.Timestamp, pd.DatetimeIndex, np.ndarray]:
    t = _ts(now)
    starts, dt = plan_grid(t, horizon, step)
    return t, starts, dt


def single(
    now: pd.Timestamp,
    starts: pd.DatetimeIndex,
    dt: np.ndarray,
    signal: str,
    values: np.ndarray,
    estimated: np.ndarray | bool,
    provenance: Provenance | Sequence[Provenance],
    events: Iterable[Event] = (),
) -> Forecast:
    est = np.broadcast_to(np.asarray(estimated, dtype=bool), (len(starts),))
    prov = [provenance] if isinstance(provenance, Provenance) else list(provenance)
    return Forecast(
        now,
        np.asarray(dt, dtype=float),
        pd.DataFrame({signal: np.asarray(values, dtype=float)}, index=starts),
        pd.DataFrame({signal: est.copy()}, index=starts),
        prov,
        list(events),
    )


def combine(forecasts: Sequence[Forecast]) -> Forecast:
    """One forecast from several made at the same ``now`` on the same grid."""
    if not forecasts:
        raise ValueError("nothing to combine")
    f0 = forecasts[0]
    for f in forecasts[1:]:
        if f.now != f0.now or not f.index.equals(f0.index):
            raise ValueError("forecasts to combine must share now and grid")
    values = pd.concat([f.values for f in forecasts], axis=1)
    if values.columns.duplicated().any():
        raise ValueError(
            f"signal(s) forecast twice: {sorted(set(values.columns[values.columns.duplicated()]))}"
        )
    est = pd.concat([f.estimated for f in forecasts], axis=1)
    return Forecast(
        f0.now,
        f0.dt,
        values,
        est,
        [p for f in forecasts for p in f.provenance],
        [e for f in forecasts for e in f.events],
    )


def check_provenance(forecast: Forecast, allow_leaky: bool = False) -> None:
    """Refuse a forecast that read past its ``now`` (§10.2: a backtest must
    prove it used only what was available)."""
    for p in forecast.provenance:
        if p.leaky and allow_leaky:
            continue
        if p.leaky:
            raise LeakError(f"{p.signal}: {p.source} sees the future by design ({p.method})")
        if p.data_until is not None and p.data_until > forecast.now:
            raise LeakError(
                f"{p.signal}: {p.source} read data up to {p.data_until}, after now = {forecast.now}"
            )
        if p.available is not None and p.available > forecast.now:
            raise LeakError(f"{p.signal}: issue available {p.available}, after now = {forecast.now}")


# -- raw forecast issues ---------------------------------------------------------


@dataclass
class Issues:
    """Raw forecast issues: rows of ``issued``, ``valid`` (and optionally
    ``end``), ``value`` — every issue kept, as a history store records them.

    ``lag``: seconds between an issue's stamp and when it could first be
    read. A numerical weather forecast stamped with its model run time
    becomes available later; a fetch time needs none.
    """

    frame: pd.DataFrame
    lag: float = 0.0
    name: str = "issues"

    def __post_init__(self) -> None:
        f = self.frame
        missing = {"issued", "valid", "value"} - set(f.columns)
        if missing:
            raise ValueError(f"issues need columns issued, valid, value; missing {sorted(missing)}")
        f = f.copy()
        for c in ("issued", "valid", "end"):
            if c in f.columns:
                f[c] = pd.to_datetime(f[c], utc=True)
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
        self.frame = f.dropna(subset=["value"]).sort_values(["issued", "valid"]).reset_index(drop=True)
        self._issues = pd.DatetimeIndex(self.frame["issued"].drop_duplicates().sort_values())

    @property
    def has_end(self) -> bool:
        return "end" in self.frame.columns and self.frame["end"].notna().any()

    def available_at(self, issued: pd.Timestamp) -> pd.Timestamp:
        return issued + pd.Timedelta(seconds=self.lag)

    def latest(self, now: TimeLike) -> pd.Timestamp | None:
        """The newest issue readable at ``now``."""
        t = _ts(now) - pd.Timedelta(seconds=self.lag)
        i = self._issues.searchsorted(t, side="right") - 1
        return self._issues[i] if i >= 0 else None

    def issue(self, issued: pd.Timestamp) -> pd.DataFrame:
        return self.frame[self.frame["issued"] == issued]

    def newest_values(self, now: TimeLike) -> pd.DataFrame:
        """For each valid time, the value from the newest issue readable at
        ``now`` (right for day-ahead prices, which a newer issue only adds
        to): columns ``valid``, (``end``), ``value``, ``issued``."""
        t = _ts(now) - pd.Timedelta(seconds=self.lag)
        f = self.frame[self.frame["issued"] <= t]
        if f.empty:
            return f
        return f.sort_values("issued").drop_duplicates("valid", keep="last").sort_values("valid")


@dataclass
class Perfect:
    """Perfect foresight: the realised values of ``signals`` from ``frame``
    (indexed by time, SI), as step means. **It reads the future by design** —
    for a perfect-foresight bound — and its provenance says so, so a backtest
    refuses it unless told this is the bound."""

    frame: pd.DataFrame
    signals: tuple[str, ...]
    kind: Mapping[str, str] = field(default_factory=dict)

    def forecast(self, now: TimeLike, horizon: float, step: float) -> Forecast:
        t, starts, dt = empty(now, horizon, step)
        out = []
        for s in self.signals:
            col = self.frame[s].dropna()
            kind = self.kind.get(s, "interval")
            vals = step_means(pd.DatetimeIndex(col.index), col.to_numpy(dtype=float), starts, dt, kind)
            last = starts[-1] + pd.Timedelta(seconds=float(dt[-1]))
            prov = Provenance(s, "perfect", "realised values", last, leaky=True)
            out.append(single(t, starts, dt, s, vals, False, prov))
        return combine(out)
