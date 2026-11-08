"""Forecasts learned from the house's own history (§7.3, §7.4, and the
fallbacks of §7.1–7.2).

Nothing here is configured by hand: each forecaster re-learns its profile
from the history **as of** ``now`` at every call, with recent days weighted
more (``halflife``), so it follows the household as data accrues — the
self-learning principle of G2 applied to the exogenous inputs.

- :class:`TimeOfWeekProfile` — a per-slot statistic (weighted mean or
  quantile) over a day or a week, in local time (habits are local).
- :class:`ProfileForecaster` — any measured signal as its learned profile
  (an irradiance proxy with no forecast of its own, say).
- :class:`OccupancyForecaster` — §7.3: the time-of-week probability of
  someone at home, overridden by the current state for the first hours.
- :class:`DrawForecaster` — §7.4: hot-water draws as a **high quantile**
  of each slot's energy over past days, optionally conditioned on occupancy.
- :class:`HoldForecaster` — the last value held (a setpoint).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import Forecast, Provenance, empty, single, until

__all__ = [
    "DrawForecaster",
    "HoldForecaster",
    "OccupancyForecaster",
    "ProfileForecaster",
    "TimeOfWeekProfile",
]


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return float(np.interp(q, c, x))


def held(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, freq: str) -> pd.Series:
    """A sample-and-hold signal (each sample holds until the next) as means
    on a regular grid ``[start, end)``."""
    s = series.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    grid = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
    if s.empty or len(grid) == 0:
        return pd.Series(dtype=float)
    v = s.reindex(s.index.union(grid)).ffill().reindex(grid)
    return v.resample(freq).mean()


@dataclass
class TimeOfWeekProfile:
    """Per-slot statistic of a series over a ``period`` (``"day"`` or
    ``"week"``), slots of ``slot`` minutes, in time zone ``tz``.

    ``quantile`` None gives the weighted mean; otherwise that quantile of
    the per-day slot values. Samples are weighted ``0.5 ** (age / halflife)``.
    A week slot with fewer than ``min_days`` days falls back to its day slot.
    """

    period: str = "day"
    slot: int = 60
    tz: str = "UTC"
    halflife: float = 14 * 86400.0
    quantile: float | None = None
    min_days: int = 3
    table: pd.Series | None = field(default=None, repr=False)
    day_table: pd.Series | None = field(default=None, repr=False)
    fitted_until: pd.Timestamp | None = None

    def _slot(self, idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        loc = idx.tz_convert(self.tz)
        mins = loc.hour * 60 + loc.minute
        day_slot = np.asarray(mins // self.slot, dtype=int)
        week_slot = np.asarray(loc.day_of_week, dtype=int) * (1440 // self.slot) + day_slot
        return day_slot, week_slot

    def fit(self, series: pd.Series) -> TimeOfWeekProfile:
        """``series``: regular slot means (or any samples, averaged per slot)."""
        s = series.dropna()
        if s.empty:
            raise ValueError("no data to learn a profile from")
        s = s.resample(f"{self.slot}min").mean().dropna()
        end = s.index.max()
        age = (end - s.index).total_seconds().to_numpy()
        w = 0.5 ** (age / self.halflife)
        day_slot, week_slot = self._slot(pd.DatetimeIndex(s.index))
        v = s.to_numpy(dtype=float)

        def stat(keys: np.ndarray) -> pd.Series:
            out = {}
            for k in np.unique(keys):
                m = keys == k
                if self.quantile is None:
                    out[k] = float((v[m] * w[m]).sum() / w[m].sum())
                else:
                    out[k] = _weighted_quantile(v[m], w[m], self.quantile)
            return pd.Series(out)

        self.day_table = stat(day_slot)
        if self.period == "week":
            counts = pd.Series(week_slot).value_counts()
            tab = stat(week_slot)
            ok = counts.reindex(tab.index).fillna(0) >= self.min_days
            self.table = tab[ok]
        elif self.period == "day":
            self.table = None
        else:
            raise ValueError(f"period must be 'day' or 'week', got {self.period!r}")
        self.fitted_until = end
        return self

    def predict(self, starts: pd.DatetimeIndex, dt: np.ndarray) -> np.ndarray:
        if self.day_table is None:
            raise RuntimeError("profile not fitted")
        mid = pd.DatetimeIndex(starts + pd.to_timedelta(np.asarray(dt) / 2, unit="s"))
        day_slot, week_slot = self._slot(mid)
        fallback = float(self.day_table.mean())
        out = self.day_table.reindex(day_slot).to_numpy(dtype=float)
        out = np.where(np.isnan(out), fallback, out)
        if self.table is not None and len(self.table):
            wk = self.table.reindex(week_slot).to_numpy(dtype=float)
            out = np.where(np.isnan(wk), out, wk)
        return out


@dataclass
class ProfileForecaster:
    """A measured signal forecast as its learned profile (mean, or a
    quantile) over the last ``window_days``. ``hold``: the series is
    sample-and-hold (a state that holds until it changes), not samples of a
    continuous quantity."""

    series: pd.Series
    signal: str
    period: str = "day"
    slot: int = 60
    window_days: float = 14.0
    halflife: float = 7 * 86400.0
    quantile: float | None = None
    hold: bool = False
    tz: str = "UTC"
    default: float | None = None

    @property
    def signals(self) -> tuple[str, ...]:
        return (self.signal,)

    def history(self, t: pd.Timestamp) -> pd.Series:
        lo = t - pd.Timedelta(days=self.window_days)
        if self.hold:
            s = until(self.series, t)
            prior = s[s.index < lo].tail(1)
            s = pd.concat([prior, s[s.index >= lo]])
            return held(s, lo, t, f"{self.slot}min")
        s = until(self.series, t)
        return s[s.index >= lo]

    def forecast(self, now, horizon: float, step: float) -> Forecast:
        t, starts, dt = empty(now, horizon, step)
        h = self.history(t)
        if h.dropna().empty:
            if self.default is None:
                raise LookupError(f"{self.signal}: no history before {t} to learn a profile from")
            prov = Provenance(self.signal, "profile", "no history: default", None, note="default")
            return single(t, starts, dt, self.signal, np.full(len(starts), self.default), True, prov)
        prof = TimeOfWeekProfile(self.period, self.slot, self.tz, self.halflife, self.quantile).fit(h)
        vals = prof.predict(starts, dt)
        prov = Provenance(
            self.signal,
            "profile",
            f"{self.period} profile over {self.window_days:g} d"
            + ("" if self.quantile is None else f", quantile {self.quantile:g}"),
            h.dropna().index.max(),
        )
        return single(t, starts, dt, self.signal, vals, True, prov)


@dataclass
class HoldForecaster:
    """The last value at ``now``, held over the horizon (a setpoint: a
    change is known only when it is made)."""

    series: pd.Series
    signal: str
    default: float | None = None

    @property
    def signals(self) -> tuple[str, ...]:
        return (self.signal,)

    def forecast(self, now, horizon: float, step: float) -> Forecast:
        t, starts, dt = empty(now, horizon, step)
        s = until(self.series.dropna(), t)
        if s.empty:
            if self.default is None:
                raise LookupError(f"{self.signal}: no value at {t}")
            v, last = self.default, None
        else:
            v, last = float(s.iloc[-1]), s.index[-1]
        prov = Provenance(self.signal, "hold", "last value held", last)
        return single(t, starts, dt, self.signal, np.full(len(starts), v), False, prov)


@dataclass
class OccupancyForecaster:
    """§7.3. ``presence``: 1 when someone is home, 0 when not, sample-and-hold.
    The forecast is the probability of someone at home: the learned
    time-of-week profile, overridden by the current state for the first
    hours, ``p = w · now + (1 − w) · profile`` with ``w = exp(−lead /
    persistence)`` (someone at home now is likely home in 30 minutes).
    ``default`` is the profile before any history exists."""

    presence: pd.Series
    signal: str = "occupancy"
    persistence: float = 2 * 3600.0
    window_days: float = 28.0
    halflife: float = 14 * 86400.0
    tz: str = "UTC"
    default: float = 1.0

    @property
    def signals(self) -> tuple[str, ...]:
        return (self.signal,)

    def forecast(self, now, horizon: float, step: float) -> Forecast:
        t, starts, dt = empty(now, horizon, step)
        s = until(self.presence.dropna().astype(float), t)
        lo = t - pd.Timedelta(days=self.window_days)
        if s.empty:
            prof = np.full(len(starts), self.default)
            current, last, method = None, None, "no presence history: default"
        else:
            prior = s[s.index < lo].tail(1)
            h = held(pd.concat([prior, s[s.index >= lo]]), max(lo, s.index[0]), t, "60min")
            if len(h.dropna()) >= 24:
                prof = (
                    TimeOfWeekProfile("week", 60, self.tz, self.halflife, min_days=2)
                    .fit(h)
                    .predict(starts, dt)
                )
                method = "time-of-week profile + current state"
            else:
                prof = np.full(len(starts), self.default)
                method = "current state, default profile"
            current, last = float(s.iloc[-1]), s.index[-1]
        if current is not None:
            lead = ((starts + pd.to_timedelta(dt / 2, unit="s")) - t).total_seconds().to_numpy()
            w = np.exp(-lead / self.persistence)
            vals = w * current + (1 - w) * prof
        else:
            vals = prof
        prov = Provenance(self.signal, "presence", method, last)
        return single(t, starts, dt, self.signal, np.clip(vals, 0, 1), True, prov)


@dataclass
class DrawForecaster:
    """§7.4: hot-water draws from detected draw events.

    ``draws``: the detected draw heat per regular step (W, ≥ 0; NaN where
    detection did not run). The forecast is a high ``quantile`` (default
    the 80th: a cold shower costs more goodwill than the energy saved) of
    the household's draws, learned over the past ``window_days``:

    - ``method="daily"`` (default): the q-quantile of the **daily** energy,
      spread over the day by the slots' mean shape. Draws are sparse — most
      days have none in a given hour — so a per-slot quantile is mostly
      zero, and summed it falls *below* the mean; the daily quantile keeps
      the level where §7.4 means it.
    - ``method="slot"``: the q-quantile of each slot's energy across days.

    ``sign`` −1 gives the signal as heat *into* the node (a draw removes heat).

    Conditioned on occupancy when an ``occupancy`` forecaster is given: the
    slot forecast is scaled by the forecast probability of someone home over
    the slot's mean presence in the history, clipped to ``[0, 1.5]``.
    """

    draws: pd.Series
    signal: str
    quantile: float = 0.8
    slot: int = 60
    period: str = "day"
    window_days: float = 28.0
    halflife: float = 14 * 86400.0
    sign: float = 1.0
    tz: str = "UTC"
    occupancy: OccupancyForecaster | None = None
    presence: pd.Series | None = None
    min_days: float = 2.0
    method: str = "daily"

    @property
    def signals(self) -> tuple[str, ...]:
        return (self.signal,)

    def forecast(self, now, horizon: float, step: float) -> Forecast:
        t, starts, dt = empty(now, horizon, step)
        lo = t - pd.Timedelta(days=self.window_days)
        s = until(self.draws, t, inclusive=False)
        s = s[s.index >= lo]
        covered = s.dropna()
        if (
            covered.empty
            or (covered.index.max() - covered.index.min()).total_seconds() < self.min_days * 86400
        ):
            prov = Provenance(self.signal, "draws", "too little history: no draws forecast", None)
            return single(t, starts, dt, self.signal, np.zeros(len(starts)), True, prov)
        # slot means (W) over covered time; a slot is kept when mostly covered
        cov = s.notna().astype(float).resample(f"{self.slot}min").mean()
        mean = s.resample(f"{self.slot}min").mean()
        mean = mean[cov >= 0.5]
        if self.method == "slot":
            prof = TimeOfWeekProfile(self.period, self.slot, self.tz, self.halflife, self.quantile).fit(mean)
            vals = np.clip(prof.predict(starts, dt), 0.0, None)
            method = f"quantile {self.quantile:g} per {self.slot}-min slot over {self.window_days:g} d"
        elif self.method == "daily":
            shape = TimeOfWeekProfile(self.period, self.slot, self.tz, self.halflife).fit(mean)
            vals = np.clip(shape.predict(starts, dt), 0.0, None)
            loc = s.copy()
            loc.index = loc.index.tz_convert(self.tz)
            daycov = loc.notna().resample("1D").mean()
            day = loc.resample("1D").mean()[daycov >= 0.8]  # W, mean over each well-covered day
            if len(day) >= 2:
                age = (day.index.max() - day.index).total_seconds().to_numpy()
                w = 0.5 ** (age / self.halflife)
                level = _weighted_quantile(day.to_numpy(dtype=float), w, self.quantile)
                mean_level = float(np.average(day.to_numpy(dtype=float), weights=w))
                vals = vals * (level / mean_level if mean_level > 0 else 1.0)
            method = (
                f"daily quantile {self.quantile:g} over {self.window_days:g} d, shaped by the "
                f"{self.slot}-min slot means"
            )
        else:
            raise ValueError(f"method must be 'daily' or 'slot', got {self.method!r}")
        if self.occupancy is not None and self.presence is not None:
            occ = self.occupancy.forecast(t, horizon, step)[self.occupancy.signal]
            ph = held(until(self.presence.dropna().astype(float), t), lo, t, f"{self.slot}min")
            if len(ph.dropna()) >= 24:
                pprof = TimeOfWeekProfile(self.period, self.slot, self.tz, self.halflife).fit(ph)
                base = pprof.predict(starts, dt)
                scale = np.where(base > 0.05, occ / np.maximum(base, 0.05), 1.0)
                vals = vals * np.clip(scale, 0.0, 1.5)
                method += ", conditioned on occupancy"
        prov = Provenance(self.signal, "draws", method, covered.index.max())
        return single(t, starts, dt, self.signal, self.sign * vals, True, prov)
