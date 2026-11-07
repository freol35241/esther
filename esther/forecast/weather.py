"""Weather (§7.1) and price (§7.2) from published forecast issues."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Forecast, Issues, Provenance, empty, single, step_means, until
from .profile import TimeOfWeekProfile

__all__ = ["PriceForecaster", "WeatherForecaster"]


@dataclass
class WeatherForecaster:
    """A numerical forecast's newest issue readable at ``now``, as step means
    on its own timestamps, **blended from the current measurement**:

        f(t) = fc(t) + (m_now − fc(now)) · exp(−(t − now) / tau)

    — a forecast is often biased at the house for the first hours (§7.1).
    ``measured`` is the house's own sensor (SI); without it, or with it
    silent for longer than ``max_measurement_age``, there is no blend.
    Beyond the issue's last point the last value is held, flagged
    estimated. ``kind``: ``"instant"`` (a temperature) or ``"interval"``
    (an accumulation or a rate over each point's interval).
    """

    issues: Issues
    signal: str
    measured: pd.Series | None = None
    tau: float = 3 * 3600.0
    kind: str = "instant"
    max_measurement_age: float = 3600.0

    @property
    def signals(self) -> tuple[str, ...]:
        return (self.signal,)

    def forecast(self, now, horizon: float, step: float) -> Forecast:
        t, starts, dt = empty(now, horizon, step)
        issued = self.issues.latest(t)
        if issued is None:
            raise LookupError(f"{self.signal}: no forecast issue readable at {t}")
        f = self.issues.issue(issued)
        times = pd.DatetimeIndex(f["valid"])
        ends = pd.DatetimeIndex(f["end"]) if self.issues.has_end else None
        vals = step_means(times, f["value"].to_numpy(), starts, dt, self.kind, ends)
        last_valid = (
            (ends.max() if ends is not None and ends.notna().any() else times.max()) if len(times) else t
        )
        mid = starts + pd.to_timedelta(dt / 2, unit="s")
        beyond = (mid > last_valid) | np.isnan(vals)
        if np.isnan(vals).all():
            raise LookupError(f"{self.signal}: issue {issued} does not cover the horizon")
        vals = pd.Series(vals).ffill().bfill().to_numpy()
        data_until = t
        note = ""
        if self.measured is not None:
            m = until(self.measured.dropna(), t)
            if len(m) and (t - m.index[-1]).total_seconds() <= self.max_measurement_age:
                fc_now = float(
                    np.interp(t.value / 1e9, times.as_unit("ns").asi8 / 1e9, f["value"].to_numpy())
                )
                bias = float(m.iloc[-1]) - fc_now
                lead = (mid - t).total_seconds().to_numpy()
                vals = vals + bias * np.exp(-lead / self.tau)
                data_until = m.index[-1]
                note = f"blended from the measurement {bias:+.2f} at now, tau {self.tau / 3600:g} h"
        prov = Provenance(
            self.signal,
            self.issues.name,
            "newest issue, step means on its own timestamps",
            max(data_until, issued) if data_until is not None else issued,
            issued=issued,
            available=self.issues.available_at(issued),
            note=note,
        )
        return single(t, starts, dt, self.signal, vals, beyond, prov)


@dataclass
class PriceForecaster:
    """Day-ahead prices as published (§7.2): for each step, the newest
    readable issue's value at its native resolution. Beyond the published
    horizon, the time-of-week profile of the last ``profile_days`` of
    realised prices, **flagged estimated** — the planner values them only
    through the terminal value (§8.6) rather than optimising against a guess.
    """

    issues: Issues
    signal: str
    profile_days: float = 28.0
    tz: str = "UTC"

    @property
    def signals(self) -> tuple[str, ...]:
        return (self.signal,)

    def realised(self, now) -> pd.Series:
        """Realised prices up to ``now``: per valid time, the newest issue."""
        nv = self.issues.newest_values(now)
        if nv.empty:
            return pd.Series(dtype=float)
        s = pd.Series(nv["value"].to_numpy(), index=pd.DatetimeIndex(nv["valid"]))
        return until(s, now, inclusive=False)

    def forecast(self, now, horizon: float, step: float) -> Forecast:
        t, starts, dt = empty(now, horizon, step)
        nv = self.issues.newest_values(t)
        if nv.empty:
            raise LookupError(f"{self.signal}: no price issue readable at {t}")
        ends = pd.DatetimeIndex(nv["end"]) if self.issues.has_end else None
        vals = step_means(pd.DatetimeIndex(nv["valid"]), nv["value"].to_numpy(), starts, dt, "interval", ends)
        est = np.isnan(vals)
        provs = [
            Provenance(
                self.signal,
                self.issues.name,
                "newest issue per valid time",
                t,
                issued=pd.Timestamp(nv["issued"].max()),
                available=self.issues.available_at(pd.Timestamp(nv["issued"].max())),
            )
        ]
        if est.any():
            hist = self.realised(t)
            hist = hist[hist.index >= t - pd.Timedelta(days=self.profile_days)]
            if hist.empty:
                raise LookupError(f"{self.signal}: no realised prices to build a profile from")
            prof = TimeOfWeekProfile(
                period="week" if len(hist) > 0 and _span_days(hist) >= 14 else "day", tz=self.tz
            )
            guess = prof.fit(hist).predict(starts[est], dt[est])
            vals[est] = guess
            provs.append(
                Provenance(
                    self.signal,
                    self.issues.name,
                    f"{prof.period}-profile of the last {self.profile_days:g} d of realised prices",
                    hist.index.max(),
                    note="estimated",
                )
            )
        return single(t, starts, dt, self.signal, vals, est, provs)


def _span_days(s: pd.Series) -> float:
    return (s.index.max() - s.index.min()).total_seconds() / 86400.0
