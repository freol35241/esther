"""Device-internal events (§7.6) and manual firing (§7.5).

- :func:`detect_episodes` finds the runs of a signal above a threshold.
- :class:`PeriodicEventForecaster` learns an event's period and duration
  from its past occurrences (detected from history as of ``now``) and
  forecasts the next ones as fixed windows — the plan treats them as fixed
  loads and MUST NOT plan to suppress them (§8.4).
- :class:`ManualFiringDetector` — a stove lit by hand: detected, never
  forecast ahead; while detected it is an exogenous heat input that decays
  after the signature ends.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Event, Forecast, Provenance, empty, single, until

__all__ = ["ManualFiringDetector", "PeriodicEventForecaster", "detect_episodes"]


def detect_episodes(
    series: pd.Series, threshold: float, min_duration: float = 0.0, max_gap: float = 0.0
) -> pd.DataFrame:
    """Runs where ``series > threshold``: ``start``, ``end`` (the first sample
    after), ``duration`` (s). Runs separated by at most ``max_gap`` seconds
    are merged; runs shorter than ``min_duration`` are dropped."""
    s = series.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=["start", "end", "duration"])
    on = s.to_numpy(dtype=float) > threshold
    t = s.index
    runs = []
    i, n = 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1]:
            j += 1
        end = t[j + 1] if j + 1 < n else t[j]
        runs.append([t[i], end])
        i = j + 1
    merged: list[list[pd.Timestamp]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]).total_seconds() <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = pd.DataFrame(merged, columns=["start", "end"])
    out["duration"] = (out.end - out.start).dt.total_seconds()
    return out[out.duration >= min_duration].reset_index(drop=True)


@dataclass
class PeriodicEventForecaster:
    """An event the device schedules itself, forecast from its observed
    period and last occurrence.

    ``detector(now)`` returns the occurrences known at ``now`` (columns
    ``start``, ``end``) from history as of then — the forecaster re-learns
    the period (median interval) and duration (median) at every call. The
    next start is the last start plus the period; an event under way at
    ``now`` runs to its start plus the duration. Output: the signal
    ``<component>.<kind>`` — the share of each step an event covers — and
    :class:`Event` windows.
    """

    component: str
    kind: str
    detector: Callable[[pd.Timestamp], pd.DataFrame]
    min_occurrences: int = 2
    margin: float = 0.0  # s, widen each window on both sides
    min_period: float = 86400.0  # s: occurrences closer than this are one (a split episode)

    @property
    def signal(self) -> str:
        return f"{self.component}.{self.kind}"

    @property
    def signals(self) -> tuple[str, ...]:
        return (self.signal,)

    def learn(self, now) -> dict:
        occ = self.detector(pd.Timestamp(now)).sort_values("start")
        occ = occ[occ.start <= now].reset_index(drop=True)
        if len(occ) > 1:  # merge occurrences closer than min_period (one event seen as two)
            keep = [0]
            for i in range(1, len(occ)):
                j = keep[-1]
                if (occ.start[i] - occ.start[j]).total_seconds() < self.min_period:
                    occ.loc[j, "end"] = max(occ.end[j], occ.end[i])
                else:
                    keep.append(i)
            occ = occ.loc[keep].reset_index(drop=True)
        out = {"n": len(occ), "period": None, "duration": None, "last": None, "until": None}
        if len(occ):
            out["last"] = occ.start.iloc[-1]
            out["until"] = min(pd.Timestamp(now), max(occ.start.max(), occ.end.max()))
            out["duration"] = float(np.median((occ.end - occ.start).dt.total_seconds()))
        if len(occ) >= self.min_occurrences:
            out["period"] = float(np.median(occ.start.diff().dt.total_seconds().dropna().to_numpy()))
        return out

    def forecast(self, now, horizon: float, step: float) -> Forecast:
        t, starts, dt = empty(now, horizon, step)
        L = self.learn(t)
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if L["last"] is not None and L["duration"] is not None:
            dur = pd.Timedelta(seconds=L["duration"])
            if L["last"] + dur > t:
                windows.append((L["last"], L["last"] + dur))
            if L["period"]:
                per = pd.Timedelta(seconds=L["period"])
                nxt = L["last"] + per
                end = t + pd.Timedelta(seconds=horizon)
                while nxt < end:
                    if nxt + dur > t:
                        windows.append((nxt, nxt + dur))
                    nxt += per
        m = pd.Timedelta(seconds=self.margin)
        windows = [(a - m, b + m) for a, b in windows]
        a = starts.as_unit("ns").asi8 / 1e9
        b = a + dt
        cover = np.zeros(len(starts))
        for lo, hi in windows:
            ov = np.clip(np.minimum(b, hi.value / 1e9) - np.maximum(a, lo.value / 1e9), 0, None)
            cover = np.maximum(cover, ov / dt)
        events = [Event(self.component, self.kind, lo, hi) for lo, hi in windows]
        method = (
            f"period {L['period'] / 86400:.2f} d, duration {L['duration'] / 3600:.2f} h "
            f"from {L['n']} occurrences"
            if L["period"]
            else f"{L['n']} occurrence(s): no period yet"
        )
        prov = Provenance(self.signal, "events", method, L["until"])
        return single(t, starts, dt, self.signal, cover, True, prov, events)


@dataclass
class ManualFiringDetector:
    """§7.5: a stove lit by hand, detected from its combustion temperature.

    Detected when the temperature is above ``wood_threshold`` (the stove's
    own wood-mode threshold), or rises by more than ``rise`` K within
    ``window`` seconds while no start was planned (``planned_lit``, a 0/1
    sample-and-hold series of what the plan had lit). While detected the
    heat is ``wood_output`` W; after the signature ends it decays with
    time constant ``decay``. Never forecast ahead: the forecast is the
    current detection's decaying tail, nothing more.
    """

    temperature: pd.Series
    signal: str
    wood_output: float
    wood_threshold: float
    rise: float = 40.0
    window: float = 900.0
    decay: float = 3600.0
    planned_lit: pd.Series | None = None

    @property
    def signals(self) -> tuple[str, ...]:
        return (self.signal,)

    def detected(self, now) -> tuple[bool, pd.Timestamp | None]:
        """Whether a manual firing is under way at ``now``, and when its
        signature was last seen."""
        t = pd.Timestamp(now)
        s = until(self.temperature.dropna(), t)
        if s.empty:
            return False, None
        recent = s[s.index >= t - pd.Timedelta(seconds=4 * self.window)]
        hot = recent[recent > self.wood_threshold]
        last = hot.index.max() if len(hot) else None
        planned = False
        if self.planned_lit is not None:
            p = until(self.planned_lit.dropna(), t)
            planned = bool(len(p) and p.iloc[-1] > 0.5)
        if not planned and len(recent) > 1:
            w = recent[recent.index >= t - pd.Timedelta(seconds=self.window)]
            if len(w) > 1 and float(w.iloc[-1] - w.min()) > self.rise:
                last = w.index[-1] if last is None else max(last, w.index[-1])
        return (last is not None and (t - last).total_seconds() <= self.window), last

    def forecast(self, now, horizon: float, step: float) -> Forecast:
        t, starts, dt = empty(now, horizon, step)
        on, last = self.detected(t)
        vals = np.zeros(len(starts))
        if last is not None:
            a = (starts - last).total_seconds().to_numpy()
            b = a + dt
            # mean of wood_output · exp(−τ/decay) over each step, τ = time since the signature
            a0, b0 = np.maximum(a, 0), np.maximum(b, 0)
            vals = self.wood_output * self.decay * (np.exp(-a0 / self.decay) - np.exp(-b0 / self.decay)) / dt
        prov = Provenance(
            self.signal,
            "manual firing",
            "detected" if on else "decaying tail" if last is not None else "none",
            last,
        )
        return single(t, starts, dt, self.signal, vals, True, prov)
