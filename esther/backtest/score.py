"""Scoring a replay (§10.2): cost on realised prices, comfort, service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["degree_hours", "prediction_errors", "score", "solve_time_summary"]


def degree_hours(temp, low, high, dt: float) -> tuple[float, float]:
    """K·h below ``low`` and above ``high`` (NaN steps skipped)."""
    t = np.asarray(temp, dtype=float)
    lo = np.broadcast_to(np.asarray(low, dtype=float), t.shape)
    hi = np.broadcast_to(np.asarray(high, dtype=float), t.shape)
    ok = np.isfinite(t) & np.isfinite(lo) & np.isfinite(hi)
    h = dt / 3600.0
    below = float(np.clip(lo[ok] - t[ok], 0, None).sum() * h)
    above = float(np.clip(t[ok] - hi[ok], 0, None).sum() * h)
    return below, above


def score(
    records: pd.DataFrame,
    dt: float,
    price: pd.Series,
    power: Mapping[str, str],
    fuel: Mapping[str, tuple[str, float]] | None = None,
    comfort: tuple[str, pd.Series, pd.Series] | None = None,
    service: tuple[str, float, pd.Series | None] | None = None,
    on: Mapping[str, str] | None = None,
    mask: pd.Series | None = None,
) -> dict[str, float]:
    """A replay's (or a recorded period's) scorecard over the steps of
    ``records`` (``mask`` true, if given):

    - ``cost``: Σ price × power × Δt over ``power`` (label → column of W),
      price in currency/kWh per step; plus Σ fuel (label → (column of kg/h,
      price per kg)).
    - ``comfort``: (column K, low K, high K) → K·h below and above.
    - ``service``: (column K, minimum K, due mask or None) → K·h below the
      minimum at the due steps, and the share of due steps below it.
    - ``on``: label → column of an on-fraction → hours on and starts.
    """
    r = records if mask is None else records[mask.reindex(records.index).fillna(False).to_numpy(bool)]
    h = dt / 3600.0
    p = price.reindex(r.index).to_numpy(dtype=float)
    out: dict[str, float] = {"steps": len(r), "hours": len(r) * h}
    total = 0.0
    for label, col in power.items():
        w = r[col].to_numpy(dtype=float)
        kwh = float(np.nansum(w) / 1000.0 * h)
        cost = float(np.nansum(w / 1000.0 * h * p))
        out[f"{label} kWh"] = kwh
        out[f"{label} cost"] = cost
        total += cost
    for label, (col, unit_price) in (fuel or {}).items():
        kg = float(np.nansum(r[col].to_numpy(dtype=float)) * h)
        out[f"{label} kg"] = kg
        out[f"{label} cost"] = kg * unit_price
        total += kg * unit_price
    out["cost"] = total
    if comfort is not None:
        col, lo, hi = comfort
        b, a = degree_hours(r[col], lo.reindex(r.index), hi.reindex(r.index), dt)
        out["comfort below K·h"] = b
        out["comfort above K·h"] = a
        t = r[col].to_numpy(dtype=float)
        out["zone mean °C"] = float(np.nanmean(t) - 273.15)
        out["zone min °C"] = float(np.nanmin(t) - 273.15)
    if service is not None:
        col, tmin, due = service
        t = r[col].to_numpy(dtype=float)
        d = np.ones(len(r), bool) if due is None else due.reindex(r.index).fillna(False).to_numpy(bool)
        ok = d & np.isfinite(t)
        out["service below K·h"] = float(np.clip(tmin - t[ok], 0, None).sum() * h)
        out["service due steps"] = int(ok.sum())
        out["service share below"] = float((t[ok] < tmin).mean()) if ok.any() else float("nan")
        out["tank min °C"] = float(np.nanmin(t[ok]) - 273.15) if ok.any() else float("nan")
    for label, col in (on or {}).items():
        v = r[col].to_numpy(dtype=float)
        out[f"{label} hours"] = float(np.nansum(v) * h)
        b = v > 0.5
        out[f"{label} starts"] = int((b[1:] & ~b[:-1]).sum())
    return out


def prediction_errors(
    log: Sequence[Mapping], realised: pd.DataFrame, columns: Mapping[str, str], leads=(1, 6, 24)
) -> pd.DataFrame:
    """RMSE of each plan's predicted trajectory against what was realised
    at ``leads`` hours (§10.2). ``log``: the controller's log (each with a
    ``trajectory`` frame); ``columns``: model state → realised column."""
    rows = []
    for state, col in columns.items():
        r = realised[col]
        for lead in leads:
            err = []
            for rec in log:
                tr = rec.get("trajectory")
                if tr is None or state not in tr:
                    continue
                t = rec["time"] + pd.Timedelta(hours=lead)
                if t not in r.index or t > tr.index[-1]:
                    continue
                pred = float(np.interp(t.value, tr.index.as_unit("ns").asi8, tr[state].to_numpy()))
                err.append(pred - float(r.loc[t]))
            e = np.asarray(err)
            rows.append(
                {
                    "state": state,
                    "lead h": lead,
                    "n": len(e),
                    "rmse K": float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
                    "bias K": float(e.mean()) if len(e) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def solve_time_summary(decisions: pd.DataFrame) -> dict[str, float]:
    """The distribution of plan times (N1) and how often a limit was hit."""
    d = decisions
    s = d["seconds"].to_numpy(dtype=float) if "seconds" in d else np.zeros(0)
    st = d["status"] if "status" in d else pd.Series(dtype=str)
    out = {
        "plans": int((st != "no plan").sum()),
        "no plan": int((st == "no plan").sum()),
        "flagged (limit)": int((st == "limit").sum()),
    }
    if len(s):
        for q in (0.5, 0.9, 0.99):
            out[f"p{int(q * 100)} s"] = float(np.quantile(s, q))
        out["max s"] = float(s.max())
        out["share < 2 s"] = float((s < 2).mean())
        out["share < 10 s"] = float((s < 10).mean())
    return out
