"""Validation on held-out data and the acceptance gate (spec §6.5, F4).

**Open-loop RMSE.** In each held-out segment, windows start every ``stride``
(default 1 h) from the segment start. A window starts from the filter's
estimate at its first step — the state a planner would start from, after the
filter has seen the measurements up to then; at the segment start that is the
initial-state prior updated with one measurement — and is simulated with
measured inputs and no measurement feedback. ``rmse[h]`` is the RMSE over all
lead times ``1 … h`` of the windows that have the full ``h`` before the
segment ends; ``end_rmse[h]`` is the RMSE at lead ``h`` alone. A window
shorter than ``h`` does not count towards ``h``.

**Persistence** — the measurement at the window start, held — is evaluated on
the same windows. It is the reference a fit must beat when there is no
current fit (and the positive control on the metric: a model that cannot beat
it has learnt nothing the planner can use).

**The gate** accepts a fit only if no parameter sits on a bound and its
``horizon`` (24 h) open-loop RMSE on the zone signal is not worse than the
reference's on the same windows. A gate with no window long enough to judge
refuses: no evidence is not a pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..core.model import LinearModel
from .data import IdData
from .fit import FitResult
from .kalman import FilterOptions, kalman, open_loop, prepare

__all__ = ["GateDecision", "Validation", "acceptance_gate", "ljung_box", "validate"]

HOUR = 3600.0


@dataclass
class Validation:
    """Metrics of one parameter vector on one held-out dataset."""

    outputs: tuple[str, ...]
    step: float
    horizons: tuple[float, ...]  # s
    stride: float
    windows: list[tuple[pd.Timestamp, int]]  # (window start, segment number)
    open_loop: pd.DataFrame  # index (output, horizon h): rmse, end_rmse, bias, n_windows
    persistence: pd.DataFrame  # same layout
    innovation: pd.DataFrame  # per output: rmse, std (standardised), n
    ljung_box: pd.DataFrame  # index (output, lag h): acf, Q, p
    bias_by_hour: pd.DataFrame  # index hour; columns (output, "innovation"|"open_loop_<h>")
    span: tuple[pd.Timestamp | None, pd.Timestamp | None]
    n_steps: int
    tz: str = "UTC"
    trajectories: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def rmse(self, output: str | None = None, horizon: float = 24 * HOUR, model: bool = True) -> float:
        out = output or self.outputs[0]
        tab = self.open_loop if model else self.persistence
        key = (out, horizon / HOUR)
        return float(tab.loc[key, "rmse"]) if key in tab.index else float("nan")

    def n_windows(self, output: str | None = None, horizon: float = 24 * HOUR) -> int:
        out = output or self.outputs[0]
        key = (out, horizon / HOUR)
        return int(self.open_loop.loc[key, "n_windows"]) if key in self.open_loop.index else 0


def ljung_box(z_segments: Sequence[np.ndarray], lags: Sequence[int]) -> pd.DataFrame:
    """Ljung–Box on standardised residuals pooled over segments (lags never
    cross a segment boundary). Returns acf at each lag and Q, p up to it."""
    z = [np.asarray(s, dtype=float) for s in z_segments if len(s) > 1]
    N = sum(len(s) for s in z)
    mu = np.concatenate(z).mean() if z else 0.0
    var = sum(((s - mu) ** 2).sum() for s in z) / max(N, 1)
    maxlag = max(lags) if lags else 0
    acf = np.zeros(maxlag + 1)
    cnt = np.zeros(maxlag + 1)
    for s in z:
        d = s - mu
        for k in range(1, min(maxlag, len(s) - 1) + 1):
            acf[k] += (d[k:] * d[:-k]).sum()
            cnt[k] += len(s) - k
    with np.errstate(invalid="ignore", divide="ignore"):
        acf = acf / (N * var)
    rows = []
    for m in lags:
        ks = np.arange(1, m + 1)
        ok = cnt[ks] > 0
        Q = N * (N + 2) * np.sum(acf[ks][ok] ** 2 / (N - ks[ok]))
        rows.append({"lag": m, "acf": acf[m], "Q": Q, "p": float(stats.chi2.sf(Q, m))})
    return pd.DataFrame(rows)


def validate(
    model: LinearModel,
    theta: Sequence[float] | FitResult,
    data: IdData,
    horizons: Sequence[float] = (1 * HOUR, 6 * HOUR, 24 * HOUR),
    stride: float = 1 * HOUR,
    tz: str = "UTC",
    options: FilterOptions | None = None,
    keep_trajectories: bool = False,
) -> Validation:
    """Score ``theta`` on ``data`` (the held-out part, §6.5)."""
    if isinstance(theta, FitResult):
        opts = options or theta.options.filter
        theta = theta.theta
    else:
        opts = options or FilterOptions()
    theta = np.asarray(theta, dtype=float)
    prep = prepare(model, data, opts)
    kf = kalman(model, prep, theta[None], store=True, options=opts)
    step = data.step
    hs = [max(1, round(h / step)) for h in horizons]
    H = max(hs)
    sstride = max(1, round(stride / step))
    starts = [(s, k) for s, n in enumerate(prep.lengths) for k in range(0, n - 1, sstride)]
    ny = len(model.outputs)
    if starts:
        xs = np.stack([kf.x_filt[0, s, k] for s, k in starts])
        pred, _ = open_loop(model, prep, theta, starts, xs, H)
        seg = np.array([s for s, _ in starts])
        k0 = np.array([k for _, k in starts])
        lead = np.arange(1, H + 1)
        kk = np.minimum(k0[:, None] + lead[None], prep.Y.shape[1] - 1)
        meas = prep.Y[seg[:, None], kk]  # (W, H, ny)
        persist = np.repeat(prep.Y[seg, k0][:, None], H, axis=1)
        err = pred - meas
        perr = persist - meas
        room = prep.lengths[seg] - 1 - k0  # steps available after the start
    else:
        err = perr = np.zeros((0, H, ny))
        room = np.zeros(0, dtype=int)
        seg = k0 = np.zeros(0, dtype=int)

    def table(e: np.ndarray) -> pd.DataFrame:
        rows = []
        for o, name in enumerate(model.outputs):
            for h_s, h in zip(horizons, hs, strict=True):
                full = room >= h
                block = e[full, :h, o]
                rows.append(
                    {
                        "output": name,
                        "horizon_h": h_s / HOUR,
                        "rmse": float(np.sqrt(np.mean(block**2))) if block.size else np.nan,
                        "end_rmse": float(np.sqrt(np.mean(e[full, h - 1, o] ** 2))) if full.any() else np.nan,
                        "bias": float(np.mean(block)) if block.size else np.nan,
                        "n_windows": int(full.sum()),
                    }
                )
        return pd.DataFrame(rows).set_index(["output", "horizon_h"])

    ol, pe = table(err), table(perr)

    # innovations (skip each segment's first step: it set the initial state)
    innov_rows, lb_rows = [], []
    lag_h = [1, 6, 24]
    lags = [max(1, round(h * HOUR / step)) for h in lag_h]
    for o, name in enumerate(model.outputs):
        es, zs = [], []
        for s, n in enumerate(prep.lengths):
            e = kf.innov[0, s, 1:n, o]
            v = kf.innov_var[0, s, 1:n, o]
            es.append(e)
            zs.append(e / np.sqrt(v))
        ecat = np.concatenate(es) if es else np.zeros(0)
        zcat = np.concatenate(zs) if zs else np.zeros(0)
        innov_rows.append(
            {
                "output": name,
                "rmse": float(np.sqrt(np.mean(ecat**2))) if ecat.size else np.nan,
                "std_standardised": float(np.std(zcat)) if zcat.size else np.nan,
                "n": int(ecat.size),
            }
        )
        lb = ljung_box(zs, lags)
        lb["output"], lb["lag_h"] = name, lag_h
        lb_rows.append(lb)
    innovation = pd.DataFrame(innov_rows).set_index("output")
    lbox = pd.concat(lb_rows).set_index(["output", "lag_h"])[["lag", "acf", "Q", "p"]]

    # bias by hour of day (local to tz)
    cols = {}
    for o, name in enumerate(model.outputs):
        hours, vals = [], []
        for s, n in enumerate(prep.lengths):
            idx = prep.index[s][1:n]
            hours.append(idx.tz_convert(tz).hour.to_numpy())
            vals.append(kf.innov[0, s, 1:n, o])
        if hours:
            cols[(name, "innovation")] = pd.Series(np.concatenate(vals)).groupby(np.concatenate(hours)).mean()
        h24 = hs[-1]
        full = room >= h24
        if full.any():
            hh, ee = [], []
            for w in np.flatnonzero(full):
                s, k = seg[w], k0[w]
                t = prep.index[s][k + 1 : k + 1 + h24]
                hh.append(t.tz_convert(tz).hour.to_numpy())
                ee.append(err[w, :h24, o])
            label = f"open_loop_{horizons[-1] / HOUR:g}h"
            cols[(name, label)] = pd.Series(np.concatenate(ee)).groupby(np.concatenate(hh)).mean()
    bias = pd.DataFrame(cols).reindex(range(24))
    bias.index.name = "hour"

    windows = [(prep.index[s][k], int(s)) for s, k in starts]
    traj = {}
    if keep_trajectories:
        traj = {"err": err, "persist_err": perr, "seg": seg, "k0": k0, "room": room}
    return Validation(
        outputs=model.outputs,
        step=step,
        horizons=tuple(float(h) for h in horizons),
        stride=float(stride),
        windows=windows,
        open_loop=ol,
        persistence=pe,
        innovation=innovation,
        ljung_box=lbox,
        bias_by_hour=bias,
        span=(data.start, data.end),
        n_steps=data.n_steps,
        tz=tz,
        trajectories=traj,
    )


@dataclass
class GateDecision:
    accepted: bool
    reasons: list[str]  # why refused; empty when accepted
    output: str
    horizon: float
    candidate_rmse: float
    reference_rmse: float
    reference: str  # "current fit" | "persistence"
    warnings: list[str] = field(default_factory=list)  # reported, not gating

    def summary(self) -> str:
        verdict = "ACCEPTED" if self.accepted else "REFUSED"
        head = (
            f"{verdict}: {self.horizon / HOUR:g} h open-loop RMSE on {self.output} "
            f"{self.candidate_rmse:.3f} K vs {self.reference} {self.reference_rmse:.3f} K"
        )
        return (
            head
            + "".join(f"\n  - {r}" for r in self.reasons)
            + "".join(f"\n  - (note) {w}" for w in self.warnings)
        )


def acceptance_gate(
    fit: FitResult,
    candidate: Validation,
    current: Validation | None = None,
    output: str | None = None,
    horizon: float = 24 * HOUR,
) -> GateDecision:
    """Accept or refuse a fit (spec §6.5). ``current`` is the current fit's
    validation on the **same** held-out windows; without one, the reference
    is persistence on the candidate's windows."""
    out = output or candidate.outputs[0]
    reasons = []
    if current is not None:
        if current.windows != candidate.windows:
            raise ValueError(
                "the current fit was validated on different windows; validate both on the same data"
            )
        ref, ref_name = current.rmse(out, horizon), "current fit"
    else:
        ref, ref_name = candidate.rmse(out, horizon, model=False), "persistence"
    cand = candidate.rmse(out, horizon)
    n = candidate.n_windows(out, horizon)
    bound = [fit.names[i] for i in np.flatnonzero(fit.on_bound)]
    if bound:
        reasons.append(f"parameter(s) on a bound: {', '.join(bound)}")
    if n == 0 or not np.isfinite(cand):
        reasons.append(f"no held-out window of {horizon / HOUR:g} h: nothing to judge the fit on")
    elif not np.isfinite(ref):
        reasons.append(f"the {ref_name} has no {horizon / HOUR:g} h RMSE on these windows")
    elif cand > ref:
        reasons.append(
            f"{horizon / HOUR:g} h open-loop RMSE {cand:.4f} K is worse than the {ref_name}'s {ref:.4f} K"
        )
    warnings = []
    if not fit.hessian_ok:
        warnings.append("the Hessian at the optimum is not positive definite: the intervals are not valid")
    weak = [fit.names[i] for i in np.flatnonzero(~fit.identified)]
    if weak:
        warnings.append(f"not identified by the data (resting on the prior): {', '.join(weak)}")
    return GateDecision(
        accepted=not reasons,
        reasons=reasons,
        warnings=warnings,
        output=out,
        horizon=horizon,
        candidate_rmse=cand,
        reference_rmse=ref,
        reference=ref_name,
    )
