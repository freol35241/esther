"""Hot-water draw detection without a flow meter (spec §6.3).

A draw is a drop in a DHW node's measured temperature faster than the tank
model explains, i.e. a large negative one-step innovation of that sensor
when the model runs with **draws set to zero**:

1. run the filter with the draw input zeroed and standardise the sensor's
   innovations, ``z = e / σ`` (σ a robust scale of the innovations, see
   :class:`DrawOptions`);
2. at a step with ``z < −k`` the draw that makes the prediction meet the
   measurement is entered as an input, and the filter goes on (a
   sequential detector: one draw cannot smear into the next); flagged
   steps up to ``merge_gap`` apart are one event, whose energy is the sum
   of its inputs — ``C_dhw × ΔT`` with ``ΔT`` the drop the model did not
   explain;
3. :func:`fit_with_draws` fits the tank with the detected draws as an input,
   re-detects with the new parameters and refits — at most three passes, or
   until the events stop changing.

Without labelled draws the detector's output is **reported, not trusted**
(§6.3). :func:`validate_draws` scores it against labels and refuses with
too few of them; :func:`inject_response` adds the model's own response to
known synthetic draws to recorded data, which tests the detector (not the
model) on the house's real noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.model import Capacity, LinearModel, Signal
from .data import IdData
from .fit import FitOptions, FitResult, fit
from .kalman import FilterOptions, _x0, kalman, prepare, system

__all__ = [
    "DrawFit",
    "DrawOptions",
    "DrawValidation",
    "detect_draws",
    "draw_series",
    "fit_with_draws",
    "inject_response",
    "node_capacity",
    "validate_draws",
]

EVENT_COLUMNS = ["time", "end", "steps", "energy", "dT", "z_min", "segment", "k_first", "k_last"]


@dataclass(frozen=True)
class DrawOptions:
    """Detector settings.

    k:             threshold on the standardised innovation (§6.3: ``−k σ``)
    min_steps:     flagged steps an event needs
    merge_gap:     flagged runs this many unflagged steps apart or closer are one
                   event (default 0: only consecutive steps; two draws one quiet
                   step apart are two draws)
    min_energy:    events below this energy (J) are dropped
    exclude:       a signal in [0, 1] (e.g. a charge on-fraction); steps where it
                   exceeds ``exclude_level`` are not flagged ("not coinciding
                   with a charge")
    scale:         what ``σ`` is: ``"robust"`` (default) — the median absolute
                   deviation of the sensor's innovations over the data, × 1.4826;
                   ``"model"`` — the filter's own innovation sd. A tank fitted
                   without its draws has absorbed them into its process noise,
                   and the filter's sd is then inflated exactly where the draws
                   are; the robust scale is not, because draws are rare.
    """

    k: float = 4.0
    min_steps: int = 1
    merge_gap: int = 0
    min_energy: float = 0.0
    exclude: str | None = None
    exclude_level: float = 0.5
    scale: str = "robust"
    filter: FilterOptions = field(default_factory=FilterOptions)


def node_capacity(model: LinearModel, theta, node: str) -> float:
    """Heat capacity (J/K) of ``node`` at ``theta``."""
    for _, t in model.terms(theta):
        if isinstance(t, Capacity) and t.node == node:
            return float(t.C)
    raise KeyError(f"no node {node!r} in the model")


def _zeroed(data: IdData, column: str | None) -> IdData:
    if column is None:
        return data
    return IdData(data.step, [f.assign(**{column: 0.0}) if column in f.columns else f for f in data.frames])


def _with_column(data: IdData, column: str, values: pd.Series) -> IdData:
    return IdData(
        data.step,
        [
            f.assign(**{column: values.reindex(f.index).fillna(0.0).to_numpy(dtype=float)})
            for f in data.frames
        ],
    )


def detect_draws(
    model: LinearModel,
    data: IdData,
    theta,
    signal: str,
    draw_input: str,
    options: DrawOptions | None = None,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect draws on the measured DHW signal ``signal``.

    The filter runs with the draw input ``draw_input`` (a column of the data,
    read by a :class:`~esther.components.loads.HotWaterDraw` on the DHW node)
    set to zero. It runs **sequentially**: at a step whose standardised
    innovation on ``signal`` is below ``−k``, the draw over the previous step
    that makes the prediction meet the measurement is entered as an input
    before the update, so the filter's state takes the drop at once instead
    of catching up over many steps (which would merge a morning's draws into
    one). Consecutive flagged steps, or steps up to ``merge_gap`` apart, are
    one event.

    ``extra`` holds columns the data need not carry (the ``exclude`` signal),
    on the same index. Returns one row per event: ``time`` (the step it was
    first seen in), ``end``, ``steps``, ``energy`` (J, positive), ``dT``
    (energy over the node's capacity, K, negative), ``z_min``, and where it
    lies in ``data`` (``segment``, ``k_first``, ``k_last``: the measured steps
    that were flagged; the heat was drawn over the step before each).
    """
    opts = options or DrawOptions()
    if signal not in model.outputs:
        raise KeyError(f"{signal!r} is not a measured output of the model")
    if Signal(draw_input) not in model.inputs:
        raise KeyError(f"the model does not read {draw_input!r}; add a HotWaterDraw with that signal")
    theta = np.asarray(theta, dtype=float)
    d0 = _zeroed(data, draw_input)
    if any(draw_input not in f.columns for f in d0.frames):
        d0 = _with_column(data, draw_input, pd.Series(dtype=float))
    prep = prepare(model, d0, opts.filter)
    sysm = system(model, prep, theta[None])
    Ad, Gd, Qd, c, R = sysm.Ad[0], sysm.Gd[0], sysm.Qd[0], sysm.c[0], sysm.R[0]
    j = model.inputs.index(Signal(draw_input))
    o = model.outputs.index(signal)
    on = prep.obs_node
    node = model.states[int(on[o])]
    C = node_capacity(model, theta, node)
    x0s, P0s = _x0(prep, c[None], model.n_states, opts.filter)
    if opts.scale not in ("robust", "model"):
        raise ValueError(f"scale must be 'robust' or 'model', got {opts.scale!r}")
    sigma = None
    if opts.scale == "robust":
        kf = kalman(model, prep, theta[None], store=True, options=opts.filter)
        e_all = kf.innov[0, :, 1:, o][prep.mask[:, 1:]]
        e_all = e_all[np.isfinite(e_all)]
        sigma = 1.4826 * float(np.median(np.abs(e_all - np.median(e_all))))
        if not sigma > 0:
            raise ValueError(f"the innovations of {signal!r} have no spread; cannot set a threshold")
    rows = []
    for s, n in enumerate(prep.lengths):
        x, P = x0s[0, s].copy(), P0s[0, s].copy()
        u = np.zeros(n)
        zs = np.zeros(n)
        ex = None
        if opts.exclude:
            src = extra if extra is not None else data.frames[s]
            ex = src[opts.exclude].reindex(prep.index[s][:n]).to_numpy(dtype=float)
        for k in range(n):
            if k:
                cb = prep.combo[s, k - 1]
                x = Ad[cb] @ x + Gd[cb] @ prep.D[s, k - 1]
                P = Ad[cb] @ P @ Ad[cb].T + Qd[cb]
            e = prep.Y[s, k] - x[on] - c
            S = P[np.ix_(on, on)] + R
            z = e[o] / (sigma if sigma is not None else np.sqrt(S[o, o]))
            zs[k] = z if np.isfinite(z) else 0.0
            excluded = ex is not None and np.nan_to_num(ex[k], nan=1.0) > opts.exclude_level
            if k > 0 and np.isfinite(z) and z < -opts.k and not excluded:
                g = Gd[cb][on[o], j]  # effect of 1 W of draw over the last step on the measured node
                if g < 0:
                    du = e[o] / g
                    x = x + Gd[cb][:, j] * du
                    u[k - 1] = du
                    e = prep.Y[s, k] - x[on] - c
            ok = np.isfinite(e)  # missing measurements are not used
            if ok.any():
                oo = on[ok]
                Sk = P[np.ix_(oo, oo)] + R[np.ix_(ok, ok)]
                Kg = np.linalg.solve(Sk, P[oo, :]).T
                x = x + Kg @ e[ok]
                P = P - Kg @ P[oo, :]
                P = 0.5 * (P + P.T)
        idx = np.flatnonzero(u > 0) + 1  # the measured steps that showed the drops
        if not len(idx):
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > opts.merge_gap + 1) + 1)
        idx_s = prep.index[s]
        for r in runs:
            if len(r) < opts.min_steps:
                continue
            a, b = int(r[0]), int(r[-1])
            energy = float(u[a - 1 : b].sum() * data.step)
            if energy < opts.min_energy:
                continue
            rows.append(
                {
                    "time": idx_s[a],
                    "end": idx_s[b] + pd.Timedelta(seconds=data.step),
                    "steps": b - a + 1,
                    "energy": energy,
                    "dT": -energy / C,
                    "z_min": float(zs[a : b + 1].min()),
                    "segment": s,
                    "k_first": a,
                    "k_last": b,
                }
            )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def draw_series(events: pd.DataFrame, data: IdData) -> pd.Series:
    """The detected draws as a heat-rate input (W) on ``data``'s grid.

    A drop first seen in the measurement of step ``a`` happened over step
    ``a − 1`` (the input of step ``k`` drives ``x_k → x_{k+1}``), so an event
    flagged on steps ``a … b`` is spread evenly over steps ``a − 1 … b − 1``.
    """
    idx = pd.DatetimeIndex(np.concatenate([f.index.as_unit("ns").asi8 for f in data.frames]), tz="UTC")
    out = pd.Series(0.0, index=idx)
    for ev in events.itertuples():
        f = data.frames[ev.segment]
        a, b = max(0, ev.k_first - 1), max(0, ev.k_last - 1)
        steps = f.index[a : b + 1]
        out.loc[steps] += ev.energy / (len(steps) * data.step)
    return out


@dataclass
class DrawFit:
    """The outer loop's result: the final fit, the events it was fitted with,
    and per pass the number of events, their energy and the objective."""

    result: FitResult
    events: pd.DataFrame
    passes: pd.DataFrame
    data: IdData  # the data with the final draw input


def fit_with_draws(
    model: LinearModel,
    data: IdData,
    signal: str,
    draw_input: str,
    passes: int = 3,
    fit_options: FitOptions | None = None,
    options: DrawOptions | None = None,
    extra: pd.DataFrame | None = None,
) -> DrawFit:
    """Fit the tank with draws detected in an outer loop (§6.3 step 3): fit
    with no draws, detect, fit with the detected draws, re-detect with the new
    parameters, … at most ``passes`` fits, or until the events stop changing.
    The model must read ``draw_input`` (a :class:`HotWaterDraw` on the DHW
    node with that signal)."""
    if Signal(draw_input) not in model.inputs:
        raise KeyError(f"the model does not read {draw_input!r}; add a HotWaterDraw with that signal")
    if not 1 <= passes <= 3:
        raise ValueError("passes must be 1, 2 or 3 (§6.3)")
    cur = _zeroed(data, draw_input)
    if any(draw_input not in f.columns for f in data.frames):
        cur = _with_column(data, draw_input, pd.Series(dtype=float))
    rows = []
    events = pd.DataFrame(columns=EVENT_COLUMNS)
    res = None
    start = None
    for p in range(passes):
        res = fit(model, cur, fit_options, start=start)
        start = res.theta
        new = detect_draws(model, cur, res.theta, signal, draw_input, options, extra)
        rows.append(
            {
                "pass": p + 1,
                "events_in": len(events),
                "energy_in_kWh": float(events["energy"].sum()) / 3.6e6 if len(events) else 0.0,
                "objective": res.objective,
                "events_found": len(new),
            }
        )
        same = len(new) == len(events) and (
            len(new) == 0 or (new["time"].to_numpy() == events["time"].to_numpy()).all()
        )
        if same and p > 0:
            break
        events = new
        if p + 1 < passes:
            cur = _with_column(data, draw_input, draw_series(events, data))
    assert res is not None
    return DrawFit(result=res, events=events, passes=pd.DataFrame(rows), data=cur)


def inject_response(
    model: LinearModel,
    theta,
    data: IdData,
    input_name: str,
    series: pd.Series,
    filter_options: FilterOptions | None = None,
) -> IdData:
    """Add to ``data``'s measured outputs the model's response to ``series``
    on input ``input_name`` (the model is linear, so the response superposes
    exactly). With ``series`` a set of known synthetic draws (negative W),
    the result is the house's real data as it would read had those draws
    happened — a test of the detector on real noise and real unmodelled
    effects, with known answers. Each segment starts from zero response."""
    if Signal(input_name) not in model.inputs:
        raise KeyError(f"the model does not read {input_name!r}")
    j = model.inputs.index(Signal(input_name))
    theta = np.asarray(theta, dtype=float)
    d0 = _with_column(data, input_name, pd.Series(dtype=float))
    prep = prepare(model, d0, filter_options)
    sysm = system(model, prep, theta[None])
    Ad, Gd = sysm.Ad[0], sysm.Gd[0]
    out = []
    for s, f in enumerate(data.frames):
        u = series.reindex(f.index).fillna(0.0).to_numpy(dtype=float)
        x = np.zeros(model.n_states)
        resp = np.zeros((len(f), model.n_states))
        for k in range(len(f)):
            resp[k] = x
            c = prep.combo[s, k]
            x = Ad[c] @ x + Gd[c][:, j] * u[k]
        g = f.copy()
        for o, sig in enumerate(model.outputs):
            g[sig] = g[sig].to_numpy(dtype=float) + resp[:, prep.obs_node[o]]
        out.append(g)
    return IdData(data.step, out)


@dataclass
class DrawValidation:
    """Detected events against labelled draws (§6.3)."""

    n_labels: int
    hits: int
    misses: pd.DataFrame
    matches: pd.DataFrame  # label time, label energy, event time, event energy, ratio
    false_alarms: int  # events within the labelled period that match no label
    passed: bool
    reason: str

    def summary(self) -> str:
        verdict = "PASSED" if self.passed else "NOT PASSED"
        return (
            f"{verdict}: {self.hits}/{self.n_labels} labelled draws found, {self.false_alarms} unlabelled "
            f"events in the labelled period; {self.reason}"
        )


def validate_draws(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    tolerance: str | pd.Timedelta = "20min",
    energy_factor: float = 10**0.5,
    min_labels: int = 3,
    period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> DrawValidation:
    """Score detected ``events`` against ``labels`` (columns ``time`` and,
    optionally, ``energy`` in J).

    The detector passes (§6.3) when there are at least ``min_labels`` labels,
    **every** label has an event within ``tolerance`` and, where a label
    gives an energy, the event's energy is within ``energy_factor`` of it
    (default √10: the same order of magnitude). False alarms are counted over
    ``period`` (default: the labels' span ± tolerance) and reported, not
    gated: unlabelled draws are expected.
    """
    tol = pd.Timedelta(tolerance)
    lab = labels.copy()
    if "energy" not in lab.columns:
        lab["energy"] = np.nan
    lab = lab.sort_values("time").reset_index(drop=True)
    ev = events.sort_values("time").reset_index(drop=True)
    used: set[int] = set()
    rows, misses = [], []
    for row in lab.itertuples():
        best, gap = None, None
        for i, e in ev.iterrows():
            if i in used:
                continue
            d = abs(e["time"] - row.time)
            if d <= tol and (gap is None or d < gap):
                best, gap = i, d
        if best is None:
            misses.append({"time": row.time, "energy": row.energy})
            continue
        used.add(best)
        e_ev = float(ev.loc[best, "energy"])
        ratio = e_ev / row.energy if np.isfinite(row.energy) and row.energy > 0 else np.nan
        rows.append(
            {"label_time": row.time, "label_energy": row.energy, "event_time": ev.loc[best, "time"],
             "event_energy": e_ev, "ratio": ratio}
        )  # fmt: skip
    matches = pd.DataFrame(
        rows, columns=["label_time", "label_energy", "event_time", "event_energy", "ratio"]
    )
    if period is None and len(lab):
        period = (lab["time"].min() - tol, lab["time"].max() + tol)
    fa = 0
    if period is not None and len(ev):
        inside = (ev["time"] >= period[0]) & (ev["time"] <= period[1])
        fa = int(sum(1 for i in np.flatnonzero(inside.to_numpy()) if i not in used))
    n = len(lab)
    ratios = matches["ratio"].dropna()
    bad_energy = ratios[(ratios > energy_factor) | (ratios < 1.0 / energy_factor)]
    if n < min_labels:
        passed, reason = False, f"too few labels ({n} < {min_labels}): the detector is reported, not trusted"
    elif misses:
        passed, reason = False, f"{len(misses)} labelled draw(s) not found within {tol}"
    elif len(bad_energy):
        passed, reason = False, f"{len(bad_energy)} energy estimate(s) off by more than ×{energy_factor:.2g}"
    else:
        passed, reason = (
            True,
            f"every label found; energies within ×{energy_factor:.2g} ({len(ratios)} with energy)",
        )
    return DrawValidation(
        n_labels=n,
        hits=len(matches),
        misses=pd.DataFrame(misses, columns=["time", "energy"]),
        matches=matches,
        false_alarms=fa,
        passed=passed,
        reason=reason,
    )
