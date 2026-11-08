"""Controller-wrapped devices (spec §4.5.6, Appendix A).

A wrapper holds no heat of its own: it models what an installed controller
does on its own, and how a decision is translated into the one input esther
can falsify. In this version a wrapper gives

- the **translation** (a supply target into a fake outdoor temperature
  through the inverse heating curve; a tank mode into a fake tank-sensor
  value), clamped to what the device accepts;
- the **device's own logic** (its heating curve, its thermostat hysteresis,
  and for the reference plant the rule that starts its one compressor), so
  that it can be **replayed** against recorded data and scored before a
  planner relies on it (:func:`replay_score`).

From M4 a wrapper also writes the device's own behaviour into the plan
(§8.3–8.4): for the reference plant, the compressor the plan runs is run by
BOOST and kept off by BLOCK, so it stops at the tank thermostat's upper limit
and at its condensing limit; the electric heater is the controller's, not
esther's, and comes as an **expected share** of the compressor's running
learned from history (:func:`fit_aux_share`) — which is how BOOST's heater
cost is priced; and device-internal events fix the decisions they occupy to
"let the device do it". :meth:`IVT490Plant.actuate` turns a plan into the
two fake-sensor inputs, and :class:`IVT490Device` replays the device's logic
step by step for a simulated plant (the backtest's digital twin).

Wrappers contribute no thermal dynamics: the physical sources they wrap
(:class:`~esther.components.sources.HeatPump`, the tank) do.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.component import Component, Number, Port, Quantity
from ..core.units import POWER, TEMPERATURE

T0 = 273.15

__all__ = [
    "MODES",
    "CurveControlledHeatSource",
    "IVT490Device",
    "IVT490Plant",
    "ReplayScore",
    "ThermostatChargedTank",
    "fit_aux_share",
    "fit_curve",
    "fit_response",
    "fit_thermostat",
    "hysteresis",
    "replay_score",
]

#: The modes of a thermostat-charged tank (§4.5.6, A.2).
MODES = ("auto", "block", "boost")


def _temp(name: str, default: float | None = None, required: bool = False, doc: str = "") -> Quantity:
    return Quantity(
        name,
        dim=TEMPERATURE,
        absolute=True,
        required=required,
        default=default,
        min=173.15,
        max=423.15,
        doc=doc,
    )


def _delta(name: str, default: float, doc: str = "") -> Quantity:
    return Quantity(name, dim=TEMPERATURE, required=False, default=default, min=-50.0, max=50.0, doc=doc)


# -- the device logic, as plain functions ------------------------------------------


def curve_target(t_out, slope: float, factor: float, pivot: float, parallel: float):
    """Heating-curve supply target, K: ``pivot + factor·slope·(pivot − T_out) + parallel``."""
    return pivot + factor * slope * (pivot - np.asarray(t_out, dtype=float)) + parallel


def curve_inverse(t_supply, slope: float, factor: float, pivot: float, parallel: float):
    """The outdoor temperature (K) at which the curve asks for ``t_supply``."""
    g = factor * slope
    if not g > 0:
        raise ValueError("a flat heating curve cannot be inverted")
    return pivot - (np.asarray(t_supply, dtype=float) - pivot - parallel) / g


def fit_curve(t_out_seen, target, factor: float = 1.0, pivot: float = 293.15) -> dict[str, float]:
    """Least-squares heating curve from recorded pairs (the outdoor value the
    controller saw, the supply target it computed), both in K. Returns
    ``slope``, ``parallel``, ``rmse`` and ``r2``; with ``factor`` the device's
    own slope scaling, ``slope`` is in the device's units."""
    x = pivot - np.asarray(t_out_seen, dtype=float)
    y = np.asarray(target, dtype=float) - pivot
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        raise ValueError("need at least three finite pairs")
    X = np.c_[x, np.ones_like(x)]
    (b, a), *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ np.array([b, a])
    ss = ((y - y.mean()) ** 2).sum()
    return {
        "slope": float(b / factor),
        "parallel": float(a),
        "rmse": float(np.sqrt((res**2).mean())),
        "r2": float(1.0 - (res**2).sum() / ss) if ss > 0 else float("nan"),
        "n": len(x),
    }


def fake_tank_sensor(mode, real, lower, upper, margin: float = 1.0):
    """The value to show a thermostat-charged tank's controller (A.2):
    ``auto`` passes the real value; ``block`` shows ``margin`` above the upper
    (stop) limit, so the tank looks full; ``boost`` shows ``margin`` below the
    lower (start) limit while the real value is below the upper limit, so the
    controller charges, and the real value once it is not."""
    mode = np.asarray(mode)
    real = np.asarray(real, dtype=float)
    lower = np.broadcast_to(np.asarray(lower, dtype=float), real.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), real.shape)
    bad = ~np.isin(mode, MODES)
    if bad.any():
        raise ValueError(f"unknown mode(s): {sorted(set(np.asarray(mode)[bad].tolist()))}; modes are {MODES}")
    out = real.copy()
    out = np.where(mode == "block", upper + margin, out)
    out = np.where((mode == "boost") & (real < upper), lower - margin, out)
    return out


def hysteresis(value, start, stop, initial: bool = False, valid=None) -> np.ndarray:
    """A thermostat: on once ``value ≤ start``, off once ``value ≥ stop``,
    otherwise as before. Arrays are per step; ``start``/``stop`` may be
    scalars. Where ``valid`` is false the state is unknown and restarts from
    ``initial``."""
    v = np.asarray(value, dtype=float)
    lo = np.broadcast_to(np.asarray(start, dtype=float), v.shape)
    hi = np.broadcast_to(np.asarray(stop, dtype=float), v.shape)
    ok = np.ones(v.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    out = np.zeros(v.shape, dtype=bool)
    on = initial
    for k in range(len(v)):
        if not ok[k] or not np.isfinite(v[k]):
            on = initial
            out[k] = on
            continue
        if v[k] <= lo[k]:
            on = True
        elif v[k] >= hi[k]:
            on = False
        out[k] = on
    return out


@dataclass(frozen=True)
class ReplayScore:
    """How well a modelled on/off rule reproduces a recorded on-fraction."""

    n: int
    agreement: float  # share of steps where the rule and the record agree (record rounded at 0.5)
    on_recall: float  # of the steps the record is on, the share the rule has on
    off_recall: float
    start_recall: float  # of the recorded starts, the share the rule has on within ``tolerance`` steps
    n_starts: int

    def summary(self) -> str:
        return (
            f"{self.n} steps: agreement {self.agreement:.1%}, on {self.on_recall:.1%}, off "
            f"{self.off_recall:.1%}; {self.n_starts} starts, {self.start_recall:.1%} reproduced"
        )


def replay_score(predicted_on, measured_fraction, valid=None, tolerance: int = 1) -> ReplayScore:
    """Score a modelled on/off rule against a recorded on-fraction per step."""
    p = np.asarray(predicted_on, dtype=bool)
    m = np.asarray(measured_fraction, dtype=float) >= 0.5
    ok = np.isfinite(np.asarray(measured_fraction, dtype=float))
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    n = int(ok.sum())
    if n == 0:
        raise ValueError("no valid steps to score")
    agree = float((p[ok] == m[ok]).mean())
    on = float(p[ok & m].mean()) if (ok & m).any() else float("nan")
    off = float((~p[ok & ~m]).mean()) if (ok & ~m).any() else float("nan")
    starts = np.flatnonzero(ok[1:] & ok[:-1] & m[1:] & ~m[:-1]) + 1
    hit = 0
    for s in starts:
        a, b = max(0, s - tolerance), min(len(p), s + tolerance + 1)
        hit += bool(p[a:b].any())
    rec = hit / len(starts) if len(starts) else float("nan")
    return ReplayScore(n, agree, on, off, rec, len(starts))


def fit_aux_share(
    compressor, heater_share, outdoor, boost=None, cold_below: float = 281.15, valid=None
) -> dict[str, float]:
    """Learn a controller's auxiliary-heat habit from history: the heater's
    expected share per step as ``δ · (share + cold_slope · max(0,
    cold_below − T_out) + boost · [mode = boost])``, by least squares over
    the steps given (``δ`` the compressor's on-fraction). The trigger of a
    controller's own timers need not be visible in what it reports; the
    expected share is what can be learned, and re-learned, from routine
    operation."""
    d = np.asarray(compressor, dtype=float)
    h = np.asarray(heater_share, dtype=float)
    cold = np.maximum(0.0, cold_below - np.asarray(outdoor, dtype=float))
    b = np.zeros_like(d) if boost is None else np.asarray(boost, dtype=float)
    ok = np.isfinite(d) & np.isfinite(h) & np.isfinite(cold) & np.isfinite(b)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    X = np.c_[d, d * cold, d * b][ok]
    y = h[ok]
    if len(y) < 10:
        raise ValueError("need at least ten steps to learn the auxiliary share")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef = np.maximum(coef, 0.0)
    res = y - X @ coef
    return {
        "aux_share": float(coef[0]),
        "aux_cold_slope": float(coef[1]),
        "aux_boost": float(coef[2]),
        "aux_cold_below": float(cold_below),
        "rmse": float(np.sqrt(np.mean(res**2))),
        "n": len(y),
    }


def fit_thermostat(
    on,
    sensor,
    start: float | None = None,
    stop: float | None = None,
    valid=None,
    resolution: float = 0.1,
) -> dict[str, float]:
    """Learn a thermostat's **effective** start and stop limits from routine
    operation: the limits at which :func:`hysteresis` on the recorded
    ``sensor`` best reproduces the recorded on-state ``on`` (an on-fraction
    per step, on at ≥ 0.5) over the ``valid`` steps. A limit passed in is
    held; the free ones are searched on a ``resolution`` grid between the
    10th and 90th percentiles of the sensor at the recorded starts (or
    stops), the two in turn when both are free.

    What a controller reports as its limits need not be where it switches —
    another rule, a filter on its sensor, a setting not reported — and the
    effective limit is what a replay of it needs. Each run of valid steps
    starts from the recorded state, so a run carried in from outside the
    valid steps (another mode) is not scored as a miss.

    Returns ``start``, ``stop``, the replay's ``agreement``, the median
    sensor at the recorded ``starts``/``stops`` (``start_median``,
    ``stop_median``) and their counts.
    """
    d = np.asarray(on, dtype=float)
    v = np.asarray(sensor, dtype=float)
    ok = np.isfinite(d) & np.isfinite(v)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    m = d >= 0.5
    prev_ok = np.r_[False, ok[:-1]]
    prev_m = np.r_[False, m[:-1]]
    starts = ok & prev_ok & m & ~prev_m
    stops = ok & prev_ok & ~m & prev_m
    if stops.sum() < 5 or starts.sum() < 5:
        raise ValueError("need at least five recorded starts and stops to learn a thermostat's limits")
    first = ok & ~prev_ok

    def replay(lo: float, hi: float) -> float:
        state = False
        hit = 0
        for k in np.flatnonzero(ok):
            if first[k]:
                state = bool(m[k])
            elif v[k] <= lo:
                state = True
            elif v[k] >= hi:
                state = False
            hit += state == m[k]
        return hit / int(ok.sum())

    def grid(at) -> np.ndarray:
        a, b = np.quantile(v[at], [0.1, 0.9])
        return np.arange(np.floor(a / resolution), np.ceil(b / resolution) + 1) * resolution

    lo = float(np.median(v[starts])) if start is None else float(start)
    hi = float(np.median(v[stops])) if stop is None else float(stop)
    for _ in range(2 if start is None and stop is None else 1):
        if stop is None:
            cand = [c for c in grid(stops) if c > lo]
            hi = max(cand, key=lambda c: replay(lo, c)) if cand else hi
        if start is None:
            cand = [c for c in grid(starts) if c < hi]
            lo = max(cand, key=lambda c: replay(c, hi)) if cand else lo
    return {
        "start": float(lo),
        "stop": float(hi),
        "agreement": float(replay(lo, hi)),
        "start_median": float(np.median(v[starts])),
        "stop_median": float(np.median(v[stops])),
        "starts": int(starts.sum()),
        "stops": int(stops.sum()),
        "n": int(ok.sum()),
    }


def fit_response(x, y, edges) -> list[tuple[float, float]]:
    """A monotone (non-decreasing) response curve of ``y`` on ``x`` from
    binned means — a device's learned habit, such as how often a circulation
    pump runs at a given supply target. Returns ``(x, y)`` knots at the bin
    means, for :func:`numpy.interp`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    knots = []
    for a, b in itertools.pairwise(edges):
        m = (x >= a) & (x < b)
        if m.sum() >= 5:
            knots.append([float(x[m].mean()), float(y[m].mean())])
    if not knots:
        raise ValueError("no bin has enough data")
    ys = np.maximum.accumulate([k[1] for k in knots])  # pool to non-decreasing
    return [(k[0], float(v)) for k, v in zip(knots, ys, strict=True)]


# -- the components -------------------------------------------------------------


class CurveControlledHeatSource(Component):
    """A device that sets its supply target from a heating curve over an
    outdoor sensor (§4.5.6). esther's supply-temperature decision is
    translated to a **fake outdoor temperature** through the inverse curve.

    Curve: ``T_supply = pivot + factor · slope · (pivot − T_out) + parallel``,
    clamped to ``[supply_min, supply_max]``. ``factor`` is the device's own
    scaling of its slope setting (K of supply per K of outdoor per unit of
    slope). The fake value is clamped to ``[outdoor_min, outdoor_max]``, the
    range the device accepts.
    """

    fields = (
        Number("slope", min=0.0),
        Number("factor", required=False, default=1.0, min=0.0, min_exclusive=True),
        _temp("pivot", default=293.15),
        _delta("parallel", 0.0, doc="parallel offset, K"),
        _temp("supply_min", default=273.15 + 10.0),
        _temp("supply_max", default=273.15 + 60.0),
        _temp("outdoor_min", default=273.15 - 40.0),
        _temp("outdoor_max", default=273.15 + 40.0),
    )
    ports = (Port("radiators", kind="component", component_type="Radiators"),)

    def check(self) -> list[tuple[str | None, str]]:
        c = self.config
        out = []
        if not c["supply_min"] < c["supply_max"]:
            out.append(("supply_max", "supply_max must exceed supply_min"))
        if not c["outdoor_min"] < c["outdoor_max"]:
            out.append(("outdoor_max", "outdoor_max must exceed outdoor_min"))
        if c["slope"] <= 0:
            out.append(("slope", "a flat curve (slope 0) cannot be steered through the outdoor sensor"))
        return out

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        return []

    def _curve(self) -> tuple[float, float, float, float]:
        c = self.config
        return c["slope"], c["factor"], c["pivot"], c["parallel"]

    def actuate(self, plan) -> list:
        """§8.7: the planned supply temperature through the inverse curve, as
        the fake outdoor value to show the device, per step."""
        from ..actuate import Command

        col = f"{self.refs['radiators']}.supply"
        if col not in plan.decisions:
            return []
        out = []
        for k, t in enumerate(plan.index):
            end = t + pd.Timedelta(seconds=float(plan.dt[k]))
            v = float(self.fake_outdoor(plan.decisions[col].iloc[k]))
            out.append(Command.of(plan, self.name, "fake_outdoor", v, t, end))
        return out

    def target(self, t_out):
        """The supply target the device computes from the outdoor value it sees, K."""
        c = self.config
        return np.clip(curve_target(t_out, *self._curve()), c["supply_min"], c["supply_max"])

    def fake_outdoor(self, t_supply):
        """The outdoor value to show the device so that it targets ``t_supply``,
        clamped to the accepted range, K."""
        c = self.config
        return np.clip(curve_inverse(t_supply, *self._curve()), c["outdoor_min"], c["outdoor_max"])


class ThermostatChargedTank(Component):
    """A device that charges a tank between a start and a stop temperature on
    its own tank sensor (§4.5.6). esther's mode (``auto``, ``block``,
    ``boost``) is translated to a fake tank-sensor value (A.2).

    ``lower``/``upper`` are the device's start and stop limits when they are
    fixed settings; a device that reports them can pass them per step to
    :meth:`fake_sensor` and :meth:`charging` instead. ``node`` is the tank
    node the device's sensor sits in.
    """

    fields = (
        _temp("lower", required=False, doc="start limit"),
        _temp("upper", required=False, doc="stop limit"),
        _delta("margin", 1.0, doc="how far past a limit the fake value is shown"),
    )
    ports = (
        Port("node", kind="node", doc="the tank node the device's sensor sits in"),
        Port("source", kind="component", required=False, doc="the source that charges the tank"),
    )

    def actuate(self, plan) -> list:
        """§8.7: BOOST while the plan runs the charging source, BLOCK
        otherwise, as the mode the fake tank sensor encodes."""
        from ..actuate import Command

        src = self.refs.get("source")
        col = next((f"{src}.{k}" for k in ("delta", "share") if src and f"{src}.{k}" in plan.decisions), None)
        if col is None:
            return []
        out = []
        for k, t in enumerate(plan.index):
            end = t + pd.Timedelta(seconds=float(plan.dt[k]))
            m = "boost" if float(plan.decisions[col].iloc[k]) > 1e-6 else "block"
            out.append(Command.of(plan, self.name, "mode", m, t, end))
        return out

    def check(self) -> list[tuple[str | None, str]]:
        lo, hi = self.config.get("lower"), self.config.get("upper")
        if lo is not None and hi is not None and not lo < hi:
            return [("upper", "the stop limit must exceed the start limit")]
        if self.config["margin"] <= 0:
            return [("margin", "must be > 0")]
        return []

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        return []

    def _limits(self, lower, upper):
        lower = self.config.get("lower") if lower is None else lower
        upper = self.config.get("upper") if upper is None else upper
        if lower is None or upper is None:
            raise ValueError(f"{self.name!r}: no start/stop limits configured or given")
        return lower, upper

    def fake_sensor(self, mode, real, lower=None, upper=None):
        lo, hi = self._limits(lower, upper)
        return fake_tank_sensor(mode, real, lo, hi, self.config["margin"])

    def charging(self, seen, lower=None, upper=None, initial: bool = False, valid=None) -> np.ndarray:
        """The device's own charging state from the sensor value it sees."""
        lo, hi = self._limits(lower, upper)
        return hysteresis(seen, lo, hi, initial, valid)


class IVT490Plant(Component):
    """The reference plant of Appendix A: one fixed-speed compressor behind
    two controller inputs, registered as ``ivt490.Plant``.

    - Heating: a :class:`CurveControlledHeatSource` with the controller's
      curve ``T_supply = 20 + 0.16 · k · (20 − T_out) + parallel`` (A.1).
    - Hot water: a :class:`ThermostatChargedTank` on the tank's middle sensor.
    - **One compressor, two demands** (A.2, H-A3 re-stated): it runs while
      hot water is wanted (the tank sensor *as the controller sees it* fell to
      its start limit and has not reached its stop limit) **or** heating is
      wanted (the jacket fell ``heating_differential`` below the curve's
      target and has not reached the target). Each demand is steered through
      its own fake sensor, so keeping the compressor off takes both.
    - Device-internal events (anti-legionella, short daily runs) are not in
      the rule; they enter as known events (§7.6).
    """

    SLOPE_FACTOR = 0.16
    PIVOT = 293.15

    plan_order = 20
    fields = (
        Number("curve_slope", min=0.0, min_exclusive=True, max=10.0),
        _delta("parallel", 0.0),
        Quantity("heater_max", dim=POWER, positive=True),
        _delta("heating_differential", 2.0, doc="jacket start below the curve target, K"),
        _delta("margin", 1.0, doc="fake tank value past a limit, K"),
        _temp("supply_max", default=273.15 + 56.0),
        _temp("supply_min", default=273.15 + 14.0, doc="the curve's floor"),
        _temp("tank_lower", default=273.15 + 49.5, doc="hot-water start limit on the tank sensor"),
        _temp("tank_upper", default=273.15 + 54.5, doc="hot-water stop limit on the tank sensor"),
        _temp(
            "tank_stop_auto",
            required=False,
            doc="where the controller left to itself (auto) actually stops charging, if not at tank_upper "
            "(learned: fit_thermostat, or esther.backtest.fit_by_replay)",
        ),
        _temp("condensing_max", default=273.15 + 60.0, doc="the compressor stops above this jacket"),
        # the controller's own heater, learned (fit_aux_share)
        Number("aux_share", required=False, default=0.0, min=0.0, max=1.0),
        Number("aux_cold_slope", required=False, default=0.0, min=0.0, max=1.0, doc="per K"),
        _temp("aux_cold_below", default=273.15 + 8.0),
        Number("aux_boost", required=False, default=0.0, min=0.0, max=1.0),
        Number("event_heater_share", required=False, default=1.0, min=0.0, max=1.0),
    )
    ports = (
        Port("tank", kind="component", component_type="JacketTank"),
        Port("radiators", kind="component", component_type="Radiators"),
        Port("heat_pump", kind="component", required=False, component_type="HeatPump"),
        Port("heater", kind="component", required=False, component_type="ElectricHeater"),
        Port("outdoor", kind="boundary", required=False, doc="the real outdoor temperature"),
    )

    def check(self) -> list[tuple[str | None, str]]:
        c = self.config
        if not c["tank_lower"] < c["tank_upper"]:
            return [("tank_upper", "the stop limit must exceed the start limit")]
        stop = c.get("tank_stop_auto")
        if stop is not None and not c["tank_lower"] < stop:
            return [("tank_stop_auto", "the stop must lie above the start limit (tank_lower)")]
        return []

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        return []

    def _curve(self) -> tuple[float, float, float, float]:
        return self.config["curve_slope"], self.SLOPE_FACTOR, self.PIVOT, self.config["parallel"]

    def supply_target(self, t_out_seen):
        """The controller's supply (and jacket) target from the outdoor value it sees, K."""
        return np.minimum(curve_target(t_out_seen, *self._curve()), self.config["supply_max"])

    def fake_outdoor(self, t_supply, outdoor_min: float = 233.15, outdoor_max: float = 313.15):
        return np.clip(curve_inverse(t_supply, *self._curve()), outdoor_min, outdoor_max)

    def fake_tank_sensor(self, mode, real, lower, upper):
        return fake_tank_sensor(mode, real, lower, upper, self.config["margin"])

    def compressor_demand(
        self,
        tank_seen,
        tank_lower,
        tank_upper,
        jacket,
        jacket_target,
        valid=None,
    ) -> dict[str, np.ndarray]:
        """The compressor's own on/off rule per step (all temperatures in K):
        ``hot_water``, ``heating`` and ``compressor`` (either)."""
        jt = np.asarray(jacket_target, dtype=float)
        hw = hysteresis(tank_seen, tank_lower, tank_upper, valid=valid)
        ht = hysteresis(jacket, jt - self.config["heating_differential"], jt, valid=valid)
        return {"hot_water": hw, "heating": ht, "compressor": hw | ht}

    def stop_limit(self, mode: str) -> float:
        """Where hot-water charging stops on the tank sensor, K: the learned
        ``tank_stop_auto`` when the controller is left to itself, the
        reported upper limit when the mode is steered (BOOST charges to it
        through the fake sensor; BLOCK shows the tank above it)."""
        c = self.config
        if mode == "auto" and c.get("tank_stop_auto") is not None:
            return float(c["tank_stop_auto"])
        return float(c["tank_upper"])

    def aux_alpha(self, t_out, boost: bool = True) -> np.ndarray:
        """The heater's expected share per unit of compressor running."""
        c = self.config
        cold = np.maximum(0.0, c["aux_cold_below"] - np.asarray(t_out, dtype=float))
        return c["aux_share"] + c["aux_cold_slope"] * cold + (c["aux_boost"] if boost else 0.0)

    # -- planning ---------------------------------------------------------------
    def _names(self, house_components) -> tuple[str | None, str | None, str, str, str]:
        hp = self.refs.get("heat_pump")
        heater = self.refs.get("heater")
        tank = house_components[self.refs["tank"]]
        jacket, dhw = tank.nodes()[0], tank.nodes()[1]
        return hp, heater, self.refs["radiators"], jacket, dhw

    def plan_prepare(self, ctx) -> None:
        hp, heater, *_ = self._names(ctx.house.components)
        if hp:
            ctx.requests.add(f"on:{hp}")
        if heater:
            ctx.requests.add(f"claim:{heater}")

    def plan(self, ctx) -> None:
        """The plant's own rules in the plan: the compressor the plan runs
        (by BOOST) stops at the thermostat's upper limit on the tank sensor
        and at the condensing limit; the heater is the controller's expected
        share of that running; device events (anti-legionella) fix the
        compressor off and the heater on for the window they occupy; the
        supply target stays under the controller's maximum."""
        c, n = self.config, ctx.n
        hp, heater, rad, jacket, dhw = self._names(ctx.house.components)
        cover = ctx.event_cover(self.name)
        ev = cover > 0
        if hp:
            delta, on = ctx.milp[f"{hp}.delta"], ctx.milp[f"{hp}.on"]
            big = 100.0
            ctx.milp.add(
                [(ctx.state(dhw)[1:], 1.0), (on, big)],
                ub=c["tank_upper"] - T0 + big,
                name=f"{self.name} tank stop",
            )
            ctx.milp.add(
                [(ctx.state(jacket)[1:], 1.0), (on, big)],
                ub=c["condensing_max"] - T0 + big,
                name=f"{self.name} condensing limit",
            )
            if ev.any():
                ctx.milp.fix(delta[ev], 0.0)
                ctx.milp.fix(on[ev], 0.0)
        if heater:
            h = ctx.milp[f"{heater}.share"]
            out_b = self.refs.get("outdoor")
            t_out = ctx.signal(ctx.boundary_signal(out_b)) if out_b else np.full(n, c["aux_cold_below"])
            alpha = self.aux_alpha(t_out)
            e = cover * c["event_heater_share"]
            if hp:
                ctx.milp.add(
                    [(h, 1.0), (ctx.milp[f"{hp}.delta"], -alpha)], e, e, name=f"{self.name} heater share"
                )
            else:
                ctx.milp.fix(h, e)
        ua = float(ctx.values(rad).get("UA_rad", 250.0)) / 1000.0
        if ctx.has_decision(f"{rad}.Q"):
            U = ctx.decision(f"{rad}.Q")
            zone = ctx.house.components[rad].refs["zone"]
            Xz = ctx.state(zone)[:-1]
            ctx.milp.add([(U, 1.0), (Xz, ua)], ub=ua * (c["supply_max"] - T0), name=f"{self.name} supply max")
            sup = lambda x: np.clip(x[Xz] + T0 + x[U] / ua, c["supply_min"], c["supply_max"])  # noqa: E731
            ctx.add_result(f"{self.name}.supply_target", sup)
            ctx.add_result(f"{self.name}.fake_outdoor", lambda x: self.fake_outdoor(sup(x)))
        if hp:
            dl = ctx.milp[f"{hp}.delta"]

            def mode(x):
                m = np.where(x[dl] > 1e-6, MODES.index("boost"), MODES.index("block")).astype(float)
                m[ev] = MODES.index("auto")
                return m

            ctx.add_result(f"{self.name}.mode", mode)
            ctx.add_result(f"{self.name}.event", lambda x: cover)

    def actuate(self, plan) -> list:
        """The plan's first steps as the two fake-sensor commands (§8.7,
        A.2): a mode per step (BOOST for the planned on-fraction of the step,
        then BLOCK; AUTO in a device event) and the supply target, which the
        interface turns into a fake outdoor value through the inverse curve
        (:meth:`fake_outdoor` gives that value too)."""
        from ..actuate import Command

        out = []
        dec = plan.decisions
        hp = self.refs.get("heat_pump")
        mcol, scol = f"{self.name}.mode", f"{self.name}.supply_target"
        for k, t in enumerate(plan.index):
            end = t + pd.Timedelta(seconds=float(plan.dt[k]))
            if mcol in dec:
                m = MODES[int(dec[mcol].iloc[k])]
                frac = float(dec[f"{hp}.delta"].iloc[k]) if hp and m == "boost" else 1.0
                split = t + pd.Timedelta(seconds=float(plan.dt[k]) * frac)
                out.append(Command.of(plan, self.name, "mode", m, t, split if m == "boost" else end))
                if m == "boost" and split < end:
                    out.append(Command.of(plan, self.name, "mode", "block", split, end))
            if scol in dec:
                ts = float(dec[scol].iloc[k])
                out.append(Command.of(plan, self.name, "supply_target", ts, t, end))
                out.append(Command.of(plan, self.name, "fake_outdoor", float(self.fake_outdoor(ts)), t, end))
        return out


@dataclass
class IVT490Device:
    """The reference plant's controller, replayed step by step for a
    simulated plant (a backtest's digital twin, §10.2): given the commands
    (mode, supply target) and the plant's temperatures, what the controller
    does — the compressor by its two demands (hot water on the tank sensor
    as faked, heating on the jacket against the curve target) and its
    condensing limit, the heater as its learned expected share, the shunt
    mixing the supply down from the jacket, and the circulation pump by its
    learned response to the supply target.

    ``pump`` is a list of ``(supply target K, on-fraction)`` knots
    (:func:`fit_response`); without it the pump always runs.
    """

    plant: IVT490Plant
    pump: list[tuple[float, float]] | None = None
    _hw: bool = False
    _ht: bool = False
    history: list[dict] = field(default_factory=list, repr=False)

    def reset(self, hot_water: bool = False, heating: bool = False) -> None:
        self._hw, self._ht = hot_water, heating

    def step(
        self,
        mode: str,
        supply_target: float,
        tank_sensor: float,
        jacket: float,
        t_out: float,
        event: float = 0.0,
        boost_fraction: float = 1.0,
    ) -> dict[str, float]:
        """One step of the controller (temperatures K). Returns the
        compressor on-fraction, heater share, supply temperature and pump
        on-fraction — the model inputs the device produces."""
        c = self.plant.config
        lo, hi = c["tank_lower"], c["tank_upper"]
        target = float(np.clip(supply_target, c["supply_min"], c["supply_max"]))
        if event > 0:
            mode = "auto"
        seen = float(fake_tank_sensor(mode, tank_sensor, lo, hi, c["margin"]))
        if seen <= lo:
            self._hw = True
        elif seen >= self.plant.stop_limit(mode):
            self._hw = False
        diff = c["heating_differential"]
        if jacket <= target - diff:
            self._ht = True
        elif jacket >= target:
            self._ht = False
        demand = self._hw or self._ht
        comp = 1.0 if (demand and jacket < c["condensing_max"] and event <= 0) else 0.0
        if mode == "boost":
            comp *= boost_fraction
        heater = float(
            np.clip(
                comp * self.plant.aux_alpha(t_out, boost=mode == "boost") + event * c["event_heater_share"],
                0,
                1,
            )
        )
        supply = min(target, jacket)
        pump = 1.0 if self.pump is None else float(np.interp(target, *zip(*self.pump, strict=True)))
        return {
            "compressor": comp,
            "heater_share": heater,
            "supply_temperature": supply,
            "circulation_pump": pump,
        }
