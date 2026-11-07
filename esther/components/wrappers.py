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

Decisions, constraints and costs are added with the planner (M4). Wrappers
contribute no thermal dynamics: the physical sources they wrap
(:class:`~esther.components.sources.HeatPump`, the tank) do.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..core.component import Component, Number, Port, Quantity
from ..core.units import POWER, TEMPERATURE

__all__ = [
    "MODES",
    "CurveControlledHeatSource",
    "IVT490Plant",
    "ReplayScore",
    "ThermostatChargedTank",
    "fit_curve",
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
    ports = (Port("node", kind="node", doc="the tank node the device's sensor sits in"),)

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

    fields = (
        Number("curve_slope", min=0.0, min_exclusive=True, max=10.0),
        _delta("parallel", 0.0),
        Quantity("heater_max", dim=POWER, positive=True),
        _delta("heating_differential", 2.0, doc="jacket start below the curve target, K"),
        _delta("margin", 1.0, doc="fake tank value past a limit, K"),
        _temp("supply_max", default=273.15 + 56.0),
    )
    ports = (
        Port("tank", kind="component", component_type="JacketTank"),
        Port("radiators", kind="component", component_type="Radiators"),
        Port("heat_pump", kind="component", required=False, component_type="HeatPump"),
        Port("heater", kind="component", required=False, component_type="ElectricHeater"),
    )

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
