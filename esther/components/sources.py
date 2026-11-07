"""Heat sources (spec §4.5.4).

In **identification** a source enters as a measured heat input (§6.4): the
integration supplies what the device reports — a compressor's on-fraction, a
heater's share of its maximum, a stove's estimated heat — and the source turns
it into watts through a fitted parameter whose prior comes from the datasheet.
In **planning** each source is a heat-rate decision into the node it heats,
tied by its :meth:`~esther.core.component.Component.plan` contribution to
what the device can do: a heat pump's on-fraction (and on/off indicators
with starts where a minimum run, a start cost or a sink limit asks for
them), a heater's share or steps, a stove's levels with ignition, minimum
burn and cool-down (§8.2–8.3). Each registers its electricity or fuel, which
the tariff prices, and its marginal cost of heat, which the terminal value
uses (§8.6).

The size of a source in watts is what fixes a model's absolute scale (§6.2).
An electric heater's heat is its electrical power, which can be measured, so
it is the natural anchor: give its ``Q`` a tight prior (or pin it). A heat
pump's heat output and a stove's delivered heat are then fitted against it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from ..core.component import Boolean, Component, Integer, Number, Port, Quantity, String, TableList
from ..core.model import Decision, Gain, ModelError, Signal, Transfer
from ..core.params import LogNormal, Param
from ..core.units import CONDUCTANCE, CURRENCY, DIMENSIONLESS, MASS_FLOW, POWER, TEMPERATURE, TIME

T0 = 273.15


def _steps(seconds: float, step: float) -> int:
    """A duration in whole planning steps (rounded up)."""
    return int(np.ceil(seconds / step - 1e-9)) if seconds > 0 else 0


def _history(dev: Mapping[str, Any], n_hist: int, step: float) -> tuple[np.ndarray, np.ndarray, int]:
    """On/off and start history before the horizon from a device's state at
    plan time: ``on`` (bool), ``on_for`` / ``off_for`` (s, how long in the
    current state). Most recent step first."""
    on = bool(dev.get("on", False))
    on_hist = np.zeros(n_hist)
    start_hist = np.zeros(n_hist)
    if on:
        k = max(1, _steps(float(dev.get("on_for", 1e9)), step))
        on_hist[: min(k, n_hist)] = 1.0
        if k <= n_hist:
            start_hist[k - 1] = 1.0
    else:
        k = _steps(float(dev.get("off_for", 1e9)), step)
        if k < n_hist:
            on_hist[k:] = 1.0  # it was burning before it stopped
    return on_hist, start_hist, (1 if on else 0)


def _on_off(ctx, name: str, delta, min_fraction: float, min_on: float, min_off: float, start_cost: float):
    """On/off indicators for an on-fraction ``delta``, with starts, a start
    cost and minimum run and off times (§8.3). Returns (on, start)."""
    from ..plan.blocks import Linear, lag, window

    n = ctx.n
    step = ctx.house.step
    n_on, n_off = _steps(min_on, step), _steps(min_off, step)
    H = max(n_on, n_off, 1) + 1
    on_h, st_h, _ = _history(ctx.devices.get(name, {}), H, step)
    on = ctx.milp.var(f"{name}.on", n, 0, 1, integer=True)
    st = ctx.milp.var(f"{name}.start", n, 0, 1)
    ctx.milp.add([(delta, 1.0), (on, -1.0)], ub=0.0, name=f"{name} on-fraction within on")
    if min_fraction > 0:
        ctx.milp.add([(delta, 1.0), (on, -min_fraction)], lb=0.0, name=f"{name} minimum on-fraction")
    t, lb, ub = (Linear.of(st) - Linear.of(on) + lag(on, 1, on_h)).rows(lb=0.0)
    ctx.milp.add(t, lb, ub, name=f"{name} start")
    if n_on > 1:
        t, lb, ub = (window(st, n_on, st_h) - Linear.of(on)).rows(ub=0.0)
        ctx.milp.add(t, lb, ub, name=f"{name} minimum run")
    if n_off > 0:
        t, lb, ub = (window(st, n_off, st_h) + lag(on, n_off, on_h)).rows(ub=1.0)
        ctx.milp.add(t, lb, ub, name=f"{name} minimum off")
    if start_cost > 0:
        ctx.add_cost(f"{name} starts", Linear.of(st, start_cost))
    ctx.hold(on, name=f"{name} held")
    return on, st


def _prior_from_field(comp: Component, pname: str, median: float, gsd: float) -> None:
    """Centre ``pname``'s default prior on a configured value, unless the
    house gave a prior of its own."""
    declared = next(p for p in type(comp).params if p.name == pname)
    if comp.param[pname] is declared:
        lo, hi = declared.bounds
        comp.param[pname] = replace(
            declared, prior=LogNormal(median, gsd), bounds=(min(lo, median / 100), hi)
        )


class HeatPump(Component):
    """Generic heat pump; a fixed-speed machine as an on-fraction of its heat output.

    *Identification:* ``on_signal`` is the compressor's on-fraction per step.
    The heat into ``sink`` is ``δ · Q`` with ``Q`` fitted (prior centred on
    ``Q_rated``). With ``sink_reference`` set, the output falls with the sink
    temperature: ``δ · (Q + dQ_dT · (T_ref − T_sink))`` — still linear in the
    states (a transfer from a fixed temperature, modulated by ``δ``). With
    ``P_el`` (the measured or rated electrical input) the fit's ``Q`` gives
    the COP directly, ``Q / P_el``; :meth:`cop` returns it.

    *Planning:* a heat-rate decision into ``sink``; the electrical power is
    ``Q / COP`` with ``COP = η T_sink / (T_sink − T_src)`` (§4.3: precomputed
    per step).

    ``source`` is where the heat comes from (outdoor air, extract air, ground).
    It is used for the COP, not as a heat path: an exhaust-air machine takes
    its heat from air that is leaving the house anyway.
    """

    fields = (
        Quantity("Q_rated", dim=POWER, positive=True, doc="rated heat output"),
        Quantity("P_el", dim=POWER, required=False, positive=True, doc="electrical input while running"),
        String("on_signal", required=False, doc="compressor on-fraction per step, 0..1"),
        Quantity(
            "sink_reference",
            dim=TEMPERATURE,
            absolute=True,
            required=False,
            min=273.15,
            max=373.15,
            doc="sink temperature at which Q applies; enables dQ_dT",
        ),
        Quantity("min_on", dim=TIME, required=False, default=0.0, min=0.0),
        Quantity("min_off", dim=TIME, required=False, default=0.0, min=0.0),
        Number(
            "min_fraction", required=False, default=0.0, min=0.0, max=1.0, doc="lowest on-fraction when on"
        ),
        Quantity("start_cost", dim=CURRENCY, required=False, default=0.0, min=0.0, doc="per start (wear)"),
        Quantity(
            "sink_max",
            dim=TEMPERATURE,
            absolute=True,
            required=False,
            min=273.15,
            max=373.15,
            doc="the sink cannot be lifted above this while it runs (a condensing limit)",
        ),
    )
    plan_order = 10
    ports = (
        Port("sink", kind="node", heats=True),
        Port("source", kind="thermal", doc="where the heat comes from: ambient, extract air, ground"),
    )
    params = (
        Param("Q", POWER, LogNormal(2000.0, 1.5), (10.0, 1e6), description="heat output while running"),
        Param(
            "dQ_dT", CONDUCTANCE, LogNormal(20.0, 3.0), (0.01, 1e4), description="output lost per K of sink"
        ),
        Param("eta", DIMENSIONLESS, LogNormal(0.45, 1.3), (0.1, 0.9), description="Carnot fraction"),
    )

    def __init__(
        self, name: str, config: Mapping[str, Any], refs: Mapping[str, Any], params: Mapping
    ) -> None:
        super().__init__(name, config, refs, params)
        _prior_from_field(self, "Q", self.config["Q_rated"], 1.5)

    def signals(self) -> list[str]:
        return [self.config["on_signal"]] if self.config.get("on_signal") else []

    def check(self) -> list[tuple[str | None, str]]:
        if self.refs["sink"] == self.refs["source"]:
            return [("source", "a heat pump's source and sink must differ")]
        return []

    def active_params(self, mode: str) -> list[str]:
        if mode == "plan":
            return ["eta"]
        return ["Q", "dQ_dT"] if self.config.get("sink_reference") is not None else ["Q"]

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        sink = self.refs["sink"]
        if mode == "plan":
            return [Decision(f"{self.name}.Q", sink)]
        on = self.config.get("on_signal")
        if not on:
            raise ModelError(
                f"{self.name!r}: identification needs the heat pump's on_signal (its heat enters as a "
                "measured input, §6.4)"
            )
        terms: list = [Gain(sink, Signal(on), p["Q"])]
        ref = self.config.get("sink_reference")
        if ref is not None:
            terms.append(Transfer(p["dQ_dT"], float(ref), sink, to=sink, modulation=on))
        return terms

    def heat_output(self, p: Mapping[str, float], t_sink: float) -> float:
        """Heat output while running, W, at sink temperature ``t_sink`` (K)."""
        ref = self.config.get("sink_reference")
        return p["Q"] if ref is None else p["Q"] + p["dQ_dT"] * (ref - t_sink)

    def cop(self, p: Mapping[str, float], t_sink: float | None = None) -> float | None:
        """``Q / P_el`` from the fitted heat output; None without ``P_el``."""
        P = self.config.get("P_el")
        if not P:
            return None
        q = p["Q"] if t_sink is None else self.heat_output(p, t_sink)
        return q / P

    def carnot_cop(self, p: Mapping[str, float], t_sink, t_src):
        """``η T_sink / (T_sink − T_src)``, temperatures in K (planning, §4.3)."""
        lift = np.maximum(np.asarray(t_sink, dtype=float) - t_src, 1.0)
        return p["eta"] * np.asarray(t_sink, dtype=float) / lift

    def plan(self, ctx) -> None:
        """On-fraction ``δ`` per step with heat ``q_k δ`` and electricity
        ``P_k δ``, where ``q_k`` (and the COP) are evaluated on the predicted
        sink temperature (§8.5). With ``P_el`` the electrical input is that
        constant while running and the heat output carries the sink
        dependence (a fixed-speed compressor); without it, ``P = q / COP``
        with the Carnot-fraction COP over the source temperature."""
        from ..plan.blocks import Linear

        name, n, c = self.name, ctx.n, self.config
        p = ctx.values(name)
        U = ctx.decision(f"{name}.Q")
        delta = ctx.milp.var(f"{name}.delta", n, 0.0, 1.0)
        sink = self.refs["sink"]
        fitted = "Q" in p and (c.get("sink_reference") is None or "dQ_dT" in p)

        def q_of(T):
            q = self.heat_output(p, T) if fitted else c["Q_rated"]
            return np.maximum(np.broadcast_to(np.asarray(q, dtype=float), np.shape(T)), 0.0)

        q = ctx.add_coefficient(f"{name}.Q", sink, q_of)
        ctx.milp.add([(U, 1.0), (delta, -q / 1000.0)], 0.0, 0.0, name=f"{name} heat")
        if c.get("P_el"):
            p_el = np.full(n, float(c["P_el"]))
        else:
            src = self.refs["source"]
            t_src = ctx.lin_point(src) if ctx.has_state(src) else ctx.signal(ctx.boundary_signal(src))
            cop_fn = lambda T: q_of(T) / np.maximum(self.carnot_cop(p, T, t_src), 1.0)  # noqa: E731
            p_el = ctx.add_coefficient(f"{name}.P_el", sink, cop_fn)
        ctx.add_electric(name, Linear.of(delta, p_el / 1000.0))
        price = ctx.shared.get("electricity_price")
        if price is not None:
            ctx.add_heat_cost(price * p_el / np.maximum(q, 1.0))
        needs = (
            f"on:{name}" in ctx.requests
            or c.get("min_on", 0) > 0
            or c.get("min_off", 0) > 0
            or c.get("start_cost", 0) > 0
            or c.get("sink_max") is not None
        )
        if needs:
            on, _ = _on_off(
                ctx,
                name,
                delta,
                c.get("min_fraction", 0.0),
                c.get("min_on", 0.0),
                c.get("min_off", 0.0),
                c.get("start_cost", 0.0),
            )
            if c.get("sink_max") is not None:
                big = 100.0
                X = ctx.state(sink)
                ctx.milp.add(
                    [(X[1:], 1.0), (on, big)], ub=c["sink_max"] - T0 + big, name=f"{name} sink limit"
                )
            ctx.add_result(f"{name}.on", lambda x, on=on: x[on])
        pin = ctx.pinned(f"{name}.delta")
        if pin is not None:
            free = np.isnan(pin)
            if (~free).any():
                ctx.milp.fix(delta[~free], pin[~free])
        ctx.add_result(f"{name}.delta", lambda x: x[delta])
        ctx.add_result(f"{name}.heat_output", lambda x: q)


class ElectricHeater(Component):
    """Immersion or flow heater, COP 1: continuous (``steps = 0``) or ``steps`` equal steps.

    *Identification:* ``on_signal`` is the heater's output as a share of
    ``max`` per step (an on-fraction for a one-step heater). The heat is
    ``share · Q``; ``Q``'s prior is centred on ``max`` and tight (gsd 1.1):
    electrical power is heat, and a measured heater is the model's best scale
    anchor (§6.2). Pin it when it has been measured.
    """

    fields = (
        Quantity("max", dim=POWER, positive=True),
        Integer("steps", required=False, default=0, min=0, max=32),
        String("on_signal", required=False, doc="output as a share of max per step, 0..1"),
        Boolean(
            "commandable",
            required=False,
            default=True,
            doc="false: a device's own controller engages it (a wrapper models when)",
        ),
    )
    plan_order = 10
    ports = (Port("node", kind="node", heats=True),)
    params = (Param("Q", POWER, LogNormal(3000.0, 1.1), (10.0, 1e6), description="heat at full output"),)

    def __init__(
        self, name: str, config: Mapping[str, Any], refs: Mapping[str, Any], params: Mapping
    ) -> None:
        super().__init__(name, config, refs, params)
        _prior_from_field(self, "Q", self.config["max"], 1.1)

    def signals(self) -> list[str]:
        return [self.config["on_signal"]] if self.config.get("on_signal") else []

    def active_params(self, mode: str) -> list[str]:
        return [] if mode == "plan" else ["Q"]

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        node = self.refs["node"]
        if mode == "plan":
            return [Decision(f"{self.name}.Q", node)]
        on = self.config.get("on_signal")
        if not on:
            raise ModelError(f"{self.name!r}: identification needs the heater's on_signal (§6.4)")
        return [Gain(node, Signal(on), p["Q"])]

    def plan(self, ctx) -> None:
        """Share ``h`` of full output per step: continuous, or in ``steps``
        equal steps. Not commandable, its share is the device's: a wrapper
        that claims it (``claim:<name>``) writes the rule; unclaimed, it is
        planned off."""
        from ..plan.blocks import Linear

        name, n, c = self.name, ctx.n, self.config
        q_max = float(ctx.values(name).get("Q", c["max"]))
        U = ctx.decision(f"{name}.Q")
        if c.get("steps") and c.get("commandable", True):
            k = ctx.milp.var(f"{name}.level", n, 0, c["steps"], integer=True)
            h = ctx.milp.var(f"{name}.share", n, 0.0, 1.0)
            ctx.milp.add([(h, 1.0), (k, -1.0 / c["steps"])], 0.0, 0.0, name=f"{name} steps")
            ctx.hold(k, name=f"{name} held")
        else:
            h = ctx.milp.var(f"{name}.share", n, 0.0, 1.0)
        if not c.get("commandable", True) and f"claim:{name}" not in ctx.requests:
            ctx.milp.fix(h, 0.0)
        ctx.milp.add([(U, 1.0), (h, -q_max / 1000.0)], 0.0, 0.0, name=f"{name} heat")
        ctx.add_electric(name, Linear.of(h, q_max / 1000.0))
        price = ctx.shared.get("electricity_price")
        if price is not None and c.get("commandable", True):
            ctx.add_heat_cost(price)
        pin = ctx.pinned(f"{name}.share")
        if pin is not None and (~np.isnan(pin)).any():
            ctx.milp.fix(h[~np.isnan(pin)], pin[~np.isnan(pin)])
        ctx.add_result(f"{name}.share", lambda x: x[h])


class Stove(Component):
    """Automatic-feed stove heating a zone's air; integer level per step.

    *Identification:* ``heat_signal`` is the integration's estimate of the
    stove's heat output in W (its nominal output per level while burning, or
    a proxy such as a flue temperature mapped through the stove's own curve).
    The heat reaching the model is ``gain · signal``, split between ``zone``
    and an optional ``mass`` node by ``f_rad``. ``gain`` is the **delivered
    fraction**: below 1 when the nominal figure overstates the heat, or part
    of it goes where the model has no node.

    *Planning:* a heat-rate decision into ``zone``; the levels, ignition ramp
    ``t_ign``, minimum burn and cool-down times arrive with the planner (M4).
    Wood burnt by hand is not planned (§7.5).
    """

    fields = (
        TableList(
            "levels",
            item_fields=(
                Quantity("heat", dim=POWER, positive=True),
                Quantity("fuel", dim=MASS_FLOW, positive=True),
            ),
        ),
        Number("efficiency", min=0.0, min_exclusive=True, max=1.0),
        String("fuel"),
        String("heat_signal", required=False, doc="estimated heat output, W (identification)"),
        Quantity("min_on", dim=TIME, required=False, default=0.0, min=0.0),
        Quantity("min_off", dim=TIME, required=False, default=0.0, min=0.0, doc="t_cool"),
        Quantity("start_cost", dim=CURRENCY, required=False, default=0.0, min=0.0, doc="per start"),
        Quantity(
            "block",
            dim=TIME,
            required=False,
            default=0.0,
            min=0.0,
            doc="levels held over clock blocks this long",
        ),
        Boolean(
            "modulating",
            required=False,
            default=False,
            doc="plan the output as continuous between the lowest and highest level (one binary a step)",
        ),
        # Stove.allowed (§8.4): someone forecast home, and never outside the allowed hours
        String("occupancy_signal", required=False, doc="forecast probability someone is home"),
        Number("min_occupancy", required=False, default=0.5, min=0.0, max=1.0),
        Integer("allowed_from", required=False, default=0, min=0, max=24, doc="local hour"),
        Integer("allowed_to", required=False, default=24, min=0, max=24, doc="local hour"),
        String("timezone", required=False, default="UTC"),
        String("allowed_signal", required=False, doc="a 0/1 predicate from the integration"),
    )
    plan_order = 10
    ports = (
        Port("zone", kind="node", heats=True),
        Port("mass", kind="node", required=False, doc="takes f_rad of the heat"),
    )
    params = (
        Param("gain", DIMENSIONLESS, LogNormal(1.0, 1.5), (1e-3, 100.0), description="delivered fraction"),
        Param("t_ign", TIME, LogNormal(900.0, 1.5), (60.0, 7200.0), description="ignition ramp"),
        Param("f_rad", DIMENSIONLESS, LogNormal(0.2, 2.0), (0.001, 0.99), description="share to mass"),
    )

    def signals(self) -> list[str]:
        return [self.config["heat_signal"]] if self.config.get("heat_signal") else []

    def check(self) -> list[tuple[str | None, str]]:
        heats = [lv["heat"] for lv in self.config["levels"]]
        if any(b <= a for a, b in pairwise(heats)):
            return [("levels", "levels must be in increasing order of heat output")]
        return []

    def active_params(self, mode: str) -> list[str]:
        if mode == "plan":
            return ["t_ign"]
        return ["gain", "f_rad"] if self.refs.get("mass") else ["gain"]

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        zone = self.refs["zone"]
        if mode == "plan":
            return [Decision(f"{self.name}.Q", zone)]
        sig = self.config.get("heat_signal")
        if not sig:
            raise ModelError(f"{self.name!r}: identification needs the stove's heat_signal (§6.4)")
        mass = self.refs.get("mass")
        if not mass:
            return [Gain(zone, Signal(sig), p["gain"])]
        f = p["f_rad"]
        return [Gain(zone, Signal(sig), (1.0 - f) * p["gain"]), Gain(mass, Signal(sig), f * p["gain"])]

    def active_heat(self, p: Mapping[str, float]) -> np.ndarray:
        """Delivered heat per level, W: the delivered fraction × the rated output."""
        return float(p.get("gain", 1.0)) * np.array([lv["heat"] for lv in self.config["levels"]])

    def ignition_loss(self, t_ign: float, step: float, n: int) -> np.ndarray:
        """Share of a level's output lost in each of the first ``n`` steps
        after a start, for an output ramping linearly over ``t_ign``."""
        out = np.zeros(n)
        for j in range(n):
            a, b = j * step, (j + 1) * step
            ts = np.linspace(a, b, 61)
            out[j] = 1.0 - float(np.mean(np.minimum(1.0, ts / max(t_ign, 1e-9))))
        return out

    def allowed(self, ctx) -> np.ndarray:
        c = self.config
        mid = ctx.index + pd.to_timedelta(ctx.dt / 2, unit="s")
        h = np.asarray(mid.tz_convert(c.get("timezone") or "UTC").hour)
        a, b = c.get("allowed_from", 0), c.get("allowed_to", 24)
        ok = (h >= a) & (h < b) if a <= b else (h >= a) | (h < b)
        if c.get("occupancy_signal"):
            ok &= ctx.signal(c["occupancy_signal"]) >= c.get("min_occupancy", 0.5)
        if c.get("allowed_signal"):
            ok &= ctx.signal(c["allowed_signal"]) > 0.5
        return ok

    def plan(self, ctx) -> None:
        """Levels as one-hot binaries per step; starts by level (exact as
        products of binaries, so continuous); the ignition ramp as lost
        output in the first ``⌈t_ign/Δt⌉`` steps, the level held through it;
        minimum burn and cool-down; forced off where not allowed (§8.3–8.4).
        A burn under way at plan time (``devices[name]``: ``on``, ``level``,
        ``on_for``/``off_for`` s) carries its history in."""
        if self.config.get("modulating"):
            return self._plan_modulating(ctx)
        from ..plan.blocks import Linear, lag, lag_sum, window

        name, n, c = self.name, ctx.n, self.config
        p = ctx.values(name)
        step = ctx.house.step
        heat = self.active_heat(p)
        fuel = np.array([lv["fuel"] for lv in c["levels"]]) * 3600.0  # kg/h
        L = len(heat)
        n_ign = max(1, _steps(float(p.get("t_ign", 900.0)), step))
        loss = self.ignition_loss(float(p.get("t_ign", 900.0)), step, n_ign)
        n_on, n_off = _steps(c.get("min_on", 0.0), step), _steps(c.get("min_off", 0.0), step)
        H = max(n_on, n_off, n_ign, 1) + 1
        dev = ctx.devices.get(name, {})
        on_h, st_h, _ = _history(dev, H, step)
        y = ctx.milp.var(f"{name}.y", (n, L), 0, 1, integer=True)
        s = ctx.milp.var(f"{name}.s", (n, L), 0.0, 1.0)
        on = Linear.of(y)  # Σ_l y_{k,l}
        prev_on = lag_sum(y, 1, on_h)
        allowed = self.allowed(ctx)
        if dev.get("on"):
            k_left = n_on - max(1, _steps(float(dev.get("on_for", 1e9)), step))
            if k_left > 0:
                allowed[:k_left] = True
        t, lb, ub = on.rows(ub=allowed.astype(float))
        ctx.milp.add(t, lb, ub, name=f"{name} one level, allowed")
        ctx.milp.add([(s, 1.0), (y, -1.0)], ub=0.0, name=f"{name} start within level")
        for lvl in range(L):
            t, lb, ub = (Linear.of(s[:, lvl]) + prev_on).rows(ub=1.0)
            ctx.milp.add(t, lb, ub, name=f"{name} start only from off")
            t, lb, ub = (Linear.of(s[:, lvl]) - Linear.of(y[:, lvl]) + prev_on).rows(lb=0.0)
            ctx.milp.add(t, lb, ub, name=f"{name} start")
            for j in range(1, n_ign):
                # the level is held through ignition
                t, lb, ub = (Linear.of(y[:, lvl]) - lag(s[:, lvl], j, 0.0)).rows(lb=0.0)
                ctx.milp.add(t, lb, ub, name=f"{name} level held while igniting")
        # the per-step start count as its own variable, for windows over it
        st = ctx.milp.var(f"{name}.start", n, 0.0, 1.0)
        t, lb, ub = (Linear.of(st) - Linear.of(s)).rows(0.0, 0.0)
        ctx.milp.add(t, lb, ub, name=f"{name} start count")
        if n_on > 1:
            t, lb, ub = (window(st, n_on, st_h) - on).rows(ub=0.0)
            ctx.milp.add(t, lb, ub, name=f"{name} minimum burn")
        if n_off > 0:
            t, lb, ub = (window(st, n_off, st_h) + lag_sum(y, n_off, on_h)).rows(ub=1.0)
            ctx.milp.add(t, lb, ub, name=f"{name} cool-down")
        # delivered heat (W → kW): levels less the ignition loss
        Q = Linear.of(y, heat / 1000.0)
        for j in range(n_ign):
            for lvl in range(L):
                Q = Q - lag(s[:, lvl], j, 0.0) * (loss[j] * heat[lvl] / 1000.0)
        U = ctx.decision(f"{name}.Q")
        t, lb, ub = (Linear.of(U) - Q).rows(0.0, 0.0)
        ctx.milp.add(t, lb, ub, name=f"{name} heat")
        ctx.add_fuel(name, Linear.of(y, fuel), c["fuel"])
        if c.get("start_cost", 0) > 0:
            ctx.add_cost(f"{name} starts", Linear.of(st, c["start_cost"]))
        fp = ctx.shared.get("fuel_price", {}).get(c["fuel"])
        if fp is not None:
            per_kwh = fp * float(np.min(fuel / np.maximum(heat / 1000.0, 1e-9)))
            ctx.add_heat_cost(np.full(n, per_kwh))
        ctx.hold(y, name=f"{name} held")
        if c.get("block", 0) > step:
            ctx.hold(y, name=f"{name} block", block=c["block"], free_hours=0.0)
        pin = ctx.pinned(f"{name}.level")
        if pin is not None:
            for k in np.flatnonzero(~np.isnan(pin)):
                lv = int(pin[k])
                ctx.milp.fix(y[k], [1.0 if j == lv - 1 else 0.0 for j in range(L)])
        ctx.add_result(f"{name}.level", lambda x: (x[y] * np.arange(1, L + 1)).sum(axis=1).round())
        ctx.add_result(f"{name}.start", lambda x: x[st])

    def _plan_modulating(self, ctx) -> None:
        """The relaxation that keeps one binary per step: lit or not, and
        when lit an output anywhere between the lowest and the highest
        level, with fuel linear in the output (fitted through the levels).
        The ignition loss is the started output's share, exact through a
        product of the start indicator and the output (McCormick, exact at
        integral points). The translator rounds the output to the nearest
        level. Starts, minimum burn, cool-down and the allowed steps are as
        in the discrete form."""
        from ..plan.blocks import Linear, lag, window

        name, n, c = self.name, ctx.n, self.config
        p = ctx.values(name)
        step = ctx.house.step
        heat = self.active_heat(p) / 1000.0  # kW
        fuel = np.array([lv["fuel"] for lv in c["levels"]]) * 3600.0  # kg/h
        lo, hi = float(heat[0]), float(heat[-1])
        f1, f0 = np.polyfit(heat, fuel, 1) if len(heat) > 1 else (fuel[0] / heat[0], 0.0)
        n_ign = max(1, _steps(float(p.get("t_ign", 900.0)), step))
        loss = self.ignition_loss(float(p.get("t_ign", 900.0)), step, n_ign)
        n_on, n_off = _steps(c.get("min_on", 0.0), step), _steps(c.get("min_off", 0.0), step)
        H = max(n_on, n_off, n_ign, 1) + 1
        dev = ctx.devices.get(name, {})
        on_h, st_h, _ = _history(dev, H, step)
        on = ctx.milp.var(f"{name}.on", n, 0, 1, integer=True)
        P = ctx.milp.var(f"{name}.power", n, 0.0, hi)
        st = ctx.milp.var(f"{name}.start", n, 0.0, 1.0)
        allowed = self.allowed(ctx)
        if dev.get("on"):
            k_left = n_on - max(1, _steps(float(dev.get("on_for", 1e9)), step))
            if k_left > 0:
                allowed[:k_left] = True
        ctx.milp.bound(on, ub=allowed.astype(float))
        ctx.milp.add([(P, 1.0), (on, -hi)], ub=0.0, name=f"{name} output when lit")
        ctx.milp.add([(P, 1.0), (on, -lo)], lb=0.0, name=f"{name} lowest level when lit")
        prev = lag(on, 1, on_h)
        for t_, lb, ub in ((Linear.of(st) + prev).rows(ub=1.0),):
            ctx.milp.add(t_, lb, ub, name=f"{name} start only from off")
        t_, lb, ub = (Linear.of(st) - Linear.of(on) + prev).rows(lb=0.0)
        ctx.milp.add(t_, lb, ub, name=f"{name} start")
        ctx.milp.add([(st, 1.0), (on, -1.0)], ub=0.0, name=f"{name} start when lit")
        if n_on > 1:
            t_, lb, ub = (window(st, n_on, st_h) - Linear.of(on)).rows(ub=0.0)
            ctx.milp.add(t_, lb, ub, name=f"{name} minimum burn")
        if n_off > 0:
            t_, lb, ub = (window(st, n_off, st_h) + lag(on, n_off, on_h)).rows(ub=1.0)
            ctx.milp.add(t_, lb, ub, name=f"{name} cool-down")
        Q = Linear.of(P)
        if n_ign:
            z = ctx.milp.var(f"{name}.ignition", (n, n_ign), 0.0, hi)
            for j in range(n_ign):
                sj = lag(st, j, 0.0)  # started j steps ago
                # z = P_k · s_{k-j}: z ≤ hi·s, z ≤ P, z ≥ P − hi(1 − s)
                t_, lb, ub = (Linear.of(z[:, j]) - sj * hi).rows(ub=0.0)
                ctx.milp.add(t_, lb, ub, name=f"{name} ignition product")
                ctx.milp.add([(z[:, j], 1.0), (P, -1.0)], ub=0.0, name=f"{name} ignition product")
                t_, lb, ub = (Linear.of(z[:, j]) - Linear.of(P) - sj * hi).rows(lb=-hi)
                ctx.milp.add(t_, lb, ub, name=f"{name} ignition product")
                Q = Q - Linear.of(z[:, j], float(loss[j]))
        U = ctx.decision(f"{name}.Q")
        t_, lb, ub = (Linear.of(U) - Q).rows(0.0, 0.0)
        ctx.milp.add(t_, lb, ub, name=f"{name} heat")
        ctx.add_fuel(name, Linear.of(on, f0) + Linear.of(P, f1), c["fuel"])
        if c.get("start_cost", 0) > 0:
            ctx.add_cost(f"{name} starts", Linear.of(st, c["start_cost"]))
        fp = ctx.shared.get("fuel_price", {}).get(c["fuel"])
        if fp is not None:
            ctx.add_heat_cost(np.full(n, fp * float(np.min(fuel / heat))))
        ctx.hold(on, name=f"{name} held")
        if c.get("block", 0) > step:
            ctx.hold(on, name=f"{name} block", block=c["block"], free_hours=0.0)
            ctx.hold(P, name=f"{name} block", block=c["block"], free_hours=0.0)
        pin = ctx.pinned(f"{name}.level")
        if pin is not None:
            k = np.flatnonzero(~np.isnan(pin))
            if len(k):
                lv = pin[k].astype(int)
                ctx.milp.fix(on[k], (lv > 0).astype(float))
                ctx.milp.fix(P[k], np.where(lv > 0, heat[np.clip(lv - 1, 0, len(heat) - 1)], 0.0))
        levels = heat

        def level(x):
            pw = x[P]
            nearest = np.abs(pw[:, None] - levels[None, :]).argmin(axis=1) + 1
            return np.where(x[on] > 0.5, nearest, 0).astype(float)

        ctx.add_result(f"{name}.level", level)
        ctx.add_result(f"{name}.power", lambda x: x[P] * 1000.0)
        ctx.add_result(f"{name}.start", lambda x: x[st])

    def actuate(self, plan) -> list:
        """§8.7: the next burn as a discrete schedule — start time and level,
        stop time — in absolute times (§9.2: the stop is part of the start)."""
        from ..actuate import stove_schedule

        col = f"{self.name}.level"
        return stove_schedule(plan, self.name, col) if col in plan.decisions else []
