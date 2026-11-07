"""Heat sources (spec §4.5.4).

In **identification** a source enters as a measured heat input (§6.4): the
integration supplies what the device reports — a compressor's on-fraction, a
heater's share of its maximum, a stove's estimated heat — and the source turns
it into watts through a fitted parameter whose prior comes from the datasheet.
In **planning** each source is a heat-rate decision into the node it heats;
levels, minimum run times and costs arrive with the planner (M4).

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

from ..core.component import Component, Integer, Number, Port, Quantity, String, TableList
from ..core.model import Decision, Gain, ModelError, Signal, Transfer
from ..core.params import LogNormal, Param
from ..core.units import CONDUCTANCE, DIMENSIONLESS, MASS_FLOW, POWER, TEMPERATURE, TIME


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
    )
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

    def carnot_cop(self, p: Mapping[str, float], t_sink: float, t_src: float) -> float:
        """``η T_sink / (T_sink − T_src)``, temperatures in K (planning, §4.3)."""
        lift = max(t_sink - t_src, 1.0)
        return p["eta"] * t_sink / lift


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
    )
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
    )
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
