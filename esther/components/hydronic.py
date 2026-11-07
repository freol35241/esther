"""Hydronic components (spec §4.5.3): radiators and tanks.

Tank capacities come from their volumes (``ρ c_p`` of water times the
volume), so a tank is a heat store of **known size**: with one heat input of
known power into it (an electric heater, COP 1) the tank fixes the model's
absolute scale (§6.2), and radiators that take their heat from the tank carry
that scale to the envelope. ``f_C`` — the share of the nominal water capacity
that behaves as the modelled nodes (steel adds to it; dead volume and
stratification take from it) — is a parameter with a tight prior; pin it to
1 to make the volumes the anchor, or leave it free to test them against a
known heater.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

from ..core.component import Component, Integer, Mapping_, Port, Quantity, String
from ..core.model import Capacity, Conductance, Decision, Measurement, ModelError, Noise, Signal, Transfer
from ..core.params import LogNormal, Param
from ..core.units import CONDUCTANCE, DIMENSIONLESS, TEMPERATURE, VOLUME, VOLUME_FLOW

#: Volumetric heat capacity of water, ρ c_p, J/(m³ K) at about 50 °C
#: (988 kg/m³ × 4181 J/(kg K)).
RHO_CP_WATER = 988.0 * 4181.0


def _sigma_w(name: str, median: float) -> Param:
    return Param(
        name, TEMPERATURE, LogNormal(median, 3.0), (1e-5, 20.0), description="process noise, K per √h"
    )


def _sigma_v(name: str, median: float) -> Param:
    return Param(name, TEMPERATURE, LogNormal(median, 3.0), (1e-4, 5.0), description="sensor noise, K")


def _f_C() -> Param:
    return Param(
        "f_C",
        DIMENSIONLESS,
        LogNormal(1.0, 1.15),
        (0.2, 5.0),
        description="effective share of the nominal water capacity",
    )


class Radiators(Component):
    """Heat from a source node (the circuit mixes from it) to a zone (§4.5.3).

    *Identification* (measurement form): with ``supply_signal`` measured,
    ``Q_rad = UA_rad (T_supply − T_z)`` — a conductance from the zone to the
    measured supply temperature, so ``UA_rad`` is fitted. If ``flow_signal``
    is given (the circulation pump's on-fraction per step, in [0, 1]) the
    conductance is scaled by it: no flow, no heat, whatever the supply sensor
    reads. The emitter law is linear; the exponent-n law is a later
    refinement (§12 question 2).

    The heat the radiators give the zone is **taken from** ``source`` (the
    tank they mix from) when that node is in the model — the return water
    brings the rest back. This is what couples a tank fit to the envelope:
    the tank's known capacity sizes ``UA_rad`` in watts. When the source is
    left out of the model (an envelope-only fit, §6.4) nothing is taken.

    *Planning* (decision form): the decision is ``Q_rad ≥ 0`` into the zone,
    taken from ``source``, with ``0 ≤ Q_rad ≤ UA_rad (T_source − T_z)`` (the
    shunt mixes down only) and, with ``supply_max``, ``Q_rad ≤ UA_rad
    (supply_max − T_z)``, at each step's start temperatures. The supply
    temperature it needs, ``T_z + Q_rad / UA_rad``, is a result of the plan.
    """

    plan_order = 15

    fields = (
        String("supply_signal", required=False, doc="measured supply temperature"),
        String("flow_signal", required=False, doc="share of each step the circuit flows, 0..1"),
        Quantity("supply_max", dim=TEMPERATURE, absolute=True, required=False, min=273.15, max=373.15),
    )
    ports = (
        Port("zone", kind="node", heats=True),
        Port("source", kind="node", required=False, doc="the node the circuit mixes from"),
    )
    params = (Param("UA_rad", CONDUCTANCE, LogNormal(250.0, 2.5), (5.0, 1e4)),)

    def signals(self) -> list[str]:
        return [s for s in (self.config.get("supply_signal"), self.config.get("flow_signal")) if s]

    def check(self) -> list[tuple[str | None, str]]:
        if self.refs.get("source") and self.refs["zone"] == self.refs["source"]:
            return [("source", "radiators cannot draw from the zone they heat")]
        return []

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        zone = self.refs["zone"]
        src = self.refs.get("source")
        if mode == "plan":
            return [Decision(f"{self.name}.Q", zone, frm=src, frm_optional=True)]
        supply = self.config.get("supply_signal")
        if not supply:
            raise ModelError(
                f"{self.name!r}: identification needs the radiators' supply_signal "
                "(the measurement form, §4.5.3)"
            )
        flow = self.config.get("flow_signal")
        if not src:
            return [Conductance(zone, Signal(supply), p["UA_rad"], flow)]
        return [
            Transfer(p["UA_rad"], Signal(supply), zone, to=zone, frm=src, modulation=flow, frm_optional=True)
        ]

    def plan(self, ctx) -> None:
        name = self.name
        if not ctx.has_decision(f"{name}.Q"):
            return
        U = ctx.decision(f"{name}.Q")
        ua = float(ctx.values(name).get("UA_rad", 250.0)) / 1000.0  # kW/K
        zone, src = self.refs["zone"], self.refs.get("source")
        Xz = ctx.state(zone)[:-1]
        if src and ctx.has_state(src):
            Xs = ctx.state(src)[:-1]
            ctx.milp.add([(U, 1.0), (Xs, -ua), (Xz, ua)], ub=0.0, name=f"{name} mixes down only")
        if self.config.get("supply_max") is not None:
            ctx.milp.add(
                [(U, 1.0), (Xz, ua)], ub=ua * (self.config["supply_max"] - 273.15), name=f"{name} supply max"
            )
        ctx.add_result(f"{name}.supply", lambda x: x[Xz] + 273.15 + x[U] / ua)


class JacketTank(Component):
    """Double-jacket tank: an outer jacket of heating water around an inner
    vessel of domestic hot water (§4.5.3).

    Nodes ``<name>.jacket`` and ``<name>.dhw``; with ``dhw_layers = 2`` the
    inner vessel is split into ``<name>.dhw`` (its lower part, where cold
    mains water enters and draws show) and ``<name>.top`` (``f_top`` of its
    volume). Sources heat the jacket; the radiator circuit draws from it;
    draws cool ``<name>.dhw`` (a :class:`~esther.components.loads.HotWaterDraw`
    on that node). Losses go to ``room``.

    Couplings (all ``Q = UA ΔT``): jacket–dhw ``UA_jd``, jacket–room
    ``UA_loss``, and with two layers dhw–top ``UA_top`` and top–room
    ``UA_loss_top``. A tank at rest stratifies and a charging tank is stirred,
    so the couplings differ between the two; ``mixing_signal`` (the share of
    each step something stirs the tank — a charge pump, a heater's convection
    — in [0, 1]) adds ``UA_mix`` to jacket–dhw and ``UA_top_mix`` to dhw–top
    while it is on. That keeps the model linear in the temperatures; the
    buoyancy it approximates is not.

    Measurements: ``jacket_signal``, ``dhw_signal``, ``top_signal``, each with
    its own noise. Unmeasured nodes should have their ``sigma_w`` pinned
    (§4.3).
    """

    fields = (
        Quantity("volume_jacket", dim=VOLUME, positive=True, max=10.0),
        Quantity("volume_dhw", dim=VOLUME, positive=True, max=10.0),
        Integer("dhw_layers", required=False, default=1, min=1, max=2),
        String("jacket_signal", required=False, doc="jacket (heating water) temperature"),
        String("dhw_signal", required=False, doc="inner vessel (lower part) temperature"),
        String("top_signal", required=False, doc="inner vessel top temperature (two layers)"),
        String("mixing_signal", required=False, doc="share of each step the tank is stirred, 0..1"),
    )
    ports = (Port("room", kind="thermal", doc="where standby losses go"),)
    params = (
        _f_C(),
        Param("UA_jd", CONDUCTANCE, LogNormal(100.0, 3.0), (0.1, 1e5), description="jacket to DHW"),
        Param(
            "UA_mix", CONDUCTANCE, LogNormal(300.0, 3.0), (0.1, 1e5), description="extra jacket–DHW, stirred"
        ),
        Param("UA_loss", CONDUCTANCE, LogNormal(2.0, 3.0), (0.01, 100.0), description="jacket to the room"),
        Param(
            "f_top", DIMENSIONLESS, LogNormal(0.3, 1.5), (0.02, 0.9), description="top layer's share of DHW"
        ),
        Param("UA_top", CONDUCTANCE, LogNormal(5.0, 3.0), (0.01, 1e4), description="DHW lower to top"),
        Param("UA_top_mix", CONDUCTANCE, LogNormal(30.0, 3.0), (0.01, 1e5), description="extra, stirred"),
        Param("UA_loss_top", CONDUCTANCE, LogNormal(1.0, 3.0), (0.01, 100.0), description="top to the room"),
        _sigma_w("sigma_w_jacket", 0.3),
        _sigma_w("sigma_w_dhw", 0.1),
        _sigma_w("sigma_w_top", 0.05),
        _sigma_v("sigma_v_jacket", 0.1),
        _sigma_v("sigma_v_dhw", 0.1),
        _sigma_v("sigma_v_top", 0.1),
    )

    @property
    def layered(self) -> bool:
        return (self.config.get("dhw_layers") or 1) == 2

    def nodes(self) -> list[str]:
        out = [f"{self.name}.jacket", f"{self.name}.dhw"]
        return [*out, f"{self.name}.top"] if self.layered else out

    def conductances(self) -> list[tuple[str, str]]:
        j, d = f"{self.name}.jacket", f"{self.name}.dhw"
        room = self.refs["room"]
        out = [(j, d), (j, room)]
        if self.layered:
            out += [(d, f"{self.name}.top"), (f"{self.name}.top", room)]
        return out

    def signals(self) -> list[str]:
        keys = ("jacket_signal", "dhw_signal", "top_signal", "mixing_signal")
        return [self.config[k] for k in keys if self.config.get(k)]

    def check(self) -> list[tuple[str | None, str]]:
        if self.config.get("top_signal") and not self.layered:
            return [("top_signal", "a top sensor needs dhw_layers = 2")]
        return []

    def capacities(self, p: Mapping[str, float]) -> dict[str, float]:
        """Heat capacity of each node, J/K."""
        f = p["f_C"] * RHO_CP_WATER
        dhw = f * self.config["volume_dhw"]
        out = {f"{self.name}.jacket": f * self.config["volume_jacket"]}
        if self.layered:
            out[f"{self.name}.dhw"] = (1.0 - p["f_top"]) * dhw
            out[f"{self.name}.top"] = p["f_top"] * dhw
        else:
            out[f"{self.name}.dhw"] = dhw
        return out

    def active_params(self, mode: str) -> list[str]:
        c = self.config
        out = ["f_C", "UA_jd", "UA_loss", "sigma_w_jacket", "sigma_w_dhw"]
        if c.get("mixing_signal"):
            out.append("UA_mix")
        if self.layered:
            out += ["f_top", "UA_top", "UA_loss_top", "sigma_w_top"]
            if c.get("mixing_signal"):
                out.append("UA_top_mix")
        for key, sv in (("jacket_signal", "sigma_v_jacket"), ("dhw_signal", "sigma_v_dhw")):
            if c.get(key):
                out.append(sv)
        if self.layered and c.get("top_signal"):
            out.append("sigma_v_top")
        return out

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        c = self.config
        j, d, t = f"{self.name}.jacket", f"{self.name}.dhw", f"{self.name}.top"
        room = self.refs["room"]
        mix = c.get("mixing_signal")
        terms: list = [Capacity(n, C) for n, C in self.capacities(p).items()]
        terms += [
            Conductance(j, d, p["UA_jd"]),
            Conductance(j, room, p["UA_loss"]),
            Noise(j, p["sigma_w_jacket"]),
            Noise(d, p["sigma_w_dhw"]),
        ]
        if mix:
            terms.append(Conductance(j, d, p["UA_mix"], mix))
        if self.layered:
            terms += [Conductance(d, t, p["UA_top"]), Conductance(t, room, p["UA_loss_top"])]
            terms.append(Noise(t, p["sigma_w_top"]))
            if mix:
                terms.append(Conductance(d, t, p["UA_top_mix"], mix))
        if c.get("jacket_signal"):
            terms.append(Measurement(c["jacket_signal"], j, p["sigma_v_jacket"]))
        if c.get("dhw_signal"):
            terms.append(Measurement(c["dhw_signal"], d, p["sigma_v_dhw"]))
        if self.layered and c.get("top_signal"):
            terms.append(Measurement(c["top_signal"], t, p["sigma_v_top"]))
        return terms


_LAYERS = {1: ("water",), 2: ("top", "bottom"), 3: ("top", "middle", "bottom")}


class StratifiedTank(Component):
    """A plain tank of ``layers`` (1–3) stacked layers of equal volume: a DHW
    cylinder or a buffer tank (§4.5.3).

    Nodes ``<name>.top``, ``<name>.middle``, ``<name>.bottom`` (two layers:
    top and bottom; one layer: ``<name>.water``). Adjacent layers exchange
    heat through ``UA_layer``; each loses ``UA_loss / layers`` to ``room``.

    **Draw-driven advection.** With ``draw_signal`` (the draw flow as a share
    of ``draw_flow_max``, in [0, 1], per step) hot water leaves from the top,
    every layer is replaced by the one below it and the bottom by mains water:
    layer ``i`` receives ``ρ c_p V̇ (T_{i+1} − T_i)``, the bottom
    ``ρ c_p V̇ (T_mains − T_bottom)``. The flow is one-sided, which is why it is
    written with :class:`~esther.core.model.Transfer` rather than a
    conductance. Sources name the layer they heat (``node = "tank.bottom"``).

    ``sensors`` maps layer names to signals, e.g. ``{ top = "t_top" }``.
    """

    fields = (
        Quantity("volume", dim=VOLUME, positive=True, max=50.0),
        Integer("layers", required=False, default=2, min=1, max=3),
        Mapping_("sensors", required=False, default={}, value_field=String("signal")),
        String("draw_signal", required=False, doc="draw flow as a share of draw_flow_max, 0..1"),
        Quantity("draw_flow_max", dim=VOLUME_FLOW, required=False, default=2e-4, positive=True),
        Quantity(
            "mains_temperature",
            dim=TEMPERATURE,
            absolute=True,
            required=False,
            default=283.15,
            min=273.15,
            max=313.15,
        ),
    )
    ports = (Port("room", kind="thermal", doc="where standby losses go"),)
    params = (
        _f_C(),
        Param("UA_layer", CONDUCTANCE, LogNormal(20.0, 3.0), (0.01, 1e5), description="between layers"),
        Param(
            "UA_loss", CONDUCTANCE, LogNormal(2.0, 3.0), (0.01, 100.0), description="whole tank to the room"
        ),
        _sigma_w("sigma_w", 0.1),
        _sigma_v("sigma_v", 0.1),
    )

    def layer_names(self) -> tuple[str, ...]:
        return _LAYERS[self.config.get("layers") or 2]

    def nodes(self) -> list[str]:
        return [f"{self.name}.{n}" for n in self.layer_names()]

    def conductances(self) -> list[tuple[str, str]]:
        ns = self.nodes()
        return [*itertools.pairwise(ns), *((n, self.refs["room"]) for n in ns)]

    def signals(self) -> list[str]:
        out = list((self.config.get("sensors") or {}).values())
        return [*out, self.config["draw_signal"]] if self.config.get("draw_signal") else out

    def check(self) -> list[tuple[str | None, str]]:
        bad = sorted(set(self.config.get("sensors") or {}) - set(self.layer_names()))
        if bad:
            return [("sensors", f"no layer {', '.join(bad)}; layers are {', '.join(self.layer_names())}")]
        return []

    def active_params(self, mode: str) -> list[str]:
        out = ["f_C", "UA_loss", "sigma_w"]
        if len(self.layer_names()) > 1:
            out.insert(1, "UA_layer")
        if self.config.get("sensors"):
            out.append("sigma_v")
        return out

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        ns = self.nodes()
        n = len(ns)
        room = self.refs["room"]
        C = p["f_C"] * RHO_CP_WATER * self.config["volume"] / n
        terms: list = []
        for node in ns:
            terms += [Capacity(node, C), Conductance(node, room, p["UA_loss"] / n), Noise(node, p["sigma_w"])]
        for a, b in itertools.pairwise(ns):
            terms.append(Conductance(a, b, p["UA_layer"]))
        draw = self.config.get("draw_signal")
        if draw:
            G = RHO_CP_WATER * self.config["draw_flow_max"]
            mains = float(self.config["mains_temperature"])
            for i, node in enumerate(ns):
                below: str | float = ns[i + 1] if i + 1 < n else mains
                terms.append(Transfer(G, below, node, to=node, modulation=draw))
        for layer, sig in (self.config.get("sensors") or {}).items():
            terms.append(Measurement(sig, f"{self.name}.{layer}", p["sigma_v"]))
        return terms
