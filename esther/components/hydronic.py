"""Hydronic components (spec §4.5.3)."""

from __future__ import annotations

from collections.abc import Mapping

from ..core.component import Component, Port, Quantity, String
from ..core.model import Conductance, Decision, ModelError, Signal
from ..core.params import LogNormal, Param
from ..core.units import CONDUCTANCE, VOLUME


class Radiators(Component):
    """Heat from a source node (the circuit mixes from it) to a zone (§4.5.3).

    *Identification* (measurement form): with ``supply_signal`` measured,
    ``Q_rad = UA_rad (T_supply − T_z)`` — a conductance from the zone to the
    measured supply temperature, so ``UA_rad`` is fitted. If ``flow_signal``
    is given (the circulation pump's on-fraction per step, in [0, 1]) the
    conductance is scaled by it: no flow, no heat, whatever the supply sensor
    reads. The emitter law is linear; the exponent-n law is a later
    refinement (§12 question 2).

    *Planning* (decision form): the decision is ``Q_rad ≥ 0`` into the zone,
    with ``0 ≤ Q_rad ≤ UA_rad (T_source − T_z)`` (the shunt mixes down only;
    the constraint arrives with the planner, M4).
    """

    fields = (
        String("supply_signal", required=False, doc="measured supply temperature"),
        String("flow_signal", required=False, doc="share of each step the circuit flows, 0..1"),
    )
    ports = (
        Port("zone", kind="node", heats=True),
        Port("source", kind="node", required=False, doc="the node the circuit mixes from (planning)"),
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
        if mode == "plan":
            return [Decision(f"{self.name}.Q", zone)]
        supply = self.config.get("supply_signal")
        if not supply:
            raise ModelError(
                f"{self.name!r}: identification needs the radiators' supply_signal "
                "(the measurement form, §4.5.3)"
            )
        return [Conductance(zone, Signal(supply), p["UA_rad"], self.config.get("flow_signal"))]


class JacketTank(Component):
    """Double-jacket tank: nodes ``<name>.jacket`` (heating water) and ``<name>.dhw``."""

    fields = (
        Quantity("volume_jacket", dim=VOLUME, positive=True, max=10.0),
        Quantity("volume_dhw", dim=VOLUME, positive=True, max=10.0),
    )
    ports = (Port("room", kind="thermal", doc="where standby losses go"),)
    params = (
        Param("UA_jd", CONDUCTANCE, LogNormal(300.0, 3.0), (1.0, 1e4), description="jacket to DHW"),
        Param("UA_loss", CONDUCTANCE, LogNormal(2.0, 3.0), (0.01, 100.0), description="to the room"),
    )

    def nodes(self) -> list[str]:
        return [f"{self.name}.jacket", f"{self.name}.dhw"]

    def conductances(self) -> list[tuple[str, str]]:
        j, d = self.nodes()
        room = self.refs["room"]
        return [(j, d), (j, room), (d, room)]
