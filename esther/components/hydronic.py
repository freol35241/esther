"""Hydronic components (spec §4.5.3)."""

from __future__ import annotations

from ..core.component import Component, Port, Quantity, String
from ..core.params import LogNormal, Param
from ..core.units import CONDUCTANCE, VOLUME


class Radiators(Component):
    """Heat from a source node (the circuit mixes from it) to a zone.

    Planning: ``0 ≤ Q_rad ≤ UA_rad (T_source − T_z)`` — the shunt mixes down only.
    """

    fields = (String("supply_signal", required=False, doc="measured supply temperature"),)
    ports = (
        Port("zone", kind="node", heats=True),
        Port("source", kind="node", doc="the node the circuit mixes from"),
    )
    params = (Param("UA_rad", CONDUCTANCE, LogNormal(250.0, 2.5), (5.0, 1e4)),)

    def signals(self) -> list[str]:
        s = self.config.get("supply_signal")
        return [s] if s else []

    def check(self) -> list[tuple[str | None, str]]:
        if self.refs["zone"] == self.refs["source"]:
            return [("source", "radiators cannot draw from the zone they heat")]
        return []


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
