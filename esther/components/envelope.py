"""Envelope components (spec §4.5.1)."""

from __future__ import annotations

from ..core.component import Component, Port, String
from ..core.params import LogNormal, Param
from ..core.units import CONDUCTANCE, HEAT_CAPACITY


class Zone(Component):
    """An air node, ``C_air dT_z/dt = Σ Q_in``."""

    params = (Param("C", HEAT_CAPACITY, LogNormal(5e6, 2.0), (1e5, 1e9), description="air and furnishings"),)

    def nodes(self) -> list[str]:
        return [self.name]


class Mass(Component):
    """Building fabric coupled to a zone: ``C_m dT_m/dt = UA (T_z − T_m) + …``. Unmeasured."""

    ports = (Port("zone", kind="node"),)
    params = (
        Param("C", HEAT_CAPACITY, LogNormal(50e6, 3.0), (1e6, 1e10)),
        Param("UA", CONDUCTANCE, LogNormal(500.0, 3.0), (1.0, 1e5), description="zone to mass"),
    )

    def nodes(self) -> list[str]:
        return [self.name]

    def conductances(self) -> list[tuple[str, str]]:
        return [(self.name, self.refs["zone"])]


class Ambient(Component):
    """A boundary: outdoor air (or ground), from a named signal."""

    fields = (String("signal", doc="the measured/forecast temperature signal"),)

    def boundaries(self) -> list[str]:
        return [self.name]

    def signals(self) -> list[str]:
        return [self.config["signal"]]


class Coupling(Component):
    """``Q = UA (T_a − T_b)`` between two nodes, or a node and a boundary."""

    ports = (Port("between", kind="thermal", count=2),)
    params = (Param("UA", CONDUCTANCE, LogNormal(100.0, 3.0), (0.1, 1e5)),)

    def conductances(self) -> list[tuple[str, str]]:
        a, b = self.refs["between"]
        return [(a, b)]

    def check(self) -> list[tuple[str | None, str]]:
        a, b = self.refs["between"]
        return [("between", f"couples {a!r} to itself")] if a == b else []
