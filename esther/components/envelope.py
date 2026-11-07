"""Envelope components (spec §4.5.1).

Noise parameters are part of each node's contract: ``sigma_w`` is the process
noise on a node, as the standard deviation (K) its temperature diffuses by in
one hour with every input known; ``sigma_v`` is a sensor's white measurement
noise (K). Both are fitted like any other parameter (§6.2).
"""

from __future__ import annotations

from collections.abc import Mapping

from ..core.component import Component, Port, String
from ..core.model import Capacity, Conductance, Measurement, Noise
from ..core.params import LogNormal, Normal, Param
from ..core.units import CONDUCTANCE, HEAT_CAPACITY, TEMPERATURE, VOLUME_FLOW

#: Volumetric heat capacity of air, ρ c_p, J/(m³ K) at about 20 °C.
RHO_CP_AIR = 1.2 * 1006.0


def _sigma_w(median: float) -> Param:
    return Param(
        "sigma_w",
        TEMPERATURE,
        LogNormal(median, 3.0),
        (1e-5, 10.0),
        description="process noise, K per √h",
    )


def _sigma_v(median: float) -> Param:
    return Param("sigma_v", TEMPERATURE, LogNormal(median, 3.0), (1e-4, 5.0), description="sensor noise, K")


class Zone(Component):
    """An air node, ``C_air dT_z/dt = Σ Q_in``.

    ``signal`` names the zone's reference sensor: it defines the zone
    temperature (no bias). Further sensors on the same node are
    :class:`Sensor` components, each with its own bias.
    """

    fields = (String("signal", required=False, doc="the zone's reference temperature sensor"),)
    params = (
        Param("C", HEAT_CAPACITY, LogNormal(5e6, 2.0), (1e5, 1e9), description="air and furnishings"),
        _sigma_w(0.1),
        _sigma_v(0.05),
    )

    def nodes(self) -> list[str]:
        return [self.name]

    def signals(self) -> list[str]:
        s = self.config.get("signal")
        return [s] if s else []

    def active_params(self, mode: str) -> list[str]:
        return ["C", "sigma_w"] + (["sigma_v"] if self.config.get("signal") else [])

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        terms: list = [Capacity(self.name, p["C"]), Noise(self.name, p["sigma_w"])]
        if self.config.get("signal"):
            terms.append(Measurement(self.config["signal"], self.name, p["sigma_v"]))
        return terms


class Mass(Component):
    """Building fabric coupled to a zone: ``C_m dT_m/dt = UA (T_z − T_m) + …``. Unmeasured."""

    ports = (Port("zone", kind="node"),)
    params = (
        Param("C", HEAT_CAPACITY, LogNormal(50e6, 3.0), (1e6, 1e10)),
        Param("UA", CONDUCTANCE, LogNormal(500.0, 3.0), (1.0, 1e5), description="zone to mass"),
        _sigma_w(0.02),
    )

    def nodes(self) -> list[str]:
        return [self.name]

    def conductances(self) -> list[tuple[str, str]]:
        return [(self.name, self.refs["zone"])]

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        return [
            Capacity(self.name, p["C"]),
            Conductance(self.name, self.refs["zone"], p["UA"]),
            Noise(self.name, p["sigma_w"]),
        ]


class Ambient(Component):
    """A boundary: outdoor air (or ground), from a named signal."""

    fields = (String("signal", doc="the measured/forecast temperature signal"),)

    def boundaries(self) -> list[str]:
        return [self.name]

    def signals(self) -> list[str]:
        return [self.config["signal"]]

    def boundary_signals(self) -> dict[str, str]:
        return {self.name: self.config["signal"]}

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        return []


class Coupling(Component):
    """``Q = UA (T_a − T_b)`` between two nodes, or a node and a boundary.

    With ``flow_signal`` (a share of each step in [0, 1]) the conductance is
    scaled by it: a path that exists only while something circulates — a
    charge pump stirring two parts of a tank."""

    fields = (String("flow_signal", required=False, doc="share of each step the path is open, 0..1"),)
    ports = (Port("between", kind="thermal", count=2),)
    params = (Param("UA", CONDUCTANCE, LogNormal(100.0, 3.0), (0.1, 1e5)),)

    def signals(self) -> list[str]:
        return [self.config["flow_signal"]] if self.config.get("flow_signal") else []

    def conductances(self) -> list[tuple[str, str]]:
        a, b = self.refs["between"]
        return [(a, b)]

    def check(self) -> list[tuple[str | None, str]]:
        a, b = self.refs["between"]
        return [("between", f"couples {a!r} to itself")] if a == b else []

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        a, b = self.refs["between"]
        return [Conductance(a, b, p["UA"], self.config.get("flow_signal"))]


class Ventilation(Component):
    """A coupling to a boundary with conductance ``ρ c_p V̇`` (W/K), the air flow
    ``V̇`` fitted or pinned. In a house with constant mechanical ventilation it
    is indistinguishable from an envelope :class:`Coupling` to the same
    boundary, and one of the two must then be pinned."""

    ports = (Port("zone", kind="node"), Port("ambient", kind="boundary"))
    params = (Param("flow", VOLUME_FLOW, LogNormal(0.03, 2.0), (1e-4, 2.0), description="air flow, m³/s"),)

    def conductances(self) -> list[tuple[str, str]]:
        return [(self.refs["zone"], self.refs["ambient"])]

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        return [Conductance(self.refs["zone"], self.refs["ambient"], RHO_CP_AIR * p["flow"])]


class Sensor(Component):
    """A further temperature sensor on a node: reads ``T_node + bias`` (spec §4.5.1:
    "a sensor on a cold wall reads low; that is a bias, not a dynamic")."""

    fields = (String("signal"),)
    ports = (Port("node", kind="node"),)
    params = (
        Param("bias", TEMPERATURE, Normal(0.0, 1.0), (-10.0, 10.0)),
        _sigma_v(0.1),
    )

    def signals(self) -> list[str]:
        return [self.config["signal"]]

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        return [Measurement(self.config["signal"], self.refs["node"], p["sigma_v"], p["bias"])]
