"""Heat gains (spec §4.5.2): solar, internal, and measured heat inputs.

Each gain can split its heat between a zone and a mass node: ``f_mass`` of it
goes to ``mass`` (the fabric absorbs sun on the floor; a stove radiates to the
walls), the rest to the zone. Without ``mass`` everything goes to the zone and
``f_mass`` is not a parameter.

Signal scales are whatever the signal measures (its SI unit after the
dataset's conversion): a gain coefficient is "W per SI unit of the signal",
which is why gain coefficients are declared dimensionless.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..core.component import Component, Integer, Port, Quantity, String
from ..core.model import Constant, Gain, Harmonic, Signal
from ..core.params import LogNormal, Normal, Param
from ..core.units import DIMENSIONLESS, POWER, TIME

MAX_HARMONICS = 4


def _f_mass(mean: float) -> Param:
    return Param("f_mass", DIMENSIONLESS, Normal(mean, 0.3), (0.0, 1.0), description="share to the mass node")


class _SplitGain(Component):
    """A gain into ``zone`` (``node``), optionally split with a ``mass`` node."""

    target_port = "zone"

    def _split(self, p: Mapping[str, float], inp, gain: float) -> list:
        node = self.refs[self.target_port]
        mass = self.refs.get("mass")
        if not mass:
            return [Gain(node, inp, gain)]
        f = p["f_mass"]
        return [Gain(node, inp, (1.0 - f) * gain), Gain(mass, inp, f * gain)]

    def _base_params(self) -> list[str]:
        return ["f_mass"] if self.refs.get("mass") else []


class SolarGain(_SplitGain):
    """``Q_sol = g_A · I`` with ``I`` an irradiance proxy (a PV inverter's output,
    a lux sensor, a clear-sky model times cloud cover). ``g_A`` is in W per SI
    unit of the proxy."""

    fields = (String("signal", doc="the irradiance proxy"),)
    ports = (
        Port("zone", kind="node", heats=True),
        Port("mass", kind="node", required=False, doc="takes f_mass of the gain"),
    )
    params = (
        Param("g_A", DIMENSIONLESS, LogNormal(0.5, 10.0), (1e-6, 1e4), description="W per unit of signal"),
        _f_mass(0.5),
    )

    def signals(self) -> list[str]:
        return [self.config["signal"]]

    def active_params(self, mode: str) -> list[str]:
        return ["g_A", *self._base_params()]

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        return self._split(p, Signal(self.config["signal"]), p["g_A"])


class InternalGain(_SplitGain):
    """Internal gains — people, cooking, appliances, standby losses of
    equipment in the heated space — as a smooth daily profile:

        Q_int(t) = Q + Σ_{k=1..harmonics} (a_k cos(2πkt/P) + b_k sin(2πkt/P))

    with period ``P`` (default one day; a week gives a time-of-week profile).
    The harmonic amplitudes have zero-mean normal priors that shrink with
    order, which is the smoothness prior of §4.5.2. ``harmonics = 0`` is a
    constant gain. Time is UTC, so a daylight-saving change shifts a local
    habit by an hour against the profile.
    """

    fields = (
        Integer("harmonics", required=False, default=0, min=0, max=MAX_HARMONICS),
        Quantity("period", dim=TIME, required=False, default=86400.0, positive=True),
    )
    ports = (
        Port("zone", kind="node", heats=True),
        Port("mass", kind="node", required=False, doc="takes f_mass of the gain"),
    )
    params = (
        Param("Q", POWER, LogNormal(300.0, 2.0), (0.1, 1e4), description="mean internal gain"),
        *(
            Param(f"{ab}{k}", POWER, Normal(0.0, 200.0 / k), (-5000.0, 5000.0))
            for k in range(1, MAX_HARMONICS + 1)
            for ab in ("a", "b")
        ),
        _f_mass(0.3),
    )

    def active_params(self, mode: str) -> list[str]:
        h = self.config.get("harmonics") or 0
        return ["Q", *(f"{ab}{k}" for k in range(1, h + 1) for ab in ("a", "b")), *self._base_params()]

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        period = self.config.get("period") or 86400.0
        terms = self._split(p, Constant(), p["Q"])
        for k in range(1, (self.config.get("harmonics") or 0) + 1):
            terms += self._split(p, Harmonic("cos", k, period), p[f"a{k}"])
            terms += self._split(p, Harmonic("sin", k, period), p[f"b{k}"])
        return terms


class HeatInput(_SplitGain):
    """A measured heat input: ``Q = gain · signal`` into ``node`` (spec §6.4:
    "sources enter as measured heat inputs where measured"; §7.5: unplanned
    heat enters as an exogenous input). With the signal in W, ``gain`` is the
    share of it that reaches the node; with a proxy (a flue temperature excess,
    a burner level) it is W per unit of the proxy."""

    target_port = "node"
    fields = (String("signal", doc="the heat rate, or a proxy proportional to it"),)
    ports = (
        Port("node", kind="node", heats=True),
        Port("mass", kind="node", required=False, doc="takes f_mass of the heat"),
    )
    params = (
        Param("gain", DIMENSIONLESS, LogNormal(1.0, 1.5), (1e-6, 1e6), description="W per unit of signal"),
        _f_mass(0.3),
    )

    def signals(self) -> list[str]:
        return [self.config["signal"]]

    def active_params(self, mode: str) -> list[str]:
        return ["gain", *self._base_params()]

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        return self._split(p, Signal(self.config["signal"]), p["gain"])
