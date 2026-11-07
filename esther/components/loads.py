"""Loads (spec §4.5.5)."""

from __future__ import annotations

from collections.abc import Mapping

from ..core.component import Component, Port, Quantity, String
from ..core.model import Gain, Signal
from ..core.units import TEMPERATURE


class HotWaterDraw(Component):
    """Energy drawn from a DHW node, ``Q_draw = ṁ c_p (T_dhw − T_mains)``.

    ``signal`` carries the draw as a heat rate in W per step, removed from
    ``node``. In identification it is the output of draw detection (§6.3,
    :mod:`esther.identify.draws`) — no flow meter is assumed; in planning, a
    forecast (§7.4). Without a signal the component contributes nothing.

    For a :class:`~esther.components.hydronic.StratifiedTank` the draw can
    instead be written as advection (its ``draw_signal``), which also moves
    the layers.
    """

    fields = (
        Quantity(
            "mains_temperature",
            dim=TEMPERATURE,
            absolute=True,
            required=False,
            default=283.15,
            min=273.15,
            max=313.15,
        ),
        String("signal", required=False, doc="draw heat rate, W"),
    )
    ports = (Port("node", kind="node"),)

    def signals(self) -> list[str]:
        return [self.config["signal"]] if self.config.get("signal") else []

    def dynamics(self, p: Mapping[str, float], mode: str) -> list:
        sig = self.config.get("signal")
        return [Gain(self.refs["node"], Signal(sig), -1.0)] if sig else []
