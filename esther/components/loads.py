"""Loads (spec §4.5.5)."""

from __future__ import annotations

from ..core.component import Component, Port, Quantity
from ..core.units import TEMPERATURE


class HotWaterDraw(Component):
    """Energy drawn from a DHW node, ``ṁ c_p (T_dhw − T_mains)``; forecast in planning."""

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
    )
    ports = (Port("node", kind="node"),)
