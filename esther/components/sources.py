"""Heat sources (spec §4.5.4)."""

from __future__ import annotations

from itertools import pairwise

from ..core.component import Component, Integer, Number, Port, Quantity, String, TableList
from ..core.params import LogNormal, Param
from ..core.units import DIMENSIONLESS, MASS_FLOW, POWER, TIME


class HeatPump(Component):
    """Generic heat pump; fixed-speed as an on-fraction of ``Q_rated``.

    COP is a Carnot fraction ``η T_sink / (T_sink − T_src)`` with ``η`` fitted.
    """

    fields = (
        Quantity("Q_rated", dim=POWER, positive=True, doc="rated heat output"),
        Quantity("min_on", dim=TIME, required=False, default=0.0, min=0.0),
        Quantity("min_off", dim=TIME, required=False, default=0.0, min=0.0),
    )
    ports = (
        Port("sink", kind="node", heats=True),
        Port("source", kind="thermal", doc="where the heat comes from: ambient, extract air, ground"),
    )
    params = (Param("eta", DIMENSIONLESS, LogNormal(0.45, 1.3), (0.1, 0.9), description="Carnot fraction"),)

    def check(self) -> list[tuple[str | None, str]]:
        if self.refs["sink"] == self.refs["source"]:
            return [("source", "a heat pump's source and sink must differ")]
        return []


class ElectricHeater(Component):
    """Immersion or flow heater, COP 1: continuous (``steps = 0``) or ``steps`` equal steps."""

    fields = (
        Quantity("max", dim=POWER, positive=True),
        Integer("steps", required=False, default=0, min=0, max=32),
    )
    ports = (Port("node", kind="node", heats=True),)


class Stove(Component):
    """Automatic-feed stove heating a zone's air; integer level per step."""

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
        Quantity("min_on", dim=TIME, required=False, default=0.0, min=0.0),
        Quantity("min_off", dim=TIME, required=False, default=0.0, min=0.0, doc="t_cool"),
    )
    ports = (Port("zone", kind="node", heats=True),)
    params = (
        Param("t_ign", TIME, LogNormal(900.0, 1.5), (60.0, 7200.0), description="ignition ramp"),
        Param("f_rad", DIMENSIONLESS, LogNormal(0.2, 2.0), (0.001, 0.99), description="share to mass"),
    )

    def check(self) -> list[tuple[str | None, str]]:
        heats = [lv["heat"] for lv in self.config["levels"]]
        if any(b <= a for a, b in pairwise(heats)):
            return [("levels", "levels must be in increasing order of heat output")]
        return []
