"""Economics and comfort (spec §4.5.7)."""

from __future__ import annotations

from ..core.component import Component, Mapping_, Number, Port, Quantity, String
from ..core.units import PRICE_PER_ENERGY, PRICE_PER_MASS, TEMPERATURE


class Tariff(Component):
    """Electricity: spot price per step + fixed per-kWh fees, then VAT. Fuel: price per kg."""

    fields = (
        String("spot_signal", doc="the spot-price signal, currency/kWh"),
        Quantity("energy_fee", dim=PRICE_PER_ENERGY, required=False, default=0.0, min=0.0),
        Number("vat", required=False, default=0.0, min=0.0, max=1.0),
        Mapping_(
            "fuel_price",
            required=False,
            default={},
            value_field=Quantity("price", dim=PRICE_PER_MASS, min=0.0),
        ),
    )

    def signals(self) -> list[str]:
        return [self.config["spot_signal"]]


class ComfortBand(Component):
    """Per zone: setpoint and band (soft), and a hard floor that is never planned below."""

    fields = (
        Quantity("setpoint", dim=TEMPERATURE, absolute=True, min=273.15, max=313.15),
        Quantity("band", dim=TEMPERATURE, min=0.0, max=10.0, doc="half-width, K"),
        Quantity("floor", dim=TEMPERATURE, absolute=True, min=273.15, max=313.15),
        Number(
            "weight",
            required=False,
            default=1.0,
            min=0.0,
            min_exclusive=True,
            doc="penalty, currency per K·h outside the band",
        ),
    )
    ports = (Port("zone", kind="node"),)

    def check(self) -> list[tuple[str | None, str]]:
        c = self.config
        if c["floor"] > c["setpoint"] - c["band"]:
            return [("floor", "the hard floor must lie at or below the lower edge of the band")]
        return []


class DhwService(Component):
    """Minimum tap temperature on a DHW node, soft with a large penalty."""

    fields = (Quantity("min_temp", dim=TEMPERATURE, absolute=True, min=283.15, max=363.15),)
    ports = (Port("node", kind="node"),)
