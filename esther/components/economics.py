"""Economics and comfort (spec §4.5.7) and their planning terms (§8.4–8.5)."""

from __future__ import annotations

import numpy as np

from ..core.component import Component, Mapping_, Number, Port, Quantity, String
from ..core.units import PRICE_PER_ENERGY, PRICE_PER_MASS, TEMPERATURE

T0 = 273.15
KWH = 3.6e6


class Tariff(Component):
    """What a bought kWh and a kg of fuel cost, in the house's one currency.

    Electricity per imported kWh::

        (fx · spot + energy_fee + Σ fees) · (1 + vat)

    - ``spot`` is the spot-price signal, in the currency the market quotes
      it in (often not the house's);
    - ``fx`` converts it to the house's currency (house currency per unit of
      the spot's; 1 when they are the same). It is a fixed rate: state its
      date and source where the house file sets it;
    - ``energy_fee`` and ``fees`` (name → price per energy) are the per-kWh
      adders — network transfer, energy tax, a supplier's markup — in the
      house's currency, **before** VAT;
    - ``vat`` applies to the spot and the adders alike.

    Fixed periodic fees change no decision and are not modelled; nor is a
    power (peak) charge. ``currency`` names the house's currency for reports.

    *Export:* none is valued. The planner plans every source's electricity as
    a bought kWh at this price, and nothing in a plan earns from export; a
    house with a solar surplus is therefore priced conservatively (its own
    surplus would cost it only the export price it forgoes). Fuel: price per
    kg, in the house's currency.

    *Planning:* every source's electricity is priced at the step's price
    (currency/kWh), its fuel at the fuel's price; the tariff contributes last.
    """

    plan_order = 90
    fields = (
        String("spot_signal", doc="the spot-price signal, the spot's currency per energy"),
        Number(
            "fx",
            required=False,
            default=1.0,
            min=0.0,
            min_exclusive=True,
            doc="house currency per spot currency",
        ),
        Quantity("energy_fee", dim=PRICE_PER_ENERGY, required=False, default=0.0, min=0.0),
        Mapping_(
            "fees",
            required=False,
            default={},
            value_field=Quantity("fee", dim=PRICE_PER_ENERGY),
        ),
        Number("vat", required=False, default=0.0, min=0.0, max=1.0),
        String("currency", required=False, doc="the house's currency, for reports"),
        Mapping_(
            "fuel_price",
            required=False,
            default={},
            value_field=Quantity("price", dim=PRICE_PER_MASS, min=0.0),
        ),
    )

    def adders(self) -> float:
        """The per-energy adders before VAT, house currency per J."""
        c = self.config
        return float(c.get("energy_fee", 0.0)) + float(sum((c.get("fees") or {}).values()))

    def price_per_kwh(self, spot_si) -> np.ndarray:
        """House currency per bought kWh from the spot signal in SI (the
        spot's currency per J)."""
        c = self.config
        spot = np.asarray(spot_si, dtype=float) * c.get("fx", 1.0)
        return (spot + self.adders()) * KWH * (1.0 + c.get("vat", 0.0))

    def signals(self) -> list[str]:
        return [self.config["spot_signal"]]

    def plan_prepare(self, ctx) -> None:
        ctx.shared["electricity_price"] = self.price_per_kwh(ctx.signal(self.config["spot_signal"]))
        ctx.shared["price_estimated"] = ctx.estimated(self.config["spot_signal"])
        ctx.shared["fuel_price"] = dict(self.config.get("fuel_price") or {})

    def plan(self, ctx) -> None:
        price = ctx.shared["electricity_price"]
        for label, expr in ctx.electric.items():
            ctx.add_cost(f"electricity {label}", expr * (price * ctx.dt_h))
        fp = ctx.shared["fuel_price"]
        for label, (expr, fuel) in ctx.fuel.items():
            if fuel not in fp:
                from ..plan.planner import PlanError

                raise PlanError("missing inputs", f"no price for fuel {fuel!r} ({label})")
            ctx.add_cost(f"fuel {label}", expr * (fp[fuel] * ctx.dt_h))


class ComfortBand(Component):
    """Per zone: setpoint and band (soft), and a hard floor that is never planned below.

    *Planning:* slack below and above the band at each step's end, priced at
    ``weight`` per K·h; the hard floor. The setpoint is fixed or a signal
    (``setpoint_signal``: a time-varying setpoint, forecast by the
    integration). From wherever the zone is at plan time, the band and the
    floor are widened to include it and close at ``recovery`` K/h — the
    problem must be feasible from any state (§8.3).
    """

    plan_order = 30
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
        Number(
            "weight_above",
            required=False,
            min=0.0,
            doc="penalty above the band, currency per K·h (default: weight)",
        ),
        String("setpoint_signal", required=False, doc="time-varying setpoint, K"),
        Quantity("recovery", dim=TEMPERATURE, required=False, default=1.0, min=0.0, doc="K per hour"),
    )
    ports = (Port("zone", kind="node"),)

    def check(self) -> list[tuple[str | None, str]]:
        c = self.config
        if c["floor"] > c["setpoint"] - c["band"]:
            return [("floor", "the hard floor must lie at or below the lower edge of the band")]
        return []

    def setpoints(self, ctx) -> np.ndarray:
        c = self.config
        if c.get("setpoint_signal"):
            return ctx.signal(c["setpoint_signal"])
        return np.full(ctx.n, c["setpoint"])

    def plan(self, ctx) -> None:
        from ..plan.blocks import Linear

        c, n, zone = self.config, ctx.n, self.refs["zone"]
        X = ctx.state(zone)[1:]
        sp = self.setpoints(ctx)
        lo, hi = sp - c["band"], sp + c["band"]
        z0 = ctx.x0_of(zone)
        rec = c.get("recovery", 1.0) * ctx.times / 3600.0
        lo = np.minimum(lo, z0 + rec)
        hi = np.maximum(hi, z0 - rec)
        floor = np.minimum(np.minimum(c["floor"], lo), z0 - 0.5 + rec)
        lo_s = ctx.milp.var(f"{self.name}.below", n, 0.0)
        hi_s = ctx.milp.var(f"{self.name}.above", n, 0.0)
        ctx.milp.add([(X, 1.0), (lo_s, 1.0)], lb=lo - T0, name=f"{self.name} band low")
        ctx.milp.add([(X, 1.0), (hi_s, -1.0)], ub=hi - T0, name=f"{self.name} band high")
        ctx.milp.bound(X, lb=floor - T0)
        w = c.get("weight", 1.0) * ctx.dt_h
        wa = (c.get("weight_above") if c.get("weight_above") is not None else c.get("weight", 1.0)) * ctx.dt_h
        ctx.add_cost(f"comfort {zone}", Linear.of(lo_s, w) + Linear.of(hi_s, wa))
        f = float(np.mean(sp - c["band"]))
        ctx.add_floor(zone, f)
        for comp in ctx.house.components.values():
            if comp.type_name == "Mass" and comp.refs.get("zone") == zone and ctx.has_state(comp.name):
                ctx.add_floor(comp.name, f)
        ctx.add_result(f"{self.name}.low", lambda x: lo)
        ctx.add_result(f"{self.name}.high", lambda x: hi)
        ctx.add_result(f"{self.name}.below", lambda x: x[lo_s])
        ctx.add_result(f"{self.name}.above", lambda x: x[hi_s])


class DhwService(Component):
    """Minimum tap temperature on a DHW node, soft with a large penalty.

    *Planning:* slack below ``min_temp`` at the **start** of each step a
    draw is forecast (``draw_signal`` nonzero; every step without one) —
    the temperature at the tap when the draw begins, not after it has
    cooled the node — priced at ``weight`` per K·h (§8.4).
    """

    plan_order = 30
    fields = (
        Quantity("min_temp", dim=TEMPERATURE, absolute=True, min=283.15, max=363.15),
        Number("weight", required=False, default=10.0, min=0.0, min_exclusive=True, doc="currency per K·h"),
        String("draw_signal", required=False, doc="the draw forecast; service is due where it is nonzero"),
    )
    ports = (Port("node", kind="node"),)

    def due(self, ctx) -> np.ndarray:
        sig = self.config.get("draw_signal")
        if not sig:
            return np.ones(ctx.n, bool)
        return np.abs(ctx.signal(sig, default=0.0)) > 1e-6

    def plan(self, ctx) -> None:
        from ..plan.blocks import Linear

        c, n, node = self.config, ctx.n, self.refs["node"]
        X = ctx.state(node)[:-1]
        due = self.due(ctx)
        s = ctx.milp.var(f"{self.name}.below", n, 0.0)
        ctx.milp.fix(s[~due], 0.0) if (~due).any() else None
        k = np.flatnonzero(due)
        if len(k):
            ctx.milp.add([(X[k], 1.0), (s[k], 1.0)], lb=c["min_temp"] - T0, name=f"{self.name} service")
        ctx.add_cost(f"hot water {node}", Linear.of(s, c.get("weight", 10.0) * ctx.dt_h))
        owner = next((cp for cp in ctx.house.components.values() if node in cp.nodes()), None)
        for nd in owner.nodes() if owner is not None else [node]:
            if ctx.has_state(nd):
                ctx.add_floor(nd, c["min_temp"])
        ctx.add_result(f"{self.name}.below", lambda x: x[s])
        ctx.add_result(f"{self.name}.due", lambda x: due.astype(float))
