"""The tariff (spec §4.5.7): spot in its own currency, converted, plus
per-kWh adders, then VAT; and the planner pricing every bought kWh with it."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic_plan import BASE, forecast, planner, x0

from esther import loads_house
from esther.house import HouseError

KWH = 3.6e6

OLD = """type = "Tariff"
spot_signal = "spot_price"
energy_fee = "0.05 EUR/kWh"
"""


def with_tariff(body: str, pellet_price: float = 0.4):
    text = BASE.format(
        horizon="12h",
        hp_extra="",
        heater_extra="",
        stove_extra="modulating = true",
        pellet_price=pellet_price,
        extra="",
    )
    assert OLD in text
    return loads_house(text.replace(OLD, 'type = "Tariff"\nspot_signal = "spot_price"\n' + body))


CONVERTED = """fx = 10.0
currency = "SEK"
fees = { transfer = "0.2 SEK/kWh", tax = "0.4 SEK/kWh" }
energy_fee = "0.05 SEK/kWh"
vat = 0.25
"""


def test_price_formula():
    h = with_tariff(CONVERTED)
    t = h["tariff"]
    spot = np.array([0.0, 0.10, -0.02]) / KWH  # spot currency per J
    # (10 · spot + 0.05 + 0.2 + 0.4) · 1.25
    expect = (10.0 * np.array([0.0, 0.10, -0.02]) + 0.65) * 1.25
    assert t.price_per_kwh(spot) == pytest.approx(expect)
    assert t.adders() * KWH == pytest.approx(0.65)
    assert t.config["currency"] == "SEK"


def test_defaults_are_the_spot_alone():
    t = with_tariff("")["tariff"]
    assert t.config["fx"] == 1.0 and t.config["vat"] == 0.0
    assert t.price_per_kwh(np.array([0.07]) / KWH) == pytest.approx([0.07])


@pytest.mark.parametrize(
    "body",
    ["fx = 0.0\n", "fx = -1.0\n", 'fees = { tax = "0.4 SEK/kg" }\n', "vat = 1.5\n"],
    ids=["fx zero", "fx negative", "fee not per energy", "vat over 1"],
)
def test_bad_tariffs_refused(body):
    with pytest.raises(HouseError):
        with_tariff(body)


def test_plan_prices_bought_electricity_with_the_tariff():
    """The plan's electricity cost is Σ P_el · price · Δt at the tariff's price
    — the converted spot, the adders and VAT — and nothing earns from export."""
    h = with_tariff(CONVERTED, pellet_price=4.0)
    now = "2026-01-10T00:00Z"
    fc = forecast(now, h, outdoor=0.0, price=lambda hr: 0.05 + 0.10 * (hr >= 16))
    p = planner(h).plan(now, x0(h), fc)
    price = h["tariff"].price_per_kwh(fc.values["spot_price"].to_numpy())
    dt_h = fc.dt / 3600.0
    for label in ("heat_pump", "heater"):
        kwh = p.decisions[f"{label}.P_el"].to_numpy() / 1000.0 * dt_h
        assert p.cost[f"electricity {label}"] == pytest.approx(float((kwh * price).sum()), rel=1e-6, abs=1e-9)
        assert (p.decisions[f"{label}.P_el"] >= -1e-9).all()
    assert all(v >= -1e-9 for k, v in p.cost.items() if k.startswith("electricity"))


def test_fx_moves_the_choice_between_electricity_and_fuel():
    """The same spot at a tenfold exchange rate makes electricity dearer than
    pellets at a fixed local price: the stove takes over (a positive control
    on the conversion reaching the plan)."""
    now = "2026-01-10T06:00Z"
    burnt = {}
    for fx in (1.0, 10.0):
        h = with_tariff(f"fx = {fx}\n", pellet_price=0.4)
        fc = forecast(now, h, outdoor=-5.0, price=0.3)
        p = planner(h).plan(now, x0(h, zone=20.0), fc)
        burnt[fx] = float(p.decisions["stove.fuel"].sum())
    assert burnt[10.0] > burnt[1.0] + 0.1
