"""A small synthetic house for the planner tests: one zone, a one-store tank
(jacket and vessel tightly coupled) with a heat pump and an electric heater,
radiators mixing from the tank, a pellet stove, a tariff, comfort and
hot-water service. Round made-up numbers; not any real house."""

from __future__ import annotations

import numpy as np
import pandas as pd

from esther import loads_house
from esther.forecast import Forecast, plan_grid
from esther.plan import Planner, PlannerOptions, SolveOptions

K = 273.15

BASE = """
schema = 1
name = "plan-test"
step = "15min"
horizon = "{horizon}"

[[component]]
type = "Ambient"
name = "outdoor"
signal = "outdoor_temperature"

[[component]]
type = "Zone"
name = "living"
signal = "living_temperature"
prior.C = {{ value = "10 MJ/K", fixed = true }}

[[component]]
type = "Coupling"
name = "envelope"
between = ["living", "outdoor"]
prior.UA = {{ value = "80 W/K", fixed = true }}

[[component]]
type = "JacketTank"
name = "tank"
volume_jacket = "50 l"
volume_dhw = "150 l"
room = "living"
prior.f_C = {{ value = 1.0, fixed = true }}
prior.UA_jd = {{ value = "2000 W/K", fixed = true }}
prior.UA_loss = {{ value = "2 W/K", fixed = true }}

[[component]]
type = "Radiators"
name = "radiators"
zone = "living"
source = "tank.jacket"
supply_signal = "supply_temperature"
prior.UA_rad = {{ value = "150 W/K", fixed = true }}

[[component]]
type = "HeatPump"
name = "heat_pump"
sink = "tank.jacket"
source = "living"
Q_rated = "2 kW"
P_el = "0.6 kW"
on_signal = "compressor"
{hp_extra}

[[component]]
type = "ElectricHeater"
name = "heater"
node = "tank.jacket"
max = "3 kW"
on_signal = "heater_share"
{heater_extra}

[[component]]
type = "Stove"
name = "stove"
zone = "living"
levels = [ {{ heat = "1 kW", fuel = "0.25 kg/h" }},
           {{ heat = "4 kW", fuel = "0.9 kg/h" }} ]
efficiency = 0.85
fuel = "pellets"
heat_signal = "stove_heat"
min_on = "1 h"
min_off = "1 h"
allowed_from = 6
allowed_to = 22
{stove_extra}

[[component]]
type = "HotWaterDraw"
name = "draws"
node = "tank.dhw"
signal = "draw_heat"

[[component]]
type = "Tariff"
spot_signal = "spot_price"
energy_fee = "0.05 EUR/kWh"
fuel_price = {{ pellets = "{pellet_price} EUR/kg" }}

[[component]]
type = "ComfortBand"
zone = "living"
setpoint = "20 °C"
band = "0.5 K"
floor = "16 °C"
weight = 2.0

[[component]]
type = "DhwService"
node = "tank.dhw"
min_temp = "45 °C"
weight = 5.0
draw_signal = "draw_heat"
{extra}
"""


def house_text(
    horizon="12h", hp_extra="", heater_extra="", stove_extra="modulating = true", pellet_price=0.4, extra=""
) -> str:
    return BASE.format(
        horizon=horizon,
        hp_extra=hp_extra,
        heater_extra=heater_extra,
        stove_extra=stove_extra,
        pellet_price=pellet_price,
        extra=extra,
    )


def house(**kw):
    return loads_house(house_text(**kw))


def forecast(now, h, outdoor=0.0, price=None, draws=None, extra=None) -> Forecast:
    """A forecast for ``house`` ``h`` made at ``now``: outdoor °C (scalar or
    function of the hour), spot price €/kWh (function of the hour), draws W."""
    now = pd.Timestamp(now)
    idx, dt = plan_grid(now, h.horizon, h.step)
    hours = np.asarray((idx.hour + idx.minute / 60).to_numpy(), dtype=float)
    out = outdoor(hours) if callable(outdoor) else np.full(len(idx), float(outdoor))
    pr = price(hours) if callable(price) else np.full(len(idx), 0.10 if price is None else float(price))
    dr = draws(hours) if callable(draws) else np.zeros(len(idx))
    vals = pd.DataFrame(
        {"outdoor_temperature": out + K, "spot_price": pr / 3.6e6, "draw_heat": dr},
        index=idx,
    )
    for k, v in (extra or {}).items():
        vals[k] = v(hours) if callable(v) else v
    est = pd.DataFrame(False, index=idx, columns=vals.columns)
    return Forecast(now, dt, vals, est, [], [])


def planner(h, accepted=True, **kw) -> Planner:
    solve = kw.pop("solve", SolveOptions(time_limit=5.0, gap=0.03))
    return Planner(h, {}, PlannerOptions(solve=solve, **kw), accepted=accepted)


def x0(h, zone=20.0, tank=50.0):
    p = Planner(h, {})
    out = {}
    for s in p.model.states:
        out[s] = (tank if s.startswith("tank") else zone) + K
    return out


WRAPPERS = """
[[component]]
type = "CurveControlledHeatSource"
name = "curve"
radiators = "radiators"
slope = 0.6

[[component]]
type = "ThermostatChargedTank"
name = "thermostat"
node = "tank.dhw"
source = "heat_pump"
lower = "48 °C"
upper = "53 °C"
"""
