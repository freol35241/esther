"""Actuation translators (spec §8.7) and the reference plant's wrapper."""

import itertools

import numpy as np
import pandas as pd
import pytest
from synthetic_plan import WRAPPERS, K, forecast, house, planner, x0

from esther import loads_house
from esther.actuate import Command, stove_schedule, translate
from esther.components.wrappers import IVT490Device, fit_aux_share, fit_response
from esther.forecast import Event
from esther.plan import Planner, PlannerOptions, SolveOptions

NOW = pd.Timestamp("2025-01-15T20:00Z")


class FakePlan:
    def __init__(self, levels, start=NOW, advisory=False):
        self.index = pd.date_range(start, periods=len(levels), freq="15min")
        self.dt = np.full(len(levels), 900.0)
        self.decisions = pd.DataFrame({"stove.level": levels}, index=self.index)
        self.id = "p1"
        self.valid_until = start + pd.Timedelta("30min")
        self.advisory = advisory


def test_stove_schedule_is_absolute_start_and_stop():
    p = FakePlan([0, 0, 2, 2, 3, 3, 0, 0])
    c = stove_schedule(p, "stove", "stove.level")
    kinds = [(x.kind, x.value, x.start) for x in c]
    assert kinds[0] == ("stop", 0, NOW)  # off now: stop now (whatever burns)
    assert kinds[1] == ("start", 2, NOW + pd.Timedelta("30min"))
    assert kinds[2] == ("level", 3, NOW + pd.Timedelta("60min"))
    assert kinds[-1] == ("stop", 0, NOW + pd.Timedelta("90min"))
    assert all(x.plan_id == "p1" and x.expires == p.valid_until for x in c)
    # a burn under way has only its stop; a burn that outlasts the horizon stops at its end
    assert [x.kind for x in stove_schedule(FakePlan([1, 1, 0]), "stove", "stove.level")] == ["stop"]
    last = stove_schedule(FakePlan([0, 1, 1]), "stove", "stove.level")[-1]
    assert last.kind == "stop" and last.start == NOW + pd.Timedelta("45min")
    assert [x.kind for x in stove_schedule(FakePlan([0, 0, 0]), "stove", "stove.level")] == ["stop"]
    # a burn the plan starts in its first step is a start now, not a burn under way
    p = FakePlan([2, 2, 0])
    p.decisions["stove.start"] = [1.0, 0.0, 0.0]
    c = stove_schedule(p, "stove", "stove.level", "stove.start")
    assert [(x.kind, x.value, x.start) for x in c] == [
        ("start", 2, NOW),
        ("stop", 0, NOW + pd.Timedelta("30min")),
    ]


def test_generic_wrappers_translate():
    h = house(extra=WRAPPERS)
    plan = planner(h).plan(NOW, x0(h, zone=19.0, tank=46.0), forecast(NOW, h, outdoor=-5.0))
    cmds = translate(plan, h)
    out = [c for c in cmds if c.device == "curve"]
    assert len(out) == len(plan.index)
    curve = h["curve"]
    k = 0
    assert out[k].value == pytest.approx(
        float(curve.fake_outdoor(plan.decisions["radiators.supply"].iloc[k]))
    )
    modes = [c for c in cmds if c.device == "thermostat"]
    on = plan.decisions["heat_pump.delta"].to_numpy() > 1e-6
    assert [m.value for m in modes] == ["boost" if o else "block" for o in on]
    assert all(c.actionable for c in cmds)


IVT = """
schema = 1
name = "ivt-test"
step = "15min"
horizon = "12h"

[[component]]
type = "Ambient"
name = "outdoor"
signal = "outdoor_temperature"

[[component]]
type = "Zone"
name = "house"
prior.C = { value = "12 MJ/K", fixed = true }

[[component]]
type = "Coupling"
name = "envelope"
between = ["house", "outdoor"]
prior.UA = { value = "60 W/K", fixed = true }

[[component]]
type = "JacketTank"
name = "tank"
volume_jacket = "55 l"
volume_dhw = "145 l"
room = "house"
prior.f_C = { value = 0.6, fixed = true }
prior.UA_jd = { value = "3000 W/K", fixed = true }

[[component]]
type = "Radiators"
name = "radiators"
zone = "house"
source = "tank.jacket"
supply_signal = "supply_temperature"
prior.UA_rad = { value = "65 W/K", fixed = true }

[[component]]
type = "HeatPump"
name = "heat_pump"
sink = "tank.jacket"
source = "house"
Q_rated = "0.9 kW"
P_el = "0.52 kW"
on_signal = "compressor"

[[component]]
type = "ElectricHeater"
name = "heater"
node = "tank.jacket"
max = "2.2 kW"
steps = 1
commandable = false
on_signal = "heater_share"

[[component]]
type = "ivt490.Plant"
name = "plant"
tank = "tank"
radiators = "radiators"
heat_pump = "heat_pump"
heater = "heater"
outdoor = "outdoor"
curve_slope = 3.0
heater_max = "2.2 kW"
aux_share = 0.05
aux_cold_slope = 0.01
aux_boost = 0.04

[[component]]
type = "Tariff"
spot_signal = "spot_price"

[[component]]
type = "ComfortBand"
zone = "house"
setpoint = "20 °C"
band = "0.5 K"
floor = "16 °C"

[[component]]
type = "DhwService"
node = "tank.dhw"
min_temp = "45 °C"
"""


def ivt_plan(events=(), tank=50.0, outdoor=0.0):
    h = loads_house(IVT)
    pl = Planner(h, {}, PlannerOptions(solve=SolveOptions(time_limit=10, gap=0.01)))
    fc = forecast(NOW, h, outdoor=outdoor, price=lambda hr: np.where((hr >= 0) & (hr < 6), 0.02, 0.4))
    fc.events.extend(events)
    x = {s: (tank if s.startswith("tank") else 20.0) + K for s in pl.model.states}
    return h, pl.plan(NOW, x, fc)


def test_ivt_plan_heater_is_the_controllers():
    h, plan = ivt_plan()
    d = plan.decisions
    alpha = h["plant"].aux_alpha(K + 0.0)
    assert np.allclose(d["heater.share"], alpha * d["heat_pump.delta"], atol=1e-6)
    tr = plan.trajectories
    on = d["heat_pump.on"].to_numpy() > 0.5
    assert (tr["tank.dhw"].to_numpy()[1:][on] <= h["plant"].config["tank_upper"] + 1e-4).all()
    modes = d["plant.mode"].to_numpy()
    assert set(np.unique(modes)) <= {1.0, 2.0}  # block / boost; auto only in events
    assert ((modes == 2.0) == (d["heat_pump.delta"].to_numpy() > 1e-6)).all()


def test_ivt_event_lets_the_device_do_it():
    ev = Event("plant", "legionella", NOW + pd.Timedelta("2h"), NOW + pd.Timedelta("3h"))
    _h, plan = ivt_plan(events=[ev])
    d = plan.decisions
    inside = (d.index >= ev.start) & (d.index < ev.end)
    assert (d["plant.mode"][inside] == 0.0).all()  # auto
    assert np.allclose(d["heat_pump.delta"][inside], 0.0)
    assert np.allclose(d["heater.share"][inside], 1.0)


def test_ivt_actuate_boost_then_block():
    h, plan = ivt_plan()
    cmds = h["plant"].actuate(plan)
    d = plan.decisions
    k = (
        int(np.argmax((d["heat_pump.delta"].to_numpy() > 0.05) & (d["heat_pump.delta"].to_numpy() < 0.95)))
        if ((d["heat_pump.delta"] > 0.05) & (d["heat_pump.delta"] < 0.95)).any()
        else None
    )
    first = [
        c
        for c in cmds
        if c.kind == "mode" and c.start < plan.index[0] + pd.Timedelta(seconds=float(plan.dt[0]))
    ]
    assert first and first[0].start == plan.index[0]
    if k is not None:
        t = plan.index[k]
        seg = [c for c in cmds if c.kind == "mode" and t <= c.start < t + pd.Timedelta("15min")]
        assert [c.value for c in seg] == ["boost", "block"]
        frac = (seg[0].end - seg[0].start).total_seconds() / 900
        assert frac == pytest.approx(d["heat_pump.delta"].iloc[k], abs=1e-6)
    sup = [c for c in cmds if c.kind == "supply_target"]
    fake = [c for c in cmds if c.kind == "fake_outdoor"]
    assert fake[0].value == pytest.approx(float(h["plant"].fake_outdoor(sup[0].value)))
    assert all(isinstance(c, Command) for c in cmds)


def test_ivt_device_two_demands():
    h = loads_house(IVT)
    dev = IVT490Device(h["plant"])
    c = h["plant"].config
    lo, hi = c["tank_lower"], c["tank_upper"]
    # BLOCK, jacket well above the target: off
    assert dev.step("block", K + 30, lo - 3, K + 50, K)["compressor"] == 0.0
    # BOOST below the stop limit: on, and the heater comes with the expected share, more under boost
    r = dev.step("boost", K + 30, hi - 2, K + 50, K)
    assert r["compressor"] == 1.0 and r["heater_share"] == pytest.approx(h["plant"].aux_alpha(K, boost=True))
    # heating demand under BLOCK: the jacket below the target less its differential
    dev.reset()
    assert dev.step("block", K + 45, lo + 3, K + 40, K)["compressor"] == 1.0
    # the condensing limit stops it whatever the demand
    dev.reset()
    assert dev.step("boost", K + 30, lo - 3, K + 61, K)["compressor"] == 0.0
    # the shunt mixes down: supply never above the jacket
    assert dev.step("auto", K + 50, K + 50, K + 40, K)["supply_temperature"] == K + 40


def test_fit_aux_share_recovers_known_coefficients():
    rng = np.random.default_rng(3)
    n = 5000
    d = (rng.uniform(size=n) > 0.5).astype(float)
    t = K + rng.uniform(-5, 15, n)
    b = (rng.uniform(size=n) > 0.8).astype(float)
    h = d * (0.04 + 0.01 * np.maximum(0, K + 8 - t) + 0.05 * b) + rng.normal(0, 0.02, n)
    f = fit_aux_share(d, h, t, b, cold_below=K + 8)
    assert f["aux_share"] == pytest.approx(0.04, abs=0.005)
    assert f["aux_cold_slope"] == pytest.approx(0.01, abs=0.001)
    assert f["aux_boost"] == pytest.approx(0.05, abs=0.005)


def test_fit_response_is_monotone():
    x = np.linspace(0, 10, 500)
    y = np.clip(x / 10 + np.sin(x) * 0.1, 0, 1)
    knots = fit_response(x, y, np.arange(0, 11, 1.0))
    ys = [k[1] for k in knots]
    assert all(b >= a for a, b in itertools.pairwise(ys))
