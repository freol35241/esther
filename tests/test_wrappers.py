"""Controller wrappers (spec §4.5.6, Appendix A): translations, the devices'
own logic, and replaying that logic against a record — with a wrong rule as
the negative control."""

from __future__ import annotations

import numpy as np
import pytest
import synthetic_tank as st

from esther import HouseError, loads_house
from esther.components.wrappers import (
    IVT490Plant,
    curve_inverse,
    curve_target,
    fake_tank_sensor,
    fit_curve,
    hysteresis,
    replay_score,
)
from esther.core.model import compose

K = 273.15

WRAPPED = (
    st.plant_text()
    + """
[[component]]
type = "CurveControlledHeatSource"
name = "curve"
radiators = "radiators"
slope = 3.0
factor = 0.16
supply_min = "15 °C"
supply_max = "55 °C"
outdoor_min = "-30 °C"
outdoor_max = "30 °C"

[[component]]
type = "ThermostatChargedTank"
name = "dhw_mode"
node = "tank.dhw"
lower = "45 °C"
upper = "52 °C"

[[component]]
type = "ivt490.Plant"
name = "plant"
tank = "tank"
radiators = "radiators"
heat_pump = "hp"
heater = "heater"
curve_slope = 3.0
heater_max = "3 kW"
"""
)


@pytest.fixture(scope="module")
def house():
    return loads_house(WRAPPED)


def test_wrappers_load_and_add_no_dynamics(house):
    m = compose(house)
    assert {"curve", "dhw_mode", "plant"} <= set(m.components)
    assert m.states == compose(loads_house(st.plant_text())).states


def test_wrapper_ports_are_type_checked():
    bad = WRAPPED.replace('tank = "tank"\nradiators', 'tank = "zone"\nradiators')
    with pytest.raises(HouseError, match="expected a JacketTank"):
        loads_house(bad)


def test_curve_round_trip_and_clamp(house):
    c = house["curve"]
    t_out = K + np.array([-10.0, 0.0, 10.0])
    target = c.target(t_out)
    assert target - K == pytest.approx([20 + 0.48 * 30, 20 + 0.48 * 20, 20 + 0.48 * 10])
    assert c.fake_outdoor(target) == pytest.approx(t_out)
    # a target the curve cannot reach inside the accepted range is clamped
    assert c.fake_outdoor(K + 90.0) == pytest.approx(K - 30.0)
    assert c.fake_outdoor(K + 5.0) == pytest.approx(K + 30.0)


def test_reference_curve_matches_its_manual_form(house):
    p = house["plant"]
    # T = 20 + 0.16 k (20 − T_out) + parallel, k = 3
    assert p.supply_target(K - 5.0) - K == pytest.approx(20 + 0.16 * 3 * 25)
    assert p.fake_outdoor(p.supply_target(K + 3.0)) == pytest.approx(K + 3.0)


def test_fit_curve_recovers_slope_and_parallel():
    rng = np.random.default_rng(0)
    t_out = K + rng.uniform(-15, 15, 400)
    target = curve_target(t_out, 2.5, 0.16, K + 20, 1.5) + rng.normal(0, 0.2, 400)
    f = fit_curve(t_out, target, factor=0.16, pivot=K + 20)
    assert f["slope"] == pytest.approx(2.5, abs=0.05)
    assert f["parallel"] == pytest.approx(1.5, abs=0.1)
    assert f["r2"] > 0.99
    with pytest.raises(ValueError, match="flat"):
        curve_inverse(K + 30, 0.0, 0.16, K + 20, 0.0)


def test_fake_tank_sensor_per_mode():
    real = K + np.array([40.0, 50.0, 53.0, 40.0, 50.0, 53.0, 40.0])
    mode = np.array(["auto", "auto", "auto", "block", "block", "boost", "boost"])
    seen = fake_tank_sensor(mode, real, K + 45, K + 52, margin=1.0)
    assert seen - K == pytest.approx([40, 50, 53, 53, 53, 53, 44])
    with pytest.raises(ValueError, match="unknown mode"):
        fake_tank_sensor(np.array(["off"]), real[:1], K + 45, K + 52)


def test_thermostat_hysteresis(house):
    v = K + np.array([50, 46, 45, 48, 51, 52, 50, 44, np.nan, 48])
    on = house["dhw_mode"].charging(v)
    assert on.tolist() == [False, False, True, True, True, False, False, True, False, False]
    assert hysteresis([1.0, 0.0, 1.0], 0.0, 1.0, initial=True).tolist() == [False, True, False]


@pytest.fixture(scope="module")
def record():
    m = st.plant()
    frame, _ = st.simulate_plant(m, st.TRUTH, days=3.0, step=300.0, seed=4, noise=False)
    return frame


def test_replay_reproduces_the_controller_and_a_wrong_rule_does_not(house, record):
    """The synthetic plant's controller: hot water on below 45 °C until 52 °C;
    heating on below the curve target until 10 K above it. The plant
    wrapper's rule with those limits reproduces the record; the same rule
    with the heating demand left out is the negative control."""
    p: IVT490Plant = house["plant"]
    # the synthetic controller's heating differential is 10 K (the plant's default is 2 K)
    p = IVT490Plant(p.name, {**p.config, "heating_differential": 10.0}, p.refs, p.param)
    target = K + 22.0 + 0.6 * (20.0 - (record.t_out - K))
    dem = p.compressor_demand(
        tank_seen=record["x:tank.dhw"],
        tank_lower=K + 45,
        tank_upper=K + 52,
        jacket=record["x:tank.jacket"],
        jacket_target=target + 10.0,
    )
    right = replay_score(dem["compressor"], record.comp)
    wrong = replay_score(dem["hot_water"], record.comp)
    assert right.agreement > 0.97 and right.start_recall > 0.9, right.summary()
    # Agreement alone barely tells them apart (the compressor is mostly on for
    # hot water); the starts do: the wrong rule misses every heating start.
    assert wrong.agreement < right.agreement
    assert wrong.start_recall < right.start_recall - 0.3, wrong.summary()
