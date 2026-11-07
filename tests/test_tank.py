"""Tanks and sources (spec §4.5.3–4.5.4): structure, synthetic recovery
(§10.1) and the scale anchor (§6.2).

The plant is a zone heated by radiators that mix from a double-jacket tank,
charged by a heat pump and an electric heater, with hot water drawn from the
inner vessel (``synthetic_tank``, simulated in closed loop). Two anchors can
fix the model's watts — the heater's known power and the tank's known water
volume — and each is tested alone, with the negative control that without
either the scale is reported as not identified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import synthetic_tank as st

from esther import HouseError, loads_house
from esther.components.hydronic import RHO_CP_WATER
from esther.core.model import Constant, Signal, compose
from esther.identify import FitOptions, IdData, fit, simulate
from esther.identify.fit import Objective
from esther.identify.kalman import prepare

QUICK = FitOptions(n_starts=1)
TOL = 0.15
NOISE = ("sigma_w", "sigma_v", "sigma_w_jacket", "sigma_w_dhw", "sigma_v_jacket", "sigma_v_dhw")


def is_noise(name: str) -> bool:
    return name.split(".", 1)[1] in NOISE


# -- structure ----------------------------------------------------------------


def test_jacket_tank_capacities_come_from_the_volumes():
    m = st.plant()
    tank = m.components["tank"]
    caps = tank.capacities({"f_C": 1.0, "f_top": 0.3})
    assert caps["tank.jacket"] == pytest.approx(0.060 * RHO_CP_WATER)
    assert caps["tank.dhw"] == pytest.approx(0.180 * RHO_CP_WATER)
    assert "tank.top" not in caps


def test_two_layer_inner_vessel_splits_its_volume():
    text = st.plant_text().replace(
        'mixing_signal = "mix"', 'mixing_signal = "mix"\ndhw_layers = 2\ntop_signal = "t_top"'
    )
    m = compose(loads_house(text))
    assert {"tank.top", "tank.dhw"} <= set(m.states)
    assert {"tank.f_top", "tank.UA_top", "tank.UA_top_mix", "tank.sigma_v_top"} <= set(m.param_names)
    caps = m.components["tank"].capacities({"f_C": 1.0, "f_top": 0.25})
    assert caps["tank.top"] == pytest.approx(0.25 * 0.180 * RHO_CP_WATER)
    assert caps["tank.top"] + caps["tank.dhw"] == pytest.approx(0.180 * RHO_CP_WATER)


def test_top_sensor_needs_two_layers():
    text = st.plant_text().replace('mixing_signal = "mix"', 'mixing_signal = "mix"\ntop_signal = "t_top"')
    with pytest.raises(HouseError, match="dhw_layers = 2"):
        loads_house(text)


def test_sources_need_their_signal_in_identification_and_are_decisions_in_planning():
    text = st.plant_text().replace('on_signal = "comp"\n', "")
    with pytest.raises(Exception, match="on_signal"):
        compose(loads_house(text))
    m = compose(loads_house(st.plant_text()), mode="plan")
    assert {"hp.Q", "heater.Q", "radiators.Q"} <= set(m.decisions)


def test_heat_pump_prior_is_centred_on_its_rating():
    m = st.plant()
    hp = m.components["hp"]
    assert hp.param["Q"].prior.median == pytest.approx(2500.0)
    assert hp.cop({"Q": 1800.0}) == pytest.approx(3.0)
    # a house prior wins over the rating
    text = st.plant_text(hp_extra='prior.Q = { median = "1 kW", gsd = 2.0 }')
    assert compose(loads_house(text)).components["hp"].param["Q"].prior.median == pytest.approx(1000.0)


def test_heat_pump_output_falls_with_its_sink():
    text = st.plant_text(hp_extra='sink_reference = "50 °C"')
    m = compose(loads_house(text))
    assert "hp.dQ_dT" in m.param_names
    th = {n: st.TRUTH[n] for n in m.param_names}
    M = m.matrices(th)
    j = m.states.index("tank.jacket")
    k = m.modulations.index("comp") + 1
    C = m.components["tank"].capacities({"f_C": 1.0})["tank.jacket"]
    one = m.inputs.index(Constant())
    # running, the output is Q + dQ_dT (T_ref − T_jacket): at 60 °C, 10 K above the reference
    q = C * (M.A[k, j, j] * (273.15 + 60) + M.E[k, j, one]) + C * M.E[0, j, m.inputs.index(Signal("comp"))]
    assert q == pytest.approx(1800.0 - 15.0 * 10.0)
    assert m.components["hp"].heat_output({"Q": 1800.0, "dQ_dT": 15.0}, 273.15 + 60) == pytest.approx(1650.0)


# -- recovery -----------------------------------------------------------------


@pytest.fixture(scope="module")
def plant_data():
    m = st.plant()
    frame, _ = st.simulate_plant(m, st.TRUTH, days=4.0, step=600.0, seed=1)
    return frame


def check(model, res, truth, tol=TOL):
    tab = res.table()
    tab["true"] = [truth[n] for n in tab.index]
    tab["lo99"], tab["hi99"] = res.interval(0.99)
    inside = (tab.lo99 <= tab.true) & (tab.true <= tab.hi99)
    assert inside.all(), f"true value outside the 99 % interval:\n{tab[~inside]}"
    assert not tab.on_bound.any()
    phys = [n for n in tab.index if not is_noise(n)]
    rel = (tab.loc[phys, "value"] / tab.loc[phys, "true"] - 1).abs()
    assert (rel < tol).all(), f"relative error above {tol}:\n{rel[rel >= tol]}"
    return tab


def test_recovers_tank_sources_and_envelope_jointly(plant_data):
    m = st.plant()
    res = fit(m, st.iddata(m, plant_data, step=600.0), QUICK)
    tab = check(m, res, st.TRUTH)
    assert tab.identified.all(), tab[~tab.identified]
    # the derived COP: heat output over the configured electrical input
    assert m.components["hp"].cop({"Q": res.params["hp.Q"]}) == pytest.approx(3.0, rel=0.05)


def test_recovers_the_heat_pumps_sink_dependence():
    text = st.plant_text(hp_extra='sink_reference = "50 °C"')
    m = compose(loads_house(text))
    frame, _ = st.simulate_plant(m, st.TRUTH, days=4.0, step=600.0, seed=2)
    res = fit(m, st.iddata(m, frame, step=600.0), QUICK)
    check(m, res, st.TRUTH, tol=0.3)


# -- the scale anchor (§6.2) ----------------------------------------------------
#
# Any heat input of known size anchors the scale — a measured draw is one too
# (its heat is ``ṁ c_p ΔT``, entered with gain −1), so these tests use a
# plant with no draws: the anchors are only the heater and the volume.


FREE_MIX = 'prior.UA_mix = { median = "200 W/K", gsd = 3.0 }\n'


@pytest.fixture(scope="module")
def dry_data():
    frame, _ = st.simulate_plant(st.plant(), st.TRUTH, days=4.0, step=600.0, seed=1, draws_per_day=0.0)
    return frame


def test_volume_alone_anchors_the_scale(dry_data):
    """Heater power unknown (wide prior), tank volume trusted: the heater's
    power is then *measured* by the tank, and the envelope's watts with it."""
    m = st.plant(
        tank_priors=FREE_MIX + "prior.f_C = { value = 1.0, fixed = true }",
        heater_prior='prior.Q = { median = "1.5 kW", gsd = 3.0 }',
    )
    res = fit(m, st.iddata(m, dry_data, step=600.0), QUICK)
    tab = check(m, res, st.TRUTH)
    assert tab.loc[["heater.Q", "env.UA", "hp.Q"], "identified"].all()
    assert not res.weak_directions


def test_without_an_anchor_the_scale_is_not_identified(dry_data):
    """Negative control: heater power and tank capacity both free with wide
    priors. Every capacity, conductance and heat input can then scale
    together; the fit must say so, not report watts."""
    m = st.plant(
        tank_priors=FREE_MIX + "prior.f_C = { median = 0.7, gsd = 3.0 }",
        heater_prior='prior.Q = { median = "1.5 kW", gsd = 3.0 }',
    )
    res = fit(m, st.iddata(m, dry_data, step=600.0), QUICK)
    tab = res.table()
    assert not tab.loc[["env.UA", "heater.Q", "hp.Q", "tank.f_C"], "identified"].any()
    loads = [set(d) for _, d in res.weak_directions]
    assert any({"tank.f_C", "heater.Q"} <= ld for ld in loads), res.weak_directions
    # ratios are still identified: the heater against the heat pump
    U = np.random.default_rng(0).multivariate_normal(res.u, res.cov_u, 2000)
    obj = Objective(m, prepare(m, st.iddata(m, dry_data, step=600.0)), res.options)
    T = obj.to_theta(U)
    i, j = res.names.index("hp.Q"), res.names.index("heater.Q")
    ratio = T[:, i] / T[:, j]
    assert np.std(np.log(ratio)) < 0.05
    assert np.median(ratio) == pytest.approx(1800 / 3000, rel=0.05)


# -- the stratified tank ----------------------------------------------------------

STRATIFIED = """\
schema = 1
name = "cylinder"

[[component]]
type = "Ambient"
name = "room"
signal = "t_room"

[[component]]
type = "StratifiedTank"
name = "cyl"
volume = "200 l"
room = "room"
layers = 3
sensors = { top = "t_top", bottom = "t_bot" }
draw_signal = "flow"
draw_flow_max = "12 l/min"
mains_temperature = "10 °C"
prior.f_C = { value = 1.0, fixed = true }
prior.sigma_w = { value = "0.02 K", fixed = true }

[[component]]
type = "ElectricHeater"
name = "element"
node = "cyl.bottom"
max = "3 kW"
on_signal = "heat"
prior.Q = { value = "3 kW", fixed = true }
"""


def test_stratified_tank_advection_conserves_energy():
    m = compose(loads_house(STRATIFIED))
    M = m.matrices(m.prior_centre())
    caps = np.full(3, RHO_CP_WATER * 0.2 / 3)
    k = m.modulations.index("flow") + 1
    G = RHO_CP_WATER * 12e-3 / 60
    x = np.array([333.15, 323.15, 300.0])  # top, middle, bottom
    order = [m.states.index(f"cyl.{n}") for n in ("top", "middle", "bottom")]
    d = np.zeros(len(m.inputs))
    d[m.inputs.index(Constant())] = 1.0
    xs = np.zeros(m.n_states)
    xs[order] = x
    dU = caps @ ((M.A[k] @ xs + M.E[k] @ d)[order])
    # the energy leaving is the flow times (T_mains − T_top): hot water out of the top
    assert dU == pytest.approx(G * (283.15 - 333.15))


def test_recovers_stratified_tank():
    m = compose(loads_house(STRATIFIED))
    rng = np.random.default_rng(5)
    step = 300.0
    n = int(3 * 86400 / step)
    idx = pd.date_range("2030-01-01", periods=n, freq="300s", tz="UTC")
    flow = np.zeros(n)
    for a in rng.integers(0, n - 3, 30):
        flow[a : a + int(rng.integers(1, 3))] = rng.uniform(0.2, 1.0)
    heat = np.zeros(n)
    for a in rng.integers(0, n - 12, 12):
        heat[a : a + int(rng.integers(3, 12))] = 1.0
    frame = pd.DataFrame({"t_room": 293.15 + rng.normal(0, 0.3, n), "flow": flow, "heat": heat}, index=idx)
    truth = {"cyl.UA_layer": 15.0, "cyl.UA_loss": 2.5, "cyl.sigma_v": 0.05}
    sim = simulate(m, truth, frame, step, x0=np.full(3, 323.15), rng=rng)
    res = fit(m, IdData.from_frame(sim, step, m.required_signals()), QUICK)
    check(m, res, truth, tol=0.25)
