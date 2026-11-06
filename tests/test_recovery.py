"""Synthetic recovery (spec §10.1, N5): every M2 component, simulated from
known parameters with process and measurement noise and gaps, fitted from its
priors.

§10.1 asks for the true values inside the reported 95 % intervals. Taken
literally on one draw that is a coin toss even for a perfect estimator: with
eight parameters, all eight land inside their 95 % intervals only ~66 % of
the time. So the requirement is tested as what it means:

- **calibration** (``test_interval_coverage``): over many independent draws,
  the 95 % intervals cover the truth at their nominal rate (≥ 85 % here; a
  study of 20 draws × 8 parameters gave 94 %);
- **per component**: on one fixed draw, every true value lies inside the 99 %
  interval, every parameter is identified, none is on a bound, and the
  capacities, conductances and gains are within ``TOL`` of the truth.

The priors are deliberately *not* centred on the truth (factors of 1.2–2
off), so a fit that stayed at the prior would fail. Every house has one heat
input of known size (``KNOWN_HEAT``): without it the data fix only ratios
(see test_gate.py, the scale control).
"""

from __future__ import annotations

import numpy as np
import pytest
import synthetic as syn

from esther.identify import FitOptions, fit

TOL = 0.15  # relative, on capacities, conductances and gains
QUICK = FitOptions(n_starts=2)

BASE_TRUTH = {
    "zone.C": 4e6,
    "zone.sigma_w": 0.05,
    "zone.sigma_v": 0.03,
    "env.UA": 120.0,
    "radiators.UA_rad": 200.0,
}
MASS_TRUTH = {"mass.C": 3e7, "mass.UA": 400.0}
NOISE = ("sigma_w", "sigma_v")


def check(model, truth, days=10.0, step=900.0, seed=0, options=QUICK):
    data, _ = syn.dataset(model, truth, days=days, step=step, seed=seed)
    assert len(data.frames) >= 2
    res = fit(model, data, options)
    tab = res.table()
    tab["true"] = [truth[n] for n in tab.index]
    tab["lo99"], tab["hi99"] = res.interval(0.99)
    inside = (tab.lo99 <= tab.true) & (tab.true <= tab.hi99)
    assert inside.all(), f"true value outside the 99 % interval:\n{tab[~inside]}"
    assert tab.identified.all(), f"not identified:\n{tab[~tab.identified]}"
    assert not tab.on_bound.any()
    # capacities, conductances, gains; noise, biases, amplitudes and shares are checked apart
    shape = ("a", "b", "f_")
    phys = [n for n in tab.index if not n.endswith(NOISE) and not n.split(".")[1].startswith(shape)]
    rel = (tab.loc[phys, "value"] / tab.loc[phys, "true"] - 1).abs()
    assert (rel < TOL).all(), f"relative error above {TOL}:\n{rel[rel >= TOL]}"
    return res


def test_interval_coverage():
    """The 95 % intervals are calibrated: over independent draws they cover
    the truth at about their nominal rate. (Too narrow intervals would pass
    every per-component test above and fail here.)"""
    m = syn.compose_house(syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT)
    hits = []
    for seed in range(12):
        data, _ = syn.dataset(m, BASE_TRUTH, days=6.0, step=900.0, seed=100 + seed)
        res = fit(m, data, FitOptions(n_starts=1))
        tv = np.array([BASE_TRUTH[n] for n in res.names])
        hits.append((res.lo95 <= tv) & (tv <= res.hi95))
    coverage = float(np.mean(hits))
    assert coverage >= 0.85, f"95 % intervals cover the truth {coverage:.0%} of the time"
    assert coverage < 1.0 or len(hits) * len(BASE_TRUTH) < 40  # not absurdly wide either


def test_default_options_follow_the_spec():
    assert FitOptions().n_starts >= 8  # §6.2


def test_recovers_one_node_with_radiators():
    m = syn.compose_house(syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT)
    check(m, BASE_TRUTH)


def test_recovers_two_node_mass():
    m = syn.compose_house(syn.MASS, syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT)
    check(m, {**BASE_TRUTH, **MASS_TRUTH}, step=600.0)


def test_recovers_mass_to_ambient_coupling():
    loss = """
[[component]]
type = "Coupling"
name = "loss"
between = ["mass", "outdoor"]
prior.UA = { median = "50 W/K", gsd = 3.0 }
"""
    m = syn.compose_house(syn.MASS, syn.ENVELOPE, loss, syn.RADIATORS, syn.KNOWN_HEAT)
    truth = {**BASE_TRUTH, **MASS_TRUTH, "loss.UA": 30.0, "mass.C": 1.5e7}
    check(m, truth, days=16.0, step=900.0, seed=1)


def test_recovers_solar_gain_split():
    solar = """
[[component]]
type = "SolarGain"
name = "solar"
zone = "zone"
mass = "mass"
signal = "sun"
prior.g_A = { median = 2.0, gsd = 5.0 }
"""
    m = syn.compose_house(syn.MASS, syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT, solar)
    truth = {**BASE_TRUTH, **MASS_TRUTH, "solar.g_A": 3.5, "solar.f_mass": 0.4}
    res = check(m, truth, days=12.0, step=900.0)
    i = res.names.index("solar.f_mass")
    assert abs(res.theta[i] - 0.4) < 0.15


def test_recovers_internal_gain_profile():
    gain = """
[[component]]
type = "InternalGain"
name = "internal"
zone = "zone"
harmonics = 1
"""
    m = syn.compose_house(syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT, gain)
    truth = {**BASE_TRUTH, "internal.Q": 450.0, "internal.a1": -150.0, "internal.b1": 120.0}
    res = check(m, truth, days=14.0)
    for n in ("internal.a1", "internal.b1"):
        i = res.names.index(n)
        assert abs(res.theta[i] - truth[n]) < 50.0


def test_recovers_sensor_bias():
    sensor = """
[[component]]
type = "Sensor"
name = "wall"
node = "zone"
signal = "t_wall"
"""
    m = syn.compose_house(syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT, sensor)
    truth = {**BASE_TRUTH, "wall.bias": -0.8, "wall.sigma_v": 0.06}
    res = check(m, truth, days=6.0)
    i = res.names.index("wall.bias")
    assert abs(res.theta[i] + 0.8) < 0.05


def test_recovers_ventilation_flow_with_the_envelope_pinned():
    vent = """
[[component]]
type = "Ventilation"
name = "vent"
zone = "zone"
ambient = "outdoor"
"""
    env = syn.ENVELOPE.replace('{ median = "150 W/K", gsd = 2.0 }', "{ value = 90, fixed = true }")
    m = syn.compose_house(env, syn.RADIATORS, syn.KNOWN_HEAT, vent)
    truth = {k: v for k, v in BASE_TRUTH.items() if k != "env.UA"}
    truth["vent.flow"] = 0.05  # 60 W/K
    res = check(m, truth)
    assert res.params["vent.flow"] == pytest.approx(0.05, rel=TOL)


def test_recovery_runs_starts_in_parallel():
    m = syn.compose_house(syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT)
    data, _ = syn.dataset(m, BASE_TRUTH, days=4.0)
    a = fit(m, data, FitOptions(n_starts=2, workers=1))
    b = fit(m, data, FitOptions(n_starts=2, workers=2))
    assert np.allclose(a.theta, b.theta)
