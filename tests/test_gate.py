"""Negative controls: the identifiability report and the acceptance gate must
be able to say no (spec §6.2, §6.5).

Each refusal test has its positive twin — the same data and fit, changed in
the one way the test is about — so a gate that refused everything would fail
here too.
"""

from __future__ import annotations

import numpy as np
import pytest
import synthetic as syn

from esther.identify import FitOptions, acceptance_gate, fit, ljung_box, validate

QUICK = FitOptions(n_starts=1)  # the start at the prior centre; tests need speed, not the spec's 8
TRUTH = {
    "zone.C": 4e6,
    "zone.sigma_w": 0.05,
    "zone.sigma_v": 0.03,
    "mass.C": 3e7,
    "mass.UA": 400.0,
    "env.UA": 120.0,
    "radiators.UA_rad": 200.0,
}
ONE_NODE = ("zone.C", "zone.sigma_w", "zone.sigma_v", "env.UA", "radiators.UA_rad")


@pytest.fixture(scope="module")
def two_node_data():
    """2R2C truth: train on 14 days, hold out the last 25 %."""
    m = syn.compose_house(syn.MASS, syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT)
    data, _ = syn.dataset(m, TRUTH, days=14.0, step=900.0, gaps=1)
    train, test = data.split(0.25)
    return m, train, test


@pytest.fixture(scope="module")
def right(two_node_data):
    m, train, test = two_node_data
    res = fit(m, train, QUICK)
    return res, validate(m, res, test)


@pytest.fixture(scope="module")
def wrong(two_node_data):
    """The same data fitted with a 1R1C structure (mis-specified: no mass)."""
    _, train, test = two_node_data
    m1 = syn.compose_house(syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT)
    res = fit(m1, train, QUICK)
    return res, validate(m1, res, test)


def test_right_structure_is_accepted_against_persistence(right):
    res, val = right
    assert val.n_windows() > 0
    g = acceptance_gate(res, val)
    assert g.accepted, g.summary()
    assert g.reference == "persistence" and g.candidate_rmse < g.reference_rmse


def test_misspecified_structure_is_refused_against_the_right_one(right, wrong):
    (r2, v2), (r1, v1) = right, wrong
    assert v1.windows == v2.windows
    refused = acceptance_gate(r1, v1, current=v2)
    assert not refused.accepted
    assert any("worse than the current fit" in r for r in refused.reasons)
    # positive twin: the right structure replaces the wrong one
    assert acceptance_gate(r2, v2, current=v1).accepted
    # (Ljung–Box is not asserted here: on a held-out set this short even the
    # true parameters can fail it by chance; it is reported, not gating.)


def test_wrong_parameters_are_refused_against_persistence(two_node_data, right):
    m, _, test = two_node_data
    res, _ = right
    theta = res.theta.copy()
    theta[res.names.index("radiators.UA_rad")] *= 4.0
    bad = validate(m, theta, test)
    g = acceptance_gate(res, bad)
    assert not g.accepted and "persistence" in g.reasons[0]


def test_parameter_on_a_bound_is_refused(two_node_data):
    _, train, test = two_node_data
    env = syn.ENVELOPE.replace(
        '{ median = "150 W/K", gsd = 2.0 }', '{ median = "40 W/K", gsd = 2.0, bounds = ["10 W/K", "60 W/K"] }'
    )
    m = syn.compose_house(syn.MASS, env, syn.RADIATORS, syn.KNOWN_HEAT)
    res = fit(m, train, QUICK)
    assert res.on_bound[res.names.index("env.UA")]
    g = acceptance_gate(res, validate(m, res, test))
    assert not g.accepted and any("env.UA" in r and "bound" in r for r in g.reasons)


def test_no_long_enough_window_is_refused(two_node_data, right):
    m, _, test = two_node_data
    res, _ = right
    short = type(test)(test.step, [f.iloc[:40] for f in test.frames])  # 10 h segments
    val = validate(m, res, short)
    assert val.n_windows() == 0
    g = acceptance_gate(res, val)
    assert not g.accepted and "nothing to judge" in g.reasons[0]
    assert acceptance_gate(res, validate(m, res, short), horizon=6 * 3600).accepted


def test_gate_refuses_to_compare_different_windows(two_node_data, right):
    m, _, test = two_node_data
    res, val = right
    other = validate(m, res, test, stride=2 * 3600)
    with pytest.raises(ValueError, match="different windows"):
        acceptance_gate(res, val, current=other)


# -- identifiability ---------------------------------------------------------


def test_common_scale_is_flagged_without_a_known_heat_input(two_node_data):
    """Every heat term free: capacities, conductances and the heater's gain
    can all scale together. The per-parameter test alone would pass them
    (each prior props the common scale up); the data-information analysis
    must flag them and name the direction."""
    _, train, _ = two_node_data
    free_heat = syn.KNOWN_HEAT.replace(
        "prior.gain = { value = 1.0, fixed = true }", "prior.gain = { median = 0.8, gsd = 1.5 }"
    )
    m = syn.compose_house(syn.MASS, syn.ENVELOPE, syn.RADIATORS, free_heat)
    res = fit(m, train, QUICK)
    flagged = {n for n, ok in zip(res.names, res.identified, strict=True) if not ok}
    assert {"zone.C", "mass.C", "mass.UA", "env.UA", "radiators.UA_rad", "heater.gain"} <= flagged
    assert not {"zone.sigma_v", "zone.sigma_w"} & flagged  # noise is still identified
    assert res.weak_directions, "the unidentified direction must be reported"
    lam, load = res.weak_directions[0]
    assert lam < 0.5 and {"zone.C", "env.UA", "heater.gain"} <= set(load)
    assert acceptance_gate(res, validate(m, res, two_node_data[2])).warnings


def test_parallel_conductances_are_flagged():
    """An envelope coupling and a ventilation to the same boundary: only
    their sum is in the data."""
    vent = '\n[[component]]\ntype = "Ventilation"\nname = "vent"\nzone = "zone"\nambient = "outdoor"\n'
    m = syn.compose_house(syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT, vent)
    truth = {"zone.C": 4e6, "zone.sigma_w": 0.05, "zone.sigma_v": 0.03, "env.UA": 90.0}
    truth.update({"radiators.UA_rad": 200.0, "vent.flow": 0.025})
    data, _ = syn.dataset(m, truth, days=6.0, step=900.0)
    res = fit(m, data, QUICK)
    ok = dict(zip(res.names, res.identified, strict=True))
    assert not ok["env.UA"] and not ok["vent.flow"]
    assert ok["zone.C"] and ok["radiators.UA_rad"]
    i, j = res.names.index("env.UA"), res.names.index("vent.flow")
    assert res.correlation[i, j] < -0.9


# -- Ljung–Box, with its own controls ------------------------------------------


def test_ljung_box_white_noise_passes_and_ar1_fails():
    rng = np.random.default_rng(0)
    white = [rng.standard_normal(500) for _ in range(3)]
    ar = []
    for _ in range(3):
        e = rng.standard_normal(500)
        x = np.zeros(500)
        for k in range(1, 500):
            x[k] = 0.5 * x[k - 1] + e[k]
        ar.append(x)
    assert ljung_box(white, [10]).p.iloc[0] > 0.05
    assert ljung_box(ar, [10]).p.iloc[0] < 1e-6
    assert ljung_box(ar, [1]).acf.iloc[0] == pytest.approx(0.5, abs=0.07)
