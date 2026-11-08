"""The live state estimate (spec §6.6, F5) on synthetic data.

A 2R2C house is simulated from known parameters with its unmeasured mass
10 K above the zone — 3.3 sd from the filter's prior, which puts every
unmeasured node at the mean of the first measurements. The estimate must
find the mass; the negative control runs the same estimate with the mass's
coupling wrong ×5 and must miss by clearly more, so the recovery test could
have failed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import synthetic as syn

from esther.identify import FilterOptions, StateError, estimate_state, simulate

TRUTH = {
    "zone.C": 4e6,
    "zone.sigma_w": 0.05,
    "zone.sigma_v": 0.03,
    "mass.C": 3e7,
    "mass.UA": 400.0,
    "env.UA": 120.0,
    "radiators.UA_rad": 200.0,
}
STEP = 600.0
HOUR = 3600.0
PRIOR_SD = FilterOptions().x0_sd_unmeasured


@pytest.fixture(scope="module")
def house():
    m = syn.compose_house(syn.MASS, syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT)
    frame = syn.inputs(2.0, STEP, seed=0)
    sim = simulate(m, TRUTH, frame, STEP, x0=[syn.K + 20.0, syn.K + 30.0], rng=np.random.default_rng(7))
    now = sim.index[0] + pd.Timedelta(hours=12)
    return m, sim, now


def truth_at(sim, t, node="mass"):
    return float(sim.loc[t, f"x:{node}"])


def test_an_unmeasured_node_is_recovered(house):
    m, sim, now = house
    est = estimate_state(m, TRUTH, sim, STEP, now, window=12 * HOUR, min_steps=6)
    assert est.time == now - pd.Timedelta(seconds=STEP)  # the newest step that has ended
    assert est.states == m.states and est.n_steps == 72
    start = sim.index[0]  # the window's first step: the prior puts the mass at the zone's measurement
    assert truth_at(sim, start) - sim.loc[start, "t_in"] > 3 * PRIOR_SD
    true = truth_at(sim, est.time)
    err = est.mean["mass"] - true
    assert abs(err) < 3 * est.sd["mass"]
    assert abs(err) < 0.1 * PRIOR_SD
    assert est.sd["mass"] < 0.1 * PRIOR_SD
    assert abs(est.mean["zone"] - truth_at(sim, est.time, "zone")) < 3 * est.sd["zone"]
    assert np.allclose(np.sqrt(np.diag(est.covariance)), [est.sd[s] for s in est.states])
    assert np.allclose(est.x, [est.mean[s] for s in est.states])


def test_a_longer_window_does_not_degrade(house):
    m, sim, now = house
    runs = [estimate_state(m, TRUTH, sim, STEP, now, window=w * HOUR, min_steps=6) for w in (2, 4, 12)]
    true = truth_at(sim, runs[0].time)
    sds = [r.sd["mass"] for r in runs]
    errs = [abs(r.mean["mass"] - true) for r in runs]
    assert sds[0] > sds[1] > sds[2]
    assert errs[2] <= errs[0] + 0.01
    assert [r.n_steps for r in runs] == [12, 24, 72]


def test_wrong_parameters_miss(house):
    # negative control: the mass's coupling ×5 — the zone's heat balance then
    # puts the mass far too close to the zone
    m, sim, now = house
    right = estimate_state(m, TRUTH, sim, STEP, now, window=12 * HOUR, min_steps=6)
    wrong = estimate_state(m, {**TRUTH, "mass.UA": 5 * TRUTH["mass.UA"]}, sim, STEP, now, window=12 * HOUR)
    true = truth_at(sim, right.time)
    miss, hit = abs(wrong.mean["mass"] - true), abs(right.mean["mass"] - true)
    assert miss > 0.5 and miss > 10 * hit
    assert miss > 3 * wrong.sd["mass"]  # and it does not know it


def test_no_look_ahead(house):
    # rows after now, and the step still running at now, are not read
    m, sim, now = house
    mid = now + pd.Timedelta(minutes=3)
    poisoned = sim.copy()
    poisoned.loc[poisoned.index >= now, "t_in"] = syn.K + 80.0
    a = estimate_state(m, TRUTH, sim, STEP, mid, window=12 * HOUR)
    b = estimate_state(m, TRUTH, poisoned, STEP, mid, window=12 * HOUR)
    assert a.time == b.time == now - pd.Timedelta(seconds=STEP)
    assert a.mean == b.mean
    c = estimate_state(m, TRUTH, poisoned, STEP, now + pd.Timedelta(seconds=STEP), window=12 * HOUR)
    assert c.mean != a.mean  # positive twin: once that step has ended it is read


def test_refusals(house):
    m, sim, now = house
    silent = sim.copy()
    silent.loc[silent.index >= now - pd.Timedelta(hours=1), "t_in"] = np.nan
    with pytest.raises(StateError) as e:
        estimate_state(m, TRUTH, silent, STEP, now, max_age=1800.0)
    assert e.value.reason == "stale"
    ok = estimate_state(m, TRUTH, silent, STEP, now, max_age=2 * HOUR)  # positive twin
    assert ok.time == now - pd.Timedelta(hours=1, seconds=STEP)

    gap = sim.copy()
    hole = (gap.index >= now - pd.Timedelta(hours=2)) & (gap.index < now - pd.Timedelta(minutes=30))
    gap.loc[hole, "t_in"] = np.nan
    with pytest.raises(StateError) as e:
        estimate_state(m, TRUTH, gap, STEP, now, min_steps=6)
    assert e.value.reason == "too short"
    assert estimate_state(m, TRUTH, gap, STEP, now, min_steps=3).n_steps == 3  # positive twin

    with pytest.raises(StateError) as e:
        estimate_state(m, TRUTH, sim, STEP, sim.index[0] - pd.Timedelta(days=2), window=12 * HOUR)
    assert e.value.reason == "no data"
    with pytest.raises(KeyError, match=r"mass\.UA"):
        estimate_state(m, {k: v for k, v in TRUTH.items() if k != "mass.UA"}, sim, STEP, now)
