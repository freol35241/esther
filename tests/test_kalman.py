"""The batched filter against a plain textbook implementation, the open-loop
simulator against the simulator, and IdData's segmenting and split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import synthetic as syn

from esther.core.model import discretise
from esther.identify import FilterOptions, IdData, kalman, open_loop, prepare, simulate, validate


def reference_nll(model, data, theta):
    """One segment, one step at a time, dense matrices, no tricks. A missing
    measurement (NaN) is left out of that step's update."""
    opts = FilterOptions()
    M = model.matrices(theta)
    nll = 0.0
    states = []
    for f in data.frames:
        D = model.input_array(f)
        Y = f[list(model.outputs)].to_numpy()
        S = f[list(model.modulations)].to_numpy() if model.modulations else np.zeros((len(f), 0))
        S = np.round(S * opts.levels) / opts.levels
        H, c, R = M.H, M.c, M.R
        n = model.n_states
        first = np.array([col[np.isfinite(col)][0] for col in Y.T])
        y0 = first - c
        x = np.full(n, y0.mean())
        sd = np.full(n, opts.x0_sd_unmeasured)
        for i in range(n):
            sel = H.argmax(axis=1) == i
            if sel.any():
                x[i] = y0[sel].mean()
                sd[i] = opts.x0_sd_measured
        P = np.diag(sd**2)
        xs = []
        for k in range(len(f)):
            ok = np.isfinite(Y[k])
            Hk, Rk = H[ok], R[np.ix_(ok, ok)]
            e = Y[k][ok] - Hk @ x - c[ok]
            if ok.any():
                Sk = Hk @ P @ Hk.T + Rk
                Kg = P @ Hk.T @ np.linalg.inv(Sk)
                x = x + Kg @ e
                P = P - Kg @ Hk @ P
                nll += 0.5 * (np.linalg.slogdet(2 * np.pi * Sk)[1] + e @ np.linalg.solve(Sk, e))
            xs.append(x.copy())
            A, E = M.at(S[k])
            d = discretise(A, E, M.Qc, data.step)
            x = d.Ad @ x + d.Gd @ D[k]
            P = d.Ad @ P @ d.Ad.T + d.Qd
        states.append(np.array(xs))
    return nll, states


TRUTH = {
    "zone.C": 4e6,
    "zone.sigma_w": 0.05,
    "zone.sigma_v": 0.03,
    "mass.C": 3e7,
    "mass.UA": 400.0,
    "env.UA": 120.0,
    "radiators.UA_rad": 200.0,
}

SENSOR = """
[[component]]
type = "Sensor"
name = "second"
node = "zone"
signal = "t_in2"
"""


@pytest.fixture(scope="module")
def two_node():
    m = syn.compose_house(syn.MASS, syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT)
    data, _ = syn.dataset(m, TRUTH, days=3, step=900.0, gaps=2)
    return m, data


def test_filter_matches_reference(two_node):
    m, data = two_node
    assert len(data.frames) == 3 and m.modulations == ("pump",)
    th = np.array([TRUTH[n] for n in m.param_names])
    ref, ref_states = reference_nll(m, data, th)
    res = kalman(m, prepare(m, data), th[None], store=True)
    assert res.nll[0] == pytest.approx(ref, rel=1e-9)
    for s, xs in enumerate(ref_states):
        assert np.allclose(res.x_filt[0, s, : len(xs)], xs, atol=1e-8)


def test_filter_matches_reference_two_sensors():
    m = syn.compose_house(syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT, SENSOR)
    truth = {"zone.C": 4e6, "zone.sigma_w": 0.05, "zone.sigma_v": 0.03, "env.UA": 120.0}
    truth.update({"radiators.UA_rad": 200.0, "second.bias": 0.5, "second.sigma_v": 0.08})
    data, _ = syn.dataset(m, truth, days=2, step=900.0, gaps=1)
    th = np.array([truth[n] for n in m.param_names])
    ref, _ = reference_nll(m, data, th)
    assert kalman(m, prepare(m, data), th[None]).nll[0] == pytest.approx(ref, rel=1e-9)


def test_missing_measurements_are_skipped_not_segment_breaks():
    m = syn.compose_house(syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT, SENSOR)
    truth = {"zone.C": 4e6, "zone.sigma_w": 0.05, "zone.sigma_v": 0.03, "env.UA": 120.0}
    truth.update({"radiators.UA_rad": 200.0, "second.bias": 0.5, "second.sigma_v": 0.08})
    _, sim = syn.dataset(m, truth, days=2, step=900.0, gaps=0)
    rng = np.random.default_rng(1)
    holes = rng.random(len(sim)) < 0.2
    sim.loc[holes, "t_in2"] = np.nan
    sim.iloc[0, sim.columns.get_loc("t_in2")] = np.nan  # missing at the very start too
    sim.loc[rng.random(len(sim)) < 0.05, "t_in"] = np.nan  # and now and then both
    strict = IdData.from_frame(sim, 900.0, m.required_signals())
    assert len(strict.frames) > 10  # every hole ends a segment
    data = IdData.from_frame(sim, 900.0, m.required_signals(), optional=["t_in", "t_in2"])
    assert len(data.frames) == 1 and data.n_steps == len(sim)
    th = np.array([truth[n] for n in m.param_names])
    ref, _ = reference_nll(m, data, th)
    kf = kalman(m, prepare(m, data), th[None], store=True)
    assert kf.nll[0] == pytest.approx(ref, rel=1e-9)
    assert np.isnan(kf.innov[0, 0, holes, 1]).all()
    # inputs cannot be optional: a missing input still ends the segment
    sim2 = sim.copy()
    sim2.iloc[50, sim2.columns.get_loc("t_out")] = np.nan
    assert len(IdData.from_frame(sim2, 900.0, m.required_signals(), optional=["t_in", "t_in2"]).frames) == 2
    # and validation scores only where both the model and persistence have a measurement
    v = validate(m, th, data)
    assert np.isfinite(v.rmse("t_in2", 6 * 3600.0))


def test_batch_equals_separate_calls(two_node):
    m, data = two_node
    prep = prepare(m, data)
    rng = np.random.default_rng(3)
    th = np.array([TRUTH[n] for n in m.param_names])
    batch = th * np.exp(rng.normal(0, 0.2, (5, len(th))))
    together = kalman(m, prep, batch).nll
    alone = [kalman(m, prep, b[None]).nll[0] for b in batch]
    assert together == pytest.approx(alone, rel=1e-12)
    # and the likelihood is not flat: the truth beats the perturbations
    assert kalman(m, prep, th[None]).nll[0] < together.min()


def test_open_loop_matches_noise_free_simulation(two_node):
    m, data = two_node
    th = np.array([TRUTH[n] for n in m.param_names])
    f = data.frames[0]
    x0 = np.array([293.0, 291.0])
    sim = simulate(m, th, f, data.step, x0=x0, noise=False)
    pred, valid = open_loop(m, prepare(m, data), th, [(0, 0)], x0[None], len(f) - 1)
    assert valid[0].all()
    assert np.allclose(pred[0, :, 0], sim["t_in"].to_numpy()[1:], atol=1e-9)


def test_prepare_refuses_non_finite(two_node):
    m, data = two_node
    bad = IdData(data.step, [data.frames[0].copy()])
    bad.frames[0].iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        prepare(m, bad)


# -- IdData -------------------------------------------------------------------


def frame(n=20, step=300):
    idx = pd.date_range("2030-01-01", periods=n, freq=f"{step}s", tz="UTC")
    return pd.DataFrame({"a": np.arange(n, dtype=float), "b": 1.0}, index=idx)


def test_iddata_segments_on_missing_values_and_index_gaps():
    f = frame()
    f.iloc[5, 0] = np.nan
    f = f.drop(f.index[12])  # a hole in the grid
    d = IdData.from_frame(f, 300.0, ["a", "b"], min_steps=2)
    assert d.lengths == [5, 6, 7]
    assert all(np.isfinite(x.to_numpy()).all() for x in d.frames)
    # a column the model does not read does not break segments
    f2 = frame()
    f2["c"] = np.nan
    assert IdData.from_frame(f2, 300.0, ["a"]).lengths == [20]


def test_iddata_min_steps_and_usable_mask():
    f = frame()
    usable = pd.Series(True, index=f.index)
    usable.iloc[2] = False
    d = IdData.from_frame(f, 300.0, ["a"], min_steps=3, usable=usable)
    assert d.lengths == [17]  # the 2-step run before the mask is dropped


def test_iddata_split_is_by_time_and_cuts_a_straddling_segment():
    f = frame(100)
    f.iloc[40, 0] = np.nan
    d = IdData.from_frame(f, 300.0, ["a"])
    train, test = d.split(0.2)
    assert train.n_steps + test.n_steps == d.n_steps
    assert test.n_steps == round(0.2 * d.n_steps)
    assert train.end <= test.start
    assert train.lengths == [40, 39] and test.lengths == [20]
