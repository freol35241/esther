"""Draw detection (spec §6.3) on a synthetic plant with known draws: the
detector finds them with the right energy, finds nothing where there is
nothing (the negative control), the label validation can say no, and the
outer loop fits the tank with what it detects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import synthetic_tank as st

from esther.identify import (
    DrawOptions,
    FitOptions,
    detect_draws,
    draw_series,
    fit_with_draws,
    inject_response,
    validate_draws,
)

OPTS = DrawOptions(k=4.0)


@pytest.fixture(scope="module")
def model():
    return st.plant()


@pytest.fixture(scope="module")
def theta(model):
    return np.array([st.TRUTH[n] for n in model.param_names])


@pytest.fixture(scope="module")
def wet(model):
    frame, events = st.simulate_plant(model, st.TRUTH, days=3.5, step=300.0, seed=3)
    data = st.iddata(model, frame, step=300.0)
    return data, events[events.time >= data.start].reset_index(drop=True)


@pytest.fixture(scope="module")
def dry(model):
    frame, _ = st.simulate_plant(model, st.TRUTH, days=3.5, step=300.0, seed=3, draws_per_day=0.0)
    return st.iddata(model, frame, step=300.0)


def test_detector_finds_every_draw_with_its_energy(model, theta, wet):
    data, truth = wet
    assert len(truth) >= 10
    ev = detect_draws(model, data, theta, "t_dhw", "q_draw", OPTS)
    v = validate_draws(ev, truth, tolerance="15min")
    assert v.passed, v.summary()
    assert v.false_alarms == 0
    assert v.matches.ratio.between(0.9, 1.1).all(), v.matches
    assert (ev.dT < 0).all()


def test_detector_finds_nothing_without_draws(model, theta, dry):
    assert len(detect_draws(model, dry, theta, "t_dhw", "q_draw", OPTS)) == 0


def test_excluded_steps_are_never_flagged(model, theta, wet):
    data, _ = wet
    extra = pd.concat(data.frames).assign(always=1.0)[["always"]]
    opts = DrawOptions(k=4.0, exclude="always")
    assert len(detect_draws(model, data, theta, "t_dhw", "q_draw", opts, extra=extra)) == 0


def test_draw_series_puts_the_energy_back(model, theta, wet):
    data, _ = wet
    ev = detect_draws(model, data, theta, "t_dhw", "q_draw", OPTS)
    q = draw_series(ev, data)
    assert q.sum() * data.step == pytest.approx(ev.energy.sum())
    # spread over the steps before each flagged step: the first event's heat starts one step early
    first = ev.iloc[0]
    k = data.frames[first.segment].index.get_loc(first.time)
    assert q.loc[data.frames[first.segment].index[k - 1]] > 0


def test_injected_draws_are_found_in_otherwise_dry_data(model, theta, wet, dry):
    """The real-data check without labels: add the model's own response to
    known draws to recorded data and require the detector to find them."""
    _, truth = wet
    q = draw_series_from_truth(truth, dry)
    injected = inject_response(model, theta, dry, "q_draw", q)
    ev = detect_draws(model, injected, theta, "t_dhw", "q_draw", OPTS)
    v = validate_draws(ev, truth, tolerance="15min")
    assert v.passed, v.summary()
    assert v.matches.ratio.between(0.85, 1.15).all(), v.matches


def draw_series_from_truth(truth: pd.DataFrame, data) -> pd.Series:
    idx = pd.concat(data.frames).index
    out = pd.Series(0.0, index=idx)
    for r in truth.itertuples():
        steps = idx[(idx >= r.time) & (idx < r.time + pd.Timedelta(seconds=r.steps * data.step))]
        out.loc[steps] += r.energy / (len(steps) * data.step)
    return out


# -- validation can say no -------------------------------------------------------


def test_validation_refuses_too_few_labels(model, theta, wet):
    data, truth = wet
    ev = detect_draws(model, data, theta, "t_dhw", "q_draw", OPTS)
    v = validate_draws(ev, truth.iloc[:2], tolerance="15min")
    assert not v.passed and "too few labels" in v.reason


def test_validation_refuses_labels_where_nothing_happened(model, theta, wet):
    data, truth = wet
    ev = detect_draws(model, data, theta, "t_dhw", "q_draw", OPTS)
    shifted = truth.assign(time=truth.time + pd.Timedelta("3h"))
    v = validate_draws(ev, shifted, tolerance="15min")
    assert not v.passed and "not found" in v.reason


def test_validation_refuses_the_wrong_order_of_magnitude(model, theta, wet):
    data, truth = wet
    ev = detect_draws(model, data, theta, "t_dhw", "q_draw", OPTS)
    v = validate_draws(ev, truth.assign(energy=truth.energy * 10), tolerance="15min")
    assert not v.passed and "energy" in v.reason
    # labels without energies are matched on time alone
    assert validate_draws(ev, truth[["time"]], tolerance="15min").passed


# -- the outer loop -------------------------------------------------------------


def test_outer_loop_fits_the_tank_with_what_it_detects(model):
    frame, truth = st.simulate_plant(model, st.TRUTH, days=3.0, step=600.0, seed=6)
    data = st.iddata(model, frame, step=600.0)
    truth = truth[truth.time >= data.start]
    zeroed = type(data)(data.step, [f.assign(q_draw=0.0) for f in data.frames])
    out = fit_with_draws(
        model, zeroed, "t_dhw", "q_draw", passes=3, fit_options=FitOptions(n_starts=1), options=OPTS
    )
    assert 2 <= len(out.passes) <= 3
    assert out.passes.objective.is_monotonic_decreasing
    v = validate_draws(out.events, truth, tolerance="20min")
    assert v.passed, v.summary()
    # without the draws the DHW node's process noise absorbs them; with them it shrinks
    first = out.passes.iloc[0]
    assert first.events_in == 0
    sw = out.result.params["tank.sigma_w_dhw"]
    assert sw < 1.0
    with pytest.raises(ValueError, match="passes"):
        fit_with_draws(model, zeroed, "t_dhw", "q_draw", passes=4)
