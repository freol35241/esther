"""A Plan as JSON (spec §9.3): every field round-trips exactly, and — the
positive control — a plan restored from its JSON warm-starts the next solve
exactly as the plan in memory does.

The control could fail: on this problem the warm start changes the result
(checked against a cold solve), and the budget is a node limit, so equal
results mean equal warm starts, not a lucky clock (N2).
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
from synthetic_plan import forecast, house, planner, x0

from esther.forecast import Event, Provenance
from esther.plan import Plan, SolveOptions, plan_from_json, plan_to_json

NOW = pd.Timestamp("2025-01-15T20:00Z")
LATER = NOW + pd.Timedelta("15min")


def cheap_nights(hours):
    return np.where((hours >= 0) & (hours < 6), 0.02, 0.30)


def draws(hours):
    return np.where((hours >= 7) & (hours < 8), 3000.0, 0.0)


def same(a, b, where="plan") -> None:
    """Exact equality, type included; NaN equals NaN."""
    assert type(a) is type(b), f"{where}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True, check_freq=False)
        same(a.index, b.index, f"{where}.index")
    elif isinstance(a, pd.DatetimeIndex):
        assert a.equals(b) and str(a.tz) == str(b.tz) and a.dtype == b.dtype, where
    elif isinstance(a, pd.Timestamp):
        assert a == b and str(a.tz) == str(b.tz) and a.utcoffset() == b.utcoffset(), where
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, where
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), where
    elif isinstance(a, dict):
        assert list(a) == list(b), where
        for k in a:
            same(a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same(x, y, f"{where}[{i}]")
    elif isinstance(a, (Provenance, Event)):
        for f in fields(a):
            same(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or a == b, f"{where}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"


@pytest.fixture(scope="module")
def setup():
    """One-hot stove levels under a node limit: hard enough that the warm
    start decides the incumbent."""
    h = house(stove_extra="", pellet_price=0.08, horizon="24h")
    pl = planner(h, solve=SolveOptions(time_limit=120, gap=0.0, node_limit=5))
    fc = forecast(NOW, h, outdoor=-8.0, price=cheap_nights, draws=draws)
    return h, pl, pl.plan(NOW, x0(h), fc)


def test_every_field_round_trips(setup):
    _h, _pl, plan = setup
    assert plan.warm  # there is a warm-start store to carry
    kolkata = pd.Timestamp("2025-01-15T23:30:00.123456789+05:30").tz_convert("Asia/Kolkata")
    # the edge cases a plan can carry: forecast provenance and events, an
    # offset time zone, NaN and the infinities, numpy scalars, tuples
    edgy = replace(
        plan,
        provenance=[
            Provenance("outdoor_temperature", "weather", "issue", NOW, issued=kolkata, available=NOW),
            Provenance("spot_price", "profile", "time of week", None, note="estimated"),
        ],
        events=[Event("thermostat", "event", NOW, NOW + pd.Timedelta("1h"), forecast=False)],
        validation={
            "accepted": True,
            "rmse": math.nan,
            "bound": math.inf,
            "low": -math.inf,
            "n": np.int64(3),
            "gap": np.float64(0.25),
            "horizons": (1.0, 6.0, 24.0),
            "$weird": {"nested": [None, "x", np.bool_(True)]},
        },
        solver={**plan.solver, "bound_inf": -math.inf},
    )
    for p in (plan, edgy):
        text = plan_to_json(p)
        json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c} in the JSON"))  # strict JSON
        back = plan_from_json(text)
        assert isinstance(back, Plan)
        for f in fields(Plan):
            same(getattr(p, f.name), getattr(back, f.name), f.name)
        assert back.flagged == p.flagged and back.advisory == p.advisory
        assert plan_to_json(back) == text  # and the restored plan writes the same JSON


def test_a_restored_plan_warm_starts_the_same(setup):
    h, pl, plan = setup
    back = plan_from_json(plan_to_json(plan))
    fc2 = forecast(LATER, h, outdoor=-8.0, price=cheap_nights, draws=draws)
    x = {s: float(plan.trajectories[s].iloc[1]) for s in plan.trajectories}
    # the same warm start and linearisation point ...
    w_mem, w_back = pl._shift_warm(plan, fc2.index), pl._shift_warm(back, fc2.index)
    same(w_mem, w_back, "warm")
    xv = np.array([x[s] for s in pl.model.states])
    same(pl._initial_lin(fc2, xv, plan), pl._initial_lin(fc2, xv, back), "lin")
    # ... and the same plan, under a deterministic budget
    a = pl.plan(LATER, x, fc2, previous=plan)
    b = pl.plan(LATER, x, fc2, previous=back)
    assert a.solver["warm_started"] and b.solver["warm_started"]
    assert a.solver["deterministic"] and b.solver["deterministic"]
    assert np.array_equal(a.decisions.to_numpy(), b.decisions.to_numpy())
    assert a.id == b.id
    # the control can fail: without the warm start the solve ends elsewhere
    cold = pl.plan(LATER, x, fc2)
    assert not cold.solver["warm_started"]
    assert cold.id != a.id and not np.array_equal(cold.decisions.to_numpy(), a.decisions.to_numpy())


def test_refusals():
    with pytest.raises(ValueError, match="not an esther plan"):
        plan_from_json(json.dumps({"id": "x"}))
    with pytest.raises(ValueError, match="lacks"):
        plan_from_json(json.dumps({"format": "esther.plan/1"}))


def test_an_unknown_type_is_refused_not_stringified(setup):
    _h, _pl, plan = setup
    with pytest.raises(TypeError, match="cannot serialise"):
        plan_to_json(replace(plan, validation={"x": object()}))
