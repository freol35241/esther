"""The M0 solver spike solves, and both formulations are the same problem.

Wall time is not asserted (CI runners vary); it is recorded in the PR and
printed by ``python benchmarks/reference_milp.py``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SPIKE = Path(__file__).resolve().parents[1] / "benchmarks" / "reference_milp.py"
spec = importlib.util.spec_from_file_location("reference_milp", SPIKE)
rm = importlib.util.module_from_spec(spec)
sys.modules["reference_milp"] = rm
spec.loader.exec_module(rm)

HAVE_LINOPY = importlib.util.find_spec("linopy") is not None


@pytest.fixture
def highs(monkeypatch):
    opts = dict(rm.HIGHS_OPTIONS)
    monkeypatch.setattr(rm, "HIGHS_OPTIONS", opts)
    return opts


def test_problem_is_144_steps_of_15_min():
    p = rm.ReferenceProblem()
    assert p.n_steps == 144 and p.dt == 900.0
    assert p.Ad.shape == (4, 4) and p.Bd.shape == (4, 4) and p.Ed.shape == (4, 3)


def test_dynamics_are_offset_invariant():
    # Solving in °C instead of K relies on this: a uniform temperature
    # (nodes and outdoor alike) with no heat input stays put.
    p = rm.ReferenceProblem()
    x = np.full(4, 7.0)
    d = np.array([7.0, 0.0, 0.0])
    assert np.allclose(p.Ad @ x + p.Ed @ d, x)


def test_easy_variant_solves_to_optimality(highs):
    highs["mip_rel_gap"] = 1e-6
    r = rm.run("cvxpy", "easy")
    assert r.status == "optimal"
    assert math.isfinite(r.objective)


@pytest.mark.skipif(not HAVE_LINOPY, reason="linopy not installed (pip install -e '.[spike]')")
def test_cvxpy_and_linopy_build_the_same_problem(highs):
    # The control on the spike itself: two independently written
    # formulations must reach the same optimum.
    highs["mip_rel_gap"] = 1e-6
    a = rm.run("cvxpy", "easy")
    b = rm.run("linopy", "easy")
    assert a.status == "optimal" and b.status == "optimal"
    assert a.objective == pytest.approx(b.objective, rel=1e-4, abs=1e-3)


@pytest.mark.parametrize("two_phase", [False, True], ids=["cold", "two-phase"])
def test_full_variant_returns_a_feasible_plan(highs, two_phase):
    highs["time_limit"] = 5.0
    p = rm.ReferenceProblem()
    r = rm.solve_cvxpy(p, two_phase=two_phase)
    assert r.status in ("optimal", "user_limit", "optimal_inaccurate"), r.status
    assert math.isfinite(r.objective)
    tr = r.trajectories
    # Decisions honour their integrality and the constraints mean what they say.
    heater = tr["heater"]
    assert np.allclose(heater, np.round(heater), atol=1e-6)
    assert heater.min() >= -1e-6 and heater.max() <= p.heater_levels + 1e-6
    assert tr["T_z"][1:].min() >= p.T_z_floor - rm.C2K - 1e-4
    assert tr["T_j"].max() <= p.T_j_max - rm.C2K + 1e-4
    on = np.round(tr["hp_on"]).astype(int)
    _assert_min_run(on, p.hp_min_on, p.hp_min_off)
    stove_on = (np.round(tr["stove_level"]) > 0).astype(int)
    assert not stove_on[~p.stove_allowed].any()
    _assert_min_run(stove_on, p.stove_min_on, p.stove_min_off)


def _assert_min_run(on: np.ndarray, min_on: int, min_off: int) -> None:
    """Every run (that ends inside the horizon) lasts >= min_on steps, and every
    gap between runs >= min_off. The state before the horizon is off."""
    edges = np.flatnonzero(np.diff(np.r_[0, on, 0]))
    starts, stops = edges[::2], edges[1::2]
    for a, b in zip(starts, stops, strict=True):
        if b < len(on):
            assert b - a >= min_on, f"run {a}..{b} shorter than {min_on}"
    for b, a in zip(stops[:-1], starts[1:], strict=True):
        assert a - b >= min_off, f"gap {b}..{a} shorter than {min_off}"


def test_min_run_checker_catches_a_short_run():
    # Positive control for the checker above.
    with pytest.raises(AssertionError):
        _assert_min_run(np.array([0, 1, 1, 0, 0, 0]), 4, 1)
    with pytest.raises(AssertionError):
        _assert_min_run(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1]), 4, 2)
