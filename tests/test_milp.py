"""The MILP layer over HiGHS (spec §8, N1, N2)."""

import numpy as np
import pytest

from esther.plan import MILP, SolveOptions
from esther.plan.blocks import Linear, lag, lag_sum, window


def knapsack():
    m = MILP()
    x = m.var("x", 4, 0, 1, integer=True, cost=[-5.0, -4.0, -3.0, -2.0])
    # one row over four variables: indices of shape (rows, terms)
    m.add([(x[None, :], np.array([[2.0, 3.0, 1.0, 4.0]]))], ub=5.0, name="weight")
    return m, x


def test_optimal_and_audit():
    m, x = knapsack()
    s = m.solve(SolveOptions(time_limit=5))
    assert s.status == "optimal" and s.ok
    assert np.allclose(s.x[x], [1, 1, 0, 0]) and s.objective == pytest.approx(-9.0)
    assert m.check(s.x) == []
    # negative control: an infeasible point is caught, by its row group's name
    bad = m.check(np.array([1.0, 1.0, 1.0, 1.0]))
    assert bad and "weight" in bad[0]
    assert any("not integral" in b for b in m.check(np.array([0.5, 0.0, 0.0, 0.0])))


def test_infeasible_has_no_solution():
    m = MILP()
    x = m.var("x", 1, 0, 1, integer=True)
    m.add([(x, 1.0)], lb=2.0)
    s = m.solve()
    assert s.status == "infeasible" and s.x is None and not s.ok


def test_warm_start_is_taken():
    m, _x = knapsack()
    s = m.solve(warm={"x": np.array([1.0, 1.0, 0.0, 0.0])})
    assert s.warm_started and s.status == "optimal"


def test_node_limit_is_deterministic_and_flagged():
    rng = np.random.default_rng(0)
    m = MILP()
    n = 60
    x = m.var("x", n, 0, 1, integer=True, cost=-rng.uniform(1, 10, n))
    for _r in range(8):
        m.add([(x[None, :], rng.uniform(1, 10, (1, n)))], ub=40.0)
    a = m.solve(SolveOptions(node_limit=3, time_limit=60, gap=0.0))
    b = m.solve(SolveOptions(node_limit=3, time_limit=60, gap=0.0))
    assert a.status in ("limit", "optimal") and a.deterministic
    assert np.array_equal(a.x, b.x)


def test_relax_and_fix_gives_a_feasible_incumbent():
    rng = np.random.default_rng(1)
    m = MILP()
    n = 24
    on = m.var("on", n, 0, 1, integer=True, cost=rng.uniform(0, 1, n))
    q = m.var("q", n, 0, 5, cost=0.1)
    m.add([(q, 1.0), (on, -5.0)], ub=0.0)
    m.add([(q, 1.0)], lb=np.full(n, 1.0))
    m.time_axis(n, ["on", "q"])
    s = m.solve(SolveOptions(time_limit=10), relax_and_fix=3)
    assert s.ok and "relax-and-fix ok" in s.reason
    assert m.check(s.x) == []


def test_linear_blocks_with_history():
    idx = np.arange(5)
    l1 = lag(idx, 1, hist=[7.0])
    assert l1.const[0] == 7.0 and (l1.coef[1:, 0] == 1).all() and l1.idx[2, 0] == 1
    w = window(idx, 3, hist=[1.0, 2.0])
    x = np.arange(5.0) * 10
    assert w.value(x)[0] == pytest.approx(0 + 1 + 2)
    assert w.value(x)[3] == pytest.approx(30 + 20 + 10)
    two = np.arange(10).reshape(5, 2)
    s = lag_sum(two, 1, hist=[1.0])
    assert s.value(np.ones(10))[0] == pytest.approx(1.0) and s.value(np.ones(10))[2] == pytest.approx(2.0)
    _t, _lb, ub = (Linear.of(idx) - 3.0).rows(ub=0.0)
    assert np.allclose(ub, 3.0)
