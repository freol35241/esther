from typing import Callable
from dataclasses import asdict
from app import opt
from app.dynamic_model import (
    ModelParameters,
    DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT,
)
from app.heat_sources import IVT490


def test_prepare_optimization_problem(pinned):
    model = ModelParameters(DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT, 0)

    problem = opt.prepare_optimization_problem(model, [0], [0], [3600], 0, 0, 0, 0, 0)

    assert list(asdict(problem).keys()) == pinned

    assert problem.size == 1
    assert isinstance(problem.objective, Callable)
    assert isinstance(problem.equality_constraints, Callable)
    assert isinstance(problem.inequality_constraints, Callable)
    assert isinstance(problem.bounds, list)


def test_solve_problem(pinned):
    model = IVT490.create_model_from_slope(4.7)

    problem = opt.prepare_optimization_problem(
        model, [1, 2], [15, 10], [3600, 3600], 20, 20, 18, 22, 50
    )

    res = opt.solve_problem(
        problem, IVT490.make_initial_guess(4.7, [15, 10]), maxiter=1000
    )

    assert res.success == True
    assert res.x == pinned.approx(rel=0.01)
