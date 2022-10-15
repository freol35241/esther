from typing import Callable
from app import opt
from app.dynamic_model import (
    ModelParameters,
    DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT,
    create_model_from_IVT490_settings,
)

import numpy as np


def test_prepare_optimization_problem():
    model = ModelParameters(DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT, 0)

    problem = opt.prepare_optimization_problem(model, [0], [0], 0, 0, 0, 0, 0)

    assert isinstance(problem.objective, Callable)
    assert isinstance(problem.equality_constraints, Callable)
    assert isinstance(problem.inequality_constraints, Callable)
    assert isinstance(problem.bounds, list)
    assert isinstance(problem.initial_guess, np.ndarray)


def test_solve_problem(pinned):
    model = create_model_from_IVT490_settings(4.7)

    problem = opt.prepare_optimization_problem(
        model, [1, 2], [20, 10], 20, 20, 18, 22, 50
    )

    res = opt.solve_problem(problem, maxiter=1000)

    assert res.success == True
    assert res.x == pinned.approx(rel=0.01)
