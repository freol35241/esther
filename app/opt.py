from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Callable
from functools import partial

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from app.dynamic_model import ModelParameters, simulate


@dataclass
class ProblemDefinition:
    objective: Callable[[np.ndarray], float]
    inequality_constraints: Callable[[np.ndarray], np.ndarray]
    equality_constraints: Callable[[np.ndarray], np.ndarray]
    bounds: List[Tuple]
    initial_guess: np.ndarray


def prepare_optimization_problem(
    model: ModelParameters,
    electricity_prices: np.ndarray,
    outdoor_temperatures: np.ndarray,
    current_indoor_temperature: float,
    requested_indoor_temperature: float,
    minimum_indoor_temperature: float,
    maximum_indoor_temperature: float,
    maximum_feed_temperature: float,
) -> ProblemDefinition:

    electricity_prices = np.asarray(electricity_prices)
    outdoor_temperatures = np.asarray(outdoor_temperatures)

    no_of_variables = len(electricity_prices)
    if len(outdoor_temperatures) != no_of_variables:
        raise RuntimeError(
            "Lengths of electricity_prices and outdoor_temperatures must match!"
        )

    delta_t = np.ones_like(electricity_prices) * 3600
    now = datetime.now()
    delta_t[0] -= now.minute * 60 + now.second

    minimum_indoor_temperature = min(
        minimum_indoor_temperature, current_indoor_temperature
    )
    maximum_indoor_temperature = max(
        maximum_indoor_temperature, current_indoor_temperature
    )

    def _objective(T_feed: np.ndarray) -> float:
        """Objective function to be optimized. The feed temperature acts as a proxy for the required power.

        Args:
            T_feed (np.ndarray): Feed temperatures, one for each hour

        Returns:
            float: Proxy for total cost
        """
        return (electricity_prices * T_feed).sum() / electricity_prices.sum()

    def _inequality_constraints(T_feed: np.ndarray) -> np.ndarray:
        T_indoor = simulate(
            model, current_indoor_temperature, T_feed, outdoor_temperatures, delta_t
        )
        lower = np.array(T_indoor) - minimum_indoor_temperature
        upper = maximum_indoor_temperature - np.array(T_indoor)
        return np.concatenate([lower, upper])

    def _equality_constraints(T_feed: np.ndarray) -> np.ndarray:
        return (
            simulate(
                model, current_indoor_temperature, T_feed, outdoor_temperatures, delta_t
            )[-1]
            - requested_indoor_temperature
        )

    return ProblemDefinition(
        objective=_objective,
        inequality_constraints=_inequality_constraints,
        equality_constraints=_equality_constraints,
        bounds=[(minimum_indoor_temperature, maximum_feed_temperature)]
        * no_of_variables,
        initial_guess=np.ones_like(outdoor_temperatures) * current_indoor_temperature,
    )


def solve_problem(problem: ProblemDefinition, **kwargs) -> OptimizeResult:
    return minimize(
        problem.objective,
        problem.initial_guess,
        method="SLSQP",
        constraints=[
            {
                "type": "ineq",
                "fun": problem.inequality_constraints,
            },
            {"type": "eq", "fun": problem.equality_constraints},
        ],
        bounds=problem.bounds,
        options=kwargs,
    )
