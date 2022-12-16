from dataclasses import dataclass
from typing import List, Tuple, Callable
import logging

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from app.dynamic_model import ModelParameters, simulate

LOGGER = logging.getLogger(__name__)


@dataclass
class ProblemDefinition:
    size: int
    objective: Callable[[np.ndarray], float]
    inequality_constraints: Callable[[np.ndarray], np.ndarray]
    equality_constraints: Callable[[np.ndarray], np.ndarray]
    bounds: List[Tuple]


def prepare_optimization_problem(
    model: ModelParameters,
    electricity_prices: np.ndarray,
    outdoor_temperatures: np.ndarray,
    delta_t: np.ndarray,
    current_indoor_temperature: float,
    requested_indoor_temperature: float,
    minimum_indoor_temperature: float,
    maximum_indoor_temperature: float,
    maximum_feed_temperature: float,
) -> ProblemDefinition:

    electricity_prices = np.asarray(electricity_prices)
    outdoor_temperatures = np.asarray(outdoor_temperatures)
    delta_t = np.asarray(delta_t)

    LOGGER.info("Preparing a new optimization problem.")
    LOGGER.debug("  model=%s", model)
    LOGGER.debug("  electricity_prices=%s", electricity_prices)
    LOGGER.debug("  outdoor_temperatures=%s", outdoor_temperatures)
    LOGGER.debug("  delta_t=%s", delta_t)
    LOGGER.debug("  current_indoor_temperature=%s", current_indoor_temperature)
    LOGGER.debug("  requested_indoor_temperature=%s", requested_indoor_temperature)
    LOGGER.debug("  minimum_indoor_temperature=%s", minimum_indoor_temperature)
    LOGGER.debug("  maximum_indoor_temperature=%s", maximum_indoor_temperature)
    LOGGER.debug("  maximum_feed_temperature=%s", maximum_feed_temperature)

    no_of_variables = len(electricity_prices)
    if len(outdoor_temperatures) != no_of_variables:
        raise RuntimeError(
            "Lengths of electricity_prices and outdoor_temperatures must match!"
        )

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
        timed_weights = delta_t / delta_t.max()
        COPs = map(model.COP_feed, T_feed) if model.COP_feed else np.ones_like(T_feed)
        COP_weights = 1 / np.array(list(COPs))
        return (
            COP_weights * timed_weights * electricity_prices * T_feed
        ).sum() / electricity_prices.sum()

    def _inequality_constraints(T_feed: np.ndarray) -> np.ndarray:
        T_indoor = simulate(
            model, current_indoor_temperature, T_feed, outdoor_temperatures, delta_t
        )
        lower = np.array(T_indoor) - minimum_indoor_temperature
        upper = maximum_indoor_temperature - np.array(T_indoor)
        return np.concatenate([lower, upper])

    def _equality_constraints(T_feed: np.ndarray) -> np.ndarray:
        indoor_temperatures = simulate(
            model, current_indoor_temperature, T_feed, outdoor_temperatures, delta_t
        )

        mean_temperature_constraint = (
            indoor_temperatures.mean() - requested_indoor_temperature
        )
        final_temperature_constraint = (
            indoor_temperatures[-1] - requested_indoor_temperature
        )

        return [mean_temperature_constraint, final_temperature_constraint]

    return ProblemDefinition(
        size=len(outdoor_temperatures),
        objective=_objective,
        inequality_constraints=_inequality_constraints,
        equality_constraints=_equality_constraints,
        bounds=[(minimum_indoor_temperature, maximum_feed_temperature)]
        * no_of_variables,
    )


def solve_problem(
    problem: ProblemDefinition, initial_guess: np.ndarray, **kwargs
) -> OptimizeResult:
    return minimize(
        problem.objective,
        initial_guess,
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
