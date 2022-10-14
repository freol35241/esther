import typing
import math
from dataclasses import dataclass

import numpy as np

DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT = 1 / 450000  # Tidskonstant: 125h


@dataclass
class ModelParameters:
    k1: float
    k2: float


def create_model_from_IVT490_settings(
    heating_curve_slope: float,
    house_cooldown_time_constant: float = DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT,
) -> ModelParameters:
    def heating_curve(slope: float, T_outdoor: float) -> float:
        """Returns the feed temperature for the given slope and outdoor temperature

        Args:
            slope (float): Heating curve slope
            T_outdoor (float): Outdoor temperature

        Returns:
            float: Feed temperature
        """
        return 20 + (-0.16 * slope) * (T_outdoor - 20)

    assumed_T_outdoor = 0.0
    assumed_T_indoor = 20.0
    resulting_T_feed = heating_curve(heating_curve_slope, assumed_T_outdoor)
    k2k1_ratio = (assumed_T_indoor - assumed_T_outdoor) / (
        resulting_T_feed - assumed_T_indoor
    )

    return ModelParameters(
        k1=house_cooldown_time_constant, k2=house_cooldown_time_constant * k2k1_ratio
    )


def _analytical_solution(
    parameters: ModelParameters,
    T_indoor_0: float,
    T_feed: float,
    T_outdoor: float,
    t: float = 3600,
) -> float:
    K = parameters.k1 + parameters.k2
    T_w = (parameters.k2 * T_feed + parameters.k1 * T_outdoor) / K
    delta = T_indoor_0 - T_w
    return T_w + delta * math.exp(-K * t)


def simulate(
    parameters: ModelParameters,
    T_indoor_now: float,
    T_feed: np.ndarray,
    T_outdoor: np.ndarray,
    delta_t: np.ndarray,
):
    T_indoor = []
    Ti = T_indoor_now

    for (Tf, To, dt) in zip(T_feed, T_outdoor, delta_t):
        Ti = _analytical_solution(parameters, Ti, Tf, To, dt)
        T_indoor.append(Ti)

    return np.asarray(T_indoor)
