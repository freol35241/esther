import math
from dataclasses import dataclass

import numpy as np

DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT = 125 * 3600  # seconds


@dataclass
class ModelParameters:
    T_outdoor_time_constant: float
    T_feed_time_constant: float


def _analytical_solution(
    parameters: ModelParameters,
    T_indoor_0: float,
    T_feed: float,
    T_outdoor: float,
    t: float = 3600,
) -> float:
    k_outdoor = 1 / parameters.T_outdoor_time_constant
    k_feed = 1 / parameters.T_feed_time_constant
    K = k_outdoor + k_feed
    T_w = (k_feed * T_feed + k_outdoor * T_outdoor) / K
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
