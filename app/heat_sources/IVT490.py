from functools import partial

import numpy as np

from app.dynamic_model import ModelParameters, DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT


def _heating_curve(slope: float, T_outdoor: float) -> float:
    """Returns the feed temperature for the given slope and outdoor temperature

    Args:
        slope (float): Heating curve slope
        T_outdoor (float): Outdoor temperature

    Returns:
        float: Feed temperature
    """
    return 20 + (-0.16 * slope) * (T_outdoor - 20)


def create_model_from_slope(
    heating_curve_slope: float,
    house_cooldown_time_constant: float = DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT,
) -> ModelParameters:

    assumed_T_outdoor = 0.0
    assumed_T_indoor = 20.0
    resulting_T_feed = _heating_curve(heating_curve_slope, assumed_T_outdoor)
    k2k1_ratio = (assumed_T_indoor - assumed_T_outdoor) / (
        resulting_T_feed - assumed_T_indoor
    )

    return ModelParameters(
        T_outdoor_time_constant=house_cooldown_time_constant,
        T_feed_time_constant=1
        / (0.8 * (1 / house_cooldown_time_constant) * k2k1_ratio),
    )


def make_initial_guess(
    heating_curve_slope: float, outdoor_temperatures: np.ndarray
) -> np.ndarray:
    return np.asarray(
        list(map(partial(_heating_curve, heating_curve_slope), outdoor_temperatures))
    )
