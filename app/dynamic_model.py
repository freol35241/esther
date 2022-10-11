import math
import numpy as np

DEFAULT_K1_VALUE = 1 / 450000  # Tidskonstant: 125h


class HeatedHouseWithSingleHeatingSource:
    def __init__(self, k1: float, k2: float):
        self._k1 = k1
        self._k2 = k2

    @staticmethod
    def _analytical_solution(
        k1: float,
        k2: float,
        T_indoor_0: float,
        T_feed: float,
        T_outdoor: float,
        t: float = 3600,
    ) -> float:
        K = k1 + k2
        T_w = (k2 * T_feed + k1 * T_outdoor) / K
        delta = T_indoor_0 - T_w
        return T_w + delta * math.exp(-K * t)

    def simulate(
        self,
        T_indoor_now: float,
        T_feed: np.ndarray,
        T_outdoor: np.ndarray,
        delta_t: float = 3600,
    ):
        T_indoor = []
        Ti = T_indoor_now

        for (Tf, To) in zip(T_feed, T_outdoor):
            Ti = self._analytical_solution(self._k1, self._k2, Ti, Tf, To, delta_t)
            T_indoor.append(Ti)

        return np.asarray(T_indoor)


class HeatedHouseWithIVT490(HeatedHouseWithSingleHeatingSource):
    def __init__(self, heating_curve_slope: float, k1: float):

        self.heating_curve_slope = heating_curve_slope

        assumed_T_outdoor = 0.0
        assumed_T_indoor = 20.0
        resulting_T_feed = self.heating_curve(
            self.heating_curve_slope, assumed_T_outdoor
        )
        k2k1_ratio = (assumed_T_indoor - assumed_T_outdoor) / (
            resulting_T_feed - assumed_T_indoor
        )

        super().__init__(k1, k1 * k2k1_ratio)

    @staticmethod
    def heating_curve(slope: float, T_outdoor: float) -> float:
        """Returns the feed temperature for the given slope and outdoor temperature

        Args:
            slope (float): Heating curve slope
            T_outdoor (float): Outdoor temperature

        Returns:
            float: Feed temperature
        """
        return 20 + (-0.16 * slope) * (T_outdoor - 20)

    @staticmethod
    def inverse_heating_curve(slope: float, T_feed: float) -> float:
        """Returns the outdoor temperature for the given slope and feed temperature

        Args:
            slope (float): Heating curve slope
            T_feed (float): Feed temperature

        Returns:
            float: Outdoor temperature
        """
        return (T_feed - 20) / (-0.16 * slope) + 20
