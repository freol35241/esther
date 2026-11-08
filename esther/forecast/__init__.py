"""Forecaster interface and implementations (spec §7).

Every forecaster implements ``forecast(now, horizon, step) → Forecast``
(:mod:`.base`): step means on the planning grid, with provenance that a
backtest checks (:func:`check_provenance`). The implementations:

- weather and price from published issues (:mod:`.weather`, §7.1–7.2);
- occupancy, draws and any measured signal from learned profiles
  (:mod:`.profile`, §7.3–7.4);
- device-internal events and manual firing (:mod:`.events`, §7.5–7.6).
"""

from .base import (
    Event,
    Forecast,
    Forecaster,
    Issues,
    LeakError,
    Perfect,
    Provenance,
    check_provenance,
    combine,
    plan_grid,
    step_means,
    until,
)
from .events import ManualFiringDetector, PeriodicEventForecaster, detect_episodes
from .profile import (
    DrawForecaster,
    HoldForecaster,
    OccupancyForecaster,
    ProfileForecaster,
    TimeOfWeekProfile,
)
from .weather import PriceForecaster, WeatherForecaster

__all__ = [
    "DrawForecaster",
    "Event",
    "Forecast",
    "Forecaster",
    "HoldForecaster",
    "Issues",
    "LeakError",
    "ManualFiringDetector",
    "OccupancyForecaster",
    "Perfect",
    "PeriodicEventForecaster",
    "PriceForecaster",
    "ProfileForecaster",
    "Provenance",
    "TimeOfWeekProfile",
    "WeatherForecaster",
    "check_provenance",
    "combine",
    "detect_episodes",
    "plan_grid",
    "step_means",
    "until",
]
