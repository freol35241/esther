"""Replay harness and scoring (spec §10).

- :class:`LinearTwin` — a plant simulated by a fitted model in
  identification mode (the house's digital twin), with a device callback
  that turns commands into the inputs the device would produce.
- :func:`run` — replays a period step by step: at each planning time the
  controller sees the plant's state and forecasts as issued before then,
  decides, and the plant advances under its commands.
- :class:`PlanningController` — esther's planner as such a controller.
- :func:`score` — cost, comfort (K·h outside the band), hot-water service,
  source hours and starts; :func:`prediction_errors` — the plans' predicted
  trajectories against what was realised, at 1/6/24 h (§10.2).
- :func:`fit_by_replay` — learns a device habit (a scalar) as the value at
  which a replay of the device on the twin reproduces the record.
- :func:`leak_check` — re-plans with every datum after ``now`` poisoned and
  requires the same plan: a forecaster that reads the future cannot pass.
"""

from .calibrate import fit_by_replay
from .harness import BacktestResult, PlanningController, leak_check, run
from .score import degree_hours, prediction_errors, score, solve_time_summary
from .twin import LinearTwin

__all__ = [
    "BacktestResult",
    "LinearTwin",
    "PlanningController",
    "degree_hours",
    "fit_by_replay",
    "leak_check",
    "prediction_errors",
    "run",
    "score",
    "solve_time_summary",
]
