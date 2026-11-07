"""Problem assembly, solver backend and the Plan type (spec §8).

- :class:`Planner` composes the house in plan mode, collects every
  component's contribution and solves (:mod:`.planner`).
- :class:`MILP` and :class:`SolveOptions` are the solver layer over HiGHS
  (:mod:`.milp`); :mod:`.blocks` holds the inter-step building blocks.
"""

from .milp import MILP, Solution, SolveOptions
from .planner import Plan, PlanContext, PlanError, Planner, PlannerOptions

__all__ = [
    "MILP",
    "Plan",
    "PlanContext",
    "PlanError",
    "Planner",
    "PlannerOptions",
    "Solution",
    "SolveOptions",
]
