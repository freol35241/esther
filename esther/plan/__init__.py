"""Problem assembly, solver backend and the Plan type (spec §8).

- :class:`Planner` composes the house in plan mode, collects every
  component's contribution and solves (:mod:`.planner`).
- :class:`MILP` and :class:`SolveOptions` are the solver layer over HiGHS
  (:mod:`.milp`); :mod:`.blocks` holds the inter-step building blocks.
- :func:`plan_to_json` / :func:`plan_from_json` — a Plan as JSON, exactly,
  so a restarted planner warm-starts (:mod:`.serialise`);
  :func:`load_planner` — a planner from a house file and its model file.
"""

from .milp import MILP, Solution, SolveOptions
from .planner import Plan, PlanContext, PlanError, Planner, PlannerOptions, load_planner
from .serialise import plan_from_json, plan_to_json

__all__ = [
    "MILP",
    "Plan",
    "PlanContext",
    "PlanError",
    "Planner",
    "PlannerOptions",
    "Solution",
    "SolveOptions",
    "load_planner",
    "plan_from_json",
    "plan_to_json",
]
