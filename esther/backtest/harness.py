"""Step-by-step replay (§10.2) and the leak check."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..actuate import Command, translate
from ..forecast.base import Forecaster, LeakError, combine
from ..plan.planner import Plan, PlanError, Planner

__all__ = ["BacktestResult", "PlanningController", "leak_check", "run"]


@dataclass
class BacktestResult:
    """``records``: one row per plant step (states, inputs, the device's
    extras). ``decisions``: one row per planning time (what the controller
    decided, how long it took, whether it fell back)."""

    records: pd.DataFrame
    decisions: pd.DataFrame
    plans: list[Plan] = field(default_factory=list, repr=False)


@dataclass
class PlanningController:
    """esther as the controller: forecasts made at ``now`` from the
    forecasters (checked for provenance, §10.2), the plant's state, a plan
    warm-started from the previous one, and the plan's commands for the
    next interval. On no plan (§8.8) it returns no commands, which the
    device callback must treat as its fallback (§9.2).

    ``devices(plant)``: the discrete devices' state at plan time (a stove's
    burn, a compressor's run), for the planner's inter-step constraints.
    ``keep_plans``: keep every :class:`Plan` (memory) or only summaries.
    ``allow_leaky``: accept forecasters that see the future by design —
    the perfect-foresight bound, and only that."""

    planner: Planner
    forecasters: Sequence[Forecaster]
    devices: Callable[[Any], Mapping[str, Mapping[str, Any]]] | None = None
    allow_leaky: bool = False
    keep_plans: bool = False
    previous: Plan | None = None
    plans: list[Plan] = field(default_factory=list, repr=False)
    log: list[dict] = field(default_factory=list, repr=False)

    def forecast(self, now: pd.Timestamp):
        h = self.planner.house
        return combine([f.forecast(now, h.horizon, h.step) for f in self.forecasters])

    def decide(self, now: pd.Timestamp, plant) -> list[Command]:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"time": now}
        try:
            fc = self.forecast(now)
            rec["forecast_s"] = time.perf_counter() - t0
            dev = self.devices(plant) if self.devices else {}
            plan = self.planner.plan(
                now, plant.state(), fc, devices=dev, previous=self.previous, allow_leaky=self.allow_leaky
            )
        except (PlanError, LookupError) as e:
            rec.update({"status": "no plan", "reason": str(e), "seconds": time.perf_counter() - t0})
            self.log.append(rec)
            self.previous = None
            return []
        except LeakError:
            raise
        self.previous = plan
        if self.keep_plans:
            self.plans.append(plan)
        s = plan.solver
        rec.update(
            {
                "status": s["status"],
                "seconds": time.perf_counter() - t0,
                "solve_seconds": s["solve_seconds"],
                "passes": len(s["passes"]),
                "gap": s["gap"],
                "warm": s["warm_started"],
                "deterministic": s["deterministic"],
                "objective": plan.cost.get("total"),
                "plan_id": plan.id,
                "advisory": plan.advisory,
                "trajectory": plan.trajectories,
            }
        )
        rec.update({f"cost:{k}": v for k, v in plan.cost.items()})
        self.log.append(rec)
        return translate(plan, self.planner.house)


def run(
    plant,
    controller,
    start,
    end,
    every: float,
    progress: Callable[[pd.Timestamp], None] | None = None,
) -> BacktestResult:
    """Replay ``[start, end)``: every ``every`` seconds the controller
    decides (``decide(now, plant) → commands``) and the plant advances
    (``advance(now, next, commands) → records``) to the next decision."""
    t = pd.Timestamp(start)
    end = pd.Timestamp(end)
    step = pd.Timedelta(seconds=every)
    parts = []
    while t < end:
        cmds = controller.decide(t, plant)
        t1 = min(t + step, end)
        parts.append(plant.advance(t, t1, cmds))
        if progress:
            progress(t)
        t = t1
    log = getattr(controller, "log", [])
    dec = pd.DataFrame([{k: v for k, v in r.items() if k != "trajectory"} for r in log])
    if len(dec):
        dec = dec.set_index("time")
    return BacktestResult(
        pd.concat(parts) if parts else pd.DataFrame(), dec, list(getattr(controller, "plans", []))
    )


def leak_check(
    make_forecasters: Callable[[pd.Timestamp | None], Sequence[Forecaster]],
    planner: Planner,
    times: Sequence[pd.Timestamp],
    state: Callable[[pd.Timestamp], Mapping[str, float]],
    devices: Callable[[pd.Timestamp], Mapping[str, Mapping[str, Any]]] | None = None,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Re-plan at each of ``times`` twice: with the forecasters built on the
    full history (``make_forecasters(None)``) and on history **poisoned
    after** ``now`` (``make_forecasters(now)``: every datum later than now
    replaced by something absurd). Information from the future, if any
    forecaster reads it, changes the plan; this raises :class:`LeakError`
    on the first difference. Each row reports the largest difference in the
    decisions and the forecasts.
    """
    rows = []
    h = planner.house
    for now in times:
        now = pd.Timestamp(now)
        out = []
        for cutoff in (None, now):
            fc = combine([f.forecast(now, h.horizon, h.step) for f in make_forecasters(cutoff)])
            out.append(fc)
        diff_fc = float(np.nanmax(np.abs(out[0].values.to_numpy() - out[1].values.to_numpy())))
        if diff_fc > tol:
            col = (out[0].values - out[1].values).abs().max().idxmax()
            raise LeakError(
                f"at {now}: the forecast of {col!r} changes when the future is poisoned ({diff_fc:.3g})"
            )
        dev = devices(now) if devices else {}
        plans = [planner.plan(now, state(now), fc, devices=dev) for fc in out]
        d = (
            plans[0].decisions.select_dtypes("number").to_numpy()
            - plans[1].decisions.select_dtypes("number").to_numpy()
        )
        diff_plan = float(np.nanmax(np.abs(d))) if d.size else 0.0
        if diff_plan > max(tol, 1e-3):
            raise LeakError(f"at {now}: the plan changes when the future is poisoned ({diff_plan:.3g})")
        rows.append({"time": now, "forecast_diff": diff_fc, "plan_diff": diff_plan})
    return pd.DataFrame(rows)
