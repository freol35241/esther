"""Actuation translators (spec §8.7): a plan into device commands.

Each component that steers a device turns the plan into
:class:`Command`\\ s (:meth:`~esther.core.component.Component.actuate`);
:func:`translate` collects them. Every command carries the plan's id and
expiry, and an **advisory** plan (§8.8: no fit has passed the gate) gives
commands that say so — :attr:`Command.actionable` is false and an executor
MUST NOT act on them.

- Fake outdoor temperature for a curve-controlled device: the supply target
  through the inverse curve, clamped to the accepted range.
- Fake tank-sensor value for a thermostat-charged tank: the mode, as the
  device's interface defines it.
- A discrete schedule for a stove: start time and level, stop time, as
  absolute times, so that an executor can hold them if the planner falls
  silent (§9.2: the stop is part of the start).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

__all__ = ["Command", "stove_schedule", "translate"]


@dataclass(frozen=True)
class Command:
    """One command for one device: ``kind`` (``"mode"``, ``"supply_target"``,
    ``"fake_outdoor"``, ``"start"``, ``"stop"``, ...) with ``value``, to hold
    over ``[start, end)``. ``expires``: the plan's expiry; ``advisory``: the
    plan is advisory and the command is not to be acted on."""

    device: str
    kind: str
    value: Any
    start: pd.Timestamp
    end: pd.Timestamp
    plan_id: str
    expires: pd.Timestamp
    advisory: bool

    @property
    def actionable(self) -> bool:
        return not self.advisory

    @classmethod
    def of(cls, plan, device: str, kind: str, value, start, end) -> Command:
        return cls(device, kind, value, start, end, plan.id, plan.valid_until, plan.advisory)


def stove_schedule(plan, device: str, level_column: str, start_column: str | None = None) -> list[Command]:
    """The plan's next burn as absolute times: ``start`` (with its level)
    and ``stop``, and level changes in between. A burn under way at the
    plan's start has no start command, only its stop — unless the plan
    starts it in its first step (``start_column``, the plan's start
    indicator), which is a start now. A plan with the stove off in its
    first step begins with a stop now: the plan does not know whether it
    burns, and a stop of a cold stove does nothing."""
    lv = plan.decisions[level_column].to_numpy()
    idx = plan.index
    end_t = idx[-1] + pd.Timedelta(seconds=float(plan.dt[-1]))
    out = []
    k = 0
    n = len(lv)
    starts_now = (
        start_column is not None
        and start_column in plan.decisions
        and lv[0] > 0
        and float(plan.decisions[start_column].iloc[0]) > 0.5
    )
    if starts_now:
        out.append(Command.of(plan, device, "start", int(lv[0]), idx[0], idx[0]))
    elif lv[0] <= 0:
        # off now: a stop now (a no-op for a stove that is not burning, and
        # the end of one that is)
        out.append(Command.of(plan, device, "stop", 0, idx[0], idx[0]))
        while k < n and lv[k] <= 0:
            k += 1
        if k == n:
            return out
        out.append(Command.of(plan, device, "start", int(lv[k]), idx[k], idx[k]))
    cur = int(lv[k])
    j = k + 1
    while j < n and lv[j] > 0:
        if int(lv[j]) != cur:
            cur = int(lv[j])
            out.append(Command.of(plan, device, "level", cur, idx[j], idx[j]))
        j += 1
    stop = idx[j] if j < n else end_t
    out.append(Command.of(plan, device, "stop", 0, stop, stop))
    return out


def translate(plan, house) -> list[Command]:
    """Every steering component's commands for ``plan``."""
    out: list[Command] = []
    for comp in house.components.values():
        out.extend(comp.actuate(plan))
    return out
