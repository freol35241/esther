"""The live state estimate (spec §6.6, F5): the filter over a recent window.

The planner starts from the state at plan time, including the nodes nothing
measures (a building mass, a tank layer). Live, that is the identification
filter run forward on the most recent measurements with the fitted
parameters: :func:`estimate_state` cuts the frame to ``[now − window, now]``,
segments it exactly as identification does (:meth:`IdData.from_frame`: a gap
ends a segment, nothing is bridged), runs :func:`~.kalman.kalman` and takes
the filtered state and covariance at the newest usable step. A batch filter
over a window costs more than a recursive one, but it is the filter the fit
and the gate used, unchanged, and it cannot drift: every estimate starts
again from the segment prior (``FilterOptions``).

Only steps that have **ended** by ``now`` are used — a step's value is its
mean, not known before its end. The estimate is of the state at the start of
the newest usable step, after that step's measurement (the filter's
convention, :mod:`.kalman`); its ``time`` says which.

It is refused (:class:`StateError`, with a reason) when there is nothing to
estimate from, when the newest usable step is older than ``max_age`` (the
measurements went silent: §8.8, no plan), or when the segment it ends is
shorter than ``min_steps`` — just after a gap, the unmeasured nodes still sit
near the prior, which knows nothing about them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.model import LinearModel
from .data import IdData
from .kalman import FilterOptions, kalman, prepare

__all__ = ["StateError", "StateEstimate", "estimate_state"]


class StateError(RuntimeError):
    """No state estimate: ``reason`` is ``"no data"``, ``"stale"`` or ``"too short"``."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class StateEstimate:
    """The filter's estimate at ``time``: per-state mean (K) and sd (K), the
    covariance in the model's state order, and how many steps of the segment
    led to it."""

    time: pd.Timestamp
    states: tuple[str, ...]
    mean: dict[str, float]
    sd: dict[str, float]
    covariance: np.ndarray
    n_steps: int

    @property
    def x(self) -> np.ndarray:
        """The mean in the model's state order."""
        return np.array([self.mean[s] for s in self.states])


def estimate_state(
    model: LinearModel,
    theta: Sequence[float] | Mapping[str, float],
    frame: pd.DataFrame,
    step: float,
    now,
    *,
    window: float = 86400.0,
    max_age: float = 1800.0,
    min_steps: int = 12,
    optional: Iterable[str] = (),
    options: FilterOptions | None = None,
) -> StateEstimate:
    """Estimate ``model``'s state at ``now`` from ``frame`` (indexed by step
    start, on a grid of ``step`` s, holding the model's inputs, modulations
    and measurements; invalid values NaN) with parameters ``theta``.

    ``window`` (s): how far back the filter starts. ``max_age`` (s): the
    oldest the newest usable step may be, measured from its start to
    ``now``. ``min_steps``: the shortest segment trusted. ``optional``
    measurements may be missing without ending a segment, as in the fit."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError(f"{now!r} has no time zone; times are UTC with an explicit offset")
    if isinstance(theta, Mapping):
        missing = [n for n in model.param_names if n not in theta]
        if missing:
            raise KeyError(f"no value for parameter(s): {', '.join(missing)}")
        theta = [float(theta[n]) for n in model.param_names]
    th = np.asarray(theta, dtype=float)
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise ValueError("the frame needs a tz-aware DatetimeIndex")
    lo = now - pd.Timedelta(seconds=float(window))
    ended = idx + pd.Timedelta(seconds=float(step)) <= now
    cut = frame[(idx >= lo) & ended]
    data = IdData.from_frame(cut, step, model.required_signals(), min_steps=1, optional=optional)
    if not data.frames:
        raise StateError("no data", f"no usable step in the {window / 3600:g} h before {now}")
    last = max(data.frames, key=lambda f: f.index[-1])
    t = last.index[-1]
    age = (now - t).total_seconds()
    if age > max_age:
        raise StateError("stale", f"newest usable step {t} is {age:.0f} s before {now}, over {max_age:g} s")
    if len(last) < min_steps:
        raise StateError("too short", f"the segment ending {t} has {len(last)} steps, fewer than {min_steps}")
    seg = IdData(data.step, [last])
    opts = options or FilterOptions()
    kf = kalman(model, prepare(model, seg, opts), th[None], store=True, options=opts)
    k = len(last) - 1
    x = kf.x_filt[0, 0, k]
    P = kf.P_filt[0, 0, k]
    P = 0.5 * (P + P.T)
    sd = np.sqrt(np.clip(np.diag(P), 0.0, None))
    states = tuple(model.states)
    return StateEstimate(
        time=t,
        states=states,
        mean=dict(zip(states, x.tolist(), strict=True)),
        sd=dict(zip(states, sd.tolist(), strict=True)),
        covariance=P,
        n_steps=len(last),
    )
