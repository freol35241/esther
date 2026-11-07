"""The digital twin: a plant simulated by its fitted model (§10.2).

The model is the identification-mode composition (:func:`compose` with
``mode="identify"``) — the one the fit produced, with the measured inputs a
device reports (a compressor's on-fraction, a heater's share, a supply
temperature, a pump's running) — so the twin is exactly the model the house's
history accepted. A **device callback** stands in for the installed
controllers: given the commands in force, the twin's temperatures and the
exogenous inputs, it returns the inputs the device would produce, and any
bookkeeping (electrical power, fuel) to record. The twin is deterministic:
no process noise, the counterfactual's expected path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.model import LinearModel, discretise

__all__ = ["LinearTwin"]

Device = Callable[[pd.Timestamp, Mapping[str, float], pd.Series, Sequence[Any], float], Mapping[str, float]]


@dataclass
class LinearTwin:
    """``exogenous``: a regular frame (step ``step`` s) of every input
    signal the model reads that the device does not produce — weather,
    draws, gains — in SI. ``device(t, x, exog_row, commands, dt)`` returns
    the device-produced inputs (and extra columns to record) for the step
    starting at ``t``; ``x`` maps each state to its temperature (K)."""

    model: LinearModel
    theta: np.ndarray
    exogenous: pd.DataFrame
    device: Device
    step: float
    x: np.ndarray
    levels: int = 20
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.M = self.model.matrices(self.theta)
        self.x = np.asarray(self.x, dtype=float).copy()
        self.signals = [u.name for u in self.model.inputs if hasattr(u, "name")]

    @property
    def states(self) -> tuple[str, ...]:
        return self.model.states

    def state(self) -> dict[str, float]:
        return dict(zip(self.model.states, self.x.tolist(), strict=True))

    def _disc(self, lev: tuple[float, ...]):
        if lev not in self._cache:
            A, E = self.M.at(list(lev))
            d = discretise(A[None], E[None], self.M.Qc[None], self.step)
            self._cache[lev] = (d.Ad[0], d.Gd[0])
        return self._cache[lev]

    def advance(self, t0: pd.Timestamp, t1: pd.Timestamp, commands: Sequence[Any] = ()) -> pd.DataFrame:
        """Simulate ``[t0, t1)`` under ``commands``; one record per step:
        the states at the step's start, the inputs used, the device's extras."""
        rows = []
        t = pd.Timestamp(t0)
        dt = pd.Timedelta(seconds=self.step)
        while t < t1:
            if t not in self.exogenous.index:
                raise KeyError(f"no exogenous inputs at {t}")
            ex = self.exogenous.loc[t]
            st = self.state()
            out = dict(self.device(t, st, ex, commands, self.step))
            row = ex.to_dict()
            row.update(out)
            missing = [
                s for s in self.signals + list(self.model.modulations) if not np.isfinite(row.get(s, np.nan))
            ]
            if missing:
                raise ValueError(f"{t}: no value for {', '.join(missing)}")
            frame = pd.DataFrame({s: [row[s]] for s in self.signals}, index=pd.DatetimeIndex([t]))
            d = self.model.input_array(frame)[0]
            lev = tuple(
                float(np.clip(np.round(row[m] * self.levels), 0, self.levels) / self.levels)
                for m in self.model.modulations
            )
            Ad, Gd = self._disc(lev)
            rec = {"time": t, **{f"x:{s}": v for s, v in st.items()}, **row}
            rows.append(rec)
            self.x = Ad @ self.x + Gd @ d
            t = t + dt
        return pd.DataFrame(rows).set_index("time") if rows else pd.DataFrame()
