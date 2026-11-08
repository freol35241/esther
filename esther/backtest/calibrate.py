"""Learning a device habit by replaying it on the twin (§10.2, §4.5.6).

Some of what a controller does is not a threshold on anything the model
holds: a compressor's hot-water run may end on a sensor the model does not
have, or on a rule the controller does not report. What the model needs is
then the **effective** value of the habit — the one at which a replay of the
device on the twin reproduces what the house recorded. :func:`fit_by_replay`
learns one such scalar from routine operation by output error: replay the
device under each candidate, compare the replayed output with the record
over the steps that count, keep the best.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

__all__ = ["fit_by_replay"]


def fit_by_replay(
    replay: Callable[[float], pd.Series],
    candidates: Sequence[float],
    observed: pd.Series,
    valid: pd.Series | None = None,
    period: str = "1D",
) -> dict:
    """The candidate whose replay best reproduces ``observed``.

    ``replay(c)`` returns the replayed output (on ``observed``'s index, or a
    superset) with the habit set to ``c``. The loss is the sum of squared
    differences of the output's **sums per** ``period`` over the ``valid``
    steps: a habit like a stop limit moves how much a device runs, not when
    each run starts, so per-step agreement would mostly score the timing.

    Returns ``best``, ``loss`` (candidate → loss), ``observed_total`` and
    ``replayed_total`` (sums over the valid steps at the best), and
    ``on_bound`` — true when the best is the first or last candidate, so the
    optimum may lie outside the grid and the value is not identified.
    """
    cand = [float(c) for c in candidates]
    if len(cand) < 3:
        raise ValueError("give at least three candidates")
    obs = observed.astype(float)
    ok = np.isfinite(obs.to_numpy())
    if valid is not None:
        ok &= valid.reindex(obs.index).fillna(False).to_numpy(bool)
    if ok.sum() == 0:
        raise ValueError("no valid steps to compare")
    obs_ok = obs[ok]
    sums = obs_ok.groupby(obs_ok.index.floor(period)).sum()
    loss, totals = {}, {}
    for c in cand:
        r = replay(c).reindex(obs.index).astype(float)[ok]
        if not np.isfinite(r.to_numpy()).all():
            raise ValueError(f"the replay at {c:g} has no value at some valid steps")
        rs = r.groupby(r.index.floor(period)).sum().reindex(sums.index)
        loss[c] = float(((rs - sums) ** 2).sum())
        totals[c] = float(r.sum())
    best = min(cand, key=lambda c: loss[c])
    return {
        "best": best,
        "loss": loss,
        "observed_total": float(obs_ok.sum()),
        "replayed_total": totals[best],
        "on_bound": best in (cand[0], cand[-1]),
    }
