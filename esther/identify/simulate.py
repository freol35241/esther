"""Simulate a model on given inputs (spec §10.1: the positive control).

:func:`simulate` draws a trajectory of the stochastic model — process noise
from the exact discrete covariance, white measurement noise — through the same
discretisation the filter uses, and returns the input frame with the measured
signals filled in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.model import LinearModel, discretise

__all__ = ["simulate"]


def simulate(
    model: LinearModel,
    theta: Sequence[float] | Mapping[str, float],
    frame: pd.DataFrame,
    step: float,
    x0: Sequence[float] | None = None,
    rng: np.random.Generator | None = None,
    noise: bool = True,
    levels: int = 20,
) -> pd.DataFrame:
    """Simulate over every row of ``frame`` (a regular grid of ``step`` s that
    holds the model's input and modulation signals). Returns a copy with the
    model's output columns set, and ``x:<state>`` columns with the true states.
    ``x0`` defaults to 20 °C in every node."""
    rng = rng or np.random.default_rng(0)
    if isinstance(theta, Mapping):
        theta = [theta[n] for n in model.param_names]
    M = model.matrices(theta)
    n = model.n_states
    D = model.input_array(frame)
    if model.modulations:
        S = frame[list(model.modulations)].to_numpy(dtype=float)
        S = np.clip(np.round(S * levels), 0, levels) / levels
    else:
        S = np.zeros((len(frame), 0))
    if S.shape[1]:
        levels_u, inv = np.unique(S, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
    else:
        levels_u, inv = np.zeros((1, 0)), np.zeros(len(frame), dtype=int)
    A = np.stack([M.at(lv)[0] for lv in levels_u])
    E = np.stack([M.at(lv)[1] for lv in levels_u])
    disc = discretise(A, E, M.Qc[None], step)
    Lq = np.stack([np.linalg.cholesky(q + 1e-18 * np.eye(n)) for q in disc.Qd])
    x = np.full(n, 293.15) if x0 is None else np.asarray(x0, dtype=float).copy()
    X = np.zeros((len(frame), n))
    for k in range(len(frame)):
        X[k] = x
        c = inv[k]
        x = disc.Ad[c] @ x + disc.Gd[c] @ D[k]
        if noise:
            x = x + Lq[c] @ rng.standard_normal(n)
    out = frame.copy()
    obs = M.H.argmax(axis=1)
    for o, sig in enumerate(model.outputs):
        y = X[:, obs[o]] + M.c[o]
        if noise:
            y = y + np.sqrt(M.R[o, o]) * rng.standard_normal(len(frame))
        out[sig] = y
    for i, s in enumerate(model.states):
        out[f"x:{s}"] = X[:, i]
    return out
