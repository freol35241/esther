"""Building blocks for component contributions to the MILP (spec §8.2–8.3).

Inter-step constraints — minimum run and off times, start indicators,
ignition ramps — refer to steps before the horizon. Those are not variables
but known history (the device's state at plan time), so each helper takes the
history as constants: ``hist[0]`` is the step just before the horizon,
``hist[1]`` the one before that, and so on.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Linear", "lag", "lag_sum", "window"]


class Linear:
    """A per-step linear expression ``Σ coef · x[idx] + const`` over ``n``
    steps: ``idx`` and ``coef`` of shape ``(n, k)``, ``const`` of shape
    ``(n,)``. Entries with coefficient 0 are absent."""

    def __init__(self, idx: np.ndarray, coef: np.ndarray, const: np.ndarray) -> None:
        self.idx = np.atleast_2d(np.asarray(idx)).reshape(len(const), -1)
        self.coef = np.asarray(coef, dtype=float).reshape(self.idx.shape)
        self.const = np.asarray(const, dtype=float)

    @classmethod
    def of(cls, idx: np.ndarray, coef: float | np.ndarray = 1.0) -> Linear:
        idx = np.asarray(idx)
        n = idx.shape[0]
        c = np.broadcast_to(np.asarray(coef, dtype=float), idx.shape)
        return cls(idx.reshape(n, -1), c.reshape(n, -1), np.zeros(n))

    def __add__(self, other: Linear | float | np.ndarray) -> Linear:
        if isinstance(other, Linear):
            return Linear(np.c_[self.idx, other.idx], np.c_[self.coef, other.coef], self.const + other.const)
        return Linear(self.idx, self.coef, self.const + np.asarray(other, dtype=float))

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self) -> Linear:
        return Linear(self.idx, -self.coef, -self.const)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Linear) else -np.asarray(other, dtype=float))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, k: float | np.ndarray) -> Linear:
        k = np.asarray(k, dtype=float)
        kk = k[:, None] if k.ndim == 1 else k
        return Linear(self.idx, self.coef * kk, self.const * k)

    __rmul__ = __mul__

    def rows(self, lb=-np.inf, ub=np.inf):
        """``(terms, lb, ub)`` for :meth:`MILP.add`, the constant moved to the bounds."""
        return (
            [(self.idx, self.coef)],
            np.asarray(lb, dtype=float) - self.const,
            np.asarray(ub, dtype=float) - self.const,
        )

    def value(self, x: np.ndarray) -> np.ndarray:
        return (self.coef * x[self.idx]).sum(axis=1) + self.const


def lag(idx: np.ndarray, k: int, hist: np.ndarray | float = 0.0) -> Linear:
    """``v_{t−k}`` for each step ``t`` of the per-step variables ``idx``
    (shape ``(n,)``); before the horizon, from ``hist``."""
    idx = np.asarray(idx)
    n = len(idx)
    h = (
        np.broadcast_to(np.asarray(hist, dtype=float), (max(k, 1),))
        if np.ndim(hist) == 0
        else np.asarray(hist)
    )
    ix = np.zeros((n, 1), dtype=int)
    C = np.zeros((n, 1))
    const = np.zeros(n)
    for t in range(n):
        s = t - k
        if s >= 0:
            ix[t, 0], C[t, 0] = idx[s], 1.0
        else:
            j = -s - 1
            ix[t, 0] = idx[0]
            const[t] = float(h[j]) if j < len(h) else 0.0
    return Linear(ix, C, const)


def window(idx: np.ndarray, w: int, hist: np.ndarray | float = 0.0) -> Linear:
    """``Σ_{j=0}^{w−1} v_{t−j}`` for each step ``t``; before the horizon,
    from ``hist``."""
    out = lag(idx, 0, hist)
    for j in range(1, w):
        out = out + lag(idx, j, hist)
    return out


def lag_sum(idx: np.ndarray, k: int, hist: np.ndarray | float = 0.0) -> Linear:
    """``Σ_j v_{t−k, j}`` over the columns of ``idx`` (shape ``(n, m)``: one-hot
    levels, say); before the horizon, the sum is ``hist``."""
    idx = np.asarray(idx)
    out = lag(idx[:, 0], k, hist)
    for j in range(1, idx.shape[1]):
        out = out + lag(idx[:, j], k, 0.0)
    return out
