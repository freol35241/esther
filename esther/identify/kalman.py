"""Continuous-discrete linear Kalman filter (spec §6.2), batched.

The likelihood of a parameter vector is the Gaussian likelihood of the
filter's one-step innovations, summed over segments; every segment restarts
from the initial-state prior (§6.1). The filter is written for a **batch** of
parameter vectors at once — all segments of all vectors advance together, one
time step per Python iteration — because the fit's finite-difference gradient
and Hessian need many evaluations at nearby points, and the cost of a step is
then shared.

Modulated conductances (:mod:`esther.core.model`) make the discrete matrices
depend on the step. Modulation signals are rounded to multiples of
``1/levels`` (default 1/20) and the model is discretised once per distinct
combination in the data.

Conventions: the state ``x_k`` is at the start of step ``k``; inputs ``d_k``
are the step's means, held over it (zero-order hold); the measurement ``y_k``
is the step's mean, taken as an observation of ``x_k``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..core.model import LinearModel, discretise
from .data import IdData

__all__ = ["FilterOptions", "KalmanResult", "Prepared", "kalman", "open_loop", "prepare", "system"]

LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class FilterOptions:
    """Initial-state prior of each segment.

    A measured node starts at its first measurement (bias removed) with sd
    ``x0_sd_measured``; an unmeasured node at the mean of the first
    measurements with sd ``x0_sd_unmeasured`` — it is the filter's job to
    find it.
    """

    x0_sd_measured: float = 1.0
    x0_sd_unmeasured: float = 3.0
    levels: int = 20


@dataclass
class Prepared:
    """A model's view of an :class:`IdData`: padded arrays, segments × steps."""

    step: float
    D: np.ndarray  # (S, L, n_in) inputs
    Y: np.ndarray  # (S, L, n_out) measurements
    mask: np.ndarray  # (S, L) bool: step exists
    combo: np.ndarray  # (S, L) int: modulation-level combination of the step
    levels: np.ndarray  # (K, n_mod) modulation values of each combination
    lengths: np.ndarray  # (S,)
    obs_node: np.ndarray  # (n_out,) state index each output measures
    index: list  # per segment, its DatetimeIndex

    @property
    def n_obs(self) -> int:
        return int(self.mask.sum()) * self.Y.shape[2]


def prepare(model: LinearModel, data: IdData, options: FilterOptions | None = None) -> Prepared:
    opts = options or FilterOptions()
    if not data.frames:
        raise ValueError("no segments")
    S = len(data.frames)
    L = max(len(f) for f in data.frames)
    nd, ny, nm = len(model.inputs), len(model.outputs), len(model.modulations)
    D = np.zeros((S, L, nd))
    Y = np.zeros((S, L, ny))
    mask = np.zeros((S, L), dtype=bool)
    mods = np.zeros((S, L, nm))
    for s, f in enumerate(data.frames):
        n = len(f)
        D[s, :n] = model.input_array(f)
        if ny:
            Y[s, :n] = f[list(model.outputs)].to_numpy(dtype=float)
        if nm:
            mods[s, :n] = f[list(model.modulations)].to_numpy(dtype=float)
        mask[s, :n] = True
    if not (np.isfinite(D).all() and np.isfinite(Y).all() and np.isfinite(mods).all()):
        raise ValueError("non-finite values in the data; build it with IdData.from_dataset/from_frame")
    if nm:
        q = np.clip(np.round(mods * opts.levels), 0, opts.levels) / opts.levels
        # The zero-padded tail must not create a combination of its own.
        q[~mask] = q[mask][0]
        levels, inv = np.unique(q.reshape(-1, nm), axis=0, return_inverse=True)
        combo = inv.reshape(S, L)
    else:
        levels, combo = np.zeros((1, 0)), np.zeros((S, L), dtype=int)
    H = model.matrices(model.prior_centre()).H
    obs_node = H.argmax(axis=1) if ny else np.zeros(0, dtype=int)
    return Prepared(
        step=data.step,
        D=D,
        Y=Y,
        mask=mask,
        combo=combo,
        levels=levels,
        lengths=np.array([len(f) for f in data.frames]),
        obs_node=obs_node,
        index=[f.index for f in data.frames],
    )


@dataclass
class System:
    """Discrete matrices for a batch of ``m`` parameter vectors on ``K`` levels."""

    Ad: np.ndarray  # (m, K, n, n)
    Gd: np.ndarray  # (m, K, n, n_in)
    Qd: np.ndarray  # (m, K, n, n)
    c: np.ndarray  # (m, n_out)
    R: np.ndarray  # (m, n_out, n_out)


def system(model: LinearModel, prep: Prepared, thetas: np.ndarray) -> System:
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    As, Es, Qs, cs, Rs = [], [], [], [], []
    s = np.c_[np.ones(len(prep.levels)), prep.levels]  # (K, 1 + n_mod)
    for th in thetas:
        M = model.matrices(th)
        As.append(np.tensordot(s, M.A, 1))
        Es.append(np.tensordot(s, M.E, 1))
        Qs.append(M.Qc)
        cs.append(M.c)
        Rs.append(M.R)
    A, E = np.stack(As), np.stack(Es)
    Qc = np.stack(Qs)[:, None]
    disc = discretise(A, E, Qc, prep.step)
    return System(disc.Ad, disc.Gd, disc.Qd, np.stack(cs), np.stack(Rs))


@dataclass
class KalmanResult:
    nll: np.ndarray  # (m,) negative log-likelihood
    # with store=True, for each parameter vector:
    innov: np.ndarray | None = None  # (m, S, L, n_out)
    innov_var: np.ndarray | None = None  # (m, S, L, n_out) diagonal of S
    x_filt: np.ndarray | None = None  # (m, S, L, n) after the update at k
    P_filt: np.ndarray | None = None  # (m, S, L, n, n)


def _x0(prep: Prepared, c: np.ndarray, n: int, opts: FilterOptions) -> tuple[np.ndarray, np.ndarray]:
    m = c.shape[0]
    y0 = prep.Y[:, 0, :][None] - c[:, None, :]  # (m, S, ny)
    S = y0.shape[1]
    x0 = np.repeat(y0.mean(axis=2, keepdims=True), n, axis=2)
    sd = np.full(n, opts.x0_sd_unmeasured)
    for i in range(n):
        sel = prep.obs_node == i
        if sel.any():
            x0[:, :, i] = y0[:, :, sel].mean(axis=2)
            sd[i] = opts.x0_sd_measured
    P0 = np.broadcast_to(np.diag(sd**2), (m, S, n, n)).copy()
    return x0, P0


def _input_drive(sysm: System, prep: Prepared) -> np.ndarray:
    """``Gd(level_k) d_k`` for every step, lanes last: (L, n, m·S)."""
    m, K, n, _ = sysm.Gd.shape
    S, L, _ = prep.D.shape
    out = np.zeros((L, n, m, S))
    if K == 1:
        out[:] = np.einsum("mij,slj->lims", sysm.Gd[:, 0], prep.D, optimize=True)
    else:
        for lv in range(K):
            sel = prep.combo == lv  # (S, L)
            if sel.any():
                s_idx, l_idx = np.nonzero(sel)
                v = np.einsum("mij,tj->tim", sysm.Gd[:, lv], prep.D[s_idx, l_idx], optimize=True)
                out[l_idx, :, :, s_idx] = v
    return out.reshape(L, n, m * S)


def kalman(
    model: LinearModel,
    prep: Prepared,
    thetas: np.ndarray,
    store: bool = False,
    options: FilterOptions | None = None,
) -> KalmanResult:
    """Run the filter for each row of ``thetas`` (natural units).

    Internally every (parameter vector, segment) pair is a *lane*, and arrays
    keep the lanes on the last axis — ``x`` is (n, B), ``P`` (n, n, B) — so
    each step is a few vectorised operations over B lanes rather than B tiny
    matrix products (an order of magnitude faster for n ≤ 3).
    """
    opts = options or FilterOptions()
    sysm = system(model, prep, thetas)
    m, K, n, _ = sysm.Ad.shape
    S, L, ny = prep.Y.shape
    B = m * S
    mi = np.repeat(np.arange(m), S)  # lane -> parameter vector
    si = np.tile(np.arange(S), m)  # lane -> segment
    on = prep.obs_node
    x0, P0 = _x0(prep, sysm.c, n, opts)  # (m, S, n), (m, S, n, n)
    x = np.ascontiguousarray(x0.reshape(B, n).T)
    P = np.ascontiguousarray(np.moveaxis(P0.reshape(B, n, n), 0, -1))
    # discrete matrices per (vector, level), lanes last
    # (n, n, m·K), C-contiguous; gathered with np.take, whose result is
    # contiguous too (fancy indexing on the last axis is not, and einsum on
    # the strided result is three times slower).
    Ad_t = np.ascontiguousarray(np.moveaxis(sysm.Ad.reshape(m * K, n, n), 0, -1))
    Qd_t = np.ascontiguousarray(np.moveaxis(sysm.Qd.reshape(m * K, n, n), 0, -1))
    drive = _input_drive(sysm, prep)  # (L, n, B)
    Y = np.ascontiguousarray(np.take(np.moveaxis(prep.Y, 0, -1), si, axis=-1))  # (L, ny, B)
    c = np.ascontiguousarray(sysm.c[mi].T)  # (ny, B)
    R = np.ascontiguousarray(np.moveaxis(sysm.R[mi], 0, -1))  # (ny, ny, B)
    E = np.zeros((L, ny, B))
    V = np.zeros((L, ny, B))
    LD = np.zeros((L, B)) if ny > 1 else None
    QF = np.zeros((L, B)) if ny > 1 else None
    if store:
        xf = np.zeros((L, n, B))
        Pf = np.zeros((L, n, n, B))
    if K == 1:
        Ad = np.take(Ad_t, mi, axis=2)
        Qd = np.take(Qd_t, mi, axis=2)
    else:
        lane_idx = np.ascontiguousarray((mi * K)[None, :] + prep.combo[si].T)  # (L, B)
    if ny == 1:
        i0 = int(on[0])
        R1 = R[0, 0]
        c1 = c[0]
    for k in range(L):
        if ny == 1:
            e = Y[k, 0] - x[i0] - c1  # (B,)
            PHt = P[:, i0]  # (n, B)
            s = PHt[i0] + R1
            Kg = PHt / s
            x = x + Kg * e
            P = P - Kg[:, None] * PHt[None]
            E[k, 0] = e
            V[k, 0] = s
        elif ny:
            e = Y[k] - x[on] - c  # (ny, B)
            PHt = P[:, on]  # (n, ny, B)
            Sk = np.moveaxis(PHt[on] + R, -1, 0)  # (B, ny, ny)
            Sinv = np.moveaxis(np.linalg.inv(Sk), 0, -1)  # (ny, ny, B)
            Kg = np.einsum("ijb,jkb->ikb", PHt, Sinv)
            x = x + np.einsum("ijb,jb->ib", Kg, e)
            P = P - np.einsum("ijb,kjb->ikb", Kg, PHt)
            LD[k] = np.linalg.slogdet(Sk)[1]
            QF[k] = np.einsum("ib,ijb,jb->b", e, Sinv, e)
            E[k] = e
            V[k] = np.einsum("iib->ib", PHt[on] + R)
        if k % 16 == 0:
            P = 0.5 * (P + P.transpose(1, 0, 2))
        if store:
            xf[k] = x
            Pf[k] = P
        if k == L - 1:
            break
        if K > 1:
            Ad = np.take(Ad_t, lane_idx[k], axis=2)
            Qd = np.take(Qd_t, lane_idx[k], axis=2)
        x = np.einsum("ijb,jb->ib", Ad, x) + drive[k]
        P = np.einsum("ikb,lkb->ilb", np.einsum("ijb,jkb->ikb", Ad, P), Ad) + Qd
    mask = prep.mask.T[:, si]  # (L, B)
    if ny == 1:
        v = V[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.log(v) + E[:, 0] ** 2 / v + LOG_2PI
    elif ny > 1:
        term = LD + QF + ny * LOG_2PI
    else:
        term = np.zeros((L, B))
    nll = 0.5 * np.where(mask, term, 0.0).reshape(L, m, S).sum(axis=(0, 2))
    res = KalmanResult(nll=nll)
    if store:

        def lanes(a: np.ndarray) -> np.ndarray:  # (L, ..., B) -> (m, S, L, ...)
            return np.moveaxis(a, -1, 0).reshape(m, S, *a.shape[:-1])

        res.innov = lanes(E)
        res.innov_var = lanes(V)
        res.x_filt = lanes(xf)
        res.P_filt = lanes(Pf)
    return res


def open_loop(
    model: LinearModel,
    prep: Prepared,
    theta: Sequence[float],
    starts: Sequence[tuple[int, int]],
    x_start: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate from ``x_start[w]`` at step ``k0`` of segment ``s`` for each
    window ``w = (s, k0)`` in ``starts``, with measured inputs and no
    measurement feedback, for ``horizon`` steps.

    Returns ``(pred, valid)``: ``pred`` (W, horizon, n_out) are the predicted
    measurements at steps ``k0 + 1 … k0 + horizon``; ``valid`` (W, horizon)
    is false past the segment end.
    """
    sysm = system(model, prep, np.asarray(theta)[None])
    Ad, Gd, c = sysm.Ad[0], sysm.Gd[0], sysm.c[0]
    W = len(starts)
    n = Ad.shape[-1]
    ny = prep.Y.shape[2]
    seg = np.array([s for s, _ in starts], dtype=int)
    k0 = np.array([k for _, k in starts], dtype=int)
    x = np.asarray(x_start, dtype=float).reshape(W, n).copy()
    pred = np.zeros((W, horizon, ny))
    valid = np.zeros((W, horizon), dtype=bool)
    L = prep.D.shape[1]
    for j in range(horizon):
        k = k0 + j  # the step whose inputs drive x_k -> x_{k+1}
        kk = np.minimum(k, L - 1)
        cb = prep.combo[seg, kk]
        d = prep.D[seg, kk]
        x = (Ad[cb] @ x[..., None])[..., 0] + (Gd[cb] @ d[..., None])[..., 0]
        pred[:, j] = x[:, prep.obs_node] + c
        valid[:, j] = (k + 1) < prep.lengths[seg]
    return pred, valid
