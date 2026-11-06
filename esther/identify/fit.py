"""Maximum a posteriori fit over physical priors (spec §6.2, F3).

The objective is the filter's negative log-likelihood plus the negative
log-prior. It is minimised in **standardised prior coordinates**: for each free
parameter, ``t`` is ``log θ`` (log-normal prior) or ``θ`` (normal prior), and
``u = (t − t_prior) / s_prior`` with ``t_prior``, ``s_prior`` the prior's centre
and spread in ``t``. The prior is then ``½ |u|²``, every coordinate is on the
same scale, and finite-difference steps can be the same for all.

- L-BFGS-B with bounds, gradient by forward differences evaluated as one
  batch; multi-start, the first start at the prior centre and the others
  drawn from the prior (§6.2: ≥ 8 starts).
- Uncertainty: the inverse of the central-difference Hessian at the optimum,
  in ``u`` (hence ``t``) space; 95 % intervals are ``t ± 1.96 sd`` mapped back.
- Identifiability: a parameter whose posterior sd in ``u`` is not smaller
  than its prior's (1) by ``identify_factor`` is reported as *not identified
  by the data*; condition number and pairwise correlations are reported.
- **Jointly unidentified directions.** The per-parameter test misses a
  combination the data cannot see while each member's prior props it up —
  the classic case is a common scale: multiplying every capacity,
  conductance and heat-input gain by the same factor leaves the
  temperatures unchanged, so without one heat input of known size the data
  fix only ratios. In ``u`` space the prior's Hessian is the identity, so the
  data's information is ``H − I``; its eigen-directions with eigenvalue below
  ``identify_factor² − 1`` are the ones along which the data shrink the prior
  sd by less than ``identify_factor``. A parameter whose posterior variance
  comes mostly (> ½) from such directions is not identified either, and the
  directions are reported.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.model import LinearModel, is_log
from .data import IdData
from .kalman import FilterOptions, Prepared, kalman, prepare

__all__ = ["FitOptions", "FitResult", "Objective", "fit"]

Z95 = 1.959963984540054
BAD = 1e10


@dataclass(frozen=True)
class FitOptions:
    n_starts: int = 8
    seed: int = 0
    maxiter: int = 400
    fd_step: float = 1e-5  # forward-difference step in u
    batch: int = 64  # parameter vectors per filter call in the Hessian
    identify_factor: float = 2.0
    correlation_flag: float = 0.95
    bound_tol: float = 1e-3  # in u: closer than this to a bound counts as on it
    workers: int = 1  # processes for the starts (set OPENBLAS_NUM_THREADS=1 when > 1)
    filter: FilterOptions = field(default_factory=FilterOptions)


class Objective:
    """Negative log-posterior in standardised coordinates ``u``."""

    def __init__(self, model: LinearModel, prep: Prepared, options: FitOptions):
        self.model, self.prep, self.opts = model, prep, options
        refs = model.free
        self.log = np.array([is_log(r.param) for r in refs], dtype=bool)
        centre = np.array([r.param.prior.centre for r in refs], dtype=float)
        spread = np.array(
            [r.param.prior.gsd if is_log(r.param) else r.param.prior.sd for r in refs], dtype=float
        )
        self.t0 = np.where(self.log, np.log(np.where(self.log, centre, 1.0)), centre)
        self.s = np.where(self.log, np.log(np.where(self.log, spread, np.e)), spread)
        lo = np.array([r.param.bounds[0] for r in refs], dtype=float)
        hi = np.array([r.param.bounds[1] for r in refs], dtype=float)
        self.lo = self.to_u(lo)
        self.hi = self.to_u(hi)
        self.n_evals = 0

    @property
    def p(self) -> int:
        return len(self.t0)

    def to_u(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        t = np.where(self.log, np.log(np.where(self.log, theta, 1.0)), theta)
        return (t - self.t0) / self.s

    def to_theta(self, u: np.ndarray) -> np.ndarray:
        t = self.t0 + np.asarray(u, dtype=float) * self.s
        return np.where(self.log, np.exp(t), t)

    def nll(self, U: np.ndarray) -> np.ndarray:
        """Data term for each row of ``U``; non-finite values become +inf."""
        U = np.atleast_2d(U)
        self.n_evals += len(U)
        with np.errstate(all="ignore"):
            v = kalman(self.model, self.prep, self.to_theta(U), options=self.opts.filter).nll
        return np.where(np.isfinite(v), v, np.inf)

    def batch(self, U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(U)
        out = np.empty(len(U))
        for i in range(0, len(U), self.opts.batch):
            chunk = U[i : i + self.opts.batch]
            out[i : i + len(chunk)] = self.nll(chunk) + 0.5 * (chunk**2).sum(axis=1)
        return out

    def value_and_grad(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        h = self.opts.fd_step
        steps = np.where(u + h <= self.hi, h, -h)
        U = np.vstack([u, u + np.diag(steps)])
        f = self.batch(U)
        if not np.isfinite(f[0]):
            # Where the filter fails (a step to a far bound), a large finite
            # value rising away from the prior centre lets the line search
            # backtrack; +inf would end the run.
            return float(BAD * (1.0 + u @ u)), 2.0 * BAD * u
        g = (f[1:] - f[0]) / steps
        g = np.where(np.isfinite(g), g, 0.0)
        return float(f[0]), g

    def hessian(self, u: np.ndarray) -> np.ndarray:
        """Central differences; steps scaled to each coordinate's curvature."""
        p = self.p
        f0 = self.batch(u[None])[0]
        h = np.full(p, 1e-3)
        for _ in range(2):
            E = np.diag(h)
            f = self.batch(np.vstack([u + E, u - E]))
            d2 = (f[:p] - 2 * f0 + f[p:]) / h**2
            sd = 1.0 / np.sqrt(np.where(d2 > 0, d2, 1.0))
            h = np.clip(0.1 * sd, 1e-5, 1e-2)
        E = np.diag(h)
        pts = [u + E, u - E]
        pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
        for i, j in pairs:
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                pts.append((u + si * E[i] + sj * E[j])[None])
        f = self.batch(np.vstack(pts))
        Hm = np.zeros((p, p))
        Hm[np.arange(p), np.arange(p)] = (f[:p] - 2 * f0 + f[p : 2 * p]) / h**2
        off = f[2 * p :].reshape(-1, 4)
        for (i, j), (pp, pm, mp, mm) in zip(pairs, off, strict=True):
            Hm[i, j] = Hm[j, i] = (pp - pm - mp + mm) / (4 * h[i] * h[j])
        return Hm


@dataclass
class FitResult:
    model: LinearModel
    names: tuple[str, ...]
    theta: np.ndarray
    u: np.ndarray
    objective: float  # negative log-posterior at the optimum
    nll: float  # its data term
    n_obs: int
    hessian: np.ndarray
    cov_u: np.ndarray
    hessian_ok: bool
    starts: pd.DataFrame
    options: FitOptions
    lo95: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hi95: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sd_u: np.ndarray = field(default_factory=lambda: np.zeros(0))
    on_bound: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    identified: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    condition: float = np.nan
    correlation: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n_evals: int = 0
    weak_share: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weak_directions: list[tuple[float, dict[str, float]]] = field(default_factory=list)
    log: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # log-transformed
    t_scale: np.ndarray = field(default_factory=lambda: np.zeros(0))  # prior spread in t

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Central ``level`` interval per parameter, from the Hessian (in ``t``)."""
        z = float(stats.norm.ppf(0.5 + level / 2))
        t = np.where(self.log, np.log(np.where(self.log, self.theta, 1.0)), self.theta)
        sd = self.sd_u * self.t_scale
        lo, hi = t - z * sd, t + z * sd
        return np.where(self.log, np.exp(lo), lo), np.where(self.log, np.exp(hi), hi)

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.names, self.theta.tolist(), strict=True))

    def high_correlations(self) -> list[tuple[str, str, float]]:
        out = []
        p = len(self.names)
        for i in range(p):
            for j in range(i + 1, p):
                r = self.correlation[i, j]
                if np.isfinite(r) and abs(r) >= self.options.correlation_flag:
                    out.append((self.names[i], self.names[j], float(r)))
        return out

    def table(self) -> pd.DataFrame:
        rows = []
        for i, ref in enumerate(self.model.free):
            pr = ref.param.prior
            rows.append(
                {
                    "parameter": self.names[i],
                    "unit": str(ref.param.unit),
                    "value": self.theta[i],
                    "lo95": self.lo95[i],
                    "hi95": self.hi95[i],
                    "prior": (
                        f"LogNormal({pr.median:.4g}, gsd {pr.gsd:g})"
                        if is_log(ref.param)
                        else f"Normal({pr.mean:.4g}, sd {pr.sd:.4g})"
                    ),
                    "sd_ratio": self.sd_u[i],
                    "weak_share": self.weak_share[i],
                    "identified": bool(self.identified[i]),
                    "on_bound": bool(self.on_bound[i]),
                }
            )
        return pd.DataFrame(rows).set_index("parameter")


def _run_start(args) -> dict:
    model, prep, opts, u0 = args
    obj = Objective(model, prep, opts)
    res = minimize(
        obj.value_and_grad,
        u0,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(obj.lo, obj.hi, strict=True)),
        options={"maxiter": opts.maxiter, "maxcor": 20},
    )
    return {
        "u0": u0,
        "u": res.x,
        "f": float(res.fun),
        "nit": int(res.nit),
        "success": bool(res.success),
        "message": str(res.message),
        "evals": obj.n_evals,
    }


def fit(model: LinearModel, data: IdData, options: FitOptions | None = None) -> FitResult:
    """MAP fit of ``model``'s free parameters to ``data`` (spec §6.2)."""
    opts = options or FitOptions()
    if not model.free:
        raise ValueError("the model has no free parameters")
    prep = prepare(model, data, opts.filter)
    obj = Objective(model, prep, opts)
    rng = np.random.default_rng(opts.seed)
    margin = 1e-6
    starts = [np.clip(np.zeros(obj.p), obj.lo + margin, obj.hi - margin)]
    for _ in range(max(0, opts.n_starts - 1)):
        starts.append(np.clip(rng.standard_normal(obj.p), obj.lo + margin, obj.hi - margin))
    jobs = [(model, prep, opts, u0) for u0 in starts]
    if opts.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=opts.workers) as ex:
            runs = list(ex.map(_run_start, jobs))
    else:
        runs = [_run_start(j) for j in jobs]
    best = min(runs, key=lambda r: r["f"])
    u = best["u"]
    Hm = obj.hessian(u)
    Hs = 0.5 * (Hm + Hm.T)
    evals = np.linalg.eigvalsh(Hs)
    ok = bool(np.all(evals > 0) and np.all(np.isfinite(Hs)))
    cov = np.linalg.inv(Hs) if ok else np.linalg.pinv(Hs)
    sd_u = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t = obj.t0 + u * obj.s
    sd_t = sd_u * obj.s
    lo_t, hi_t = t - Z95 * sd_t, t + Z95 * sd_t
    lo = np.where(obj.log, np.exp(lo_t), lo_t)
    hi = np.where(obj.log, np.exp(hi_t), hi_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd_u, sd_u)
    on_bound = (u - obj.lo < opts.bound_tol) | (obj.hi - u < opts.bound_tol)
    lam, vec = np.linalg.eigh(Hs - np.eye(obj.p))
    weak = lam < opts.identify_factor**2 - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        weak_var = (vec[:, weak] ** 2 / np.maximum(lam[weak] + 1.0, 1e-12)).sum(axis=1)
        share = np.clip(weak_var / np.where(sd_u > 0, sd_u**2, np.inf), 0.0, 1.0)
    directions = []
    for j in np.flatnonzero(weak):
        load = {
            model.param_names[i]: float(vec[i, j])
            for i in np.argsort(-np.abs(vec[:, j]))
            if abs(vec[i, j]) >= 0.2
        }
        directions.append((float(lam[j]), load))
    identified = ok & (sd_u * opts.identify_factor <= 1.0) & (share <= 0.5)
    cond = float(evals.max() / evals.min()) if ok else np.inf
    starts_df = pd.DataFrame(
        [
            {
                "start": i,
                "objective": r["f"],
                "iterations": r["nit"],
                "converged": r["success"],
                "message": r["message"],
            }
            for i, r in enumerate(runs)
        ]
    )
    data_nll = float(obj.nll(u[None])[0])
    return FitResult(
        model=model,
        names=model.param_names,
        theta=obj.to_theta(u),
        u=u,
        objective=float(best["f"]),
        nll=data_nll,
        n_obs=prep.n_obs,
        hessian=Hs,
        cov_u=cov,
        hessian_ok=ok,
        starts=starts_df,
        options=opts,
        lo95=lo,
        hi95=hi,
        sd_u=sd_u,
        on_bound=on_bound,
        identified=identified,
        condition=cond,
        correlation=corr,
        n_evals=sum(r["evals"] for r in runs) + obj.n_evals,
        weak_share=share,
        weak_directions=directions,
        log=obj.log,
        t_scale=obj.s,
    )
