"""A small sparse MILP builder over HiGHS (spec §8, N1, N2).

The planner writes its problem as blocks of variables and vectorised rows of
linear constraints, and this module hands the matrix to HiGHS directly
(``highspy``). Building the matrices here rather than through a modelling
layer does two things the planner needs: a **warm start from any point** —
the previous plan shifted by a step, given to HiGHS as a (partial) solution
it completes and uses as its first incumbent — and a build that costs
milliseconds, so a re-plan every step is cheap (§3.2: the M0 spike chose
cvxpy for its build time; this is faster still and gives the warm start
cvxpy's HiGHS interface does not).

Stopping (N1, N2): a solve stops at the relative gap, at a **node limit**
(deterministic: the same problem stops at the same point on any machine) or
at a wall-clock **time limit** (a safety net; a solve stopped by it is
flagged as not reproducible). In both limit cases the best incumbent is
returned, flagged; with none, there is no solution.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp

__all__ = ["MILP", "Solution", "SolveOptions"]

INF = float("inf")


@dataclass(frozen=True)
class SolveOptions:
    """How long and how hard to solve.

    ``gap``: relative MIP gap at which the solve stops as optimal.
    ``time_limit``: wall-clock seconds (N1: under 10 s, should be under 2 s).
    ``node_limit``: branch-and-bound nodes, the deterministic budget (N2);
    ``None`` for none. ``threads`` 1 and a fixed ``seed`` keep HiGHS
    deterministic.
    """

    gap: float = 0.01
    time_limit: float = 10.0
    node_limit: int | None = None
    threads: int = 1
    seed: int = 0
    presolve: bool = True
    verbose: bool = False


@dataclass
class Solution:
    """What a solve returned.

    ``status``: ``"optimal"`` (within the gap), ``"limit"`` (a node or time
    limit with an incumbent — usable, flagged), ``"infeasible"``, or
    ``"failed"`` (no solution: a limit without an incumbent, or a solver
    error). ``deterministic`` is false when the wall clock stopped the solve.
    """

    status: str
    x: np.ndarray | None
    objective: float
    bound: float
    gap: float
    seconds: float
    nodes: int
    deterministic: bool
    reason: str = ""
    warm_started: bool = False

    @property
    def ok(self) -> bool:
        return self.x is not None and self.status in ("optimal", "limit")


def _column(c: np.ndarray, ndim: int) -> np.ndarray:
    """A per-row coefficient (shape ``(m,)``) against 2-d indices ``(m, k)``
    broadcasts along the rows, not the columns."""
    if ndim == 2 and c.ndim == 1:
        return c[:, None]
    return c


@dataclass
class _Block:
    name: str
    start: int
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    def idx(self) -> np.ndarray:
        return np.arange(self.start, self.start + self.size).reshape(self.shape)


@dataclass
class MILP:
    """Variables in named blocks, rows added in vectorised groups.

    A *term* is ``(idx, coef)``: ``idx`` an integer array of variable
    indices of shape ``(m,)`` or ``(m, k)``, ``coef`` broadcastable to it.
    :meth:`add` adds ``m`` rows ``lb ≤ Σ_terms Σ_k coef·x[idx] ≤ ub``.
    """

    blocks: dict[str, _Block] = field(default_factory=dict)
    _lb: list[np.ndarray] = field(default_factory=list)
    _ub: list[np.ndarray] = field(default_factory=list)
    _int: list[np.ndarray] = field(default_factory=list)
    _cost: list[np.ndarray] = field(default_factory=list)
    _rows: list[np.ndarray] = field(default_factory=list)
    _cols: list[np.ndarray] = field(default_factory=list)
    _vals: list[np.ndarray] = field(default_factory=list)
    _rlb: list[np.ndarray] = field(default_factory=list)
    _rub: list[np.ndarray] = field(default_factory=list)
    n_vars: int = 0
    n_rows: int = 0
    offset: float = 0.0
    _row_groups: dict[str, tuple[int, int]] = field(default_factory=dict)
    _extra_cost: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    # -- variables ---------------------------------------------------------
    def var(
        self,
        name: str,
        shape: int | Sequence[int],
        lb: float | np.ndarray = 0.0,
        ub: float | np.ndarray = INF,
        integer: bool = False,
        cost: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """A block of variables; returns their indices with ``shape``."""
        if name in self.blocks:
            raise ValueError(f"variable block {name!r} defined twice")
        shp = (shape,) if isinstance(shape, int | np.integer) else tuple(shape)
        b = _Block(name, self.n_vars, shp)
        self.blocks[name] = b
        size = b.size
        self._lb.append(np.broadcast_to(np.asarray(lb, dtype=float), shp).reshape(-1).copy())
        self._ub.append(np.broadcast_to(np.asarray(ub, dtype=float), shp).reshape(-1).copy())
        self._int.append(np.full(size, bool(integer)))
        self._cost.append(np.broadcast_to(np.asarray(cost, dtype=float), shp).reshape(-1).copy())
        self.n_vars += size
        return b.idx()

    def __getitem__(self, name: str) -> np.ndarray:
        return self.blocks[name].idx()

    def __contains__(self, name: str) -> bool:
        return name in self.blocks

    def add_cost(self, idx: np.ndarray, coef: float | np.ndarray) -> None:
        """Add ``coef · x[idx]`` to the objective (accumulates)."""
        idx = np.asarray(idx)
        c = np.broadcast_to(np.asarray(coef, dtype=float), idx.shape).reshape(-1)
        self._extra_cost.append((idx.reshape(-1).copy(), c.copy()))

    def fix(self, idx: np.ndarray, value: float | np.ndarray) -> None:
        """Pin variables to values (both bounds)."""
        lb, ub = self._bounds()
        idx = np.asarray(idx).reshape(-1)
        v = np.broadcast_to(np.asarray(value, dtype=float), idx.shape)
        lb[idx] = v
        ub[idx] = v
        self._set_bounds(lb, ub)

    def bound(self, idx: np.ndarray, lb: float | np.ndarray | None = None, ub=None) -> None:
        """Tighten (intersect) variable bounds."""
        lo, hi = self._bounds()
        idx = np.asarray(idx).reshape(-1)
        if lb is not None:
            lo[idx] = np.maximum(lo[idx], np.broadcast_to(np.asarray(lb, dtype=float), idx.shape))
        if ub is not None:
            hi[idx] = np.minimum(hi[idx], np.broadcast_to(np.asarray(ub, dtype=float), idx.shape))
        self._set_bounds(lo, hi)

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self._lb), np.concatenate(self._ub)

    def _set_bounds(self, lb: np.ndarray, ub: np.ndarray) -> None:
        self._lb, self._ub = [lb], [ub]

    # -- rows --------------------------------------------------------------
    def add(
        self,
        terms: Sequence[tuple[np.ndarray, float | np.ndarray]],
        lb: float | np.ndarray = -INF,
        ub: float | np.ndarray = INF,
        name: str | None = None,
    ) -> np.ndarray:
        """Add a group of rows; returns their row indices."""
        if not terms:
            raise ValueError("a row needs at least one term")
        m = None
        rows, cols, vals = [], [], []
        for idx, coef in terms:
            idx = np.asarray(idx)
            if idx.ndim == 0:
                idx = idx[None]
            if m is None:
                m = idx.shape[0]
            elif idx.shape[0] != m:
                raise ValueError(f"terms disagree on the number of rows: {idx.shape[0]} vs {m}")
            idx2 = idx.reshape(m, -1)
            c2 = np.broadcast_to(_column(np.asarray(coef, dtype=float), idx.ndim), idx.shape).reshape(m, -1)
            r = np.repeat(np.arange(m), idx2.shape[1]).reshape(m, -1)
            keep = c2 != 0
            rows.append(r[keep] + self.n_rows)
            cols.append(idx2[keep])
            vals.append(c2[keep])
        assert m is not None
        self._rows.extend(rows)
        self._cols.extend(cols)
        self._vals.extend(vals)
        self._rlb.append(np.broadcast_to(np.asarray(lb, dtype=float), (m,)).copy())
        self._rub.append(np.broadcast_to(np.asarray(ub, dtype=float), (m,)).copy())
        out = np.arange(self.n_rows, self.n_rows + m)
        if name:
            self._row_groups[name] = (self.n_rows, self.n_rows + m)
        self.n_rows += m
        return out

    # -- assembly and solve ------------------------------------------------
    def arrays(self):
        """``(c, A (csc), row_lb, row_ub, col_lb, col_ub, integer)``."""
        c = np.concatenate(self._cost) if self._cost else np.zeros(0)
        for idx, coef in self._extra_cost:
            np.add.at(c, idx, coef)
        lb, ub = self._bounds()
        integ = np.concatenate(self._int)
        if self._rows:
            r = np.concatenate(self._rows)
            k = np.concatenate(self._cols)
            v = np.concatenate(self._vals)
        else:
            r = k = np.zeros(0, int)
            v = np.zeros(0)
        A = sp.coo_matrix((v, (r, k)), shape=(self.n_rows, self.n_vars)).tocsc()
        A.sum_duplicates()
        rlb = np.concatenate(self._rlb) if self._rlb else np.zeros(0)
        rub = np.concatenate(self._rub) if self._rub else np.zeros(0)
        return c, A, rlb, rub, lb, ub, integ

    def check(self, x: np.ndarray, tol: float = 1e-6) -> list[str]:
        """Constraint, bound and integrality violations of a point ``x``,
        each named by its block or row group — the planner's own audit of a
        plan (a plan that violates a constraint must be caught)."""
        _c, A, rlb, rub, lb, ub, integ = self.arrays()
        out = []
        ax = A @ x
        scale = 1.0 + np.abs(ax)
        bad = np.flatnonzero((ax < rlb - tol * scale) | (ax > rub + tol * scale))
        for i in bad[:50]:
            grp = next((g for g, (a, b) in self._row_groups.items() if a <= i < b), "?")
            out.append(f"row {i} ({grp}): {ax[i]:.6g} not in [{rlb[i]:.6g}, {rub[i]:.6g}]")
        badv = np.flatnonzero((x < lb - tol * (1 + np.abs(lb))) | (x > ub + tol * (1 + np.abs(ub))))
        for i in badv[:50]:
            out.append(f"variable {self._var_name(i)}: {x[i]:.6g} not in [{lb[i]:.6g}, {ub[i]:.6g}]")
        badi = np.flatnonzero(integ & (np.abs(x - np.round(x)) > 1e-5))
        for i in badi[:50]:
            out.append(f"variable {self._var_name(i)}: {x[i]:.6g} is not integral")
        return out

    def _var_name(self, i: int) -> str:
        for b in self.blocks.values():
            if b.start <= i < b.start + b.size:
                return f"{b.name}{list(np.unravel_index(i - b.start, b.shape))}"
        return str(i)

    def value(self, x: np.ndarray, name: str) -> np.ndarray:
        return x[self[name]]

    def solve(
        self,
        options: SolveOptions | None = None,
        warm: Mapping[str, np.ndarray] | None = None,
        relax_and_fix: int = 0,
    ) -> Solution:
        """Solve with HiGHS. ``warm``: block name → values (NaN = not given),
        a partial solution HiGHS completes into its first incumbent.

        ``relax_and_fix`` > 1: first build an incumbent by **relax-and-fix**
        over that many windows of the time axis (blocks whose first axis is
        time, :meth:`time_axis`): window ``w``'s integers are solved with the
        earlier windows' fixed and the later ones relaxed; then the full
        problem is solved from that incumbent, within the same time limit.
        """
        opts = options or SolveOptions()
        t0 = time.perf_counter()
        arr = self.arrays()
        _c, _A, _rlb, _rub, lb, ub, integ = arr
        warm_x = self._warm_vector(warm, lb, ub, integ) if warm else None
        rf_info = None
        if relax_and_fix > 1 and integ.any() and self._n_steps:
            x_rf, rf_info = self._relax_and_fix(arr, relax_and_fix, opts, warm_x, t0)
            if x_rf is not None:
                warm_x = x_rf
        left = max(0.05, opts.time_limit - (time.perf_counter() - t0))
        sol = self._run(arr, opts, left, warm_x, opts.node_limit)
        sol.seconds = time.perf_counter() - t0
        if rf_info is not None:
            sol.reason = (sol.reason + " " if sol.reason else "") + rf_info
            sol.warm_started = sol.warm_started or rf_info.startswith("relax-and-fix ok")
        return sol

    # -- time axis (for relax-and-fix) ------------------------------------------
    _n_steps: int = 0
    _time_blocks: tuple[str, ...] = ()

    def time_axis(self, n_steps: int, blocks: Sequence[str]) -> None:
        """Declare that ``blocks``' first axis is the ``n_steps`` time steps."""
        self._n_steps = int(n_steps)
        self._time_blocks = tuple(blocks)

    def _step_of(self) -> np.ndarray:
        t = np.full(self.n_vars, -1, dtype=int)
        for name in self._time_blocks:
            b = self.blocks[name]
            if b.shape and b.shape[0] in (self._n_steps, self._n_steps + 1):
                k = np.arange(b.shape[0]).reshape((-1,) + (1,) * (len(b.shape) - 1))
                t[b.idx()] = np.broadcast_to(np.minimum(k, self._n_steps - 1), b.shape)
        return t

    def _warm_vector(self, warm, lb, ub, integ) -> np.ndarray | None:
        x = np.full(self.n_vars, np.nan)
        any_ = False
        for name, vals in warm.items():
            if name not in self.blocks:
                continue
            ix = self[name].reshape(-1)
            v = np.asarray(vals, dtype=float).reshape(-1)
            if v.shape != ix.shape:
                continue
            ok = np.isfinite(v)
            vv = np.clip(v[ok], lb[ix[ok]], ub[ix[ok]])
            x[ix[ok]] = np.where(integ[ix[ok]], np.round(vv), vv)
            any_ = True
        return x if any_ else None

    def _relax_and_fix(self, arr, windows: int, opts: SolveOptions, warm_x, t0):
        c, A, rlb, rub, lb, ub, integ = arr
        step = self._step_of()
        edges = np.linspace(0, self._n_steps, windows + 1).round().astype(int)
        lb2, ub2 = lb.copy(), ub.copy()
        x = None
        budget = 0.5 * opts.time_limit
        for w in range(windows):
            a, b = edges[w], edges[w + 1]
            if time.perf_counter() - t0 > budget:
                return None, f"relax-and-fix stopped at window {w + 1}/{windows} (time)"
            mask = integ & (((step >= a) & (step < b)) | (step < 0))
            left = max(0.05, (budget - (time.perf_counter() - t0)) / (windows - w))
            sub = self._run(
                (c, A, rlb, rub, lb2, ub2, mask), replace(opts, gap=max(opts.gap, 0.05)), left, None, None
            )
            if sub.x is None:
                return None, f"relax-and-fix failed at window {w + 1}/{windows} ({sub.status})"
            x = sub.x
            fix = integ & (step >= a) & (step < b)
            v = np.round(x[fix])
            lb2[fix] = v
            ub2[fix] = v
        if x is None:
            return None, "relax-and-fix: nothing to do"
        rest = integ & (step < 0)
        out = x.copy()
        out[rest] = np.round(out[rest])
        return out, f"relax-and-fix ok ({windows} windows, {time.perf_counter() - t0:.2f} s)"

    def _run(self, arr, opts: SolveOptions, time_limit: float, warm_x, node_limit) -> Solution:
        import highspy

        t0 = time.perf_counter()
        c, A, rlb, rub, lb, ub, integ = arr
        lp = highspy.HighsLp()
        lp.num_col_ = self.n_vars
        lp.num_row_ = self.n_rows
        lp.col_cost_ = c
        lp.col_lower_ = lb
        lp.col_upper_ = ub
        lp.row_lower_ = rlb
        lp.row_upper_ = rub
        lp.offset_ = float(self.offset)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data
        lp.a_matrix_.num_col_ = self.n_vars
        lp.a_matrix_.num_row_ = self.n_rows
        has_int = bool(integ.any())
        if has_int:
            lp.integrality_ = [
                highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous for i in integ
            ]
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(opts.verbose))
        h.setOptionValue("threads", int(opts.threads))
        h.setOptionValue("random_seed", int(opts.seed))
        h.setOptionValue("mip_rel_gap", float(opts.gap))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("presolve", "on" if opts.presolve else "off")
        if node_limit is not None:
            h.setOptionValue("mip_max_nodes", int(node_limit))
        h.passModel(lp)
        warmed = False
        if warm_x is not None and has_int:
            ok = np.isfinite(warm_x)
            # integers only: HiGHS completes the rest with an LP
            sel = ok & integ
            ix = np.flatnonzero(sel).astype(np.int32)
            if len(ix):
                warmed = h.setSolution(len(ix), ix, warm_x[ix]) == highspy.HighsStatus.kOk
        h.run()
        secs = time.perf_counter() - t0
        st = h.getModelStatus()
        info = h.getInfo()
        S = highspy.HighsModelStatus
        nodes = int(getattr(info, "mip_node_count", 0) or 0)
        primal_ok = int(getattr(info, "primal_solution_status", 0) or 0) == 2
        x = np.asarray(h.getSolution().col_value, dtype=float) if primal_ok else None
        obj = float(info.objective_function_value) if primal_ok else float("nan")
        bound = float(getattr(info, "mip_dual_bound", obj)) if has_int else obj
        gap = float(getattr(info, "mip_gap", 0.0)) if has_int else 0.0
        det = True
        reason = ""
        if st == S.kOptimal:
            status = "optimal"
        elif st in (S.kTimeLimit, S.kSolutionLimit, S.kIterationLimit, S.kInterrupt) or (
            node_limit is not None and nodes >= node_limit
        ):
            status = "limit" if x is not None else "failed"
            det = st != S.kTimeLimit
            reason = st.name
        elif st in (S.kInfeasible, S.kUnboundedOrInfeasible):
            status = "infeasible"
            x = None
            reason = st.name
        else:
            status = "limit" if x is not None else "failed"
            reason = st.name
        return Solution(status, x, obj, bound, gap, secs, nodes, det, reason, warmed)
