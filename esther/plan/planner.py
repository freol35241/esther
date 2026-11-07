"""The planner (spec §8): problem assembly, the solve, and the Plan.

The composed model in **plan mode** gives the dynamics: its decisions are the
heat rates of the sources and radiators. Everything else is contributed by
the components through :meth:`~esther.core.component.Component.plan` —
on-fractions and on/off indicators, stove levels with start, burn and
cool-down rules, mixing limits, comfort and hot-water slack, the tariff —
so that the core composes contributions and knows nothing about heat pumps
(§4.1).

Internally the problem is built in °C (offset from 273.15 K; the dynamics
are affine in the offset, which is carried exactly), kW and kWh, so that
the matrix spans a few orders of magnitude rather than nine (the M0 spike's
finding).

Speed (N1): integer decisions are free per step for the first
``free_hours`` and held over clock-aligned ``block`` intervals after that
(move blocking); every solve is warm-started from the previous plan, shifted
to the new grid; and a solve that reaches its limit returns its incumbent,
flagged. The COP (or a source's heat output that depends on its sink) is
evaluated on a predicted sink temperature and the problem re-solved, warm,
when the plan moved it by more than ``cop_tolerance`` (§8.5, at most
``max_passes`` passes).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..core.model import Capacity, LinearModel, compose, discretise
from ..forecast.base import Event, Forecast, check_provenance, plan_grid
from .blocks import Linear
from .milp import MILP, Solution, SolveOptions

if TYPE_CHECKING:  # pragma: no cover
    from ..house import House

__all__ = ["Plan", "PlanContext", "PlanError", "Planner", "PlannerOptions"]

T0 = 273.15
KWH = 3.6e6


class PlanError(RuntimeError):
    """No plan (§8.8): infeasible, a limit without an incumbent, stale or
    missing inputs, or a solution that fails the planner's own audit."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class PlannerOptions:
    """``free_hours``/``block``: move blocking of integer decisions.
    ``max_age``: per forecast signal, the oldest issue (s) the planner
    accepts (§8.8 staleness). ``validity``: how long the plan's commands
    stay valid (s). ``terminal_factor`` scales the terminal value (§8.6)."""

    solve: SolveOptions = field(default_factory=SolveOptions)
    free_hours: float = 6.0
    block: float = 3600.0
    cop_tolerance: float = 0.05
    max_passes: int = 2
    terminal_factor: float = 0.9
    comfort_recovery: float = 1.0  # K/h: how fast a widened band closes (§8.3)
    validity: float = 1800.0
    max_age: Mapping[str, float] = field(default_factory=dict)
    warm_start: bool = True
    relax_and_fix: int = 4  # windows of the cold-start heuristic; 0 or 1 for none
    audit: bool = True


@dataclass
class PlanContext:
    """What a component sees while it contributes to the problem."""

    planner: Planner
    milp: MILP
    now: pd.Timestamp
    index: pd.DatetimeIndex
    dt: np.ndarray
    forecast: Forecast
    x0: np.ndarray
    devices: Mapping[str, Mapping[str, Any]]
    lin: Mapping[str, np.ndarray]
    pins: Mapping[str, np.ndarray]
    X: np.ndarray = field(init=False)
    U: np.ndarray = field(init=False)
    electric: dict[str, Linear] = field(default_factory=dict)
    fuel: dict[str, tuple[Linear, str]] = field(default_factory=dict)
    costs: dict[str, Linear] = field(default_factory=dict)
    heat_cost: list[np.ndarray] = field(default_factory=list)
    floors: dict[str, float] = field(default_factory=dict)
    results: dict[str, Callable[[np.ndarray], np.ndarray]] = field(default_factory=dict)
    coefficients: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray], np.ndarray]] = field(
        default_factory=dict
    )
    requests: set[str] = field(default_factory=set)
    shared: dict[str, Any] = field(default_factory=dict)

    # -- shape --------------------------------------------------------------
    @property
    def model(self) -> LinearModel:
        return self.planner.model

    @property
    def house(self) -> House:
        return self.planner.house

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def dt_h(self) -> np.ndarray:
        return self.dt / 3600.0

    @property
    def times(self) -> np.ndarray:
        """Seconds from ``now`` to each step's end."""
        return np.cumsum(self.dt)

    def values(self, component: str) -> dict[str, float]:
        return self.planner.values.get(component, {})

    # -- variables ----------------------------------------------------------
    def state(self, node: str) -> np.ndarray:
        """Indices of ``node``'s temperature (°C) at steps 0..n."""
        return self.X[:, self.model.states.index(node)]

    def has_state(self, node: str) -> bool:
        return node in self.model.states

    def x0_of(self, node: str) -> float:
        return float(self.x0[self.model.states.index(node)])

    def decision(self, name: str) -> np.ndarray:
        """Indices of a heat-rate decision (kW) per step."""
        return self.U[:, self.model.decisions.index(name)]

    def has_decision(self, name: str) -> bool:
        return name in self.model.decisions

    def signal(self, name: str, default: float | None = None) -> np.ndarray:
        if name in self.forecast:
            v = self.forecast[name]
            if np.isnan(v).any():
                raise PlanError("missing inputs", f"forecast of {name!r} has gaps")
            return v
        if default is not None:
            return np.full(self.n, float(default))
        raise PlanError("missing inputs", f"no forecast of {name!r}")

    def estimated(self, name: str) -> np.ndarray:
        if name in self.forecast.estimated.columns:
            return self.forecast.estimated[name].to_numpy(dtype=bool)
        return np.zeros(self.n, bool)

    def boundary_signal(self, boundary: str) -> str:
        return self.model.boundary_signal.get(boundary, boundary)

    def event_cover(self, component: str, kind: str | None = None) -> np.ndarray:
        """Share of each step covered by the component's events (§7.6)."""
        a = self.index.as_unit("ns").asi8 / 1e9
        b = a + self.dt
        cover = np.zeros(self.n)
        for e in self.forecast.events:
            if e.component != component or (kind is not None and e.kind != kind):
                continue
            ov = np.clip(np.minimum(b, e.end.value / 1e9) - np.maximum(a, e.start.value / 1e9), 0, None)
            cover = np.maximum(cover, ov / self.dt)
        return cover

    def lin_point(self, node: str) -> np.ndarray:
        """The predicted temperature (K) of ``node`` at each step's start, for
        coefficients that depend on it (COP, a heat output; §8.5)."""
        return self.lin[node][: self.n]

    # -- move blocking -------------------------------------------------------
    def hold_steps(self, block: float | None = None, free_hours: float | None = None) -> np.ndarray:
        """Steps whose integer decisions repeat the previous step's: after
        ``free_hours``, all but the first step of each clock-aligned block."""
        opts = self.planner.options
        free = (opts.free_hours if free_hours is None else free_hours) * 3600.0
        blk = int(opts.block if block is None else block)
        start = self.times - self.dt  # seconds from now to each step's start
        absolute = self.index.as_unit("ns").asi8 // 10**9
        on_boundary = (absolute % blk) == 0
        k = np.arange(self.n)
        return k[(start >= free) & ~on_boundary & (k > 0)]

    def hold(
        self, idx: np.ndarray, name: str = "hold", block: float | None = None, free_hours: float | None = None
    ) -> None:
        """Integer decisions ``idx`` (first axis = steps) held over blocks
        (move blocking; the planner's options unless given)."""
        h = self.hold_steps(block, free_hours)
        if len(h) == 0:
            return
        idx = np.asarray(idx)
        a = idx[h].reshape(len(h), -1)
        b = idx[h - 1].reshape(len(h), -1)
        for j in range(a.shape[1]):
            self.milp.add([(a[:, j], 1.0), (b[:, j], -1.0)], 0.0, 0.0, name=name)

    # -- costs and bookkeeping -----------------------------------------------
    def add_cost(self, label: str, expr: Linear) -> None:
        """A named cost term (currency), reported in the plan's breakdown."""
        self.milp.add_cost(expr.idx, expr.coef)
        self.milp.offset += float(expr.const.sum())
        self.costs[label] = self.costs[label] + expr if label in self.costs else expr

    def add_electric(self, label: str, expr: Linear) -> None:
        """Electrical power (kW) per step drawn by ``label``; priced by the tariff."""
        self.electric[label] = expr

    def add_fuel(self, label: str, expr: Linear, fuel: str) -> None:
        """Fuel (kg/h) per step burnt by ``label``."""
        self.fuel[label] = (expr, fuel)

    def add_heat_cost(self, cost_per_kwh: np.ndarray) -> None:
        """A source's marginal cost of heat (currency/kWh) per step, for the
        terminal value (§8.6)."""
        self.heat_cost.append(np.asarray(cost_per_kwh, dtype=float))

    def add_floor(self, node: str, floor_k: float) -> None:
        self.floors[node] = min(floor_k, self.floors.get(node, np.inf))

    def add_result(self, name: str, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.results[name] = fn

    def add_coefficient(self, name: str, node: str, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """A coefficient evaluated on ``node``'s predicted temperature
        (§8.5); the planner re-solves when the plan moves it."""
        c = np.asarray(fn(self.lin_point(node)), dtype=float)
        self.coefficients[name] = (node, fn, c)
        return c

    def pinned(self, name: str) -> np.ndarray | None:
        """A manual override for decision ``name`` (§8.4): values per step,
        NaN where free."""
        v = self.pins.get(name)
        return None if v is None else np.asarray(v, dtype=float)[: self.n]


@dataclass
class Plan:
    """The planner's output (§9.3)."""

    id: str
    created: pd.Timestamp
    valid_until: pd.Timestamp
    mode: str  # "active" or "advisory"
    index: pd.DatetimeIndex
    dt: np.ndarray
    decisions: pd.DataFrame
    trajectories: pd.DataFrame  # K, at step starts plus the horizon's end
    cost: dict[str, float]
    solver: dict[str, Any]
    provenance: list
    state: dict[str, float]
    events: list[Event]
    model_version: str = ""
    validation: dict[str, Any] = field(default_factory=dict)
    warm: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)

    @property
    def flagged(self) -> bool:
        """A best-so-far solution: a limit was reached before the gap."""
        return self.solver.get("status") != "optimal"

    @property
    def advisory(self) -> bool:
        return self.mode != "active"

    def first(self, name: str) -> float:
        return float(self.decisions[name].iloc[0])


class Planner:
    """Plans a house (§8). ``values``: the parameter values, flat
    ``"component.param"`` (a fit's, plus any planning parameters; anything
    missing takes its prior centre). ``accepted``: whether the model passed
    the acceptance gate — if not, every plan is **advisory** (§8.8)."""

    def __init__(
        self,
        house: House,
        values: Mapping[str, float],
        options: PlannerOptions | None = None,
        accepted: bool = True,
        model_version: str = "",
        validation: Mapping[str, Any] | None = None,
    ) -> None:
        self.house = house
        self.options = options or PlannerOptions()
        self.accepted = accepted
        self.model_version = model_version
        self.validation = dict(validation or {})
        self.model = compose(house, mode="plan")
        flat = dict(values)
        theta = [flat.get(r.name, r.param.prior.centre) for r in self.model.free]
        self.theta = np.array(theta, dtype=float)
        per: dict[str, dict[str, float]] = {n: {} for n in house.components}
        for name, comp in house.components.items():
            for p in comp.param.values():
                per[name][p.name] = p.fixed if p.fixed is not None else p.prior.centre
        for k, v in flat.items():
            c, _, p = k.partition(".")
            if c in per:
                per[c][p] = float(v)
        self.values = per
        self.M = self.model.matrices(self.theta)
        self.capacity = {t.node: t.C for _, t in self.model.terms(self.theta) if isinstance(t, Capacity)}
        self._disc: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        order = sorted(house.components.items(), key=lambda kv: getattr(kv[1], "plan_order", 50))
        self.order = [name for name, _ in order]

    # -- the dynamics ---------------------------------------------------------
    def _discrete(
        self, dt: float, levels: tuple[float, ...] = ()
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (round(float(dt), 6), levels)
        if key not in self._disc:
            A, E = self.M.at(list(levels) if levels else [0.0] * len(self.model.modulations))
            nd = self.M.B.shape[1]
            G = np.c_[self.M.B, E]
            d = discretise(A[None], G[None], self.M.Qc[None], float(dt))
            self._disc[key] = (d.Ad[0], d.Gd[0][:, :nd], d.Gd[0][:, nd:])
        return self._disc[key]

    def required_signals(self) -> list[str]:
        return [u.name for u in self.model.inputs if hasattr(u, "name")]

    # -- one build -----------------------------------------------------------
    def build(
        self,
        now: pd.Timestamp,
        x0: np.ndarray,
        forecast: Forecast,
        devices: Mapping[str, Mapping[str, Any]],
        lin: Mapping[str, np.ndarray],
        pins: Mapping[str, np.ndarray],
    ) -> PlanContext:
        index, dt = forecast.index, forecast.dt
        n, nx, nd = len(index), self.model.n_states, len(self.model.decisions)
        milp = MILP()
        ctx = PlanContext(self, milp, now, index, dt, forecast, x0, devices, lin, pins)
        ctx.X = milp.var("x", (n + 1, nx), lb=-np.inf, ub=np.inf)
        ctx.U = milp.var("u", (n, nd), lb=0.0, ub=np.inf)
        # x_{k+1} = Ad x_k + Bd u_k + Gd d_k, in °C and kW
        frame = pd.DataFrame(index=index)
        for u in self.model.inputs:
            if hasattr(u, "name"):
                frame[u.name] = ctx.signal(u.name)
        D = self.model.input_array(frame)
        milp.add([(ctx.X[0], 1.0)], x0 - T0, x0 - T0, name="initial state")
        # A modulation in plan mode (a tank stirred while it charges, say) is
        # taken at its forecast level, or 0 without one: the model must stay
        # linear in the decisions.
        mods = np.zeros((n, len(self.model.modulations)))
        for j, m in enumerate(self.model.modulations):
            if m in forecast:
                mods[:, j] = np.clip(np.round(forecast[m] * 20) / 20, 0, 1)
        keys = [(round(float(dt[k]), 6), tuple(mods[k])) for k in range(n)]
        for key in sorted(set(keys)):
            ks = np.array([k for k in range(n) if keys[k] == key])
            Ad, Bd, Gd = self._discrete(key[0], key[1])
            rhs = D[ks] @ Gd.T + ((Ad - np.eye(nx)) @ np.full(nx, T0))[None, :]
            for i in range(nx):
                terms = [(ctx.X[ks + 1, i], 1.0), (ctx.X[ks][:, :], -np.broadcast_to(Ad[i], (len(ks), nx)))]
                if nd:
                    terms.append((ctx.U[ks][:, :], -1000.0 * np.broadcast_to(Bd[i], (len(ks), nd))))
                milp.add(terms, rhs[:, i], rhs[:, i], name=f"dynamics {self.model.states[i]}")
        comps = self.house.components
        for name in self.order:
            comps[name].plan_prepare(ctx)
        for name in self.order:
            comps[name].plan(ctx)
        self._terminal(ctx)
        milp.time_axis(n, [b for b in milp.blocks if b != "x"])
        return ctx

    def _terminal(self, ctx: PlanContext) -> None:
        """§8.6: heat stored above each node's floor at the horizon's end is
        worth what it would cost to replace, at the mean of the estimated part
        of the price forecast (all of it if none is estimated)."""
        if not ctx.floors or not ctx.heat_cost:
            return
        cost = np.min(np.vstack(ctx.heat_cost), axis=0)  # cheapest source per step
        est = np.zeros(ctx.n, bool)
        for c in ctx.forecast.estimated.columns:
            est |= ctx.forecast.estimated[c].to_numpy(dtype=bool) & ("price" in c)
        per_kwh = float(np.mean(cost[est])) if est.any() else float(np.mean(cost))
        per_kwh *= self.options.terminal_factor
        idx, coef, const = [], [], 0.0
        for node, floor in ctx.floors.items():
            if node not in self.capacity:
                continue
            c_kwh = self.capacity[node] / KWH  # kWh/K
            idx.append(ctx.state(node)[-1])
            coef.append(-c_kwh * per_kwh)
            const += c_kwh * per_kwh * (floor - T0)
        if idx:
            ctx.add_cost("terminal value", Linear(np.array([idx]), np.array([coef]), np.array([const])))
            ctx.shared["terminal_per_kwh"] = per_kwh

    # -- planning ------------------------------------------------------------
    def check_inputs(self, now: pd.Timestamp, forecast: Forecast, allow_leaky: bool = False) -> None:
        """§8.8: refuse stale inputs, and a forecast that read past now."""
        if forecast.now != now:
            raise PlanError("stale inputs", f"forecast made for {forecast.now}, planning at {now}")
        check_provenance(forecast, allow_leaky=allow_leaky)
        for p in forecast.provenance:
            age_max = self.options.max_age.get(p.signal)
            if age_max is None:
                continue
            ref = p.issued or p.data_until
            if ref is None or (now - ref).total_seconds() > age_max:
                raise PlanError("stale inputs", f"{p.signal}: newest issue {ref}, older than {age_max:g} s")

    def plan(
        self,
        now,
        x0: Mapping[str, float] | np.ndarray,
        forecast: Forecast,
        devices: Mapping[str, Mapping[str, Any]] | None = None,
        previous: Plan | None = None,
        pins: Mapping[str, np.ndarray] | None = None,
        allow_leaky: bool = False,
    ) -> Plan:
        """Plan from state ``x0`` (K, per state or in the model's order) on
        ``forecast`` (made at ``now`` on :func:`plan_grid`)."""
        t_start = time.perf_counter()
        now = pd.Timestamp(now)
        self.check_inputs(now, forecast, allow_leaky)
        if isinstance(x0, Mapping):
            missing = [s for s in self.model.states if s not in x0]
            if missing:
                raise PlanError("missing inputs", f"no state estimate for {', '.join(missing)}")
            x = np.array([float(x0[s]) for s in self.model.states])
        else:
            x = np.asarray(x0, dtype=float)
        if not np.isfinite(x).all():
            raise PlanError("missing inputs", "the state estimate has NaN")
        devices = dict(devices or {})
        pins = dict(pins or {})
        n = len(forecast.index)
        lin = self._initial_lin(forecast, x, previous)
        warm = self._shift_warm(previous, forecast.index) if (previous and self.options.warm_start) else None
        passes = []
        sol: Solution | None = None
        ctx: PlanContext | None = None
        budget = self.options.solve.time_limit  # for the whole plan, every pass included (N1)
        for p in range(self.options.max_passes):
            left = budget - (time.perf_counter() - t_start)
            if p > 0 and left < 0.1 * budget:
                passes[-1]["unresolved"] = True  # no time for the re-linearised pass: keep the first
                break
            ctx = self.build(now, x, forecast, devices, lin, pins)
            rf = self.options.relax_and_fix if warm is None else 0
            opts = replace(self.options.solve, time_limit=max(left, 0.05))
            sol = ctx.milp.solve(opts, warm, relax_and_fix=rf)
            passes.append(
                {
                    "status": sol.status,
                    "seconds": sol.seconds,
                    "gap": sol.gap,
                    "nodes": sol.nodes,
                    "objective": sol.objective,
                    "bound": sol.bound,
                    "warm": sol.warm_started,
                    "note": sol.reason,
                    "deterministic": sol.deterministic,
                    "vars": ctx.milp.n_vars,
                    "rows": ctx.milp.n_rows,
                    "integers": int(np.concatenate(ctx.milp._int).sum()),
                }
            )
            if not sol.ok:
                raise PlanError(
                    "no plan", f"{sol.status} ({sol.reason}) after {sol.seconds:.2f} s in pass {p + 1}"
                )
            if self.options.audit:
                bad = ctx.milp.check(sol.x, tol=1e-5)
                if bad:
                    raise PlanError("plan fails its audit", "; ".join(bad[:5]))
            traj = sol.x[ctx.X] + T0
            new_lin = {s: traj[:, i] for i, s in enumerate(self.model.states)}
            moved = 0.0
            for _, (node, fn, used) in ctx.coefficients.items():
                c = np.asarray(fn(new_lin[node][:n]), dtype=float)
                ref = np.maximum(np.abs(used), 0.1 * float(np.max(np.abs(used))) + 1e-9)
                rel = np.abs(c - used) / ref
                moved = max(moved, float(rel.max()) if rel.size else 0.0)
            passes[-1]["coefficient_moved"] = moved
            if moved <= self.options.cop_tolerance or p + 1 == self.options.max_passes:
                break
            lin = new_lin
            warm = {name: sol.x[ctx.milp[name]] for name in ctx.milp.blocks}
        assert sol is not None and ctx is not None and sol.x is not None
        return self._result(now, x, forecast, ctx, sol, passes, time.perf_counter() - t_start)

    # -- warm start and linearisation ----------------------------------------
    def _initial_lin(self, forecast: Forecast, x: np.ndarray, previous: Plan | None) -> dict[str, np.ndarray]:
        n = len(forecast.index)
        out = {}
        for i, s in enumerate(self.model.states):
            if previous is not None and s in previous.trajectories.columns:
                tr = previous.trajectories[s]
                t_prev = tr.index.as_unit("ns").asi8 / 1e9
                t_new = np.r_[forecast.index.as_unit("ns").asi8 / 1e9, 0.0]
                t_new[-1] = t_new[-2] + forecast.dt[-1]
                v = np.interp(t_new, t_prev, tr.to_numpy(dtype=float))
                v[0] = x[i]
                out[s] = v
            else:
                out[s] = np.full(n + 1, x[i])
        return out

    @staticmethod
    def _shift_warm(previous: Plan, index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        out = {}
        t_new = index.as_unit("ns").asi8
        for name, (t_prev, vals) in previous.warm.items():
            j = np.searchsorted(t_prev, t_new, side="right") - 1
            j = np.clip(j, 0, len(t_prev) - 1)
            out[name] = np.asarray(vals)[j]
        return out

    # -- the result -----------------------------------------------------------
    def _result(self, now, x, forecast, ctx, sol: Solution, passes, seconds) -> Plan:
        X = sol.x[ctx.X] + T0
        U = sol.x[ctx.U] * 1000.0
        end = forecast.index[-1] + pd.Timedelta(seconds=float(forecast.dt[-1]))
        t_idx = forecast.index.append(pd.DatetimeIndex([end]))
        traj = pd.DataFrame(X, index=t_idx, columns=list(self.model.states))
        dec = pd.DataFrame(U, index=forecast.index, columns=list(self.model.decisions))
        for name, fn in ctx.results.items():
            dec[name] = fn(sol.x)
        price = {}
        for label, expr in ctx.electric.items():
            dec[f"{label}.P_el"] = expr.value(sol.x) * 1000.0
        for label, (expr, _) in ctx.fuel.items():
            dec[f"{label}.fuel"] = expr.value(sol.x)
        cost = {label: float(expr.value(sol.x).sum()) for label, expr in ctx.costs.items()}
        cost["total"] = float(sum(cost.values()))
        price.update(cost)
        status = "optimal" if all(p["status"] == "optimal" for p in passes) else "limit"
        solver = {
            "status": status,
            "passes": passes,
            "seconds": seconds,
            "solve_seconds": float(sum(p["seconds"] for p in passes)),
            "gap": passes[-1]["gap"],
            "deterministic": all(p["deterministic"] for p in passes),
            "warm_started": bool(passes[0]["warm"]),
            "terminal_per_kwh": ctx.shared.get("terminal_per_kwh"),
        }
        t_ns = forecast.index.as_unit("ns").asi8
        warm = {}
        integ = np.concatenate(ctx.milp._int)
        for name in ctx.milp.blocks:
            if name in ("x", "u"):
                continue
            ix = ctx.milp[name]
            on_grid = bool(ix.shape) and ix.shape[0] == len(forecast.index)
            if on_grid and bool(integ[ix.reshape(-1)].any()):
                warm[name] = (t_ns, sol.x[ix])
        digest = hashlib.sha256(
            np.ascontiguousarray(np.round(sol.x, 6)).tobytes() + str(now).encode()
        ).hexdigest()[:12]
        return Plan(
            id=f"{now.strftime('%Y%m%dT%H%M%SZ')}-{digest}",
            created=now,
            valid_until=now + pd.Timedelta(seconds=self.options.validity),
            mode="active" if self.accepted else "advisory",
            index=forecast.index,
            dt=forecast.dt,
            decisions=dec,
            trajectories=traj,
            cost=price,
            solver=solver,
            provenance=list(forecast.provenance),
            state=dict(zip(self.model.states, x.tolist(), strict=True)),
            events=list(forecast.events),
            model_version=self.model_version,
            validation=self.validation,
            warm=warm,
        )

    def grid(self, now) -> tuple[pd.DatetimeIndex, np.ndarray]:
        return plan_grid(now, self.house.horizon, self.house.step)
