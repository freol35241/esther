"""M0 solver spike: a synthetic reference MILP, solved with cvxpy and linopy.

Spec §11 M0 exit: "spike solves 144 steps in < 2 s". Spec §3.2 / §12.1 leave
cvxpy vs linopy open, to be decided by this spike. Both formulations below
build the *same* problem; their optimal objectives must agree, which is the
check that neither formulation is quietly solving something easier.

The plant is synthetic and generic — numbers of the right order for a
detached house with a heat pump, not any particular house:

    states (4)   zone air T_z, building mass T_m, tank jacket T_j, DHW vessel T_d
    heat pump    fixed speed into the jacket: on/off binary + on-fraction,
                 minimum run 1 h, minimum off 30 min, start cost,
                 condensing limit on the jacket while it runs
    heater       electric, into the jacket, integer level 0..3 × 3 kW
    stove        heats the zone: 3 one-hot levels, start indicator, minimum
                 burn 2 h, minimum off (cool-down) 1 h, ignition ramp
                 (reduced output for 2 steps after a start), forbidden at night
    radiators    Q_rad ≤ UA_rad (T_j − T_z): the shunt can only mix down
    loads        hot-water draws (forecast energy per step) on the DHW node
    objective    electricity at a spot price + fuel + starts + comfort slack
                 (K·h, both sides of a band) + DHW service slack − terminal
                 value of stored heat

Horizon 36 h at 15 min = 144 steps. Exact zero-order-hold discretisation.

Run it:  python benchmarks/reference_milp.py [--backend B] [--variant V] [--two-phase]
         [--time-limit S] [--gap G] [--repeat N]

Findings (2026-11-05; Ryzen 5 PRO 3500U laptop, WSL2 devcontainer, HiGHS
1.15.1 single-threaded, 1 % relative gap; full table in the M0 PR):

- The modelling layer is not where the time goes, except that linopy's build
  costs ~3 s here against cvxpy's ~0.02 s. Same solver, same optimum on the
  "easy" variant (the cross-check). Hence cvxpy.
- The solve is dominated by formulation. "easy" (heater as on-fraction, jacket
  capped at the condensing limit, no stove) is optimal in ~0.5 s. Each of the
  stove's levels/min-burn, the integer heater levels, and the jacket's narrow
  window (condensing limit) on its own keeps HiGHS from proving 1 % in 10 s:
  the root LP sits 5-8 % below the best known solution, mostly because the LP
  burns the stove fractionally.
- "full" cold: incumbent 115 at 2 s, 66 at 10 s (best known ~32.5, bound ~30).
  Two-phase (stove forbidden first, then warm-started): 38.3 at 2 s, 32.7 at
  10 s. The exit criterion "144 steps in < 2 s" is met only by "easy"; see the
  PR for what that means for M4.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm

C2K = 273.15
J_PER_KWH = 3.6e6


@dataclass
class ReferenceProblem:
    """Data for the synthetic reference MILP. All internal values are SI."""

    n_steps: int = 144
    dt: float = 900.0  # s

    # Envelope
    C_z: float = 5e6  # J/K
    C_m: float = 50e6
    UA_zo: float = 110.0  # W/K, zone to outdoor
    UA_zm: float = 600.0  # zone to mass
    UA_mo: float = 25.0  # mass to outdoor
    # Tank (double jacket)
    C_j: float = 60 * 4186.0  # 60 l of water
    C_d: float = 150 * 4186.0  # 150 l
    UA_jd: float = 300.0
    UA_jz: float = 1.5  # jacket losses to the room
    UA_dz: float = 1.5
    # Radiators
    UA_rad: float = 250.0
    Q_rad_max: float = 8000.0
    # Heat pump
    Q_hp_rated: float = 2000.0  # W heat
    hp_min_fraction: float = 0.3
    hp_min_on: int = 4  # steps
    hp_min_off: int = 2
    hp_T_max: float = 58 + C2K  # condensing limit on the jacket
    eta_carnot: float = 0.40
    T_sink_cop: float = 50 + C2K
    c_start_hp: float = 0.05  # currency per start
    # Electric heater
    heater_step: float = 3000.0  # W
    heater_levels: int = 3
    # Stove
    stove_heat: tuple[float, ...] = (1000.0, 3500.0, 7000.0)  # W
    stove_fuel: tuple[float, ...] = (0.25, 0.80, 1.55)  # kg/h
    stove_min_on: int = 8
    stove_min_off: int = 4
    stove_ign_steps: int = 2
    stove_ign_loss: float = 500.0  # W less than the level's output during ignition
    stove_forbidden_hours: tuple[int, int] = (22, 7)
    c_start_stove: float = 1.0
    fuel_price: float = 3.0  # currency/kg
    # Comfort and service
    w_comfort: float = 5.0  # currency per K·h
    w_dhw: float = 50.0  # currency per K·step
    T_dhw_min: float = 45 + C2K
    T_z_floor: float = 16 + C2K
    T_j_max: float = 80 + C2K
    # Terminal value
    terminal_factor: float = 0.9
    # Solve-shaping options (see the module docstring and the PR for timings)
    heater_continuous: bool = False  # heater as a continuous on-fraction instead of integer levels
    block_after: int | None = None  # from this step on, integer decisions are held for block_len steps
    block_len: int = 4
    stove_block: int = 1  # stove decisions held over blocks of this many steps (from step 0)

    x0: np.ndarray = field(default_factory=lambda: np.array([20.5, 20.0, 45.0, 48.0]) + C2K)
    start_hour: float = 12.0

    def __post_init__(self) -> None:
        n = self.n_steps
        hours = (self.start_hour + np.arange(n) * self.dt / 3600.0) % 24
        self.hours = hours
        # Cold weather: the heat pump alone cannot carry the house.
        self.T_out = -8 + 4 * np.sin(2 * np.pi * (hours - 9) / 24) + C2K
        # Spot price, currency/kWh: morning and evening peaks, cheap nights.
        self.price = (
            0.35
            + 0.9 * np.exp(-(((hours - 8) / 1.5) ** 2))
            + 1.6 * np.exp(-(((hours - 18) / 2.0) ** 2))
            - 0.15 * ((hours < 5) | (hours >= 23))
        )
        self.Q_int = np.full(n, 300.0)
        # DHW draws, W averaged over the step: a morning and an evening draw.
        draw_kwh = np.zeros(n)
        draw_kwh[(hours >= 6.5) & (hours < 7.5)] = 0.6
        draw_kwh[(hours >= 20) & (hours < 21)] = 0.9
        self.Q_draw = draw_kwh * J_PER_KWH / self.dt
        # Comfort band
        night = (hours >= 23) | (hours < 6)
        self.T_set = np.where(night, 19.0, 21.0) + C2K
        self.band = 0.5
        a, b = self.stove_forbidden_hours
        self.stove_allowed = ~((hours >= a) | (hours < b))
        self.cop = self.eta_carnot * self.T_sink_cop / (self.T_sink_cop - self.T_out)
        self.Ad, self.Bd, self.Ed = self._discretise()

    # x = [T_z, T_m, T_j, T_d]; u = [Q_hp, Q_el, Q_rad, Q_stove]; d = [T_out, Q_int, Q_draw]
    def _continuous(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        C = np.array([self.C_z, self.C_m, self.C_j, self.C_d])
        G = np.zeros((4, 4))  # conductances between nodes

        def couple(i: int, j: int, ua: float) -> None:
            G[i, j] += ua
            G[j, i] += ua

        couple(0, 1, self.UA_zm)
        couple(2, 3, self.UA_jd)
        couple(2, 0, self.UA_jz)
        couple(3, 0, self.UA_dz)
        to_out = np.array([self.UA_zo, self.UA_mo, 0.0, 0.0])
        A = (G - np.diag(G.sum(axis=1) + to_out)) / C[:, None]
        B = (
            np.array(
                [
                    [0, 0, 1, 1],  # zone: radiators, stove
                    [0, 0, 0, 0],
                    [1, 1, -1, 0],  # jacket: hp, heater, minus radiators
                    [0, 0, 0, 0],
                ],
                dtype=float,
            )
            / C[:, None]
        )
        E = (
            np.array(
                [
                    [self.UA_zo, 1, 0],
                    [self.UA_mo, 0, 0],
                    [0, 0, 0],
                    [0, 0, -1],
                ],
                dtype=float,
            )
            / C[:, None]
        )
        return A, B, E

    def _discretise(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        A, B, E = self._continuous()
        nx, nu, nd = A.shape[0], B.shape[1], E.shape[1]
        M = np.zeros((nx + nu + nd, nx + nu + nd))
        M[:nx, :nx] = A
        M[:nx, nx : nx + nu] = B
        M[:nx, nx + nu :] = E
        Phi = expm(M * self.dt)
        return Phi[:nx, :nx], Phi[:nx, nx : nx + nu], Phi[:nx, nx + nu :]

    def solver_units(self) -> SimpleNamespace:
        """The problem in the units the MILP is built in: °C, kW, kWh.

        The model is SI (spec N4); the solver is not told so. With W and J the
        constraint matrix spans ~9 orders of magnitude and HiGHS slows down
        badly; in kW and °C it spans ~4. The dynamics are offset-invariant
        (every node's conductances sum to its diagonal), so °C needs no
        affine correction.
        """
        kw = 1e3
        Bd = self.Bd * kw
        Ed = self.Ed.copy()
        Ed[:, 1:] *= kw  # Q_int, Q_draw columns are powers
        C_kwh = np.array([self.C_z, self.C_m, self.C_j, self.C_d]) / J_PER_KWH  # kWh/K
        per_kwh = self.terminal_factor * self.price.mean() / self.cop.mean()
        tw = C_kwh * per_kwh
        floors = np.array([self.T_z_floor, self.T_z_floor, 40 + C2K, 40 + C2K]) - C2K
        return SimpleNamespace(
            Ad=self.Ad,
            Bd=Bd,
            Ed=Ed,
            x0=self.x0 - C2K,
            T_out=self.T_out - C2K,
            T_set=self.T_set - C2K,
            Q_int=self.Q_int / kw,
            Q_draw=self.Q_draw / kw,
            Q_hp_rated=self.Q_hp_rated / kw,
            heater_step=self.heater_step / kw,
            stove_heat=np.array(self.stove_heat) / kw,
            stove_ign_loss=self.stove_ign_loss / kw,
            UA_rad=self.UA_rad / kw,
            Q_rad_max=self.Q_rad_max / kw,
            hp_T_max=self.hp_T_max - C2K,
            T_j_max=self.T_j_max - C2K,
            T_z_floor=self.T_z_floor - C2K,
            T_dhw_min=self.T_dhw_min - C2K,
            terminal_weight=tw,
            terminal_offset=float(tw @ floors),
        )

    def blocked_steps(self, after: int | None, length: int) -> np.ndarray:
        """Steps whose decisions must equal the previous step's (move blocking):
        from step ``after`` on, decisions change only every ``length`` steps."""
        k = np.arange(self.n_steps)
        if after is None or length <= 1:
            return k[:0]
        return k[(k > after) & ((k - after) % length != 0)]


@dataclass
class SpikeResult:
    backend: str
    status: str
    objective: float
    build_s: float  # modelling-layer construction
    solve_s: float  # wall time of the solve call (incl. canonicalisation / export)
    solver_s: float | None  # time HiGHS itself reports
    total_s: float
    trajectories: dict[str, np.ndarray]


HIGHS_OPTIONS = {
    "time_limit": 10.0,
    "mip_rel_gap": 0.01,  # 1 %: on this problem ~0.3 currency over 36 h
    "threads": 1,
    "random_seed": 0,
    "output_flag": False,
}
VERBOSE = False


def solve_cvxpy(p: ReferenceProblem, two_phase: bool = False) -> SpikeResult:
    """Build and solve with cvxpy + HiGHS.

    ``two_phase``: first solve with the stove forbidden (an easy problem: no
    stove binaries in play), then re-solve the same, parametrised problem with
    the real stove permissions, warm-started from the first solution. Both
    phases share one time limit. This is the MPC re-solve pattern — a
    parameter changes, the previous solution seeds the next — used here to
    give HiGHS a good incumbent early.
    """
    import cvxpy as cp

    t0 = time.perf_counter()
    s = p.solver_units()
    n, dt_h = p.n_steps, p.dt / 3600.0
    allowed = cp.Parameter(n, nonneg=True)
    X = cp.Variable((n + 1, 4))
    delta = cp.Variable(n, nonneg=True)
    hp_on = cp.Variable(n, boolean=True)
    hp_start = cp.Variable(n, boolean=True)
    heater = cp.Variable(n, integer=not p.heater_continuous)
    Q_rad = cp.Variable(n, nonneg=True)
    L = len(s.stove_heat)
    y = cp.Variable((n, L), boolean=True)
    s_start = cp.Variable(n, boolean=True)
    sig_lo = cp.Variable(n, nonneg=True)
    sig_hi = cp.Variable(n, nonneg=True)
    sig_dhw = cp.Variable(n, nonneg=True)
    # A variable fixed at 1 carries the objective's constant into HiGHS, so
    # that its relative MIP gap is measured on the real cost.
    one = cp.Variable()

    heat = s.stove_heat
    fuel = np.array(p.stove_fuel)
    Q_hp = s.Q_hp_rated * delta
    Q_el = s.heater_step * heater
    stove_on = cp.sum(y, axis=1)

    def lag(v, k: int, init: float = 0.0):
        """v shifted k steps later; the k steps before the horizon are ``init``."""
        if k == 0:
            return v
        return cp.hstack([np.full(k, init), v[: n - k]])

    def window(v, w: int):
        return sum(lag(v, j) for j in range(w))

    Q_stove = y @ heat - s.stove_ign_loss * window(s_start, p.stove_ign_steps)
    U = cp.vstack([Q_hp, Q_el, Q_rad, Q_stove]).T
    D = np.column_stack([s.T_out, s.Q_int, s.Q_draw])
    Tz, Tj, Td = X[1:, 0], X[1:, 2], X[1:, 3]

    cons = [
        one == 1,
        X[0] == s.x0,
        X[1:] == X[:-1] @ s.Ad.T + U @ s.Bd.T + D @ s.Ed.T,
        # heat pump
        delta <= hp_on,
        delta >= p.hp_min_fraction * hp_on,
        hp_start >= hp_on - lag(hp_on, 1),
        window(hp_start, p.hp_min_on) <= hp_on,
        window(hp_start, p.hp_min_off) <= 1 - lag(hp_on, p.hp_min_off),
        X[:-1, 2] <= s.hp_T_max + (s.T_j_max - s.hp_T_max) * (1 - hp_on),
        # heater
        heater >= 0,
        heater <= p.heater_levels,
        # radiators: mix down only
        Q_rad <= s.UA_rad * (X[:-1, 2] - X[:-1, 0]),
        Q_rad <= s.Q_rad_max,
        # stove
        stove_on <= 1,
        stove_on <= allowed,
        s_start >= stove_on - lag(stove_on, 1),
        window(s_start, p.stove_min_on) <= stove_on,
        window(s_start, p.stove_min_off) <= 1 - lag(stove_on, p.stove_min_off),
        # comfort band (soft), hard floor, tank limits, DHW service (soft)
        Tz >= s.T_set - p.band - sig_lo,
        Tz <= s.T_set + p.band + sig_hi,
        Tz >= s.T_z_floor,
        Tj <= s.T_j_max,
        Td >= s.T_dhw_min - sig_dhw,
    ]
    P = Q_hp / p.cop + Q_el  # kW electrical
    cost = (
        cp.sum(cp.multiply(p.price, P)) * dt_h
        + p.fuel_price * cp.sum(y @ fuel) * dt_h
        + p.c_start_hp * cp.sum(hp_start)
        + p.c_start_stove * cp.sum(s_start)
        + p.w_comfort * cp.sum(sig_lo + sig_hi) * dt_h
        + p.w_dhw * cp.sum(sig_dhw)
        - (s.terminal_weight @ X[n] - s.terminal_offset * one)
    )
    blk = p.blocked_steps(p.block_after, p.block_len)
    if blk.size:
        for v in (hp_on, heater):
            cons.append(v[blk] == v[blk - 1])
        cons.append(hp_start[blk] == 0)
    sblk = np.union1d(blk, p.blocked_steps(0, p.stove_block))
    if sblk.size:
        cons.append(y[sblk, :] == y[sblk - 1, :])
        cons.append(s_start[sblk] == 0)
    prob = cp.Problem(cp.Minimize(cost), cons)
    t1 = time.perf_counter()
    opts = dict(HIGHS_OPTIONS)
    if two_phase:
        allowed.value = np.zeros(n)
        prob.solve(solver=cp.HIGHS, verbose=VERBOSE, **opts)
        opts["time_limit"] = max(0.05, HIGHS_OPTIONS["time_limit"] - (time.perf_counter() - t1))
    allowed.value = p.stove_allowed.astype(float)
    prob.solve(solver=cp.HIGHS, verbose=VERBOSE, warm_start=two_phase, **opts)
    t2 = time.perf_counter()
    stats = prob.solver_stats
    return SpikeResult(
        backend=f"cvxpy {cp.__version__}" + (" 2-phase" if two_phase else ""),
        status=prob.status,
        objective=float(prob.value) if prob.value is not None else float("nan"),
        build_s=t1 - t0,
        solve_s=t2 - t1,
        solver_s=stats.solve_time if stats else None,
        total_s=t2 - t0,
        trajectories={
            "T_z": X.value[:, 0],
            "T_j": X.value[:, 2],
            "T_d": X.value[:, 3],
            "hp_on": hp_on.value,
            "heater": heater.value,
            "stove_level": y.value @ np.arange(1, L + 1),
        }
        if X.value is not None
        else {},
    )


def solve_linopy(p: ReferenceProblem) -> SpikeResult:
    import linopy
    import pandas as pd
    import xarray as xr

    warnings.filterwarnings("ignore", module="linopy")
    t0 = time.perf_counter()
    s = p.solver_units()
    n, dt_h = p.n_steps, p.dt / 3600.0
    t = pd.RangeIndex(n, name="t")
    m = linopy.Model()

    def da(a) -> xr.DataArray:
        return xr.DataArray(np.asarray(a, dtype=float), coords=[t])

    names = ["T_z", "T_m", "T_j", "T_d"]
    # state after step k
    X = {nm: m.add_variables(coords=[t], name=nm) for nm in names}
    delta = m.add_variables(lower=0, upper=1, coords=[t], name="delta")
    hp_on = m.add_variables(binary=True, coords=[t], name="hp_on")
    hp_start = m.add_variables(binary=True, coords=[t], name="hp_start")
    heater = m.add_variables(
        lower=0, upper=p.heater_levels, integer=not p.heater_continuous, coords=[t], name="heater"
    )
    Q_rad = m.add_variables(lower=0, upper=s.Q_rad_max, coords=[t], name="Q_rad")
    L = len(s.stove_heat)
    y = [m.add_variables(binary=True, coords=[t], name=f"stove_{level}") for level in range(L)]
    s_start = m.add_variables(binary=True, coords=[t], name="stove_start")
    sig_lo = m.add_variables(lower=0, coords=[t], name="sig_lo")
    sig_hi = m.add_variables(lower=0, coords=[t], name="sig_hi")
    sig_dhw = m.add_variables(lower=0, coords=[t], name="sig_dhw")
    one = m.add_variables(lower=1, upper=1, name="one")  # carries the objective constant

    def lag(v, k: int):
        return v if k == 0 else v.shift(t=k).fillna(0)

    def window(v, w: int):
        return sum(lag(v, j) for j in range(1, w)) + v

    first = da(np.eye(1, n, 0)[0])

    def prev_state(i: int):
        """State at the start of step k: x_{k}, the known x0 when k = 0."""
        return lag(X[names[i]], 1) + s.x0[i] * first

    Q_hp = s.Q_hp_rated * delta
    Q_el = s.heater_step * heater
    stove_on = sum(y[1:], y[0] * 1)
    Q_stove = sum(s.stove_heat[lv] * y[lv] for lv in range(L)) - s.stove_ign_loss * window(
        s_start, p.stove_ign_steps
    )
    U = [Q_hp, Q_el, Q_rad, Q_stove]
    D = [da(s.T_out), da(s.Q_int), da(s.Q_draw)]
    for i, nm in enumerate(names):
        rhs = sum(s.Ad[i, j] * prev_state(j) for j in range(4) if s.Ad[i, j] != 0)
        rhs = rhs + sum(s.Bd[i, j] * U[j] for j in range(4) if s.Bd[i, j] != 0)
        rhs = rhs + sum(s.Ed[i, j] * D[j] for j in range(3) if s.Ed[i, j] != 0)
        m.add_constraints(X[nm] - rhs == 0, name=f"dyn_{nm}")

    Tz_prev, Tj_prev = prev_state(0), prev_state(2)
    m.add_constraints(delta - hp_on <= 0, name="hp_frac_hi")
    m.add_constraints(delta - p.hp_min_fraction * hp_on >= 0, name="hp_frac_lo")
    m.add_constraints(hp_start - hp_on + lag(hp_on, 1) >= 0, name="hp_start")
    m.add_constraints(window(hp_start, p.hp_min_on) - hp_on <= 0, name="hp_min_on")
    m.add_constraints(window(hp_start, p.hp_min_off) + lag(hp_on, p.hp_min_off) <= 1, name="hp_min_off")
    m.add_constraints(Tj_prev + (s.T_j_max - s.hp_T_max) * hp_on <= s.T_j_max, name="hp_condensing")
    m.add_constraints(Q_rad - s.UA_rad * (Tj_prev - Tz_prev) <= 0, name="mixing")
    m.add_constraints(stove_on <= da(p.stove_allowed), name="stove_allowed")
    m.add_constraints(s_start - stove_on + lag(stove_on, 1) >= 0, name="stove_start")
    m.add_constraints(window(s_start, p.stove_min_on) - stove_on <= 0, name="stove_min_on")
    m.add_constraints(
        window(s_start, p.stove_min_off) + lag(stove_on, p.stove_min_off) <= 1,
        name="stove_min_off",
    )
    m.add_constraints(X["T_z"] + sig_lo >= da(s.T_set - p.band), name="comfort_lo")
    m.add_constraints(X["T_z"] - sig_hi <= da(s.T_set + p.band), name="comfort_hi")
    m.add_constraints(X["T_z"] >= s.T_z_floor, name="floor")
    m.add_constraints(X["T_j"] <= s.T_j_max, name="tank_max")
    m.add_constraints(X["T_d"] + sig_dhw >= s.T_dhw_min, name="dhw")

    el = (da(p.price / p.cop) * Q_hp + da(p.price) * Q_el) * dt_h
    fuel = sum(p.stove_fuel[lv] * y[lv] for lv in range(L)) * (p.fuel_price * dt_h)
    tw = s.terminal_weight
    last = da(np.eye(1, n, n - 1)[0])
    terminal = sum(tw[i] * last * X[nm] for i, nm in enumerate(names))
    objective = (
        el
        + fuel
        + p.c_start_hp * hp_start
        + p.c_start_stove * s_start
        + (p.w_comfort * dt_h) * (sig_lo + sig_hi)
        + p.w_dhw * sig_dhw
        - terminal
    ).sum() + s.terminal_offset * one
    m.add_objective(objective)

    def held(steps: np.ndarray) -> xr.DataArray:
        mask = np.zeros(n, dtype=bool)
        mask[steps] = True
        return xr.DataArray(mask, coords=[t])

    blk = p.blocked_steps(p.block_after, p.block_len)
    if blk.size:
        for v in (hp_on, heater):
            m.add_constraints(v - lag(v, 1) == 0, mask=held(blk))
        m.add_constraints(1 * hp_start == 0, mask=held(blk))
    sblk = np.union1d(blk, p.blocked_steps(0, p.stove_block))
    if sblk.size:
        for v in y:
            m.add_constraints(v - lag(v, 1) == 0, mask=held(sblk))
        m.add_constraints(1 * s_start == 0, mask=held(sblk))
    t1 = time.perf_counter()
    status, condition = m.solve(solver_name="highs", **HIGHS_OPTIONS)
    t2 = time.perf_counter()
    ok = status == "ok"
    obj = float(m.objective.value) if ok else float("nan")
    sol = {nm: X[nm].solution.values for nm in names} if ok else {}
    return SpikeResult(
        backend=f"linopy {linopy.__version__}",
        status=condition,
        objective=obj,
        build_s=t1 - t0,
        solve_s=t2 - t1,
        solver_s=None,
        total_s=t2 - t0,
        trajectories={
            "T_z": np.r_[s.x0[0], sol["T_z"]],
            "T_j": np.r_[s.x0[2], sol["T_j"]],
            "T_d": np.r_[s.x0[3], sol["T_d"]],
            "hp_on": hp_on.solution.values,
            "heater": heater.solution.values,
            "stove_level": sum((lv + 1) * y[lv].solution.values for lv in range(L)),
        }
        if ok
        else {},
    )


BACKENDS = {"cvxpy": solve_cvxpy, "linopy": solve_linopy}


def machine() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    import highspy

    try:
        hv = highspy.Highs().version()
    except Exception:
        hv = "?"
    return (
        f"{cpu}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}, HiGHS {hv} (threads=1)"
    )


# Named variants of the reference problem. "full" is the problem as described
# in the module docstring; each other variant removes one source of difficulty,
# so that the timings show where the solve time goes.
VARIANTS: dict[str, dict] = {
    "full": {},
    "heater-fraction": {"heater_continuous": True},
    "no-stove": {"stove_forbidden_hours": (0, 24)},
    "tank-capped": {"heater_continuous": True, "T_j_max": 58 + C2K},
    "easy": {"heater_continuous": True, "T_j_max": 58 + C2K, "stove_forbidden_hours": (0, 24)},
}


def run(backend: str, variant: str, two_phase: bool = False) -> SpikeResult:
    p = ReferenceProblem(**VARIANTS[variant])
    if backend == "cvxpy":
        return solve_cvxpy(p, two_phase=two_phase)
    return BACKENDS[backend](p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--backend", choices=sorted(BACKENDS), action="append")
    ap.add_argument("--variant", choices=list(VARIANTS), action="append")
    ap.add_argument("--two-phase", action="store_true", help="cvxpy only; see solve_cvxpy")
    ap.add_argument("--time-limit", type=float, default=HIGHS_OPTIONS["time_limit"])
    ap.add_argument("--gap", type=float, default=HIGHS_OPTIONS["mip_rel_gap"], help="relative MIP gap")
    args = ap.parse_args()
    HIGHS_OPTIONS.update(time_limit=args.time_limit, mip_rel_gap=args.gap)
    warnings.filterwarnings("ignore", message="Solution may be inaccurate")
    print(machine())
    print(f"144 steps x 15 min; HiGHS time_limit {args.time_limit:g} s, mip_rel_gap {args.gap:g}\n")
    print(
        f"{'backend':<24}{'variant':<17}{'status':<12}{'objective':>10}{'build':>8}{'solve':>8}{'total':>8}"
        "   median s; max total"
    )
    for name in args.backend or sorted(BACKENDS):
        for variant in args.variant or list(VARIANTS):
            runs = [run(name, variant, args.two_phase) for _ in range(args.repeat)]
            r = runs[0]
            med = {
                k: statistics.median(getattr(x, k) for x in runs) for k in ("build_s", "solve_s", "total_s")
            }
            print(
                f"{r.backend:<24}{variant:<17}{r.status:<12}{r.objective:>10.3f}"
                f"{med['build_s']:>8.3f}{med['solve_s']:>8.3f}{med['total_s']:>8.3f}"
                f"   {max(x.total_s for x in runs):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
