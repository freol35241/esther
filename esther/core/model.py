"""The composed linear thermal model (spec §4.3, §4.4).

Each component contributes **terms** — capacities, conductances, gains,
noise, measurements, decisions — as functions of its own parameter values.
:func:`compose` collects the components of a house that have dynamics into a
:class:`LinearModel`; for any parameter vector, :meth:`LinearModel.matrices`
assembles the continuous-time model

    dx/dt = A x + B u + E d + w,        y = H x + c + v

with ``x`` the node temperatures (K), ``u`` the decisions (W), ``d`` the
exogenous inputs (boundary temperatures, gain signals, time features), ``w``
process noise of intensity ``Qc`` and ``v`` measurement noise of covariance
``R``. The core knows nothing about what a term means physically; it only sums
heat flows into nodes and divides by their capacity.

A conductance may be **modulated** by a signal ``s`` in [0, 1] — a radiator
circuit whose pump runs part of the time carries ``s · UA``. Then ``A`` and
``E`` depend on the step: ``A_k = A_0 + Σ_j s_{j,k} A_j`` (and likewise
``E``). The model stays linear in the states; :func:`discretise` evaluates it
on each distinct modulation level.

Discretisation is exact zero-order hold (matrix exponential), with Van Loan's
method for the process-noise covariance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.linalg import expm

from .params import LogNormal, Param

if TYPE_CHECKING:  # pragma: no cover
    from ..house import House
    from .component import Component

__all__ = [
    "Capacity",
    "Conductance",
    "Constant",
    "Decision",
    "Discrete",
    "Gain",
    "Harmonic",
    "LinearModel",
    "Matrices",
    "Measurement",
    "ModelError",
    "Noise",
    "ParamRef",
    "Signal",
    "compose",
    "discretise",
]

SECONDS_PER_HOUR = 3600.0


class ModelError(ValueError):
    """The house cannot be composed into a model; the message names the component."""


# -- inputs -----------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """A named, measured (or forecast) signal, in SI units."""

    name: str

    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant:
    """The input that is always 1 (a constant heat rate is ``gain × 1``)."""

    def label(self) -> str:
        return "1"


@dataclass(frozen=True)
class Harmonic:
    """``cos`` or ``sin`` of ``2π · order · t / period``, ``t`` in seconds since the
    epoch (UTC). A smooth periodic profile is a sum of these."""

    fn: str  # "cos" | "sin"
    order: int
    period: float = 86400.0

    def label(self) -> str:
        return f"{self.fn}{self.order}/{self.period:g}s"

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        w = 2.0 * np.pi * self.order * t / self.period
        return np.cos(w) if self.fn == "cos" else np.sin(w)


Input = Signal | Constant | Harmonic


# -- terms ------------------------------------------------------------------


@dataclass(frozen=True)
class Capacity:
    """Node ``node`` has heat capacity ``C`` (J/K) and one temperature state."""

    node: str
    C: float


@dataclass(frozen=True)
class Conductance:
    """``Q = UA (T_b − T_a)`` into ``a`` (and its opposite into ``b``).

    ``a`` is a node. ``b`` is a node, a boundary (by name; resolved to the
    boundary's signal) or a :class:`Signal` (a measured temperature such as a
    radiator supply). ``modulation`` names a signal in [0, 1] that scales the
    conductance per step.
    """

    a: str
    b: str | Signal
    UA: float
    modulation: str | None = None


@dataclass(frozen=True)
class Gain:
    """``Q = gain × input`` into ``node`` (W)."""

    node: str
    input: Input
    gain: float


@dataclass(frozen=True)
class Decision:
    """A heat-rate decision ``u`` (W) entering ``node`` with factor ``gain``."""

    name: str
    node: str
    gain: float = 1.0


@dataclass(frozen=True)
class Noise:
    """Process noise on ``node``: a diffusion whose standard deviation grows by
    ``sigma`` kelvin per square-root hour (``Qc = sigma² / 3600`` K²/s)."""

    node: str
    sigma: float


@dataclass(frozen=True)
class Measurement:
    """Signal ``signal`` reads ``T_node + bias`` with white noise of sd ``sigma`` (K)."""

    signal: str
    node: str
    sigma: float
    bias: float = 0.0


Term = Capacity | Conductance | Gain | Decision | Noise | Measurement


# -- the model --------------------------------------------------------------


@dataclass(frozen=True)
class ParamRef:
    """A model parameter: component ``component``'s parameter ``param``."""

    component: str
    param: Param

    @property
    def name(self) -> str:
        return f"{self.component}.{self.param.name}"


@dataclass
class Matrices:
    """Continuous-time matrices for one parameter vector.

    ``A[0]``/``E[0]`` are the unmodulated parts; ``A[j]``/``E[j]`` (j ≥ 1) are
    multiplied by modulation signal ``j − 1``.
    """

    A: np.ndarray  # (1 + n_mod, n, n)
    E: np.ndarray  # (1 + n_mod, n, n_in)
    B: np.ndarray  # (n, n_dec)
    Qc: np.ndarray  # (n, n)
    H: np.ndarray  # (n_out, n)
    c: np.ndarray  # (n_out,)
    R: np.ndarray  # (n_out, n_out)

    def at(self, levels: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """``(A, E)`` with the modulation signals at ``levels``."""
        s = np.r_[1.0, np.asarray(levels, dtype=float)]
        return np.tensordot(s, self.A, 1), np.tensordot(s, self.E, 1)


@dataclass
class LinearModel:
    """A house's composed linear model, built by :func:`compose`."""

    components: dict[str, Component]
    states: tuple[str, ...]
    inputs: tuple[Input, ...]
    modulations: tuple[str, ...]
    decisions: tuple[str, ...]
    outputs: tuple[str, ...]  # measured signals, in H's row order
    free: tuple[ParamRef, ...]
    fixed: dict[str, float]  # "component.param" -> pinned value
    boundary_signal: dict[str, str]
    mode: str = "identify"
    skipped: tuple[str, ...] = ()
    _active: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)

    # -- parameters --------------------------------------------------------
    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.free)

    @property
    def n_states(self) -> int:
        return len(self.states)

    def prior_centre(self) -> np.ndarray:
        return np.array([r.param.prior.centre for r in self.free])

    def values(self, theta: Sequence[float] | Mapping[str, float]) -> dict[str, dict[str, float]]:
        """Per component, its parameter values: free from ``theta``, fixed from the house."""
        if isinstance(theta, Mapping):
            missing = [n for n in self.param_names if n not in theta]
            if missing:
                raise KeyError(f"no value for parameter(s): {', '.join(missing)}")
            flat = {n: float(theta[n]) for n in self.param_names}
        else:
            th = np.asarray(theta, dtype=float)
            if th.shape != (len(self.free),):
                raise ValueError(
                    f"theta has shape {th.shape}; the model has {len(self.free)} free parameters"
                )
            flat = dict(zip(self.param_names, th.tolist(), strict=True))
        flat.update(self.fixed)
        out: dict[str, dict[str, float]] = {name: {} for name in self.components}
        for key, v in flat.items():
            comp, _, p = key.partition(".")
            out[comp][p] = v
        return out

    # -- assembly ----------------------------------------------------------
    def terms(self, theta: Sequence[float] | Mapping[str, float]) -> list[tuple[str, Term]]:
        vals = self.values(theta)
        out = []
        for name, comp in self.components.items():
            out.extend((name, self._orient(t)) for t in comp.dynamics(vals[name], self.mode))
        return out

    def _orient(self, t: Term) -> Term:
        """A conductance written boundary-first is turned round: ``a`` is the node."""
        if isinstance(t, Conductance) and isinstance(t.a, str) and t.a in self.boundary_signal:
            if isinstance(t.b, Signal) or t.b in self.boundary_signal:
                raise ModelError(f"a conductance between two boundaries ({t.a!r}, {t.b!r}) has no state")
            return Conductance(t.b, t.a, t.UA, t.modulation)
        return t

    def matrices(self, theta: Sequence[float] | Mapping[str, float]) -> Matrices:
        terms = self.terms(theta)
        n, ni, nm = len(self.states), len(self.inputs), len(self.modulations)
        si = {s: i for i, s in enumerate(self.states)}
        ii = {u: i for i, u in enumerate(self.inputs)}
        mi = {m: j + 1 for j, m in enumerate(self.modulations)}
        di = {d: i for i, d in enumerate(self.decisions)}
        oi = {o: i for i, o in enumerate(self.outputs)}
        C = np.zeros(n)
        for _, t in terms:
            if isinstance(t, Capacity):
                C[si[t.node]] = t.C
        A = np.zeros((1 + nm, n, n))
        E = np.zeros((1 + nm, n, ni))
        B = np.zeros((n, len(self.decisions)))
        Qc = np.zeros((n, n))
        H = np.zeros((len(self.outputs), n))
        c = np.zeros(len(self.outputs))
        r = np.zeros(len(self.outputs))
        for _, t in terms:
            if isinstance(t, Conductance):
                a = si[t.a]
                j = mi[t.modulation] if t.modulation else 0
                A[j, a, a] -= t.UA / C[a]
                if isinstance(t.b, Signal) or t.b in self.boundary_signal:
                    sig = t.b if isinstance(t.b, Signal) else Signal(self.boundary_signal[t.b])
                    E[j, a, ii[sig]] += t.UA / C[a]
                else:
                    b = si[t.b]
                    A[j, a, b] += t.UA / C[a]
                    A[j, b, b] -= t.UA / C[b]
                    A[j, b, a] += t.UA / C[b]
            elif isinstance(t, Gain):
                k = si[t.node]
                E[0, k, ii[t.input]] += t.gain / C[k]
            elif isinstance(t, Decision):
                k = si[t.node]
                B[k, di[t.name]] += t.gain / C[k]
            elif isinstance(t, Noise):
                Qc[si[t.node], si[t.node]] += t.sigma**2 / SECONDS_PER_HOUR
            elif isinstance(t, Measurement):
                o = oi[t.signal]
                H[o, si[t.node]] = 1.0
                c[o] = t.bias
                r[o] = t.sigma**2
        return Matrices(A=A, E=E, B=B, Qc=Qc, H=H, c=c, R=np.diag(r))

    # -- data --------------------------------------------------------------
    def required_signals(self) -> list[str]:
        """Every signal the model reads: inputs, modulations and outputs."""
        out: list[str] = []
        for s in [*(u.name for u in self.inputs if isinstance(u, Signal)), *self.modulations, *self.outputs]:
            if s not in out:
                out.append(s)
        return out

    def input_array(self, frame: pd.DataFrame) -> np.ndarray:
        """``d`` per row of ``frame`` (indexed by UTC step start), shape (len, n_in)."""
        t = frame.index.as_unit("ns").asi8 / 1e9 if len(frame) else np.zeros(0)
        cols = []
        for u in self.inputs:
            if isinstance(u, Signal):
                cols.append(frame[u.name].to_numpy(dtype=float))
            elif isinstance(u, Constant):
                cols.append(np.ones(len(frame)))
            else:
                cols.append(u.evaluate(t))
        return np.column_stack(cols) if cols else np.zeros((len(frame), 0))

    def describe(self) -> str:
        lines = [
            f"states:      {', '.join(self.states)}",
            f"inputs:      {', '.join(u.label() for u in self.inputs) or '-'}",
            f"modulations: {', '.join(self.modulations) or '-'}",
            f"outputs:     {', '.join(self.outputs) or '-'}",
            f"decisions:   {', '.join(self.decisions) or '-'}",
            f"free:        {', '.join(self.param_names) or '-'}",
        ]
        if self.fixed:
            lines.append("fixed:       " + ", ".join(f"{k}={v:g}" for k, v in self.fixed.items()))
        if self.skipped:
            lines.append(f"skipped:     {', '.join(self.skipped)} (no dynamics in this esther)")
        return "\n".join(lines)


def _has_dynamics(comp: Component) -> bool:
    from .component import Component

    return type(comp).dynamics is not Component.dynamics


def compose(
    house: House | Mapping[str, Component],
    include: Iterable[str] | None = None,
    mode: str = "identify",
) -> LinearModel:
    """Compose a house (or a mapping of components) into a :class:`LinearModel`.

    ``include`` restricts the model to the named components (the fitting
    split, §6.4: the envelope is fitted without the tank). Without it, every
    component that has dynamics is included and the others are listed in
    ``skipped``. ``mode`` is ``"identify"`` (measured inputs, e.g. a radiator
    supply temperature) or ``"plan"`` (decisions).
    """
    if mode not in ("identify", "plan"):
        raise ValueError(f"mode must be 'identify' or 'plan', got {mode!r}")
    comps_all = dict(house.components) if hasattr(house, "components") else dict(house)
    if include is not None:
        names = list(include)
        unknown = [n for n in names if n not in comps_all]
        if unknown:
            raise ModelError(f"no component named {', '.join(map(repr, unknown))}")
        comps = {n: comps_all[n] for n in names}
        bad = [n for n, c in comps.items() if not _has_dynamics(c)]
        if bad:
            raise ModelError(
                "component(s) without dynamics in this esther: "
                + ", ".join(f"{n!r} ({comps[n].type_name})" for n in bad)
            )
        skipped: tuple[str, ...] = ()
    else:
        comps = {n: c for n, c in comps_all.items() if _has_dynamics(c)}
        skipped = tuple(n for n in comps_all if n not in comps)

    boundary_signal: dict[str, str] = {}
    for c in comps.values():
        boundary_signal.update(c.boundary_signals())

    free: list[ParamRef] = []
    fixed: dict[str, float] = {}
    for name, c in comps.items():
        for pname in c.active_params(mode):
            p = c.param[pname]
            if p.fixed is not None:
                fixed[f"{name}.{pname}"] = p.fixed
            else:
                free.append(ParamRef(name, p))

    model = LinearModel(
        components=comps,
        states=(),
        inputs=(),
        modulations=(),
        decisions=(),
        outputs=(),
        free=tuple(free),
        fixed=fixed,
        boundary_signal=boundary_signal,
        mode=mode,
        skipped=skipped,
    )
    # One pass over the terms at the prior centre fixes the structure (which
    # terms exist does not depend on parameter values).
    terms = model.terms(model.prior_centre())
    states: list[str] = []
    owner: dict[str, str] = {}
    for cname, t in terms:
        if isinstance(t, Capacity):
            if t.node in owner:
                raise ModelError(f"{cname!r}: node {t.node!r} already has a capacity from {owner[t.node]!r}")
            owner[t.node] = cname
            states.append(t.node)
    inputs: list[Input] = []
    mods: list[str] = []
    decisions: list[str] = []
    outputs: list[str] = []

    def add(lst: list, item: Any) -> None:
        if item not in lst:
            lst.append(item)

    def node(cname: str, n: str, what: str) -> None:
        if n not in owner:
            hint = " (a boundary is not a node)" if n in boundary_signal else ""
            raise ModelError(f"{cname!r}: {what} {n!r} is not a node in this model{hint}")

    adj: dict[str, set[str]] = {s: set() for s in states}
    for cname, t in terms:
        if isinstance(t, Conductance):
            node(cname, t.a, "node")
            if isinstance(t.b, Signal):
                add(inputs, t.b)
                adj[t.a].add("<boundary>")
            elif t.b in boundary_signal:
                add(inputs, Signal(boundary_signal[t.b]))
                adj[t.a].add("<boundary>")
            else:
                node(cname, t.b, "node")
                adj[t.a].add(t.b)
                adj[t.b].add(t.a)
            if t.modulation:
                add(mods, t.modulation)
        elif isinstance(t, Gain):
            node(cname, t.node, "node")
            add(inputs, t.input)
        elif isinstance(t, Decision):
            node(cname, t.node, "node")
            if t.name in decisions:
                raise ModelError(f"{cname!r}: decision {t.name!r} defined twice")
            decisions.append(t.name)
        elif isinstance(t, Noise):
            node(cname, t.node, "node")
        elif isinstance(t, Measurement):
            node(cname, t.node, "node")
            if t.signal in outputs:
                raise ModelError(f"{cname!r}: signal {t.signal!r} is already a measurement")
            outputs.append(t.signal)
    # Every state needs a conductance path to a boundary (§4.4).
    seen = {s for s in states if "<boundary>" in adj[s]}
    frontier = list(seen)
    while frontier:
        for nb in adj[frontier.pop()]:
            if nb != "<boundary>" and nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    for s in states:
        if s not in seen:
            raise ModelError(
                f"{owner[s]!r}: node {s!r} has no conductance path to a boundary in this model, "
                "so its temperature would be unbounded"
            )
    if mode == "identify" and not outputs:
        raise ModelError("no measurements: identification needs at least one measured node")
    model.states = tuple(states)
    model.inputs = tuple(inputs)
    model.modulations = tuple(mods)
    model.decisions = tuple(decisions)
    model.outputs = tuple(outputs)
    return model


# -- discretisation ---------------------------------------------------------


@dataclass
class Discrete:
    """Discrete-time matrices, batched over any leading axes (parameter
    vectors, modulation levels) of the continuous ones."""

    Ad: np.ndarray  # (..., n, n)
    Gd: np.ndarray  # (..., n, n_in)
    Qd: np.ndarray  # (..., n, n)


def discretise(A: np.ndarray, E: np.ndarray, Qc: np.ndarray, dt: float) -> Discrete:
    """Exact zero-order-hold discretisation over ``dt`` seconds.

    ``A`` (..., n, n), ``E`` (..., n, m), ``Qc`` broadcastable to (..., n, n).
    ``x_{k+1} = Ad x_k + Gd d_k + w_k``, ``Cov w_k = Qd`` (Van Loan).
    """
    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    n, m = A.shape[-1], E.shape[-1]
    lead = A.shape[:-2]
    Qc = np.broadcast_to(np.asarray(Qc, dtype=float), (*lead, n, n))
    M = np.zeros((*lead, n + m, n + m))
    M[..., :n, :n] = A
    M[..., :n, n:] = E
    F = expm(M * dt)
    Ad, Gd = F[..., :n, :n], F[..., :n, n:]
    V = np.zeros((*lead, 2 * n, 2 * n))
    V[..., :n, :n] = -A
    V[..., :n, n:] = Qc
    V[..., n:, n:] = np.swapaxes(A, -1, -2)
    G = expm(V * dt)
    F22t = np.swapaxes(G[..., n:, n:], -1, -2)  # = Ad
    Qd = F22t @ G[..., :n, n:]
    Qd = 0.5 * (Qd + np.swapaxes(Qd, -1, -2))
    return Discrete(Ad=np.ascontiguousarray(Ad), Gd=np.ascontiguousarray(Gd), Qd=Qd)


def is_log(p: Param) -> bool:
    return isinstance(p.prior, LogNormal)
