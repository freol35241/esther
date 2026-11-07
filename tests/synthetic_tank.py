"""A synthetic tank plant for the M3 recovery tests (spec §10.1).

One zone heated by radiators that mix from the jacket of a double-jacket
tank; the jacket is charged by a fixed-speed heat pump and an electric
heater; hot water is drawn from the inner vessel. Unlike the envelope tests,
the inputs cannot be generated open-loop — an uncontrolled tank drifts off —
so the plant is simulated **in closed loop** with a controller of the usual
kind: the compressor runs on a hot-water thermostat and on a heating demand
from a curve, the circulation pump stops while hot water is charged, the
supply is mixed down from the jacket, and the heater comes in when the inner
vessel runs cold. Draws are random, morning and evening, of known energy.

Every number here is invented; nothing is any real house.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from esther import loads_house
from esther.core.model import Signal, compose, discretise
from esther.identify import IdData

K = 273.15

PLANT = """\
schema = 1
name = "synthetic-tank"

[[component]]
type = "Ambient"
name = "outdoor"
signal = "t_out"

[[component]]
type = "Zone"
name = "zone"
signal = "t_in"
prior.C = {{ median = "6 MJ/K", gsd = 2.0 }}

[[component]]
type = "Coupling"
name = "env"
between = ["zone", "outdoor"]
prior.UA = {{ median = "200 W/K", gsd = 2.0 }}

[[component]]
type = "Radiators"
name = "radiators"
zone = "zone"
source = "tank.jacket"
supply_signal = "t_sup"
flow_signal = "pump"
prior.UA_rad = {{ median = "300 W/K", gsd = 2.5 }}

[[component]]
type = "JacketTank"
name = "tank"
volume_jacket = "60 l"
volume_dhw = "180 l"
room = "zone"
jacket_signal = "t_jacket"
dhw_signal = "t_dhw"
mixing_signal = "mix"
{tank_priors}

[[component]]
type = "HeatPump"
name = "hp"
sink = "tank.jacket"
source = "zone"
Q_rated = "2.5 kW"
P_el = "0.6 kW"
on_signal = "comp"
{hp_extra}

[[component]]
type = "ElectricHeater"
name = "heater"
node = "tank.jacket"
max = "3 kW"
on_signal = "heat"
{heater_prior}

[[component]]
type = "HotWaterDraw"
name = "draw"
node = "tank.dhw"
signal = "q_draw"
"""

TRUTH = {
    "zone.C": 4e6,
    "zone.sigma_w": 0.05,
    "zone.sigma_v": 0.03,
    "env.UA": 80.0,
    "radiators.UA_rad": 180.0,
    "tank.f_C": 1.0,
    "tank.UA_jd": 40.0,
    "tank.UA_mix": 400.0,
    "tank.UA_loss": 3.0,
    "tank.sigma_w_jacket": 0.3,
    "tank.sigma_w_dhw": 0.05,
    "tank.sigma_v_jacket": 0.1,
    "tank.sigma_v_dhw": 0.05,
    "hp.Q": 1800.0,
    "hp.dQ_dT": 15.0,
    "heater.Q": 3000.0,
}

# The anchors, as a house file would write them.
ANCHORED = {
    "tank_priors": (
        'prior.f_C = { median = 1.0, gsd = 1.15 }\nprior.UA_mix = { median = "200 W/K", gsd = 3.0 }'
    ),
    "hp_extra": "",
    "heater_prior": 'prior.Q = { value = "3 kW", fixed = true }',
}


def plant_text(**over: str) -> str:
    return PLANT.format(**{**ANCHORED, **over})


def plant(**over: str):
    return compose(loads_house(plant_text(**over)))


def simulate_plant(
    model,
    truth: dict[str, float],
    days: float = 6.0,
    step: float = 300.0,
    seed: int = 0,
    draws_per_day: float = 4.0,
    noise: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate ``model`` (built by :func:`plant`) in closed loop at ``truth``.

    Returns ``(frame, events)``: the frame holds every input, modulation and
    measured signal (and ``x:<state>`` true states); ``events`` the true draws
    (``time`` of the step they start in, ``energy`` J, ``steps``)."""
    rng = np.random.default_rng(seed)
    n = round(days * 86400 / step)
    idx = pd.date_range("2030-01-01", periods=n, freq=f"{int(step)}s", tz="UTC")
    t = np.arange(n) * step
    drift = np.cumsum(rng.normal(0.0, 0.1, n)) * np.sqrt(step / 600.0)
    drift -= np.linspace(0, drift[-1], n)
    t_out = K + 5.0 + 5.0 * np.sin(2 * np.pi * (t / 86400 - 0.375)) + drift

    # draws: morning and evening, 0.3-2.5 kWh over 1-3 steps
    q_draw = np.zeros(n)
    events = []
    for d in range(int(np.ceil(days))):
        for _ in range(rng.poisson(draws_per_day)):
            hour = rng.choice([rng.uniform(6, 9), rng.uniform(17, 22)])
            a = int((d * 86400 + hour * 3600) / step)
            m = int(rng.integers(1, 4))
            if a + m >= n or q_draw[a : a + m].any() or (a > 0 and q_draw[a - 1]):
                continue
            e = rng.uniform(0.3, 2.5) * 3.6e6
            q_draw[a : a + m] = e / (m * step)
            events.append({"time": idx[a], "energy": e, "steps": m})
    # a few heater blocks on their own, so the heater is excited apart from the compressor
    heat_block = np.zeros(n, dtype=bool)
    for d in range(int(np.ceil(days))):
        for _ in range(2):
            a = int((d * 86400 + rng.uniform(0, 24) * 3600) / step)
            heat_block[a : a + int(rng.integers(1, 4) * 1200 / step)] = True

    names = list(model.param_names)
    M = model.matrices([truth[p] for p in names])
    si = {s: i for i, s in enumerate(model.states)}
    j, dh, z = si["tank.jacket"], si["tank.dhw"], si["zone"]
    cache: dict[tuple, object] = {}

    def disc(levels: tuple):
        if levels not in cache:
            A, E = M.at(levels)
            cache[levels] = discretise(A, E, M.Qc, step)
        return cache[levels]

    x = np.zeros(model.n_states)
    x[z], x[j], x[dh] = K + 20.0, K + 45.0, K + 50.0
    X = np.zeros((n, model.n_states))
    cols = {k: np.zeros(n) for k in ("t_sup", "pump", "comp", "heat", "mix")}
    hw = ht = False
    for k in range(n):
        target = K + 22.0 + 0.6 * (20.0 - (t_out[k] - K))
        if x[dh] < K + 45.0:
            hw = True
        elif x[dh] > K + 52.0:
            hw = False
        if x[j] < target:
            ht = True
        elif x[j] > target + 10.0:
            ht = False
        comp = 1.0 if (hw or ht) else 0.0
        heat = 1.0 if (heat_block[k] or x[dh] < K + 40.0) else 0.0
        pump = 0.0 if (hw and x[dh] < K + 50.0) else 1.0
        t_sup = min(x[j] - 1.0, target) if pump else x[z] + 2.0
        mix = max(comp, heat)
        vals = {"t_out": t_out[k], "t_sup": t_sup, "pump": pump, "comp": comp, "heat": heat, "mix": mix}
        vals["q_draw"] = q_draw[k]
        for key in ("t_sup", "pump", "comp", "heat", "mix"):
            cols[key][k] = vals[key]
        dvec = np.array([vals[u.name] if isinstance(u, Signal) else 1.0 for u in model.inputs])
        lv = tuple(vals[m] for m in model.modulations)
        dd = disc(lv)
        X[k] = x
        x = dd.Ad @ x + dd.Gd @ dvec
        if noise:
            x = x + np.linalg.cholesky(dd.Qd + 1e-18 * np.eye(len(x))) @ rng.standard_normal(len(x))
    frame = pd.DataFrame({"t_out": t_out, "q_draw": q_draw, **cols}, index=idx)
    obs = M.H.argmax(axis=1)
    for o, sig in enumerate(model.outputs):
        y = X[:, obs[o]] + M.c[o]
        if noise:
            y = y + np.sqrt(M.R[o, o]) * rng.standard_normal(n)
        frame[sig] = y
    for i, s in enumerate(model.states):
        frame[f"x:{s}"] = X[:, i]
    return frame, pd.DataFrame(events, columns=["time", "energy", "steps"])


def iddata(model, frame: pd.DataFrame, step: float = 300.0, burn_in: float = 0.5) -> IdData:
    frame = frame.iloc[int(burn_in * 86400 / step) :]
    return IdData.from_frame(frame, step, model.required_signals())
