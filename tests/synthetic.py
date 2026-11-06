"""Synthetic houses and excitation for the recovery tests (spec §10.1).

Inputs are generated, not recorded: an outdoor temperature with a daily cycle
and weather drift, a radiator supply held at random levels for hours at a
time, a circulation pump that stops now and then, stove-like heat pulses of
known size, and a bell-shaped daily irradiance with cloudy days. The house is
then simulated with :func:`esther.identify.simulate` from known parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from esther import loads_house
from esther.core.model import compose
from esther.identify import IdData, simulate

K = 273.15

HEADER = """\
schema = 1
name = "synthetic"

[[component]]
type = "Ambient"
name = "outdoor"
signal = "t_out"

[[component]]
type = "Zone"
name = "zone"
signal = "t_in"
prior.C = { median = "5 MJ/K", gsd = 2.0 }
"""

MASS = """
[[component]]
type = "Mass"
name = "mass"
zone = "zone"
prior.C = { median = "40 MJ/K", gsd = 3.0 }
# Process noise on an unmeasured node is barely identified (and on real data,
# left free, it soaks up model error); pinned, as a house file would.
prior.sigma_w = { value = "0.01 K", fixed = true }
"""

ENVELOPE = """
[[component]]
type = "Coupling"
name = "env"
between = ["zone", "outdoor"]
prior.UA = { median = "150 W/K", gsd = 2.0 }
"""

RADIATORS = """
[[component]]
type = "Radiators"
zone = "zone"
supply_signal = "t_sup"
flow_signal = "pump"
"""

# The anchor: a heat input of known size (spec §6.2's scale argument).
KNOWN_HEAT = """
[[component]]
type = "HeatInput"
name = "heater"
node = "zone"
signal = "q_heat"
prior.gain = { value = 1.0, fixed = true }
"""


def inputs(days: float, step: float, seed: int = 0, start: str = "2030-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = round(days * 86400 / step)
    idx = pd.date_range(start, periods=n, freq=f"{int(step)}s", tz="UTC")
    t = np.arange(n) * step
    drift = np.cumsum(rng.normal(0.0, 0.15, n)) * np.sqrt(step / 600.0)
    drift -= np.linspace(0, drift[-1], n)  # keep it bounded
    t_out = K + 3.0 + 5.0 * np.sin(2 * np.pi * (t / 86400 - 0.375)) + drift
    hold_lo, hold_hi = int(2 * 3600 / step), int(8 * 3600 / step)
    lev: list[float] = []
    while len(lev) < n:
        lev += [rng.uniform(22.0, 45.0)] * int(rng.integers(hold_lo, hold_hi))
    t_sup = K + np.array(lev[:n])
    pump = np.ones(n)
    i = 0
    while i < n:
        if rng.random() < 0.25:
            pump[i : i + int(rng.integers(1, max(2, hold_lo)))] = 0.0
        i += int(rng.integers(hold_lo // 2 + 1, hold_hi))
    q = np.zeros(n)
    for d in range(int(np.ceil(days))):
        if rng.random() < 0.6:
            a = int((d * 86400 + rng.uniform(6, 20) * 3600) / step)
            q[a : a + int(rng.integers(max(1, hold_lo // 2), hold_lo * 2))] = rng.choice([1500.0, 3000.0])
    tod = (t % 86400) / 86400
    cloud = np.repeat(rng.uniform(0.15, 1.0, int(np.ceil(days)) + 1), int(86400 / step))[:n]
    sun = np.clip(np.sin(np.pi * (tod - 0.3) / 0.4), 0.0, None) ** 1.5 * cloud * 600.0
    sun[(tod < 0.3) | (tod > 0.7)] = 0.0
    return pd.DataFrame({"t_out": t_out, "t_sup": t_sup, "pump": pump, "q_heat": q, "sun": sun}, index=idx)


def house(*parts: str):
    return loads_house(HEADER + "".join(parts))


def dataset(
    model,
    truth: dict[str, float],
    days: float = 12.0,
    step: float = 600.0,
    seed: int = 0,
    gaps: int = 2,
    burn_in: float = 2.0,
) -> tuple[IdData, pd.DataFrame]:
    """Simulate ``model`` at ``truth``; drop a burn-in and cut ``gaps`` holes
    of a few hours so the data come as several segments."""
    rng = np.random.default_rng(seed + 1000)
    frame = inputs(days + burn_in, step, seed)
    x0 = np.full(model.n_states, K + 20.0)
    sim = simulate(model, truth, frame, step, x0=x0, rng=rng)
    sim = sim.iloc[int(burn_in * 86400 / step) :].copy()
    n = len(sim)
    for g in range(gaps):
        a = int((g + 1) * n / (gaps + 1))
        sim.iloc[a : a + int(3 * 3600 / step), sim.columns.get_loc(model.outputs[0])] = np.nan
    return IdData.from_frame(sim, step, model.required_signals()), sim


def compose_house(*parts: str):
    return compose(house(*parts))
