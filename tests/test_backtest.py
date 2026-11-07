"""The replay harness (spec §10.2): the twin, the loop, scoring, and the
leak check with its negative control."""

import numpy as np
import pandas as pd
import pytest
from synthetic_plan import K, house, planner, x0

from esther.backtest import LinearTwin, PlanningController, degree_hours, leak_check, run, score
from esther.core.model import compose
from esther.forecast import (
    Issues,
    LeakError,
    Provenance,
    WeatherForecaster,
    plan_grid,
)
from esther.forecast.base import single
from esther.identify import simulate

NOW = pd.Timestamp("2025-01-15T20:00Z")


def twin_parts():
    h = house()
    m = compose(h)  # identification mode: the device's measured inputs
    theta = m.prior_centre()
    idx = pd.date_range(NOW - pd.Timedelta("1h"), periods=24 * 12 + 24, freq="5min")
    rng = np.random.default_rng(0)
    ex = pd.DataFrame(
        {
            "outdoor_temperature": K + 2 + rng.normal(0, 0.3, len(idx)),
            "supply_temperature": K + 35.0,
            "compressor": (rng.uniform(size=len(idx)) > 0.5).astype(float),
            "heater_share": 0.0,
            "stove_heat": 0.0,
            "draw_heat": 0.0,
        },
        index=idx,
    )
    return h, m, theta, ex


def test_twin_reproduces_simulate():
    # positive control: with the device passing the recorded inputs through,
    # the twin is esther's own deterministic simulation
    _h, m, theta, ex = twin_parts()
    x0_ = np.array([K + 20, K + 50, K + 50])
    tw = LinearTwin(m, theta, ex, lambda t, x, row, cmds, dt: {}, 300.0, x0_)
    rec = tw.advance(ex.index[0], ex.index[-1] + pd.Timedelta("5min"))
    sim = simulate(m, theta, ex, 300.0, x0=x0_, noise=False)
    for _i, s in enumerate(m.states):
        assert np.allclose(rec[f"x:{s}"].to_numpy(), sim[f"x:{s}"].to_numpy())


def test_degree_hours_and_score():
    b, a = degree_hours(np.array([19.0, 20.0, 22.0]) + K, K + 19.5, K + 21.0, 1800)
    assert b == pytest.approx(0.25) and a == pytest.approx(0.5)
    idx = pd.date_range(NOW, periods=4, freq="15min")
    rec = pd.DataFrame(
        {"P": [1000.0, 0, 1000, 0], "fuel": [0, 2.0, 0, 0], "z": K + 20.0, "on": [1.0, 0, 1, 0]}, index=idx
    )
    price = pd.Series([0.1, 0.1, 0.5, 0.5], index=idx)
    s = score(rec, 900, price, {"hp": "P"}, fuel={"pellets": ("fuel", 0.4)}, on={"hp": "on"})
    assert s["hp kWh"] == pytest.approx(0.5) and s["hp cost"] == pytest.approx(0.25 * 0.1 + 0.25 * 0.5)
    assert s["pellets kg"] == pytest.approx(0.5) and s["cost"] == pytest.approx(0.15 + 0.2)
    assert s["hp starts"] == 1 and s["hp hours"] == pytest.approx(0.5)


class Truth:
    """Forecasters on a synthetic history, honest or reading ahead."""

    def __init__(self, cutoff=None, cheat=False):
        idx = pd.date_range(NOW - pd.Timedelta(days=3), NOW + pd.Timedelta(days=3), freq="1h")
        t = K + 2 + 4 * np.sin(np.arange(len(idx)) / 5)
        s = pd.Series(t, index=idx)
        if cutoff is not None:
            s = s.where(s.index <= cutoff, s + 1000.0)  # poison what is not yet known
        self.series = s
        rows = [
            {
                "issued": i,
                "valid": i + pd.Timedelta(hours=h),
                "value": float(s.iloc[min(k + h, len(s) - 1)] if cheat else s.iloc[k]),
            }
            for k, i in enumerate(idx)
            for h in range(48)
        ]
        self.issues = Issues(pd.DataFrame(rows))
        self.cheat = cheat

    def forecasters(self):
        out = [
            WeatherForecaster(self.issues, "outdoor_temperature"),
            Const("spot_price", 0.1 / 3.6e6),
            Const("draw_heat", 0.0),
        ]
        if self.cheat:
            out[0] = Ahead(self.series)
        return out


class Const:
    def __init__(self, signal, v):
        self.signal, self.v = signal, v
        self.signals = (signal,)

    def forecast(self, now, horizon, step):
        idx, dt = plan_grid(now, horizon, step)
        return single(
            pd.Timestamp(now),
            idx,
            dt,
            self.signal,
            np.full(len(idx), self.v),
            False,
            Provenance(self.signal, "const", "c", None),
        )


class Ahead:
    """Reads the outdoor temperature that is coming — and claims not to."""

    def __init__(self, series):
        self.series = series
        self.signals = ("outdoor_temperature",)

    def forecast(self, now, horizon, step):
        idx, dt = plan_grid(now, horizon, step)
        v = np.interp(idx.as_unit("ns").asi8, self.series.index.as_unit("ns").asi8, self.series.to_numpy())
        return single(
            pd.Timestamp(now),
            idx,
            dt,
            "outdoor_temperature",
            v,
            False,
            Provenance("outdoor_temperature", "ahead", "x", pd.Timestamp(now)),
        )


def test_leak_check_passes_honest_and_catches_a_cheat():
    h = house(heater_extra="", stove_extra="modulating = true")
    pl = planner(h)
    times = [NOW]
    state = lambda t: x0(h)  # noqa: E731
    ok = leak_check(lambda c: Truth(c).forecasters(), pl, times, state)
    assert (ok.plan_diff <= 1e-3).all()
    with pytest.raises(LeakError, match="outdoor_temperature"):
        leak_check(lambda c: Truth(c, cheat=True).forecasters(), pl, times, state)


def test_closed_loop_run_and_fallback():
    h = house()
    pl = planner(h)
    m = compose(h)
    theta = m.prior_centre()
    idx = pd.date_range(NOW, periods=12 * 3, freq="5min")
    ex = pd.DataFrame({"outdoor_temperature": K + 2.0, "stove_heat": 0.0, "draw_heat": 0.0}, index=idx)

    def device(t, x, row, cmds, dt):
        q = 0.0
        for c in cmds:
            if c.device == "curve" and c.start <= t < c.end:
                q = 1.0
        return {"supply_temperature": x["tank.jacket"], "compressor": 1.0, "heater_share": 0.0, "q": q}

    tw = LinearTwin(m, theta, ex, device, 300.0, np.array([x0(h)[s] for s in m.states]))
    ctl = PlanningController(pl, Truth().forecasters())
    res = run(tw, ctl, NOW, NOW + pd.Timedelta("3h"), 900.0)
    assert len(res.records) == 36 and len(res.decisions) == 12
    assert (res.decisions.status != "no plan").all()
    assert res.decisions.warm.iloc[1:].all()  # each plan warm-started from the one before

    class Broken:
        signals = ("spot_price",)

        def forecast(self, now, horizon, step):
            raise LookupError("no price issue readable")

    ctl2 = PlanningController(pl, [*Truth().forecasters()[:1], Broken(), Const("draw_heat", 0.0)])
    tw2 = LinearTwin(m, theta, ex, device, 300.0, np.array([x0(h)[s] for s in m.states]))
    res2 = run(tw2, ctl2, NOW, NOW + pd.Timedelta("30min"), 900.0)
    assert (res2.decisions.status == "no plan").all()  # §8.8: no plan, the device's own fallback runs
