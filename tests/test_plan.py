"""The planner (spec §8) on a synthetic house: dynamics, constraints,
policy, objective, the audit (negative controls), determinism, warm start."""

import numpy as np
import pandas as pd
import pytest
from synthetic_plan import K, forecast, house, planner, x0

from esther.forecast import Event, LeakError, Provenance
from esther.plan import MILP, PlanError, SolveOptions

NOW = pd.Timestamp("2025-01-15T20:00Z")  # the 12 h horizon holds the cheap night and the morning draw


def cheap_nights(hours):
    return np.where((hours >= 0) & (hours < 6), 0.02, 0.30)


@pytest.fixture(scope="module")
def base_plan():
    h = house()
    pl = planner(h)
    fc = forecast(
        NOW, h, outdoor=0.0, price=cheap_nights, draws=lambda hr: np.where((hr >= 7) & (hr < 8), 3000.0, 0)
    )
    return h, pl, fc, pl.plan(NOW, x0(h), fc)


def resimulate(pl, plan, fc):
    """The plan's decisions through the plan-mode model, step by step."""
    m = pl.model
    x = np.array([plan.state[s] for s in m.states])
    frame = pd.DataFrame({u.name: fc[u.name] for u in m.inputs if hasattr(u, "name")}, index=fc.index)
    D = m.input_array(frame)
    U = plan.decisions[list(m.decisions)].to_numpy()
    out = [x]
    for k, dt in enumerate(fc.dt):
        Ad, Bd, Gd = pl._discrete(float(dt))
        x = Ad @ x + Bd @ U[k] + Gd @ D[k]
        out.append(x)
    return np.array(out)


def test_plan_follows_the_dynamics(base_plan):
    # positive control: the plan's trajectory is its decisions through the model
    _h, pl, fc, plan = base_plan
    sim = resimulate(pl, plan, fc)
    assert np.allclose(sim, plan.trajectories.to_numpy(), atol=1e-4)
    assert plan.solver["status"] in ("optimal", "limit") and plan.solver["gap"] < 0.05
    assert plan.flagged == (plan.solver["status"] != "optimal")
    assert plan.mode == "active"


def test_mixing_limit_and_service(base_plan):
    _h, pl, _fc, plan = base_plan
    tr, dec = plan.trajectories, plan.decisions
    ua = pl.values["radiators"]["UA_rad"]
    lim = ua * (tr["tank.jacket"].to_numpy()[:-1] - tr["living"].to_numpy()[:-1])
    assert (dec["radiators.Q"].to_numpy() <= lim + 1e-3).all()
    due = dec["dhw_service.due"].to_numpy() > 0
    assert due.any()
    # the tank is at service temperature when the forecast draw begins
    assert (
        tr["tank.dhw"].to_numpy()[:-1][due] >= K + 45.0 - 1e-3 - dec["dhw_service.below"].to_numpy()[due]
    ).all()


def test_heat_pump_runs_when_power_is_cheap(base_plan):
    _h, _pl, _fc, plan = base_plan
    hours = (plan.index.hour + plan.index.minute / 60).to_numpy()
    cheap = (hours >= 0) & (hours < 6)
    e = plan.decisions["heat_pump.P_el"].to_numpy() * plan.dt
    assert e[cheap].sum() / cheap.sum() > 2 * e[~cheap].sum() / (~cheap).sum()


def test_stove_respects_allowed_hours_and_min_burn():
    h = house(pellet_price=0.01)  # pellets nearly free: the stove is the source of choice
    pl = planner(h)
    fc = forecast(NOW, h, outdoor=-10.0, price=0.5)
    plan = pl.plan(NOW, x0(h, zone=19.6), fc)
    lv = plan.decisions["stove.level"].to_numpy()
    loc = plan.index.tz_convert("UTC").hour
    assert (lv[(loc < 6) | (loc >= 22)] == 0).all()
    assert (lv > 0).any()
    lit = lv > 0
    starts = np.flatnonzero(lit[1:] & ~lit[:-1]) + 1
    for s in starts:  # every burn lasts at least the minimum (1 h = 4 steps) unless the horizon ends
        run = np.argmax(~lit[s:]) if (~lit[s:]).any() else len(lit) - s
        assert run >= 4
    stops = np.flatnonzero(~lit[1:] & lit[:-1]) + 1
    for s in stops:  # and a cool-down of an hour before the next start
        nxt = starts[starts > s]
        if len(nxt):
            assert nxt[0] - s >= 4


@pytest.mark.parametrize("form", ["modulating = true", ""])
def test_stove_ignition_loses_output(form):
    # both forms: the output continuous between levels (one binary per
    # step) and the discrete one-hot levels
    h = house(pellet_price=0.01, stove_extra=form)
    pl = planner(h)
    plan = pl.plan(NOW, x0(h, zone=19.6), forecast(NOW, h, outdoor=-10.0, price=0.5))
    lv = plan.decisions["stove.level"].to_numpy()
    q = plan.decisions["stove.Q"].to_numpy()
    lit = lv > 0
    s = np.flatnonzero(lit & ~np.r_[False, lit[:-1]])
    assert len(s)
    k = s[0]
    full = (
        plan.decisions["stove.power"].to_numpy()[k]
        if "stove.power" in plan.decisions
        else [1000.0, 4000.0][int(lv[k]) - 1]
    )
    assert q[k] < full - 1.0  # the first step after a start delivers less
    assert (lv[(plan.index.hour < 6) | (plan.index.hour >= 22)] == 0).all()


def test_a_burn_under_way_is_carried_in():
    h = house(pellet_price=10.0)  # pellets dear: the plan would not light it
    pl = planner(h)
    fc = forecast(NOW, h, outdoor=5.0, price=0.05)
    plan = pl.plan(NOW, x0(h), fc, devices={"stove": {"on": True, "on_for": 900.0, "level": 1}})
    lv = plan.decisions["stove.level"].to_numpy()
    assert (lv[:3] > 0).all()  # min burn of 4 steps: 3 more to go
    plan2 = pl.plan(NOW, x0(h), fc, devices={"stove": {"on": False, "off_for": 900.0}})
    assert (plan2.decisions["stove.level"].to_numpy() == 0).all()


def test_terminal_value_keeps_the_store():
    h = house()
    fc = forecast(NOW, h, outdoor=5.0, price=0.10)
    with_tv = planner(h).plan(NOW, x0(h, tank=55.0), fc)
    without = planner(h, terminal_factor=0.0).plan(NOW, x0(h, tank=55.0), fc)
    end = lambda p: p.trajectories["tank.dhw"].iloc[-1]  # noqa: E731
    assert end(with_tv) >= end(without) - 1e-6
    assert with_tv.solver["terminal_per_kwh"] > 0


def test_comfort_band_widens_from_a_cold_start():
    h = house()
    fc = forecast(NOW, h, outdoor=-5.0)
    plan = planner(h).plan(NOW, x0(h, zone=15.0, tank=40.0), fc)  # below the 16 °C floor
    z = plan.trajectories["living"].to_numpy()
    assert z[-1] > z[0]
    assert plan.decisions["comfort_band.low"].iloc[0] < K + 19.5


def test_audit_catches_a_violating_plan(monkeypatch):
    # negative control: a solver that returns a point violating a constraint
    # must not produce a plan
    h = house()
    pl = planner(h)
    fc = forecast(NOW, h)
    real = MILP.solve

    def corrupt(self, *a, **k):
        sol = real(self, *a, **k)
        u = self["u"]
        j = pl.model.decisions.index("radiators.Q")
        sol.x = sol.x.copy()
        sol.x[u[0, j]] += 50.0  # 50 kW more radiator heat than the tank can give
        return sol

    monkeypatch.setattr(MILP, "solve", corrupt)
    with pytest.raises(PlanError, match="audit"):
        pl.plan(NOW, x0(h), fc)


def test_milp_check_names_the_violated_rows(base_plan):
    _h, pl, fc, plan = base_plan
    lin = pl._initial_lin(fc, np.array([plan.state[s] for s in pl.model.states]), None)
    ctx = pl.build(NOW, np.array([plan.state[s] for s in pl.model.states]), fc, {}, lin, {})
    sol = ctx.milp.solve(SolveOptions(time_limit=5, gap=0.03))
    assert ctx.milp.check(sol.x) == []
    x = sol.x.copy()
    x[ctx.milp["stove.on"]] = 0.5  # fractional: not integral
    bad = ctx.milp.check(x)
    assert any("not integral" in b for b in bad)


def test_deterministic():
    # N2: the same inputs give the same plan
    h = house()
    fc = forecast(NOW, h, outdoor=-2.0, price=cheap_nights)
    a = planner(h, solve=SolveOptions(time_limit=60, gap=0.02, node_limit=50)).plan(NOW, x0(h), fc)
    b = planner(h, solve=SolveOptions(time_limit=60, gap=0.02, node_limit=50)).plan(NOW, x0(h), fc)
    assert a.solver["deterministic"]
    assert np.array_equal(a.decisions.to_numpy(), b.decisions.to_numpy())
    assert a.id == b.id


def test_warm_start_from_the_previous_plan(base_plan):
    h, pl, _fc, plan = base_plan
    later = NOW + pd.Timedelta("15min")
    fc2 = forecast(later, h, outdoor=0.0, price=cheap_nights)
    x = {s: float(plan.trajectories[s].iloc[1]) for s in plan.trajectories}
    p2 = pl.plan(later, x, fc2, previous=plan)
    assert p2.solver["warm_started"]


def test_first_step_shortened_to_the_grid():
    h = house()
    now = pd.Timestamp("2025-01-15T12:07Z")
    plan = planner(h).plan(now, x0(h), forecast(now, h))
    assert plan.dt[0] == 8 * 60 and (plan.dt[1:] == 900).all()
    assert plan.index[1] == pd.Timestamp("2025-01-15T12:15Z")


def test_heat_output_relinearised():
    # a heat output falling with its sink: the plan is re-solved when the
    # planned sink moves the coefficient (§8.5)
    h = house(hp_extra='sink_reference = "45 °C"')
    pl = planner(h)
    pl.values["heat_pump"]["Q"] = 2000.0
    pl.values["heat_pump"]["dQ_dT"] = 40.0
    plan = pl.plan(NOW, x0(h, tank=35.0), forecast(NOW, h, outdoor=-5.0, price=cheap_nights))
    ps = plan.solver["passes"]
    assert len(ps) == 2 and ps[0]["coefficient_moved"] > 0.05


def test_stale_inputs_refused():
    h = house()
    pl = planner(h, max_age={"spot_price": 3600.0})
    fc = forecast(NOW, h)
    fc.provenance.append(
        Provenance("spot_price", "test", "x", NOW - pd.Timedelta("1h"), issued=NOW - pd.Timedelta("5h"))
    )
    with pytest.raises(PlanError, match="stale"):
        pl.plan(NOW, x0(h), fc)
    fc.provenance[-1] = Provenance("spot_price", "test", "x", NOW + pd.Timedelta("1h"))
    with pytest.raises(LeakError):
        pl.plan(NOW, x0(h), fc)


def test_advisory_when_the_model_is_not_accepted():
    from synthetic_plan import WRAPPERS

    h = house(extra=WRAPPERS)
    plan = planner(h, accepted=False).plan(NOW, x0(h), forecast(NOW, h))
    assert plan.advisory
    from esther.actuate import translate

    cmds = translate(plan, h)
    assert cmds and all(not c.actionable and c.plan_id == plan.id for c in cmds)


def test_unclaimed_uncommandable_heater_stays_off():
    h = house(heater_extra="commandable = false")
    plan = planner(h).plan(NOW, x0(h, zone=17.0, tank=40.0), forecast(NOW, h, outdoor=-15.0))
    assert np.allclose(plan.decisions["heater.share"].to_numpy(), 0.0)


def test_manual_override_pins_a_decision():
    h = house()
    pins = {"heat_pump.delta": np.r_[np.ones(4), np.full(200, np.nan)]}
    plan = planner(h).plan(NOW, x0(h), forecast(NOW, h, price=0.9), pins=pins)
    assert np.allclose(plan.decisions["heat_pump.delta"].to_numpy()[:4], 1.0)


def test_events_reach_the_context():
    h = house()
    fc = forecast(NOW, h)
    fc.events.append(Event("heat_pump", "x", NOW + pd.Timedelta("1h"), NOW + pd.Timedelta("2h")))
    pl = planner(h)
    x = np.array([x0(h)[s] for s in pl.model.states])
    ctx = pl.build(NOW, x, fc, {}, pl._initial_lin(fc, x, None), {})
    c = ctx.event_cover("heat_pump")
    assert c.sum() == pytest.approx(4.0) and c[:4].sum() == 0


def test_terminal_value_prices_at_the_estimated_part():
    # §8.6: the mean over the part of the price forecast that is a guess
    h = house()
    fc = forecast(NOW, h, price=lambda hr: np.where(hr < 2, 0.5, 0.1))
    est = (fc.index.hour >= 2) & (fc.index.hour < 8)
    fc.estimated["spot_price"] = est
    plan = planner(h).plan(NOW, x0(h), fc)
    price = fc["spot_price"] * 3.6e6 + 0.05  # €/kWh with the fee
    hp = price * 0.6 / 2.0  # the heat pump's marginal cost of heat
    expect = 0.9 * min(hp[est].mean(), price[est].mean())
    assert plan.solver["terminal_per_kwh"] == pytest.approx(expect, rel=1e-6)
