"""Forecasters (spec §7) on synthetic histories, with the provenance checks
that let a backtest prove it used only what was known."""

import numpy as np
import pandas as pd
import pytest

from esther.forecast import (
    DrawForecaster,
    Forecast,
    HoldForecaster,
    Issues,
    LeakError,
    ManualFiringDetector,
    OccupancyForecaster,
    Perfect,
    PeriodicEventForecaster,
    PriceForecaster,
    ProfileForecaster,
    Provenance,
    TimeOfWeekProfile,
    WeatherForecaster,
    check_provenance,
    combine,
    detect_episodes,
    plan_grid,
    step_means,
)

NOW = pd.Timestamp("2025-01-15T12:07Z")


def test_plan_grid_shortens_the_first_step():
    idx, dt = plan_grid(NOW, 36 * 3600, 900)
    assert idx[0] == NOW and dt[0] == 8 * 60
    assert idx[1] == pd.Timestamp("2025-01-15T12:15Z") and (dt[1:] == 900).all()
    assert idx[-1] + pd.Timedelta(seconds=dt[-1]) >= NOW + pd.Timedelta("36h")
    # on a boundary: a full first step; a sliver merges into the next
    idx, dt = plan_grid(pd.Timestamp("2025-01-15T12:15Z"), 3600, 900)
    assert dt[0] == 900 and len(idx) == 4
    idx, dt = plan_grid(pd.Timestamp("2025-01-15T12:14:30Z"), 3600, 900)
    assert dt[0] == 930


def test_step_means_are_exact():
    t = pd.date_range("2025-01-15T12:00Z", periods=5, freq="1h")
    v = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    starts = pd.DatetimeIndex(["2025-01-15T12:00Z", "2025-01-15T13:30Z"])
    m = step_means(t, v, starts, np.array([3600.0, 1800.0]), "instant")
    assert m == pytest.approx([5.0, 17.5])  # mean of a linear ramp over each step
    m = step_means(t, v, starts, np.array([3600.0, 1800.0]), "interval")
    assert m == pytest.approx([0.0, 10.0])  # each value holds to the next


def test_step_means_any_datetime_unit():
    # a history read with microsecond timestamps: the same answer (this bit
    # the weather blend once — asi8 is in the index's own unit)
    t = pd.date_range("2025-01-15T12:00Z", periods=5, freq="1h").as_unit("us")
    v = np.arange(5.0)
    starts = pd.DatetimeIndex(["2025-01-15T12:00Z"]).as_unit("us")
    assert step_means(t, v, starts, np.array([3600.0]), "instant") == pytest.approx([0.5])


def issues(lag=0.0, unit="ns"):
    rows = []
    for k, iss in enumerate(pd.date_range("2025-01-15T00:00Z", periods=6, freq="3h")):
        for h in range(0, 48):
            valid = iss + pd.Timedelta(hours=h)
            rows.append({"issued": iss, "valid": valid, "value": 270.0 + k + 0.1 * h})
    f = pd.DataFrame(rows)
    f["issued"] = f["issued"].dt.as_unit(unit)
    f["valid"] = f["valid"].dt.as_unit(unit)
    return Issues(f, lag=lag, name="nwp")


def test_issue_readable_only_after_its_lag():
    iss = issues(lag=3600.0)
    assert iss.latest(pd.Timestamp("2025-01-15T12:30Z")) == pd.Timestamp("2025-01-15T09:00Z")
    assert iss.latest(pd.Timestamp("2025-01-15T13:00Z")) == pd.Timestamp("2025-01-15T12:00Z")
    assert iss.latest(pd.Timestamp("2025-01-14T23:00Z")) is None


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_weather_blends_from_the_measurement(unit):
    iss = issues(unit=unit)
    measured = pd.Series([280.0], index=pd.DatetimeIndex([NOW - pd.Timedelta("5min")]))
    plain = WeatherForecaster(iss, "outdoor_temperature").forecast(NOW, 36 * 3600, 900)
    blend = WeatherForecaster(iss, "outdoor_temperature", measured=measured, tau=3600).forecast(
        NOW, 36 * 3600, 900
    )
    d = blend["outdoor_temperature"] - plain["outdoor_temperature"]
    fc_now = 270.0 + 4 + 0.1 * (7 / 60)  # the 12:00 issue at 12:07
    assert d[0] == pytest.approx((280.0 - fc_now) * np.exp(-(4 * 60) / 3600), rel=1e-3)
    assert abs(d[-1]) < 1e-6  # the blend has decayed
    p = blend.provenance[0]
    assert p.issued == pd.Timestamp("2025-01-15T12:00Z") and p.data_until <= NOW
    check_provenance(blend)


def test_weather_holds_past_the_issue_and_flags_it():
    iss = issues()
    fc = WeatherForecaster(iss, "t").forecast(pd.Timestamp("2025-01-15T12:00Z"), 72 * 3600, 900)
    est = fc.estimated["t"].to_numpy()
    assert est[-1] and not est[0]


def test_price_known_then_estimated():
    rows = []
    for day in range(30):
        iss = pd.Timestamp("2025-01-01T11:00Z") + pd.Timedelta(days=day)
        start = iss.normalize() + pd.Timedelta(days=1)
        for q in range(96):
            v = start + pd.Timedelta(minutes=15 * q)
            rows.append(
                {
                    "issued": iss,
                    "valid": v,
                    "end": v + pd.Timedelta("15min"),
                    "value": 1e-8 * (1 + (v.hour >= 17) * 2),
                }
            )
    f = pd.DataFrame(rows)
    pf = PriceForecaster(Issues(f), "spot_price", profile_days=14)
    now = pd.Timestamp("2025-01-20T09:00Z")  # tomorrow not yet published
    fc = pf.forecast(now, 36 * 3600, 900)
    est = fc.estimated["spot_price"].to_numpy()
    assert not est[0] and est[-1]
    hours = fc.index.hour
    v = fc["spot_price"]
    assert (
        v[(hours >= 17) & (hours < 23) & est].mean() > 2 * v[(hours < 6) & est].mean()
    )  # the profile learned the evening
    assert fc.provenance[-1].data_until < now


def test_check_provenance_refuses_the_future():
    idx, dt = plan_grid(NOW, 3600, 900)
    fc = Forecast(
        NOW,
        dt,
        pd.DataFrame({"x": np.zeros(len(idx))}, index=idx),
        pd.DataFrame({"x": np.zeros(len(idx), bool)}, index=idx),
        [Provenance("x", "cheat", "x", NOW + pd.Timedelta("1h"))],
    )
    with pytest.raises(LeakError):
        check_provenance(fc)
    perfect = Perfect(
        pd.DataFrame({"x": np.arange(10.0)}, index=pd.date_range(NOW, periods=10, freq="15min")), ("x",)
    )
    pfc = perfect.forecast(NOW, 3600, 900)
    with pytest.raises(LeakError, match="by design"):
        check_provenance(pfc)
    check_provenance(pfc, allow_leaky=True)


def test_profile_recovers_a_known_day():
    idx = pd.date_range("2025-01-01T00:00Z", "2025-01-15T00:00Z", freq="1h", inclusive="left")
    truth = 100.0 * (idx.hour >= 18)
    s = pd.Series(truth + np.random.default_rng(0).normal(0, 1, len(idx)), index=idx)
    prof = TimeOfWeekProfile("day", 60).fit(s)
    starts = pd.date_range("2025-01-15T00:00Z", periods=24, freq="1h")
    p = prof.predict(starts, np.full(24, 3600.0))
    assert np.allclose(p, 100.0 * (starts.hour >= 18), atol=1.5)
    fc = ProfileForecaster(s, "x").forecast(pd.Timestamp("2025-01-15T00:00Z"), 24 * 3600, 3600)
    assert fc.provenance[0].data_until < pd.Timestamp("2025-01-15T00:00Z")


def test_occupancy_follows_now_then_the_profile():
    # home every evening, away by day; at now (noon) someone is unexpectedly home
    ch = []
    for d in range(21):
        day = pd.Timestamp("2025-01-01T00:00Z") + pd.Timedelta(days=d)
        ch += [(day + pd.Timedelta("8h"), 0.0), (day + pd.Timedelta("17h"), 1.0)]
    ch.append((pd.Timestamp("2025-01-22T11:00Z"), 1.0))
    pres = pd.Series([v for _, v in ch], index=pd.DatetimeIndex([t for t, _ in ch]))
    fc = OccupancyForecaster(pres, persistence=3600).forecast(
        pd.Timestamp("2025-01-22T12:00Z"), 24 * 3600, 900
    )
    p = fc["occupancy"]
    assert p[0] > 0.8  # home now
    assert p[8] < 0.3  # two hours on, the profile (away by day)
    assert p[24] > 0.8  # the evening


def test_draws_daily_quantile_covers_most_days():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01T00:00Z", "2025-01-29T00:00Z", freq="5min", inclusive="left")
    q = pd.Series(0.0, index=idx)
    for d in range(28):  # a morning shower at a varying time, a bath some evenings
        day = pd.Timestamp("2025-01-01T00:00Z") + pd.Timedelta(days=d)
        t = day + pd.Timedelta(hours=int(rng.integers(6, 12)))
        q.loc[t : t + pd.Timedelta("10min")] = 6000.0 * rng.uniform(0.5, 1.5)
        if rng.uniform() < 0.15:
            q.loc[day + pd.Timedelta("19h") : day + pd.Timedelta("19h30min")] = 8000.0
    now = pd.Timestamp("2025-01-29T00:00Z")
    daily = DrawForecaster(q, "draw_heat", quantile=0.8).forecast(now, 24 * 3600, 900)
    slot = DrawForecaster(q, "draw_heat", quantile=0.8, method="slot").forecast(now, 24 * 3600, 900)
    e_daily = daily["draw_heat"].sum() * 900 / 3.6e6
    e_slot = slot["draw_heat"].sum() * 900 / 3.6e6
    days = q.resample("1D").mean() * 86400 / 3.6e6
    assert e_daily >= np.quantile(days, 0.7)  # a high quantile of the daily energy
    assert e_slot < days.mean()  # a per-slot quantile of sparse events falls below the mean
    hours = daily.index.hour
    assert daily["draw_heat"][(hours >= 6) & (hours < 12)].mean() > 5 * daily["draw_heat"][hours == 3].mean()
    neg = DrawForecaster(q, "neg", sign=-1.0).forecast(now, 24 * 3600, 900)
    assert (neg["neg"] <= 0).all()


def test_periodic_events_learned_and_merged():
    starts = [pd.Timestamp("2025-01-01T04:00Z") + pd.Timedelta(days=15 * k, hours=1.5 * k) for k in range(4)]
    occ = pd.DataFrame({"start": starts, "end": [s + pd.Timedelta("1h") for s in starts]})
    # one occurrence seen as two pieces 20 minutes apart
    split = pd.concat(
        [
            occ,
            pd.DataFrame(
                {"start": [starts[2] + pd.Timedelta("80min")], "end": [starts[2] + pd.Timedelta("2h")]}
            ),
        ]
    ).sort_values("start")
    ev = PeriodicEventForecaster("dev", "raise", lambda now: split[split.start <= now])
    now = starts[-1] + pd.Timedelta(days=14)
    fc = ev.forecast(now, 72 * 3600, 900)
    assert len(fc.events) == 1
    expect = starts[-1] + pd.Timedelta(days=15, hours=1.5)
    assert abs((fc.events[0].start - expect).total_seconds()) < 3600
    assert fc["dev.raise"].sum() > 0
    early = ev.forecast(starts[0] + pd.Timedelta(days=1), 36 * 3600, 900)
    assert early.events == []  # one occurrence: no period yet


def test_detect_episodes():
    idx = pd.date_range("2025-01-01T00:00Z", periods=20, freq="5min")
    s = pd.Series([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)
    ep = detect_episodes(s, 0.5, min_duration=600, max_gap=300)
    assert len(ep) == 1 and ep.start[0] == idx[1] and ep.end[0] == idx[7]


def test_manual_firing_detected_never_forecast_ahead():
    idx = pd.date_range("2025-01-01T10:00Z", periods=24, freq="5min")
    temp = pd.Series(np.r_[np.full(12, 40.0), np.linspace(60, 300, 12)], index=idx)
    det = ManualFiringDetector(temp, "manual_heat", wood_output=7000.0, wood_threshold=280.0, decay=1800.0)
    now = idx[-1]
    on, _last = det.detected(now)
    assert on
    fc = det.forecast(now, 6 * 3600, 900)
    v = fc["manual_heat"]
    assert v[0] > 1000 and v[-1] < 1.0 and (np.diff(v) <= 1e-9).all()  # the tail decays, nothing new
    planned = pd.Series([1.0], index=pd.DatetimeIndex([idx[0]]))
    cool = pd.Series(
        np.r_[np.full(12, 40.0), np.linspace(60, 200, 12)], index=idx
    )  # rising, below wood, but planned
    assert not ManualFiringDetector(cool, "m", 7000.0, 280.0, window=1800, planned_lit=planned).detected(now)[
        0
    ]
    assert ManualFiringDetector(cool, "m", 7000.0, 280.0, window=1800).detected(now)[0]


def test_hold_and_combine():
    s = pd.Series([290.0, 292.0], index=pd.DatetimeIndex(["2025-01-01T00:00Z", NOW - pd.Timedelta("1h")]))
    h = HoldForecaster(s, "sp").forecast(NOW, 3600, 900)
    assert (h["sp"] == 292.0).all()
    w = WeatherForecaster(issues(), "t").forecast(NOW, 3600, 900)
    both = combine([h, w])
    assert set(both.values.columns) == {"sp", "t"} and len(both.provenance) == 2
    with pytest.raises(ValueError):
        combine([h, h])
