"""Dataset builder (spec §6.1): hold-aggregation, masks, exclusions, segments.

Every detector is tested with an injected fault it must find (positive
control) and, where a negative result is relied on, the same data with the
rule switched off, which must then let the fault through.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from esther.data import DatasetConfig, SignalSpec, SpikeRule, Window, build_dataset
from esther.data.resample import hold_integrals, mask_samples

T0 = pd.Timestamp("2025-01-01T00:00:00Z")


def series(seconds, values, name="x") -> pd.Series:
    idx = pd.DatetimeIndex([T0 + pd.Timedelta(seconds=float(s)) for s in seconds], name="time")
    return pd.Series(values, index=idx, name=name)


def cfg(**signals: SignalSpec) -> DatasetConfig:
    return DatasetConfig(step=300.0, min_segment=0.0, tolerance=60.0, signals=signals)


# -- hold_integrals ----------------------------------------------------------


def test_hold_integrals_by_hand():
    # Value 2 from t=0 to 100, value 4 from 100, held at most 60 s (tolerance):
    # held over [0, 60) and [100, 160).
    t = np.array([0.0, 100.0])
    v = np.array([2.0, 4.0])
    edges = np.array([0.0, 50.0, 120.0, 200.0])
    integ, cov = hold_integrals(t, v, edges, tolerance=60.0, breaks=np.array([]))
    np.testing.assert_allclose(cov, [50, 10 + 20, 40])
    np.testing.assert_allclose(integ, [100, 2 * 10 + 4 * 20, 4 * 40])


def test_hold_stops_at_a_break():
    t = np.array([0.0])
    _integ, cov = hold_integrals(t, np.array([1.0]), np.array([0.0, 100.0]), 1e9, breaks=np.array([30.0]))
    assert cov[0] == pytest.approx(30.0)


def test_hold_integrals_empty():
    integ, cov = hold_integrals(np.array([]), np.array([]), np.array([0.0, 300.0]), 60.0, np.array([]))
    assert integ.tolist() == [0.0] and cov.tolist() == [0.0]


# -- aggregation ---------------------------------------------------------------


def test_analog_mean_and_unit():
    # 10 s samples of a ramp 0..: the step mean of the held values is known.
    t = np.arange(0, 600, 10)
    s = series(t, t / 10.0)
    ds = build_dataset(cfg(x=SignalSpec("x", unit="°C")), {"x": s}, T0, T0 + pd.Timedelta("10min"))
    held_means = [np.mean(t[(t >= a) & (t < a + 300)] / 10.0) for a in (0, 300)]
    np.testing.assert_allclose(ds.values["x"], np.array(held_means) + 273.15)
    assert ds.valid["x"].all() and len(ds.segments) == 1


def test_binary_on_fraction():
    # On for 3 of 5 minutes: δ = 0.6 (spec §6.1's own example).
    s = series([0, 180], [True, False])
    c = DatasetConfig(
        step=300.0, min_segment=0.0, tolerance=600.0, signals={"c": SignalSpec("c", kind="binary")}
    )
    ds = build_dataset(c, {"c": s}, T0, T0 + pd.Timedelta("5min"))
    assert ds.values["c"].iloc[0] == pytest.approx(0.6)


def test_binary_accepts_words_and_numbers():
    s = series([0, 60, 120, 180, 240], ["on", "OFF", 1, 0.0, "true"])
    c = DatasetConfig(
        step=300.0, min_segment=0.0, tolerance=600.0, signals={"c": SignalSpec("c", kind="binary")}
    )
    ds = build_dataset(c, {"c": s}, T0, T0 + pd.Timedelta("5min"))
    assert ds.values["c"].iloc[0] == pytest.approx((60 + 60 + 60) / 300)


def test_categorical_majority():
    s = series([0, 100, 160], ["a", "b", "a"])
    c = DatasetConfig(
        step=300.0, min_segment=0.0, tolerance=600.0, signals={"m": SignalSpec("m", kind="categorical")}
    )
    ds = build_dataset(c, {"m": s}, T0, T0 + pd.Timedelta("5min"))
    assert ds.values["m"].iloc[0] == "a"  # 240 s of a, 60 s of b


# -- gaps and segments -------------------------------------------------------


def gappy():
    # 10 s samples for 2 h with a 30 min hole starting at 50 min.
    t = np.arange(0, 7200, 10)
    t = t[(t < 3000) | (t >= 4800)]
    return series(t, np.full(len(t), 20.0))


def test_injected_gap_splits_the_segment():
    ds = build_dataset(cfg(x=SignalSpec("x")), {"x": gappy()}, T0, T0 + pd.Timedelta("2h"))
    assert [(s.start - T0, s.end - T0) for s in ds.segments] == [
        (pd.Timedelta("0min"), pd.Timedelta("50min")),
        (pd.Timedelta("80min"), pd.Timedelta("120min")),
    ]
    assert not ds.valid["x"].iloc[10:16].any() and np.isnan(ds.values["x"].iloc[10:16]).all()


def test_gap_within_tolerance_is_held_control():
    # Control for the test above: with a tolerance longer than the hole, the
    # same data is one segment — so it was the gap that split it.
    c = dataclasses.replace(cfg(x=SignalSpec("x")), tolerance=3600.0)
    ds = build_dataset(c, {"x": gappy()}, T0, T0 + pd.Timedelta("2h"))
    assert len(ds.segments) == 1


def test_short_segments_are_dropped():
    c = dataclasses.replace(cfg(x=SignalSpec("x")), min_segment=3600.0)
    ds = build_dataset(c, {"x": gappy()}, T0, T0 + pd.Timedelta("2h"))
    assert len(ds.segments) == 0 and ds.usable.sum() == 18  # 10 + 8 usable steps, none ≥ 1 h


def test_optional_signal_does_not_break_segments():
    other = series(np.arange(0, 7200, 10), np.full(720, 1.0))
    c = cfg(x=SignalSpec("x", required=False), y=SignalSpec("y"))
    ds = build_dataset(c, {"x": gappy(), "y": other}, T0, T0 + pd.Timedelta("2h"))
    assert len(ds.segments) == 1 and not ds.valid["x"].all()


# -- exclusions --------------------------------------------------------------


def steady(n=720):
    return series(np.arange(0, n * 10, 10), np.full(n, 20.0))


def test_global_exclusion_window():
    w = Window(T0 + pd.Timedelta("32min"), T0 + pd.Timedelta("41min"), "outage")
    c = dataclasses.replace(cfg(x=SignalSpec("x")), exclude=(w,))
    ds = build_dataset(c, {"x": steady()}, T0, T0 + pd.Timedelta("2h"))
    # Steps 30–35, 35–40 and 40–45 overlap [32, 41).
    assert ds.excluded.to_numpy().nonzero()[0].tolist() == [6, 7, 8]
    assert next(s.end - T0 for s in ds.segments) == pd.Timedelta("30min")
    assert ds.segments[1].start - T0 == pd.Timedelta("45min")
    assert ds.report.loc["x", "masked_excluded"] == 54  # 9 min of 10 s samples


def test_per_signal_exclusion():
    w = Window(T0 + pd.Timedelta("30min"), T0 + pd.Timedelta("40min"))
    c = cfg(x=SignalSpec("x", exclude=(w,)))
    ds = build_dataset(c, {"x": steady()}, T0, T0 + pd.Timedelta("2h"))
    assert not ds.excluded.any()
    assert ds.valid["x"].to_numpy().nonzero()[0].tolist() == [i for i in range(24) if i not in (6, 7)]


def test_hold_does_not_cross_an_exclusion():
    # One sample before a window, tolerance huge: the hold must stop at the window.
    s = series([0, 3000], [1.0, 1.0])
    w = Window(T0 + pd.Timedelta("10min"), T0 + pd.Timedelta("20min"))
    c = DatasetConfig(
        step=300.0, min_segment=0.0, tolerance=1e6, exclude=(w,), signals={"x": SignalSpec("x")}
    )
    ds = build_dataset(c, {"x": s}, T0, T0 + pd.Timedelta("50min"))
    assert ds.coverage["x"].iloc[4] == 0.0  # 20–25 min: nothing held across the window


# -- masks ---------------------------------------------------------------------


def spiky():
    t = np.arange(0, 3600, 10)
    v = 20.0 + 0.001 * t
    v[100] += 15.0  # an ADC spike at t = 1000 s
    v[200:203] = -99.9  # a sentinel at t = 2000–2020 s
    v[250] = 99.0  # out of range at t = 2500 s
    return series(t, v)


def test_mask_rules_find_what_was_injected():
    spec = SignalSpec("x", range=(-40.0, 60.0), invalid_values=(-99.9,), spike=SpikeRule(3.0, 5))
    ds = build_dataset(cfg(x=spec), {"x": spiky()}, T0, T0 + pd.Timedelta("1h"))
    m = ds.masked["x"]
    secs = ((m.index - T0).total_seconds()).astype(int).tolist()
    assert dict(zip(secs, m.tolist(), strict=True)) == {
        1000: "spike",
        2000: "invalid_value",
        2010: "invalid_value",
        2020: "invalid_value",
        2500: "range",
    }
    # The masked spike is gone from the step mean (step 15–20 min holds t = 1000).
    assert ds.values["x"].iloc[3] == pytest.approx(np.mean(20.0 + 0.001 * np.arange(900, 1200, 10)), abs=1e-3)
    assert ds.valid["x"].all()  # masked samples are held over by their neighbours


def test_spike_survives_without_the_rule_control():
    spec = SignalSpec("x", range=(-40.0, 60.0), invalid_values=(-99.9,))
    ds = build_dataset(cfg(x=spec), {"x": spiky()}, T0, T0 + pd.Timedelta("1h"))
    assert "spike" not in ds.masked["x"].tolist()
    clean = np.mean(20.0 + 0.001 * np.arange(900, 1200, 10))
    assert ds.values["x"].iloc[3] == pytest.approx(clean + 15.0 / 30, abs=1e-3)  # one of 30 samples +15 K


def test_spike_rule_spares_a_genuine_step():
    t = np.arange(0, 600, 10.0)
    v = np.where(t < 300, 20.0, 40.0)
    rule = mask_samples(t, v, SignalSpec("x", spike=SpikeRule(3.0, 5)), [])
    assert (rule < 0).all()


def test_stuck_sensor():
    t = np.arange(0, 7200, 10.0)
    v = 20 + 0.01 * np.sin(t / 100)
    v[360:] = 21.0  # frozen for the second hour
    rule = mask_samples(t, v, SignalSpec("x", stuck=1800.0), [])
    assert (rule[360:] == 3).all() and (rule[:360] < 0).all()
    assert (mask_samples(t, v, SignalSpec("x"), []) < 0).all()  # control: no rule, no mask


def test_report_counts():
    spec = SignalSpec("x", range=(-40.0, 60.0), invalid_values=(-99.9,), spike=SpikeRule(3.0, 5))
    ds = build_dataset(cfg(x=spec), {"x": spiky()}, T0, T0 + pd.Timedelta("1h"))
    r = ds.report.loc["x"]
    assert (r.samples, r.masked_spike, r.masked_invalid_value, r.masked_range) == (360, 1, 3, 1)
    assert r.steps_valid == 12 and "12 steps" in ds.summary()


# -- grid and inputs -----------------------------------------------------------


def test_grid_is_epoch_aligned():
    s = series([7, 400], [1.0, 1.0])
    ds = build_dataset(cfg(x=SignalSpec("x")), {"x": s})
    assert ds.index[0] == T0 and ds.index.tz is not None


def test_missing_samples_and_naive_index_are_refused():
    with pytest.raises(KeyError):
        build_dataset(cfg(x=SignalSpec("x")), {})
    naive = pd.Series([1.0], index=pd.DatetimeIndex(["2025-01-01"]))
    with pytest.raises(ValueError, match="tz-aware"):
        build_dataset(cfg(x=SignalSpec("x")), {"x": naive}, T0, T0 + pd.Timedelta("5min"))


def test_segment_frames():
    ds = build_dataset(cfg(x=SignalSpec("x")), {"x": gappy()}, T0, T0 + pd.Timedelta("2h"))
    frames = list(ds.segment_frames())
    assert [len(f) for f in frames] == [s.n_steps for s in ds.segments] == [10, 8]
