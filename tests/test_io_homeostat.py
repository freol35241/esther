"""esther.io.homeostat on synthetic history stores built here, and the reader
feeding the dataset builder end to end with injected faults it must find."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esther.data import DatasetConfig, SignalSpec, SpikeRule, Window, build_dataset, read_signals
from esther.io.homeostat import Address, HomeostatError, HomeostatHistory

SCHEMA = """
CREATE TABLE rooms (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE series (
  id INTEGER PRIMARY KEY, class TEXT NOT NULL, entity TEXT NOT NULL, aspect TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT '', row_count INTEGER NOT NULL DEFAULT 0,
  oldest_ts INTEGER, newest_ts INTEGER, UNIQUE (class, entity, aspect, source));
CREATE TABLE samples (
  series_id INTEGER NOT NULL REFERENCES series (id), ts INTEGER NOT NULL,
  room_id INTEGER NOT NULL REFERENCES rooms (id), kind INTEGER NOT NULL, value NOT NULL,
  PRIMARY KEY (series_id, ts)) WITHOUT ROWID;
"""

T0 = pd.Timestamp("2025-03-01T00:00:00Z")
US = 1_000_000


def us(seconds: float) -> int:
    return T0.value // 1000 + round(seconds * US)


def make_store(
    path: Path, data: dict[tuple[str, str, str], list[tuple[int, int, object]]], first_id: int = 1
):
    """``data``: (class, entity, aspect) → [(ts µs, kind, value)]. Series ids start
    at ``first_id`` so that two files number the same series differently."""
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.execute("insert into rooms values (1, 'global')")
    for i, ((cls, ent, asp), rows) in enumerate(data.items(), first_id):
        ts = [r[0] for r in rows]
        con.execute(
            "insert into series values (?,?,?,?,'',?,?,?)",
            (i, cls, ent, asp, len(rows), min(ts, default=None), max(ts, default=None)),
        )
        con.executemany("insert into samples values (?,?,1,?,?)", [(i, t, k, v) for t, k, v in rows])
    con.commit()
    con.close()


TEMP = ("state", "outdoor_sensor", "temperature")
COMP = ("state", "boiler", "compressor")
MODE = ("state", "boiler", "mode")
CMD = ("cmd", "boiler", "mode")


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """An archive for the first hour and a live file from 50 min on (an
    overlap of 10 min, where the live file holds different values)."""
    d = tmp_path / "data"
    (d / "archive").mkdir(parents=True)
    t_arch = np.arange(0, 3600, 10)
    t_live = np.arange(3000, 7200, 10)
    make_store(
        d / "archive" / "history-2025-02.db",
        {
            TEMP: [(us(t), 1, 5.0) for t in t_arch],
            COMP: [(us(0), 0, 1), (us(180), 0, 0)],
            MODE: [(us(0), 2, "auto")],
        },
    )
    make_store(
        d / "history.db",
        {
            CMD: [(us(10), 1, 2)],
            MODE: [(us(3000), 2, "boost")],
            COMP: [(us(3000), 0, 1)],
            TEMP: [(us(t), 1, 7.0) for t in t_live],
        },
        first_id=40,
    )
    return d


def test_address_round_trip():
    for text in ("a/b", "cmd:a/b", "forecast:a/b@src", "a/b.c@x"):
        assert str(Address.parse(text)) == text
    assert Address.parse("a/b").cls == "state"
    with pytest.raises(ValueError):
        Address.parse("nope")


def test_merges_files_by_key_not_id(store: Path):
    h = HomeostatHistory.open_directory(store)
    assert [p.name for p in h.paths] == ["history-2025-02.db", "history.db"]
    s = h.read("outdoor_sensor/temperature")
    assert s.index.is_monotonic_increasing and s.index.is_unique and str(s.index.tz) == "UTC"
    assert len(s) == 720  # 0–7190 s every 10 s, the overlap counted once
    # In the overlap the later file (live) wins.
    assert s[T0 + pd.Timedelta(seconds=3000)] == 7.0 and s[T0 + pd.Timedelta(seconds=2990)] == 5.0


def test_kinds_decode(store: Path):
    h = HomeostatHistory.open_directory(store)
    c = h.read("boiler/compressor")
    assert c.dtype == bool and c.tolist() == [True, False, True]
    m = h.read("boiler/mode")
    assert m.tolist() == ["auto", "boost"]
    assert h.read("cmd:boiler/mode").tolist() == [2.0]


def test_time_window(store: Path):
    h = HomeostatHistory.open_directory(store)
    s = h.read("outdoor_sensor/temperature", T0 + pd.Timedelta("10min"), T0 + pd.Timedelta("20min"))
    assert len(s) == 60 and s.index[0] == T0 + pd.Timedelta("10min")
    with pytest.raises(ValueError, match="time zone"):
        h.read("outdoor_sensor/temperature", "2025-03-01T00:10:00")


def test_unknown_address_raises(store: Path):
    h = HomeostatHistory.open_directory(store)
    with pytest.raises(KeyError):
        h.read("no/such")
    # Known to one file only is fine.
    assert len(h.read("cmd:boiler/mode")) == 1


def test_catalogue(store: Path):
    cat = HomeostatHistory.open_directory(store).catalogue().set_index("address")
    assert cat.loc["outdoor_sensor/temperature", "files"] == 2
    assert cat.loc["outdoor_sensor/temperature", "rows"] == 360 + 420
    assert cat.loc["cmd:boiler/mode", "files"] == 1
    assert cat.loc["outdoor_sensor/temperature", "oldest"] == T0


def test_opens_read_only(store: Path):
    # A write through the reader's own connection must fail, in both modes.
    for immutable in (False, True):
        h = HomeostatHistory([store / "history.db"], immutable=immutable)
        con = h._connect(h.paths[0])
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("insert into rooms values (2, 'x')")
        con.close()
    # Control: the same statement on a normal connection succeeds.
    con = sqlite3.connect(store / "history.db")
    con.execute("insert into rooms values (2, 'x')")
    con.rollback()
    con.close()


def test_not_a_history_file(tmp_path: Path):
    p = tmp_path / "x.db"
    sqlite3.connect(p).close()
    with pytest.raises(HomeostatError, match="no such table"):
        HomeostatHistory([p]).read("a/b")
    with pytest.raises(FileNotFoundError):
        HomeostatHistory([tmp_path / "missing.db"])


# -- end to end: injected gap, spike and exclusion must all be found ---------------


@pytest.fixture
def faulty(tmp_path: Path) -> Path:
    t = np.arange(0, 6 * 3600, 10)
    v = 4.0 + 0.0001 * t
    spike_at = 7200  # s
    v[t == spike_at] += 20.0
    keep = ~((t >= 3 * 3600) & (t < 3 * 3600 + 1800))  # 30 min hole at 03:00
    rows = [(us(a), 1, float(b)) for a, b in zip(t[keep], v[keep], strict=True)]
    # Compressor: on 3 of every 5 minutes.
    comp = [r for k in range(0, 6 * 12) for r in ((us(k * 300), 0, 1), (us(k * 300 + 180), 0, 0))]
    half = us(3 * 3600)
    d = tmp_path / "e2e"
    (d / "archive").mkdir(parents=True)
    make_store(d / "archive" / "history-a.db", {TEMP: [r for r in rows if r[0] < half], COMP: comp[:72]})
    make_store(d / "history.db", {COMP: comp[72:], TEMP: [r for r in rows if r[0] >= half]}, first_id=9)
    return d


E2E = DatasetConfig(
    step=300.0,
    min_segment=1800.0,
    tolerance=60.0,
    exclude=(Window(T0 + pd.Timedelta("5h"), T0 + pd.Timedelta("5h10min"), "test outage"),),
    signals={
        "outdoor_temperature": SignalSpec("outdoor_temperature", unit="°C", spike=SpikeRule(5.0, 5)),
        "compressor": SignalSpec("compressor", kind="binary", tolerance=600.0),
    },
)
ADDR = {"outdoor_temperature": "outdoor_sensor/temperature", "compressor": "boiler/compressor"}


def test_end_to_end_finds_injected_faults(faulty: Path):
    h = HomeostatHistory.open_directory(faulty, immutable=True)
    start, end = T0, T0 + pd.Timedelta("6h")
    ds = build_dataset(E2E, read_signals(h, ADDR, E2E, start, end), start, end)

    m = ds.masked["outdoor_temperature"]
    assert m[m == "spike"].index.tolist() == [T0 + pd.Timedelta(seconds=7200)]
    excl = m[m == "excluded"].index
    assert len(excl) == 60 and excl.min() == T0 + pd.Timedelta("5h") and set(m) == {"spike", "excluded"}

    bounds = [
        ((s.start - T0) / pd.Timedelta("1min"), (s.end - T0) / pd.Timedelta("1min")) for s in ds.segments
    ]
    assert bounds == [(0, 180), (210, 300), (310, 360)]  # the gap at 03:00, the exclusion at 05:00

    on = ds.values["compressor"][ds.usable]
    np.testing.assert_allclose(on, 0.6)
    assert ds.values["outdoor_temperature"].iloc[0] == pytest.approx(4.0 + 0.0001 * 145 + 273.15)


def test_end_to_end_control_without_faults(faulty: Path):
    # Control: switch the spike rule and the exclusion off — the spike is no
    # longer masked and only the real gap breaks segments.
    cfg = DatasetConfig(
        step=300.0,
        min_segment=1800.0,
        tolerance=60.0,
        signals={
            "outdoor_temperature": SignalSpec("outdoor_temperature", unit="°C"),
            "compressor": SignalSpec("compressor", kind="binary", tolerance=600.0),
        },
    )
    h = HomeostatHistory.open_directory(faulty)
    start, end = T0, T0 + pd.Timedelta("6h")
    ds = build_dataset(cfg, read_signals(h, ADDR, cfg, start, end), start, end)
    assert len(ds.masked["outdoor_temperature"]) == 0
    assert len(ds.segments) == 2


def test_read_signals_needs_every_address(faulty: Path):
    h = HomeostatHistory.open_directory(faulty)
    with pytest.raises(KeyError, match="compressor"):
        read_signals(
            h, {"outdoor_temperature": "outdoor_sensor/temperature"}, E2E, T0, T0 + pd.Timedelta("1h")
        )
