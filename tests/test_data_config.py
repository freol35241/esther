"""The [dataset] table: a positive control, then the refusals (spec §6.1)."""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

import pytest

from esther import HouseError, load_house, loads_house
from esther.data import parse_dataset_config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "houses"

CONFIG = """\
[dataset]
step = "5min"
min_segment = "2h"
tolerance = "10min"

[[dataset.exclude]]
start = 2025-01-10T06:00:00Z
duration = "8 h"
reason = "power cut"

[[dataset.exclude]]
start = "2025-02-01T00:00:00+01:00"
end = "2025-02-02T00:00:00+01:00"

[dataset.signal.outdoor_temperature]
unit = "°C"
range = [-45, 45]
invalid_values = [-99.9]
spike = { threshold = 4.0, window = 7 }
stuck = "6 h"
tolerance = "30min"

[dataset.signal.compressor]
kind = "binary"

[dataset.signal.mode]
kind = "categorical"
required = false
"""


def parse(text: str):
    return parse_dataset_config(tomllib.loads(text)["dataset"])


def test_valid_config_parses():
    cfg, problems = parse(CONFIG)
    assert problems == []
    assert cfg.step == 300 and cfg.min_segment == 7200 and cfg.tolerance == 600
    assert [w.reason for w in cfg.exclude] == ["power cut", ""]
    w0, w1 = cfg.exclude
    assert w0.end - w0.start == dt.timedelta(hours=8)
    assert w1.start == dt.datetime(2025, 1, 31, 23, tzinfo=dt.UTC)
    out = cfg.signals["outdoor_temperature"]
    assert out.range == (-45, 45) and out.invalid_values == (-99.9,)
    assert out.spike.threshold == 4.0 and out.spike.window == 7
    assert out.stuck == 6 * 3600 and cfg.tolerance_of("outdoor_temperature") == 1800
    assert cfg.tolerance_of("compressor") == 600
    assert out.to_si() == (1.0, 273.15)
    assert cfg.required == ("outdoor_temperature", "compressor")


def test_defaults():
    cfg, problems = parse('[dataset.signal.x]\nunit = "kW"\n')
    assert problems == []
    assert cfg.step == 300 and cfg.min_coverage == 1.0 and cfg.exclude == ()
    assert cfg.signals["x"].to_si() == (1000.0, 0.0)


@pytest.mark.parametrize(
    ("old", "new", "key", "fragment"),
    [
        ('step = "5min"', 'step = "5 kg"', "dataset.step", "dimension"),
        ('step = "5min"', 'stepp = "5min"', "dataset.stepp", "unknown key"),
        (
            'duration = "8 h"',
            'duration = "8 h"\nend = 2025-01-11T00:00:00Z',
            "dataset.exclude[1]",
            "exactly one",
        ),
        (
            "start = 2025-01-10T06:00:00Z",
            "start = 2025-01-10T06:00:00",
            "dataset.exclude[1].start",
            "no UTC offset",
        ),
        (
            'end = "2025-02-02T00:00:00+01:00"',
            'end = "2025-01-02T00:00:00+01:00"',
            "dataset.exclude[2]",
            "before it starts",
        ),
        ("range = [-45, 45]", "range = [45, -45]", "dataset.signal.outdoor_temperature.range", "low < high"),
        ('unit = "°C"', 'unit = "parsecs"', "dataset.signal.outdoor_temperature.unit", "unknown unit"),
        ("window = 7", "window = 4", "dataset.signal.outdoor_temperature.spike.window", "odd integer"),
        ('kind = "binary"', 'kind = "boolean"', "dataset.signal.compressor.kind", "must be one of"),
        ('kind = "binary"', 'kind = "binary"\nstuck = "1 h"', "dataset.signal.compressor.stuck", "normal"),
        ("required = false", "required = false\nrange = [0, 1]", "dataset.signal.mode", "categorical"),
    ],
)
def test_refusals(old, new, key, fragment):
    assert old in CONFIG
    cfg, problems = parse(CONFIG.replace(old, new, 1))
    assert cfg is None
    hits = [m for k, m in problems if k == key]
    assert hits and any(fragment in m for m in hits), problems


def test_no_required_signal_is_refused():
    cfg, problems = parse("[dataset.signal.a]\nrequired = false\n")
    assert cfg is None and any("no required signal" in m for _, m in problems)


HOUSE = """\
schema = 1
name = "test"

[[component]]
type = "Ambient"
name = "outdoor"
signal = "outdoor_temperature"

[[component]]
type = "Zone"
name = "house"

[[component]]
type = "Coupling"
between = ["house", "outdoor"]

"""


def test_house_carries_its_dataset():
    house = loads_house(HOUSE + CONFIG)
    assert house.dataset is not None
    assert set(house.dataset.signals) == {"outdoor_temperature", "compressor", "mode"}
    assert loads_house(HOUSE).dataset is None


def test_house_reports_dataset_problems_with_lines():
    bad = HOUSE + CONFIG.replace("range = [-45, 45]", "range = [45, -45]")
    with pytest.raises(HouseError) as e:
        loads_house(bad)
    (issue,) = e.value.issues
    assert issue.key == "dataset.signal.outdoor_temperature.range"
    assert bad.splitlines()[issue.line - 1].startswith("range = [45, -45]")


def test_examples_with_dataset_load():
    house = load_house(EXAMPLES / "multi_source.toml")
    assert house.dataset is not None and "compressor" in house.dataset.signals
