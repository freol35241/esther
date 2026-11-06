"""Dataset configuration (spec §6.1): the ``[dataset]`` table of a house file.

What a dataset needs to know about a house's history — which named signals it
holds, how each is aggregated into a step, what an implausible value of each
looks like, and which periods are not representative — is a property of the
house, so it is declared next to the components rather than in code::

    [dataset]
    step = "5min"
    min_segment = "2h"

    [[dataset.exclude]]
    start = 2025-01-10T06:00:00Z
    duration = "8 h"
    reason = "power cut"

    [dataset.signal.outdoor_temperature]
    unit = "°C"
    range = [-45, 45]
    spike = { threshold = 4.0 }

    [dataset.signal.compressor]
    kind = "binary"

Signal *names* only. Mapping a name to where its history lives (a database
series, a bus topic) is the integration layer's business (§5, §9) and is passed
to :func:`esther.data.read_signals` separately.

Masks (``range``, ``invalid_values``, ``spike``, ``stuck``) are written in the
signal's **source** units, the units its history is recorded in — that is how a
sensor fault is observed and described. ``unit`` converts the aggregated values
to SI on output (N4).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core.units import TIME, UnitError, parse_quantity, si_conversion

__all__ = [
    "DatasetConfig",
    "SignalSpec",
    "SpikeRule",
    "Window",
    "parse_dataset_config",
]

Kind = Literal["analog", "binary", "categorical"]
KINDS: tuple[str, ...] = ("analog", "binary", "categorical")

_DATASET_KEYS = ("step", "min_segment", "min_coverage", "tolerance", "exclude", "signal")
_SIGNAL_KEYS = (
    "kind",
    "required",
    "unit",
    "tolerance",
    "min_coverage",
    "range",
    "invalid_values",
    "spike",
    "stuck",
    "exclude",
)
_WINDOW_KEYS = ("start", "end", "duration", "reason")
_SPIKE_KEYS = ("threshold", "window")


@dataclass(frozen=True)
class Window:
    """A half-open UTC time range ``[start, end)``."""

    start: dt.datetime
    end: dt.datetime
    reason: str = ""

    def overlaps(self, start: dt.datetime, end: dt.datetime) -> bool:
        return self.start < end and self.end > start


@dataclass(frozen=True)
class SpikeRule:
    """A sample further than ``threshold`` (source units) from the median of
    the ``window`` samples centred on it is a spike. Catches spikes shorter
    than half the window; a longer excursion is a level, not a spike."""

    threshold: float
    window: int = 5


@dataclass(frozen=True)
class SignalSpec:
    name: str
    kind: Kind = "analog"
    required: bool = True
    unit: str | None = None
    tolerance: float | None = None  # s; None: the dataset default
    min_coverage: float | None = None  # None: the dataset default
    range: tuple[float, float] | None = None
    invalid_values: tuple[float, ...] = ()
    spike: SpikeRule | None = None
    stuck: float | None = None  # s
    exclude: tuple[Window, ...] = ()

    def to_si(self) -> tuple[float, float]:
        """``(factor, offset)`` with ``si = factor * value + offset``."""
        if self.unit is None or self.kind != "analog":
            return 1.0, 0.0
        factor, offset, _ = si_conversion(self.unit)
        return factor, offset


@dataclass(frozen=True)
class DatasetConfig:
    step: float = 300.0  # s, §6.1: 5 min for identification
    min_segment: float = 3600.0  # s; shorter usable runs are not segments
    min_coverage: float = 1.0  # fraction of a step a signal must cover
    tolerance: float = 900.0  # s; a sample stays valid this long at most
    exclude: tuple[Window, ...] = ()
    signals: dict[str, SignalSpec] = field(default_factory=dict)

    def tolerance_of(self, name: str) -> float:
        t = self.signals[name].tolerance
        return self.tolerance if t is None else t

    def coverage_of(self, name: str) -> float:
        c = self.signals[name].min_coverage
        return self.min_coverage if c is None else c

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(n for n, s in self.signals.items() if s.required)


Problems = list[tuple[str, str]]  # (dotted key, message)


def parse_dataset_config(table: Any, prefix: str = "dataset") -> tuple[DatasetConfig | None, Problems]:
    """Parse a ``[dataset]`` table. Returns the config (None if invalid) and
    every problem found, each as ``(dotted key, message)``."""
    p: Problems = []
    if not isinstance(table, dict):
        return None, [(prefix, "must be a table")]
    for k in sorted(set(table) - set(_DATASET_KEYS)):
        p.append((f"{prefix}.{k}", f"unknown key {k!r}; expected one of {', '.join(_DATASET_KEYS)}"))

    step = _duration(table, "step", "5min", prefix, p, positive=True)
    min_segment = _duration(table, "min_segment", "1h", prefix, p, positive=False)
    tolerance = _duration(table, "tolerance", "15min", prefix, p, positive=True)
    min_coverage = _fraction(table, "min_coverage", 1.0, prefix, p)

    exclude = _windows(table.get("exclude", []), f"{prefix}.exclude", p)

    signals: dict[str, SignalSpec] = {}
    raw_signals = table.get("signal", {})
    if not isinstance(raw_signals, dict):
        p.append((f"{prefix}.signal", "must be a table of signal tables ([dataset.signal.<name>])"))
        raw_signals = {}
    for name, spec in raw_signals.items():
        s = _signal(name, spec, f"{prefix}.signal.{name}", p)
        if s is not None:
            signals[name] = s
    if not raw_signals:
        p.append((f"{prefix}.signal", "no signals: declare at least one [dataset.signal.<name>]"))
    elif not any(s.required for s in signals.values()) and len(signals) == len(raw_signals):
        p.append((f"{prefix}.signal", "no required signal: segments need at least one"))

    if p:
        return None, p
    return (
        DatasetConfig(
            step=step,
            min_segment=min_segment,
            min_coverage=min_coverage,
            tolerance=tolerance,
            exclude=tuple(exclude),
            signals=signals,
        ),
        p,
    )


# -- helpers -----------------------------------------------------------------


def _duration(t: dict, key: str, default: str, prefix: str, p: Problems, *, positive: bool) -> float:
    try:
        v = parse_quantity(t.get(key, default), TIME)
    except UnitError as e:
        p.append((f"{prefix}.{key}", str(e)))
        return math.nan
    if not math.isfinite(v) or v < 0 or (positive and v == 0):
        p.append((f"{prefix}.{key}", f"must be a {'positive' if positive else 'non-negative'} duration"))
    return v


def _fraction(t: dict, key: str, default: float, prefix: str, p: Problems) -> float:
    v = t.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int | float) or not 0 < v <= 1:
        p.append((f"{prefix}.{key}", f"must be a number in (0, 1], got {v!r}"))
        return math.nan
    return float(v)


def _number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _instant(v: Any, key: str, p: Problems) -> dt.datetime | None:
    if isinstance(v, str):
        try:
            v = dt.datetime.fromisoformat(v)
        except ValueError:
            p.append((key, f"cannot parse {v!r} as a date-time"))
            return None
    if not isinstance(v, dt.datetime):
        p.append((key, f"must be a date-time with a UTC offset, got {v!r}"))
        return None
    if v.tzinfo is None:
        p.append((key, f"{v.isoformat()} has no UTC offset; write e.g. {v.isoformat()}Z"))
        return None
    return v.astimezone(dt.UTC)


def _windows(v: Any, prefix: str, p: Problems) -> list[Window]:
    if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
        p.append((prefix, "must be an array of tables ([[...exclude]])"))
        return []
    out = []
    for i, w in enumerate(v, 1):
        key = f"{prefix}[{i}]"
        for k in sorted(set(w) - set(_WINDOW_KEYS)):
            p.append((f"{key}.{k}", f"unknown key {k!r}; expected {', '.join(_WINDOW_KEYS)}"))
        start = _instant(w.get("start"), f"{key}.start", p)
        if ("end" in w) == ("duration" in w):
            p.append((key, "give exactly one of 'end' and 'duration'"))
            continue
        if "end" in w:
            end = _instant(w["end"], f"{key}.end", p)
        else:
            try:
                d = parse_quantity(w["duration"], TIME)
            except UnitError as e:
                p.append((f"{key}.duration", str(e)))
                continue
            end = start + dt.timedelta(seconds=d) if start is not None else None
        reason = w.get("reason", "")
        if not isinstance(reason, str):
            p.append((f"{key}.reason", "must be a string"))
            continue
        if start is None or end is None:
            continue
        if not end > start:
            p.append((key, f"ends ({end.isoformat()}) before it starts ({start.isoformat()})"))
            continue
        out.append(Window(start, end, reason))
    return out


def _signal(name: str, spec: Any, key: str, p: Problems) -> SignalSpec | None:
    n0 = len(p)
    if not isinstance(spec, dict):
        p.append((key, "must be a table"))
        return None
    for k in sorted(set(spec) - set(_SIGNAL_KEYS)):
        p.append((f"{key}.{k}", f"unknown key {k!r}; expected one of {', '.join(_SIGNAL_KEYS)}"))
    kw: dict[str, Any] = {"name": name}

    kind = spec.get("kind", "analog")
    if kind not in KINDS:
        p.append((f"{key}.kind", f"must be one of {', '.join(KINDS)}; got {kind!r}"))
    kw["kind"] = kind

    req = spec.get("required", True)
    if not isinstance(req, bool):
        p.append((f"{key}.required", "must be true or false"))
    kw["required"] = req

    if "unit" in spec:
        u = spec["unit"]
        try:
            if not isinstance(u, str):
                raise UnitError(f"must be a unit string, got {u!r}")
            si_conversion(u)
        except UnitError as e:
            p.append((f"{key}.unit", str(e)))
        if kind != "analog":
            p.append((f"{key}.unit", f"only analog signals carry a unit; {name!r} is {kind}"))
        kw["unit"] = u

    if "tolerance" in spec:
        kw["tolerance"] = _duration(spec, "tolerance", "", key, p, positive=True)
    if "min_coverage" in spec:
        kw["min_coverage"] = _fraction(spec, "min_coverage", 1.0, key, p)

    if kind == "categorical" and any(k in spec for k in ("range", "invalid_values", "spike")):
        p.append((key, "range, invalid_values and spike apply to numeric signals, not categorical ones"))

    if "range" in spec:
        r = spec["range"]
        if not (isinstance(r, list) and len(r) == 2 and all(_number(x) for x in r) and r[0] < r[1]):
            p.append((f"{key}.range", f"must be [low, high] with low < high, got {r!r}"))
        else:
            kw["range"] = (float(r[0]), float(r[1]))

    if "invalid_values" in spec:
        iv = spec["invalid_values"]
        if not (isinstance(iv, list) and all(_number(x) for x in iv)):
            p.append((f"{key}.invalid_values", f"must be an array of numbers, got {iv!r}"))
        else:
            kw["invalid_values"] = tuple(float(x) for x in iv)

    if "spike" in spec:
        sp = spec["spike"]
        if not isinstance(sp, dict):
            p.append((f"{key}.spike", "must be a table { threshold = ..., window = ... }"))
        else:
            for k in sorted(set(sp) - set(_SPIKE_KEYS)):
                p.append((f"{key}.spike.{k}", f"unknown key {k!r}; expected {', '.join(_SPIKE_KEYS)}"))
            th, win = sp.get("threshold"), sp.get("window", 5)
            if not (_number(th) and th > 0):
                p.append((f"{key}.spike.threshold", f"must be a positive number, got {th!r}"))
            elif not (isinstance(win, int) and not isinstance(win, bool) and win >= 3 and win % 2 == 1):
                p.append((f"{key}.spike.window", f"must be an odd integer ≥ 3, got {win!r}"))
            else:
                kw["spike"] = SpikeRule(float(th), win)

    if "stuck" in spec:
        kw["stuck"] = _duration(spec, "stuck", "", key, p, positive=True)
        if kind == "binary":
            p.append((f"{key}.stuck", "a binary signal holding its value is normal; stuck does not apply"))

    if "exclude" in spec:
        kw["exclude"] = tuple(_windows(spec["exclude"], f"{key}.exclude", p))

    return SignalSpec(**kw) if len(p) == n0 else None
