"""A Plan as JSON, and back (spec §9.3), so that a restarted planner warm-starts.

A plan restored from its JSON is the plan: every field round-trips exactly —
floats through their shortest repr (exact for IEEE doubles), NaN and the
infinities as tagged strings so the text stays strict JSON, timestamps as
nanoseconds since the epoch with their time zone, arrays with their dtype and
shape, DataFrames column by column with each column's dtype. The warm-start
store (``Plan.warm``: step-start times in ns and the integer decisions' values)
is carried too, so that ``previous=`` a restored plan gives the next solve the
same warm start as the plan in memory (M4: a cold start under a short budget
can return a poor incumbent).

Anything the encoder does not know is refused rather than stringified: a
round trip that silently changed a type would not be one.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd

from ..forecast.base import Event, Provenance
from .planner import Plan

__all__ = ["plan_from_json", "plan_to_json"]

FORMAT = "esther.plan/1"
_SPECIAL = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _float(x: float) -> float | dict[str, str]:
    if math.isfinite(x):
        return x
    return {"$float": "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")}


def _array_data(a: np.ndarray) -> list:
    if a.dtype.kind == "f":
        return [_float(float(v)) for v in a.reshape(-1)]
    if a.dtype.kind in "iub":
        return a.reshape(-1).tolist()
    if a.dtype.kind in "OU":
        return [_enc(v) for v in a.reshape(-1)]
    raise TypeError(f"cannot serialise an array of dtype {a.dtype}")


def _ts(t: pd.Timestamp) -> dict[str, Any]:
    return {"$ts": int(t.as_unit("ns").value), "tz": None if t.tz is None else str(t.tz)}


def _index(idx: pd.DatetimeIndex) -> dict[str, Any]:
    return {"$index": idx.as_unit("ns").asi8.tolist(), "tz": None if idx.tz is None else str(idx.tz)}


def _enc(v: Any) -> Any:
    """A value as JSON, tagged where JSON alone would lose its type."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, np.bool_):
        return {"$np": "bool", "v": bool(v)}
    if isinstance(v, np.integer):
        return {"$np": str(v.dtype), "v": int(v)}
    if isinstance(v, np.floating):
        return {"$np": str(v.dtype), "v": _float(float(v))}
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return _float(v)
    if isinstance(v, pd.Timestamp):
        return _ts(v)
    if isinstance(v, pd.Timedelta):
        return {"$td": int(v.as_unit("ns").value)}
    if isinstance(v, pd.DatetimeIndex):
        return _index(v)
    if isinstance(v, np.ndarray):
        return {"$array": _array_data(v), "dtype": v.dtype.str, "shape": list(v.shape)}
    if isinstance(v, pd.DataFrame):
        return _frame(v)
    if isinstance(v, Provenance):
        return {"$provenance": {f.name: _enc(getattr(v, f.name)) for f in fields(v)}}
    if isinstance(v, Event):
        return {"$event": {f.name: _enc(getattr(v, f.name)) for f in fields(v)}}
    if isinstance(v, tuple):
        return {"$tuple": [_enc(x) for x in v]}
    if isinstance(v, list):
        return [_enc(x) for x in v]
    if isinstance(v, dict):
        bad = [k for k in v if not isinstance(k, str)]
        if bad:
            raise TypeError(f"cannot serialise a dict with non-string keys: {bad[:3]}")
        if any(k.startswith("$") for k in v):
            return {"$dict": {k: _enc(x) for k, x in v.items()}}
        return {k: _enc(x) for k, x in v.items()}
    raise TypeError(f"cannot serialise {type(v).__name__} in a plan")


def _frame(df: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("a plan's frames are indexed by time")
    cols = [str(c) for c in df.columns]
    if cols != list(df.columns) or len(set(cols)) != len(cols):
        raise TypeError("a plan's frames need unique string column names")
    return {
        "$frame": {
            "index": _index(df.index),
            "columns": cols,
            "data": [_enc(df[c].to_numpy()) for c in cols],
        }
    }


def _dec_float(x: Any) -> float:
    if isinstance(x, dict):
        return _SPECIAL[x["$float"]]
    return float(x)


def _dec_ts(d: dict) -> pd.Timestamp:
    t = pd.Timestamp(d["$ts"], unit="ns", tz="UTC" if d["tz"] is not None else None)
    return t.tz_convert(d["tz"]) if d["tz"] is not None else t


def _dec_index(d: dict) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(np.asarray(d["$index"], dtype="int64").view("M8[ns]"))
    return idx.tz_localize("UTC").tz_convert(d["tz"]) if d["tz"] is not None else idx


def _dec(v: Any) -> Any:
    if isinstance(v, list):
        return [_dec(x) for x in v]
    if not isinstance(v, dict):
        return v
    if "$float" in v:
        return _dec_float(v)
    if "$np" in v:
        if v["$np"] == "bool":
            return np.bool_(v["v"])
        t = np.dtype(v["$np"])
        return t.type(_dec_float(v["v"]) if t.kind == "f" else v["v"])
    if "$ts" in v:
        return _dec_ts(v)
    if "$td" in v:
        return pd.Timedelta(v["$td"], unit="ns")
    if "$index" in v:
        return _dec_index(v)
    if "$array" in v:
        dt = np.dtype(v["dtype"])
        data = v["$array"]
        if dt.kind == "f":
            flat = np.array([_dec_float(x) for x in data], dtype=dt)
        elif dt.kind in "OU":
            flat = np.array([_dec(x) for x in data], dtype=dt)
        else:
            flat = np.array(data, dtype=dt)
        return flat.reshape(v["shape"])
    if "$frame" in v:
        f = v["$frame"]
        idx = _dec_index(f["index"])
        cols = {c: _dec(d) for c, d in zip(f["columns"], f["data"], strict=True)}
        return pd.DataFrame(cols, index=idx, columns=f["columns"])
    if "$provenance" in v:
        return Provenance(**{k: _dec(x) for k, x in v["$provenance"].items()})
    if "$event" in v:
        return Event(**{k: _dec(x) for k, x in v["$event"].items()})
    if "$tuple" in v:
        return tuple(_dec(x) for x in v["$tuple"])
    if "$dict" in v:
        return {k: _dec(x) for k, x in v["$dict"].items()}
    return {k: _dec(x) for k, x in v.items()}


def plan_to_json(plan: Plan, indent: int | None = None) -> str:
    """The plan as strict JSON (no bare NaN or Infinity)."""
    doc = {"format": FORMAT}
    for f in fields(plan):
        v = getattr(plan, f.name)
        if f.name == "warm":
            v = {k: {"t": np.asarray(t), "x": np.asarray(x)} for k, (t, x) in v.items()}
        doc[f.name] = _enc(v)
    return json.dumps(doc, indent=indent, allow_nan=False)


def plan_from_json(text: str) -> Plan:
    doc = json.loads(text)
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        got = doc.get("format") if isinstance(doc, dict) else type(doc).__name__
        raise ValueError(f"not an esther plan (format {got!r}, expected {FORMAT!r})")
    kw = {}
    for f in fields(Plan):
        if f.name not in doc:
            raise ValueError(f"plan JSON lacks {f.name!r}")
        v = _dec(doc[f.name])
        if f.name == "warm":
            v = {k: (w["t"], w["x"]) for k, w in v.items()}
        kw[f.name] = v
    return Plan(**kw)
