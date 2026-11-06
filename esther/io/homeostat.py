"""Reader for a homeostat history store (optional extra: ``esther[homeostat]``).

A homeostat history store is SQLite, possibly split over several files — a
live database plus sealed monthly archives — with, in each file::

    series(id, class, entity, aspect, source, row_count, oldest_ts, newest_ts)
    samples(series_id, ts, room_id, kind, value)     -- ts in µs since the epoch, UTC
    rooms(id, name)

``kind`` is 0 for a boolean, 1 for a number, 2 for a string. Series ids are
local to each file, so files are joined on ``(class, entity, aspect, source)``.

Files are only ever opened read-only. ``mode=ro`` takes SQLite's shared lock
like any reader, which is correct for a file another process is writing but
needs a writable directory when the file has a journal; ``immutable=True``
takes no locks and assumes the file does not change — right for a sealed
archive, and for a live file only when the caller accepts that a write
landing mid-read can surface as an error or a torn read. Every read opens a
fresh connection, so an immutable reader never mixes cached pages with new
ones across reads.

An *address* names one series: ``"entity/aspect"`` for the ``state`` class,
``"class:entity/aspect"`` for another, with ``"@source"`` appended for a
sourced series.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

__all__ = ["Address", "HomeostatError", "HomeostatHistory"]

TimeLike = dt.datetime | pd.Timestamp | str


class HomeostatError(RuntimeError):
    """A history file could not be read."""


@dataclass(frozen=True)
class Address:
    entity: str
    aspect: str
    cls: str = "state"
    source: str = ""

    @classmethod
    def parse(cls, text: str) -> Address:
        body, _, source = text.partition("@")
        klass, sep, rest = body.partition(":")
        if not sep:
            klass, rest = "state", body
        entity, sep, aspect = rest.partition("/")
        if not (sep and entity and aspect and klass):
            raise ValueError(f"bad address {text!r}: expected [class:]entity/aspect[@source]")
        return cls(entity, aspect, klass, source)

    def __str__(self) -> str:
        s = f"{self.entity}/{self.aspect}"
        if self.cls != "state":
            s = f"{self.cls}:{s}"
        return f"{s}@{self.source}" if self.source else s


def _us(x: TimeLike | None) -> int | None:
    if x is None:
        return None
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        raise ValueError(f"{x!r} has no time zone; give a UTC offset")
    return ts.value // 1000


class HomeostatHistory:
    """Merged, read-only view over one or more history files.

    ``paths`` are read in the order given; where two files hold a sample for
    the same series and timestamp, the later file wins. Give archives first
    and the live file last (:meth:`open_directory` does).
    """

    def __init__(self, paths: Iterable[str | Path], *, immutable: bool = False) -> None:
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("no history files given")
        for p in self.paths:
            if not p.is_file():
                raise FileNotFoundError(p)
        self.immutable = immutable

    @classmethod
    def open_directory(
        cls,
        directory: str | Path,
        *,
        live: str = "history.db",
        archives: str = "archive/history-*.db",
        immutable: bool = False,
    ) -> HomeostatHistory:
        """Archives matching ``archives`` (sorted by name), then ``live`` if present."""
        d = Path(directory)
        paths = sorted(d.glob(archives))
        if (d / live).is_file():
            paths.append(d / live)
        return cls(paths, immutable=immutable)

    # -- connections -------------------------------------------------------
    def _connect(self, path: Path) -> sqlite3.Connection:
        flag = "immutable=1" if self.immutable else "mode=ro"
        uri = f"file:{quote(str(path.resolve()))}?{flag}"
        try:
            con = sqlite3.connect(uri, uri=True)
            con.execute("select 1 from series limit 1").fetchall()
        except sqlite3.OperationalError as e:
            hint = ""
            if "readonly" in str(e) and not self.immutable:
                hint = (
                    " — SQLite needs to write next to the file to read it in mode=ro (a journal in"
                    " a read-only directory); open with immutable=True if a torn read is acceptable"
                )
            raise HomeostatError(f"{path}: {e}{hint}") from e
        return con

    # -- catalogue ---------------------------------------------------------
    def catalogue(self) -> pd.DataFrame:
        """Every series across the files: class, entity, aspect, source, rows,
        oldest and newest timestamp (UTC), and how many files hold it."""
        frames = []
        for p in self.paths:
            with closing(self._connect(p)) as con:
                rows = con.execute(
                    "select class, entity, aspect, source, row_count, oldest_ts, newest_ts from series"
                ).fetchall()
            frames.append(
                pd.DataFrame(
                    rows, columns=["class", "entity", "aspect", "source", "rows", "oldest", "newest"]
                )
            )
        df = pd.concat(frames, ignore_index=True)
        key = ["class", "entity", "aspect", "source"]
        out = df.groupby(key, sort=True).agg(
            rows=("rows", "sum"), oldest=("oldest", "min"), newest=("newest", "max"), files=("rows", "size")
        )
        for c in ("oldest", "newest"):
            out[c] = pd.to_datetime(out[c], unit="us", utc=True)
        out = out.reset_index()
        out.insert(
            0, "address", [str(Address(e, a, c, s)) for c, e, a, s in out[key].itertuples(index=False)]
        )
        return out

    # -- reading -----------------------------------------------------------
    def read(
        self, address: str | Address, start: TimeLike | None = None, end: TimeLike | None = None
    ) -> pd.Series:
        """Samples of one series in ``[start, end)``, merged over the files.

        Index: UTC timestamps. Values: float for numbers, bool for booleans,
        object (str) for strings; a series mixing kinds comes back as object.
        An address no file knows raises :class:`KeyError`.
        """
        a = address if isinstance(address, Address) else Address.parse(address)
        t0, t1 = _us(start), _us(end)
        ts_parts, val_parts, kind_parts = [], [], []
        found = False
        for p in self.paths:
            with closing(self._connect(p)) as con:
                row = con.execute(
                    "select id from series where class=? and entity=? and aspect=? and source=?",
                    (a.cls, a.entity, a.aspect, a.source),
                ).fetchone()
                if row is None:
                    continue
                found = True
                q = "select ts, kind, value from samples where series_id=?"
                args: list[object] = [row[0]]
                if t0 is not None:
                    q += " and ts>=?"
                    args.append(t0)
                if t1 is not None:
                    q += " and ts<?"
                    args.append(t1)
                try:
                    rows = con.execute(q + " order by ts", args).fetchall()
                except sqlite3.DatabaseError as e:
                    raise HomeostatError(f"{p}: reading {a}: {e}") from e
            if rows:
                ts, kind, val = zip(*rows, strict=True)
                ts_parts.append(np.asarray(ts, dtype=np.int64))
                kind_parts.append(np.asarray(kind, dtype=np.int8))
                val_parts.append(np.asarray(val, dtype=object))
        if not found:
            raise KeyError(f"no series {a} in {', '.join(p.name for p in self.paths)}")
        name = str(a)
        if not ts_parts:
            return pd.Series([], index=pd.DatetimeIndex([], tz="UTC", name="time"), dtype=float, name=name)
        ts = np.concatenate(ts_parts)
        kind = np.concatenate(kind_parts)
        val = np.concatenate(val_parts)
        # Stable sort, then keep the last of each timestamp: the later file wins.
        order = np.argsort(ts, kind="stable")
        ts, kind, val = ts[order], kind[order], val[order]
        last = np.r_[ts[1:] != ts[:-1], True]
        ts, kind, val = ts[last], kind[last], val[last]
        index = pd.DatetimeIndex(pd.to_datetime(ts, unit="us", utc=True), name="time")
        kinds = set(np.unique(kind).tolist())
        if kinds == {1}:
            data = pd.Series(val.astype(float), index=index, name=name)
        elif kinds == {0}:
            data = pd.Series(val.astype(float) != 0, index=index, name=name)
        else:
            data = pd.Series(val, index=index, name=name, dtype=object)
        return data

    def read_many(
        self, addresses: Sequence[str], start: TimeLike | None = None, end: TimeLike | None = None
    ) -> dict[str, pd.Series]:
        return {a: self.read(a, start, end) for a in addresses}
