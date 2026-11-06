"""Identification data: contiguous runs of steps with every signal a model reads.

A :class:`~esther.data.Dataset` marks the steps where its *required* signals
are valid. A model may read more (a solar proxy, a pump on-fraction), so
:meth:`IdData.from_dataset` re-segments on exactly the model's signals: a step
where any of them is missing ends the segment (§6.1: nothing is interpolated
across a gap). Every frame in an :class:`IdData` is finite in those columns
and on a regular grid.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["IdData"]


@dataclass
class IdData:
    """Segments of identification data on a regular grid of ``step`` seconds."""

    step: float
    frames: list[pd.DataFrame]

    # -- construction ------------------------------------------------------
    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        step: float,
        signals: Iterable[str],
        min_steps: int = 2,
        usable: pd.Series | None = None,
    ) -> IdData:
        """Split ``frame`` (indexed by step start) into runs where every one of
        ``signals`` is finite, the index advances by exactly ``step`` and
        ``usable`` (if given) is true. Runs shorter than ``min_steps`` are dropped."""
        cols = list(dict.fromkeys(signals))
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"no column for signal(s): {', '.join(missing)}")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError("the frame needs a tz-aware DatetimeIndex")
        sub = frame[cols].astype(float)
        ok = np.isfinite(sub.to_numpy()).all(axis=1) if cols else np.ones(len(sub), dtype=bool)
        if usable is not None:
            ok &= usable.reindex(frame.index, fill_value=False).to_numpy(dtype=bool)
        t = frame.index.as_unit("ns").asi8
        step_ns = round(step * 1e9)
        brk = np.r_[True, np.diff(t) != step_ns]
        frames = []
        start = None
        for i in range(len(sub) + 1):
            end_run = i == len(sub) or not ok[i] or (brk[i] and start is not None)
            if end_run and start is not None:
                if i - start >= min_steps:
                    frames.append(sub.iloc[start:i])
                start = None
            if i < len(sub) and ok[i] and start is None:
                start = i
        return cls(step=float(step), frames=frames)

    @classmethod
    def from_dataset(cls, ds, signals: Iterable[str], extra: pd.DataFrame | None = None) -> IdData:
        """Segments of a :class:`~esther.data.Dataset` usable for ``signals``.

        ``extra`` adds derived columns (same index) before segmenting — inputs
        the integration computes from the dataset's signals."""
        values = ds.values if extra is None else ds.values.join(extra, how="left")
        min_steps = max(2, int(np.ceil(ds.config.min_segment / ds.config.step - 1e-9)))
        return cls.from_frame(values, ds.config.step, signals, min_steps=min_steps, usable=ds.usable)

    # -- properties --------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return sum(len(f) for f in self.frames)

    @property
    def lengths(self) -> list[int]:
        return [len(f) for f in self.frames]

    @property
    def start(self) -> pd.Timestamp | None:
        return self.frames[0].index[0] if self.frames else None

    @property
    def end(self) -> pd.Timestamp | None:
        if not self.frames:
            return None
        return self.frames[-1].index[-1] + pd.Timedelta(seconds=self.step)

    def columns(self, cols: Sequence[str]) -> IdData:
        return IdData(self.step, [f[list(cols)] for f in self.frames])

    # -- splitting ---------------------------------------------------------
    def split(self, holdout: float = 0.2, min_steps: int = 2) -> tuple[IdData, IdData]:
        """``(train, test)``: the most recent ``holdout`` share of the steps,
        by time, is held out (§6.5; never random rows). A segment that
        straddles the split time is cut in two."""
        if not 0.0 < holdout < 1.0:
            raise ValueError(f"holdout must be in (0, 1), got {holdout}")
        if not self.frames:
            raise ValueError("no data to split")
        idx = pd.DatetimeIndex(np.concatenate([f.index.as_unit("ns").asi8 for f in self.frames]), tz="UTC")
        n_test = round(holdout * len(idx))
        if n_test < 1 or n_test >= len(idx):
            raise ValueError(f"cannot hold out {holdout:.0%} of {len(idx)} steps")
        t_split = idx.sort_values()[len(idx) - n_test]
        train, test = [], []
        for f in self.frames:
            a, b = f[f.index < t_split], f[f.index >= t_split]
            if len(a) >= min_steps:
                train.append(a)
            if len(b) >= min_steps:
                test.append(b)
        return IdData(self.step, train), IdData(self.step, test)

    def summary(self) -> str:
        if not self.frames:
            return "no segments"
        ls = np.array(self.lengths)
        h = self.step / 3600
        return (
            f"{len(ls)} segments, {ls.sum()} steps ({ls.sum() * h / 24:.1f} d) "
            f"from {self.start} to {self.end}; "
            f"segment length median {np.median(ls) * h:.1f} h, longest {ls.max() * h:.1f} h"
        )
