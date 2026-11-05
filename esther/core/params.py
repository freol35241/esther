"""Parameters and their priors (spec §4.2).

A component declares each parameter it has with a unit, a weak default prior
and bounds. The house description may replace the prior or pin the value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .units import Dimension, UnitError, parse_quantity

__all__ = ["LogNormal", "Normal", "Param", "Prior", "parse_prior"]


@dataclass(frozen=True)
class LogNormal:
    """Log-normal prior: ``median`` in SI, geometric standard deviation ``gsd`` > 1."""

    median: float
    gsd: float

    def check(self) -> list[str]:
        errs = []
        if not self.median > 0:
            errs.append(f"log-normal median must be > 0, got {self.median:g}")
        if not self.gsd > 1:
            errs.append(f"gsd must be > 1 (1 would be a pinned value; use fixed = true), got {self.gsd:g}")
        return errs

    @property
    def centre(self) -> float:
        return self.median


@dataclass(frozen=True)
class Normal:
    """Normal prior, for quantities that may be zero or negative (a sensor bias)."""

    mean: float
    sd: float

    def check(self) -> list[str]:
        return [] if self.sd > 0 else [f"sd must be > 0, got {self.sd:g}"]

    @property
    def centre(self) -> float:
        return self.mean


Prior = LogNormal | Normal


@dataclass(frozen=True)
class Param:
    """A model parameter: ``Param("UA", unit=CONDUCTANCE, prior=LogNormal(150, 2.0), bounds=(10, 2000))``.

    ``fixed`` holds the pinned value when the house pins it; then it is not fitted.
    Positive physical quantities (log-normal prior) are fitted in log space.
    """

    name: str
    unit: Dimension
    prior: Prior
    bounds: tuple[float, float]
    fixed: float | None = None
    description: str = ""

    @property
    def log_transform(self) -> bool:
        return isinstance(self.prior, LogNormal)

    def check(self) -> list[str]:
        lo, hi = self.bounds
        errs = list(self.prior.check())
        if not lo < hi:
            errs.append(f"bounds must satisfy lower < upper, got ({lo:g}, {hi:g})")
        if self.log_transform and lo <= 0:
            errs.append(f"a log-normal parameter needs a positive lower bound, got {lo:g}")
        if self.fixed is not None:
            if not lo <= self.fixed <= hi:
                errs.append(f"fixed value {self.fixed:g} is outside the bounds ({lo:g}, {hi:g})")
        elif not lo < self.prior.centre < hi:
            errs.append(f"prior centre {self.prior.centre:g} is outside the bounds ({lo:g}, {hi:g})")
        return errs


_PRIOR_KEYS = {
    frozenset({"median", "gsd"}),
    frozenset({"mean", "sd"}),
    frozenset({"value", "fixed"}),
}


def parse_prior(param: Param, spec: object) -> Param:
    """Apply a house-file prior table to a declared parameter.

    Accepted forms::

        prior.C  = { median = "8 MJ/K", gsd = 2.0 }     # log-normal
        prior.b  = { mean = "0 K", sd = "0.5 K" }        # normal
        prior.V  = { value = "150 l", fixed = true }     # pinned
        prior.UA = { median = "120 W/K", gsd = 2.0, bounds = ["10 W/K", "1000 W/K"] }

    Raises ``ValueError`` with a message naming the problem.
    """
    if not isinstance(spec, dict):
        raise ValueError("a prior must be a table, e.g. { median = ..., gsd = ... }")
    keys = set(spec)
    bounds_spec = spec.get("bounds")
    keys.discard("bounds")
    if frozenset(keys) not in _PRIOR_KEYS:
        raise ValueError(f"prior keys {sorted(keys)} not understood; use median+gsd, mean+sd or value+fixed")

    def q(key: str) -> float:
        try:
            return parse_quantity(spec[key], param.unit)
        except UnitError as e:
            raise ValueError(f"{key}: {e}") from None

    def number(key: str) -> float:
        v = spec[key]
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ValueError(f"{key} must be a finite number, got {v!r}")
        return float(v)

    bounds = param.bounds
    if bounds_spec is not None:
        if not (isinstance(bounds_spec, list) and len(bounds_spec) == 2):
            raise ValueError("bounds must be a two-element array [lower, upper]")
        try:
            bounds = tuple(parse_quantity(b, param.unit) for b in bounds_spec)  # type: ignore[assignment]
        except UnitError as e:
            raise ValueError(f"bounds: {e}") from None

    if "median" in keys:
        new = replace(param, prior=LogNormal(q("median"), number("gsd")), bounds=bounds)
    elif "mean" in keys:
        new = replace(param, prior=Normal(q("mean"), q("sd")), bounds=bounds)
    else:
        if spec["fixed"] is not True:
            raise ValueError("fixed must be true when a value is given")
        new = replace(param, fixed=q("value"), bounds=bounds)
    errs = new.check()
    if errs:
        raise ValueError("; ".join(errs))
    return new
