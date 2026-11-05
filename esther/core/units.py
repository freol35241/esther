"""Quantities with units, parsed from configuration strings into SI (spec N4).

Internal units are SI: W, J, K, s, kg (and m³ for volumes). Configuration
writes values as strings with a unit — ``"8 MJ/K"``, ``"55 l"``, ``"15min"``,
``"0.15 kg/h"``. :func:`parse_quantity` turns such a string into an SI float and
checks it has the dimension the caller expects.

Temperatures are the one affine case: ``"21 °C"`` is an absolute temperature
and becomes 294.15 K; ``"5 K"`` is a difference and stays 5. A Celsius value is
only accepted where an absolute temperature is asked for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "CONDUCTANCE",
    "DIMENSIONLESS",
    "ENERGY",
    "ENERGY_PER_MASS",
    "HEAT_CAPACITY",
    "MASS",
    "MASS_FLOW",
    "POWER",
    "PRICE_PER_ENERGY",
    "PRICE_PER_MASS",
    "TEMPERATURE",
    "TIME",
    "VOLUME",
    "VOLUME_FLOW",
    "Dimension",
    "UnitError",
    "parse_quantity",
]

CELSIUS_OFFSET = 273.15


class UnitError(ValueError):
    """A unit string could not be parsed, or has the wrong dimension."""


@dataclass(frozen=True)
class Dimension:
    """Exponents of the base dimensions (kg, m, s, K, currency)."""

    kg: int = 0
    m: int = 0
    s: int = 0
    K: int = 0
    cur: int = 0

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(*(a + b for a, b in zip(self._t(), other._t(), strict=True)))

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(*(a - b for a, b in zip(self._t(), other._t(), strict=True)))

    def _t(self) -> tuple[int, ...]:
        return (self.kg, self.m, self.s, self.K, self.cur)

    def __str__(self) -> str:
        return _DIMENSION_NAMES.get(self, _format_exponents(self))


def _format_exponents(d: Dimension) -> str:
    parts = [
        f"{name}^{exp}" if exp != 1 else name
        for name, exp in zip(("kg", "m", "s", "K", "currency"), d._t(), strict=True)
        if exp
    ]
    return " ".join(parts) or "dimensionless"


DIMENSIONLESS = Dimension()
MASS = Dimension(kg=1)
LENGTH = Dimension(m=1)
TIME = Dimension(s=1)
TEMPERATURE = Dimension(K=1)
CURRENCY = Dimension(cur=1)
VOLUME = Dimension(m=3)
ENERGY = Dimension(kg=1, m=2, s=-2)
POWER = Dimension(kg=1, m=2, s=-3)
HEAT_CAPACITY = ENERGY / TEMPERATURE
CONDUCTANCE = POWER / TEMPERATURE
MASS_FLOW = MASS / TIME
VOLUME_FLOW = VOLUME / TIME
ENERGY_PER_MASS = ENERGY / MASS
PRICE_PER_ENERGY = CURRENCY / ENERGY
PRICE_PER_MASS = CURRENCY / MASS

_DIMENSION_NAMES = {
    DIMENSIONLESS: "dimensionless",
    MASS: "mass",
    TIME: "time",
    TEMPERATURE: "temperature",
    VOLUME: "volume",
    ENERGY: "energy",
    POWER: "power",
    HEAT_CAPACITY: "heat capacity",
    CONDUCTANCE: "conductance",
    MASS_FLOW: "mass flow",
    VOLUME_FLOW: "volume flow",
    ENERGY_PER_MASS: "energy per mass",
    PRICE_PER_ENERGY: "price per energy",
    PRICE_PER_MASS: "price per mass",
}

# Atomic units: symbol -> (factor to SI, dimension).
_ATOMS: dict[str, tuple[float, Dimension]] = {
    "1": (1.0, DIMENSIONLESS),
    "%": (0.01, DIMENSIONLESS),
    "kg": (1.0, MASS),
    "g": (1e-3, MASS),
    "t": (1e3, MASS),
    "m": (1.0, LENGTH),
    "m3": (1.0, VOLUME),
    "m³": (1.0, VOLUME),
    "l": (1e-3, VOLUME),
    "L": (1e-3, VOLUME),
    "s": (1.0, TIME),
    "min": (60.0, TIME),
    "h": (3600.0, TIME),
    "d": (86400.0, TIME),
    "K": (1.0, TEMPERATURE),
    "W": (1.0, POWER),
    "kW": (1e3, POWER),
    "MW": (1e6, POWER),
    "J": (1.0, ENERGY),
    "kJ": (1e3, ENERGY),
    "MJ": (1e6, ENERGY),
    "GJ": (1e9, ENERGY),
    "Wh": (3600.0, ENERGY),
    "kWh": (3.6e6, ENERGY),
    "MWh": (3.6e9, ENERGY),
    # Currency is a dimension of its own with no fixed unit: esther does not
    # convert between currencies, so every price in a house must use the same
    # one. "SEK", "EUR", ... all map to the same base.
    "cur": (1.0, CURRENCY),
    "SEK": (1.0, CURRENCY),
    "EUR": (1.0, CURRENCY),
    "NOK": (1.0, CURRENCY),
    "DKK": (1.0, CURRENCY),
    "USD": (1.0, CURRENCY),
}

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_QUANTITY_RE = re.compile(rf"^\s*(?P<num>{_NUMBER})\s*(?P<unit>.*?)\s*$")
_CELSIUS = ("°C", "degC", "C")


def _parse_unit(expr: str) -> tuple[float, Dimension]:
    """Parse ``a·b/c`` style unit expressions; ``^n`` exponents allowed."""
    if not expr:
        return 1.0, DIMENSIONLESS
    factor, dim = 1.0, DIMENSIONLESS
    # Split on '/' keeping track of which side each atom is on; everything
    # after the first '/' is in the denominator (so "J/kg/K" = J/(kg K)).
    tokens = re.split(r"(/|\*|·|\s+)", expr)
    sign = 1
    for tok in tokens:
        if tok is None or tok.strip() == "" or tok in ("*", "·"):
            continue
        if tok == "/":
            sign = -1
            continue
        base, _, exp_s = tok.partition("^")
        try:
            exp = int(exp_s) if exp_s else 1
        except ValueError as e:
            raise UnitError(f"bad exponent in unit {expr!r}") from e
        if base not in _ATOMS:
            raise UnitError(f"unknown unit {base!r} in {expr!r}")
        f, d = _ATOMS[base]
        for _ in range(exp):
            if sign > 0:
                factor, dim = factor * f, dim * d
            else:
                factor, dim = factor / f, dim / d
    return factor, dim


def parse_quantity(
    value: str | int | float,
    expect: Dimension,
    *,
    absolute_temperature: bool = False,
) -> float:
    """Return ``value`` in SI units, checking its dimension against ``expect``.

    A bare number is taken to be in SI already (the "default unit for that kind
    of quantity", spec §5) — except for absolute temperatures, where a bare
    number is °C, the unit configuration is written in.
    """
    if isinstance(value, bool):
        raise UnitError(f"expected a {expect} quantity, got a boolean")
    if isinstance(value, int | float):
        if absolute_temperature and expect == TEMPERATURE:
            return float(value) + CELSIUS_OFFSET
        return float(value)
    if not isinstance(value, str):
        raise UnitError(f"expected a {expect} quantity, got {type(value).__name__}")
    m = _QUANTITY_RE.match(value)
    if not m:
        raise UnitError(f"cannot parse quantity {value!r}")
    number = float(m["num"])
    unit = m["unit"]
    if unit in _CELSIUS:
        if not (absolute_temperature and expect == TEMPERATURE):
            raise UnitError(
                f"{value!r} is an absolute temperature in °C; "
                f"expected {expect}" + (" (write differences in K)" if expect == TEMPERATURE else "")
            )
        return number + CELSIUS_OFFSET
    factor, dim = _parse_unit(unit)
    if dim != expect:
        raise UnitError(f"{value!r} has dimension {dim}, expected {expect}")
    return number * factor
