import math

import pytest

from esther.core.units import (
    CONDUCTANCE,
    DIMENSIONLESS,
    ENERGY,
    HEAT_CAPACITY,
    MASS,
    MASS_FLOW,
    POWER,
    PRICE_PER_ENERGY,
    TEMPERATURE,
    TIME,
    VOLUME,
    UnitError,
    parse_quantity,
)


@pytest.mark.parametrize(
    ("text", "dim", "si"),
    [
        ("8 MJ/K", HEAT_CAPACITY, 8e6),
        ("120 W/K", CONDUCTANCE, 120.0),
        ("55 l", VOLUME, 0.055),
        ("15min", TIME, 900.0),
        ("36h", TIME, 129600.0),
        ("0.15 kg/h", MASS_FLOW, 0.15 / 3600),
        ("6.8 kW", POWER, 6800.0),
        ("2 kWh", ENERGY, 7.2e6),
        ("5 K", TEMPERATURE, 5.0),
        ("0.10 EUR/kWh", PRICE_PER_ENERGY, 0.10 / 3.6e6),
        ("50 %", DIMENSIONLESS, 0.5),
        ("1e3 W", POWER, 1000.0),
        ("4186 J/kg/K", ENERGY / MASS / TEMPERATURE, 4186.0),
    ],
)
def test_parse(text, dim, si):
    assert math.isclose(parse_quantity(text, dim), si, rel_tol=1e-12)


def test_bare_number_is_si():
    assert parse_quantity(120, CONDUCTANCE) == 120.0


def test_celsius_only_for_absolute_temperature():
    assert parse_quantity("21 °C", TEMPERATURE, absolute_temperature=True) == pytest.approx(294.15)
    assert parse_quantity(21, TEMPERATURE, absolute_temperature=True) == pytest.approx(294.15)
    with pytest.raises(UnitError, match="differences in K"):
        parse_quantity("5 °C", TEMPERATURE)


@pytest.mark.parametrize(
    ("text", "dim", "match"),
    [
        ("8 MW", HEAT_CAPACITY, "dimension power, expected heat capacity"),
        ("8 furlongs", TIME, "unknown unit"),
        ("fast", TIME, "cannot parse"),
        (True, TIME, "boolean"),
        ("21 °C", POWER, "absolute temperature"),
    ],
)
def test_refusals(text, dim, match):
    with pytest.raises(UnitError, match=match):
        parse_quantity(text, dim)
