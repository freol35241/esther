from app import dynamic_model
from app.heat_sources import IVT490

import math
import pytest


def test_simulate_steady_state():
    model = IVT490.create_model_from_slope(4.7)

    assert dynamic_model.simulate(model, 20, [20], [20], [3600]) == 20


def test_simulate_cooldown():
    model = IVT490.create_model_from_slope(4.7)

    assert dynamic_model.simulate(model, 20, [20], [0], [3600]) < 20


def test_simulate_heatup():
    model = IVT490.create_model_from_slope(4.7)

    assert dynamic_model.simulate(model, 20, [40], [20], [3600]).item() > 20


def test_simulate_timeconstant():
    model = dynamic_model.ModelParameters(
        k1=dynamic_model.DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT, k2=0
    )

    out = dynamic_model.simulate(
        model, 20, [20], [0], [1 / dynamic_model.DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT]
    )

    # Should have declined by about 63.2% during this time (i.e. one timeconstant)
    assert out.item() == pytest.approx(20 * math.exp(-1))
