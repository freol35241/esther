from app import dynamic_model

import math
import pytest
from dataclasses import asdict


def test_create_from_IVT490_settings(pinned):
    assert (
        asdict(dynamic_model.create_model_from_IVT490_settings(4.7)) == pinned.approx()
    )


def test_simulate_steady_state():
    model = dynamic_model.create_model_from_IVT490_settings(4.7)

    assert dynamic_model.simulate(model, 20, [20], [20], [3600]) == 20


def test_simulate_cooldown():
    model = dynamic_model.create_model_from_IVT490_settings(4.7)

    assert dynamic_model.simulate(model, 20, [20], [0], [3600]) < 20


def test_simulate_heatup():
    model = dynamic_model.create_model_from_IVT490_settings(4.7)

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
