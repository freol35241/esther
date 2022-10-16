from app.heat_sources import IVT490

from dataclasses import asdict


def test_create_from_slope(pinned):
    assert asdict(IVT490.create_model_from_slope(4.7)) == pinned.approx()


def test_make_initial_guess(pinned):
    assert IVT490.make_initial_guess(4.7, [10, 20, 0, -10]) == pinned.approx()
