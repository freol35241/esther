from app.utils import T_eff


def test_T_eff(pinned):
    assert T_eff(0, 10) == pinned.approx()
