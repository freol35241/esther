"""The general heat path (core.model.Transfer) and decisions with a source:
matrices against hand-computed values, energy conservation, and the
fitting-split rule for an optional source."""

from __future__ import annotations

import numpy as np
import pytest

from esther import loads_house
from esther.components.envelope import Ambient
from esther.core.component import Component
from esther.core.model import (
    Capacity,
    Constant,
    Decision,
    Measurement,
    ModelError,
    Signal,
    Transfer,
    compose,
)


class Terms(Component):
    """A component that contributes exactly the terms it is given."""

    def __init__(self, name: str, terms: list) -> None:
        super().__init__(name, {}, {}, {})
        self._terms = terms

    def dynamics(self, p, mode):
        return self._terms

    def active_params(self, mode):
        return []


CA, CB = 2e6, 5e5


def build(*terms, mode: str = "identify"):
    base = [Capacity("a", CA), Capacity("b", CB), Measurement("y", "a", 0.1)]
    comps = {"out": Ambient("out", {"signal": "t_out"}, {}, {}), "t": Terms("t", [*base, *terms])}
    return compose(comps, mode=mode)


def test_two_sided_transfer_conserves_energy():
    m = build(Transfer(50.0, Signal("t_sup"), "a", to="a", frm="b", modulation="pump"))
    M = m.matrices([])
    a, b = m.states.index("a"), m.states.index("b")
    sup = m.inputs.index(Signal("t_sup"))
    # Q = 50 (T_sup − T_a), modulated: into a, out of b
    assert M.A[1, a, a] == pytest.approx(-50 / CA)
    assert M.E[1, a, sup] == pytest.approx(50 / CA)
    assert M.A[1, b, a] == pytest.approx(50 / CB)
    assert M.E[1, b, sup] == pytest.approx(-50 / CB)
    # watts into a equal watts out of b, for any state and input
    C = np.array([CA if s == "a" else CB for s in m.states])
    assert C @ M.A[1] == pytest.approx(np.zeros(2), abs=1e-12)
    assert C @ M.E[1] == pytest.approx(np.zeros(len(m.inputs)), abs=1e-12)
    assert np.all(M.A[0] == 0)  # nothing unmodulated


def test_fixed_temperature_enters_through_the_constant_input():
    m = build(Transfer(10.0, 323.15, "a", to="a"), Transfer(1.0, "a", "b", to="b"))
    M = m.matrices([])
    a, b = m.states.index("a"), m.states.index("b")
    one = m.inputs.index(Constant())
    assert M.E[0, a, one] == pytest.approx(10 * 323.15 / CA)
    assert M.A[0, a, a] == pytest.approx(-10 / CA)
    # one-sided between nodes: b is pulled towards a, a feels nothing
    assert M.A[0, b, a] == pytest.approx(1 / CB) and M.A[0, b, b] == pytest.approx(-1 / CB)
    assert M.A[0, a, b] == 0


def test_boundary_temperature_resolves_to_its_signal():
    m = build(Transfer(5.0, "a", "out", to=None, frm="a"), Transfer(1.0, "a", "b", to="b", frm="a"))
    M = m.matrices([])
    a = m.states.index("a")
    out = m.inputs.index(Signal("t_out"))
    assert M.A[0, a, a] == pytest.approx(-5 / CA - 1 / CA)
    assert M.E[0, a, out] == pytest.approx(5 / CA)


def test_optional_source_is_dropped_outside_the_model_and_required_otherwise():
    t = Transfer(50.0, Signal("t_sup"), "a", to="a", frm="tank", frm_optional=True)
    m = build(t, Transfer(1.0, "out", "b", to="b"))
    M = m.matrices([])
    C = np.array([CA if s == "a" else CB for s in m.states])
    assert (C @ M.E[0])[m.inputs.index(Signal("t_sup"))] == pytest.approx(50.0)  # heat from nowhere
    with pytest.raises(ModelError, match="not a node in this model"):
        build(Transfer(50.0, Signal("t_sup"), "a", to="a", frm="tank"))


def test_a_transfer_must_touch_a_node():
    with pytest.raises(ModelError, match="at least one node"):
        build(Transfer(1.0, "a", "b", to=None, frm=None))


def test_decision_taken_from_a_source():
    m = build(Decision("q", "a", frm="b"), Transfer(1.0, "out", "b", to="b", frm="a"), mode="plan")
    B = m.matrices([]).B
    assert B[m.states.index("a"), 0] == pytest.approx(1 / CA)
    assert B[m.states.index("b"), 0] == pytest.approx(-1 / CB)


RADIATORS_ON_TANK = """\
schema = 1
name = "t"

[[component]]
type = "Ambient"
name = "out"
signal = "t_out"

[[component]]
type = "Zone"
name = "z"
signal = "t_in"

[[component]]
type = "Coupling"
between = ["z", "out"]

[[component]]
type = "JacketTank"
name = "tank"
volume_jacket = "50 l"
volume_dhw = "150 l"
room = "z"
jacket_signal = "t_j"

[[component]]
type = "Radiators"
zone = "z"
source = "tank.jacket"
supply_signal = "t_sup"
flow_signal = "pump"
"""


def test_radiators_take_from_the_tank_what_they_give_the_zone():
    h = loads_house(RADIATORS_ON_TANK)
    m = compose(h)
    M = m.matrices(m.prior_centre())
    z, j = m.states.index("z"), m.states.index("tank.jacket")
    sup = m.inputs.index(Signal("t_sup"))
    Cz = h["z"].param["C"].prior.median
    Cj = h["tank"].capacities({"f_C": 1.0})["tank.jacket"]
    assert Cz * M.E[1, z, sup] == pytest.approx(-Cj * M.E[1, j, sup])
    assert Cz * M.E[1, z, sup] == pytest.approx(h["radiators"].param["UA_rad"].prior.median)
    # envelope-only split: the tank left out, the radiators still heat the zone
    env = compose(h, include=["out", "z", "coupling", "radiators"])
    assert env.states == ("z",)
    assert env.matrices(env.prior_centre()).E[1, 0, env.inputs.index(Signal("t_sup"))] > 0
