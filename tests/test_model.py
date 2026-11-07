"""Composition and discretisation (spec §4.3, §4.4): matrices from terms,
exact ZOH against closed forms and fine integration, and the refusals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esther import load_house, loads_house
from esther.components.envelope import RHO_CP_AIR
from esther.core.model import Constant, Harmonic, ModelError, Signal, compose, discretise

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "houses"

BASE = """\
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
"""

COUPLING = """
[[component]]
type = "Coupling"
name = "env"
between = ["z", "out"]
"""


def model(*parts: str, mode: str = "identify"):
    return compose(loads_house(BASE + "".join(parts)), mode=mode)


def test_one_node_matrices():
    m = model(
        COUPLING,
        """
[[component]]
type = "Radiators"
zone = "z"
supply_signal = "t_sup"
flow_signal = "pump"

[[component]]
type = "InternalGain"
zone = "z"
""",
    )
    assert m.states == ("z",)
    assert m.inputs == (Signal("t_out"), Signal("t_sup"), Constant())
    assert m.modulations == ("pump",) and m.outputs == ("t_in",)
    th = {
        "z.C": 2e6,
        "z.sigma_w": 0.3,
        "z.sigma_v": 0.1,
        "env.UA": 100.0,
        "radiators.UA_rad": 50.0,
        "internal_gain.Q": 400.0,
    }
    M = m.matrices(th)
    assert M.A[0, 0, 0] == pytest.approx(-100 / 2e6)
    assert M.A[1, 0, 0] == pytest.approx(-50 / 2e6)  # modulated: pump on adds UA_rad
    assert M.E[0, 0].tolist() == pytest.approx([100 / 2e6, 0.0, 400 / 2e6])
    assert M.E[1, 0].tolist() == pytest.approx([0.0, 50 / 2e6, 0.0])
    assert M.Qc[0, 0] == pytest.approx(0.3**2 / 3600)
    assert M.H.tolist() == [[1.0]] and M.R[0, 0] == pytest.approx(0.01)
    A_on, E_on = M.at([1.0])
    assert A_on[0, 0] == pytest.approx(-150 / 2e6) and E_on[0, 1] == pytest.approx(50 / 2e6)
    A_half, _ = M.at([0.5])
    assert A_half[0, 0] == pytest.approx(-125 / 2e6)


def test_two_node_and_splits():
    m = model(
        COUPLING,
        """
[[component]]
type = "Mass"
name = "m"
zone = "z"

[[component]]
type = "SolarGain"
zone = "z"
mass = "m"
signal = "sun"

[[component]]
type = "Ventilation"
zone = "z"
ambient = "out"

[[component]]
type = "Sensor"
node = "z"
signal = "t_in2"
""",
    )
    th = dict(zip(m.param_names, m.prior_centre(), strict=True))
    th.update({"z.C": 1e6, "m.C": 1e7, "m.UA": 200.0, "env.UA": 80.0, "solar_gain.g_A": 2.0})
    th.update({"solar_gain.f_mass": 0.25, "ventilation.flow": 0.05, "sensor.bias": -0.4})
    M = m.matrices(th)
    vent = RHO_CP_AIR * 0.05
    expect = [[-(80 + 200 + vent) / 1e6, 200 / 1e6], [200 / 1e7, -200 / 1e7]]
    assert np.allclose(M.A[0], expect, rtol=1e-12, atol=0)
    sun = m.inputs.index(Signal("sun"))
    assert M.E[0, :, sun].tolist() == pytest.approx([0.75 * 2 / 1e6, 0.25 * 2 / 1e7])
    assert m.outputs == ("t_in", "t_in2") and M.c.tolist() == [0.0, -0.4]
    assert M.H.tolist() == [[1.0, 0.0], [1.0, 0.0]]


def test_coupling_boundary_first_is_the_same():
    a = model(COUPLING)
    b = model(COUPLING.replace('["z", "out"]', '["out", "z"]'))
    th = a.prior_centre()
    assert np.allclose(a.matrices(th).A, b.matrices(th).A)
    assert np.allclose(a.matrices(th).E, b.matrices(th).E)


def test_internal_gain_harmonics():
    m = model(COUPLING, '\n[[component]]\ntype = "InternalGain"\nzone = "z"\nharmonics = 2\n')
    assert "internal_gain.a2" in m.param_names and "internal_gain.a3" not in m.param_names
    assert Harmonic("sin", 2, 86400.0) in m.inputs
    t = np.array([0.0, 21600.0])
    assert Harmonic("cos", 1).evaluate(t) == pytest.approx([1.0, 0.0], abs=1e-12)


def test_fixed_parameters_are_not_free():
    m = model(COUPLING.replace('"out"]', '"out"]\nprior.UA = { value = "90 W/K", fixed = true }'))
    assert "env.UA" not in m.param_names and m.fixed == {"env.UA": 90.0}
    assert m.matrices(m.prior_centre()).A[0, 0, 0] == pytest.approx(-90 / 5e6)


def test_plan_mode_radiators_are_a_decision():
    m = model(COUPLING, '\n[[component]]\ntype = "Radiators"\nname = "rad"\nzone = "z"\n', mode="plan")
    assert m.decisions == ("rad.Q",)
    assert m.matrices(m.prior_centre()).B[0, 0] == pytest.approx(1 / 5e6)


def test_plan_mode_composes_the_tank_and_sources():
    h = load_house(EXAMPLES / "multi_source.toml")
    m = compose(h, mode="plan")
    assert set(m.skipped) == {"tariff", "comfort_band", "dhw_service"}
    assert {"tank.jacket", "tank.dhw"} <= set(m.states)
    assert set(m.decisions) == {"radiators.Q", "heat_pump.Q", "immersion.Q", "stove.Q"}
    B = m.matrices(m.prior_centre()).B
    j, z, r = m.states.index("tank.jacket"), m.states.index("living"), m.decisions.index("radiators.Q")
    # the radiators' heat enters the zone and leaves the jacket they mix from
    assert B[z, r] > 0 > B[j, r]


# -- refusals --------------------------------------------------------------


def test_refuses_radiators_without_supply_in_identification():
    with pytest.raises(ModelError, match="supply_signal"):
        model(COUPLING, '\n[[component]]\ntype = "Radiators"\nzone = "z"\n')


def test_refuses_unbounded_node():
    # The house is fine; the sub-model without the coupling is not (§6.4 split).
    h = loads_house(BASE + COUPLING)
    assert compose(h, include=["out", "z", "env"]).states == ("z",)
    with pytest.raises(ModelError, match="no conductance path"):
        compose(h, include=["out", "z"])


def test_refuses_no_measurement():
    with pytest.raises(ModelError, match="no measurements"):
        compose(loads_house(BASE.replace('signal = "t_in"\n', "") + COUPLING))


def test_refuses_include_of_unmodelled_or_unknown():
    h = load_house(EXAMPLES / "multi_source.toml")
    with pytest.raises(ModelError, match="without dynamics"):
        compose(h, include=["tariff"], mode="plan")
    with pytest.raises(ModelError, match="no component named"):
        compose(h, include=["nope"])


def test_refuses_a_node_outside_the_model():
    h = loads_house(
        BASE
        + COUPLING
        + '\n[[component]]\ntype = "Mass"\nname = "m"\nzone = "z"\n'
        + '\n[[component]]\ntype = "HeatInput"\nnode = "m"\nsignal = "q"\n'
    )
    with pytest.raises(ModelError, match="not a node in this model"):
        compose(h, include=["out", "z", "env", "heat_input"])


# -- discretisation ----------------------------------------------------------


def test_discretise_scalar_closed_form():
    a, e, q, dt = -1e-4, 3e-5, 2e-6, 600.0
    d = discretise(np.array([[a]]), np.array([[e]]), np.array([[q]]), dt)
    assert d.Ad[0, 0] == pytest.approx(np.exp(a * dt), rel=1e-12)
    assert d.Gd[0, 0] == pytest.approx(e * (np.exp(a * dt) - 1) / a, rel=1e-10)
    assert d.Qd[0, 0] == pytest.approx(q * (np.exp(2 * a * dt) - 1) / (2 * a), rel=1e-10)


def test_discretise_matches_fine_integration():
    # Control: an independent integrator (RK4 at 1 s) must agree with the
    # matrix exponential for a two-node system with a held input.
    A = np.array([[-3e-4, 1e-4], [2e-5, -2e-5]])
    E = np.array([[1e-4, 5e-5], [0.0, 1e-6]])
    d = discretise(A, E, np.zeros((2, 2)), 900.0)
    x0, u = np.array([293.0, 290.0]), np.array([275.0, 2.0])
    x = x0.copy()

    def f(x):
        return A @ x + E @ u

    for _ in range(900):
        k1 = f(x)
        k2 = f(x + 0.5 * k1)
        k3 = f(x + 0.5 * k2)
        k4 = f(x + k3)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    assert d.Ad @ x0 + d.Gd @ u == pytest.approx(x, abs=1e-9)


def test_discretise_batched():
    A = np.array([[[-1e-4]], [[-2e-4]]])
    E = np.ones((2, 1, 1)) * 1e-5
    d = discretise(A, E, np.full((1, 1), 1e-6), 300.0)
    assert d.Ad.shape == (2, 1, 1)
    assert d.Ad[1, 0, 0] == pytest.approx(np.exp(-2e-4 * 300))
