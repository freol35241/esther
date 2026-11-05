"""House loader: a positive control, then one test per refusal path.

Every refusal is a small mutation of ``VALID``, and ``test_valid_base_loads``
proves ``VALID`` itself loads — so each refusal test shows that *its* mutation,
and nothing else, is what the loader refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from esther import HouseError, load_house, loads_house
from esther.components.envelope import Zone
from esther.core.component import Component, Port
from esther.core.registry import Registry

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "houses"

VALID = """\
schema = 1
name = "test"
step = "15min"
horizon = "36h"

[[component]]
type = "Ambient"
name = "outdoor"
signal = "outdoor_temperature"

[[component]]
type = "Zone"
name = "house"
prior.C = { median = "8 MJ/K", gsd = 2.0 }

[[component]]
type = "Mass"
name = "fabric"
zone = "house"

[[component]]
type = "Coupling"
between = ["house", "outdoor"]
prior.UA = { median = "120 W/K", gsd = 2.0 }

[[component]]
type = "JacketTank"
name = "tank"
volume_jacket = "60 l"
volume_dhw = "180 l"
room = "house"

[[component]]
type = "Radiators"
zone = "house"
source = "tank.jacket"

[[component]]
type = "HeatPump"
name = "hp"
sink = "tank.jacket"
source = "outdoor"
Q_rated = "2 kW"

[[component]]
type = "Stove"
name = "stove"
zone = "house"
levels = [ { heat = "1 kW", fuel = "0.25 kg/h" }, { heat = "7 kW", fuel = "1.5 kg/h" } ]
efficiency = 0.85
fuel = "pellets"
"""


def mutate(old: str, new: str, text: str = VALID) -> str:
    assert old in text, f"test bug: {old!r} not in the base house"
    return text.replace(old, new, 1)


def refused(text: str) -> HouseError:
    with pytest.raises(HouseError) as exc:
        loads_house(text, source="h.toml")
    return exc.value


def line_of(text: str, needle: str) -> int:
    return next(i for i, ln in enumerate(text.splitlines(), 1) if needle in ln)


# -- positive controls ------------------------------------------------------


def test_valid_base_loads():
    h = loads_house(VALID)
    assert h.name == "test"
    assert h.step == 900.0 and h.horizon == 36 * 3600.0 and h.n_steps == 144
    assert set(h.nodes) == {"house", "fabric", "tank.jacket", "tank.dhw"}
    assert h.boundaries == ("outdoor",)
    assert h["house"].param["C"].prior.median == 8e6
    assert h["coupling"].param["UA"].prior.median == 120.0  # default name
    assert h["radiators"].refs == {"zone": "house", "source": "tank.jacket"}
    assert h["stove"].config["levels"][1]["fuel"] == pytest.approx(1.5 / 3600)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.toml")), ids=lambda p: p.name)
def test_examples_load(path):
    h = load_house(path)
    assert h.components and h.nodes and h.boundaries


def test_multi_source_example_contents():
    h = load_house(EXAMPLES / "multi_source.toml")
    assert h.of_type("Coupling") == [h["envelope"], h["fabric_loss"]]
    assert h["tariff"].config["fuel_price"]["pellets"] == pytest.approx(0.40)
    assert h["tariff"].config["energy_fee"] == pytest.approx(0.10 / 3.6e6)
    assert h["comfort_band"].config["setpoint"] == pytest.approx(294.15)
    assert h["immersion"].config["steps"] == 3
    assert h["stove"].param["t_ign"].prior.median == 1200.0
    assert set(h.signals) == {"outdoor_temperature", "supply_temperature", "spot_price"}


def test_pinned_parameter():
    h = loads_house(
        mutate('prior.C = { median = "8 MJ/K", gsd = 2.0 }', 'prior.C = { value = "8 MJ/K", fixed = true }')
    )
    assert h["house"].param["C"].fixed == 8e6


def test_all_problems_reported_at_once():
    text = mutate('name = "test"\n', 'name = "test"\ncolour = 3\n')
    text = mutate('type = "Zone"', 'type = "Zon"', text)
    text = mutate('Q_rated = "2 kW"', 'Q_rated = "2 kg"', text)
    err = refused(text)
    assert len(err.issues) >= 3
    msg = str(err)
    assert "colour" in msg and "'Zon'" in msg and "Q_rated" in msg


# -- refusals: file and top level -------------------------------------------


def test_toml_syntax_error_has_line():
    text = mutate('signal = "outdoor_temperature"', "signal = outdoor_temperature")
    err = refused(text)
    assert "TOML syntax error" in str(err)
    assert err.issues[0].line == line_of(text, "signal = outdoor_temperature")


def test_unknown_top_level_key():
    text = mutate('horizon = "36h"', 'horizon = "36h"\nhorizn = "1h"')
    err = refused(text)
    assert err.issues[0].line == 5
    assert "unknown top-level key 'horizn'" in str(err)


def test_missing_schema():
    assert "missing 'schema'" in str(refused(mutate("schema = 1\n", "")))


def test_unsupported_schema():
    assert "schema 2 not supported" in str(refused(mutate("schema = 1", "schema = 2")))


def test_horizon_not_whole_steps():
    err = refused(mutate('step = "15min"', 'step = "7min"'))
    assert "not a whole number of steps" in str(err) and err.issues[0].key == "horizon"


def test_step_with_wrong_unit():
    err = refused(mutate('step = "15min"', 'step = "15 kW"'))
    assert "expected time" in str(err) and err.issues[0].key == "step"


def test_no_components():
    text = VALID[: VALID.index("[[component]]")]
    assert "no components" in str(refused(text))


# -- refusals: component types and keys ------------------------------------


def test_unknown_component_type_suggests():
    text = mutate('type = "Stove"', 'type = "Stov"')
    err = refused(text)
    assert "unknown component type 'Stov'" in str(err) and "did you mean 'Stove'" in str(err)
    assert err.issues[0].line == line_of(text, 'type = "Stov"')


def test_missing_type():
    assert "missing 'type'" in str(refused(mutate('type = "Stove"\n', "")))


def test_unknown_component_key():
    text = mutate("efficiency = 0.85", "efficiency = 0.85\nefficency = 0.9")
    err = refused(text)
    assert "unknown key 'efficency' for Stove" in str(err)
    assert err.issues[0].line == line_of(text, "efficency")


def test_missing_required_field():
    assert "missing required key 'Q_rated'" in str(refused(mutate('Q_rated = "2 kW"\n', "")))


# -- refusals: names ----------------------------------------------------------


def test_duplicate_name():
    text = mutate('name = "stove"', 'name = "hp"')
    err = refused(text)
    assert "duplicate name 'hp'" in str(err)
    hp_lines = [i for i, ln in enumerate(text.splitlines(), 1) if ln == 'name = "hp"']
    assert err.issues[0].line == hp_lines[1]  # the Stove's line, not the HeatPump's
    assert err.issues[0].component.startswith("component #8 'hp' (Stove)")


def test_explicit_name_clashing_with_default_name():
    err = refused(mutate('name = "stove"', 'name = "coupling"'))
    assert "default name 'coupling' is taken" in str(err)


def test_invalid_name():
    assert "invalid name 'a.b'" in str(refused(mutate('name = "stove"', 'name = "a.b"')))


def test_second_unnamed_of_a_type_gets_indexed_name():
    text = VALID + '\n[[component]]\ntype = "Coupling"\nbetween = ["fabric", "outdoor"]\n'
    h = loads_house(text)
    assert h["coupling_2"].refs["between"] == ["fabric", "outdoor"]


# -- refusals: ports ----------------------------------------------------------


def test_missing_port():
    err = refused(mutate('zone = "house"\nlevels', "levels"))
    assert "missing port 'zone' (the node it heats)" in str(err)


def test_port_names_nothing():
    text = mutate('sink = "tank.jacket"', 'sink = "tank.jaket"')
    err = refused(text)
    assert "no node named 'tank.jaket'" in str(err) and "did you mean 'tank.jacket'" in str(err)
    assert err.issues[0].line == line_of(text, "tank.jaket")


def test_port_names_component_with_several_nodes():
    err = refused(mutate('source = "tank.jacket"', 'source = "tank"'))
    assert "several nodes; name one of: tank.jacket, tank.dhw" in str(err)


def test_port_names_component_without_node():
    text = (
        VALID
        + '\n[[component]]\ntype = "ComfortBand"\nzone = "hp"\n'
        + 'setpoint = "21 °C"\nband = "0.5 K"\nfloor = "17 °C"\n'
    )
    assert "'hp' is a HeatPump, which has no thermal node" in str(refused(text))


def test_source_cannot_heat_a_boundary():
    err = refused(mutate('zone = "house"\nlevels', 'zone = "outdoor"\nlevels'))
    assert "a source must heat a node, not a boundary" in str(err)


def test_port_wrong_shape():
    err = refused(mutate('between = ["house", "outdoor"]', 'between = ["house"]'))
    assert "must be an array of 2 names" in str(err)


def test_coupling_to_itself():
    err = refused(mutate('between = ["house", "outdoor"]', 'between = ["house", "house"]'))
    assert "couples 'house' to itself" in str(err)


def test_no_cascade_from_a_failed_component():
    # The tank fails on its own; Radiators and HeatPump, which reference
    # tank.jacket, must not add "no node named" noise on top.
    err = refused(mutate('volume_jacket = "60 l"', 'volume_jacket = "60 kg"'))
    assert len(err.issues) == 1 and "volume_jacket" in str(err)


# -- refusals: the thermal graph ---------------------------------------------


def test_node_without_path_to_boundary():
    # Without the envelope coupling, house, fabric and the tank float.
    err = refused(
        mutate(
            'type = "Coupling"\nbetween = ["house", "outdoor"]\n'
            'prior.UA = { median = "120 W/K", gsd = 2.0 }\n',
            'type = "Zone"\nname = "spare"\n',
        )
    )
    msg = str(err)
    assert "node 'house' has no conductance path to a boundary" in msg
    assert "node 'fabric' has no conductance path" in msg
    assert "node 'spare' has no conductance path" in msg


def test_no_boundary_at_all():
    text = VALID.replace(
        'type = "Ambient"\nname = "outdoor"\nsignal = "outdoor_temperature"',
        'type = "Zone"\nname = "outdoor"',
    )
    err = refused(text)
    assert "no boundary in the house" in str(err)


# -- refusals: values, units and priors --------------------------------------


def test_unit_mismatch_in_field():
    text = mutate('Q_rated = "2 kW"', 'Q_rated = "2 kWh"')
    err = refused(text)
    assert "has dimension energy, expected power" in str(err)
    assert err.issues[0].line == line_of(text, "Q_rated")


def test_unit_mismatch_in_prior():
    text = mutate(
        'prior.UA = { median = "120 W/K", gsd = 2.0 }', 'prior.UA = { median = "120 W", gsd = 2.0 }'
    )
    err = refused(text)
    assert "median: '120 W' has dimension power, expected conductance" in str(err)
    assert err.issues[0].line == line_of(text, "prior.UA")


def test_value_out_of_range():
    assert "must be ≤ 1" in str(refused(mutate("efficiency = 0.85", "efficiency = 1.2")))


def test_non_positive_quantity():
    assert "must be > 0" in str(refused(mutate('Q_rated = "2 kW"', 'Q_rated = "0 kW"')))


def test_unknown_parameter():
    err = refused(mutate("prior.UA = { median", "prior.U = { median"))
    assert "Coupling has no parameter 'U'" in str(err) and "did you mean 'UA'" in str(err)


def test_prior_gsd_must_exceed_one():
    err = refused(
        mutate('gsd = 2.0 }\n\n[[component]]\ntype = "Mass"', 'gsd = 1.0 }\n\n[[component]]\ntype = "Mass"')
    )
    assert "gsd must be > 1" in str(err)


def test_prior_outside_bounds():
    err = refused(mutate('median = "8 MJ/K"', 'median = "8 kJ/K"'))
    assert "outside the bounds" in str(err)


def test_prior_bad_keys():
    err = refused(mutate('{ median = "8 MJ/K", gsd = 2.0 }', '{ median = "8 MJ/K" }'))
    assert "prior keys ['median'] not understood" in str(err)


def test_fixed_must_be_true():
    err = refused(
        mutate('prior.C = { median = "8 MJ/K", gsd = 2.0 }', 'prior.C = { value = "8 MJ/K", fixed = false }')
    )
    assert "fixed must be true" in str(err)


def test_component_cross_field_check():
    err = refused(mutate('{ heat = "7 kW", fuel', '{ heat = "0.5 kW", fuel'))
    assert "increasing order of heat output" in str(err)


def test_table_list_item_errors():
    err = refused(mutate('{ heat = "1 kW", fuel = "0.25 kg/h" }', '{ heat = "1 kW", fuel = "0.25 kg" }'))
    assert "item 1, fuel" in str(err) and "expected mass flow" in str(err)


# -- the registry ---------------------------------------------------------


def test_custom_registry_resolves_third_party_type():
    class Boiler(Component):
        ports = (Port("node", heats=True),)

    reg = Registry()  # still sees the built-ins through entry points
    reg.register(Boiler, "acme.Boiler")
    text = VALID + '\n[[component]]\ntype = "acme.Boiler"\nnode = "tank.jacket"\n'
    h = loads_house(text, registry=reg)
    assert type(h["boiler"]) is Boiler  # default name from the last dotted part


def test_registry_without_entry_points_knows_nothing():
    reg = Registry(use_entry_points=False)
    reg.register(Zone)
    err = refused_with(VALID, reg)
    assert "unknown component type 'Ambient'" in str(err)


def refused_with(text: str, reg: Registry) -> HouseError:
    with pytest.raises(HouseError) as exc:
        loads_house(text, registry=reg)
    return exc.value
