from importlib.metadata import EntryPoint

import pytest

from esther.components.envelope import Zone
from esther.core import registry as registry_mod
from esther.core.component import Component
from esther.core.registry import ENTRY_POINT_GROUP, Registry, RegistryError

BUILT_INS = {
    "Zone",
    "Mass",
    "Ambient",
    "Coupling",
    "Radiators",
    "JacketTank",
    "HeatPump",
    "ElectricHeater",
    "Stove",
    "HotWaterDraw",
    "Tariff",
    "ComfortBand",
    "DhwService",
}


class Boiler(Component):
    pass


def fake_entry_points(*eps):
    def entry_points(group):
        assert group == ENTRY_POINT_GROUP
        return list(eps)

    return entry_points


def ep(name, value):
    return EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)


def test_built_ins_come_through_entry_points():
    # The installed package's own entry points — the positive control that
    # the mechanism a third party uses is the one esther uses.
    reg = Registry()
    assert set(reg.names()) >= BUILT_INS
    assert reg.resolve("Zone") is Zone
    assert Zone.type_name == "Zone"


def test_third_party_entry_point(monkeypatch):
    monkeypatch.setattr(
        registry_mod, "entry_points", fake_entry_points(ep("acme.Boiler", f"{__name__}:Boiler"))
    )
    reg = Registry()
    assert "acme.Boiler" in reg
    assert reg.resolve("acme.Boiler") is Boiler


def test_unknown_type():
    with pytest.raises(RegistryError, match="unknown component type 'Nope'"):
        Registry().resolve("Nope")


def test_broken_plugin_is_reported_by_name(monkeypatch):
    monkeypatch.setattr(
        registry_mod, "entry_points", fake_entry_points(ep("acme.Broken", "no_such_module_xyz:Thing"))
    )
    reg = Registry()
    with pytest.raises(RegistryError, match=r"'acme.Broken' \(no_such_module_xyz:Thing\) failed to load"):
        reg.resolve("acme.Broken")


def test_entry_point_must_be_a_component(monkeypatch):
    monkeypatch.setattr(registry_mod, "entry_points", fake_entry_points(ep("acme.Bad", "math:sqrt")))
    with pytest.raises(RegistryError, match="not a Component subclass"):
        Registry().resolve("acme.Bad")


def test_conflicting_entry_points(monkeypatch):
    monkeypatch.setattr(
        registry_mod,
        "entry_points",
        fake_entry_points(ep("X", f"{__name__}:Boiler"), ep("X", "esther.components.envelope:Zone")),
    )
    with pytest.raises(RegistryError, match="registered more than once"):
        Registry().resolve("X")


def test_register_conflict():
    reg = Registry(use_entry_points=False)
    reg.register(Boiler, "B")
    with pytest.raises(RegistryError, match="already registered"):
        reg.register(Zone, "B")


def test_register_rejects_non_component():
    with pytest.raises(RegistryError, match="not a Component subclass"):
        Registry(use_entry_points=False).register(int, "int")  # type: ignore[arg-type]
