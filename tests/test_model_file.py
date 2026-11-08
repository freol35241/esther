"""The fitted-model file (spec §9): round trip, the content hash, the house
check, and the gate verdict reaching the planner (§8.8: refused → advisory).

Each refusal has its positive twin: the same file loaded against the house it
was fitted for, an accepted file next to the refused one.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import synthetic as syn
from synthetic_plan import forecast, house_text, x0

from esther import loads_house
from esther.identify import (
    FittedModel,
    GateDecision,
    ModelFileError,
    fitted_model,
    load_model,
    model_version,
    save_model,
    validate,
)
from esther.plan import Planner, load_planner

TRUTH = {
    "zone.C": 4e6,
    "zone.sigma_w": 0.05,
    "zone.sigma_v": 0.03,
    "mass.C": 3e7,
    "mass.UA": 400.0,
    "env.UA": 120.0,
    "radiators.UA_rad": 200.0,
}
NOW = pd.Timestamp("2025-01-15T20:00Z")


def gate(accepted: bool) -> GateDecision:
    return GateDecision(
        accepted=accepted,
        reasons=[] if accepted else ["24 h open-loop RMSE 0.9 K is worse than the persistence's 0.5 K"],
        output="t_in",
        horizon=86400.0,
        candidate_rmse=0.3 if accepted else 0.9,
        reference_rmse=0.5,
        reference="persistence",
        warnings=["not identified by the data (resting on the prior): mass.C"],
    )


@pytest.fixture(scope="module")
def validation():
    """A real Validation (the true parameters on held-out synthetic data), so
    the metrics written are the tables the gate sees."""
    m = syn.compose_house(syn.MASS, syn.ENVELOPE, syn.RADIATORS, syn.KNOWN_HEAT)
    data, _ = syn.dataset(m, TRUTH, days=4.0, step=900.0, gaps=1)
    _, test = data.split(0.5)
    return validate(m, [TRUTH[n] for n in m.param_names], test)


def test_round_trip(tmp_path, validation):
    text = house_text()
    prov = {"fitted_on": "synthetic", "by": "test"}
    fm = fitted_model(text, TRUTH, gate(True), validation, provenance=prov)
    path = tmp_path / "model.json"
    save_model(path, fm)
    back = load_model(path, house_text=text)
    assert back == fm
    assert back.version == fm.version
    assert back.values == TRUTH and back.accepted and back.reasons == []
    assert back.warnings == gate(True).warnings
    assert back.gate == {"output": "t_in", "reference": "persistence"}
    # the metrics are the validation's tables, as numbers (NaN as null) — strict JSON
    doc = json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"bare {c} in the file"))
    ol = doc["metrics"]["open_loop"]["t_in"]
    assert set(ol) == {"1", "6", "24"}
    assert ol["6"]["rmse"] == pytest.approx(validation.open_loop.loc[("t_in", 6.0), "rmse"])
    assert doc["metrics"]["gate"]["candidate_rmse"] == 0.3

    def leaves(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from leaves(x)
        else:
            yield v

    numbers = list(leaves(doc["metrics"]))
    assert all(x is None or (isinstance(x, (int, float)) and math.isfinite(x)) for x in numbers)


def test_version_is_a_content_hash():
    h = "0" * 64
    a = model_version({"zone.C": 4e6, "env.UA": 120.0}, h)
    # key order, the way a number is written, int vs float vs numpy: the same
    assert model_version({"env.UA": 120, "zone.C": 4000000.0}, h) == a
    assert model_version({"env.UA": np.float64("1.2e2"), "zone.C": np.float32(4e6)}, h) == a
    # any change of any value, by one ulp, changes it; so does the house
    for k in ("zone.C", "env.UA"):
        v = {"zone.C": 4e6, "env.UA": 120.0}
        v[k] = float(np.nextafter(v[k], np.inf))
        assert model_version(v, h) != a
    assert model_version({"zone.C": 4e6, "env.UA": 120.0, "mass.C": 3e7}, h) != a
    assert model_version({"zone.C": 4e6, "env.UA": 120.0}, "1" + h[1:]) != a
    with pytest.raises(ModelFileError, match="not finite"):
        model_version({"zone.C": math.nan}, h)


def test_house_mismatch_is_refused(tmp_path):
    text = house_text()
    path = tmp_path / "model.json"
    save_model(path, fitted_model(text, {}, gate(True)))
    assert load_model(path, house_text=text).accepted  # positive twin: the house it was fitted for
    other = house_text(pellet_price=0.5)
    with pytest.raises(ModelFileError, match="fitted for the house"):
        load_model(path, house_text=other)
    with pytest.raises(ModelFileError, match="fitted for the house"):
        load_model(path, house_text=text + "\n")  # the hash is of the text: any edit counts


def test_an_edited_file_is_refused(tmp_path):
    text = house_text()
    path = tmp_path / "model.json"
    save_model(path, fitted_model(text, TRUTH, gate(True)))
    doc = json.loads(path.read_text())
    doc["values"]["env.UA"] = 121.0
    path.write_text(json.dumps(doc))
    with pytest.raises(ModelFileError, match="edited after it was written"):
        load_model(path, house_text=text)
    with pytest.raises(ModelFileError, match="not an esther model file"):
        FittedModel.from_json(json.dumps({"values": {}}))


def test_no_gate_is_not_a_pass():
    fm = fitted_model(house_text(), TRUTH, None)
    assert not fm.accepted and fm.reasons


def test_gate_verdict_reaches_the_planner(tmp_path):
    text = house_text()
    hp = tmp_path / "house.toml"
    hp.write_text(text)
    pl0 = Planner(loads_house(text), {})
    free = pl0.model.free[0].name
    values = {free: 1.5 * pl0.model.free[0].param.prior.centre}
    plans = {}
    for ok in (False, True):
        mp = tmp_path / f"model-{ok}.json"
        fm = fitted_model(text, values, gate(ok))
        save_model(mp, fm)
        pl = load_planner(hp, mp)
        assert pl.theta[0] == values[free]  # the file's values, not the priors
        assert pl.model_version == fm.version and pl.accepted is ok
        h = pl.house
        plans[ok] = pl.plan(NOW, x0(h), forecast(NOW, h))
        assert plans[ok].model_version == fm.version
        assert plans[ok].validation["accepted"] is ok
    assert plans[False].mode == "advisory" and plans[False].advisory
    assert plans[True].mode == "active"  # positive twin
    other = tmp_path / "other.toml"
    other.write_text(house_text(pellet_price=0.5))
    with pytest.raises(ModelFileError):
        load_planner(other, tmp_path / "model-True.json")
