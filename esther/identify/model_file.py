"""The fitted-model file (spec §9): a fit, written once, loaded by a live planner.

A live planner does not identify anything (re-fitting is a scheduled batch
job, §6.6); it loads the result of one. The file holds

- ``values``: the parameter values, flat ``"component.param"`` — a fit's,
  several fits' merged (the fitting split, §6.4), and any learned device
  habit (§10.2's replay);
- the **gate decision** that admitted them (accepted, and why not), and the
  validation metrics the gate saw — numbers only;
- free-form ``provenance`` (strings: what was fitted on what, by whom);
- ``house_sha256``, the hash of the house file's text the values were fitted
  for. A parameter belongs to a structure: loaded against another house, the
  same numbers mean something else, so :func:`load_model` refuses a mismatch;
- ``version``, a content hash over a canonical JSON of the values and the
  house hash — the model version a plan carries (§9.3). It does not depend on
  key order or on how a number was written, and changes with any value.

A refused gate is recorded, not hidden: a planner built from such a file plans
in **advisory** mode (§8.8).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fit import FitResult
from .validate import GateDecision, Validation

__all__ = [
    "FittedModel",
    "ModelFileError",
    "fitted_model",
    "gate_metrics",
    "house_sha256",
    "load_model",
    "model_version",
    "save_model",
]

FORMAT = "esther.model/1"


class ModelFileError(ValueError):
    """A model file that cannot be used: malformed, edited after it was written
    (its version no longer matches its content), or fitted for another house."""


def house_sha256(text: str) -> str:
    """The sha256 of a house file's text (UTF-8), hex."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _values(values: Mapping[str, float]) -> dict[str, float]:
    out = {}
    for k, v in values.items():
        x = float(v)
        if not math.isfinite(x):
            raise ModelFileError(f"value of {k!r} is not finite ({x})")
        out[str(k)] = x
    return out


def model_version(values: Mapping[str, float], house: str) -> str:
    """Content hash of ``values`` and the house hash ``house``: sha256 over
    canonical JSON (sorted keys, each number as the shortest repr of its
    float), first 16 hex digits."""
    doc = {"house_sha256": house, "values": _values(values)}
    canon = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def _number(x: Any) -> float | int | None:
    """A JSON number; NaN and infinities become None."""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    x = float(x)
    return x if math.isfinite(x) else None


def _table(df) -> dict[str, Any]:
    """A metrics table (index (output, horizon) or output) as nested dicts of numbers."""
    out: dict[str, Any] = {}
    for key, row in df.iterrows():
        cells = {str(c): _number(v) for c, v in row.items()}
        if isinstance(key, tuple):
            out.setdefault(str(key[0]), {})[f"{float(key[1]):g}"] = cells
        else:
            out[str(key)] = cells
    return out


def gate_metrics(gate: GateDecision | None, validation: Validation | None = None) -> dict[str, Any]:
    """The numbers behind a gate decision: the gated RMSE and its reference,
    and, with the candidate's :class:`Validation`, its open-loop,
    persistence and innovation tables (open-loop keyed by output, then
    horizon in hours)."""
    out: dict[str, Any] = {}
    if gate is not None:
        out["gate"] = {
            "horizon_s": _number(gate.horizon),
            "candidate_rmse": _number(gate.candidate_rmse),
            "reference_rmse": _number(gate.reference_rmse),
        }
    if validation is not None:
        out["open_loop"] = _table(validation.open_loop)
        out["persistence"] = _table(validation.persistence)
        out["innovation"] = _table(validation.innovation)
        out["n_steps"] = int(validation.n_steps)
        out["step_s"] = _number(validation.step)
    return out


@dataclass(frozen=True)
class FittedModel:
    """A fit as a live planner loads it. ``version`` is derived, never stored
    independently: see :func:`model_version`."""

    values: dict[str, float]
    house_sha256: str
    accepted: bool
    reasons: list[str] = field(default_factory=list)  # why refused; empty when accepted
    warnings: list[str] = field(default_factory=list)  # reported, not gating
    gate: dict[str, str] = field(default_factory=dict)  # the gate's output and reference
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _values(self.values))
        bad = [k for k, v in self.provenance.items() if not (isinstance(k, str) and isinstance(v, str))]
        if bad:
            raise ModelFileError(f"provenance must map strings to strings: {bad}")

    @property
    def version(self) -> str:
        return model_version(self.values, self.house_sha256)

    # -- JSON ----------------------------------------------------------------
    def to_json(self) -> str:
        doc = {
            "format": FORMAT,
            "version": self.version,
            "house_sha256": self.house_sha256,
            "values": dict(sorted(self.values.items())),
            "accepted": bool(self.accepted),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "gate": dict(self.gate),
            "metrics": self.metrics,
            "provenance": dict(self.provenance),
        }
        return json.dumps(doc, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str, house_text: str | None = None) -> FittedModel:
        """Parse a model file. With ``house_text``, refuse a file fitted for
        another house. A file whose stored version does not match its content
        is refused either way."""
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            raise ModelFileError(f"not JSON: {e}") from e
        if not isinstance(doc, dict) or doc.get("format") != FORMAT:
            got = doc.get("format") if isinstance(doc, dict) else type(doc).__name__
            raise ModelFileError(f"not an esther model file (format {got!r}, expected {FORMAT!r})")
        missing = [k for k in ("version", "house_sha256", "values", "accepted") if k not in doc]
        if missing:
            raise ModelFileError(f"model file lacks {', '.join(missing)}")
        m = cls(
            values=doc["values"],
            house_sha256=doc["house_sha256"],
            accepted=bool(doc["accepted"]),
            reasons=list(doc.get("reasons", [])),
            warnings=list(doc.get("warnings", [])),
            gate=dict(doc.get("gate", {})),
            metrics=dict(doc.get("metrics", {})),
            provenance=dict(doc.get("provenance", {})),
        )
        if m.version != doc["version"]:
            raise ModelFileError(
                f"model file says version {doc['version']}, its content hashes to {m.version}: "
                "edited after it was written"
            )
        if house_text is not None:
            h = house_sha256(house_text)
            if h != m.house_sha256:
                raise ModelFileError(
                    f"model {m.version} was fitted for the house with sha256 {m.house_sha256[:12]}…, "
                    f"not this one ({h[:12]}…); re-fit, or load the house it was fitted for"
                )
        return m


def fitted_model(
    house_text: str,
    values: Mapping[str, float] | FitResult,
    gate: GateDecision | None,
    validation: Validation | None = None,
    provenance: Mapping[str, str] | None = None,
) -> FittedModel:
    """A :class:`FittedModel` for the house ``house_text`` from a fit (or
    merged values) and the gate decision on it. Without a gate decision the
    model is not accepted: no evidence is not a pass (§6.5)."""
    vals = values.params if isinstance(values, FitResult) else dict(values)
    if gate is None:
        accepted, reasons, warnings, g = False, ["no acceptance-gate decision"], [], {}
    else:
        accepted, reasons, warnings = gate.accepted, list(gate.reasons), list(gate.warnings)
        g = {"output": gate.output, "reference": gate.reference}
    return FittedModel(
        values=vals,
        house_sha256=house_sha256(house_text),
        accepted=bool(accepted),
        reasons=reasons,
        warnings=warnings,
        gate=g,
        metrics=gate_metrics(gate, validation),
        provenance=dict(provenance or {}),
    )


def save_model(path: str | Path, model: FittedModel) -> None:
    Path(path).write_text(model.to_json(), encoding="utf-8")


def load_model(path: str | Path, house_text: str) -> FittedModel:
    """Load a model file for the house whose text is ``house_text``; refuse
    one fitted for another house (:class:`ModelFileError`)."""
    return FittedModel.from_json(Path(path).read_text(encoding="utf-8"), house_text=house_text)
