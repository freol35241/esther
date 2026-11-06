"""The component contract (spec §4.1) — the part of it the house loader needs.

A component class declares, as class attributes:

- ``fields``: its configuration keys, each with a kind and (for quantities) a
  dimension, so that the loader can parse and check them;
- ``ports``: the keys that name other things in the house — nodes, boundaries
  or components — resolved at composition (§4.4);
- ``params``: its parameters with default priors and bounds (§4.2).

and, as methods, what it adds to the thermal graph: the nodes it owns, the
boundaries it imposes and the conductances it creates. That is enough to
validate a house (every port resolves, every node has a path to a boundary,
every source has a node to heat).

From M2, a component also contributes its **dynamics and measurements**
(§4.1, §4.3): :meth:`Component.dynamics` returns linear terms
(:mod:`esther.core.model`) as a function of the component's parameter values,
and :meth:`Component.active_params` says which of its parameters those terms
use. A component that does not override ``dynamics`` has none in this
version (the tank and source types until M3) and is skipped by
:func:`esther.core.model.compose`.

The remaining contributions — decisions with their constraints and cost, and
actuation — are added in M4. They are not stubbed here: an interface written
before its first implementation would be a guess with the authority of code.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from .params import Param
from .units import Dimension, UnitError, parse_quantity

__all__ = [
    "Boolean",
    "Component",
    "Field",
    "FieldError",
    "Integer",
    "Mapping_",
    "Number",
    "Port",
    "Quantity",
    "String",
    "TableList",
]


class FieldError(ValueError):
    """A configuration value is malformed. The message names the problem only;
    the loader adds where it is."""


@dataclass(frozen=True)
class Field:
    """A configuration key. Subclasses implement :meth:`parse`."""

    name: str
    required: bool = True
    default: Any = None
    doc: str = ""

    def parse(self, value: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class Quantity(Field):
    """A value with a unit, stored in SI. ``absolute`` for temperatures in °C."""

    dim: Dimension = field(default_factory=Dimension)
    absolute: bool = False
    min: float | None = None  # SI, inclusive
    max: float | None = None
    positive: bool = False

    def parse(self, value: Any) -> float:
        try:
            v = parse_quantity(value, self.dim, absolute_temperature=self.absolute)
        except UnitError as e:
            raise FieldError(str(e)) from None
        if not math.isfinite(v):
            raise FieldError(f"must be finite, got {value!r}")
        if self.positive and not v > 0:
            raise FieldError(f"must be > 0, got {value!r}")
        if self.min is not None and v < self.min:
            raise FieldError(f"{value!r} is below the minimum ({self.min:g} SI)")
        if self.max is not None and v > self.max:
            raise FieldError(f"{value!r} is above the maximum ({self.max:g} SI)")
        return v


@dataclass(frozen=True)
class Number(Field):
    """A dimensionless real number with optional open/closed range."""

    min: float | None = None
    max: float | None = None
    min_exclusive: bool = False

    def parse(self, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise FieldError(f"must be a number, got {value!r}")
        v = float(value)
        if not math.isfinite(v):
            raise FieldError(f"must be finite, got {value!r}")
        if self.min is not None and (v <= self.min if self.min_exclusive else v < self.min):
            op = ">" if self.min_exclusive else "≥"
            raise FieldError(f"must be {op} {self.min:g}, got {v:g}")
        if self.max is not None and v > self.max:
            raise FieldError(f"must be ≤ {self.max:g}, got {v:g}")
        return v


@dataclass(frozen=True)
class Integer(Field):
    min: int | None = None
    max: int | None = None

    def parse(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise FieldError(f"must be an integer, got {value!r}")
        if self.min is not None and value < self.min:
            raise FieldError(f"must be ≥ {self.min}, got {value}")
        if self.max is not None and value > self.max:
            raise FieldError(f"must be ≤ {self.max}, got {value}")
        return value


@dataclass(frozen=True)
class String(Field):
    choices: tuple[str, ...] | None = None

    def parse(self, value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise FieldError(f"must be a non-empty string, got {value!r}")
        if self.choices is not None and value not in self.choices:
            raise FieldError(f"must be one of {', '.join(self.choices)}; got {value!r}")
        return value


@dataclass(frozen=True)
class Boolean(Field):
    def parse(self, value: Any) -> bool:
        if not isinstance(value, bool):
            raise FieldError(f"must be true or false, got {value!r}")
        return value


@dataclass(frozen=True)
class TableList(Field):
    """An array of inline tables, each with the given sub-fields."""

    item_fields: tuple[Field, ...] = ()
    min_items: int = 1

    def parse(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise FieldError("must be an array of tables")
        if len(value) < self.min_items:
            raise FieldError(f"needs at least {self.min_items} item(s), got {len(value)}")
        known = {f.name: f for f in self.item_fields}
        out = []
        for i, item in enumerate(value, 1):
            if not isinstance(item, dict):
                raise FieldError(f"item {i} must be a table, got {item!r}")
            unknown = sorted(set(item) - set(known))
            if unknown:
                raise FieldError(
                    f"item {i}: unknown key(s) {', '.join(unknown)}; expected {', '.join(known)}"
                )
            parsed = {}
            for f in self.item_fields:
                if f.name in item:
                    try:
                        parsed[f.name] = f.parse(item[f.name])
                    except FieldError as e:
                        raise FieldError(f"item {i}, {f.name}: {e}") from None
                elif f.required:
                    raise FieldError(f"item {i}: missing key {f.name}")
                else:
                    parsed[f.name] = f.default
            out.append(parsed)
        return out


@dataclass(frozen=True)
class Mapping_(Field):
    """A table from free-form keys to values of one field kind."""

    value_field: Field = field(default_factory=lambda: Number("value"))

    def parse(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise FieldError("must be a table")
        out = {}
        for k, v in value.items():
            try:
                out[k] = self.value_field.parse(v)
            except FieldError as e:
                raise FieldError(f"{k}: {e}") from None
        return out


PortKind = Literal["node", "boundary", "thermal", "component"]


@dataclass(frozen=True)
class Port:
    """A key whose value names something else in the house.

    kind:
        ``node``      a thermal node (has a temperature state)
        ``boundary``  an imposed temperature
        ``thermal``   either of the two
        ``component`` another component, by name
    count:
        ``None`` for a single name, ``n`` for an array of exactly ``n`` names.
    heats:
        this port is where the component delivers heat; it must be a node
        (heating a boundary is meaningless) — "a source has a node to heat".
    """

    name: str
    kind: PortKind = "node"
    count: int | None = None
    required: bool = True
    heats: bool = False
    component_type: str | None = None  # for kind="component"
    doc: str = ""


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


class Component:
    """Base class for house components.

    Subclasses set ``fields``, ``ports`` and ``params`` and override the graph
    methods as needed. The registry gives each class its ``type_name``.
    """

    type_name: ClassVar[str] = ""
    fields: ClassVar[tuple[Field, ...]] = ()
    ports: ClassVar[tuple[Port, ...]] = ()
    params: ClassVar[tuple[Param, ...]] = ()
    doc: ClassVar[str] = ""

    def __init__(
        self,
        name: str,
        config: Mapping[str, Any],
        refs: Mapping[str, str | list[str] | None],
        params: Mapping[str, Param],
    ) -> None:
        self.name = name
        self.config = dict(config)
        self.refs = dict(refs)
        self.param = dict(params)

    # -- the thermal graph -------------------------------------------------
    def nodes(self) -> list[str]:
        """Thermal nodes this component owns (fully qualified names)."""
        return []

    def boundaries(self) -> list[str]:
        """Boundary temperatures this component imposes."""
        return []

    def conductances(self) -> list[tuple[str, str]]:
        """Heat paths with a conductance, as pairs of node/boundary names."""
        return []

    def signals(self) -> list[str]:
        """Named signals this component reads (spec §5: named, not addressed)."""
        return []

    def check(self) -> list[tuple[str | None, str]]:
        """Cross-field sanity checks: ``[(key or None, message), ...]``."""
        return []

    # -- dynamics (spec §4.3) ----------------------------------------------
    def dynamics(self, p: Mapping[str, float], mode: str) -> list[Any]:
        """Linear terms (:mod:`esther.core.model`) for parameter values ``p``
        (this component's, by local name). ``mode`` is ``"identify"`` or
        ``"plan"``. Which terms exist MUST NOT depend on the values in ``p``."""
        raise NotImplementedError(f"{self.type_name or type(self).__name__} has no dynamics yet")

    def active_params(self, mode: str) -> list[str]:
        """The parameters :meth:`dynamics` uses in ``mode`` (default: all)."""
        return [p.name for p in self.params]

    def boundary_signals(self) -> dict[str, str]:
        """Boundary name → the signal that gives its temperature."""
        return {}

    # -- helpers -----------------------------------------------------------
    @classmethod
    def declared_keys(cls) -> set[str]:
        return {f.name for f in cls.fields} | {p.name for p in cls.ports}

    @staticmethod
    def valid_name(name: str) -> bool:
        return bool(_NAME_RE.match(name))

    def __repr__(self) -> str:
        return f"<{self.type_name or type(self).__name__} {self.name!r}>"
