"""House description loading and validation (spec §5).

A house is one TOML file::

    schema = 1
    name = "example"
    step = "15min"
    horizon = "36h"

    [[component]]
    type = "Zone"
    name = "living"
    prior.C = { median = "6 MJ/K", gsd = 2.0 }
    ...

Validation runs before anything else and reports *every* problem it finds,
each with the file line and the component it belongs to, as one
:class:`HouseError`. Refused: TOML syntax errors, unknown top-level or
component keys, unknown component types, missing names/ports/fields,
duplicate names, values with the wrong unit or out of range, priors that do
not fit their parameter, ports that name nothing (or the wrong kind of
thing), sources that would heat a boundary, and nodes with no conductance
path to a boundary (their temperature would be unbounded, §4.4).

An optional ``[dataset]`` table declares what identification needs to know
about the house's history — signals, masks, exclusion windows (§6.1); it is
validated by :func:`esther.data.parse_dataset_config` and its problems are
reported in the same list.
"""

from __future__ import annotations

import difflib
import math
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.component import Component, FieldError, Port
from .core.params import Param, parse_prior
from .core.registry import Registry, RegistryError, default_registry
from .core.units import TIME, UnitError, parse_quantity
from .data.config import DatasetConfig, parse_dataset_config

__all__ = ["SCHEMA_VERSION", "House", "HouseError", "Issue", "load_house", "loads_house"]

SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = ("schema", "name", "step", "horizon", "component", "dataset")
_COMPONENT_BASE_KEYS = ("type", "name", "prior")


@dataclass(frozen=True)
class Issue:
    """One validation problem."""

    message: str
    line: int | None = None
    component: str | None = None  # e.g. "component #3 'stove' (Stove)"
    key: str | None = None

    def format(self, source: str) -> str:
        loc = f"{source}:{self.line}" if self.line else source
        parts = [loc]
        if self.component:
            parts.append(self.component)
        if self.key:
            parts.append(self.key)
        return ": ".join(parts) + f": {self.message}"


class HouseError(ValueError):
    """The house description is invalid. ``issues`` lists every problem found."""

    def __init__(self, issues: Iterable[Issue], source: str) -> None:
        self.issues = sorted(issues, key=lambda i: (i.line or 0, i.component or "", i.key or ""))
        self.source = source
        n = len(self.issues)
        lines = "\n".join("  " + i.format(source) for i in self.issues)
        super().__init__(f"{source}: {n} problem{'s' if n != 1 else ''} in house description\n{lines}")


@dataclass(frozen=True)
class House:
    """A validated house description."""

    name: str
    schema: int
    step: float  # s
    horizon: float  # s
    components: dict[str, Component]
    nodes: tuple[str, ...]
    boundaries: tuple[str, ...]
    signals: tuple[str, ...]
    source: str = "<string>"
    order: tuple[str, ...] = field(default=())
    dataset: DatasetConfig | None = None  # the [dataset] table (spec §6.1), if any

    @property
    def n_steps(self) -> int:
        return round(self.horizon / self.step)

    def __getitem__(self, name: str) -> Component:
        return self.components[name]

    def of_type(self, type_name: str) -> list[Component]:
        return [c for c in self.components.values() if c.type_name == type_name]


class _SourceMap:
    """Line numbers for keys. tomllib keeps no positions, so they are found in
    the text: the n-th ``[[component]]`` header, then the key inside its block.
    A key that cannot be located falls back to the block's header line."""

    _HEADER = re.compile(r"^\s*\[\[\s*component\s*\]\]")
    _TABLE = re.compile(r"^\s*\[")
    _OTHER = re.compile(r"^\s*\[(?!\[?\s*component\s*[\].])")

    def __init__(self, text: str) -> None:
        self.lines = text.splitlines()
        self.headers = [i + 1 for i, ln in enumerate(self.lines) if self._HEADER.match(ln)]

    def _find(self, key: str, start: int, end: int) -> int | None:
        dotted = r"\s*\.\s*".join(re.escape(p) for p in key.split("."))
        pat = re.compile(rf'^\s*(?:{dotted}|"{re.escape(key)}")\s*=')
        for i in range(start, min(end, len(self.lines))):
            if pat.match(self.lines[i]):
                return i + 1
        return None

    def table(self, dotted: str) -> int | None:
        """Line of a key under a ``[table]`` / ``[[table]]`` header: the most
        specific header that is a prefix of ``dotted``, then the key in it."""
        parts = re.sub(r"\[\d+\]", "", dotted).split(".")
        for n in range(len(parts), 0, -1):
            head = ".".join(parts[:n])
            pat = re.compile(rf"^\s*\[\[?\s*{re.escape(head)}\s*\]\]?")
            hits = [i for i, ln in enumerate(self.lines) if pat.match(ln)]
            if hits:
                start = hits[0] + 1
                end = next(
                    (i for i in range(start, len(self.lines)) if self._TABLE.match(self.lines[i])),
                    len(self.lines),
                )
                rest = ".".join(parts[n:])
                return (self._find(rest, start, end) if rest else None) or start
        return None

    def top(self, key: str | None = None) -> int | None:
        if key is None:
            return None
        end = next((i for i, ln in enumerate(self.lines) if self._TABLE.match(ln)), len(self.lines))
        return self._find(key, 0, end)

    def component(self, idx: int, key: str | None = None) -> int | None:
        if idx >= len(self.headers):
            return None  # e.g. components written as an inline array
        start = self.headers[idx]
        if key is None:
            return start
        end = self.headers[idx + 1] - 1 if idx + 1 < len(self.headers) else len(self.lines)
        # ...and a block ends at any other table, e.g. a [dataset] after the components.
        other = next((i for i in range(start, end) if self._OTHER.match(self.lines[i])), end)
        return self._find(key, start, other) or start


def _snake(type_name: str) -> str:
    last = type_name.rsplit(".", 1)[-1]
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", last).lower()


def _suggest(word: str, options: Iterable[str]) -> str:
    close = difflib.get_close_matches(word, list(options), n=3)
    return f" (did you mean {', '.join(repr(c) for c in close)}?)" if close else ""


@dataclass
class _Raw:
    idx: int
    table: dict[str, Any]
    cls: type[Component] | None = None
    name: str | None = None
    explicit_name: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    refs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Param] = field(default_factory=dict)
    ok: bool = True

    def label(self) -> str:
        t = self.table.get("type")
        t = t if isinstance(t, str) else "?"
        n = f" {self.name!r}" if self.name else ""
        return f"component #{self.idx + 1}{n} ({t})"


def load_house(path: str | Path, registry: Registry | None = None) -> House:
    """Load and validate a house TOML file. Raises :class:`HouseError`."""
    p = Path(path)
    return loads_house(p.read_text(encoding="utf-8"), registry=registry, source=str(p))


def loads_house(text: str, registry: Registry | None = None, source: str = "<string>") -> House:
    """Load and validate a house description from a string. Raises :class:`HouseError`."""
    reg = registry if registry is not None else default_registry()
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        m = re.search(r"at line (\d+)", str(e))
        raise HouseError([Issue(f"TOML syntax error: {e}", int(m[1]) if m else None)], source) from e
    return _Loader(doc, _SourceMap(text), reg, source).run()


class _Loader:
    def __init__(self, doc: dict[str, Any], smap: _SourceMap, reg: Registry, source: str) -> None:
        self.doc, self.smap, self.reg, self.source = doc, smap, reg, source
        self.issues: list[Issue] = []

    # -- reporting ---------------------------------------------------------
    def top_issue(self, key: str | None, msg: str) -> None:
        self.issues.append(Issue(msg, self.smap.top(key), None, key))

    def comp_issue(self, raw: _Raw, key: str | None, msg: str) -> None:
        raw.ok = False
        self.issues.append(Issue(msg, self.smap.component(raw.idx, key), raw.label(), key))

    # -- passes ------------------------------------------------------------
    def run(self) -> House:
        name, schema, step, horizon = self.top_level()
        dataset = self.dataset()
        raws = self.components()
        components = self.instantiate(raws)
        nodes, boundaries = self.graph(raws, components)
        self.resolve_ports(raws, components, nodes, boundaries)
        if not self.issues:
            self.reachability(raws, components, nodes, boundaries)
        for raw in raws:
            comp = components.get(raw.name or "")
            if comp is not None and raw.ok:
                for key, msg in comp.check():
                    self.comp_issue(raw, key, msg)
        if self.issues:
            raise HouseError(self.issues, self.source)
        signals: list[str] = []
        for c in components.values():
            signals.extend(s for s in c.signals() if s not in signals)
        return House(
            name=name,
            schema=schema,
            step=step,
            horizon=horizon,
            components=components,
            nodes=tuple(nodes),
            boundaries=tuple(boundaries),
            signals=tuple(signals),
            source=self.source,
            order=tuple(r.name for r in raws if r.name),
            dataset=dataset,
        )

    def top_level(self) -> tuple[str, int, float, float]:
        d = self.doc
        for k in sorted(set(d) - set(_TOP_LEVEL_KEYS)):
            self.top_issue(k, f"unknown top-level key {k!r}{_suggest(k, _TOP_LEVEL_KEYS)}")
        schema = d.get("schema")
        if schema is None:
            self.top_issue(None, f"missing 'schema' (this esther reads schema = {SCHEMA_VERSION})")
        elif schema != SCHEMA_VERSION or isinstance(schema, bool):
            self.top_issue("schema", f"schema {schema!r} not supported; this esther reads {SCHEMA_VERSION}")
        name = d.get("name")
        if not (isinstance(name, str) and name):
            self.top_issue("name" if "name" in d else None, "missing or empty house 'name'")
            name = "?"

        def duration(key: str, default: str) -> float:
            try:
                v = parse_quantity(d.get(key, default), TIME)
            except UnitError as e:
                self.top_issue(key, str(e))
                return math.nan
            if not v > 0:
                self.top_issue(key, f"must be a positive duration, got {d.get(key)!r}")
                return math.nan
            return v

        step = duration("step", "15min")
        horizon = duration("horizon", "36h")
        if math.isfinite(step) and math.isfinite(horizon):
            n = horizon / step
            if abs(n - round(n)) > 1e-9 or round(n) < 1:
                self.top_issue(
                    "horizon", f"horizon ({horizon:g} s) is not a whole number of steps ({step:g} s)"
                )
        return name, int(schema) if isinstance(schema, int) else 0, step, horizon

    def dataset(self) -> DatasetConfig | None:
        if "dataset" not in self.doc:
            return None
        cfg, problems = parse_dataset_config(self.doc["dataset"])
        for key, msg in problems:
            self.issues.append(Issue(msg, self.smap.table(key), None, key))
        return cfg

    def components(self) -> list[_Raw]:
        comps = self.doc.get("component")
        if comps is None:
            self.top_issue(None, "no components: a house needs at least one [[component]]")
            return []
        if not isinstance(comps, list) or not all(isinstance(c, dict) for c in comps):
            self.top_issue("component", "'component' must be an array of tables ([[component]])")
            return []
        raws = [_Raw(i, c) for i, c in enumerate(comps)]
        for raw in raws:
            self.parse_component(raw)
        self.assign_names(raws)
        return raws

    def parse_component(self, raw: _Raw) -> None:
        t = raw.table
        if "name" in t:
            if isinstance(t["name"], str) and Component.valid_name(t["name"]):
                raw.name, raw.explicit_name = t["name"], True
            else:
                self.comp_issue(
                    raw, "name", f"invalid name {t['name']!r}: letters, digits, '_' and '-', no dots"
                )
        typ = t.get("type")
        if not isinstance(typ, str):
            self.comp_issue(raw, None, "missing 'type'")
            return
        try:
            cls = self.reg.resolve(typ)
        except RegistryError as e:
            self.comp_issue(raw, "type", f"{e}{_suggest(typ, self.reg.names())}")
            return
        raw.cls = cls

        allowed = set(_COMPONENT_BASE_KEYS) | cls.declared_keys()
        for k in sorted(set(t) - allowed):
            self.comp_issue(raw, k, f"unknown key {k!r} for {typ}{_suggest(k, allowed)}")

        for f in cls.fields:
            if f.name in t:
                try:
                    raw.config[f.name] = f.parse(t[f.name])
                except FieldError as e:
                    self.comp_issue(raw, f.name, str(e))
            elif f.required:
                self.comp_issue(raw, None, f"missing required key {f.name!r}")
            else:
                raw.config[f.name] = f.default

        for port in cls.ports:
            self.parse_port(raw, port)

        declared = {p.name: p for p in cls.params}
        raw.params = dict(declared)
        priors = t.get("prior", {})
        if not isinstance(priors, dict):
            self.comp_issue(raw, "prior", "'prior' must be a table of parameter priors")
            return
        for pname, spec in priors.items():
            key = f"prior.{pname}"
            if pname not in declared:
                known = ", ".join(declared) or "none"
                self.comp_issue(
                    raw,
                    key,
                    f"{typ} has no parameter {pname!r} (its parameters: {known})" + _suggest(pname, declared),
                )
                continue
            try:
                raw.params[pname] = parse_prior(declared[pname], spec)
            except ValueError as e:
                self.comp_issue(raw, key, str(e))

    def parse_port(self, raw: _Raw, port: Port) -> None:
        t = raw.table
        if port.name not in t:
            if port.required:
                what = "the node it heats" if port.heats else f"a {port.kind}"
                self.comp_issue(raw, None, f"missing port {port.name!r} ({what})")
            else:
                raw.refs[port.name] = None
            return
        v = t[port.name]
        if port.count is None:
            if not (isinstance(v, str) and v):
                self.comp_issue(raw, port.name, f"must name a {port.kind}, got {v!r}")
                return
        else:
            if not (
                isinstance(v, list) and len(v) == port.count and all(isinstance(x, str) and x for x in v)
            ):
                self.comp_issue(raw, port.name, f"must be an array of {port.count} names, got {v!r}")
                return
        raw.refs[port.name] = v

    def assign_names(self, raws: list[_Raw]) -> None:
        owners: dict[str, _Raw] = {}
        for raw in raws:
            if raw.explicit_name:
                assert raw.name is not None
                if raw.name in owners:
                    first = owners[raw.name]
                    self.comp_issue(
                        raw, "name", f"duplicate name {raw.name!r}, already used by {first.label()}"
                    )
                else:
                    owners[raw.name] = raw
        counts: dict[str, int] = {}
        for raw in raws:
            if raw.explicit_name or raw.cls is None or "name" in raw.table:
                continue
            base = _snake(raw.cls.type_name)
            counts[base] = counts.get(base, 0) + 1
            candidate = base if counts[base] == 1 else f"{base}_{counts[base]}"
            if candidate in owners:
                self.comp_issue(
                    raw,
                    None,
                    f"unnamed, and its default name {candidate!r} is taken by {owners[candidate].label()};"
                    " give it a name",
                )
                continue
            raw.name = candidate
            owners[candidate] = raw

    def instantiate(self, raws: list[_Raw]) -> dict[str, Component]:
        out: dict[str, Component] = {}
        for raw in raws:
            if raw.ok and raw.cls is not None and raw.name is not None:
                out[raw.name] = raw.cls(raw.name, raw.config, raw.refs, raw.params)
        return out

    def graph(self, raws: list[_Raw], comps: dict[str, Component]) -> tuple[list[str], list[str]]:
        nodes: list[str] = []
        boundaries: list[str] = []
        owner: dict[str, str] = {}
        by_name = {r.name: r for r in raws if r.name}
        for name, c in comps.items():
            for n in c.nodes():
                if n in owner:
                    self.comp_issue(by_name[name], None, f"node {n!r} is already defined by {owner[n]!r}")
                else:
                    owner[n] = name
                    nodes.append(n)
            for b in c.boundaries():
                if b in owner:
                    self.comp_issue(by_name[name], None, f"boundary {b!r} is already defined by {owner[b]!r}")
                else:
                    owner[b] = name
                    boundaries.append(b)
        return nodes, boundaries

    def resolve_ports(
        self,
        raws: list[_Raw],
        comps: dict[str, Component],
        nodes: list[str],
        boundaries: list[str],
    ) -> None:
        node_set, bnd_set = set(nodes), set(boundaries)
        # A reference to a component that itself failed to load is not
        # reported again: the root cause is already in the list.
        failed = {r.table["name"] for r in raws if not r.ok and isinstance(r.table.get("name"), str)}
        failed |= {r.name for r in raws if not r.ok and r.name}
        for raw in raws:
            comp = comps.get(raw.name or "")
            if comp is None:
                continue
            for port in comp.ports:
                v = comp.refs.get(port.name)
                if v is None:
                    continue
                for ref in [v] if isinstance(v, str) else v:
                    if ref in failed or ref.split(".", 1)[0] in failed:
                        continue
                    msg = self.port_problem(port, ref, comps, node_set, bnd_set)
                    if msg:
                        self.comp_issue(raw, port.name, msg)

    def port_problem(
        self,
        port: Port,
        ref: str,
        comps: dict[str, Component],
        nodes: set[str],
        boundaries: set[str],
    ) -> str | None:
        if port.kind == "component":
            if ref not in comps:
                return f"no component named {ref!r}{_suggest(ref, comps)}"
            if port.component_type and comps[ref].type_name != port.component_type:
                return f"{ref!r} is a {comps[ref].type_name}, expected a {port.component_type}"
            return None
        want_node = port.kind in ("node", "thermal")
        want_bnd = port.kind in ("boundary", "thermal")
        if (ref in nodes and want_node) or (ref in boundaries and want_bnd):
            return None
        if ref in boundaries:
            if port.heats:
                return f"{ref!r} is a boundary; a source must heat a node, not a boundary"
            return f"{ref!r} is a boundary; expected a node"
        if ref in nodes:
            return f"{ref!r} is a node; expected a boundary"
        if ref in comps:
            owned = comps[ref].nodes()
            if owned:
                return f"{ref!r} is a component with several nodes; name one of: {', '.join(owned)}"
            return f"{ref!r} is a {comps[ref].type_name}, which has no thermal node"
        pool = (nodes if want_node else set()) | (boundaries if want_bnd else set())
        return f"no {port.kind} named {ref!r}{_suggest(ref, pool)}"

    def reachability(
        self,
        raws: list[_Raw],
        comps: dict[str, Component],
        nodes: list[str],
        boundaries: list[str],
    ) -> None:
        adj: dict[str, set[str]] = {n: set() for n in [*nodes, *boundaries]}
        for c in comps.values():
            for a, b in c.conductances():
                adj[a].add(b)
                adj[b].add(a)
        seen = set(boundaries)
        frontier = list(boundaries)
        while frontier:
            for nb in adj[frontier.pop()]:
                if nb not in seen:
                    seen.add(nb)
                    frontier.append(nb)
        by_name = {r.name: r for r in raws if r.name}
        for name, c in comps.items():
            for n in c.nodes():
                if n not in seen:
                    hint = (
                        "there is no boundary in the house (add an Ambient)"
                        if not boundaries
                        else ("add a Coupling to a boundary or to a coupled node")
                    )
                    self.comp_issue(
                        by_name[name],
                        None,
                        f"node {n!r} has no conductance path to a boundary, so its temperature "
                        f"would be unbounded; {hint}",
                    )
