"""Component type registry (spec §5).

Types are resolved by name. Every type — esther's own included — is
registered through the ``esther.components`` entry-point group, so a
third-party package adds one in its own ``pyproject.toml``::

    [project.entry-points."esther.components"]
    "acme.Boiler" = "acme_esther.boiler:Boiler"

and nothing in esther changes. Entry points are loaded lazily, on first use of
the name, so one broken plug-in does not stop houses that do not use it.

A :class:`Registry` can also be populated directly (``register``), which tests
and notebooks use.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

from .component import Component

__all__ = ["ENTRY_POINT_GROUP", "Registry", "RegistryError", "default_registry"]

ENTRY_POINT_GROUP = "esther.components"


class RegistryError(LookupError):
    pass


class Registry:
    def __init__(self, *, use_entry_points: bool = True) -> None:
        self._classes: dict[str, type[Component]] = {}
        self._entry_points: dict[str, list[EntryPoint]] | None = None
        self._use_entry_points = use_entry_points

    def register(self, cls: type[Component], name: str | None = None) -> type[Component]:
        name = name or cls.type_name or cls.__name__
        if not (isinstance(cls, type) and issubclass(cls, Component)):
            raise RegistryError(f"{cls!r} is not a Component subclass")
        if name in self._classes and self._classes[name] is not cls:
            raise RegistryError(f"component type {name!r} is already registered")
        cls.type_name = name
        self._classes[name] = cls
        return cls

    def _eps(self) -> dict[str, list[EntryPoint]]:
        if self._entry_points is None:
            found: dict[str, list[EntryPoint]] = {}
            if self._use_entry_points:
                for ep in entry_points(group=ENTRY_POINT_GROUP):
                    found.setdefault(ep.name, []).append(ep)
            self._entry_points = found
        return self._entry_points

    def names(self) -> list[str]:
        return sorted(set(self._classes) | set(self._eps()))

    def __contains__(self, name: str) -> bool:
        return name in self._classes or name in self._eps()

    def resolve(self, name: str) -> type[Component]:
        if name in self._classes:
            return self._classes[name]
        eps = self._eps().get(name)
        if not eps:
            raise RegistryError(f"unknown component type {name!r}")
        targets = {ep.value for ep in eps}
        if len(targets) > 1:
            raise RegistryError(
                f"component type {name!r} is registered more than once: {', '.join(sorted(targets))}"
            )
        try:
            cls = eps[0].load()
        except Exception as e:
            raise RegistryError(f"component type {name!r} ({eps[0].value}) failed to load: {e}") from e
        if not (isinstance(cls, type) and issubclass(cls, Component)):
            raise RegistryError(f"entry point {name!r} ({eps[0].value}) is not a Component subclass")
        return self.register(cls, name)


_default: Registry | None = None


def default_registry() -> Registry:
    """The process-wide registry, backed by installed entry points."""
    global _default
    if _default is None:
        _default = Registry()
    return _default
