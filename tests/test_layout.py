"""Package layout per spec §3.1, and the io import boundary."""

import ast
import importlib
from pathlib import Path

import pytest

import esther

ROOT = Path(esther.__file__).parent
SUBPACKAGES = ["core", "components", "data", "identify", "forecast", "plan", "actuate", "backtest", "io"]
NO_IO = ["core", "components", "identify", "forecast", "plan", "actuate", "backtest"]


@pytest.mark.parametrize("name", [*SUBPACKAGES, "house"])
def test_module_exists(name):
    importlib.import_module(f"esther.{name}")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # The package a module's relative imports resolve against: for both
    # esther/plan/x.py and esther/plan/__init__.py it is esther.plan.
    package = list(path.relative_to(ROOT.parent).with_suffix("").parts[:-1])
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                mod = ".".join([*base, node.module] if node.module else base)
            else:
                mod = node.module or ""
            out.add(mod)
            out |= {f"{mod}.{a.name}" for a in node.names}
    return out


def test_import_scanner_sees_an_io_import():
    # Positive control: the scanner must flag an io import when there is one.
    fake = ROOT / "plan" / "_probe_for_test.py"
    fake.write_text("from ..io import homeostat\n")
    try:
        assert "esther.io" in _imports(fake)
    finally:
        fake.unlink()


@pytest.mark.parametrize("sub", NO_IO)
def test_no_io_imports(sub):
    for path in (ROOT / sub).rglob("*.py"):
        bad = {m for m in _imports(path) if m == "esther.io" or m.startswith("esther.io.")}
        assert not bad, f"{path.relative_to(ROOT.parent)} imports {sorted(bad)} (spec §3.1)"
