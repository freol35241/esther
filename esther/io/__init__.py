"""OPTIONAL readers — extras only, e.g. :mod:`esther.io.homeostat`.

Nothing in core, components, identify, forecast, plan, actuate or backtest may
import from here (spec §3.1); tests/test_layout.py enforces it. Readers are not
imported by this package: ``from esther.io.homeostat import HomeostatHistory``."""
