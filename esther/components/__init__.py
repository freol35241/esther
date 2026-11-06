"""The component library (spec §4.5).

Each module defines component classes; they are registered by name through the
``esther.components`` entry points in ``pyproject.toml`` — the same mechanism a
third-party package uses.

Status: every class declares its configuration, ports and parameters (enough
for the house loader to validate a house) and its place in the thermal graph.
The envelope (``Zone``, ``Mass``, ``Coupling``, ``Ambient``, ``Ventilation``,
``Sensor``), the gains (``SolarGain``, ``InternalGain``, ``HeatInput``) and
``Radiators`` also contribute their dynamics (M2). Tanks, sources and wrappers
follow in M3; decisions, constraints and costs in M4.
"""
