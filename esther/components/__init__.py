"""The component library (spec §4.5).

Each module defines component classes; they are registered by name through the
``esther.components`` entry points in ``pyproject.toml`` — the same mechanism a
third-party package uses.

Status: every class declares its configuration, ports and parameters (enough
for the house loader to validate a house) and its place in the thermal graph.
The envelope (``Zone``, ``Mass``, ``Coupling``, ``Ambient``, ``Ventilation``,
``Sensor``), the gains (``SolarGain``, ``InternalGain``, ``HeatInput``) and
``Radiators`` contribute their dynamics since M2; the tanks (``JacketTank``,
``StratifiedTank``), the sources (``HeatPump``, ``ElectricHeater``,
``Stove``) and ``HotWaterDraw`` since M3, as measured inputs for
identification and as heat-rate decisions for planning. The controller
wrappers (``CurveControlledHeatSource``, ``ThermostatChargedTank``,
``ivt490.Plant``) model a device's own logic and its translations (M3).
Integer decisions, constraints and costs arrive with the planner in M4.
"""
