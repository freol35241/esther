"""The component library (spec §4.5).

Each module defines component classes; they are registered by name through the
``esther.components`` entry points in ``pyproject.toml`` — the same mechanism a
third-party package uses.

M0 status: these classes declare their configuration, ports and parameters
(enough for the house loader to validate a house) and their place in the
thermal graph. Their dynamics, decisions, constraints and costs arrive in
M2 (envelope, gains, radiators) and M3 (tanks, sources, wrappers).
"""
