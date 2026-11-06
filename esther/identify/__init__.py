"""Identification (spec §6): Kalman filter, likelihood, MAP fit, validation.

- :class:`IdData` — contiguous segments of the signals a model reads.
- :func:`kalman` — the batched continuous-discrete Kalman filter.
- :func:`fit` — MAP over the priors, multi-start L-BFGS-B, Hessian intervals,
  identifiability.
- :func:`validate` and :func:`acceptance_gate` — held-out open-loop metrics
  and the §6.5 gate; :func:`report_markdown` writes them up.
- :func:`simulate` — synthetic data from a model (§10.1).
"""

from .data import IdData
from .fit import FitOptions, FitResult, fit
from .kalman import FilterOptions, kalman, open_loop, prepare
from .report import comparison_table, report_markdown
from .simulate import simulate
from .validate import GateDecision, Validation, acceptance_gate, ljung_box, validate

__all__ = [
    "FilterOptions",
    "FitOptions",
    "FitResult",
    "GateDecision",
    "IdData",
    "Validation",
    "acceptance_gate",
    "comparison_table",
    "fit",
    "kalman",
    "ljung_box",
    "open_loop",
    "prepare",
    "report_markdown",
    "simulate",
    "validate",
]
