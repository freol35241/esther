"""Identification (spec §6): Kalman filter, likelihood, MAP fit, validation.

- :class:`IdData` — contiguous segments of the signals a model reads.
- :func:`kalman` — the batched continuous-discrete Kalman filter.
- :func:`fit` — MAP over the priors, multi-start L-BFGS-B, Hessian intervals,
  identifiability.
- :func:`validate` and :func:`acceptance_gate` — held-out open-loop metrics
  and the §6.5 gate; :func:`report_markdown` writes them up.
- :func:`simulate` — synthetic data from a model (§10.1).
- :func:`estimate_state` — the live state estimate: the filter over a
  recent window (§6.6).
- :class:`FittedModel`, :func:`save_model`, :func:`load_model` — a fit as
  the file a live planner loads (§9).
- :func:`detect_draws`, :func:`fit_with_draws`, :func:`validate_draws` —
  hot-water draws from the tank's temperature signature (§6.3).
"""

from .data import IdData
from .draws import (
    DrawFit,
    DrawOptions,
    DrawValidation,
    detect_draws,
    draw_series,
    fit_with_draws,
    inject_response,
    validate_draws,
)
from .fit import FitOptions, FitResult, fit
from .kalman import FilterOptions, kalman, open_loop, prepare
from .model_file import (
    FittedModel,
    ModelFileError,
    fitted_model,
    gate_metrics,
    house_sha256,
    load_model,
    model_version,
    save_model,
)
from .report import comparison_table, report_markdown
from .simulate import simulate
from .state import StateError, StateEstimate, estimate_state
from .validate import GateDecision, Validation, acceptance_gate, ljung_box, validate

__all__ = [
    "DrawFit",
    "DrawOptions",
    "DrawValidation",
    "FilterOptions",
    "FitOptions",
    "FitResult",
    "FittedModel",
    "GateDecision",
    "IdData",
    "ModelFileError",
    "StateError",
    "StateEstimate",
    "Validation",
    "acceptance_gate",
    "comparison_table",
    "detect_draws",
    "draw_series",
    "estimate_state",
    "fit",
    "fit_with_draws",
    "fitted_model",
    "gate_metrics",
    "house_sha256",
    "inject_response",
    "kalman",
    "ljung_box",
    "load_model",
    "model_version",
    "open_loop",
    "prepare",
    "report_markdown",
    "save_model",
    "simulate",
    "validate",
    "validate_draws",
]
