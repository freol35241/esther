"""A validation report in Markdown (spec §6.5, F4): the fit, its uncertainty
and identifiability, the held-out metrics against persistence, the residual
diagnostics, and the gate's decision with its numbers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .fit import FitResult
from .validate import GateDecision, Validation

__all__ = ["comparison_table", "report_markdown"]


def _cell(v, floatfmt: str) -> str:
    if isinstance(v, bool | np.bool_):
        return "yes" if v else "no"
    if isinstance(v, int | np.integer):
        return str(int(v))
    if isinstance(v, float | np.floating):
        return "" if not np.isfinite(v) else format(float(v), floatfmt)
    return str(v)


def _md(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    """A pandas frame as a Markdown table, keeping each column's own type."""
    cols = [str(c) for c in df.columns]
    idx_name = ", ".join(str(n) for n in df.index.names if n is not None) or ""
    lines = ["| " + " | ".join([idx_name, *cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for i in range(len(df)):
        key = df.index[i]
        k = ", ".join(_cell(x, "g") for x in (key if isinstance(key, tuple) else (key,)))
        cells = [_cell(df[c].iloc[i], floatfmt) for c in df.columns]
        lines.append("| " + " | ".join([k, *cells]) + " |")
    return "\n".join(lines)


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def report_markdown(
    fit: FitResult,
    validation: Validation,
    gate: GateDecision | None = None,
    title: str = "Fit and validation",
) -> str:
    """The full report for one fit."""
    out = [f"## {title}", "", "### Model", "", "```", fit.model.describe(), "```", ""]
    tol = 1e-3 * max(1.0, abs(fit.objective))
    n_best = int((np.abs(fit.starts.objective - fit.objective) < tol).sum())
    out += [
        "### Fit",
        "",
        f"MAP over {len(fit.names)} free parameters, {fit.n_obs} observations; "
        f"negative log-posterior {fit.objective:.2f} (data term {fit.nll:.2f}); "
        f"{len(fit.starts)} starts, {n_best} "
        f"reached the best value; {fit.n_evals} filter evaluations.",
        "",
    ]
    tab = fit.table().copy()
    tab["value"] = tab["value"].map(_fmt)
    tab["95 % interval"] = [f"{_fmt(a)} … {_fmt(b)}" for a, b in zip(fit.lo95, fit.hi95, strict=True)]
    tab["identified"] = ["yes" if v else "**no**" for v in tab["identified"]]
    tab["on_bound"] = ["**yes**" if v else "no" for v in tab["on_bound"]]
    tab = tab[["unit", "value", "95 % interval", "prior", "sd_ratio", "weak_share", "identified", "on_bound"]]
    out += [_md(tab, ".3g"), ""]
    out += [
        f"Hessian {'positive definite' if fit.hessian_ok else '**not positive definite**'}, "
        f"condition number {fit.condition:.3g}. `sd_ratio` is the posterior sd over the prior's "
        "in log space (log-normal priors) or natural space (normal priors); `weak_share` the share of "
        "the posterior variance from directions the data barely inform.",
        "",
    ]
    corr = fit.high_correlations()
    if corr:
        out += [
            "Pairwise correlations at or above "
            f"{fit.options.correlation_flag:g}: " + "; ".join(f"{a} / {b} {r:+.3f}" for a, b, r in corr),
            "",
        ]
    for lam, load in fit.weak_directions:
        out += [
            "Direction not identified by the data (information "
            f"{lam:.3g}): " + ", ".join(f"{k} {v:+.2f}" for k, v in load.items()),
            "",
        ]
    a, b = validation.span
    out += [
        "### Held-out validation",
        "",
        f"{validation.n_steps} steps from {a} to {b}; windows every {validation.stride / 3600:g} h "
        f"({len(validation.windows)} windows).",
        "",
    ]
    ol = validation.open_loop[["rmse", "end_rmse", "bias", "n_windows"]].copy()
    ol["persistence_rmse"] = validation.persistence["rmse"]
    ol["persistence_end_rmse"] = validation.persistence["end_rmse"]
    out += ["Open-loop simulation from the filter's state (K):", "", _md(ol, ".3f"), ""]
    out += ["One-step innovations:", "", _md(validation.innovation, ".4g"), ""]
    out += [
        "Ljung–Box on the standardised innovations (reported, not gating):",
        "",
        _md(validation.ljung_box, ".3g"),
        "",
    ]
    bias = validation.bias_by_hour
    if not bias.empty:
        summ = pd.DataFrame(
            {"max |bias| (K)": bias.abs().max(), "hour of max": bias.abs().idxmax().astype(int)}
        )
        out += [f"Bias by hour of day ({validation.tz}), summary:", "", _md(summ, ".3f"), ""]
    if gate is not None:
        out += ["### Acceptance gate", "", "```", gate.summary(), "```", ""]
    return "\n".join(out)


def comparison_table(
    entries: Sequence[tuple[str, FitResult | None, Validation]],
    output: str | None = None,
) -> pd.DataFrame:
    """One row per candidate structure: size, fit, and held-out RMSE at each horizon."""
    rows = []
    for name, fit, val in entries:
        out = output or val.outputs[0]
        row = {"structure": name}
        if fit is not None:
            row.update(
                params=len(fit.names),
                nll=fit.nll,
                not_identified=int((~fit.identified).sum()),
                on_bound=int(fit.on_bound.sum()),
            )
        for h in val.horizons:
            row[f"rmse_{h / 3600:g}h"] = val.rmse(out, h)
        row["innov_rmse"] = float(val.innovation.loc[out, "rmse"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("structure")
