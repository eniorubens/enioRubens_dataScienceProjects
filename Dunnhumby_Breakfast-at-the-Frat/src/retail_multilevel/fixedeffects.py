"""Fixed effects of price and promotion: step-up F0..F3 with the random part of the best null model.

All steps share the random part ``RANDOM`` (from notebook 02's best null model, M4) and differ in
the fixed part. The step-up compares ML fits (the fixed part changes, so REML is not comparable;
D14) with naive chi-square LR tests (fixed effects are not on a boundary). The final model F3 is
refitted by REML to report coefficients and variances. Fitting needs R (lme4 through rpy2).
"""

from __future__ import annotations

import pandas as pd

from retail_multilevel.features import FOURIER_TERMS
from retail_multilevel.multilevel import LmerFit, fit_lm, fit_lmer, fits_table, lr_test, to_r_frame
from retail_multilevel.nullmodels import DEFAULT_CONTROL

RANDOM = "(1 | STORE_NUM) + (1 | UPC) + (1 | STORE_NUM:UPC)"

_PRICE = "LOG_DISC + LOG_BASE_DEV + LOG_BASE_MEAN"
_PROMO = "FEATURE + DISPLAY + TPR_ONLY"
_SEASON = "TREND + " + " + ".join(FOURIER_TERMS(3))

FE_MODELS: list[tuple[str, str]] = [
    ("F0_null", "LOG_UNITS ~ 1"),
    ("F1_price", f"LOG_UNITS ~ 1 + {_PRICE}"),
    ("F2_promo", f"LOG_UNITS ~ 1 + {_PRICE} + {_PROMO}"),
    ("F3_season", f"LOG_UNITS ~ 1 + {_PRICE} + {_PROMO} + {_SEASON}"),
]

LR_PAIRS: list[tuple[str, str]] = [("F0_null", "F1_price"), ("F1_price", "F2_promo"), ("F2_promo", "F3_season")]

DEFAULT_FE_CONTROL = DEFAULT_CONTROL

_R_COLS = [
    "LOG_UNITS", "STORE_NUM", "UPC", "LOG_DISC", "LOG_BASE_DEV", "LOG_BASE_MEAN",
    "FEATURE", "DISPLAY", "TPR_ONLY", "TREND", *FOURIER_TERMS(3),
]  # fmt: skip


def _stack(frames: dict[str, pd.DataFrame], col: str = "model") -> pd.DataFrame:
    out = pd.concat([d.assign(**{col: name}) for name, d in frames.items()], ignore_index=True)
    return out[[col, *[c for c in out.columns if c != col]]]


def _with_random(formula: str) -> str:
    return f"{formula} + {RANDOM}"


def fixef_with_ci(fixef: pd.DataFrame) -> pd.DataFrame:
    """Add ``z = estimate / std_error`` and 95% Wald limits ``ci_low, ci_high`` (estimate +- 1.96 se)."""
    out = fixef.copy()
    out["z"] = out["estimate"] / out["std_error"]
    out["ci_low"] = out["estimate"] - 1.96 * out["std_error"]
    out["ci_high"] = out["estimate"] + 1.96 * out["std_error"]
    return out


def pseudo_r2(var_null: pd.DataFrame, var_final: pd.DataFrame) -> pd.DataFrame:
    """Proportional reduction of each variance component from the null to the final model.

    ``reduction = 1 - var_final / var_null`` (multilevel pseudo-R2 per level; Raudenbush & Bryk).
    It can be negative when the final model has a larger variance at that level.

    Parameters
    ----------
    var_null, var_final : pandas.DataFrame
        ``varcomp`` tables (``group, term, variance, sd``) of the REML null and final fits.

    Returns
    -------
    pandas.DataFrame
        Columns ``group, var_null, var_final, reduction``.
    """
    a = var_null[["group", "variance"]].rename(columns={"variance": "var_null"})
    b = var_final[["group", "variance"]].rename(columns={"variance": "var_final"})
    out = a.merge(b, on="group", how="inner")
    out["reduction"] = 1 - out["var_final"] / out["var_null"]
    return out


def lr_rows(fits: list[LmerFit], pairs: list[tuple[str, str]] = LR_PAIRS) -> pd.DataFrame:
    """LR tests of the nested fixed-effect steps with the naive chi-square p-value (``boundary=False``)."""
    by_name = {f.name: f for f in fits}
    rows = [lr_test(by_name[r], by_name[f], boundary=False) for r, f in pairs]
    return pd.DataFrame(rows, columns=["restricted", "full", "stat", "df", "p_value", "p_naive"])


def fit_f3_reml(df: pd.DataFrame, control: str | None = DEFAULT_FE_CONTROL) -> dict[str, pd.DataFrame]:
    """Fit only the final model F3 by REML (used by sensitivity runs on a row subset).

    Parameters
    ----------
    df : pandas.DataFrame
        Rows with the columns required by :func:`fit_fixed_effects`.
    control : str, optional
        R expression for ``lmerControl``.

    Returns
    -------
    dict of pandas.DataFrame
        ``fixef`` (``term, estimate, std_error, z, ci_low, ci_high``), ``varcomp_final`` and ``fits``
        (one-row ``fits_table`` with ``nobs``, log-likelihood, messages).
    """
    r_df = to_r_frame(df[_R_COLS], ["STORE_NUM", "UPC"])
    f3 = fit_lmer(r_df, _with_random(FE_MODELS[-1][1]), "F3_season", reml=True, control=control)
    return {"fixef": fixef_with_ci(f3.fixef), "varcomp_final": f3.varcomp, "fits": fits_table([f3])}


def fit_fixed_effects(df: pd.DataFrame, control: str | None = DEFAULT_FE_CONTROL) -> dict[str, pd.DataFrame]:
    """Fit the fixed-effects step-up and the comparison OLS models.

    Parameters
    ----------
    df : pandas.DataFrame
        Training rows with the columns of :func:`retail_multilevel.features.add_model_features`
        plus ``LOG_UNITS, STORE_NUM, UPC`` (no missing values in the model columns).
    control : str, optional
        R expression for ``lmerControl`` (default bobyqa).

    Returns
    -------
    dict of pandas.DataFrame
        ``fits`` (ML fits F0..F3), ``lr`` (naive chi-square LR tests, columns ``restricted, full,
        stat, df, p_value, p_naive``), ``fixef`` (F3 by REML: ``term, estimate, std_error, z, ci_low,
        ci_high``), ``varcomp_final`` (F3 REML), ``varcomp_steps`` (ML variances, leading ``model``),
        ``pseudo_r2`` (``group, var_null, var_final, reduction``; F0 vs F3, both REML) and
        ``ols_pooled`` (``model, term, estimate, std_error`` for plain OLS and OLS with UPC fixed
        effects, UPC dummies omitted).
    """
    r_df = to_r_frame(df[_R_COLS], ["STORE_NUM", "UPC"])
    fits = [fit_lmer(r_df, _with_random(f), name, reml=False, control=control) for name, f in FE_MODELS]
    f3_reml = fit_lmer(r_df, _with_random(FE_MODELS[-1][1]), "F3_season", reml=True, control=control)
    f0_reml = fit_lmer(r_df, _with_random(FE_MODELS[0][1]), "F0_null", reml=True, control=control)

    fixed3 = FE_MODELS[-1][1]
    ols = fit_lm(r_df, fixed3, "ols_pooled", reml=False)
    ols_upc = fit_lm(r_df, fixed3 + " + factor(UPC)", "ols_upc_fe", reml=False)
    ols_tab = _stack(
        {
            "ols_pooled": ols.fixef,
            "ols_upc_fe": ols_upc.fixef[~ols_upc.fixef["term"].str.startswith("factor(UPC)")],
        }
    )
    return {
        "fits": fits_table(fits),
        "lr": lr_rows(fits),
        "fixef": fixef_with_ci(f3_reml.fixef),
        "varcomp_final": f3_reml.varcomp,
        "varcomp_steps": _stack({f.name: f.varcomp for f in fits}),
        "pseudo_r2": pseudo_r2(f0_reml.varcomp, f3_reml.varcomp),
        "ols_pooled": ols_tab.reset_index(drop=True),
    }
