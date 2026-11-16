"""Inferential diagnostics for the purchase-time OLS and logistic models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

DEFAULT_OLS_FORMULA = (
    "delivery_delay_days ~ promised_lead_days + total_price + "
    "C(customer_state, Treatment(\"SP\"))"
)
DEFAULT_LOGIT_FORMULA = (
    "sla_breach ~ promised_lead_days + total_price + "
    "C(customer_state, Treatment(\"SP\"))"
)


def fit_ols_inferential(
    frame: pd.DataFrame,
    formula: str = DEFAULT_OLS_FORMULA,
    cluster_col: str = "primary_seller_id",
) -> dict[str, object]:
    """Fit OLS with HC3 and seller-clustered covariance estimates."""
    model = smf.ols(formula=formula, data=frame).fit(cov_type="HC3")
    if cluster_col not in frame:
        raise KeyError(f"cluster column not found: {cluster_col}")
    clustered = smf.ols(formula=formula, data=frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame[cluster_col]}
    )
    return {"hc3": model, "clustered": clustered}


def coefficient_table(result: object) -> pd.DataFrame:
    """Return coefficients, uncertainty intervals, and p-values."""
    interval = result.conf_int()
    return pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient": result.params.to_numpy(),
            "std_error": result.bse.to_numpy(),
            "p_value": result.pvalues.to_numpy(),
            "ci_lower": interval.iloc[:, 0].to_numpy(),
            "ci_upper": interval.iloc[:, 1].to_numpy(),
        }
    )


def vif_table(result: object) -> pd.DataFrame:
    """Calculate VIF on the fitted design matrix, excluding the intercept."""
    design = result.model.exog
    names = result.model.exog_names
    rows = []
    for index, name in enumerate(names):
        if name == "Intercept":
            continue
        try:
            value = variance_inflation_factor(design, index)
        except (np.linalg.LinAlgError, ValueError):
            value = float("inf")
        rows.append({"term": name, "vif": float(value)})
    return pd.DataFrame(rows)


def ols_diagnostics(result: object) -> dict[str, object]:
    """Return residual, Breusch-Pagan, and Cook-distance diagnostics."""
    influence = OLSInfluence(result)
    residuals = pd.DataFrame(
        {
            "fitted": result.fittedvalues,
            "residual": result.resid,
            "studentized_residual": influence.resid_studentized_internal,
            "cooks_distance": influence.cooks_distance[0],
        }
    )
    bp = het_breuschpagan(result.resid, result.model.exog)
    return {
        "residuals": residuals,
        "breusch_pagan": {
            "lm_stat": float(bp[0]),
            "lm_p_value": float(bp[1]),
            "f_stat": float(bp[2]),
            "f_p_value": float(bp[3]),
        },
        "max_cooks_distance": float(residuals["cooks_distance"].max()),
    }


def standardized_ols(
    frame: pd.DataFrame,
    formula: str = DEFAULT_OLS_FORMULA,
    numeric_columns: tuple[str, ...] = (
        "promised_lead_days",
        "total_price",
    ),
) -> object:
    """Fit the same OLS after z-scoring continuous predictors."""
    data = frame.copy()
    for column in numeric_columns:
        mean = data[column].mean()
        std = data[column].std(ddof=0)
        data[column] = (data[column] - mean) / (std if std else 1.0)
    return smf.ols(formula=formula, data=data).fit(cov_type="HC3")


def fit_logit_inferential(
    frame: pd.DataFrame,
    formula: str = DEFAULT_LOGIT_FORMULA,
) -> tuple[object, pd.DataFrame]:
    """Fit a logistic model and return odds ratios with 95% intervals."""
    data = frame.copy()
    data["sla_breach"] = data["sla_breach"].astype(int)
    result = smf.logit(formula=formula, data=data).fit(disp=False)
    interval = result.conf_int()
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "odds_ratio": np.exp(result.params.to_numpy()),
            "ci_lower": np.exp(interval.iloc[:, 0].to_numpy()),
            "ci_upper": np.exp(interval.iloc[:, 1].to_numpy()),
            "p_value": result.pvalues.to_numpy(),
        }
    )
    return result, table
