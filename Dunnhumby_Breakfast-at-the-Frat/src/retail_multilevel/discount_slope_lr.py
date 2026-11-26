"""Compare the nested C0 and C1 discount models on the same training rows.

The alternative adds two slope variances. Under the joint null both lie on the
boundary, so the usual one-component 50:50 mixture does not calibrate this test.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd

from retail_multilevel.features import PROMO_FLAGS, add_model_features, pair_means
from retail_multilevel.fixedeffects import DEFAULT_FE_CONTROL, RANDOM
from retail_multilevel.multilevel import LmerFit, fit_lmer, to_r_frame
from retail_multilevel.p1 import CANDIDATES, FIXED_FORMULA, FIXED_TERMS, RETRY_CONTROL, SINGULAR_MSG
from retail_multilevel.split import FIRST_TEST_WEEK

C0_FORMULA = f"{FIXED_FORMULA} + {RANDOM}"
_C1_SLOPES = next(terms for candidate, _, terms in CANDIDATES if candidate == "C1")
C1_FORMULA = f"{C0_FORMULA} + {_C1_SLOPES}"
_GROUPS = ["STORE_NUM", "UPC"]
_MODEL_VALUES = ["LOG_UNITS", *FIXED_TERMS]
_LR_ROUNDOFF = 1e-5


def _check_training(rows: pd.DataFrame) -> None:
    """Require a nonempty modelling sample in the original training window."""
    weeks = rows["WEEK"].to_numpy(dtype=float, na_value=np.nan)
    if not weeks.size or not np.all(np.isfinite(weeks) & (weeks >= 0) & (weeks < FIRST_TEST_WEEK)):
        raise ValueError(f"a nonempty sample from weeks 0 to {FIRST_TEST_WEEK - 1} is required")
    if not np.all(weeks == np.floor(weeks)):
        raise ValueError("WEEK must contain integer week indices")
    if "IN_MODEL" in rows and not rows["IN_MODEL"].eq(True).fillna(False).all():
        raise ValueError("all rows must belong to the IN_MODEL sample")


def _check_model_values(rows: pd.DataFrame) -> None:
    """Prevent silent omission of rows or creation of a missing-value factor level."""
    if rows[_GROUPS].isna().any().any():
        raise ValueError("STORE_NUM and UPC must be present on every training row")
    if not np.isfinite(rows[_MODEL_VALUES].to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("the response and all model predictors must be finite on every training row")


def prepare_training(train: pd.DataFrame) -> pd.DataFrame:
    """Create the F3 predictors using only the supplied eligible training rows.

    Subsets of the training window are allowed. Rows outside it are rejected before
    computing the pair means. Eligibility is checked when ``IN_MODEL`` is present.
    """
    _check_training(train)
    if train["PAIR_ID"].isna().any():
        raise ValueError("PAIR_ID must be present on every training row")
    if not train[PROMO_FLAGS].isin([0, 1]).all().all():
        raise ValueError("promotion indicators must contain only 0 or 1")
    work = add_model_features(train, pair_means(train))
    _check_model_values(work)
    return work


def _fit_ml(r_df, formula: str, name: str) -> LmerFit:
    """Fit one model and retry once when its optimiser reports nonconvergence."""
    fit = fit_lmer(r_df, formula, name, reml=False, control=DEFAULT_FE_CONTROL)
    if any(message != SINGULAR_MSG for message in fit.messages):
        fit = fit_lmer(r_df, formula, name, reml=False, control=RETRY_CONTROL)
        if any(message != SINGULAR_MSG for message in fit.messages):
            raise RuntimeError(f"{name} did not converge after retry: {' | '.join(fit.messages)}")
    return fit


def fit_observed(df: pd.DataFrame) -> tuple[LmerFit, LmerFit]:
    """Fit C0 and C1 by ML on one shared training design.

    A converged fit on the boundary is retained for the observed comparison.
    A singular C0 must be handled by the calibration caller before simulation.
    """
    _check_training(df)
    _check_model_values(df)
    r_df = to_r_frame(df[[*_GROUPS, *_MODEL_VALUES]], _GROUPS)
    c0 = _fit_ml(r_df, C0_FORMULA, "C0")
    c1 = _fit_ml(r_df, C1_FORMULA, "C1")
    if c0.nobs != len(df) or c1.nobs != len(df):
        raise RuntimeError("both models must include every row of their shared training design")
    likelihood_ratio(c0, c1)
    return c0, c1


def likelihood_ratio(c0: LmerFit, c1: LmerFit) -> float:
    """Return twice the ML log-likelihood gain of the nested C1 model over C0.

    Only a tiny negative difference from numerical roundoff is set to zero.
    A materially negative value signals an invalid fit comparison.
    """
    if c0.formula != C0_FORMULA or c1.formula != C1_FORMULA or c0.reml or c1.reml:
        raise ValueError("the comparison requires the declared C0 and C1 formulas fitted by ML")
    if c0.nobs <= 0 or c0.nobs != c1.nobs:
        raise ValueError("C0 and C1 must have the same positive number of observations")
    if not c0.fixef["term"].equals(c1.fixef["term"]) or c1.npar != c0.npar + 2:
        raise ValueError("C0 and C1 must share fixed effects and differ by two slope variance parameters")
    statistic = 2.0 * (c1.loglik - c0.loglik)
    if not isfinite(statistic) or statistic < -_LR_ROUNDOFF:
        raise ValueError(f"invalid nested likelihood-ratio statistic: {statistic}")
    return max(0.0, statistic)
