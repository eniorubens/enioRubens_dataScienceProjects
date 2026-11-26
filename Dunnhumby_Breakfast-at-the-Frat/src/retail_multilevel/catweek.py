"""Sensitivity of F3 to common category x week shocks: F3 versus F3 + ``(1 | CATEGORY:WEEK)``.

Both models are REML fits on the *same* training rows (one R data frame is built once and passed to
both fits), with the same fixed part (F3_season) and the random part ``RANDOM``. The sensitivity adds a
random intercept for each category-week cell. The grouping factor is materialised as a ``CATWEEK``
string column (``"<CATEGORY>|<WEEK>"``) and converted to an R factor, which is equivalent to the lme4
term ``(1 | CATEGORY:WEEK)`` and keeps the formula free of interaction parsing. A category x week
intercept captures contemporaneous deviations only; the persistence of shocks between weeks is read from
the week-mean autocorrelation in :mod:`retail_multilevel.diagnostics`.

Everything except the fitting and the residual extraction is pure pandas; R is imported lazily.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from retail_multilevel import diagnostics
from retail_multilevel.fixedeffects import DEFAULT_FE_CONTROL, FE_MODELS, _with_random
from retail_multilevel.multilevel import LmerFit, icc, lr_test
from retail_multilevel.split import FIRST_TEST_WEEK

CATWEEK_COL = "CATWEEK"
CATWEEK_TERM = f"(1 | {CATWEEK_COL})"
KEY_TERMS = ["LOG_DISC", "LOG_BASE_DEV", "LOG_BASE_MEAN", "FEATURE", "DISPLAY", "TPR_ONLY", "TREND"]
REFERENCE = "F3_ref"
SENSITIVITY = "F3_catweek"
FALLBACK_CONTROL = 'lmerControl(optimizer="nloptwrap")'
SINGULAR_MESSAGE = "singular fit"


def reference_formula() -> str:
    """Return the F3 formula with the random part of the project (the C0 reference)."""
    return _with_random(FE_MODELS[-1][1])


def sensitivity_formula() -> str:
    """Return the F3 formula plus the category x week random intercept."""
    return f"{reference_formula()} + {CATWEEK_TERM}"


def add_catweek(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the ``CATWEEK`` grouping column (``"<CATEGORY>|<WEEK>"``).

    Parameters
    ----------
    df : pandas.DataFrame
        Rows with ``CATEGORY`` and ``WEEK``.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If any ``CATEGORY`` or ``WEEK`` is missing.
    """
    if df["CATEGORY"].isna().any() or df["WEEK"].isna().any():
        raise ValueError("CATEGORY and WEEK must not be missing")
    out = df.copy()
    out[CATWEEK_COL] = out["CATEGORY"].astype(str) + "|" + out["WEEK"].astype(int).astype(str)
    return out


def check_training(df: pd.DataFrame, first_test_week: int = FIRST_TEST_WEEK) -> None:
    """Raise if any row is outside the training window (``WEEK`` must be below ``first_test_week``)."""
    if int(df["WEEK"].max()) >= first_test_week:
        raise RuntimeError(f"only training weeks (< {first_test_week}) are allowed")


def counts(df: pd.DataFrame) -> dict:
    """Return the sizes of the estimation sample (rows, pairs, stores, UPCs, categories, cells, weeks)."""
    cw = df[CATWEEK_COL] if CATWEEK_COL in df.columns else add_catweek(df)[CATWEEK_COL]
    return {
        "n_rows": int(len(df)),
        "n_pairs": int(df["PAIR_ID"].nunique()),
        "n_stores": int(df["STORE_NUM"].nunique()),
        "n_upcs": int(df["UPC"].nunique()),
        "n_categories": int(df["CATEGORY"].nunique()),
        "n_catweek_levels": int(cw.nunique()),
        "n_weeks": int(df["WEEK"].nunique()),
        "max_week": int(df["WEEK"].max()),
        "n_missing_category": int(df["CATEGORY"].isna().sum()),
    }


def convergence_messages(fit: LmerFit) -> list[str]:
    """Return the fit's messages other than the singularity flag."""
    return [m for m in fit.messages if m != SINGULAR_MESSAGE]


def is_singular(fit: LmerFit) -> bool:
    """Return whether the fit was flagged singular."""
    return SINGULAR_MESSAGE in fit.messages


def fit_with_retry(
    r_df,
    formula: str,
    name: str,
    control: str | None = DEFAULT_FE_CONTROL,
    fitter: Callable[..., LmerFit] | None = None,
) -> tuple[LmerFit, list[dict]]:
    """Fit by REML; with convergence messages, refit once with ``nloptwrap`` and keep the cleaner fit.

    The retry replaces the first fit only if it has no convergence messages. A singular fit alone does
    not trigger a retry.

    Returns
    -------
    fit : LmerFit
        The fit kept.
    attempts : list of dict
        One row per attempt: ``model, optimizer, messages, singular, seconds, npar, nobs, loglik_reml,
        used``.
    """
    if fitter is None:
        from retail_multilevel.multilevel import fit_lmer as fitter
    first = fitter(r_df, formula, name, reml=True, control=control)
    fits = [("bobyqa", first)]
    kept = first
    if convergence_messages(first):
        second = fitter(r_df, formula, name, reml=True, control=FALLBACK_CONTROL)
        fits.append(("nloptwrap", second))
        if not convergence_messages(second):
            kept = second
    rows = [
        {
            "model": name,
            "optimizer": optimizer,
            "messages": " | ".join(f.messages),
            "singular": is_singular(f),
            "seconds": float(f.seconds),
            "npar": int(f.npar),
            "nobs": int(f.nobs),
            "loglik_reml": float(f.loglik),
            "used": f is kept,
        }
        for optimizer, f in fits
    ]
    return kept, rows


def fit_both(
    df: pd.DataFrame,
    control: str | None = DEFAULT_FE_CONTROL,
    frame_builder: Callable | None = None,
    fitter: Callable[..., LmerFit] | None = None,
) -> tuple[dict[str, LmerFit], pd.DataFrame, int]:
    """Fit the reference and the sensitivity model on one shared R frame (identical rows).

    Parameters
    ----------
    df : pandas.DataFrame
        Training rows with the F3 model columns plus ``CATEGORY`` and ``WEEK``.
    control : str, optional
        R expression for ``lmerControl`` of the first attempt.
    frame_builder, fitter : callable, optional
        Injection points for tests; default to :func:`retail_multilevel.multilevel.to_r_frame` and
        :func:`retail_multilevel.multilevel.fit_lmer`.

    Returns
    -------
    fits : dict of LmerFit
        Keys ``REFERENCE`` and ``SENSITIVITY``.
    convergence : pandas.DataFrame
        All attempts (see :func:`fit_with_retry`).
    n_rows : int
        Rows given to both fits.
    """
    from retail_multilevel.fixedeffects import _R_COLS

    if frame_builder is None:
        from retail_multilevel.multilevel import to_r_frame as frame_builder
    work = add_catweek(df)
    r_df = frame_builder(work[[*_R_COLS, CATWEEK_COL]], ["STORE_NUM", "UPC", CATWEEK_COL])
    fits, rows = {}, []
    for name, formula in ((REFERENCE, reference_formula()), (SENSITIVITY, sensitivity_formula())):
        fits[name], attempts = fit_with_retry(r_df, formula, name, control, fitter)
        rows += attempts
    return fits, pd.DataFrame(rows), len(work)


def conditional_residuals(fit: LmerFit, df: pd.DataFrame) -> pd.DataFrame:
    """Return ``diagnostics.ID_COLS`` of ``df`` plus ``resid`` (conditional, ``y - Xb - Zu``) of ``fit``."""
    from retail_multilevel.rbridge import get_robjects

    ro = get_robjects()
    cond = np.asarray(ro.r("function(m) as.numeric(resid(m))")(fit.r_model), dtype=float)
    if cond.size != len(df):
        raise RuntimeError("residual length differs from the number of rows")
    res = df[diagnostics.ID_COLS].copy().reset_index(drop=True)
    res["resid"] = cond
    return res


def fixef_compare(ref: LmerFit, sens: LmerFit, terms: list[str] = KEY_TERMS) -> pd.DataFrame:
    """Side-by-side fixed effects of the two fits.

    Returns
    -------
    pandas.DataFrame
        Columns ``term, est_ref, se_ref, est_catweek, se_catweek, diff, se_ratio`` (``diff`` is
        sensitivity minus reference; ``se_ratio`` is sensitivity SE over reference SE).
    """
    a = ref.fixef.set_index("term").loc[terms]
    b = sens.fixef.set_index("term").loc[terms]
    out = pd.DataFrame(
        {
            "term": terms,
            "est_ref": a["estimate"].to_numpy(),
            "se_ref": a["std_error"].to_numpy(),
            "est_catweek": b["estimate"].to_numpy(),
            "se_catweek": b["std_error"].to_numpy(),
        }
    )
    out["diff"] = out["est_catweek"] - out["est_ref"]
    out["se_ratio"] = out["se_catweek"] / out["se_ref"]
    return out


def varcomp_compare(fits: dict[str, LmerFit]) -> pd.DataFrame:
    """Variance components of every fit with each one's share of the total (``multilevel.icc``).

    Returns
    -------
    pandas.DataFrame
        Columns ``model, group, variance, sd, share``.
    """
    rows = []
    for name, fit in fits.items():
        shares = icc(fit.varcomp).set_index("group")["share"]
        v = fit.varcomp[["group", "variance", "sd"]].copy()
        v["share"] = v["group"].map(shares)
        v.insert(0, "model", name)
        rows.append(v)
    return pd.concat(rows, ignore_index=True)


def convergence_table(attempts: pd.DataFrame) -> pd.DataFrame:
    """Return the attempts table with the canonical column order."""
    cols = ["model", "optimizer", "messages", "singular", "seconds", "npar", "nobs", "loglik_reml", "used"]
    return attempts[cols].reset_index(drop=True)


def descriptive_lr(ref: LmerFit, sens: LmerFit) -> dict:
    """LR statistic of the added variance with the 50:50 chi2(0)/chi2(1) mixture p-value (descriptive).

    The fixed parts are identical and both fits are REML, so the comparison is valid; the p-value is
    the Self & Liang boundary mixture for one variance and is reported as a description, not as a
    significance claim.
    """
    return lr_test(ref, sens, boundary=True)


def residual_summaries(res_by_model: dict[str, pd.DataFrame], max_lag: int = 4) -> dict[str, pd.DataFrame]:
    """Lag correlations, week summaries and week shocks of the conditional residuals of every model.

    Returns
    -------
    dict of pandas.DataFrame
        ``lag_corr`` (``model, lag, corr, n``), ``week_summary`` (one row per model) and
        ``week_shocks`` (``model, WEEK, mean_resid, n, expected_sd``).
    """
    lags, summ, shocks = [], [], []
    for name, res in res_by_model.items():
        lc = diagnostics.lag_correlations(res, max_lag, "resid").rename(columns={"n_pairs_of_obs": "n"})
        lags.append(lc.assign(model=name))
        weeks, summary = diagnostics.week_shocks(res, "resid")
        summ.append({"model": name, **summary})
        shocks.append(weeks.assign(model=name))
    lag = pd.concat(lags, ignore_index=True)[["model", "lag", "corr", "n"]]
    wk = pd.concat(shocks, ignore_index=True)
    return {
        "lag_corr": lag,
        "week_summary": pd.DataFrame(summ),
        "week_shocks": wk[["model", *[c for c in wk.columns if c != "model"]]],
    }
