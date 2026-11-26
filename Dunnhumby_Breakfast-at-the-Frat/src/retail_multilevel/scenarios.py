"""Pricing and promotion scenarios of the reference model F3 (REML), with joint covariance and observed support.

The scenario contrast against a regular week is a linear combination of the fixed effects of F3
(:func:`retail_multilevel.interpret.scenario_contrast`). This module refits F3 exactly as the stored
``fixed_effects`` run did, extracts the joint covariance matrix ``vcov(m)`` from lme4, and counts the
training rows that realise each scenario (exact modality combination, discount within a window).
Only the training weeks are used. Fitting needs R (lme4 through rpy2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_multilevel.fixedeffects import _R_COLS, FE_MODELS, RANDOM
from retail_multilevel.interpret import Scenario
from retail_multilevel.multilevel import LmerFit, _ro, fit_lmer, to_r_frame
from retail_multilevel.nullmodels import DEFAULT_CONTROL
from retail_multilevel.split import FIRST_TEST_WEEK

SUPPORT_TOL = 0.03
MAX_TRAIN_WEEK = FIRST_TEST_WEEK - 1  # 103


def assert_training_only(df: pd.DataFrame) -> None:
    """Raise if any row is from a test week (``WEEK`` above 103)."""
    if len(df) and int(df["WEEK"].max()) > MAX_TRAIN_WEEK:
        raise ValueError(f"rows from the test period present (max WEEK {int(df['WEEK'].max())} > {MAX_TRAIN_WEEK})")


def vcov_from_r(fit: LmerFit) -> pd.DataFrame:
    """Joint covariance matrix of the fixed effects, ``as.matrix(vcov(m))`` of the lme4 fit.

    Parameters
    ----------
    fit : LmerFit
        Fit with an R model handle.

    Returns
    -------
    pandas.DataFrame
        Square frame indexed and labelled by fixed-effect term.
    """
    ro = _ro()
    res = ro.r(
        """function(m) {
          v <- as.matrix(vcov(m))
          list(terms = rownames(v), values = as.vector(v), n = nrow(v))
        }"""
    )(fit.r_model)
    terms = [str(s) for s in res.rx2("terms")]
    n = int(res.rx2("n")[0])
    values = np.asarray(list(res.rx2("values")), dtype=float).reshape(n, n, order="F")
    return pd.DataFrame(values, index=terms, columns=terms)


def fit_f3_with_vcov(df: pd.DataFrame, control: str | None = DEFAULT_CONTROL) -> tuple[LmerFit, pd.DataFrame]:
    """Refit F3 by REML (same formula, columns and control as the fixed-effects run) and return its vcov.

    Parameters
    ----------
    df : pandas.DataFrame
        Training rows with the model features (``WEEK`` up to 103 only).
    control : str, optional
        R expression for ``lmerControl``.

    Returns
    -------
    tuple
        The :class:`~retail_multilevel.multilevel.LmerFit` and the covariance frame of the fixed effects.
    """
    assert_training_only(df)
    r_df = to_r_frame(df[_R_COLS], ["STORE_NUM", "UPC"])
    name, fixed = FE_MODELS[-1]
    fit = fit_lmer(r_df, f"{fixed} + {RANDOM}", name, reml=True, control=control)
    return fit, vcov_from_r(fit)


def support_table(
    df: pd.DataFrame, scenarios: list[Scenario], tol: float = SUPPORT_TOL, group: str | None = None
) -> pd.DataFrame:
    """Observed support of each scenario in the training rows.

    A row supports a scenario when ``FEATURE``, ``DISPLAY`` and ``TPR_ONLY`` all equal the scenario's
    (exact modality combination) and ``DISCOUNT = 1 - PRICE / BASE_PRICE`` lies within ``tol`` of the
    scenario's discount (boundary included). ``share`` is relative to all rows of ``df`` (or of the group);
    the ``*_observed_base`` columns restrict support to rows with an observed base price (``BASE_IMPUTED``
    false), with ``share_observed_base`` relative to the observed-base rows. ``n_rows_fallback`` counts the
    support rows whose base price fell back to the price (``BASE_FALLBACK``). A scenario without support
    rows has ``outside_support = True`` and stays in the table.

    Parameters
    ----------
    df : pandas.DataFrame
        Training rows with ``DISCOUNT, FEATURE, DISPLAY, TPR_ONLY, PAIR_ID, WEEK, BASE_IMPUTED, BASE_FALLBACK``.
    scenarios : list of tuple
        ``(name, discount_fraction, feature, display, tpr)``.
    tol : float
        Half-width of the discount window (fraction of the base price).
    group : str, optional
        Column to cut by; the support is then counted within each level (the model itself has common slopes).

    Returns
    -------
    pandas.DataFrame
        Columns ``[group,] scenario, n_rows, n_pairs, share, n_rows_observed_base, n_pairs_observed_base,
        share_observed_base, n_rows_fallback, outside_support``.
    """
    assert_training_only(df)
    if group is not None:
        parts = [support_table(g, scenarios, tol).assign(**{group: key}) for key, g in df.groupby(group, sort=True)]
        out = pd.concat(parts, ignore_index=True)
        return out[[group, *[c for c in out.columns if c != group]]]
    disc = df["DISCOUNT"].to_numpy(dtype=float)
    flags = {c: df[c].to_numpy().astype(bool) for c in ("FEATURE", "DISPLAY", "TPR_ONLY")}
    observed = ~df["BASE_IMPUTED"].to_numpy().astype(bool)
    fallback = df["BASE_FALLBACK"].to_numpy().astype(bool)
    pairs = df["PAIR_ID"].to_numpy()
    n_all, n_obs = len(df), int(observed.sum())
    rows = []
    for name, discount, feature, display, tpr in scenarios:
        mask = (
            (flags["FEATURE"] == bool(feature))
            & (flags["DISPLAY"] == bool(display))
            & (flags["TPR_ONLY"] == bool(tpr))
            & (np.abs(disc - discount) <= tol + 1e-12)
        )
        m_obs = mask & observed
        rows.append(
            {
                "scenario": name,
                "n_rows": int(mask.sum()),
                "n_pairs": int(pd.Series(pairs[mask]).nunique()),
                "share": float(mask.sum()) / max(n_all, 1),
                "n_rows_observed_base": int(m_obs.sum()),
                "n_pairs_observed_base": int(pd.Series(pairs[m_obs]).nunique()),
                "share_observed_base": float(m_obs.sum()) / max(n_obs, 1),
                "n_rows_fallback": int((mask & fallback).sum()),
                "outside_support": not bool(mask.any()),
            }
        )
    return pd.DataFrame(rows)
