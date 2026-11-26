"""Per-pair OLS discount slopes on the training weeks (evidence for partial pooling).

The specification is the one of the P1 per-pair OLS, ``LOG_UNITS ~ LOG_DISC + FEATURE + DISPLAY + TPR_ONLY``
fitted separately for each store x product pair with at least :data:`retail_multilevel.p1.MIN_WEEKS_OLS` weeks.
The fitting itself is :func:`retail_multilevel.p1.pair_ols_fit`, including its per-term coverage handling: a
term without variation in the pair, or collinear with the previous ones, is not estimated. Pairs whose
``LOG_DISC`` is not estimated therefore have no slope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_multilevel import p1
from retail_multilevel.features import add_model_features, pair_means, upc_means

SUMMARY_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
ALL = "all"
POSITIVE_TOL = 1e-9
SUMMARY_COLUMNS = ["stratum", "n_pairs", "n_positive", "share_positive_pct", "q05", "q25", "median", "q75", "q95"]


def pair_slopes(train: pd.DataFrame, min_weeks: int = p1.MIN_WEEKS_OLS) -> pd.DataFrame:
    """Fit one OLS per pair and return the ``LOG_DISC`` slope with the history stratum.

    The model features are built on ``train`` itself, as the fit window of P1 does, and then
    :func:`retail_multilevel.p1.pair_ols_fit` is applied with the terms ``LOG_DISC, FEATURE, DISPLAY, TPR_ONLY``.

    Parameters
    ----------
    train : pandas.DataFrame
        Training rows with ``PAIR_ID, UPC, WEEK, WEEK_OF_YEAR, LOG_UNITS, LOG_PRICE, LOG_BASE_PRICE`` and the
        promotion flags.
    min_weeks : int
        Minimum number of weeks of a pair to be fitted.

    Returns
    -------
    pandas.DataFrame
        One row per pair of ``train``: ``PAIR_ID``, ``n_weeks``, ``has_ols`` (fitted, at least ``min_weeks``
        weeks), ``slope`` (``LOG_DISC`` coefficient; ``NaN`` when the pair was not fitted or its ``LOG_DISC``
        was not estimated) and ``stratum`` (``<26``, ``26-77`` or ``>=78`` weeks of history).
    """
    feats = add_model_features(train, pair_means(train), upc_means(train)).reset_index(drop=True)
    info, coef, est = p1.pair_ols_fit(feats, p1.OLS_TERMS, min_weeks)
    out = info.copy()
    out["slope"] = np.where(est[:, 1], coef[:, 1], np.nan)
    out["stratum"] = p1.history_stratum(out["n_weeks"]).to_numpy()
    return out


def summarize_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    """Summarise the per-pair slopes overall and by history stratum.

    Parameters
    ----------
    slopes : pandas.DataFrame
        Output of :func:`pair_slopes`; only rows with a non-missing ``slope`` are used.

    Returns
    -------
    pandas.DataFrame
        One row for ``all`` and one per stratum of :data:`retail_multilevel.p1.STRATA`, with columns
        ``stratum, n_pairs, n_positive, share_positive_pct`` (slope above zero) and the quantiles
        ``q05, q25, median, q75, q95``. A stratum without pairs has zero counts and ``NaN`` for the rest.
    """
    valid = slopes.loc[slopes["slope"].notna()]
    rows = []
    for name in (ALL, *p1.STRATA):
        values = (valid["slope"] if name == ALL else valid.loc[valid["stratum"] == name, "slope"]).to_numpy(float)
        n = len(values)
        q = np.quantile(values, SUMMARY_QUANTILES) if n else np.full(len(SUMMARY_QUANTILES), np.nan)
        n_pos = int((values > POSITIVE_TOL).sum())
        rows.append(
            {
                "stratum": name,
                "n_pairs": n,
                "n_positive": n_pos,
                "share_positive_pct": 100.0 * n_pos / n if n else np.nan,
                **dict(zip(SUMMARY_COLUMNS[4:], q, strict=True)),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
