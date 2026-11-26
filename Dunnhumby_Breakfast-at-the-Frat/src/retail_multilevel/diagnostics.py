"""Residual diagnostics of the final fixed-effects model: serial dependence and common week shocks.

The functions are pure over a *residual frame* with ``PAIR_ID``, ``WEEK`` and a residual column
(default ``resid``); ``CATEGORY`` is needed only by the by-category summaries. :func:`fit_f3_residuals`
is the only part that needs R: it refits F3 by REML and returns the conditional residuals
``y - X b - Z u`` and the marginal residuals ``y - X b``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLS = ["PAIR_ID", "WEEK", "STORE_NUM", "UPC", "CATEGORY"]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _lag_pairs(res: pd.DataFrame, lag: int, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the residuals at WEEK t and at WEEK t + ``lag`` of the same pair (both observed)."""
    codes, _ = pd.factorize(res["PAIR_ID"])
    week = res["WEEK"].to_numpy(dtype=np.int64)
    span = int(week.max()) + abs(lag) + 2
    key = codes.astype(np.int64) * span + week
    values = pd.Series(res[col].to_numpy(dtype=float), index=key)
    if not values.index.is_unique:
        raise ValueError("(PAIR_ID, WEEK) must identify rows uniquely")
    later = values.reindex(key + lag).to_numpy()
    ok = ~np.isnan(later)
    return values.to_numpy()[ok], later[ok]


def lag_correlations(res: pd.DataFrame, max_lag: int = 4, col: str = "resid") -> pd.DataFrame:
    """Within-pair correlation between the residual at WEEK t and at WEEK t + k (calendar lag).

    Only comparisons where both weeks are observed for the pair enter; pairs are pooled.

    Parameters
    ----------
    res : pandas.DataFrame
        Columns ``PAIR_ID``, ``WEEK`` and ``col``; ``(PAIR_ID, WEEK)`` unique.
    max_lag : int
        Largest lag in weeks.
    col : str
        Residual column.

    Returns
    -------
    pandas.DataFrame
        Columns ``lag``, ``corr``, ``n_pairs_of_obs`` (number of comparisons), ``k = 1..max_lag``.
    """
    rows = []
    for k in range(1, max_lag + 1):
        a, b = _lag_pairs(res, k, col)
        rows.append({"lag": k, "corr": _pearson(a, b), "n_pairs_of_obs": int(a.size)})
    return pd.DataFrame(rows)


def consecutive_correlation(res: pd.DataFrame, col: str = "resid") -> dict:
    """Correlation between consecutive *observed* rows of a pair, whatever the calendar gap.

    Returns
    -------
    dict
        ``corr`` and ``n`` (number of comparisons).
    """
    ordered = res.sort_values(["PAIR_ID", "WEEK"], kind="stable")
    prev = ordered.groupby("PAIR_ID", sort=False)[col].shift(1)
    ok = prev.notna().to_numpy()
    a = prev.to_numpy(dtype=float)[ok]
    b = ordered[col].to_numpy(dtype=float)[ok]
    return {"corr": _pearson(a, b), "n": int(a.size)}


def lag1_by_category(res: pd.DataFrame, col: str = "resid") -> pd.DataFrame:
    """Lag-1 (calendar) correlation within each ``CATEGORY``; columns ``CATEGORY, corr, n_pairs_of_obs``."""
    rows = []
    for cat, part in res.groupby("CATEGORY", sort=True):
        a, b = _lag_pairs(part, 1, col)
        rows.append({"CATEGORY": cat, "corr": _pearson(a, b), "n_pairs_of_obs": int(a.size)})
    return pd.DataFrame(rows)


def _between_share(res: pd.DataFrame, by: list[str], col: str) -> tuple[float, float]:
    """Return the between-group share of the sum of squares and its expectation under independence."""
    x = res[col].to_numpy(dtype=float)
    grand = x.mean()
    total = float(((x - grand) ** 2).sum())
    g = res.groupby(by, sort=False)[col].agg(["mean", "size"])
    between = float((g["size"] * (g["mean"] - grand) ** 2).sum())
    return between / total, (len(g) - 1) / x.size


def week_shocks(res: pd.DataFrame, col: str = "resid") -> tuple[pd.DataFrame, dict]:
    """Compare the per-week mean residual with what independent rows would produce (common week shocks).

    Parameters
    ----------
    res : pandas.DataFrame
        Columns ``PAIR_ID``, ``WEEK``, ``col`` and, for the category summary, ``CATEGORY``.
    col : str
        Residual column.

    Returns
    -------
    weeks : pandas.DataFrame
        One row per week: ``WEEK, mean_resid, n, expected_sd`` (``sqrt(sigma2 / n)``, the sd of the
        week mean if rows were independent).
    summary : dict
        ``sigma2`` (residual variance), ``var_week_means``, ``var_expected_independent`` (mean over
        weeks of ``sigma2 / n_w``), ``variance_ratio``, ``share_week`` (between-week share of the sum
        of squares) with ``share_week_expected`` (``(W - 1) / N`` under independence),
        ``autocorr_week_means`` (lag 1 of the week-mean series over consecutive calendar weeks) and,
        when ``CATEGORY`` is present, ``share_category_week`` and its expectation.
    """
    x = res[col].to_numpy(dtype=float)
    sigma2 = float(x.var(ddof=1))
    g = res.groupby("WEEK")[col].agg(mean_resid="mean", n="size").reset_index()
    g["expected_sd"] = np.sqrt(sigma2 / g["n"])
    series = g.set_index("WEEK")["mean_resid"]
    series = series.reindex(range(int(series.index.min()), int(series.index.max()) + 1)).to_numpy()
    lead, lag = series[1:], series[:-1]
    ok = ~(np.isnan(lead) | np.isnan(lag))
    var_means = float(g["mean_resid"].var(ddof=1))
    var_expected = float((sigma2 / g["n"]).mean())
    share, share_exp = _between_share(res, ["WEEK"], col)
    summary = {
        "sigma2": sigma2,
        "n_weeks": int(len(g)),
        "var_week_means": var_means,
        "var_expected_independent": var_expected,
        "variance_ratio": var_means / var_expected,
        "share_week": share,
        "share_week_expected": share_exp,
        "autocorr_week_means": _pearson(lead[ok], lag[ok]),
    }
    if "CATEGORY" in res.columns:
        s2, e2 = _between_share(res, ["CATEGORY", "WEEK"], col)
        summary |= {"share_category_week": s2, "share_category_week_expected": e2}
    return g, summary


def residual_diagnostics(res: pd.DataFrame, col: str = "resid", max_lag: int = 4) -> dict:
    """Run all diagnostics for one residual column.

    Returns
    -------
    dict
        Tables ``lag_corr``, ``lag1_by_category`` (when ``CATEGORY`` exists), ``week_shocks`` and the
        dicts ``consecutive`` and ``week_summary``.
    """
    weeks, summary = week_shocks(res, col)
    out = {
        "lag_corr": lag_correlations(res, max_lag, col),
        "consecutive": consecutive_correlation(res, col),
        "week_shocks": weeks,
        "week_summary": summary,
    }
    if "CATEGORY" in res.columns:
        out["lag1_by_category"] = lag1_by_category(res, col)
    return out


def fit_f3_residuals(df: pd.DataFrame, control: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Refit F3 by REML and return its residuals attached to the identifier columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Training rows with the model columns of :func:`retail_multilevel.fixedeffects.fit_f3_reml`
        and the identifiers ``PAIR_ID, WEEK, STORE_NUM, UPC, CATEGORY``.
    control : str, optional
        R expression for ``lmerControl``.

    Returns
    -------
    res : pandas.DataFrame
        Identifiers plus ``resid_cond`` (``y - Xb - Zu``) and ``resid_marg`` (``y - Xb``).
    info : dict
        ``sigma_resid``, the fit's ``nobs`` and its messages.
    """
    from retail_multilevel.fixedeffects import _R_COLS, FE_MODELS, _with_random
    from retail_multilevel.multilevel import fit_lmer, to_r_frame
    from retail_multilevel.rbridge import get_robjects

    ro = get_robjects()
    r_df = to_r_frame(df[_R_COLS], ["STORE_NUM", "UPC"])
    fit = fit_lmer(r_df, _with_random(FE_MODELS[-1][1]), "F3_season", reml=True, control=control)
    out = ro.r(
        """function(m) list(cond = as.numeric(resid(m)),
                           marg = as.numeric(lme4::getME(m, "y") - predict(m, re.form = NA)))"""
    )(fit.r_model)
    cond = np.asarray(out.rx2("cond"), dtype=float)
    marg = np.asarray(out.rx2("marg"), dtype=float)
    if cond.size != len(df):
        raise RuntimeError("residual length differs from the number of rows")
    res = df[ID_COLS].copy().reset_index(drop=True)
    res["resid_cond"] = cond
    res["resid_marg"] = marg
    sigma = float(fit.varcomp.loc[fit.varcomp["group"] == "Residual", "sd"].iloc[0])
    return res, {"sigma_resid": sigma, "nobs": fit.nobs, "messages": fit.messages}
