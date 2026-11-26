"""Descriptive statistics on the panel: no models, no plotting, no I/O."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from retail_multilevel.split import FIRST_TEST_WEEK, is_test

BELOW_BASE_RATIO = 0.95  # price below 95% of base price counts as a price cut
PROMO_TYPES = ["FEATURE", "DISPLAY", "TPR_ONLY"]


def panel_overview(panel: pd.DataFrame) -> pd.Series:
    """Summarise the size and time span of the panel.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with ``STORE_NUM``, ``UPC``, ``PAIR_ID``, ``WEEK`` and ``WEEK_END_DATE``.

    Returns
    -------
    pandas.Series
        Index ``rows, stores, upcs, pairs, weeks, first_week, last_week``.
    """
    return pd.Series(
        {
            "rows": len(panel),
            "stores": panel["STORE_NUM"].nunique(),
            "upcs": panel["UPC"].nunique(),
            "pairs": panel["PAIR_ID"].nunique(),
            "weeks": panel["WEEK"].nunique(),
            "first_week": panel["WEEK_END_DATE"].min(),
            "last_week": panel["WEEK_END_DATE"].max(),
        }
    )


def promo_rates(panel: pd.DataFrame, by: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Share of rows with each promotion flag, overall or by group.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with the promotion flags, ``PRICE`` and ``BASE_PRICE``.
    by : str or list of str, optional
        Grouping column(s). ``None`` gives a single row indexed ``"all"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``FEATURE, DISPLAY, TPR_ONLY, ANY_PROMO, BELOW_BASE`` (shares in [0, 1])
        and ``n`` (number of rows). ``BELOW_BASE`` is ``PRICE < 0.95 * BASE_PRICE``.
    """
    flags = (
        panel[PROMO_TYPES]
        .astype(float)
        .assign(
            ANY_PROMO=panel["ANY_PROMO"].astype(float),
            BELOW_BASE=(panel["PRICE"] < BELOW_BASE_RATIO * panel["BASE_PRICE"]).astype(float),
        )
    )
    if by is None:
        out = flags.mean().to_frame("all").T
        out["n"] = len(flags)
        return out
    keys = [by] if isinstance(by, str) else list(by)
    grouped = flags.groupby([panel[k] for k in keys], dropna=False)
    out = grouped.mean()
    out["n"] = grouped.size()
    return out


def weekly_units(panel: pd.DataFrame, first_test_week: int = FIRST_TEST_WEEK) -> pd.DataFrame:
    """Total units per week, with the train/test indicator.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with ``WEEK_END_DATE``, ``WEEK`` and ``UNITS``.
    first_test_week : int, default FIRST_TEST_WEEK
        First ``WEEK`` index of the test period.

    Returns
    -------
    pandas.DataFrame
        Columns ``WEEK_END_DATE, WEEK, UNITS, IS_TEST``, sorted by week.
    """
    out = panel.groupby(["WEEK_END_DATE", "WEEK"], as_index=False)["UNITS"].sum()
    out = out.sort_values("WEEK", ignore_index=True)
    out["IS_TEST"] = is_test(out, first_test_week)
    return out[["WEEK_END_DATE", "WEEK", "UNITS", "IS_TEST"]]


def pair_coverage(panel: pd.DataFrame, first_test_week: int = FIRST_TEST_WEEK) -> pd.DataFrame:
    """Count the distinct weeks of each store x UPC pair before and from the test start.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with ``PAIR_ID``, ``STORE_NUM``, ``UPC`` and ``WEEK``.
    first_test_week : int, default FIRST_TEST_WEEK
        First ``WEEK`` index of the test period.

    Returns
    -------
    pandas.DataFrame
        One row per ``PAIR_ID``; columns ``PAIR_ID, STORE_NUM, UPC, n_weeks_train, n_weeks_test``.
    """
    test = is_test(panel, first_test_week)
    base = panel.groupby("PAIR_ID", as_index=False)[["STORE_NUM", "UPC"]].first()
    n_train = panel.loc[~test].groupby("PAIR_ID")["WEEK"].nunique().rename("n_weeks_train")
    n_test = panel.loc[test].groupby("PAIR_ID")["WEEK"].nunique().rename("n_weeks_test")
    out = base.join(n_train, on="PAIR_ID").join(n_test, on="PAIR_ID")
    out[["n_weeks_train", "n_weeks_test"]] = out[["n_weeks_train", "n_weeks_test"]].fillna(0).astype(int)
    return out


def price_variation(panel: pd.DataFrame) -> pd.DataFrame:
    """Within-pair price variation.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with ``PAIR_ID``, ``LOG_PRICE``, ``PRICE`` and ``BASE_PRICE``.

    Returns
    -------
    pandas.DataFrame
        One row per ``PAIR_ID``: ``n_weeks``, ``sd_log_price`` (``LOG_PRICE`` standard
        deviation with ddof=1, NaN for a single week) and ``share_below_base`` (share of
        weeks with ``PRICE < 0.95 * BASE_PRICE``).
    """
    df = panel.assign(_below=(panel["PRICE"] < BELOW_BASE_RATIO * panel["BASE_PRICE"]).astype(float))
    g = df.groupby("PAIR_ID")
    out = pd.DataFrame(
        {
            "n_weeks": g.size(),
            "sd_log_price": g["LOG_PRICE"].std(ddof=1),
            "share_below_base": g["_below"].mean(),
        }
    )
    return out.reset_index()


def variance_shares(
    panel: pd.DataFrame, y: str = "LOG_UNITS", groups: Sequence[str] = ("STORE_NUM", "UPC", "PAIR_ID")
) -> pd.DataFrame:
    """Descriptive share of the variance of ``y`` lying between the groups of each grouping.

    The share is the weighted variance of the group means around the grand mean (weights =
    group sizes) divided by the total variance of ``y`` (both with ddof=0), i.e. the
    descriptive eta-squared. It is computed without any model, one grouping at a time, so
    the shares of non-nested groupings do not sum to 1.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel containing ``y`` and the grouping columns.
    y : str, default "LOG_UNITS"
        Response column.
    groups : sequence of str
        Grouping columns, evaluated one at a time.

    Returns
    -------
    pandas.DataFrame
        Columns ``group, n_groups, between_share``, one row per grouping, in the given order.
    """
    values = panel[y]
    total = values.var(ddof=0)
    rows = []
    for g in groups:
        means = values.groupby(panel[g]).transform("mean")
        between = ((means - values.mean()) ** 2).mean()
        rows.append({"group": g, "n_groups": panel[g].nunique(), "between_share": between / total})
    return pd.DataFrame(rows, columns=["group", "n_groups", "between_share"])


def raw_promo_lift(panel: pd.DataFrame) -> pd.DataFrame:
    """Raw promotion lift in log units, relative to weeks without any promotion.

    Descriptive only: ``LOG_UNITS`` is demeaned within ``PAIR_ID``; the baseline is the mean
    demeaned value of rows with ``ANY_PROMO == False``; the lift of each promotion type is the
    mean demeaned value of rows with that flag on, minus the baseline. It is confounded with
    price cuts and with the other promotions, so it is not causal.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with ``PAIR_ID``, ``LOG_UNITS``, ``ANY_PROMO`` and the promotion flags.

    Returns
    -------
    pandas.DataFrame
        Index ``promo`` (``FEATURE, DISPLAY, TPR_ONLY``); columns ``lift_log``, ``lift_pct``
        (``exp(lift_log) - 1``) and ``n_weeks`` (rows with the flag on).
    """
    demeaned = panel["LOG_UNITS"] - panel.groupby("PAIR_ID")["LOG_UNITS"].transform("mean")
    baseline = demeaned[~panel["ANY_PROMO"].astype(bool)].mean()
    rows = {}
    for promo in PROMO_TYPES:
        on = panel[promo] == 1
        lift = demeaned[on].mean() - baseline
        rows[promo] = {"lift_log": lift, "lift_pct": np.exp(lift) - 1.0, "n_weeks": int(on.sum())}
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["lift_log", "lift_pct", "n_weeks"])
    out.index.name = "promo"
    return out
