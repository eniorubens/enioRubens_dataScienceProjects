"""Descriptive audit of one anomalous week against its neighbouring weeks (protocol P1, E5).

All functions are pure over a *row frame* with the columns ``WEEK``, ``PAIR_ID``, ``STORE_NUM``, ``UNITS``,
``VISITS``, ``HHS``, ``SPEND``, ``PRICE``, ``FEATURE``, ``DISPLAY`` and ``TPR_ONLY``. The grouped summaries also
need ``STATE`` and ``CATEGORY`` when those are the grouping columns. Nothing here fits a model or drops a row:
the functions only describe, so that the evidence about a week can be read before deciding anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_COLS = ["rows", "pairs", "stores", "units", "visits", "hhs", "spend", "units_per_row", "units_per_visit",
                "visits_per_hh", "price_mean", "feature_share", "display_share", "tpr_share"]
SPEND_TOL = 0.02  # relative tolerance of SPEND / UNITS against PRICE


def neighbour_weeks(weeks, week: int, k: int = 4) -> list[int]:
    """Return the observed weeks within ``k`` weeks of ``week``, ``week`` itself excluded.

    Parameters
    ----------
    weeks : array-like
        Weeks present in the data.
    week : int
        Focal week.
    k : int
        Half-width of the window, in weeks.

    Returns
    -------
    list of int
        Sorted neighbouring weeks.
    """
    present = {int(w) for w in np.unique(np.asarray(weeks))}
    return [w for w in range(week - k, week + k + 1) if w != week and w in present]


def week_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise every week: rows, pairs, stores, volumes, price level and promotion shares.

    Parameters
    ----------
    df : pandas.DataFrame
        Row frame (see the module docstring).

    Returns
    -------
    pandas.DataFrame
        One row per ``WEEK`` with the columns in :data:`PROFILE_COLS`.
    """
    g = df.groupby("WEEK")
    out = pd.DataFrame(
        {
            "rows": g.size(),
            "pairs": g["PAIR_ID"].nunique(),
            "stores": g["STORE_NUM"].nunique(),
            "units": g["UNITS"].sum(),
            "visits": g["VISITS"].sum(),
            "hhs": g["HHS"].sum(),
            "spend": g["SPEND"].sum(),
            "price_mean": g["PRICE"].mean(),
            "feature_share": g["FEATURE"].mean(),
            "display_share": g["DISPLAY"].mean(),
            "tpr_share": g["TPR_ONLY"].mean(),
        }
    )
    out["units_per_row"] = out["units"] / out["rows"]
    out["units_per_visit"] = out["units"] / out["visits"]
    out["visits_per_hh"] = out["visits"] / out["hhs"]
    return out[PROFILE_COLS].reset_index()


def compare_to_neighbours(profile: pd.DataFrame, week: int, k: int = 4) -> pd.DataFrame:
    """Compare each profile column of ``week`` with the neighbouring weeks.

    Parameters
    ----------
    profile : pandas.DataFrame
        Output of :func:`week_profile`.
    week : int
        Focal week.
    k : int
        Half-width of the neighbour window.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric, value, nb_mean, nb_min, nb_max, ratio, z, n_neighbours`` (``ratio = value / nb_mean``;
        ``z`` uses the neighbours' standard deviation).
    """
    nb = profile[profile["WEEK"].isin(neighbour_weeks(profile["WEEK"], week, k))]
    row = profile.loc[profile["WEEK"] == week].iloc[0]
    rows = []
    for col in [c for c in profile.columns if c != "WEEK"]:
        sd = nb[col].std(ddof=1)
        mean = nb[col].mean()
        rows.append(
            {
                "metric": col,
                "value": float(row[col]),
                "nb_mean": float(mean),
                "nb_min": float(nb[col].min()),
                "nb_max": float(nb[col].max()),
                "ratio": float(row[col] / mean) if mean else np.nan,
                "z": float((row[col] - mean) / sd) if sd and sd > 0 else np.nan,
                "n_neighbours": len(nb),
            }
        )
    return pd.DataFrame(rows)


def group_ratio(df: pd.DataFrame, week: int, group: str, k: int = 4, value: str = "UNITS") -> pd.DataFrame:
    """Total of ``value`` per group in ``week`` against the neighbours' mean weekly total.

    Parameters
    ----------
    df : pandas.DataFrame
        Row frame.
    week : int
        Focal week.
    group : str
        Grouping column (``STORE_NUM``, ``STATE``, ``CATEGORY``...).
    k : int
        Half-width of the neighbour window.
    value : str
        Column summed.

    Returns
    -------
    pandas.DataFrame
        Columns ``group, rows, nb_rows, total, nb_mean, ratio, deficit, deficit_share``; ``deficit`` is
        ``nb_mean - total`` and ``deficit_share`` its share of the summed positive deficits. Sorted by
        decreasing deficit.
    """
    nbw = neighbour_weeks(df["WEEK"], week, k)
    cur = df[df["WEEK"] == week].groupby(group).agg(rows=(value, "size"), total=(value, "sum"))
    nb = df[df["WEEK"].isin(nbw)].groupby(group).agg(nb_rows=(value, "size"), nb_total=(value, "sum"))
    out = pd.DataFrame({"nb_rows": nb["nb_rows"] / len(nbw), "nb_mean": nb["nb_total"] / len(nbw)}).join(
        cur, how="outer"
    ).fillna(0.0)
    out["ratio"] = out["total"] / out["nb_mean"].replace(0, np.nan)
    out["deficit"] = out["nb_mean"] - out["total"]
    pos = out["deficit"].clip(lower=0).sum()
    out["deficit_share"] = out["deficit"].clip(lower=0) / pos if pos > 0 else 0.0
    out.index.name = "group"
    out = out.reset_index()
    return out[["group", "rows", "nb_rows", "total", "nb_mean", "ratio", "deficit", "deficit_share"]].sort_values(
        "deficit", ascending=False
    ).reset_index(drop=True)


def pair_presence(df: pd.DataFrame, week: int) -> dict:
    """Count the pairs present, missing and new in ``week`` relative to the adjacent weeks.

    Parameters
    ----------
    df : pandas.DataFrame
        Row frame.
    week : int
        Focal week.

    Returns
    -------
    dict
        ``n_pairs``, ``n_pairs_prev``, ``n_pairs_next``, ``missing_both`` (present in ``week-1`` and ``week+1``
        but absent in ``week``) and ``new_vs_both`` (present in ``week`` and absent from both neighbours).
    """
    cur = set(df.loc[df["WEEK"] == week, "PAIR_ID"])
    prev = set(df.loc[df["WEEK"] == week - 1, "PAIR_ID"])
    nxt = set(df.loc[df["WEEK"] == week + 1, "PAIR_ID"])
    return {
        "n_pairs": len(cur),
        "n_pairs_prev": len(prev),
        "n_pairs_next": len(nxt),
        "missing_both": len((prev & nxt) - cur),
        "new_vs_both": len(cur - prev - nxt),
    }


def missing_pairs(df: pd.DataFrame, week: int) -> pd.DataFrame:
    """Return the pairs present in both adjacent weeks but absent in ``week``.

    Parameters
    ----------
    df : pandas.DataFrame
        Row frame with ``PAIR_ID`` and, optionally, ``STORE_NUM``, ``UPC``, ``STATE`` and ``CATEGORY``.
    week : int
        Focal week.

    Returns
    -------
    pandas.DataFrame
        One row per missing pair, with the descriptive columns found in ``df``.
    """
    cur = set(df.loc[df["WEEK"] == week, "PAIR_ID"])
    prev = set(df.loc[df["WEEK"] == week - 1, "PAIR_ID"])
    nxt = set(df.loc[df["WEEK"] == week + 1, "PAIR_ID"])
    keep = [c for c in ("PAIR_ID", "STORE_NUM", "UPC", "STATE", "CATEGORY") if c in df.columns]
    return df.loc[df["PAIR_ID"].isin((prev & nxt) - cur), keep].drop_duplicates("PAIR_ID").reset_index(drop=True)


def pair_ratio_summary(
    df: pd.DataFrame, weeks: list[int], k: int = 4, low: tuple[float, ...] = (0.5, 0.25)
) -> pd.DataFrame:
    """Distribution of the pair-level units ratio, ``UNITS`` in a week over the pair's mean in its neighbours.

    Each week in ``weeks`` is compared with its own neighbours, so neighbouring weeks act as placebos for the
    focal one. Only pairs observed in the week and in at least one neighbour enter.

    Parameters
    ----------
    df : pandas.DataFrame
        Row frame.
    weeks : list of int
        Weeks to evaluate.
    k : int
        Half-width of the neighbour window.
    low : tuple of float
        Thresholds for the share of pairs whose ratio is below them.

    Returns
    -------
    pandas.DataFrame
        Per week: ``n_pairs``, ``ratio_median``, ``ratio_p10``, ``ratio_p90`` and ``share_below_<t>``.
    """
    rows = []
    for w in weeks:
        nbw = neighbour_weeks(df["WEEK"], w, k)
        nb = df[df["WEEK"].isin(nbw)].groupby("PAIR_ID")["UNITS"].mean()
        cur = df[df["WEEK"] == w].set_index("PAIR_ID")["UNITS"]
        ratio = (cur / nb.reindex(cur.index)).dropna()
        rec = {
            "WEEK": w,
            "n_pairs": int(ratio.size),
            "ratio_median": float(ratio.median()),
            "ratio_p10": float(ratio.quantile(0.1)),
            "ratio_p90": float(ratio.quantile(0.9)),
        }
        for t in low:
            rec[f"share_below_{t}"] = float((ratio < t).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def integrity_by_week(df: pd.DataFrame, weeks: list[int] | None = None, spend_tol: float = SPEND_TOL) -> pd.DataFrame:
    """Count obviously corrupted values per week.

    Parameters
    ----------
    df : pandas.DataFrame
        Row frame; ``(PAIR_ID, WEEK)`` identifies a row.
    weeks : list of int, optional
        Weeks to report; all when omitted.
    spend_tol : float
        Relative tolerance of ``SPEND / UNITS`` against ``PRICE``.

    Returns
    -------
    pandas.DataFrame
        Per week: ``duplicate_keys``, ``nonpositive_units``, ``nonpositive_price``, ``spend_mismatch``,
        ``visits_gt_units``, ``hhs_gt_visits``, ``upv_gt_5`` (units per visit above 5), ``vph_gt_3`` (visits
        per household above 3) and ``nan_cells``.
    """
    d = df if weeks is None else df[df["WEEK"].isin(weeks)]
    units = d["UNITS"].to_numpy(dtype=float)
    price = d["PRICE"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.abs(d["SPEND"].to_numpy(dtype=float) / units - price) / price
    flags = pd.DataFrame(
        {
            "WEEK": d["WEEK"].to_numpy(),
            "duplicate_keys": d.duplicated(["PAIR_ID", "WEEK"], keep=False).to_numpy(),
            "nonpositive_units": units <= 0,
            "nonpositive_price": ~(price > 0),
            "spend_mismatch": np.nan_to_num(gap, nan=1.0, posinf=1.0) > spend_tol,
            "visits_gt_units": (d["VISITS"] > d["UNITS"]).to_numpy(),
            "hhs_gt_visits": (d["HHS"] > d["VISITS"]).to_numpy(),
            "upv_gt_5": (d["UNITS"] > 5 * d["VISITS"]).to_numpy(),
            "vph_gt_3": (d["VISITS"] > 3 * d["HHS"]).to_numpy(),
            "nan_cells": d[["UNITS", "VISITS", "HHS", "SPEND", "PRICE"]].isna().any(axis=1).to_numpy(),
        }
    )
    return flags.groupby("WEEK").sum().astype(int).reset_index()


def with_week_index(tx: pd.DataFrame, date_col: str = "WEEK_END_DATE", origin=None) -> pd.DataFrame:
    """Add ``WEEK`` (0-based, 7-day steps from ``origin`` or the first date) and ``PAIR_ID`` to transactions.

    Parameters
    ----------
    tx : pandas.DataFrame
        Raw transactions with ``date_col``, ``STORE_NUM`` and ``UPC``.
    date_col : str
        Week-ending date column.
    origin : datetime-like, optional
        Date of week 0; defaults to the minimum date.

    Returns
    -------
    pandas.DataFrame
        Copy with integer ``WEEK`` and string ``PAIR_ID = STORE_NUM_UPC``.
    """
    dates = pd.to_datetime(tx[date_col])
    start = dates.min() if origin is None else pd.Timestamp(origin)
    out = tx.copy()
    out["WEEK"] = ((dates - start).dt.days // 7).astype(int)
    out["PAIR_ID"] = out["STORE_NUM"].astype(str) + "_" + out["UPC"].astype(str)
    return out


def classify_week(compare: pd.DataFrame, integrity: pd.DataFrame, week: int, k: int = 4) -> dict:
    """Apply the conservative evidence rule: a data error is *proven* only by missing stores or impossible values.

    Parameters
    ----------
    compare : pandas.DataFrame
        Output of :func:`compare_to_neighbours` for ``week``.
    integrity : pandas.DataFrame
        Output of :func:`integrity_by_week`, covering ``week`` and its neighbours.
    week : int
        Focal week.
    k : int
        Half-width of the neighbour window.

    Returns
    -------
    dict
        ``category`` (``"proven data error"`` or ``"unexplained shock"``) and ``reasons`` (list of str).
        Reasons are the stores missing against the neighbours and the impossible-value counts
        (``duplicate_keys``, ``nonpositive_units``, ``spend_mismatch``, ``upv_gt_5``) clearly above the
        neighbours' maximum.
    """
    reasons = []
    c = compare.set_index("metric")
    if c.loc["stores", "value"] < c.loc["stores", "nb_max"]:
        reasons.append(f"stores present {c.loc['stores', 'value']:.0f} < neighbour maximum {c.loc['stores', 'nb_max']:.0f}")
    ig = integrity.set_index("WEEK")
    nbw = [w for w in neighbour_weeks(ig.index, week, k) if w in ig.index]
    for col in ("duplicate_keys", "nonpositive_units", "spend_mismatch", "upv_gt_5"):
        if week in ig.index and nbw and ig.loc[week, col] > max(5, 3 * ig.loc[nbw, col].max()):
            reasons.append(f"{col} = {ig.loc[week, col]} against neighbour maximum {ig.loc[nbw, col].max()}")
    return {"category": "proven data error" if reasons else "unexplained shock", "reasons": reasons}
