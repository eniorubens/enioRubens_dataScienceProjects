"""Funnel and channel performance on the canonical user base."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the path from impacted users to retained users.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical user-level base with ``user_id``, ``converted_flag``
        and ``retained_flag``.

    Returns
    -------
    pd.DataFrame
        Columns ``Metric`` and ``Value`` with impacted users, conversions,
        retained users among those converted, the conversion rate over all
        rows and the retention rate among converted users, both in percent.
    """
    conversions = int(df["converted_flag"].sum())
    retained = int(df[df["converted_flag"] == 1]["retained_flag"].sum())
    return pd.DataFrame(
        {
            "Metric": [
                "Impacted Users",
                "Conversions",
                "Retained (among converted)",
                "Conversion Rate (%)",
                "Retention Rate (%)",
            ],
            "Value": [
                df["user_id"].nunique(),
                conversions,
                retained,
                round((conversions / len(df)) * 100, 2),
                round((retained / max(conversions, 1)) * 100, 2),
            ],
        }
    )


def analyze_channel_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute conversion and retention by marketing channel.

    Retention only counts users who converted and were then retained, and
    the retention rate is bounded to the 0-100 range.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical user-level base with ``marketing_channel``,
        ``user_id``, ``converted_flag``, ``retained_flag`` and
        optionally ``date_canceled``.

    Returns
    -------
    pd.DataFrame
        One row per channel with impacted users, conversions, retained,
        cancellations and the conversion and retention rates in percent,
        sorted by conversion rate, best first.
    """
    channel_df = df.copy()
    channel_df["retained_after_conversion"] = (
        (channel_df["converted_flag"] == 1)
        & (channel_df["retained_flag"] == 1)
    ).astype(int)

    aggregations = {
        "impacted_users": ("user_id", "nunique"),
        "conversions": ("converted_flag", "sum"),
        "retained": ("retained_after_conversion", "sum"),
    }
    if "date_canceled" in channel_df.columns:
        aggregations["cancellations"] = (
            "date_canceled",
            lambda values: values.notna().sum(),
        )

    summary = (
        channel_df.groupby("marketing_channel")
        .agg(**aggregations)
        .reset_index()
    )
    if "cancellations" not in summary.columns:
        summary["cancellations"] = 0

    summary["conversion_rate_%"] = round(
        (summary["conversions"] / summary["impacted_users"]) * 100, 2
    )
    retention = (
        summary["retained"] / summary["conversions"].replace(0, np.nan)
    ) * 100
    summary["retention_rate_%"] = (
        retention.fillna(0).clip(lower=0, upper=100).round(2)
    )
    return summary.sort_values("conversion_rate_%", ascending=False)
