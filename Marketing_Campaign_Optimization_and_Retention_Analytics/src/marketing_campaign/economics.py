"""Customer value by segment and simulated ROI by channel.

Costs and revenues are explicit assumptions (see
:mod:`marketing_campaign.config`); the results are comparative scenarios,
not observed financial accounting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketing_campaign.config import (
    COST_PER_IMPRESSION,
    DEFAULT_COST,
    ORGANIC_CONVERSION_RATE,
    RETENTION_LTV_SHARE,
    REVENUE_PER_CONVERSION,
)


@dataclass(frozen=True)
class LtvSummary:
    """Historical customer value, per customer and per segment.

    Attributes
    ----------
    by_customer : pd.DataFrame
        One row per user with total revenue, total cost, segment and
        ``historical_ltv`` (revenue minus cost).
    by_segment : pd.DataFrame
        Mean, median, standard deviation and size of the value per
        segment, highest mean first.
    global_ltv : float
        Mean historical value across all customers.
    segment_ltv_map : dict
        Mean historical value keyed by segment name.
    """

    by_customer: pd.DataFrame
    by_segment: pd.DataFrame
    global_ltv: float
    segment_ltv_map: dict[str, float]


def estimate_ltv(
    df: pd.DataFrame,
    cost_per_impression: dict[str, float] | None = None,
    revenue_per_conversion: float = REVENUE_PER_CONVERSION,
) -> LtvSummary:
    """Estimate the historical value of each customer and segment.

    The value of a customer is the revenue of its conversions minus the
    cost of the impressions it received.

    Parameters
    ----------
    df : pd.DataFrame
        Segmented user base with ``user_id``, ``marketing_channel``,
        ``converted_flag`` and ``Segment``.
    cost_per_impression : dict, optional
        Cost of an impression by channel. Defaults to
        ``config.COST_PER_IMPRESSION``; unknown channels use
        ``config.DEFAULT_COST``.
    revenue_per_conversion : float, default=45.0
        Revenue assumed for each conversion.

    Returns
    -------
    LtvSummary
        Per-customer and per-segment historical value.
    """
    costs = cost_per_impression or COST_PER_IMPRESSION
    frame = df.copy()
    frame["revenue"] = frame["converted_flag"] * revenue_per_conversion
    frame["cost"] = frame["marketing_channel"].map(costs).fillna(DEFAULT_COST)

    by_customer = (
        frame.groupby("user_id")
        .agg(
            total_revenue=("revenue", "sum"),
            total_cost=("cost", "sum"),
            segment=("Segment", "first"),
        )
        .assign(historical_ltv=lambda d: d["total_revenue"] - d["total_cost"])
        .reset_index()
    )
    by_segment = (
        by_customer.groupby("segment")["historical_ltv"]
        .agg(
            ltv_mean="mean",
            ltv_median="median",
            ltv_std="std",
            n_customers="count",
        )
        .reset_index()
        .sort_values("ltv_mean", ascending=False)
    )
    by_segment["ltv_std"] = by_segment["ltv_std"].fillna(0)
    return LtvSummary(
        by_customer=by_customer,
        by_segment=by_segment,
        global_ltv=float(by_customer["historical_ltv"].mean()),
        segment_ltv_map=by_segment.set_index("segment")["ltv_mean"].to_dict(),
    )


def simulate_roi(
    channel_summary: pd.DataFrame,
    df: pd.DataFrame,
    ltv: LtvSummary,
    cost_per_impression: dict[str, float] | None = None,
    revenue_per_conversion: float = REVENUE_PER_CONVERSION,
    organic_conversion_rate: float = ORGANIC_CONVERSION_RATE,
) -> pd.DataFrame:
    """Simulate the total and incremental ROI of each channel.

    Revenue combines the conversions with a share of the value of the
    retained users, using the value of the channel's majority segment.
    The incremental ROI discounts what would have converted organically.

    Parameters
    ----------
    channel_summary : pd.DataFrame
        Output of
        :func:`marketing_campaign.funnel.analyze_channel_performance`.
    df : pd.DataFrame
        Segmented user base, used for the baseline conversion and the
        majority segment of each channel.
    ltv : LtvSummary
        Output of :func:`estimate_ltv`.
    cost_per_impression : dict, optional
        Cost of an impression by channel.
    revenue_per_conversion : float, default=45.0
        Revenue assumed for each conversion.
    organic_conversion_rate : float, default=0.20
        Share of the observed conversion rate assumed to happen without
        the campaign.

    Returns
    -------
    pd.DataFrame
        ``channel_summary`` plus total cost, majority segment, reference
        value, estimated revenue and profit, incremental revenue, ROI and
        incremental ROI in percent.
    """
    costs = cost_per_impression or COST_PER_IMPRESSION
    roi = channel_summary.copy()
    roi["total_cost"] = (
        roi["marketing_channel"].map(costs).fillna(DEFAULT_COST)
        * roi["impacted_users"]
    )

    majority_segment = (
        df.dropna(subset=["Segment"])
        .groupby("marketing_channel")["Segment"]
        .agg(lambda values: values.mode().iloc[0])
    )
    roi["majority_segment"] = roi["marketing_channel"].map(majority_segment)
    roi["ltv_reference"] = (
        roi["majority_segment"].map(ltv.segment_ltv_map).fillna(ltv.global_ltv)
    )

    roi["estimated_revenue"] = (
        roi["conversions"] * revenue_per_conversion
        + roi["retained"] * roi["ltv_reference"] * RETENTION_LTV_SHARE
    )
    roi["estimated_profit"] = roi["estimated_revenue"] - roi["total_cost"]
    roi["roi_%"] = _percent_of_cost(roi["estimated_profit"], roi["total_cost"])

    baseline_conversions = (
        roi["impacted_users"]
        * df["converted_flag"].mean()
        * organic_conversion_rate
    )
    baseline_revenue = baseline_conversions * revenue_per_conversion
    roi["incremental_revenue"] = roi["estimated_revenue"] - baseline_revenue
    roi["roi_incremental_%"] = _percent_of_cost(
        roi["incremental_revenue"] - roi["total_cost"], roi["total_cost"]
    )
    return roi


def _percent_of_cost(gain: pd.Series, cost: pd.Series) -> pd.Series:
    """Express a gain as a percentage of cost, with 0 when cost is 0.

    Parameters
    ----------
    gain : pd.Series
        Profit or incremental gain.
    cost : pd.Series
        Cost the gain is measured against.

    Returns
    -------
    pd.Series
        ``gain / cost * 100`` rounded to two decimals; zero-cost rows
        become 0.
    """
    return (gain / cost.replace(0, np.nan) * 100).round(2).fillna(0)
