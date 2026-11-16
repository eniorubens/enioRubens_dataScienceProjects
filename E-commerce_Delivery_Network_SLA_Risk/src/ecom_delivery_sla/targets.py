"""Target construction and temporal validity rules."""

from __future__ import annotations

import pandas as pd

DATE_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)


def add_target_columns(orders: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with parsed dates and delivery target columns."""
    result = orders.copy()
    for column in DATE_COLUMNS:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["delivery_delay_days"] = (
        result["order_delivered_customer_date"]
        - result["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86_400
    result["delivery_delay_calendar_days"] = (
        result["order_delivered_customer_date"].dt.normalize()
        - result["order_estimated_delivery_date"].dt.normalize()
    ).dt.days
    result["actual_lead_days"] = (
        result["order_delivered_customer_date"]
        - result["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    result["promised_lead_days"] = (
        result["order_estimated_delivery_date"]
        - result["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    result["sla_breach"] = result["delivery_delay_calendar_days"] > 0
    return result


def build_supervised_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Build the valid supervised population without silently repairing anomalies."""
    prepared = add_target_columns(orders)
    valid = (
        prepared["order_status"].eq("delivered")
        & prepared["order_delivered_customer_date"].notna()
        & prepared["order_estimated_delivery_date"].notna()
    )
    return prepared.loc[valid].copy()
