"""Reproducible temporal anomaly rules for the Olist orders table."""

from __future__ import annotations

import pandas as pd

ANOMALY_RULES = (
    "dispatch_before_purchase",
    "delivery_before_dispatch",
    "undelivered_with_delivery_date",
    "unexpected_status_with_dispatch",
    "estimated_before_purchase",
    "delivery_before_purchase",
)

_POLICY = {
    "dispatch_before_purchase": (
        "sensitivity",
        "Retain the raw row, but exclude the event from as-of historical features.",
    ),
    "delivery_before_dispatch": (
        "sensitivity",
        "Retain the raw row and report sensitivity; the chronology is inconsistent.",
    ),
    "undelivered_with_delivery_date": (
        "exclude",
        "Exclude from the supervised population until status/date semantics are resolved.",
    ),
    "unexpected_status_with_dispatch": (
        "sensitivity",
        "Retain for quality audit, but do not infer a completed delivery outcome.",
    ),
    "estimated_before_purchase": (
        "exclude",
        "A promise before purchase violates the prediction-time contract.",
    ),
    "delivery_before_purchase": (
        "exclude",
        "A delivery before purchase is temporally impossible for this contract.",
    ),
}


def _parsed_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the audit timestamps parsed safely."""
    result = orders.copy()
    for column in (
        "order_purchase_timestamp",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        result[column] = pd.to_datetime(
            result[column], errors="coerce", format="mixed"
        )
    return result


def audit_temporal_anomalies(orders: pd.DataFrame) -> pd.DataFrame:
    """Return one row per ``(rule, order_id)`` temporal anomaly.

    The rules mirror the six source-quality checks documented in the Phase 1
    pre-analysis. The function only classifies evidence; it never repairs or
    drops source rows.
    """
    data = _parsed_orders(orders)
    purchase = data["order_purchase_timestamp"]
    dispatch = data["order_delivered_carrier_date"]
    delivery = data["order_delivered_customer_date"]
    estimated = data["order_estimated_delivery_date"]
    status = data["order_status"].fillna("")
    masks = {
        "dispatch_before_purchase": dispatch.notna() & purchase.notna() & (dispatch < purchase),
        "delivery_before_dispatch": delivery.notna() & dispatch.notna() & (delivery < dispatch),
        "undelivered_with_delivery_date": delivery.notna() & status.ne("delivered"),
        "unexpected_status_with_dispatch": dispatch.notna() & ~status.isin(["shipped", "delivered"]),
        "estimated_before_purchase": estimated.notna() & purchase.notna() & (estimated < purchase),
        "delivery_before_purchase": delivery.notna() & purchase.notna() & (delivery < purchase),
    }
    columns = [
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    rows = []
    for rule in ANOMALY_RULES:
        selected = data.loc[masks[rule], columns].copy()
        selected.insert(0, "rule", rule)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["rule", "order_id"], kind="stable"
    ).reset_index(drop=True)


def anomaly_policy_summary(anomalies: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize anomaly counts, policy decisions, and justifications.

    Parameters
    ----------
    anomalies:
        Optional output from :func:`audit_temporal_anomalies`. When omitted,
        counts are zero because this function deliberately does not read raw
        CSVs or hide an I/O dependency inside the policy layer.
    """
    counts = (
        anomalies.groupby("rule").size()
        if anomalies is not None and not anomalies.empty
        else pd.Series(dtype="int64")
    )
    return pd.DataFrame(
        [
            {
                "rule": rule,
                "count": int(counts.get(rule, 0)),
                "decision": _POLICY[rule][0],
                "justification": _POLICY[rule][1],
            }
            for rule in ANOMALY_RULES
        ]
    )
