import pandas as pd

from ecom_delivery_sla.audit import (
    ANOMALY_RULES,
    anomaly_policy_summary,
    audit_temporal_anomalies,
)


def test_temporal_audit_has_one_synthetic_case_per_rule():
    rows = [
        ("dispatch", "delivered", "2020-01-02", "2020-01-01", "2020-01-04", "2020-01-05"),
        ("delivery_dispatch", "delivered", "2020-01-01", "2020-01-04", "2020-01-03", "2020-01-05"),
        ("undelivered", "shipped", "2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"),
        ("unexpected", "canceled", "2020-01-01", "2020-01-02", None, "2020-01-05"),
        ("estimated", "delivered", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-01"),
        ("delivery", "delivered", "2020-01-02", "2020-01-01", "2020-01-01 12:00:00", "2020-01-05"),
    ]
    orders = pd.DataFrame(
        [
            {
                "order_id": name,
                "order_status": status,
                "order_purchase_timestamp": purchase,
                "order_delivered_carrier_date": dispatch,
                "order_delivered_customer_date": delivery,
                "order_estimated_delivery_date": estimated,
            }
            for name, status, purchase, dispatch, delivery, estimated in rows
        ]
    )

    anomalies = audit_temporal_anomalies(orders)
    counts = anomalies.groupby("rule").size().to_dict()

    assert set(counts) == set(ANOMALY_RULES)
    assert all(counts[rule] >= 1 for rule in ANOMALY_RULES)
    assert not anomalies.duplicated(["rule", "order_id"]).any()


def test_anomaly_policy_summary_preserves_decisions_and_counts():
    anomalies = pd.DataFrame({"rule": ["delivery_before_purchase", "delivery_before_purchase"]})

    summary = anomaly_policy_summary(anomalies)
    row = summary.loc[summary["rule"] == "delivery_before_purchase"].iloc[0]

    assert row["count"] == 2
    assert row["decision"] == "exclude"
    assert len(summary) == 6
