import pandas as pd

from ecom_delivery_sla.targets import add_target_columns, build_supervised_orders


def sample_orders() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["a", "b", "c"],
        "order_status": ["delivered", "delivered", "canceled"],
        "order_purchase_timestamp": ["2018-01-01 10:00:00"] * 3,
        "order_approved_at": ["2018-01-01 10:01:00"] * 3,
        "order_delivered_carrier_date": ["2018-01-03", "2018-01-04", None],
        "order_delivered_customer_date": [
            "2018-01-11",
            "2018-01-08",
            None,
        ],
        "order_estimated_delivery_date": [
            "2018-01-10",
            "2018-01-10",
            "2018-01-10",
        ],
    })


def test_target_definition_and_population():
    prepared = add_target_columns(sample_orders())
    supervised = build_supervised_orders(sample_orders())

    assert prepared.loc[0, "delivery_delay_days"] == 1
    assert prepared.loc[0, "delivery_delay_calendar_days"] == 1
    assert bool(prepared.loc[0, "sla_breach"])
    assert prepared.loc[1, "delivery_delay_days"] == -2
    assert len(supervised) == 2
    assert supervised["order_id"].tolist() == ["a", "b"]


def test_same_calendar_day_at_later_hour_is_not_a_breach():
    orders = sample_orders().iloc[[0]].copy()
    orders.loc[orders.index[0], "order_delivered_customer_date"] = (
        "2018-01-10 15:00:00"
    )

    prepared = add_target_columns(orders)

    assert prepared.loc[0, "delivery_delay_days"] == 0.625
    assert prepared.loc[0, "delivery_delay_calendar_days"] == 0
    assert not bool(prepared.loc[0, "sla_breach"])
