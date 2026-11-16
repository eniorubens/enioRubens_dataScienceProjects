import pandas as pd

from ecom_delivery_sla.eda import category_metrics, monthly_metrics, state_metrics
from ecom_delivery_sla.io import schema_summary


def supervised_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["a", "b", "c", "d"],
        "customer_id": ["c1", "c2", "c1", "c3"],
        "order_purchase_timestamp": pd.to_datetime(["2018-01-01", "2018-01-15", "2018-02-01", "2018-02-15"]),
        "delivery_delay_days": [1, -2, 3, -1],
        "sla_breach": [True, False, True, False],
    })


def test_monthly_and_state_metrics_use_order_granularity():
    data = supervised_sample()
    customers = pd.DataFrame({"customer_id": ["c1", "c2", "c3"], "customer_state": ["SP", "RJ", "SP"]})

    monthly = monthly_metrics(data)
    state = state_metrics(data, customers, min_orders=1)

    assert monthly.loc["2018-01", "orders"] == 2
    assert monthly.loc["2018-01", "breaches"] == 1
    assert state.loc["SP", "orders"] == 3
    assert state.loc["SP", "breaches"] == 2


def test_category_metrics_aggregates_items_before_grouping():
    data = supervised_sample()
    items = pd.DataFrame({"order_id": ["a", "a", "b", "c", "d"], "product_id": [1, 2, 3, 4, 5]})
    products = pd.DataFrame({"product_id": [1, 2, 3, 4, 5], "product_category_name": ["x", "x", "y", "x", "z"]})

    result = category_metrics(data, items, products, min_orders=1)

    assert result.loc["x", "orders"] == 2
    assert result.loc["x", "breaches"] == 2


def test_schema_summary_reports_each_field_and_observed_type():
    tables = {"orders": pd.DataFrame({"order_id": ["a"], "amount": [1.5]})}

    result = schema_summary(tables)

    assert result["field"].tolist() == ["order_id", "amount"]
    assert result["data_type"].tolist() == ["object", "float64"]
    assert result["non_null_pct"].tolist() == [1.0, 1.0]
