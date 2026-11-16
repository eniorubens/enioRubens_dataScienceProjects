import numpy as np
import pandas as pd
import pytest

from ecom_delivery_sla.network import build_network_features


def _network_fixture():
    orders = pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_status": ["delivered", "delivered", "delivered"],
            "order_purchase_timestamp": ["2018-01-01", "2018-01-15", "2018-02-15"],
            "order_approved_at": ["2018-01-01", "2018-01-15", "2018-02-15"],
            "order_delivered_carrier_date": ["2018-01-02", "2018-01-16", "2018-02-16"],
            "order_delivered_customer_date": ["2018-02-01", "2018-01-20", "2018-02-20"],
            "order_estimated_delivery_date": ["2018-01-20", "2018-02-01", "2018-03-01"],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_item_id": [1, 1, 1],
            "product_id": ["p", "p", "p"],
            "seller_id": ["s1", "s1", "s2"],
            "price": [100.0, 100.0, 50.0],
            "shipping_limit_date": ["2018-01-03", "2018-01-17", "2018-02-17"],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3"],
            "customer_state": ["SP", "SP", "RJ"],
        }
    )
    orders["customer_id"] = ["c1", "c2", "c3"]
    return orders, items, customers


def test_network_history_is_as_of_and_excludes_future_delivery():
    orders, items, customers = _network_fixture()

    result = build_network_features(orders, items, customers)

    row_b = result.loc[result["order_id"] == "b"].iloc[0]
    row_c = result.loc[result["order_id"] == "c"].iloc[0]
    assert row_b["seller_hist_orders"] == 0
    assert row_b["seller_cold_start"] == 1
    assert row_c["seller_hist_orders"] == 0
    assert row_c["seller_cold_start"] == 1
    assert row_c["edge_hist_orders"] == 0


def test_network_returns_one_row_per_order_and_cold_start_for_new_edge():
    orders, items, customers = _network_fixture()

    result = build_network_features(orders, items, customers)

    assert result["order_id"].is_unique
    assert set(result["order_id"]) == {"a", "b", "c"}
    assert result.loc[result["order_id"] == "c", "edge_cold_start"].item() == 1


def _frames(rows):
    """Build orders/items/customers from compact synthetic rows.

    Each row is a dict with order_id, seller_id (or a list of (seller, price)),
    state, purchase, carrier, delivered, estimated and shipping_limit.
    """
    orders, items, customers = [], [], []
    for row in rows:
        orders.append(
            {
                "order_id": row["order_id"],
                "customer_id": f"c_{row['order_id']}",
                "order_status": "delivered",
                "order_purchase_timestamp": row["purchase"],
                "order_approved_at": row["purchase"],
                "order_delivered_carrier_date": row["carrier"],
                "order_delivered_customer_date": row["delivered"],
                "order_estimated_delivery_date": row["estimated"],
            }
        )
        customers.append({"customer_id": f"c_{row['order_id']}", "customer_state": row["state"]})
        for number, (seller, price) in enumerate(row["sellers"], start=1):
            items.append(
                {
                    "order_id": row["order_id"],
                    "order_item_id": number,
                    "product_id": "p",
                    "seller_id": seller,
                    "price": price,
                    "shipping_limit_date": row["limit"],
                }
            )
    return pd.DataFrame(orders), pd.DataFrame(items), pd.DataFrame(customers)


def _late_row(order_id, seller, state, purchase, delivered, carrier=None, limit=None):
    purchase_ts = pd.Timestamp(purchase)
    return {
        "order_id": order_id,
        "sellers": [(seller, 100.0)],
        "state": state,
        "purchase": purchase,
        "carrier": carrier or str((purchase_ts + pd.Timedelta(days=3)).date()),
        "delivered": delivered,
        "estimated": str((purchase_ts + pd.Timedelta(days=10)).date()),
        "limit": limit or str((purchase_ts + pd.Timedelta(days=1)).date()),
    }


def test_cold_start_edge_destination_and_dispatch_use_global_asof_prior():
    rows = [
        _late_row(f"h{i}", "s1", "SP", f"2018-01-{i + 1:02d}", f"2018-01-{i + 20:02d}")
        for i in range(5)
    ]
    rows.append(_late_row("z", "s2", "RJ", "2018-03-01", "2018-03-20"))
    orders, items, customers = _frames(rows)

    result = build_network_features(orders, items, customers)

    row = result.loc[result["order_id"] == "z"].iloc[0]
    # 5 delivered orders, all late: (5 + 20 * 0.5) / (5 + 20) = 0.6, not 0.5.
    expected = (5 + 20 * 0.5) / (5 + 20)
    assert row["edge_cold_start"] == 1
    assert row["edge_hist_breach_rate"] == pytest.approx(expected)
    assert row["dest_hist_breach_rate"] == pytest.approx(expected)
    assert row["seller_hist_breach_rate"] == pytest.approx(expected)
    # Dispatch: 5 single-seller dispatches, all after the shipping limit.
    assert row["seller_hist_dispatch_late_rate"] == pytest.approx(expected)
    assert expected != 0.5


def test_dispatch_before_own_purchase_is_excluded_from_history():
    rows = [
        # Dispatch (01-01) precedes its own purchase (01-05) and is "late"
        # versus a limit of 2017-12-30; it must not be counted.
        _late_row(
            "x", "s1", "SP", "2018-01-05", "2018-01-20",
            carrier="2018-01-01", limit="2017-12-30",
        ),
        _late_row("y", "s1", "SP", "2018-01-10", "2018-01-25"),
    ]
    orders, items, customers = _frames(rows)

    result = build_network_features(orders, items, customers).set_index("order_id")

    # x never sees itself and y does not count x's pre-purchase dispatch.
    assert result.loc["x", "seller_hist_dispatch_late_rate"] == pytest.approx(0.5)
    assert result.loc["y", "seller_hist_dispatch_late_rate"] == pytest.approx(0.5)


def test_delivery_before_own_purchase_is_excluded_from_history():
    rows = [
        _late_row("w", "s1", "SP", "2018-02-01", "2018-01-15"),
        _late_row("v", "s1", "SP", "2018-03-01", "2018-03-20"),
    ]
    orders, items, customers = _frames(rows)

    result = build_network_features(orders, items, customers).set_index("order_id")

    assert result.loc["v", "seller_hist_orders"] == 0
    assert result.loc["v", "seller_cold_start"] == 1
    assert result.loc["v", "edge_hist_orders"] == 0


def test_aggregation_two_sellers_one_cold_start_ignores_nan_weights():
    rows = [
        _late_row("p1", "s1", "SP", "2018-01-01", "2018-01-20"),
        _late_row("p2", "s1", "SP", "2018-01-02", "2018-01-21"),
    ]
    multi = _late_row("m", "s1", "SP", "2018-03-01", "2018-03-20")
    multi["sellers"] = [("s1", 300.0), ("s2", 100.0)]
    rows.append(multi)
    orders, items, customers = _frames(rows)

    result = build_network_features(orders, items, customers).set_index("order_id")

    prior = (2 + 20 * 0.5) / (2 + 20)
    s1_rate = (2 + 20 * prior) / (2 + 20)
    row = result.loc["m"]
    assert row["seller_cold_start"] == 1  # max: one cold seller marks the order
    assert row["seller_hist_orders"] == 0  # min: least-evidence seller
    assert row["seller_hist_breach_rate"] == pytest.approx(max(s1_rate, prior))
    expected_mean = (300 * s1_rate + 100 * prior) / 400
    assert row["seller_hist_breach_rate_value_weighted_mean"] == pytest.approx(expected_mean)
    # seller_days_since_last is NaN for the cold seller: max keeps s1's value.
    assert row["seller_days_since_last"] > 0
    assert row["seller_days_since_last"] == pytest.approx(
        (pd.Timestamp("2018-03-01") - pd.Timestamp("2018-01-21")).days
    )
    # Single-seller order: max equals the value-weighted mean.
    single = result.loc["p2"]
    assert single["seller_hist_breach_rate"] == pytest.approx(
        single["seller_hist_breach_rate_value_weighted_mean"]
    )
    assert result.index.is_unique


def test_aggregate_pair_features_does_not_fill_missing_with_zero():
    from ecom_delivery_sla.network import _aggregate_pair_features

    pair = pd.DataFrame(
        {
            "order_id": ["a", "a", "b"],
            "item_value": [10.0, 30.0, 0.0],
            "seller_hist_breach_rate": [0.2, 0.6, 0.4],
            "seller_cold_start": [0, 0, 1],
            "seller_days_since_last": [np.nan, 8.0, np.nan],
            "seller_hist_dispatch_late_rate": [0.1, 0.1, 0.1],
            "edge_hist_breach_rate": [0.3, 0.3, 0.3],
            "edge_cold_start": [0, 0, 1],
            "edge_hist_median_lead_days": [np.nan, np.nan, np.nan],
            "seller_hist_orders": [5, 7, 0],
            "edge_hist_orders": [1, 2, 0],
        }
    )

    result = _aggregate_pair_features(pair).set_index("order_id")

    assert result.loc["a", "seller_days_since_last_value_weighted_mean"] == pytest.approx(8.0)
    assert result.loc["a", "seller_hist_breach_rate_value_weighted_mean"] == pytest.approx(
        (10 * 0.2 + 30 * 0.6) / 40
    )
    assert np.isnan(result.loc["b", "seller_days_since_last_value_weighted_mean"])
    assert np.isnan(result.loc["a", "edge_hist_median_lead_days_value_weighted_mean"])
    # Zero total weight falls back to equal weights.
    assert result.loc["b", "seller_hist_breach_rate_value_weighted_mean"] == pytest.approx(0.4)
    assert not any(column.startswith("seller_cold_start_") for column in result.columns)
