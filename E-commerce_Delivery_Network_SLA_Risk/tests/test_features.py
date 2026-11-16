import pandas as pd
import pytest

from ecom_delivery_sla.features import (
    build_candidate_order_features,
    feature_catalog,
    feature_quality_summary,
)


def test_candidate_features_keep_order_granularity_and_exclude_target():
    orders = pd.DataFrame({
        "order_id": ["o1", "o2"],
        "customer_id": ["c1", "c2"],
        "order_purchase_timestamp": ["2018-01-01", "2018-01-02"],
        "order_estimated_delivery_date": ["2018-01-10", "2018-01-12"],
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o1", "o2"],
        "order_item_id": [1, 2, 1],
        "product_id": ["p1", "p2", "p1"],
        "seller_id": ["s1", "s1", "s1"],
        "price": [10.0, 20.0, 30.0],
        "freight_value": [2.0, 3.0, 4.0],
    })
    products = pd.DataFrame({"product_id": ["p1", "p2"], "product_category_name": ["a", "b"]})
    payments = pd.DataFrame({
        "order_id": ["o1", "o2"],
        "payment_sequential": [1, 1],
        "payment_type": ["credit_card", "boleto"],
        "payment_installments": [2, 1],
        "payment_value": [35.0, 34.0],
    })
    customers = pd.DataFrame({"customer_id": ["c1", "c2"], "customer_state": ["SP", "RJ"], "customer_zip_code_prefix": [100, 100]})
    sellers = pd.DataFrame({"seller_id": ["s1"], "seller_state": ["SP"], "seller_zip_code_prefix": [200]})
    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": [100, 200],
        "geolocation_lat": [0.0, 0.0],
        "geolocation_lng": [0.0, 1.0],
        "geolocation_city": ["A", "B"],
        "geolocation_state": ["SP", "SP"],
    })

    result = build_candidate_order_features(orders, items, products, payments, customers, sellers, geolocation)

    assert len(result) == 2
    assert result["order_id"].is_unique
    assert result.loc[result["order_id"] == "o1", "item_count"].item() == 2
    assert result["payment_record_missing"].eq(0).all()
    assert result["distance_missing"].eq(0).all()
    assert "delivery_delay_days" not in result.columns
    assert "sla_breach" not in result.columns


def test_feature_catalog_has_feature_statuses():
    catalog = feature_catalog()

    assert "available_at_purchase" in set(catalog["status"])
    assert "available_at_approval" in set(catalog["status"])
    assert not catalog["status"].str.startswith("pending").any()
    assert not catalog.query("use_in_model == True")["feature"].isin(
        ["payment_value_sum", "distance_min_km", "distance_max_km", "handling_days_min", "handling_days_max"]
    ).any()


def test_catalog_drives_model_feature_lists_without_quality_controls():
    from ecom_delivery_sla.modeling import BASELINE_FEATURES

    catalog = feature_catalog()
    allowed = set(catalog.query("status == 'available_at_purchase' and use_in_model == True")["feature"])

    assert set(BASELINE_FEATURES) == allowed
    assert not any(feature.startswith("pending") for feature in BASELINE_FEATURES)
    assert "payment_value_sum" not in BASELINE_FEATURES
    assert "distance_min_km" not in BASELINE_FEATURES
    assert "distance_max_km" not in BASELINE_FEATURES


def test_feature_quality_summary_describes_missingness_without_dropping_rows():
    candidate_features = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "distance_mean_km": [100.0, None],
            "payment_count": [1.0, None],
        }
    )

    summary = feature_quality_summary(candidate_features)

    distance_row = summary.loc[summary["feature"] == "distance_mean_km"].iloc[0]
    payment_row = summary.loc[summary["feature"] == "payment_count"].iloc[0]
    assert distance_row["missing"] == 1
    assert "missing_indicator" in distance_row["missing_action"]
    assert "missing_indicator" in payment_row["missing_action"]
    assert len(candidate_features) == 2


def test_new_features_aggregate_two_sellers_at_order_grain():
    orders = pd.DataFrame({
        "order_id": ["o1"],
        "customer_id": ["c1"],
        "order_purchase_timestamp": ["2017-11-20 10:00:00"],
        "order_estimated_delivery_date": ["2017-11-30"],
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o1"],
        "order_item_id": [1, 2],
        "product_id": ["p1", "p2"],
        "seller_id": ["s1", "s2"],
        "price": [100.0, 20.0],
        "freight_value": [10.0, 5.0],
        "shipping_limit_date": ["2017-11-22", "2017-11-23"],
    })
    products = pd.DataFrame({
        "product_id": ["p1", "p2"],
        "product_category_name": ["a", "b"],
        "product_weight_g": [1000.0, 500.0],
        "product_length_cm": [10.0, 10.0],
        "product_height_cm": [10.0, 5.0],
        "product_width_cm": [10.0, 5.0],
    })
    payments = pd.DataFrame({
        "order_id": ["o1"],
        "payment_sequential": [1],
        "payment_type": ["credit_card"],
        "payment_installments": [2],
        "payment_value": [135.0],
    })
    customers = pd.DataFrame({
        "customer_id": ["c1"],
        "customer_state": ["SP"],
        "customer_city": ["sao paulo"],
        "customer_zip_code_prefix": [100],
    })
    sellers = pd.DataFrame({
        "seller_id": ["s1", "s2"],
        "seller_state": ["SP", "RJ"],
        "seller_zip_code_prefix": [200, 300],
    })
    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": [100, 200, 300],
        "geolocation_lat": [0.0, 0.0, 1.0],
        "geolocation_lng": [0.0, 1.0, 1.0],
        "geolocation_city": ["A", "B", "C"],
        "geolocation_state": ["SP", "SP", "RJ"],
    })

    result = build_candidate_order_features(
        orders, items, products, payments, customers, sellers, geolocation
    )

    row = result.iloc[0]
    assert len(result) == 1
    assert row["seller_count"] == 2
    assert row["total_weight_g"] == 1500.0
    assert row["total_volume_cm3"] == 1250.0
    assert row["seller_state"] == "SP"
    assert row["customer_is_capital"] == 1
    assert row["handling_days_min"] == 1.5833333333333333
    assert row["handling_days_max"] == 2.5833333333333335
    assert result["freight_per_kg"].dtype == "float64"


def _network_inputs():
    orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c2", "c3"],
        "order_status": ["delivered", "delivered", "shipped"],
        "order_purchase_timestamp": ["2018-01-01", "2018-03-01", "2018-03-05"],
        "order_approved_at": ["2018-01-01", "2018-03-01", "2018-03-05"],
        "order_delivered_carrier_date": ["2018-01-03", "2018-03-03", "2018-03-06"],
        "order_delivered_customer_date": ["2018-01-20", "2018-03-20", None],
        "order_estimated_delivery_date": ["2018-01-10", "2018-03-12", "2018-03-15"],
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "order_item_id": [1, 1, 1],
        "product_id": ["p1", "p1", "p1"],
        "seller_id": ["s1", "s1", "s1"],
        "price": [10.0, 20.0, 30.0],
        "freight_value": [2.0, 3.0, 4.0],
        "shipping_limit_date": ["2018-01-02", "2018-03-02", "2018-03-07"],
    })
    products = pd.DataFrame({"product_id": ["p1"], "product_category_name": ["a"]})
    payments = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "payment_sequential": [1, 1, 1],
        "payment_type": ["credit_card"] * 3,
        "payment_installments": [1, 1, 1],
        "payment_value": [12.0, 23.0, 34.0],
    })
    customers = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3"],
        "customer_state": ["SP", "SP", "SP"],
        "customer_zip_code_prefix": [100, 100, 100],
    })
    sellers = pd.DataFrame({"seller_id": ["s1"], "seller_state": ["SP"], "seller_zip_code_prefix": [200]})
    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": [100, 200],
        "geolocation_lat": [0.0, 0.0],
        "geolocation_lng": [0.0, 1.0],
        "geolocation_city": ["A", "B"],
        "geolocation_state": ["SP", "SP"],
    })
    return orders, items, products, payments, customers, sellers, geolocation


def test_network_features_are_in_catalog_model_lists_and_order_table():
    from ecom_delivery_sla.modeling import NUMERIC_FEATURES
    from ecom_delivery_sla.network import NETWORK_FEATURE_COLUMNS

    result = build_candidate_order_features(*_network_inputs())
    catalog = feature_catalog().set_index("feature")

    assert result["order_id"].is_unique
    assert set(result["order_id"]) == {"o1", "o2", "o3"}  # unlabeled o3 stays
    for column in NETWORK_FEATURE_COLUMNS:
        assert column in NUMERIC_FEATURES
        assert column in result.columns
        assert catalog.loc[column, "status"] == "available_at_purchase"
        assert catalog.loc[column, "source"] == "network"
    assert catalog.loc["promised_lead_days", "use_in_model"]


def test_network_features_in_order_table_have_no_future_leak():
    result = build_candidate_order_features(*_network_inputs()).set_index("order_id")

    # o1 is the first purchase: no history. o2 sees o1 (delivered 01-20).
    # o3 purchased 03-05 does not see o2, delivered only on 03-20.
    assert result.loc["o1", "seller_cold_start"] == 1
    assert result.loc["o2", "seller_hist_orders"] == 1
    assert result.loc["o3", "seller_hist_orders"] == 1
    assert result.loc["o2", "seller_cold_start"] == 0


def test_distance_range_is_max_minus_min_for_two_sellers_and_zero_for_one():
    orders = pd.DataFrame({
        "order_id": ["o1", "o2"],
        "customer_id": ["c1", "c2"],
        "order_purchase_timestamp": ["2017-11-20 10:00:00", "2017-11-21 10:00:00"],
        "order_estimated_delivery_date": ["2017-11-30", "2017-12-01"],
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o1", "o2"],
        "order_item_id": [1, 2, 1],
        "product_id": ["p1", "p2", "p1"],
        "seller_id": ["s1", "s2", "s1"],
        "price": [100.0, 20.0, 30.0],
        "freight_value": [10.0, 5.0, 4.0],
        "shipping_limit_date": ["2017-11-22", "2017-11-23", "2017-11-22"],
    })
    products = pd.DataFrame({
        "product_id": ["p1", "p2"], "product_category_name": ["a", "b"],
        "product_weight_g": [1000.0, 500.0], "product_length_cm": [10.0, 10.0],
        "product_height_cm": [10.0, 5.0], "product_width_cm": [10.0, 5.0],
    })
    payments = pd.DataFrame({
        "order_id": ["o1", "o2"], "payment_sequential": [1, 1],
        "payment_type": ["credit_card", "boleto"], "payment_installments": [2, 1],
        "payment_value": [135.0, 34.0],
    })
    customers = pd.DataFrame({
        "customer_id": ["c1", "c2"], "customer_state": ["SP", "SP"],
        "customer_city": ["sao paulo", "sao paulo"],
        "customer_zip_code_prefix": [100, 100],
    })
    sellers = pd.DataFrame({
        "seller_id": ["s1", "s2"], "seller_state": ["SP", "RJ"],
        "seller_zip_code_prefix": [200, 300],
    })
    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": [100, 200, 300],
        "geolocation_lat": [0.0, 0.0, 1.0],
        "geolocation_lng": [0.0, 1.0, 1.0],
        "geolocation_city": ["A", "B", "C"],
        "geolocation_state": ["SP", "SP", "RJ"],
    })

    result = build_candidate_order_features(
        orders, items, products, payments, customers, sellers, geolocation
    ).set_index("order_id")

    two = result.loc["o1"]
    assert two["seller_count"] == 2
    assert two["distance_max_km"] > two["distance_min_km"] > 0
    assert two["distance_range_km"] == pytest.approx(
        two["distance_max_km"] - two["distance_min_km"]
    )
    assert result.loc["o2", "distance_range_km"] == 0.0
    entry = feature_catalog().set_index("feature").loc["distance_range_km"]
    assert entry["status"] == "available_at_purchase"
    assert entry["feature_type"] == "numeric"
    assert not entry["use_in_model"]
    from ecom_delivery_sla.modeling import BASELINE_FEATURES

    assert "distance_range_km" not in BASELINE_FEATURES
