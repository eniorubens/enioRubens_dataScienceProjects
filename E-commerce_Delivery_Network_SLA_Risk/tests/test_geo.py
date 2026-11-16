import numpy as np
import pandas as pd

from ecom_delivery_sla.geo import build_order_geography, build_route_demo_pairs, composition_summary, haversine_km


def test_haversine_returns_zero_for_same_point():
    distance = haversine_km(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))

    assert distance[0] == 0.0


def test_order_geography_keeps_one_row_per_order_and_counts_composition():
    orders = pd.DataFrame({
        "order_id": ["o1"],
        "customer_id": ["c1"],
        "sla_breach": [True],
        "delivery_delay_days": [2.0],
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o1"],
        "order_item_id": [1, 2],
        "product_id": ["p1", "p2"],
        "seller_id": ["s1", "s2"],
    })
    customers = pd.DataFrame({
        "customer_id": ["c1"],
        "customer_zip_code_prefix": [100],
        "customer_state": ["SP"],
    })
    sellers = pd.DataFrame({
        "seller_id": ["s1", "s2"],
        "seller_zip_code_prefix": [200, 300],
        "seller_state": ["SP", "RJ"],
    })
    products = pd.DataFrame({
        "product_id": ["p1", "p2"],
        "product_category_name": ["a", "b"],
    })
    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": [100, 200, 300],
        "geolocation_lat": [0.0, 0.0, 1.0],
        "geolocation_lng": [0.0, 1.0, 0.0],
        "geolocation_city": ["A", "B", "C"],
        "geolocation_state": ["SP", "SP", "RJ"],
    })

    result = build_order_geography(orders, items, customers, sellers, products, geolocation)

    assert len(result) == 1
    assert result.loc[0, "item_count"] == 2
    assert result.loc[0, "distinct_category_count"] == 2
    assert result.loc[0, "seller_count"] == 2
    assert result.loc[0, "origin_state_count"] == 2

    composition = composition_summary(result, dimension="products")

    assert composition["orders"].sum() == 1


def test_route_demo_pairs_uses_single_seller_orders():
    orders = pd.DataFrame({"order_id": ["o1"], "customer_id": ["c1"]})
    items = pd.DataFrame({"order_id": ["o1"], "order_item_id": [1], "seller_id": ["s1"]})
    customers = pd.DataFrame({"customer_id": ["c1"], "customer_zip_code_prefix": [100]})
    sellers = pd.DataFrame({"seller_id": ["s1"], "seller_zip_code_prefix": [200]})
    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": [100, 200],
        "geolocation_lat": [0.0, 0.0],
        "geolocation_lng": [0.0, 1.0],
        "geolocation_city": ["A", "B"],
        "geolocation_state": ["SP", "SP"],
    })

    result = build_route_demo_pairs(orders, items, customers, sellers, geolocation)

    assert len(result) == 1
    assert result.loc[0, "haversine_distance_km"] > 100
