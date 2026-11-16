"""Geographic and order-composition features for exploratory analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from .labels import get_labels

EARTH_RADIUS_KM = 6_371.0088


def haversine_km(
    latitude_1: pd.Series | np.ndarray,
    longitude_1: pd.Series | np.ndarray,
    latitude_2: pd.Series | np.ndarray,
    longitude_2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Calculate great-circle distance in kilometres between coordinate pairs."""
    lat_1 = np.radians(np.asarray(latitude_1, dtype=float))
    lon_1 = np.radians(np.asarray(longitude_1, dtype=float))
    lat_2 = np.radians(np.asarray(latitude_2, dtype=float))
    lon_2 = np.radians(np.asarray(longitude_2, dtype=float))
    delta_lat = lat_2 - lat_1
    delta_lon = lon_2 - lon_1
    haversine = np.sin(delta_lat / 2) ** 2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


def aggregate_geolocation(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated CEP-prefix observations into approximate coordinates."""

    def modal_value(values: pd.Series) -> object:
        modes = values.dropna().mode()
        return modes.iloc[0] if not modes.empty else pd.NA

    return (
        geolocation.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            geolocation_lat=("geolocation_lat", "median"),
            geolocation_lng=("geolocation_lng", "median"),
            geolocation_city=("geolocation_city", modal_value),
            geolocation_state=("geolocation_state", modal_value),
        )
    )


def build_order_geography(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    products: pd.DataFrame,
    geolocation: pd.DataFrame,
) -> pd.DataFrame:
    """Build order-level distance and composition metrics without changing the target grain."""
    geo = aggregate_geolocation(geolocation)
    customer_geo = customers.merge(
        geo,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(
        columns={
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
            "geolocation_state": "customer_geo_state",
        }
    )
    seller_geo = sellers.merge(
        geo,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(
        columns={
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
            "geolocation_state": "seller_geo_state",
        }
    )
    item_rows = (
        items.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
        .merge(customer_geo, on="customer_id", how="left")
        .merge(seller_geo, on="seller_id", how="left")
        .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    )
    valid_coordinates = item_rows[["seller_lat", "seller_lng", "customer_lat", "customer_lng"]].notna().all(axis=1)
    item_rows["distance_km"] = np.nan
    item_rows.loc[valid_coordinates, "distance_km"] = haversine_km(
        item_rows.loc[valid_coordinates, "seller_lat"],
        item_rows.loc[valid_coordinates, "seller_lng"],
        item_rows.loc[valid_coordinates, "customer_lat"],
        item_rows.loc[valid_coordinates, "customer_lng"],
    )
    item_rows["same_state"] = item_rows["seller_state"].eq(item_rows["customer_state"])
    order_metrics = (
        item_rows.groupby("order_id")
        .agg(
            item_count=("order_item_id", "size"),
            distinct_product_count=("product_id", "nunique"),
            distinct_category_count=("product_category_name", "nunique"),
            seller_count=("seller_id", "nunique"),
            origin_state_count=("seller_state", "nunique"),
            distance_min_km=("distance_km", "min"),
            distance_mean_km=("distance_km", "mean"),
            distance_max_km=("distance_km", "max"),
            coordinate_coverage=("distance_km", lambda values: values.notna().mean()),
            same_state_share=("same_state", "mean"),
        )
        .reset_index()
    )
    return orders.merge(order_metrics, on="order_id", how="left")


def build_route_demo_pairs(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    geolocation: pd.DataFrame,
) -> pd.DataFrame:
    """Return single-seller order pairs suitable for a route demonstration."""
    geo = aggregate_geolocation(geolocation)
    customer_geo = customers.merge(
        geo,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"})
    seller_geo = sellers.merge(
        geo,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"})
    item_rows = (
        items.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
        .merge(customer_geo, on="customer_id", how="left")
        .merge(seller_geo, on="seller_id", how="left")
    )
    seller_counts = item_rows.groupby("order_id")["seller_id"].nunique()
    single_seller_orders = seller_counts[seller_counts == 1].index
    item_rows = item_rows[item_rows["order_id"].isin(single_seller_orders)].copy()
    valid = item_rows[["seller_lat", "seller_lng", "customer_lat", "customer_lng"]].notna().all(axis=1)
    item_rows = item_rows.loc[valid].copy()
    item_rows["haversine_distance_km"] = haversine_km(
        item_rows["seller_lat"],
        item_rows["seller_lng"],
        item_rows["customer_lat"],
        item_rows["customer_lng"],
    )
    return (
        item_rows.sort_values(["order_id", "order_item_id"])
        .drop_duplicates("order_id")
        [[
            "order_id",
            "customer_id",
            "seller_id",
            "seller_lat",
            "seller_lng",
            "customer_lat",
            "customer_lng",
            "haversine_distance_km",
        ]]
        .reset_index(drop=True)
    )


def distance_band_summary(
    order_context: pd.DataFrame,
    bins: tuple[float, ...] = (0, 100, 300, 600, 1_000, np.inf),
) -> pd.DataFrame:
    """Summarize order volume and breach rate by mean distance band."""
    data = order_context.copy()
    labels = ["0-100", "100-300", "300-600", "600-1000", "1000+"]
    data["distance_band_km"] = pd.cut(data["distance_mean_km"], bins=bins, labels=labels, right=False)
    result = data.groupby("distance_band_km", observed=False).agg(
        orders=("order_id", "size"),
        breaches=("sla_breach", "sum"),
        breach_rate=("sla_breach", "mean"),
        median_delay=("delivery_delay_days", "median"),
    )
    return result.reset_index()


def composition_summary(order_context: pd.DataFrame, dimension: str = "category") -> pd.DataFrame:
    """Summarize breach rate by a selected order-composition dimension."""
    data = order_context.copy()
    definitions = {
        "items": ("item_count", [0, 1, 2, 3, 4, 5, np.inf], ["1 item", "2 items", "3 items", "4 items", "5 items", "6+ items"]),
        "products": ("distinct_product_count", [0, 1, 2, 3, 4, 5, np.inf], ["1 product", "2 products", "3 products", "4 products", "5 products", "6+ products"]),
        "category": ("distinct_category_count", [0, 1, 2, 3, np.inf], ["1 category", "2 categories", "3 categories", "4+ categories"]),
        "sellers": ("seller_count", [0, 1, 2, 3, np.inf], ["1 seller", "2 sellers", "3 sellers", "4+ sellers"]),
    }
    if dimension not in definitions:
        raise ValueError(f"dimension must be one of {sorted(definitions)}")
    column, bins, labels = definitions[dimension]
    data["composition_band"] = pd.cut(data[column], bins=bins, labels=labels, right=True)
    return (
        data.groupby("composition_band", observed=False)
        .agg(
            orders=("order_id", "size"),
            breaches=("sla_breach", "sum"),
            breach_rate=("sla_breach", "mean"),
            median_delay=("delivery_delay_days", "median"),
            median_items=("item_count", "median"),
            median_sellers=("seller_count", "median"),
        )
        .reset_index()
    )


def plot_distance_band_breach(order_context: pd.DataFrame, language: str = "pt") -> object:
    """Plot descriptive breach rates across approximate distance bands."""
    labels = get_labels(language)
    summary = distance_band_summary(order_context)
    return px.bar(
        summary,
        x="distance_band_km",
        y="breach_rate",
        text="breach_rate",
        labels={"distance_band_km": labels["geo"]["distance_band"], "breach_rate": labels["geo"]["breach_rate"]},
        title=labels["geo"]["distance_title"],
    ).update_traces(texttemplate="%{text:.1%}", textposition="outside")


def plot_composition_breach(
    order_context: pd.DataFrame,
    language: str = "pt",
    dimension: str = "category",
) -> object:
    """Plot descriptive breach rates across product-category composition bands."""
    labels = get_labels(language)
    summary = composition_summary(order_context, dimension=dimension)
    title = labels["geo"][f"{dimension}_title"]
    return px.bar(
        summary,
        x="composition_band",
        y="breach_rate",
        text="breach_rate",
        labels={"composition_band": labels["geo"]["composition"], "breach_rate": labels["geo"]["breach_rate"]},
        title=title,
    ).update_traces(texttemplate="%{text:.1%}", textposition="outside")
