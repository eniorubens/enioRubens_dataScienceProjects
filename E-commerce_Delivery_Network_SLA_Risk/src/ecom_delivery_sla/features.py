"""Order-level features and the temporal feature-availability catalog."""

from __future__ import annotations

import unicodedata

import pandas as pd

from .geo import build_order_geography
from .network import NETWORK_AUXILIARY_COLUMNS, NETWORK_FEATURE_COLUMNS, build_network_features

EVENT_DATES = {
    "black_friday": {
        2016: pd.Timestamp("2016-11-25"),
        2017: pd.Timestamp("2017-11-24"),
        2018: pd.Timestamp("2018-11-23"),
        2019: pd.Timestamp("2019-11-29"),
    },
    "christmas": {
        year: pd.Timestamp(f"{year}-12-25") for year in range(2016, 2020)
    },
}

STATE_CAPITALS = {
    "AC": "rio branco", "AL": "maceio", "AP": "macapa", "AM": "manaus",
    "BA": "salvador", "CE": "fortaleza", "DF": "brasilia", "ES": "vitoria",
    "GO": "goiania", "MA": "sao luis", "MT": "cuiaba", "MS": "campo grande",
    "MG": "belo horizonte", "PA": "belem", "PB": "joao pessoa", "PR": "curitiba",
    "PE": "recife", "PI": "teresina", "RJ": "rio de janeiro", "RN": "natal",
    "RS": "porto alegre", "RO": "porto velho", "RR": "boa vista", "SC": "florianopolis",
    "SP": "sao paulo", "SE": "aracaju", "TO": "palmas",
}

FEATURE_CATALOG = pd.DataFrame(
    [
        ("order_id", "order", "quality_control", "identifier", False, "Order identifier"),
        ("customer_id", "customer", "quality_control", "identifier", False, "Order-level customer identifier"),
        ("primary_seller_id", "sellers", "quality_control", "identifier", False, "Seller with highest item value in the order"),
        ("order_purchase_timestamp", "order", "available_at_purchase", "datetime", False, "Purchase timestamp"),
        ("order_estimated_delivery_date", "order", "available_at_purchase", "datetime", False, "Estimated delivery date in the order data"),
        ("promised_lead_days", "order", "available_at_purchase", "numeric", True, "Estimated promise minus purchase"),
        ("customer_state", "customer", "available_at_purchase", "categorical", True, "Customer destination state"),
        ("customer_is_capital", "customer", "available_at_purchase", "numeric", True, "Customer city matches the state capital"),
        ("item_count", "order_items", "available_at_purchase", "numeric", True, "Number of item rows"),
        ("distinct_product_count", "order_items", "available_at_purchase", "numeric", True, "Number of distinct products"),
        ("distinct_category_count", "products", "available_at_purchase", "numeric", True, "Number of distinct categories"),
        ("seller_count", "order_items", "available_at_purchase", "numeric", True, "Number of sellers in the order"),
        ("origin_state_count", "sellers", "available_at_purchase", "numeric", True, "Number of seller-origin states"),
        ("seller_state", "sellers", "available_at_purchase", "categorical", True, "State of the seller with highest item value"),
        ("total_price", "order_items", "available_at_purchase", "numeric", True, "Sum of item prices"),
        ("total_freight_value", "order_items", "available_at_purchase", "numeric", True, "Sum of item freight values"),
        ("mean_item_price", "order_items", "available_at_purchase", "numeric", True, "Mean item price"),
        ("total_weight_g", "products", "available_at_purchase", "numeric", True, "Sum of product weights in grams"),
        ("max_weight_g", "products", "available_at_purchase", "numeric", True, "Maximum product weight in grams"),
        ("total_volume_cm3", "products", "available_at_purchase", "numeric", True, "Sum of product volumes in cubic centimetres"),
        ("freight_per_kg", "order_items", "available_at_purchase", "numeric", True, "Total freight divided by total weight in kilograms"),
        ("payment_count", "payments", "available_at_purchase", "numeric", True, "Number of payment records"),
        ("payment_installments_max", "payments", "available_at_purchase", "numeric", True, "Maximum installment count"),
        ("primary_payment_type", "payments", "available_at_purchase", "categorical", True, "Most frequent payment type"),
        ("distance_mean_km", "geolocation", "available_at_purchase", "numeric", True, "Mean approximate seller-customer Haversine distance"),
        ("same_state_share", "geolocation", "available_at_purchase", "numeric", True, "Share of item rows with seller and customer in the same state"),
        ("purchase_month", "order", "available_at_purchase", "numeric", True, "Calendar month of purchase"),
        ("purchase_dayofweek", "order", "available_at_purchase", "numeric", True, "Day of week of purchase, Monday=0"),
        ("purchase_hour", "order", "available_at_purchase", "numeric", True, "Hour of purchase"),
        ("days_to_black_friday", "calendar", "available_at_purchase", "numeric", True, "Days until the next fixed Black Friday date"),
        ("days_to_christmas", "calendar", "available_at_purchase", "numeric", True, "Days until the next fixed Christmas date"),
        ("seller_hist_orders", "network", "available_at_purchase", "numeric", True, "Prior delivered orders of the least-experienced seller, as-of purchase"),
        ("seller_hist_breach_rate", "network", "available_at_purchase", "numeric", True, "Smoothed prior SLA-breach rate of the seller (max over sellers), as-of purchase"),
        ("seller_cold_start", "network", "available_at_purchase", "numeric", True, "Any seller in the order has no prior delivered order"),
        ("seller_days_since_last", "network", "available_at_purchase", "numeric", True, "Days since the seller's latest known delivery (max over sellers)"),
        ("seller_hist_dispatch_late_rate", "network", "available_at_purchase", "numeric", True, "Smoothed prior late-dispatch rate of the seller, as-of purchase"),
        ("edge_hist_orders", "network", "available_at_purchase", "numeric", True, "Prior delivered orders on the seller-to-state edge (min over sellers)"),
        ("edge_hist_breach_rate", "network", "available_at_purchase", "numeric", True, "Smoothed prior SLA-breach rate of the seller-to-state edge"),
        ("edge_cold_start", "network", "available_at_purchase", "numeric", True, "Any seller-to-state edge in the order has no prior delivered order"),
        ("edge_hist_median_lead_days", "network", "available_at_purchase", "numeric", True, "Prior median actual lead time on the seller-to-state edge"),
        ("dest_hist_breach_rate", "network", "available_at_purchase", "numeric", True, "Smoothed prior SLA-breach rate of the destination state"),
        ("dest_hist_median_lead_days", "network", "available_at_purchase", "numeric", True, "Prior median actual lead time of the destination state"),
        ("seller_out_degree", "network", "available_at_purchase", "numeric", True, "Number of destination states already served by the seller (max over sellers)"),
        ("dest_in_degree", "network", "available_at_purchase", "numeric", True, "Number of seller-destination edges already seen for the destination state"),
        ("promise_slack", "network", "available_at_purchase", "numeric", True, "Promised lead time minus prior median lead time (edge, then destination, then global)"),
        ("promise_vs_recent_median", "order", "available_at_purchase", "numeric", False, "Promised lead time divided by the median promise of the same destination state in the previous 30 days (F2 post-hoc variant E only; computed by posthoc.promise_vs_recent_median)"),
        ("seller_hist_breach_rate_value_weighted_mean", "network", "available_at_purchase", "numeric", False, "Item-value weighted seller breach rate; redundant with the max for single-seller orders"),
        ("edge_hist_breach_rate_value_weighted_mean", "network", "available_at_purchase", "numeric", False, "Item-value weighted edge breach rate; redundant with the max for single-seller orders"),
        ("seller_hist_dispatch_late_rate_value_weighted_mean", "network", "available_at_purchase", "numeric", False, "Item-value weighted late-dispatch rate; redundant with the max for single-seller orders"),
        ("dest_hist_orders", "network", "available_at_purchase", "numeric", False, "Prior delivered orders of the destination state (audit only)"),
        ("handling_days_min", "order_items", "available_at_approval", "numeric", False, "Minimum shipping-limit minus purchase in days"),
        ("handling_days_max", "order_items", "available_at_approval", "numeric", False, "Maximum shipping-limit minus purchase in days"),
        ("payment_value_sum", "payments", "quality_control", "numeric", False, "Payment sum retained only for redundancy audit"),
        ("distance_min_km", "geolocation", "quality_control", "numeric", False, "Minimum distance retained only for redundancy audit"),
        ("distance_max_km", "geolocation", "quality_control", "numeric", False, "Maximum distance retained only for redundancy audit"),
        ("distance_range_km", "geolocation", "available_at_purchase", "numeric", False, "Maximum minus minimum seller-customer distance (candidate for the distance ablation)"),
        ("coordinate_coverage", "geolocation", "quality_control", "numeric", False, "Share of item rows with coordinates"),
        ("distance_missing", "geolocation", "quality_control", "numeric", False, "Order has no mean distance"),
        ("payment_record_missing", "payments", "quality_control", "numeric", False, "Order has no payment record"),
    ],
    columns=["feature", "source", "status", "feature_type", "use_in_model", "definition"],
)

MISSINGNESS_ACTIONS = {
    "distance_mean_km": "retain_order_impute_on_train_and_add_missing_indicator",
    "payment_count": "retain_order_impute_on_train_and_add_missing_indicator",
    "payment_installments_max": "retain_order_impute_on_train_and_add_missing_indicator",
    "primary_payment_type": "retain_order_impute_on_train_and_add_missing_indicator",
    "coordinate_coverage": "retain_as_quality_control",
}


def feature_catalog() -> pd.DataFrame:
    """Return the feature catalog with temporal availability and usage flags."""
    return FEATURE_CATALOG.copy()


def feature_quality_summary(candidate_features: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and the approved handling action.

    Parameters
    ----------
    candidate_features:
        Order-level feature table.

    Returns
    -------
    pandas.DataFrame
        One row per feature with missingness, catalog status, and handling action.
    """
    quality = pd.DataFrame(
        {
            "feature": candidate_features.columns,
            "missing": candidate_features.isna().sum().to_numpy(),
            "missing_rate": candidate_features.isna().mean().to_numpy(),
            "data_type": candidate_features.dtypes.astype(str).to_numpy(),
        }
    )
    summary = quality.merge(FEATURE_CATALOG, on="feature", how="left")
    summary["missing_action"] = summary["feature"].map(
        MISSINGNESS_ACTIONS
    ).fillna("no_missing_observed_or_review_if_missing")
    return summary


def _mode(values: pd.Series) -> object:
    modes = values.dropna().mode()
    return modes.iloc[0] if not modes.empty else pd.NA


def _normalise_city(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    return "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _days_to_event(purchases: pd.Series, event_name: str) -> pd.Series:
    """Return non-negative days to the next fixed event date."""
    event_map = EVENT_DATES[event_name]

    def next_date(value: object) -> float:
        if pd.isna(value):
            return float("nan")
        timestamp = pd.Timestamp(value).normalize()
        for year in range(timestamp.year, timestamp.year + 3):
            event_date = event_map.get(year)
            if event_date is not None and event_date >= timestamp:
                return float((event_date - timestamp).days)
        return float("nan")

    return purchases.map(next_date)


def _ensure_product_columns(products: pd.DataFrame) -> pd.DataFrame:
    result = products.copy()
    for column in (
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ):
        if column not in result:
            result[column] = pd.NA
    return result


def build_order_features(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    payments: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    geolocation: pd.DataFrame,
) -> pd.DataFrame:
    """Build one feature row per order without attaching delivery targets.

    ``shipping_limit_date`` is included only as an approval-time diagnostic.
    It is marked ``available_at_approval`` and excluded from the purchase-time
    modeling matrix. Seller/edge/destination network features are joined from
    :func:`build_network_features`; they use only delivery and dispatch
    outcomes known strictly before each purchase, so every order in
    ``orders`` (including unlabeled ones) may be passed.
    """
    if orders["order_id"].duplicated().any():
        raise ValueError("orders must contain one row per order_id")

    product_columns = [
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    product_data = _ensure_product_columns(products)[product_columns]
    item_data = items.merge(product_data, on="product_id", how="left")
    purchase_lookup = orders[["order_id", "order_purchase_timestamp"]].copy()
    purchase_lookup["order_purchase_timestamp"] = pd.to_datetime(
        purchase_lookup["order_purchase_timestamp"], errors="coerce"
    )
    item_data = item_data.merge(purchase_lookup, on="order_id", how="left")
    item_data["product_volume_cm3"] = item_data[
        ["product_length_cm", "product_height_cm", "product_width_cm"]
    ].prod(axis=1, min_count=3)
    shipping_limit = item_data["shipping_limit_date"] if "shipping_limit_date" in item_data else pd.Series(pd.NaT, index=item_data.index)
    item_data["handling_days"] = (
        pd.to_datetime(shipping_limit, errors="coerce")
        - item_data["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400

    item_features = (
        item_data.groupby("order_id")
        .agg(
            item_count=("order_item_id", "size"),
            distinct_product_count=("product_id", "nunique"),
            distinct_category_count=("product_category_name", "nunique"),
            total_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            mean_item_price=("price", "mean"),
            total_weight_g=("product_weight_g", lambda values: values.sum(min_count=1)),
            max_weight_g=("product_weight_g", "max"),
            total_volume_cm3=("product_volume_cm3", lambda values: values.sum(min_count=1)),
            handling_days_min=("handling_days", "min"),
            handling_days_max=("handling_days", "max"),
        )
        .reset_index()
    )
    weight_kg = item_features["total_weight_g"].astype("float64").div(1_000)
    item_features["freight_per_kg"] = item_features[
        "total_freight_value"
    ].astype("float64").div(weight_kg.where(weight_kg > 0))

    seller_values = (
        item_data.assign(item_value=item_data["price"].fillna(0))
        .groupby(["order_id", "seller_id"], as_index=False)["item_value"]
        .sum()
        .sort_values(["order_id", "item_value", "seller_id"], ascending=[True, False, True])
        .drop_duplicates("order_id")
        .merge(
            sellers[["seller_id", "seller_state"]].drop_duplicates("seller_id"),
            on="seller_id",
            how="left",
        )[["order_id", "seller_id", "seller_state"]]
        .rename(columns={"seller_id": "primary_seller_id"})
    )
    payment_features = (
        payments.groupby("order_id")
        .agg(
            payment_count=("payment_sequential", "size"),
            payment_value_sum=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
            primary_payment_type=("payment_type", _mode),
        )
        .reset_index()
    )
    customer_columns = ["customer_id", "customer_state"]
    if "customer_city" in customers:
        customer_columns.append("customer_city")
    customer_features = customers[customer_columns].drop_duplicates("customer_id")
    geography = build_order_geography(
        orders, items, customers, sellers, products, geolocation
    )
    geography_columns = [
        "order_id",
        "seller_count",
        "origin_state_count",
        "distance_min_km",
        "distance_mean_km",
        "distance_max_km",
        "coordinate_coverage",
        "same_state_share",
    ]
    result = (
        orders[
            [
                "order_id",
                "customer_id",
                "order_purchase_timestamp",
                "order_estimated_delivery_date",
            ]
        ]
        .merge(customer_features, on="customer_id", how="left")
        .merge(item_features, on="order_id", how="left")
        .merge(payment_features, on="order_id", how="left")
        .merge(seller_values, on="order_id", how="left")
        .merge(geography[geography_columns], on="order_id", how="left")
    )
    result["distance_range_km"] = result["distance_max_km"] - result["distance_min_km"]
    result["distance_missing"] = result["distance_mean_km"].isna().astype("int8")
    result["payment_record_missing"] = result["payment_count"].isna().astype("int8")
    purchase = pd.to_datetime(result["order_purchase_timestamp"], errors="coerce")
    estimated = pd.to_datetime(result["order_estimated_delivery_date"], errors="coerce")
    result["promised_lead_days"] = (estimated - purchase).dt.total_seconds() / 86_400
    result["purchase_month"] = purchase.dt.month
    result["purchase_dayofweek"] = purchase.dt.dayofweek
    result["purchase_hour"] = purchase.dt.hour
    result["days_to_black_friday"] = _days_to_event(purchase, "black_friday")
    result["days_to_christmas"] = _days_to_event(purchase, "christmas")
    if "customer_city" in result:
        result["customer_is_capital"] = [
            int(_normalise_city(city) == STATE_CAPITALS.get(state, ""))
            if not pd.isna(city) and not pd.isna(state) else pd.NA
            for city, state in zip(
                result["customer_city"], result["customer_state"], strict=False
            )
        ]
    else:
        result["customer_is_capital"] = pd.NA
    network = build_network_features(orders, items, customers, sellers)
    result = result.merge(
        network[["order_id", *NETWORK_FEATURE_COLUMNS, *NETWORK_AUXILIARY_COLUMNS]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    return result


def build_candidate_order_features(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    payments: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    geolocation: pd.DataFrame,
) -> pd.DataFrame:
    """Backward-compatible alias for the all-orders scoring feature builder."""
    return build_order_features(
        orders, items, products, payments, customers, sellers, geolocation
    )
