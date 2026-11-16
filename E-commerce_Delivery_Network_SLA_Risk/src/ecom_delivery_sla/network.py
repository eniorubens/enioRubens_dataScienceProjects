"""As-of seller-to-state network features for order-level prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .targets import DATE_COLUMNS, add_target_columns

SMOOTHING_STRENGTH = 20.0
DEFAULT_PRIOR = 0.5


def _asof_lookup(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    by: str | list[str] | None = None,
) -> pd.DataFrame:
    """Join the latest strictly earlier event without a Python order loop."""
    by_columns = [by] if isinstance(by, str) else (by or [])
    left_sorted = left.sort_values(left_on).reset_index(drop=True)
    right_sorted = right.sort_values(right_on).reset_index(drop=True)
    return pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_on,
        right_on=right_on,
        by=by,
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_history"),
    ).drop(columns=[column for column in by_columns if column.endswith("_history")], errors="ignore")


def _cumulative_events(
    events: pd.DataFrame,
    group: str | list[str],
    event_time: str = "event_time",
) -> pd.DataFrame:
    """Add cumulative count, breach sum and lead median to event rows."""
    result = events.sort_values(event_time).copy()
    groupby = result.groupby(group, sort=False)
    result["history_n"] = groupby.cumcount() + 1
    result["history_breach_sum"] = groupby["breach"].cumsum()
    result["history_median_lead_days"] = groupby["lead_days"].transform(
        lambda values: values.expanding().median()
    )
    return result


def _smoothed_rate(
    history_sum: pd.Series,
    history_n: pd.Series,
    prior: pd.Series,
) -> pd.Series:
    """Apply an empirical-Bayes-style prior with documented fixed strength."""
    return (history_sum.fillna(0) + SMOOTHING_STRENGTH * prior) / (
        history_n.fillna(0) + SMOOTHING_STRENGTH
    )


def _pair_rows(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Build one current order-seller row with an item-value weight."""
    customer_state = customers[["customer_id", "customer_state"]].drop_duplicates(
        "customer_id"
    )
    rows = (
        items[["order_id", "seller_id", "price"]]
        .merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]], on="order_id", how="left")
        .merge(customer_state, on="customer_id", how="left")
        .assign(item_value=lambda frame: frame["price"].fillna(0.0))
        .groupby(["order_id", "seller_id", "customer_state", "order_purchase_timestamp"], as_index=False)
        .agg(item_value=("item_value", "sum"))
    )
    rows["order_purchase_timestamp"] = pd.to_datetime(
        rows["order_purchase_timestamp"], errors="coerce", format="mixed"
    )
    rows["edge_key"] = rows["seller_id"].astype(str) + "|" + rows["customer_state"].astype(str)
    return rows


def _outcome_events(orders: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Attach delivery outcomes to seller and seller-state exposure rows."""
    outcome_columns = [
        "order_id",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "sla_breach",
        "actual_lead_days",
    ]
    return pairs.merge(orders[outcome_columns], on="order_id", how="inner").rename(
        columns={"order_delivered_customer_date": "event_time"}
    )


def _delivery_events(events: pd.DataFrame) -> pd.DataFrame:
    """Keep delivery outcomes that happen on or after their own purchase.

    A delivery dated before the order's own purchase is temporally impossible
    (``delivery_before_purchase`` in :mod:`audit`) and must not feed history.
    """
    valid = events["event_time"].notna() & (
        events["event_time"] >= events["order_purchase_timestamp"]
    )
    eligible = events.loc[valid].copy()
    eligible["breach"] = eligible["sla_breach"].astype(float)
    eligible["lead_days"] = eligible["actual_lead_days"]
    return eligible


def _dispatch_events(
    pairs: pd.DataFrame, events: pd.DataFrame, items: pd.DataFrame
) -> pd.DataFrame:
    """Build single-seller dispatch-lateness events.

    Dispatches dated before the order's own purchase
    (``dispatch_before_purchase``) are excluded from the history, following
    the audit policy.
    """
    seller_counts = pairs.groupby("order_id")["seller_id"].nunique()
    single_orders = seller_counts[seller_counts == 1].index
    if "shipping_limit_date" not in items:
        items = items.assign(shipping_limit_date=pd.NaT)
    dispatch_limit = (
        items.loc[items["order_id"].isin(single_orders)]
        .assign(
            shipping_limit_date=lambda frame: pd.to_datetime(
                frame["shipping_limit_date"], errors="coerce", format="mixed"
            )
        )
        .groupby("order_id", as_index=False)["shipping_limit_date"]
        .max()
    )
    dispatch = events.merge(dispatch_limit, on="order_id", how="inner")
    valid = (
        dispatch["order_delivered_carrier_date"].notna()
        & dispatch["shipping_limit_date"].notna()
        & (dispatch["order_delivered_carrier_date"] >= dispatch["order_purchase_timestamp"])
    )
    dispatch = dispatch.loc[valid].copy()
    dispatch["event_time"] = dispatch["order_delivered_carrier_date"]
    dispatch["breach"] = (dispatch["event_time"] > dispatch["shipping_limit_date"]).astype(float)
    dispatch["lead_days"] = np.nan
    return dispatch


def _global_asof_priors(
    current_orders: pd.DataFrame,
    delivery_events: pd.DataFrame,
    dispatch_events: pd.DataFrame,
) -> pd.DataFrame:
    """Compute global as-of priors once per order.

    Returns one row per ``order_id`` with the smoothed global breach rate, the
    smoothed global late-dispatch rate and the global median lead time, all
    using only events strictly earlier than the order purchase. Before any
    event exists the rates fall back to ``DEFAULT_PRIOR``.
    """
    base = current_orders[["order_id", "order_purchase_timestamp"]].drop_duplicates(
        "order_id"
    )
    result = base[["order_id"]].copy()
    specs = (
        (delivery_events, "global_breach_prior", True),
        (dispatch_events, "global_dispatch_prior", False),
    )
    for events, name, with_median in specs:
        unique = events.drop_duplicates("order_id").assign(global_key="all")
        cumulative = _cumulative_events(unique, "global_key")
        lookup = _asof_lookup(
            base.assign(global_key="all"),
            cumulative[
                [
                    "global_key",
                    "event_time",
                    "history_n",
                    "history_breach_sum",
                    "history_median_lead_days",
                ]
            ],
            "order_purchase_timestamp",
            "event_time",
            by="global_key",
        )
        rates = pd.DataFrame(
            {
                "order_id": lookup["order_id"],
                name: _smoothed_rate(
                    lookup["history_breach_sum"],
                    lookup["history_n"],
                    pd.Series(DEFAULT_PRIOR, index=lookup.index),
                ),
            }
        )
        if with_median:
            rates["global_hist_median_lead_days"] = lookup["history_median_lead_days"]
        result = result.merge(rates, on="order_id", how="left", validate="one_to_one")
    return result


def _merge_seller_history(
    pairs: pd.DataFrame, delivery_events: pd.DataFrame, priors: pd.DataFrame
) -> pd.DataFrame:
    """Add seller delivery history and seller recency as-of each purchase."""
    seller_events = _cumulative_events(delivery_events, "seller_id")
    current = pairs[["order_id", "seller_id", "order_purchase_timestamp"]].copy()
    lookup = _asof_lookup(
        current,
        seller_events[
            [
                "seller_id",
                "event_time",
                "history_n",
                "history_breach_sum",
                "history_median_lead_days",
            ]
        ],
        "order_purchase_timestamp",
        "event_time",
        by="seller_id",
    ).rename(
        columns={
            "history_n": "seller_hist_orders",
            "history_breach_sum": "seller_hist_breach_sum",
            "history_median_lead_days": "seller_hist_median_lead_days",
            "event_time": "seller_last_outcome",
        }
    )
    lookup = lookup.merge(
        priors[["order_id", "global_breach_prior"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    )
    lookup["seller_hist_breach_rate"] = _smoothed_rate(
        lookup["seller_hist_breach_sum"],
        lookup["seller_hist_orders"],
        lookup["global_breach_prior"],
    )
    lookup["seller_cold_start"] = lookup["seller_hist_orders"].isna().astype("int8")
    lookup["seller_hist_orders"] = lookup["seller_hist_orders"].fillna(0).astype("int64")
    lookup["seller_days_since_last"] = (
        lookup["order_purchase_timestamp"] - lookup["seller_last_outcome"]
    ).dt.total_seconds() / 86_400
    return lookup.drop(
        columns=[
            "seller_hist_breach_sum",
            "global_breach_prior",
            "seller_last_outcome",
            "seller_hist_median_lead_days",
        ]
    )


def _merge_edge_history(
    pairs: pd.DataFrame, delivery_events: pd.DataFrame, priors: pd.DataFrame
) -> pd.DataFrame:
    """Add seller-state edge history as-of the current purchase."""
    edge_events = _cumulative_events(delivery_events, "edge_key")
    current = pairs[["order_id", "edge_key", "order_purchase_timestamp"]]
    lookup = _asof_lookup(
        current,
        edge_events[
            [
                "edge_key",
                "event_time",
                "history_n",
                "history_breach_sum",
                "history_median_lead_days",
            ]
        ],
        "order_purchase_timestamp",
        "event_time",
        by="edge_key",
    ).rename(
        columns={
            "history_n": "edge_hist_orders",
            "history_breach_sum": "edge_hist_breach_sum",
            "history_median_lead_days": "edge_hist_median_lead_days",
        }
    )
    lookup = lookup.merge(
        priors[["order_id", "global_breach_prior"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    )
    lookup["edge_hist_breach_rate"] = _smoothed_rate(
        lookup["edge_hist_breach_sum"],
        lookup["edge_hist_orders"],
        lookup["global_breach_prior"],
    )
    lookup["edge_cold_start"] = lookup["edge_hist_orders"].isna().astype("int8")
    lookup["edge_hist_orders"] = lookup["edge_hist_orders"].fillna(0).astype("int64")
    return lookup.drop(columns=["edge_hist_breach_sum", "global_breach_prior"])


def _merge_destination_history(
    pairs: pd.DataFrame, delivery_events: pd.DataFrame, priors: pd.DataFrame
) -> pd.DataFrame:
    """Add destination-state history and historical graph degree."""
    destination_events = _cumulative_events(delivery_events, "customer_state")
    current = pairs[["order_id", "customer_state", "order_purchase_timestamp"]].drop_duplicates("order_id")
    destination = _asof_lookup(
        current,
        destination_events[
            ["customer_state", "event_time", "history_n", "history_breach_sum", "history_median_lead_days"]
        ],
        "order_purchase_timestamp",
        "event_time",
        by="customer_state",
    ).rename(
        columns={
            "history_n": "dest_hist_orders",
            "history_breach_sum": "dest_hist_breach_sum",
            "history_median_lead_days": "dest_hist_median_lead_days",
        }
    )
    destination = destination.merge(
        priors[["order_id", "global_breach_prior"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    destination["dest_hist_breach_rate"] = _smoothed_rate(
        destination["dest_hist_breach_sum"],
        destination["dest_hist_orders"],
        destination["global_breach_prior"],
    )
    destination = destination.drop(columns=["dest_hist_breach_sum", "global_breach_prior"])
    destination["dest_hist_orders"] = destination["dest_hist_orders"].fillna(0).astype("int64")

    first_edges = (
        delivery_events.groupby(["seller_id", "customer_state"], as_index=False)["event_time"]
        .min()
        .sort_values("event_time")
    )
    first_edges["edge_count"] = first_edges.groupby("seller_id").cumcount() + 1
    first_edges["edge_dest_count"] = first_edges.groupby("customer_state").cumcount() + 1
    seller_degree = _asof_lookup(
        pairs[["order_id", "seller_id", "order_purchase_timestamp"]],
        first_edges[["seller_id", "event_time", "edge_count"]],
        "order_purchase_timestamp",
        "event_time",
        by="seller_id",
    )[["order_id", "edge_count"]].rename(columns={"edge_count": "seller_out_degree"})
    seller_degree = seller_degree.groupby("order_id", as_index=False)[
        "seller_out_degree"
    ].max()
    dest_degree = _asof_lookup(
        current,
        first_edges[["customer_state", "event_time", "edge_dest_count"]],
        "order_purchase_timestamp",
        "event_time",
        by="customer_state",
    )[["order_id", "edge_dest_count"]].rename(columns={"edge_dest_count": "dest_in_degree"})
    return destination.merge(seller_degree, on="order_id", how="left").merge(
        dest_degree, on="order_id", how="left"
    )


def _merge_dispatch_history(
    pairs: pd.DataFrame, dispatch_events: pd.DataFrame, priors: pd.DataFrame
) -> pd.DataFrame:
    """Add single-seller dispatch lateness history as-of purchase."""
    seller_dispatch = _cumulative_events(dispatch_events, "seller_id")
    lookup = _asof_lookup(
        pairs[["order_id", "seller_id", "order_purchase_timestamp"]],
        seller_dispatch[["seller_id", "event_time", "history_n", "history_breach_sum"]],
        "order_purchase_timestamp",
        "event_time",
        by="seller_id",
    ).rename(
        columns={"history_n": "seller_hist_dispatch_orders", "history_breach_sum": "seller_hist_dispatch_late_sum"}
    )
    lookup = lookup.merge(
        priors[["order_id", "global_dispatch_prior"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    )
    lookup["seller_hist_dispatch_late_rate"] = _smoothed_rate(
        lookup["seller_hist_dispatch_late_sum"],
        lookup["seller_hist_dispatch_orders"],
        lookup["global_dispatch_prior"],
    )
    lookup["seller_hist_dispatch_orders"] = lookup[
        "seller_hist_dispatch_orders"
    ].fillna(0).astype("int64")
    return lookup.drop(columns=["seller_hist_dispatch_late_sum", "global_dispatch_prior"])


# Pair-level columns reduced per order. Rates and recency use the worst case
# (max); history counts use the least-evidence seller (min); cold-start flags
# use max so one cold seller marks the whole order.
_MAX_COLUMNS = (
    "seller_hist_breach_rate",
    "seller_cold_start",
    "seller_days_since_last",
    "seller_hist_dispatch_late_rate",
    "edge_hist_breach_rate",
    "edge_cold_start",
    "edge_hist_median_lead_days",
)
_MIN_COLUMNS = ("seller_hist_orders", "edge_hist_orders")
# Continuous columns that also get an item-value weighted mean.
_WEIGHTED_COLUMNS = (
    "seller_hist_breach_rate",
    "seller_hist_dispatch_late_rate",
    "seller_days_since_last",
    "edge_hist_breach_rate",
    "edge_hist_median_lead_days",
)


def _aggregate_pair_features(pair_features: pd.DataFrame) -> pd.DataFrame:
    """Reduce order-seller histories to one row per order without Python loops.

    Rates, recency and cold-start flags use ``max``; history counts use
    ``min``. Continuous columns also get ``<col>_value_weighted_mean`` equal to
    ``sum(value * weight) / sum(weight)`` per order, with weight = item value
    (equal weights when the order total is not positive). Missing values are
    ignored in both numerator and denominator, never filled with zero; the mean
    stays NaN only when every seller value is missing.
    """
    data = pair_features.reset_index(drop=True)
    order_key = data["order_id"]
    weight = data["item_value"].astype(float).clip(lower=0).fillna(0.0)
    order_total = weight.groupby(order_key).transform("sum")
    weight = weight.where(order_total > 0, 1.0)
    aggregations = {column: "max" for column in _MAX_COLUMNS}
    aggregations.update({column: "min" for column in _MIN_COLUMNS})
    reduced = data.groupby("order_id", sort=False).agg(aggregations)
    weighted = {}
    for column in _WEIGHTED_COLUMNS:
        valid = data[column].notna()
        numerator = (data[column] * weight).where(valid).groupby(order_key, sort=False).sum(min_count=1)
        denominator = weight.where(valid).groupby(order_key, sort=False).sum(min_count=1)
        weighted[f"{column}_value_weighted_mean"] = numerator / denominator
    return pd.concat([reduced, pd.DataFrame(weighted)], axis=1).reset_index()


NETWORK_FEATURE_COLUMNS = (
    "seller_hist_orders",
    "seller_hist_breach_rate",
    "seller_cold_start",
    "seller_days_since_last",
    "seller_hist_dispatch_late_rate",
    "edge_hist_orders",
    "edge_hist_breach_rate",
    "edge_cold_start",
    "edge_hist_median_lead_days",
    "dest_hist_breach_rate",
    "dest_hist_median_lead_days",
    "seller_out_degree",
    "dest_in_degree",
    "promise_slack",
)
NETWORK_AUXILIARY_COLUMNS = (
    "seller_hist_breach_rate_value_weighted_mean",
    "edge_hist_breach_rate_value_weighted_mean",
    "seller_hist_dispatch_late_rate_value_weighted_mean",
    "dest_hist_orders",
)


def build_network_features(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build seller-to-destination features using only known prior outcomes.

    A prior outcome is eligible only when its delivery or dispatch event is
    strictly earlier than the current order purchase and not earlier than its
    own purchase. The current order is therefore never part of its own history.
    Rates are smoothed toward the global as-of rate known before each purchase
    (computed once per order). ``shipping_limit_date`` is used only for the
    single-seller dispatch-lateness diagnostic.

    Returns one row per order in ``orders`` with ``NETWORK_FEATURE_COLUMNS``
    and ``NETWORK_AUXILIARY_COLUMNS``.
    """
    del sellers  # reserved for future seller metadata; IDs are in order_items
    outcome_columns = [column for column in DATE_COLUMNS if column not in orders]
    prepared = add_target_columns(orders.assign(**dict.fromkeys(outcome_columns, pd.NaT)))
    pairs = _pair_rows(prepared, items, customers)
    events = _outcome_events(prepared, pairs)
    delivery_events = _delivery_events(events)
    dispatch_events = _dispatch_events(pairs, events, items)
    current_orders = pairs[["order_id", "order_purchase_timestamp"]].drop_duplicates("order_id")
    priors = _global_asof_priors(current_orders, delivery_events, dispatch_events)
    seller = _merge_seller_history(pairs, delivery_events, priors)
    edge = _merge_edge_history(pairs, delivery_events, priors)
    destination = _merge_destination_history(pairs, delivery_events, priors)
    dispatch = _merge_dispatch_history(pairs, dispatch_events, priors)
    pair_features = (
        pairs.merge(seller, on=["order_id", "seller_id", "order_purchase_timestamp"], how="left")
        .merge(edge, on=["order_id", "edge_key", "order_purchase_timestamp"], how="left")
        .merge(dispatch, on=["order_id", "seller_id", "order_purchase_timestamp"], how="left")
    )
    aggregated = _aggregate_pair_features(pair_features)
    result = aggregated.merge(destination, on="order_id", how="left")
    result = result.merge(
        prepared[["order_id", "promised_lead_days"]], on="order_id", how="left"
    )
    result = result.merge(
        priors[["order_id", "global_hist_median_lead_days"]], on="order_id", how="left"
    )
    edge_median = result["edge_hist_median_lead_days"].fillna(
        result["dest_hist_median_lead_days"]
    )
    result["promise_slack"] = result["promised_lead_days"] - edge_median.fillna(
        result["global_hist_median_lead_days"]
    )
    columns = ["order_id", *NETWORK_FEATURE_COLUMNS, *NETWORK_AUXILIARY_COLUMNS]
    return prepared[["order_id"]].merge(result[columns], on="order_id", how="left")
