"""Temporal coverage and leakage-aware split helpers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .labels import get_labels
from .targets import DATE_COLUMNS, add_target_columns


def _parsed_timestamp(data: pd.DataFrame, timestamp_col: str) -> pd.Series:
    if timestamp_col not in data.columns:
        raise KeyError(f"timestamp column not found: {timestamp_col}")
    timestamp = pd.to_datetime(data[timestamp_col], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"timestamp column contains invalid or missing values: {timestamp_col}")
    return timestamp


def temporal_monthly_summary(
    data: pd.DataFrame,
    timestamp_col: str = "order_purchase_timestamp",
) -> pd.DataFrame:
    """Summarize order volume and cumulative share by purchase month."""
    timestamp = _parsed_timestamp(data, timestamp_col)
    summary = (
        timestamp.dt.to_period("M")
        .value_counts()
        .sort_index()
        .rename_axis("purchase_month")
        .rename("orders")
        .reset_index()
    )
    summary["purchase_month"] = summary["purchase_month"].astype(str)
    summary["cumulative_share"] = summary["orders"].cumsum() / len(data)
    return summary


def temporal_split(
    data: pd.DataFrame,
    timestamp_col: str,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Split rows chronologically into train, validation, and test partitions.

    ``train_end`` and ``validation_end`` are exclusive boundaries. The test
    partition contains rows on or after ``validation_end``. The function does
    not shuffle or reset indices, preserving traceability to the source table.
    """
    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must be earlier than validation_end")

    timestamp = _parsed_timestamp(data, timestamp_col)
    masks = {
        "train": timestamp < train_cutoff,
        "validation": (timestamp >= train_cutoff) & (timestamp < validation_cutoff),
        "test": timestamp >= validation_cutoff,
    }
    splits = {name: data.loc[mask].copy() for name, mask in masks.items()}
    empty = [name for name, frame in splits.items() if frame.empty]
    if empty:
        raise ValueError(f"temporal split produced empty partitions: {empty}")
    return splits


def temporal_split_summary(
    splits: dict[str, pd.DataFrame],
    timestamp_col: str = "order_purchase_timestamp",
) -> pd.DataFrame:
    """Summarize size and date range for temporal partitions."""
    total = sum(len(frame) for frame in splits.values())
    rows: list[dict[str, object]] = []
    for split_name, frame in splits.items():
        timestamp = _parsed_timestamp(frame, timestamp_col)
        rows.append(
            {
                "split": split_name,
                "orders": len(frame),
                "share": len(frame) / total,
                "first_purchase": timestamp.min(),
                "last_purchase": timestamp.max(),
            }
        )
    return pd.DataFrame(rows)


def expanding_window_folds(
    data: pd.DataFrame,
    timestamp_col: str = "order_purchase_timestamp",
    first_val_month: str | pd.Period = "2017-10",
    last_val_month: str | pd.Period = "2018-06",
    embargo_days: int = 30,
    test_start: str | pd.Timestamp = "2018-07-01",
) -> list[dict[str, pd.DataFrame]]:
    """Create monthly expanding-window folds without exposing the holdout.

    Each validation fold is one calendar month. Training rows end strictly
    before the validation start minus ``embargo_days``. Rows on or after
    ``test_start`` are never returned, even if a caller requests a later
    validation month.
    """
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    timestamp = _parsed_timestamp(data, timestamp_col)
    first = pd.Period(first_val_month, freq="M")
    last = pd.Period(last_val_month, freq="M")
    if first > last:
        raise ValueError("first_val_month must not be later than last_val_month")
    holdout_start = pd.Timestamp(test_start)
    folds: list[dict[str, pd.DataFrame]] = []
    for period in pd.period_range(first, last, freq="M"):
        validation_start = period.start_time
        validation_end = period.end_time + pd.Timedelta(nanoseconds=1)
        train_end = validation_start - pd.Timedelta(days=embargo_days)
        train_mask = timestamp < train_end
        validation_mask = (
            (timestamp >= validation_start)
            & (timestamp < validation_end)
            & (timestamp < holdout_start)
        )
        train = data.loc[train_mask & (timestamp < holdout_start)].copy()
        validation = data.loc[validation_mask].copy()
        if train.empty or validation.empty:
            continue
        folds.append(
            {
                "fold": period.strftime("%Y-%m"),
                "train": train,
                "validation": validation,
            }
        )
    if not folds:
        raise ValueError("no non-empty expanding-window folds were produced")
    return folds


def expanding_fold_summary(
    folds: list[dict[str, pd.DataFrame]],
    timestamp_col: str = "order_purchase_timestamp",
    target_col: str = "sla_breach",
) -> pd.DataFrame:
    """Summarize each expanding-window fold: periods, sizes, and prevalence."""
    rows = []
    for fold in folds:
        train_time = _parsed_timestamp(fold["train"], timestamp_col)
        validation_time = _parsed_timestamp(fold["validation"], timestamp_col)
        rows.append(
            {
                "fold": fold["fold"],
                "train_start": train_time.min(),
                "train_end": train_time.max(),
                "validation_start": validation_time.min(),
                "validation_end": validation_time.max(),
                "train_orders": len(fold["train"]),
                "validation_orders": len(fold["validation"]),
                "train_prevalence": float(fold["train"][target_col].astype(float).mean()),
                "validation_prevalence": float(
                    fold["validation"][target_col].astype(float).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def right_censoring_sensitivity(
    orders: pd.DataFrame,
    start: str | pd.Timestamp = "2018-07-01",
    end: str | pd.Timestamp = "2018-09-01",
) -> pd.DataFrame:
    """Describe how right censoring changes the label population in a period.

    Shipped orders without a delivery date are censored: their outcome is not
    yet observed. Two scenarios are reported, with descriptive counts only (no
    model metrics and no manufactured labels in any modeling frame):

    ``exclude``
        Supervised population = delivered orders with valid dates; censored
        orders are left out (the project's main scenario).
    ``include_as_breach``
        Upper-bound sensitivity that counts every censored order as a breach.

    Returns one row per scenario with ``supervised_orders``, ``breaches``,
    ``censored_orders`` and ``prevalence`` (breaches / supervised_orders).
    """
    purchase = _parsed_timestamp(orders, "order_purchase_timestamp")
    absent = dict.fromkeys([c for c in DATE_COLUMNS if c not in orders], pd.NaT)
    prepared = add_target_columns(orders.assign(**absent))
    status = prepared["order_status"].fillna("")
    period = (purchase >= pd.Timestamp(start)) & (purchase < pd.Timestamp(end))
    supervised = (
        period
        & status.eq("delivered")
        & prepared["order_delivered_customer_date"].notna()
        & prepared["order_estimated_delivery_date"].notna()
    )
    censored = int(
        (period & status.eq("shipped") & prepared["order_delivered_customer_date"].isna()).sum()
    )
    supervised_count = int(supervised.sum())
    breaches = int(prepared.loc[supervised, "sla_breach"].sum())

    def row(scenario: str, extra: int) -> dict[str, object]:
        population = supervised_count + extra
        violations = breaches + extra
        return {
            "scenario": scenario,
            "supervised_orders": population,
            "breaches": violations,
            "censored_orders": censored,
            "prevalence": violations / population if population else float("nan"),
        }

    return pd.DataFrame([row("exclude", 0), row("include_as_breach", censored)])


def plot_temporal_split(
    splits: dict[str, pd.DataFrame],
    timestamp_col: str = "order_purchase_timestamp",
    language: str = "pt",
) -> go.Figure:
    """Plot chronological train, validation, and test partitions."""
    labels = get_labels(language)["temporal"]
    summary = temporal_split_summary(splits, timestamp_col=timestamp_col)
    colors = {"train": "#2563EB", "validation": "#0F766E", "test": "#F59E0B"}
    figure = go.Figure()
    for row in summary.itertuples(index=False):
        start = pd.Timestamp(row.first_purchase)
        end = pd.Timestamp(row.last_purchase)
        figure.add_trace(
            go.Scatter(
                x=[start, end],
                y=[row.split, row.split],
                mode="lines",
                line={"color": colors[row.split], "width": 28},
                hovertemplate=(
                    f"{row.split}: {int(row.orders):,} orders"
                    "<br>%{x|%Y-%m-%d}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        figure.add_annotation(
            x=start + (end - start) / 2,
            y=row.split,
            text=f"{int(row.orders):,} orders",
            showarrow=False,
            font={"color": "white", "size": 12},
        )
    figure.update_layout(
        title=labels["title"],
        xaxis={"title": labels["xaxis"], "type": "date", "showgrid": True},
        yaxis={
            "title": "",
            "categoryorder": "array",
            "categoryarray": ["test", "validation", "train"],
        },
        template="plotly_white",
        height=420,
        margin={"l": 40, "r": 40, "t": 70, "b": 60},
    )
    figure.update_yaxes(
        tickvals=["train", "validation", "test"],
        ticktext=[labels["train"], labels["validation"], labels["test"]],
    )
    return figure
