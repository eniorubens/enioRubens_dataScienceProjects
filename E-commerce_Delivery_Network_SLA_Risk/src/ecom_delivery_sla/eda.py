"""Reusable descriptive tables and Plotly figures for the first EDA."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from .labels import get_labels


def inventory_summary(tables: dict[str, pd.DataFrame], language: str = "pt") -> pd.DataFrame:
    """Summarize table names, row counts, and column counts."""
    labels = get_labels(language)
    rows = []
    for name, table in tables.items():
        rows.append({"tabela": labels["inventory"].get(name, name), "linhas": len(table), "colunas": table.shape[1]})
    return pd.DataFrame(rows)


def target_summary(supervised: pd.DataFrame, language: str = "pt") -> pd.DataFrame:
    """Return the core delivery-delay and SLA-breach summary."""
    labels = get_labels(language)
    values = [
        len(supervised),
        int(supervised["sla_breach"].sum()),
        supervised["sla_breach"].mean(),
        supervised["delivery_delay_days"].median(),
        supervised["delivery_delay_days"].quantile(.95),
        supervised["delivery_delay_days"].quantile(.99),
        int((supervised["delivery_delay_days"] > 30).sum()),
        int((supervised["delivery_delay_days"] < -30).sum()),
    ]
    return pd.DataFrame({labels["columns"]["metric"]: labels["target_metrics"], labels["columns"]["value"]: values})


def lead_summary(supervised: pd.DataFrame, language: str = "pt") -> pd.DataFrame:
    """Summarize promised and actual lead-time quantiles."""
    labels = get_labels(language)
    columns = labels["columns"]
    return pd.DataFrame({
        columns["measure"]: labels["lead_measures"],
        columns["median"]: [supervised["promised_lead_days"].median(), supervised["actual_lead_days"].median()],
        columns["p95"]: [supervised["promised_lead_days"].quantile(.95), supervised["actual_lead_days"].quantile(.95)],
    })


def monthly_metrics(supervised: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly order volume, breaches, and delay metrics."""
    data = supervised.copy()
    data["purchase_month"] = data["order_purchase_timestamp"].dt.to_period("M").astype(str)
    result = data.groupby("purchase_month").agg(
        orders=("order_id", "size"),
        breaches=("sla_breach", "sum"),
        median_delay=("delivery_delay_days", "median"),
        p95_delay=("delivery_delay_days", lambda series: series.quantile(.95)),
    )
    result["breach_rate"] = result["breaches"] / result["orders"]
    return result


def monthly_summary(supervised: pd.DataFrame, language: str = "pt") -> pd.DataFrame:
    """Return monthly metrics with localized presentation labels."""
    labels = get_labels(language)["columns"]
    return monthly_metrics(supervised).rename(columns={
        "orders": labels["orders"], "breaches": labels["breaches"],
        "median_delay": labels["median_delay"], "p95_delay": labels["p95_delay"],
        "breach_rate": labels["breach_rate"],
    }).rename_axis(labels["month"])


def state_metrics(supervised: pd.DataFrame, customers: pd.DataFrame, min_orders: int = 500) -> pd.DataFrame:
    """Calculate customer-state metrics above a minimum order count."""
    result = (
        supervised[["order_id", "customer_id", "sla_breach", "delivery_delay_days"]]
        .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
        .groupby("customer_state")
        .agg(orders=("order_id", "size"), breaches=("sla_breach", "sum"), median_delay=("delivery_delay_days", "median"))
    )
    result["breach_rate"] = result["breaches"] / result["orders"]
    return result.loc[result["orders"] >= min_orders].sort_values("breach_rate")


def state_summary(supervised: pd.DataFrame, customers: pd.DataFrame, language: str = "pt", min_orders: int = 500) -> pd.DataFrame:
    """Return customer-state metrics with localized presentation labels."""
    labels = get_labels(language)["columns"]
    return state_metrics(supervised, customers, min_orders=min_orders).rename(columns={
        "orders": labels["orders"], "breaches": labels["breaches"],
        "median_delay": labels["median_delay"], "breach_rate": labels["breach_rate"],
    }).rename_axis(labels["state"])


def _dominant_category(items: pd.DataFrame, products: pd.DataFrame) -> pd.Series:
    """Assign each order the modal non-null product category."""
    merged = items[["order_id", "product_id"]].merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    return merged.groupby("order_id")["product_category_name"].agg(
        lambda values: values.dropna().mode().iloc[0] if not values.dropna().empty else "unknown"
    ).rename("dominant_category")


def category_metrics(supervised: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame, min_orders: int = 1000) -> pd.DataFrame:
    """Calculate category metrics after reducing item rows to orders."""
    result = (
        supervised[["order_id", "sla_breach", "delivery_delay_days"]]
        .join(_dominant_category(items, products), on="order_id")
        .groupby("dominant_category")
        .agg(orders=("order_id", "size"), breaches=("sla_breach", "sum"), median_delay=("delivery_delay_days", "median"))
    )
    result["breach_rate"] = result["breaches"] / result["orders"]
    return result.loc[result["orders"] >= min_orders].sort_values("breach_rate", ascending=False)


def category_summary(supervised: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame, language: str = "pt", min_orders: int = 1000) -> pd.DataFrame:
    """Return category metrics with localized presentation labels."""
    labels = get_labels(language)["columns"]
    return category_metrics(supervised, items, products, min_orders=min_orders).rename(columns={
        "orders": labels["orders"], "breaches": labels["breaches"],
        "median_delay": labels["median_delay"], "breach_rate": labels["breach_rate"],
    }).rename_axis(labels["category"])


def anomaly_summary(supervised: pd.DataFrame, language: str = "pt") -> pd.DataFrame:
    """Count temporal anomalies in the supervised population."""
    labels = get_labels(language)
    columns = labels["columns"]
    values = [
        int((supervised["order_delivered_customer_date"] < supervised["order_delivered_carrier_date"]).sum()),
        int(supervised["order_delivered_carrier_date"].isna().sum()),
        int((supervised["order_approved_at"] < supervised["order_purchase_timestamp"]).sum()),
        int((supervised["order_delivered_customer_date"] < supervised["order_purchase_timestamp"]).sum()),
    ]
    return pd.DataFrame({columns["rule"]: labels["anomaly_rules"], columns["cases"]: values})


def plot_delay_distribution(supervised: pd.DataFrame, language: str = "pt"):
    """Build a Plotly histogram of delivery delay."""
    labels = get_labels(language)
    data = supervised.loc[supervised["delivery_delay_days"].between(-60, 60), ["delivery_delay_days"]]
    figure = px.histogram(data, x="delivery_delay_days", nbins=60, title=labels["plots"]["delay"], labels={"delivery_delay_days": labels["axes"]["delay"]})
    figure.add_vline(x=0, line_dash="dash", line_color="#b2182b")
    figure.add_vline(x=supervised["delivery_delay_days"].median(), line_dash="dot", line_color="#1b7837")
    figure.update_layout(yaxis_title=labels["axes"]["orders"], showlegend=False)
    return figure


def plot_lead_scatter(supervised: pd.DataFrame, language: str = "pt", sample_size: int = 5000):
    """Build a Plotly scatter plot of promised lead time and delay."""
    labels = get_labels(language)
    sample = supervised.sample(n=min(sample_size, len(supervised)), random_state=42)
    figure = px.scatter(sample, x="promised_lead_days", y="delivery_delay_days", color="sla_breach", opacity=.25, title=labels["plots"]["lead"], labels={"promised_lead_days": labels["axes"]["promised"], "delivery_delay_days": labels["axes"]["relative_delay"], "sla_breach": labels["axes"]["breach"]}, color_discrete_map={False: "#2f6f9f", True: "#b2182b"})
    figure.update_traces(marker={"size": 5})
    figure.add_hline(y=0, line_color="black")
    return figure


def plot_monthly_breach(supervised: pd.DataFrame, language: str = "pt"):
    """Build a Plotly line chart of monthly breach rates."""
    labels = get_labels(language)
    data = monthly_metrics(supervised).reset_index()
    figure = px.line(data, x="purchase_month", y="breach_rate", markers=True, title=labels["plots"]["monthly"], labels={"purchase_month": labels["axes"]["month"], "breach_rate": labels["axes"]["breach"]})
    figure.update_yaxes(tickformat=".1%")
    return figure


def plot_state_breach(supervised: pd.DataFrame, customers: pd.DataFrame, language: str = "pt", min_orders: int = 500):
    """Build a Plotly bar chart of breach rates by customer state."""
    labels = get_labels(language)
    data = state_metrics(supervised, customers, min_orders=min_orders).reset_index()
    figure = px.bar(data, x="breach_rate", y="customer_state", orientation="h", title=labels["plots"]["state"], labels={"breach_rate": labels["axes"]["breach"], "customer_state": labels["axes"]["state"]}, text="breach_rate")
    figure.update_xaxes(tickformat=".1%")
    figure.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    return figure


def plot_category_breach(supervised: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame, language: str = "pt", min_orders: int = 1000, top_n: int = 12):
    """Build a Plotly bar chart of the highest category breach rates."""
    labels = get_labels(language)
    data = category_metrics(supervised, items, products, min_orders=min_orders).head(top_n).sort_values("breach_rate").reset_index()
    figure = px.bar(data, x="breach_rate", y="dominant_category", orientation="h", title=labels["plots"]["category"], labels={"breach_rate": labels["axes"]["breach"], "dominant_category": labels["axes"]["category"]}, text="breach_rate")
    figure.update_xaxes(tickformat=".1%")
    figure.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    return figure
