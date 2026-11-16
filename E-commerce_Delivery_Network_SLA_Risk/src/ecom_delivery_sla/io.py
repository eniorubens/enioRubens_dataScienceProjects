"""Input/output helpers for the Olist source tables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

TABLE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "categories": "product_category_name_translation.csv",
}

EDA_TABLES = ("orders", "customers", "items", "products")


def read_olist_tables(
    data_dir: str | Path,
    table_names: Iterable[str] = EDA_TABLES,
) -> dict[str, pd.DataFrame]:
    """Read selected Olist tables without changing the raw source files."""
    data_path = Path(data_dir)
    selected = tuple(table_names)
    unknown = sorted(set(selected) - set(TABLE_FILES))
    if unknown:
        raise ValueError(f"Unknown Olist tables: {unknown}")

    tables: dict[str, pd.DataFrame] = {}
    for name in selected:
        file_path = data_path / TABLE_FILES[name]
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        tables[name] = pd.read_csv(file_path)
    return tables


def schema_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per field with observed type and completeness."""
    rows: list[dict[str, object]] = []
    for table_name, table in tables.items():
        for position, (field, dtype) in enumerate(table.dtypes.items(), start=1):
            rows.append(
                {
                    "dataset": table_name,
                    "rows": len(table),
                    "field_order": position,
                    "field": field,
                    "data_type": str(dtype),
                    "non_null_pct": table[field].notna().mean(),
                }
            )
    return pd.DataFrame(rows)
