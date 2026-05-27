"""
data.py
-------
Data loading, cleaning and RFM feature engineering.

All user-visible strings pass through ``t()`` so print() / display()
output respects the active language without code duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.multilang import t
from src.viz import fmt


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "Customer ID": ("CustomerID", "CustomerId"),
    "Invoice": ("InvoiceNo",),
    "Price": ("UnitPrice",),
}

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
)

_INVOICE_DATE_FORMATS: tuple[str, ...] = (
    "%m/%d/%y %H:%M",
    "%Y-%m-%d",
)


def _log_step(steps: list[dict], name: str, before: int, after: int) -> None:
    steps.append({
        t("Step"): name,
        t("Before"): before,
        t("After"): after,
        t("Removed"): before - after,
        t("% removed"): round((before - after) / before * 100, 2) if before else 0,
    })


def _mode_or_unknown(series: pd.Series) -> str:
    """Return the modal value of *series*, or ``'Unknown'`` when empty."""
    mode = series.dropna().mode()
    return str(mode.iloc[0]) if not mode.empty else "Unknown"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected transaction columns across common Online Retail schemas."""
    out = df.copy()
    out = out.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    rename_map: dict[str, str] = {}
    for target, aliases in _COLUMN_ALIASES.items():
        if target in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                rename_map[alias] = target
                break

    if rename_map:
        out = out.rename(columns=rename_map)

    missing = [col for col in _REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise KeyError(
            f"Missing required columns after normalization: {missing}. "
            f"Available columns: {list(out.columns)}"
        )

    return out


def _parse_invoice_dates(series: pd.Series) -> pd.Series:
    """Parse invoice dates with explicit formats to avoid pandas inference warnings."""
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.astype(str)

    for date_format in _INVOICE_DATE_FORMATS:
        current = pd.to_datetime(remaining, format=date_format, errors="coerce")
        mask = current.notna() & parsed.isna()
        if mask.any():
            parsed.loc[mask] = current.loc[mask]
        if parsed.notna().all():
            break

    return parsed


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the Online Retail CSV into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Location of the local retail CSV file (for example,
        ``online_retail.csv`` or ``online_retail_II.csv``).

    Returns
    -------
    pd.DataFrame
        Raw data without any cleaning.
    """
    df = pd.read_csv(path)
    print(
        t("Clean dataset: {} rows and {} columns").format(
            fmt(df.shape[0]), fmt(df.shape[1])
        )
    )
    return df


def clean(raw_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove invalid records and engineer ``Revenue``.

    Cleaning steps (logged when *verbose* is ``True``):

    1. Drop rows with missing ``Customer ID``.
    2. Parse ``InvoiceDate``; drop unparseable rows.
    3. Remove cancellations (``Invoice`` starts with ``'C'``).
    4. Remove ``Quantity <= 0``.
    5. Remove ``Price <= 0``.
    6. Drop rows with missing ``Description``.

    Parameters
    ----------
    raw_df : pd.DataFrame
    verbose : bool
        Print the cleaning log table when ``True``.

    Returns
    -------
    pd.DataFrame
        Cleaned data with ``Revenue`` column added.
    """
    df = _normalize_columns(raw_df)
    steps: list[dict[str, Any]] = []

    before = len(df)
    df = df.dropna(subset=["Customer ID"])
    _log_step(steps, "Remove missing Customer ID", before, len(df))

    before = len(df)
    df["InvoiceDate"] = _parse_invoice_dates(df["InvoiceDate"])
    df = df.dropna(subset=["InvoiceDate"])
    _log_step(steps, "Convert InvoiceDate", before, len(df))

    before = len(df)
    df = df.loc[~df["Invoice"].astype(str).str.startswith("C", na=False)]
    _log_step(steps, "Remove cancellations", before, len(df))

    before = len(df)
    df = df.loc[df["Quantity"] > 0]
    _log_step(steps, "Remove Quantity <= 0", before, len(df))

    before = len(df)
    df = df.loc[df["Price"] > 0]
    _log_step(steps, "Remove Price <= 0", before, len(df))

    before = len(df)
    df = df.dropna(subset=["Description"])
    _log_step(steps, "Remove missing Description", before, len(df))

    df["Customer ID"] = df["Customer ID"].astype("Int64").astype(str)
    df["Revenue"] = df["Quantity"] * df["Price"]
    df = df.reset_index(drop=True)

    if verbose:
        from IPython.display import display

        display(pd.DataFrame(steps))
        print(
            t("Clean dataset: {} rows and {} columns").format(
                fmt(df.shape[0]), fmt(df.shape[1])
            )
        )
        print(t("Unique customers: {}").format(fmt(df["Customer ID"].nunique())))
        print(t("Total revenue: {}").format(fmt(df["Revenue"].sum(), 2)))

    return df


def build_rfm(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions into one RFM row per customer.

    Features produced:

    ============== =====================================================
    Column         Description
    ============== =====================================================
    Recency        Days since last purchase (lower = more recent)
    Frequency      Number of unique invoices
    Monetary       Total revenue
    AverageTicket  Monetary / Frequency (zero-division safe)
    QuantityTotal  Sum of item quantities
    UniqueProducts Count of distinct SKUs purchased
    CountryMode    Most frequent country for this customer
    ============== =====================================================

    Parameters
    ----------
    clean_df : pd.DataFrame
        Output of :func:`clean`.

    Returns
    -------
    pd.DataFrame
        One row per ``Customer ID``.
    """
    snapshot_date = clean_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(t("Reference date: {}").format(f"{snapshot_date:%Y-%m-%d}"))

    rfm = (
        clean_df.groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
            QuantityTotal=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
            CountryMode=("Country", _mode_or_unknown),
        )
        .reset_index()
    )

    rfm["AverageTicket"] = np.where(
        rfm["Frequency"] > 0,
        rfm["Monetary"] / rfm["Frequency"],
        0.0,
    )

    rfm = rfm[
        [
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
            "AverageTicket",
            "QuantityTotal",
            "UniqueProducts",
            "CountryMode",
        ]
    ]

    print(t("RFM table: {} customers").format(fmt(rfm.shape[0])))
    return rfm
