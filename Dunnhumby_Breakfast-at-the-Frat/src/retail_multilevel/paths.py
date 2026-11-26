"""Project paths. Every module resolves files from here, never from the working directory."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Licensed dunnhumby files, downloaded manually (never versioned).
DATASET_DIR = ROOT / "dataset"
RAW_XLSX = DATASET_DIR / "dunnhumby - Breakfast at the Frat.xlsx"

DATA_DIR = ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"  # one Parquet per Excel sheet, as read
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned analytical panel

OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
SITE_DIR = ROOT / "site"

# Excel sheet name -> short name used for the Parquet files.
SHEETS: dict[str, str] = {
    "dh Store Lookup": "stores",
    "dh Products Lookup": "products",
    "dh Transaction Data": "transactions",
}
