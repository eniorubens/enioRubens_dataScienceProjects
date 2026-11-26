"""Raw data I/O: the licensed Excel workbook is read once and cached as Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_multilevel.paths import INTERIM_DIR, RAW_XLSX, SHEETS


def interim_path(name: str, interim_dir: Path = INTERIM_DIR) -> Path:
    """Return the Parquet path for the sheet short name ``name``."""
    return interim_dir / f"{name}.parquet"


def read_sheet(sheet: str, xlsx: Path = RAW_XLSX) -> pd.DataFrame:
    """Read one sheet of the dunnhumby workbook.

    Row 1 is a title; the header is on row 2 (``header=1``).
    """
    df = pd.read_excel(xlsx, sheet_name=sheet, header=1)
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df.dropna(axis=1, how="all")


def excel_to_parquet(
    xlsx: Path = RAW_XLSX, interim_dir: Path = INTERIM_DIR, overwrite: bool = False
) -> dict[str, Path]:
    """Cache every analysis sheet as Parquet in ``interim_dir`` (as read, no cleaning)."""
    if not xlsx.exists():
        raise FileNotFoundError(f"{xlsx} not found. See 'How to get the data' in README.md.")
    interim_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for sheet, name in SHEETS.items():
        path = interim_path(name, interim_dir)
        if overwrite or not path.exists():
            read_sheet(sheet, xlsx).to_parquet(path, index=False)
        out[name] = path
    return out


def load_interim(name: str, interim_dir: Path = INTERIM_DIR) -> pd.DataFrame:
    """Load a cached sheet (``'stores'``, ``'products'`` or ``'transactions'``)."""
    path = interim_path(name, interim_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/ingest.py")
    return pd.read_parquet(path)
