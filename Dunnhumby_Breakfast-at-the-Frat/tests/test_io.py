"""Excel -> Parquet caching on a tiny synthetic workbook shaped like the dunnhumby file."""

import pandas as pd
import pytest

from retail_multilevel.io import excel_to_parquet, load_interim
from retail_multilevel.paths import SHEETS


@pytest.fixture
def xlsx(tmp_path):
    path = tmp_path / "book.xlsx"
    frames = {
        "dh Store Lookup": pd.DataFrame({"store_id ": [1, 2], "SEG_VALUE_NAME": ["VALUE", "UPSCALE"]}),
        "dh Products Lookup": pd.DataFrame({"UPC": [10], "DESCRIPTION": ["x"]}),
        "dh Transaction Data": pd.DataFrame({"STORE_NUM": [1], "UPC": [10], "UNITS": [3]}),
    }
    with pd.ExcelWriter(path) as writer:
        for sheet, df in frames.items():
            pd.DataFrame([["Title row"]]).to_excel(writer, sheet_name=sheet, header=False, index=False)
            df.to_excel(writer, sheet_name=sheet, startrow=1, index=False)
    return path


def test_excel_to_parquet_reads_header_on_row_two(xlsx, tmp_path):
    out_dir = tmp_path / "interim"
    paths = excel_to_parquet(xlsx, out_dir)
    assert set(paths) == set(SHEETS.values())
    stores = load_interim("stores", out_dir)
    assert list(stores.columns) == ["STORE_ID", "SEG_VALUE_NAME"]  # stripped and upper-cased
    assert len(stores) == 2


def test_cache_is_reused_unless_overwrite(xlsx, tmp_path):
    out_dir = tmp_path / "interim"
    excel_to_parquet(xlsx, out_dir)
    marker = pd.DataFrame({"X": [0]})
    marker.to_parquet(out_dir / "products.parquet")
    excel_to_parquet(xlsx, out_dir)
    assert list(load_interim("products", out_dir).columns) == ["X"]
    excel_to_parquet(xlsx, out_dir, overwrite=True)
    assert "UPC" in load_interim("products", out_dir).columns


def test_missing_files_raise_helpful_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="How to get the data"):
        excel_to_parquet(tmp_path / "nope.xlsx", tmp_path)
    with pytest.raises(FileNotFoundError, match="ingest.py"):
        load_interim("stores", tmp_path)
