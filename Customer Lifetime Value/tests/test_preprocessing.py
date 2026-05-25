"""
test_preprocessing.py — Tests for OnlineRetailPreprocessor.
"""

from __future__ import annotations

import io
import textwrap

import pandas as pd
import pytest

from src.preprocessing import OnlineRetailPreprocessor


@pytest.fixture
def preprocessor() -> OnlineRetailPreprocessor:
    return OnlineRetailPreprocessor()


@pytest.fixture
def clean_df(sample_transactions, preprocessor) -> pd.DataFrame:
    cleaned, _ = preprocessor.clean(sample_transactions)
    return cleaned


# ── load() ────────────────────────────────────────────────────────────────────

def test_load_normalizes_column_names(tmp_path, preprocessor):
    """Columns with different casing are normalised to canonical names."""
    csv_content = textwrap.dedent(
        """\
        invoiceno,quantity,invoicedate,unitprice,customerid
        536365,6,12/1/10 8:26,2.55,17850
        """
    )
    path = tmp_path / "test.csv"
    path.write_text(csv_content)
    df = preprocessor.load(path)
    for col in OnlineRetailPreprocessor.REQUIRED_COLUMNS:
        assert col in df.columns, f"Expected column '{col}' missing after normalisation"


def test_load_raises_on_missing_columns(tmp_path, preprocessor):
    """ValueError raised if a required column is absent after normalisation."""
    csv_content = "invoiceno,quantity,invoicedate\n536365,6,12/1/10 8:26\n"
    path = tmp_path / "test.csv"
    path.write_text(csv_content)
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocessor.load(path)


# ── clean() ───────────────────────────────────────────────────────────────────

def test_clean_removes_null_customers(sample_transactions, preprocessor):
    """Rows with NaN CustomerID must be removed."""
    cleaned, _ = preprocessor.clean(sample_transactions)
    assert cleaned["CustomerID"].isna().sum() == 0


def test_clean_removes_cancellations(sample_transactions, preprocessor):
    """Rows whose InvoiceNo starts with 'C' must be removed."""
    cleaned, _ = preprocessor.clean(sample_transactions)
    has_cancellation = cleaned["InvoiceNo"].str.upper().str.startswith("C").any()
    assert not has_cancellation


def test_clean_removes_negative_quantity(sample_transactions, preprocessor):
    """Rows with Quantity <= 0 must be removed."""
    cleaned, _ = preprocessor.clean(sample_transactions)
    assert (cleaned["Quantity"] > 0).all()


def test_clean_removes_negative_price(sample_transactions, preprocessor):
    """Rows with UnitPrice <= 0 must be removed."""
    cleaned, _ = preprocessor.clean(sample_transactions)
    assert (cleaned["UnitPrice"] > 0).all()


def test_clean_creates_total_price(sample_transactions, preprocessor):
    """TotalPrice == Quantity * UnitPrice for every row."""
    cleaned, _ = preprocessor.clean(sample_transactions)
    assert "TotalPrice" in cleaned.columns
    expected = (cleaned["Quantity"] * cleaned["UnitPrice"]).round(6)
    actual = cleaned["TotalPrice"].round(6)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_clean_applies_outlier_cap(sample_transactions, preprocessor):
    """TotalPrice values above the quantile cap are removed."""
    # Reconstruct the intermediate state (after row removal, before capping) to
    # derive the same threshold that clean() uses internally.
    df = sample_transactions.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%y %H:%M", errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])
    df = df.loc[~df["InvoiceNo"].astype(str).str.upper().str.startswith("C", na=False)]
    df = df.loc[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    cap = df["TotalPrice"].quantile(preprocessor.OUTLIER_QUANTILE)

    cleaned, _ = preprocessor.clean(sample_transactions)
    assert (cleaned["TotalPrice"] <= cap + 1e-9).all()


def test_clean_returns_summary_dataframe(sample_transactions, preprocessor):
    """cleaning_summary is a DataFrame with expected columns."""
    _, summary = preprocessor.clean(sample_transactions)
    assert isinstance(summary, pd.DataFrame)
    for col in ("step", "rows", "customers"):
        assert col in summary.columns, f"Column '{col}' missing from cleaning_summary"


# ── build_rfm() ────────────────────────────────────────────────────────────────

def _make_clean_df() -> pd.DataFrame:
    """Small clean transaction DataFrame for RFM tests."""
    rows = []
    for cid in ["1001", "1002", "1003"]:
        for offset_days in [0, 30, 90]:
            rows.append(
                {
                    "CustomerID": cid,
                    "InvoiceDate": pd.Timestamp("2010-12-01") + pd.Timedelta(days=offset_days),
                    "InvoiceNo": f"INV{cid}{offset_days}",
                    "TotalPrice": 50.0,
                    "Quantity": 2,
                    "UnitPrice": 25.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_rfm_output_columns(preprocessor):
    """RFM DataFrame must have the expected columns."""
    df = _make_clean_df()
    rfm = preprocessor.build_rfm(df)
    for col in ("customer_id", "frequency", "recency", "T", "monetary_value"):
        assert col in rfm.columns, f"Column '{col}' missing from RFM"


def test_build_rfm_removes_zero_T(preprocessor):
    """Customers observed only on the last day (T == 0) must be excluded."""
    df = _make_clean_df()
    # Add a customer with a single transaction on the observation period end
    last_date = df["InvoiceDate"].max()
    extra = pd.DataFrame(
        [{"CustomerID": "9999", "InvoiceDate": last_date, "InvoiceNo": "INV9999", "TotalPrice": 10.0, "Quantity": 1, "UnitPrice": 10.0}]
    )
    df = pd.concat([df, extra], ignore_index=True)
    rfm = preprocessor.build_rfm(df)
    assert (rfm["T"] > 0).all()


# ── calibration_holdout_split() ───────────────────────────────────────────────

def test_calibration_holdout_split_no_overlap(preprocessor):
    """
    Calibration and holdout DataFrames share customer IDs but the column sets
    are disjoint so there is no temporal overlap.
    """
    df = _make_clean_df()
    cal, hold = preprocessor.calibration_holdout_split(df)
    assert "frequency_cal" in cal.columns
    assert "frequency_holdout" in hold.columns
    # Same customers appear in both
    assert set(cal["customer_id"]) == set(hold["customer_id"])


# ── get_modeling_base() ───────────────────────────────────────────────────────

def test_get_modeling_base_filters_correctly(sample_rfm, preprocessor):
    """Only customers with frequency > 0 and monetary_value > 0 are returned."""
    base = preprocessor.get_modeling_base(sample_rfm)
    assert (base["frequency"] > 0).all()
    assert (base["monetary_value"] > 0).all()
    assert len(base) < len(sample_rfm)
