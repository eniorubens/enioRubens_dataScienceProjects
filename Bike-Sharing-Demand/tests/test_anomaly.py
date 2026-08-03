"""Tests for the shared 2020-style anomaly diagnosis in src/seasonal.py."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.seasonal import (  # noqa: E402
    AnomalyDiagnosis,
    anomaly_diagnosis,
    anomaly_mask,
    anomaly_report_table,
    plot_anomaly_diagnosis,
)

TARGET = "Rented Bike Count"


@pytest.fixture
def three_year_df() -> pd.DataFrame:
    """Hourly demand for 2019-2021; 2020 months 8-11 collapse below threshold."""
    dates = pd.date_range("2019-01-01", "2021-12-31 23:00", freq="h")
    year = dates.year
    month = dates.month
    # Multiplicative growth: 2019 -> 1000, 2021 -> 2000; 2020 sits exactly at
    # the geometric mean so non-collapsed months have ratio == 1.
    geo = float(np.sqrt(1000.0 * 2000.0))
    level = np.where(year == 2019, 1000.0, np.where(year == 2020, geo, 2000.0))
    demand = level.astype(float)
    # Collapse Aug-Nov 2020 to ~10% of expected.
    collapse = (year == 2020) & np.isin(month, [8, 9, 10, 11])
    demand = np.where(collapse, demand * 0.10, demand)
    return pd.DataFrame({"DateTime": dates, "Year": year, "Month": month, TARGET: demand})


def test_flags_collapsed_months(three_year_df):
    diag = anomaly_diagnosis(three_year_df, TARGET)
    assert isinstance(diag, AnomalyDiagnosis)
    assert diag.anomalous_months == [8, 9, 10, 11]
    # Non-collapsed months sit at ~1.0 (2020 == geometric mean of 2019/2021).
    normal = diag.ratio.drop(index=[8, 9, 10, 11])
    assert np.allclose(normal.values, 1.0, atol=1e-6)


def test_derives_year_month_from_datetime(three_year_df):
    no_cols = three_year_df.drop(columns=["Year", "Month"])
    diag = anomaly_diagnosis(no_cols, TARGET)
    assert diag.anomalous_months == [8, 9, 10, 11]


def test_anomaly_mask_matches_rule(three_year_df):
    diag = anomaly_diagnosis(three_year_df, TARGET)
    mask = anomaly_mask(three_year_df, diag)
    expected = three_year_df["Year"].eq(2020) & three_year_df["Month"].isin([8, 9, 10, 11])
    assert mask.equals(expected)
    assert mask.index.equals(three_year_df.index)


def test_report_table_is_localized_pt(three_year_df):
    diag = anomaly_diagnosis(three_year_df, TARGET)
    table = anomaly_report_table(diag)
    assert "Referência geométrica" in table.columns
    assert "Sinalizado" in table.columns
    assert table.index.name == "Mês"
    assert bool(table["Sinalizado"].loc[8]) is True
    assert bool(table["Sinalizado"].loc[1]) is False


def test_plot_returns_fig_axes(three_year_df):
    diag = anomaly_diagnosis(three_year_df, TARGET)
    fig, axes = plot_anomaly_diagnosis(diag)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    plt.close(fig)
