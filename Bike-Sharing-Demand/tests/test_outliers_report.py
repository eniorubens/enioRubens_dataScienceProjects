"""Tests for the outlier rate-matrix report builder in src/outliers.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import outliers

TARGET = "Rented Bike Count"


def _two_year_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2018-01-01", "2019-12-31 23:00", freq="h")
    months = dates.month
    return pd.DataFrame(
        {
            "DateTime": dates,
            "Hour": dates.hour,
            TARGET: rng.integers(0, 3000, size=len(dates)).astype(float),
            "Seasons": np.where(
                months.isin([12, 1, 2]),
                "Winter",
                np.where(
                    months.isin([3, 4, 5]),
                    "Spring",
                    np.where(months.isin([6, 7, 8]), "Summer", "Autumn"),
                ),
            ),
        }
    )


def test_rate_matrix_is_localized_and_keeps_contract():
    df = _two_year_df()
    summary = outliers.iqr_outlier_summary_by_season_year(df)
    # Internal analytic contract must be untouched (English keys).
    assert {"Season", "Meteorological year", "Outlier rate (%)"} <= set(summary.columns)

    complete = outliers.complete_meteorological_years(summary)
    assert complete == [2018, 2019]  # partial Dec-2019 met-year excluded

    matrix = outliers.outlier_rate_matrix(summary, complete)
    assert list(matrix.index) == complete
    assert list(matrix.columns) == ["Ano completo", "Inverno", "Primavera", "Verão", "Outono"]
