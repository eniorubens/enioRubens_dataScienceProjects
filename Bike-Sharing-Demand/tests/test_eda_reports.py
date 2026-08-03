"""Tests for the notebook-01 report/plot builders extracted into src/eda.py.

These functions must return testable data objects, keep the input frame
unmutated, localize visible labels to PT (passthrough) without renaming the
internal schema, and — for plots — return (fig, axes) without showing them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import eda  # noqa: E402

TARGET = "Rented Bike Count"


@pytest.fixture
def multi_year_df() -> pd.DataFrame:
    """Two full years of hourly rows with a deliberate coverage gap and NaNs.

    Mirrors ``read_data()``: ``DateTime`` is the calendar day at midnight and the
    hour lives in a separate ``Hour`` column (so ``DateTime + Hour`` rebuilds the
    hourly stamp without double-counting).
    """
    rng = np.random.default_rng(0)
    stamps = pd.date_range("2018-01-01", "2019-12-31 23:00", freq="h")
    months = stamps.month
    df = pd.DataFrame(
        {
            "DateTime": stamps.normalize(),
            "Hour": stamps.hour,
            TARGET: rng.integers(0, 3000, size=len(stamps)).astype(float),
            "Temperature(C)": rng.uniform(-10, 35, size=len(stamps)),
            "Cloud Cover (oktas)": rng.uniform(0, 8, size=len(stamps)),
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
    # Punch a 5-hour coverage gap (Jun 10 2018, hours 2..6) and inject NaNs.
    gap = (df["DateTime"] == "2018-06-10") & df["Hour"].between(2, 6)
    df = df.loc[~gap].reset_index(drop=True)
    df.loc[10:14, "Cloud Cover (oktas)"] = np.nan
    return df


def _timestamp(df: pd.DataFrame) -> pd.Series:
    return df["DateTime"] + pd.to_timedelta(df["Hour"], unit="h")


def test_schema_summary_shape_and_pt_labels(multi_year_df):
    before = multi_year_df.copy()
    out = eda.schema_summary(multi_year_df)
    assert list(out.index) == list(multi_year_df.columns)  # internal schema intact
    assert set(out.columns) == {"Tipo", "Não nulos", "Ausentes", "Valores únicos"}
    pd.testing.assert_frame_equal(multi_year_df, before)  # no mutation


def test_missing_values_summary_counts(multi_year_df):
    summary, stats = eda.missing_values_summary(multi_year_df)
    assert stats["total_missing"] == int(multi_year_df.isna().sum().sum())
    assert stats["total_missing"] == 5  # the five injected NaNs
    assert stats["missing_col"] in summary.columns


def test_temporal_coverage_totals(multi_year_df):
    ts = _timestamp(multi_year_df)
    coverage, stats = eda.temporal_coverage(multi_year_df, ts)
    assert stats["observed_rows"] == len(multi_year_df)
    assert stats["missing_hours"] == 5  # the punched gap
    assert stats["coverage_col"] in coverage.columns


def test_observed_period_gap_runs(multi_year_df):
    ts = _timestamp(multi_year_df)
    period, gaps, stats = eda.observed_period_summary(multi_year_df, ts)
    assert stats["missing_hour_count"] == 5
    assert stats["n_runs"] == 1
    assert stats["max_run_hours"] == 5


def test_daily_hourly_profile_and_plot(multi_year_df):
    ts = _timestamp(multi_year_df)
    profile = eda.daily_hourly_profile(multi_year_df, ts, TARGET)
    assert len(profile["season_summary"]) == 4
    fig, axes = eda.plot_daily_and_hourly(profile)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    plt.close(fig)


def test_yearly_growth_matches_groupby(multi_year_df):
    yearly = eda.yearly_growth_summary(multi_year_df, TARGET)
    inline = multi_year_df.groupby(pd.to_datetime(multi_year_df["DateTime"]).dt.year)[TARGET].mean()
    assert np.allclose(yearly["Média horária"].values, inline.values)
    fig, axes = eda.plot_yearly_growth(yearly)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_weather_missing_summary_and_plot(multi_year_df):
    cols = ["Temperature(C)", "Cloud Cover (oktas)"]
    summary = eda.weather_missing_summary(multi_year_df, cols)
    assert summary.loc["Cloud Cover (oktas)", "Valores ausentes"] == 5
    fig, ax = eda.plot_weather_missing_by_year(multi_year_df, cols)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_localize_preview_translates_values_without_mutating_input():
    df = pd.DataFrame(
        {
            "Seasons": ["Winter", "Summer"],
            "Holiday": ["Holiday", "No Holiday"],
            "Functioning Day": ["Yes", "No"],
            "Rented Bike Count": [10, 20],
        }
    )
    before = df.copy()
    out = eda.localize_preview(df)
    assert list(out["Seasons"]) == ["Inverno", "Verão"]
    assert list(out["Holiday"]) == ["Feriado", "Sem feriado"]
    assert list(out["Functioning Day"]) == ["Sim", "Não"]
    assert list(out["Rented Bike Count"]) == [10, 20]
    # internal dataframe stays in the original (English) schema
    assert list(df["Seasons"]) == ["Winter", "Summer"]
    assert list(df["Holiday"]) == ["Holiday", "No Holiday"]
    pd.testing.assert_frame_equal(df, before)


def test_localize_preview_ignores_missing_columns():
    df = pd.DataFrame({"Rented Bike Count": [1, 2]})
    out = eda.localize_preview(df)
    pd.testing.assert_frame_equal(out, df)
