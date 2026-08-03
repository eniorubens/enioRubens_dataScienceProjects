"""Tests for the nb02 detrended-frame helpers in src/seasonal.py."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import seasonal  # noqa: E402

TARGET = "Rented Bike Count"


@pytest.fixture
def full_frame() -> pd.DataFrame:
    stamps = pd.date_range("2018-01-01", periods=48, freq="h")
    return pd.DataFrame(
        {
            "Date": stamps,
            TARGET: np.arange(48, dtype=float) + 1,
            "Year": stamps.year,
            "Month": stamps.month,
            "met_year": stamps.year + (stamps.month == 12).astype(int),
            "Functioning Day": "Yes",
            "is_anomalous_2020": 0,
            "Temperature(C)": np.linspace(-5, 20, 48),
        }
    )


def test_build_analysis_frame_drops_and_keeps(full_frame):
    demand_index = pd.Series(np.linspace(0.5, 1.5, 48), index=full_frame.index)
    demand_index.iloc[:5] = np.nan  # edge rows without a baseline
    mv, meta = seasonal.build_analysis_frame(full_frame, demand_index, TARGET)

    assert len(mv) == 43  # NaN index rows dropped
    for dropped in ("Date", "Year", "met_year", "Functioning Day"):
        assert dropped not in mv.columns
    # target replaced by the index
    assert np.allclose(mv[TARGET].to_numpy(), demand_index.dropna().to_numpy())
    # meta keeps the raw count and calendar keys for the retained rows
    assert {"Year", "Month", "met_year", "raw_count"} <= set(meta.columns)
    assert len(meta) == 43


def test_time_index_correlations_returns_two_floats(full_frame):
    demand_index = pd.Series(np.linspace(0.5, 1.5, 48), index=full_frame.index)
    corr_idx, corr_raw = seasonal.time_index_correlations(full_frame, demand_index, TARGET)
    assert isinstance(corr_idx, float) and isinstance(corr_raw, float)
    # raw demand here is a perfect ramp -> corr(time, raw) == 1
    assert corr_raw == pytest.approx(1.0)


def test_plot_baseline_and_index_returns_fig(full_frame):
    demand_index = pd.Series(np.linspace(0.5, 1.5, 48), index=full_frame.index)
    fig, axes = seasonal.plot_baseline_and_index(full_frame, demand_index, TARGET)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    plt.close(fig)


def test_seasonal_shape_and_plot(full_frame):
    demand_index = pd.Series(np.linspace(0.5, 1.5, 48), index=full_frame.index)
    shape = seasonal.seasonal_shape_by_met_year(full_frame, demand_index)
    assert list(shape.index) == [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    fig, ax = seasonal.plot_seasonal_shape_by_met_year(full_frame, demand_index, min_months=0)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
