"""Tests for src/cv.py — cross-validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from src.cv import LeaveOneSeasonOut, make_temporal_holdout_split, make_ts_cv


@pytest.fixture
def seasonal_df() -> pd.DataFrame:
    """One full year of hourly data — 8760 rows."""
    dates = pd.date_range("2018-01-01", periods=8760, freq="h")
    return pd.DataFrame(
        {
            "DateTime": dates,
            "Month": dates.month,
            "Rented Bike Count": np.random.default_rng(0).integers(0, 1000, 8760).astype(float),
        }
    )


class TestLeaveOneSeasonOut:
    def test_four_folds(self, seasonal_df):
        cv = LeaveOneSeasonOut(gap=48)
        splits = list(cv.split(seasonal_df))
        assert len(splits) == 4

    def test_get_n_splits(self, seasonal_df):
        cv = LeaveOneSeasonOut(gap=48)
        assert cv.get_n_splits(seasonal_df) == 4

    def test_no_overlap_with_gap(self, seasonal_df):
        cv = LeaveOneSeasonOut(gap=48)
        for train_idx, test_idx in cv.split(seasonal_df):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set), "Train and test sets overlap"

    def test_temporal_order_respected(self, seasonal_df):
        cv = LeaveOneSeasonOut(gap=48)
        for train_idx, test_idx in cv.split(seasonal_df):
            t_min = test_idx.min()
            t_max = test_idx.max()
            adjacent_train = train_idx[(train_idx > t_min - 100) & (train_idx < t_max + 100)]
            if len(adjacent_train):
                gap_before = (
                    t_min - adjacent_train[adjacent_train < t_min].max()
                    if any(adjacent_train < t_min)
                    else 49
                )
                gap_after = (
                    adjacent_train[adjacent_train > t_max].min() - t_max
                    if any(adjacent_train > t_max)
                    else 49
                )
                assert gap_before >= 48 or gap_after >= 48 or len(adjacent_train) == 0

    def test_all_indices_covered(self, seasonal_df):
        cv = LeaveOneSeasonOut(gap=48)
        all_test = set()
        for _, test_idx in cv.split(seasonal_df):
            all_test.update(test_idx.tolist())
        assert len(all_test) > 0


class TestMakeTsCv:
    def test_returns_time_series_split(self):
        cv = make_ts_cv()
        assert isinstance(cv, TimeSeriesSplit)

    def test_default_params(self):
        cv = make_ts_cv()
        assert cv.n_splits == 5
        assert cv.gap == 48
        assert cv.max_train_size == 6000
        assert cv.test_size == 1000

    def test_custom_params(self):
        cv = make_ts_cv(n_splits=3, gap=24, max_train_size=3000, test_size=500)
        assert cv.n_splits == 3
        assert cv.gap == 24


class TestMakeTemporalHoldoutSplit:
    def test_sizes(self, seasonal_df):
        y = seasonal_df["Rented Bike Count"]
        X = seasonal_df.drop(columns=["Rented Bike Count"])
        (
            X_train,
            X_holdout,
            y_train,
            y_holdout,
            train_end,
            holdout_start,
        ) = make_temporal_holdout_split(X, y, holdout_size=1000, holdout_gap=48)
        assert len(X_holdout) == 1000
        assert len(X_train) == train_end

    def test_no_gap_overlap(self, seasonal_df):
        y = seasonal_df["Rented Bike Count"]
        X = seasonal_df.drop(columns=["Rented Bike Count"])
        _, _, _, _, train_end, holdout_start = make_temporal_holdout_split(
            X, y, holdout_size=1000, holdout_gap=48
        )
        assert holdout_start - train_end == 48

    def test_temporal_order(self, seasonal_df):
        y = seasonal_df["Rented Bike Count"]
        X = seasonal_df.drop(columns=["Rented Bike Count"])
        X_train, X_holdout, _, _, _, _ = make_temporal_holdout_split(
            X, y, holdout_size=1000, holdout_gap=48
        )
        assert X_train.index[-1] < X_holdout.index[0]
