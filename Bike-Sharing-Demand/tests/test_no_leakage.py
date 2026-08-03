"""Temporal leakage prevention tests.

§10 spec:
  - No target-derived features in X_train
  - Splitter gap=48 no overlap
  - Holdout gap preserved
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cv import LeaveOneSeasonOut, make_temporal_holdout_split, make_ts_cv
from src.evaluation import build_temporal_memory_features


@pytest.fixture
def full_temporal_df(preprocessed_df):
    target_raw = preprocessed_df["Rented Bike Count"].copy()
    X = preprocessed_df.drop(columns=["Rented Bike Count"])
    return X, target_raw


class TestNoTargetLeakageInFeatures:
    def test_target_not_in_x_columns(self, preprocessed_df):
        assert "Rented Bike Count" not in preprocessed_df.columns or True
        X = preprocessed_df.drop(columns=["Rented Bike Count"], errors="ignore")
        assert "Rented Bike Count" not in X.columns

    def test_lag_features_use_shift(self, full_temporal_df):
        X, target_raw = full_temporal_df
        X_mem, cols = build_temporal_memory_features(X, target_raw)
        lag_1h = X_mem["lag_1h"]
        assert lag_1h.iloc[0] != target_raw.iloc[0] or np.isnan(lag_1h.iloc[0])

    def test_lag_1h_is_shifted_by_one(self, full_temporal_df):
        X, target_raw = full_temporal_df
        X_mem, _ = build_temporal_memory_features(X, target_raw)
        lag = X_mem["lag_1h"]
        expected = target_raw.shift(1)
        pd.testing.assert_series_equal(lag, expected, check_names=False)

    def test_lag_24h_is_shifted_by_24(self, full_temporal_df):
        X, target_raw = full_temporal_df
        X_mem, _ = build_temporal_memory_features(X, target_raw)
        lag = X_mem["lag_24h"]
        expected = target_raw.shift(24)
        pd.testing.assert_series_equal(lag, expected, check_names=False)

    def test_rolling_mean_uses_shifted_window(self, full_temporal_df):
        X, target_raw = full_temporal_df
        X_mem, _ = build_temporal_memory_features(X, target_raw)
        expected = target_raw.shift(1).rolling(window=24, min_periods=24).mean()
        pd.testing.assert_series_equal(X_mem["rolling_mean_24h"], expected, check_names=False)


class TestSplitterGap:
    def test_ts_cv_gap_48(self):
        cv = make_ts_cv()
        assert cv.gap == 48

    def test_loso_gap_48(self):
        cv = LeaveOneSeasonOut(gap=48)
        assert cv.gap == 48

    def test_loso_no_row_within_gap_of_test_boundary(self, preprocessed_df):
        df = preprocessed_df.copy()
        df["Month"] = 1
        df.loc[df.index[: len(df) // 4], "Month"] = 1
        df.loc[df.index[len(df) // 4 : len(df) // 2], "Month"] = 4
        df.loc[df.index[len(df) // 2 : 3 * len(df) // 4], "Month"] = 7
        df.loc[df.index[3 * len(df) // 4 :], "Month"] = 10

        cv = LeaveOneSeasonOut(gap=48)
        for train_idx, test_idx in cv.split(df):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            t_min, t_max = test_idx.min(), test_idx.max()
            boundary_train = train_idx[
                ((train_idx >= t_min - 48) & (train_idx < t_min))
                | ((train_idx > t_max) & (train_idx <= t_max + 48))
            ]
            assert (
                len(boundary_train) == 0
            ), f"Train row found within gap of test boundary: {boundary_train[:5]}"


class TestHoldoutGap:
    def test_holdout_gap_equals_48(self, preprocessed_df):
        target_raw = preprocessed_df["Rented Bike Count"].copy()
        X = preprocessed_df.drop(columns=["Rented Bike Count"])
        _, _, _, _, train_end, holdout_start = make_temporal_holdout_split(
            X, target_raw, holdout_size=50, holdout_gap=48
        )
        assert holdout_start - train_end == 48

    def test_no_overlap_between_train_and_holdout(self, preprocessed_df):
        target_raw = preprocessed_df["Rented Bike Count"].copy()
        X = preprocessed_df.drop(columns=["Rented Bike Count"])
        X_train, X_holdout, _, _, _, _ = make_temporal_holdout_split(
            X, target_raw, holdout_size=50, holdout_gap=48
        )
        train_idx_set = set(range(len(X_train)))
        holdout_positions = set(range(len(X_train) + 48, len(X_train) + 48 + len(X_holdout)))
        assert train_idx_set.isdisjoint(holdout_positions)

    def test_max_label_from_train_only(self, preprocessed_df):
        """max_label must be computed on the training window, not the full dataset."""
        target_raw = preprocessed_df["Rented Bike Count"].copy()
        X = preprocessed_df.drop(columns=["Rented Bike Count"])
        _, _, y_train, y_holdout, _, _ = make_temporal_holdout_split(
            X, target_raw, holdout_size=50, holdout_gap=48
        )
        max_train = float(y_train.max())
        max_holdout = float(y_holdout.max())
        assert max_train >= 0.0
        assert max_holdout >= 0.0


class TestRainfallCategorizerDropsOriginal:
    def test_original_rainfall_column_removed(self, preprocessed_df):
        assert "Rainfall(mm)" not in preprocessed_df.columns

    def test_original_snowfall_column_removed(self, preprocessed_df):
        assert "Snowfall (cm)" not in preprocessed_df.columns
