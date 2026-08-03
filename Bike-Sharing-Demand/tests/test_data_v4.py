"""v4-specific tests: the Seoul_public_bicycle schema adapter and LeaveOneYearOut.

These run against the real dataset/Seoul_public_bicycle.csv (7.4 MB, ships with
the project) because the adapter's whole job is faithfulness to that file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cv import LeaveOneYearOut
from src.data import _DEFAULT_DATASET, read_data

pytestmark = pytest.mark.skipif(
    not Path(_DEFAULT_DATASET).exists(),
    reason="real v4 dataset not present",
)


@pytest.fixture(scope="module")
def v4_df() -> pd.DataFrame:
    return read_data()


class TestV4SchemaAdapter:
    def test_v3_compatible_columns_present(self, v4_df):
        expected = {
            "DateTime",
            "Rented Bike Count",
            "Hour",
            "Temperature(C)",
            "Humidity(%)",
            "Wind speed (m/s)",
            "Visibility (10m)",
            "Dew point temperature(C)",
            "Solar Radiation (MJ/m2)",
            "Rainfall(mm)",
            "Snowfall (cm)",
            "Seasons",
            "Holiday",
            "Functioning Day",
        }
        assert expected <= set(v4_df.columns)

    def test_new_v4_features_present(self, v4_df):
        assert {"Sunshine (hr)", "Cloud Cover (oktas)", "Ground Temp(C)"} <= set(v4_df.columns)

    def test_period_and_row_count(self, v4_df):
        assert len(v4_df) == 81018
        assert v4_df["DateTime"].min() == pd.Timestamp("2015-09-19")
        assert v4_df["DateTime"].max() == pd.Timestamp("2024-12-31")

    def test_sorted_and_no_duplicate_timestamps(self, v4_df):
        stamps = v4_df["DateTime"] + pd.to_timedelta(v4_df["Hour"], unit="h")
        assert stamps.is_monotonic_increasing
        assert not stamps.duplicated().any()

    def test_visibility_converted_to_tens_of_meters(self, v4_df):
        # Compare against the source column directly: adapter divides meters by 10.
        raw = pd.read_csv(
            _DEFAULT_DATASET, usecols=["datetime", "visibility"], parse_dates=["datetime"]
        ).sort_values("datetime", ignore_index=True)
        assert np.allclose(
            v4_df["Visibility (10m)"].to_numpy(),
            (raw["visibility"] / 10.0).to_numpy(),
            equal_nan=True,
        )

    def test_zero_when_missing_columns_have_no_nans(self, v4_df):
        for col in ("Rainfall(mm)", "Snowfall (cm)", "Sunshine (hr)", "Solar Radiation (MJ/m2)"):
            assert v4_df[col].isna().sum() == 0, col

    def test_categorical_values_match_v3_vocabulary(self, v4_df):
        assert set(v4_df["Seasons"].unique()) == {"Winter", "Spring", "Summer", "Autumn"}
        assert set(v4_df["Holiday"].unique()) <= {"Holiday", "No Holiday"}
        assert set(v4_df["Functioning Day"].unique()) == {"Yes"}

    def test_optimizer_feature_lists_are_satisfied(self, v4_df):
        from src.feature_engineering import build_preprocessing_pipeline
        from src.optimizer import RegressionOptimizer

        df = build_preprocessing_pipeline().fit_transform(v4_df.copy())
        for col in RegressionOptimizer._NUMERICAL_FEATURES:
            assert col in df.columns, col
        for col in RegressionOptimizer._CATEGORICAL_FEATURES:
            assert col in df.columns, col


class TestLeaveOneYearOut:
    @pytest.fixture(scope="class")
    def X(self, v4_df):
        X = v4_df.copy()
        X["Year"] = X["DateTime"].dt.year
        return X

    def test_nine_full_year_folds(self, X):
        loyo = LeaveOneYearOut(gap=48)
        assert loyo.get_n_splits(X) == 9  # 2016..2024; partial 2015 excluded

    def test_partial_2015_never_a_test_fold_but_trains(self, X):
        loyo = LeaveOneYearOut(gap=48)
        years = X["Year"].to_numpy()
        for train_idx, test_idx in loyo.split(X):
            assert 2015 not in set(years[test_idx])
        # 2015 rows participate in training of at least the last fold
        train_idx, test_idx = list(loyo.split(X))[-1]
        assert 2015 in set(years[train_idx])

    def test_gap_trimmed_around_test_block(self, X):
        gap = 48
        loyo = LeaveOneYearOut(gap=gap)
        train_idx, test_idx = next(loyo.split(X))
        t_min, t_max = test_idx.min(), test_idx.max()
        inside_gap = train_idx[(train_idx >= t_min - gap) & (train_idx <= t_max + gap)]
        assert len(inside_gap) == 0

    def test_test_folds_are_disjoint_and_cover_full_years_only(self, X):
        loyo = LeaveOneYearOut(gap=48)
        seen: set = set()
        years = X["Year"].to_numpy()
        for _, test_idx in loyo.split(X):
            as_set = set(test_idx.tolist())
            assert not (as_set & seen)
            seen |= as_set
            assert len(set(years[test_idx])) == 1

    def test_requires_year_column_or_datetime_index(self):
        loyo = LeaveOneYearOut()
        bad = pd.DataFrame({"a": np.arange(10)})
        with pytest.raises(ValueError, match="Year"):
            next(loyo.split(bad))

    # --- meteorological year (Dec->Nov): keep each winter intact ---

    @pytest.fixture(scope="class")
    def X_met(self, v4_df):
        X = v4_df.copy()
        X["Year"] = X["DateTime"].dt.year
        X["Month"] = X["DateTime"].dt.month
        return X

    def test_meteorological_assigns_december_to_next_year(self, X_met):
        loyo = LeaveOneYearOut(gap=48, meteorological=True)
        met = loyo._get_years(X_met).to_numpy()
        cal_year = X_met["Year"].to_numpy()
        month = X_met["Month"].to_numpy()
        # December -> next meteorological year; all other months unchanged.
        dec = month == 12
        assert (met[dec] == cal_year[dec] + 1).all()
        assert (met[~dec] == cal_year[~dec]).all()

    def test_meteorological_never_splits_a_winter(self, X_met):
        loyo = LeaveOneYearOut(gap=48, meteorological=True)
        month = X_met["Month"].to_numpy()
        cal_year = X_met["Year"].to_numpy()
        # A winter instance is identified by (Dec year Y) + (Jan/Feb year Y+1);
        # under meteorological grouping all three share met-year Y+1, so each
        # winter's rows must land in exactly one test fold (never split).
        for _, test_idx in loyo.split(X_met):
            met_years = set((cal_year[test_idx] + (month[test_idx] == 12)).tolist())
            assert len(met_years) == 1

    def test_meteorological_requires_month(self, v4_df):
        loyo = LeaveOneYearOut(meteorological=True)
        no_month = v4_df.copy()
        no_month["Year"] = no_month["DateTime"].dt.year  # Year present, Month absent
        with pytest.raises(ValueError, match="Month"):
            next(loyo.split(no_month))
