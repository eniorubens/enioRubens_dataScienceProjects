"""Tests for src/cv.py's v4 additions: ExpandingMeteorologicalYearSplit and
split_dev_holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.cv import (
    _MONTH_TO_SEASON,
    ExpandingMeteorologicalYearSplit,
    expanding_meteorological_year_report,
    split_dev_holdout,
)
from src.seasonal import meteorological_year


@pytest.fixture(scope="module")
def raw_v4_df() -> pd.DataFrame:
    """Coarse (6h) synthetic frame in the raw loader shape (DateTime + Hour,
    no pre-existing 'Date' column), spanning Sep/2015 through Nov/2024 so all
    five expanding folds plus the sealed holdout are present."""
    dates = pd.date_range("2015-09-01", "2024-11-30 23:00", freq="6h")
    rng = np.random.default_rng(0)
    n = len(dates)
    return pd.DataFrame(
        {
            "DateTime": dates.normalize(),
            "Hour": dates.hour,
            "Rented Bike Count": rng.integers(0, 1000, n).astype(float),
            "Temperature(C)": rng.uniform(-15, 35, n),
        }
    )


class TestSplitDevHoldout:
    def test_holdout_row_count_matches_date_range(self, raw_v4_df):
        _, _, summary = split_dev_holdout(raw_v4_df)
        timestamps = pd.to_datetime(raw_v4_df["DateTime"])
        expected = int(
            (
                (timestamps >= pd.Timestamp("2023-12-01"))
                & (timestamps <= pd.Timestamp("2024-11-30"))
            ).sum()
        )
        assert summary.n_rows == expected
        assert summary.sealed is True

    def test_dev_set_is_strictly_before_holdout_start(self, raw_v4_df):
        X_dev, _, summary = split_dev_holdout(raw_v4_df)
        assert (pd.to_datetime(X_dev["DateTime"]) < summary.start).all()

    def test_summary_carries_no_row_data(self, raw_v4_df):
        _, _, summary = split_dev_holdout(raw_v4_df)
        assert not hasattr(summary, "X")
        assert not hasattr(summary, "y")
        assert not hasattr(summary, "X_holdout")
        assert not hasattr(summary, "y_holdout")

    def test_target_removed_from_X_dev(self, raw_v4_df):
        X_dev, y_dev, _ = split_dev_holdout(raw_v4_df)
        assert "Rented Bike Count" not in X_dev.columns
        assert len(X_dev) == len(y_dev)

    def test_summary_describes_the_development_window(self, raw_v4_df):
        X_dev, _, summary = split_dev_holdout(raw_v4_df)
        assert summary.n_dev_rows == len(X_dev)
        assert summary.dev_end < summary.start


@pytest.fixture(scope="module")
def raw_with_post_holdout() -> pd.DataFrame:
    """Hourly frame running to 31/12/2024, like the real v4 source.

    The holdout closes on 30/11/2024, so the last 744 hours belong to neither
    split. This is the shape that made ``df.loc[~holdout_mask]`` wrong.
    """
    dates = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="1h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "DateTime": dates.normalize(),
            "Hour": dates.hour,
            "Rented Bike Count": rng.integers(0, 1000, len(dates)).astype(float),
            "Temperature(C)": rng.uniform(-15, 35, len(dates)),
        }
    )


class TestPostHoldoutRowsAreDiscarded:
    """December/2024 is later than the holdout and must reach no split.

    Feeding it back into development would train the model on hours recorded
    *after* the period it is judged on — the leak that ``~holdout_mask``
    introduced.
    """

    def test_the_744_december_2024_hours_stay_out_of_development(self, raw_with_post_holdout):
        X_dev, _, summary = split_dev_holdout(raw_with_post_holdout)
        assert summary.n_post_holdout_rows == 744
        timestamps = pd.to_datetime(X_dev["DateTime"]) + pd.to_timedelta(X_dev["Hour"], unit="h")
        assert (timestamps < pd.Timestamp("2023-12-01")).all()
        assert timestamps.max() == pd.Timestamp("2023-11-30 23:00")

    def test_the_discarded_window_is_reported(self, raw_with_post_holdout):
        _, _, summary = split_dev_holdout(raw_with_post_holdout)
        assert summary.post_holdout_start == pd.Timestamp("2024-12-01 00:00")
        assert summary.post_holdout_end == pd.Timestamp("2024-12-31 23:00")

    def test_the_three_windows_partition_the_source(self, raw_with_post_holdout):
        _, _, summary = split_dev_holdout(raw_with_post_holdout)
        total = summary.n_dev_rows + summary.n_rows + summary.n_post_holdout_rows
        assert total == len(raw_with_post_holdout)

    def test_development_is_not_the_complement_of_the_holdout(self, raw_with_post_holdout):
        """The regression this fix exists for: the complement is 744 rows larger."""
        X_dev, _, summary = split_dev_holdout(raw_with_post_holdout)
        timestamps = pd.to_datetime(raw_with_post_holdout["DateTime"]) + pd.to_timedelta(
            raw_with_post_holdout["Hour"], unit="h"
        )
        holdout_mask = (timestamps >= pd.Timestamp("2023-12-01")) & (
            timestamps <= pd.Timestamp("2024-11-30 23:59:59")
        )
        assert len(X_dev) == int((~holdout_mask).sum()) - 744

    def test_no_holdout_row_reaches_development(self, raw_with_post_holdout):
        X_dev, _, _ = split_dev_holdout(raw_with_post_holdout)
        timestamps = pd.to_datetime(X_dev["DateTime"]) + pd.to_timedelta(X_dev["Hour"], unit="h")
        inside = (timestamps >= pd.Timestamp("2023-12-01")) & (
            timestamps <= pd.Timestamp("2024-11-30 23:59:59")
        )
        assert not inside.any()


@pytest.fixture(scope="module")
def real_v4_frame():
    """The actual v4 source, or a skip when it is not on this machine."""
    from src.data import read_data

    try:
        return read_data()
    except Exception as exc:  # pragma: no cover - depends on the local checkout
        pytest.skip(f"dataset v4 indisponível: {exc}")


class TestSplitDevHoldoutOnTheRealDataset:
    """The same guarantee, asserted against the file the notebook actually loads."""

    def test_source_really_extends_past_the_holdout(self, real_v4_frame):
        timestamps = pd.to_datetime(real_v4_frame["DateTime"]) + pd.to_timedelta(
            real_v4_frame["Hour"], unit="h"
        )
        assert timestamps.max() == pd.Timestamp("2024-12-31 23:00")

    def test_december_2024_does_not_enter_development(self, real_v4_frame):
        X_dev, y_dev, summary = split_dev_holdout(real_v4_frame)
        timestamps = pd.to_datetime(X_dev["DateTime"]) + pd.to_timedelta(X_dev["Hour"], unit="h")
        assert summary.n_post_holdout_rows == 744
        assert timestamps.max() < pd.Timestamp("2023-12-01")
        assert not (timestamps >= pd.Timestamp("2024-12-01")).any()
        assert len(X_dev) == len(y_dev) == summary.n_dev_rows


class TestExpandingMeteorologicalYearSplitFoldDates:
    EXPECTED_TEST_STARTS = [
        "2018-12-01",
        "2019-12-01",
        "2020-12-01",
        "2021-12-01",
        "2022-12-01",
    ]
    EXPECTED_TEST_ENDS = [
        "2019-11-30",
        "2020-11-30",
        "2021-11-30",
        "2022-11-30",
        "2023-11-30",
    ]

    def test_five_folds(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        assert splitter.get_n_splits(X_dev) == 5
        assert len(list(splitter.split(X_dev))) == 5

    def test_exact_test_block_boundaries(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        timestamps = splitter._get_timestamps(X_dev).reset_index(drop=True)

        for (_, test_idx), start, end in zip(
            splitter.split(X_dev), self.EXPECTED_TEST_STARTS, self.EXPECTED_TEST_ENDS
        ):
            test_ts = timestamps.iloc[test_idx]
            assert test_ts.min().normalize() == pd.Timestamp(start)
            assert test_ts.max().normalize() == pd.Timestamp(end)

    def test_train_strictly_before_test(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        timestamps = splitter._get_timestamps(X_dev).reset_index(drop=True)
        for train_idx, test_idx in splitter.split(X_dev):
            assert timestamps.iloc[train_idx].max() < timestamps.iloc[test_idx].min()

    def test_real_gap_is_at_least_48_hours(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit(gap=48)
        timestamps = splitter._get_timestamps(X_dev).reset_index(drop=True)
        for train_idx, test_idx in splitter.split(X_dev):
            gap_hours = (
                timestamps.iloc[test_idx].min() - timestamps.iloc[train_idx].max()
            ) / pd.Timedelta(hours=1)
            assert gap_hours >= 48

    def test_four_seasons_present_in_every_test_block(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        timestamps = splitter._get_timestamps(X_dev).reset_index(drop=True)
        for _, test_idx in splitter.split(X_dev):
            months = timestamps.iloc[test_idx].dt.month
            seasons = set(months.map(_MONTH_TO_SEASON))
            assert seasons == {"Winter", "Spring", "Summer", "Autumn"}

    def test_no_overlap_between_train_and_test(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        for train_idx, test_idx in splitter.split(X_dev):
            assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))

    def test_train_never_contains_a_future_meteorological_year(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        timestamps = splitter._get_timestamps(X_dev).reset_index(drop=True)
        met_year = meteorological_year(timestamps)
        for train_idx, test_idx in splitter.split(X_dev):
            test_year = met_year.iloc[test_idx[0]]
            assert (met_year.iloc[train_idx] < test_year).all()

    def test_folds_are_expanding_not_leave_one_out(self, raw_v4_df):
        """Later folds' training windows must be supersets of earlier ones —
        a year that was a test block earlier legitimately joins training once
        a later fold's test block has moved past it (unlike leave-one-year-out,
        which would keep excluding it forever)."""
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        train_sets = [set(train_idx.tolist()) for train_idx, _ in splitter.split(X_dev)]
        for earlier, later in zip(train_sets, train_sets[1:]):
            assert earlier.issubset(later)

    def test_holdout_rows_never_appear_in_any_fold(self, raw_v4_df):
        X_dev, _, summary = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        timestamps = splitter._get_timestamps(X_dev).reset_index(drop=True)
        for train_idx, test_idx in splitter.split(X_dev):
            all_idx = np.concatenate([train_idx, test_idx])
            assert (timestamps.iloc[all_idx] < summary.start).all()

    def test_sklearn_cross_validate_compatible(self, raw_v4_df):
        X_dev, y_dev, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        # cross_validate hands the *full* X to cv.split() itself (it needs the
        # DateTime/Hour columns); the pipeline below selects the numeric
        # column before fitting Ridge.
        pipeline = Pipeline(
            steps=[
                ("select", FunctionTransformer(lambda df: df[["Temperature(C)"]].fillna(0.0))),
                ("ridge", Ridge()),
            ]
        )
        result = cross_validate(
            pipeline, X_dev, y_dev, cv=splitter, scoring="neg_mean_absolute_error"
        )
        assert len(result["test_score"]) == 5


class TestExpandingMeteorologicalYearReport:
    def test_five_rows_with_expected_columns(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit()
        report = expanding_meteorological_year_report(X_dev, splitter)
        assert len(report) == 5
        assert "Gap real (horas)" in report.columns
        assert (report["Gap real (horas)"] >= 48).all()
        assert not report["Treino contém anos futuros"].any()


class TestBoundedRecentTrainingWindow:
    def test_rolling_window_keeps_only_the_configured_meteorological_years(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit(max_train_years=4)
        timestamps = splitter._get_timestamps(X_dev).reset_index(drop=True)
        met_year = meteorological_year(timestamps)

        for train_idx, test_idx in splitter.split(X_dev):
            test_year = int(met_year.iloc[test_idx[0]])
            train_years = set(met_year.iloc[train_idx].astype(int))
            assert train_years
            assert min(train_years) >= test_year - 4
            assert max(train_years) < test_year

    def test_test_blocks_remain_full_four_season_years(self, raw_v4_df):
        X_dev, _, _ = split_dev_holdout(raw_v4_df)
        splitter = ExpandingMeteorologicalYearSplit(max_train_years=4)
        timestamps = splitter._get_timestamps(X_dev).reset_index(drop=True)
        for _, test_idx in splitter.split(X_dev):
            seasons = set(timestamps.iloc[test_idx].dt.month.map(_MONTH_TO_SEASON))
            assert seasons == {"Winter", "Spring", "Summer", "Autumn"}

    def test_invalid_window_is_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            ExpandingMeteorologicalYearSplit(max_train_years=0)
