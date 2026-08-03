"""Tests for src/feature_engineering.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import (
    CloudCoverCategorizer,
    DateAsColumnsTransformer,
    DateHourJoiner,
    DropDateTransformer,
    ElapsedHoursTransformer,
    RainfallCategorizer,
    RushHourTransformer,
    SnowfallCategorizer,
    SunshineCategorizer,
    TargetAsFloatTransformer,
    TimePeriodTransformer,
    WeekdayWeekStatusTransformer,
    build_preprocessing_pipeline,
)


class TestDateHourJoiner:
    def test_inserts_datetime_at_position_0(self, seoul_df):
        out = DateHourJoiner().fit_transform(seoul_df)
        assert out.columns[0] == "DateTime"
        assert pd.api.types.is_datetime64_any_dtype(out["DateTime"])

    def test_row_count_unchanged(self, seoul_df):
        out = DateHourJoiner().fit_transform(seoul_df)
        assert len(out) == len(seoul_df)


class TestTimePeriodTransformer:
    def test_adds_time_period_column(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        out = TimePeriodTransformer(date_column="DateTime", hour_column="Hour").fit_transform(df)
        assert "Time_Period" in out.columns

    def test_period_values(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        out = TimePeriodTransformer(date_column="DateTime", hour_column="Hour").fit_transform(df)
        valid = {"Dawn", "Morning", "Afternoon", "Evening"}
        assert set(out["Time_Period"].unique()).issubset(valid)


class TestWeekdayWeekStatusTransformer:
    def test_adds_weekday_columns(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        out = WeekdayWeekStatusTransformer().fit_transform(df)
        for col in ("Weekday", "DayNumberOnWeek", "WeekStatus"):
            assert col in out.columns

    def test_weekstatus_values(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        out = WeekdayWeekStatusTransformer().fit_transform(df)
        assert set(out["WeekStatus"].unique()).issubset({"Weekday", "Weekend"})


class TestRushHourTransformer:
    def test_adds_rush_hour_column(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        df = WeekdayWeekStatusTransformer().fit_transform(df)
        out = RushHourTransformer().fit_transform(df)
        assert "Rush_Hour" in out.columns

    def test_rush_hour_values(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        df = WeekdayWeekStatusTransformer().fit_transform(df)
        out = RushHourTransformer().fit_transform(df)
        assert set(out["Rush_Hour"].unique()).issubset({"Rush", "No Rush"})

    def test_weekends_and_holidays_are_not_rush_hours(self):
        df = pd.DataFrame(
            {
                "Hour": [8, 8, 17],
                "WeekStatus": ["Weekend", "Weekday", "Weekday"],
                "Functioning Day": ["Yes", "Yes", "Yes"],
                "Holiday": ["No Holiday", "Holiday", "No Holiday"],
            }
        )
        out = RushHourTransformer().fit_transform(df)

        assert out["Rush_Hour"].tolist() == ["No Rush", "No Rush", "Rush"]

    def test_rush_period_separates_morning_and_evening_peaks(self):
        df = pd.DataFrame(
            {
                "Hour": [8, 17, 12],
                "WeekStatus": ["Weekday"] * 3,
                "Functioning Day": ["Yes"] * 3,
                "Holiday": ["No Holiday"] * 3,
            }
        )
        out = RushHourTransformer().fit_transform(df)

        assert out["Rush_Period"].tolist() == ["Morning Rush", "Evening Rush", "Non-Rush"]


class TestRainfallCategorizer:
    def test_adds_rainfall_cat_drops_original(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        out = RainfallCategorizer().fit_transform(df)
        assert "Rainfall Cat" in out.columns
        assert "Rainfall(mm)" not in out.columns

    def test_categories(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        out = RainfallCategorizer().fit_transform(df)
        valid = {"No Rain", "Light Rain", "Moderate Rain", "Heavy Rain"}
        assert set(out["Rainfall Cat"].cat.categories).issubset(valid)


class TestSnowfallCategorizer:
    def test_adds_snowfall_cat_drops_original(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        out = SnowfallCategorizer().fit_transform(df)
        assert "Snowfall Cat" in out.columns
        assert "Snowfall (cm)" not in out.columns


class TestSunshineCategorizer:
    """v4-only: Sunshine (hr) -> Sunshine Cat, numeric kept, no-op when absent."""

    def test_adds_category_keeps_numeric(self):
        df = pd.DataFrame({"Sunshine (hr)": [0.0, 0.2, 0.5, 1.0]})
        out = SunshineCategorizer().fit_transform(df)
        assert "Sunshine Cat" in out.columns
        assert "Sunshine (hr)" in out.columns  # original kept (unlike Rainfall/Snowfall)
        assert out["Sunshine Cat"].dtype.name == "category"
        assert set(out["Sunshine Cat"].cat.categories).issubset(
            {"No Sun", "Low Sun", "Moderate Sun", "Full Sun"}
        )
        assert out.loc[0, "Sunshine Cat"] == "No Sun"

    def test_noop_when_column_absent(self, seoul_df):
        # The synthetic (v3-like) fixture has no Sunshine column.
        out = SunshineCategorizer().fit_transform(seoul_df.copy())
        assert "Sunshine Cat" not in out.columns


class TestCloudCoverCategorizer:
    """v4-only: Cloud Cover (oktas) -> Cloud Cover Cat, numeric kept, no-op when absent."""

    def test_adds_category_keeps_numeric(self):
        df = pd.DataFrame({"Cloud Cover (oktas)": [0.0, 2.0, 5.0, 10.0, np.nan]})
        out = CloudCoverCategorizer().fit_transform(df)
        assert "Cloud Cover Cat" in out.columns
        assert "Cloud Cover (oktas)" in out.columns
        assert out["Cloud Cover Cat"].dtype.name == "category"
        assert set(out["Cloud Cover Cat"].cat.categories).issubset(
            {"Clear", "Partly Cloudy", "Overcast"}
        )
        assert out.loc[0, "Cloud Cover Cat"] == "Clear"
        assert pd.isna(out.loc[4, "Cloud Cover Cat"])  # missing stays NaN

    def test_noop_when_column_absent(self, seoul_df):
        out = CloudCoverCategorizer().fit_transform(seoul_df.copy())
        assert "Cloud Cover Cat" not in out.columns


class TestTargetAsFloatTransformer:
    def test_target_is_float(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df)
        out = TargetAsFloatTransformer().fit_transform(df)
        assert out["Rented Bike Count"].dtype == float


class TestDropDateTransformer:
    def test_drops_datetime_keeps_date(self, preprocessed_df):
        """DateTime (raw) is dropped; Date (combined date+hour) survives for
        downstream time-series use — e.g. transformed_df1['Date'] in the notebook."""
        assert "DateTime" not in preprocessed_df.columns
        assert "Date" in preprocessed_df.columns


class TestBuildPreprocessingPipeline:
    def test_pipeline_runs_end_to_end(self, seoul_df):
        pipe = build_preprocessing_pipeline()
        out = pipe.fit_transform(seoul_df)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(seoul_df)

    def test_datetime_removed_date_kept(self, preprocessed_df):
        assert "DateTime" not in preprocessed_df.columns
        assert "Date" in preprocessed_df.columns

    def test_target_float(self, preprocessed_df):
        assert preprocessed_df["Rented Bike Count"].dtype == float

    def test_category_columns_present(self, preprocessed_df):
        for col in (
            "Rainfall Cat",
            "Snowfall Cat",
            "Time_Period",
            "Weekday",
            "WeekStatus",
            "Rush_Hour",
            "Rush_Period",
        ):
            assert col in preprocessed_df.columns, f"Missing column: {col}"


class TestNoInputMutation:
    """The v4 optimizer's CV pipeline calls fit_transform() once per fold; a
    transformer that mutates its input in place would corrupt whatever the
    caller still holds a reference to across folds/estimators. Each
    transformer touched for that fix must leave its input untouched."""

    def test_date_hour_joiner_does_not_mutate_input(self, seoul_df):
        before = seoul_df.copy()
        DateHourJoiner().fit_transform(seoul_df)
        pd.testing.assert_frame_equal(seoul_df, before)

    def test_weekday_week_status_does_not_mutate_input(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df.copy())
        before = df.copy()
        WeekdayWeekStatusTransformer().fit_transform(df)
        pd.testing.assert_frame_equal(df, before)

    def test_snowfall_categorizer_does_not_mutate_input(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df.copy())
        before = df.copy()
        SnowfallCategorizer().fit_transform(df)
        pd.testing.assert_frame_equal(df, before)

    def test_date_as_columns_does_not_mutate_input(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df.copy())
        df = WeekdayWeekStatusTransformer().fit_transform(df)  # parses 'Date' to datetime
        before = df.copy()
        DateAsColumnsTransformer().fit_transform(df)
        pd.testing.assert_frame_equal(df, before)

    def test_drop_date_transformer_does_not_mutate_input(self, seoul_df):
        df = DateHourJoiner().fit_transform(seoul_df.copy())
        before = df.copy()
        DropDateTransformer().fit_transform(df)
        pd.testing.assert_frame_equal(df, before)


class TestElapsedHoursTransformer:
    """v4-only, additive feature for the model-selection pipeline: hours
    elapsed since the training fold's own earliest timestamp. Derived purely
    from the timestamp (no target dependency), so it carries no leakage risk
    under temporal CV."""

    def test_fit_anchors_to_input_min(self):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=5, freq="h")})
        transformer = ElapsedHoursTransformer().fit(df)
        assert transformer.reference_ == df["Date"].min()

    def test_transform_computes_hours_since_reference(self):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=5, freq="h")})
        out = ElapsedHoursTransformer().fit(df).transform(df)
        assert out["Elapsed_Hours"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_transform_on_a_later_block_uses_the_fitted_reference(self):
        """Simulates a CV fold: fit on train, transform on test — the test
        block's elapsed hours must be measured from *train's* anchor, not
        recomputed from the test block's own (later) minimum."""
        train = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=5, freq="h")})
        test = pd.DataFrame({"Date": pd.date_range("2020-01-02", periods=2, freq="h")})
        transformer = ElapsedHoursTransformer().fit(train)
        out = transformer.transform(test)
        assert out["Elapsed_Hours"].tolist() == [24.0, 25.0]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame(
            {"Date": pd.date_range("2020-01-01", periods=3, freq="h"), "x": [1, 2, 3]}
        )
        before = df.copy()
        ElapsedHoursTransformer().fit_transform(df)
        pd.testing.assert_frame_equal(df, before)

    def test_fit_ignores_y(self):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=3, freq="h")})
        without_y = ElapsedHoursTransformer().fit(df, y=None)
        with_y = ElapsedHoursTransformer().fit(df, y=pd.Series([1, 2, 3]))
        assert without_y.reference_ == with_y.reference_
