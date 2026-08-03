"""Sklearn-compatible feature engineering transformers.

Faithful port of notebook cells:
  [10] DateHourJoiner
  [11] TimePeriodTransformer
  [12] WeekdayWeekStatusTransformer
  [13] RushHourTransformer
  [15] AsCategoricalTransformer
  [50] RainfallCategorizer
  [51] SnowfallCategorizer
  [67] TargetAsFloatTransformer
  [68] DateAsColumnsTransformer
  [70] DropDateTransformer

Rule (§4): in-place mutations and side-effects are preserved exactly as
in the source notebook. Preservation of behaviour > function purity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# DateHourJoiner
# ---------------------------------------------------------------------------


class DateHourJoiner(BaseEstimator, TransformerMixin):
    """Join Date and Hour columns into a unique column and insert it as the first column."""

    def __init__(
        self,
        date_col: str = "DateTime",
        hour_col: str = "Hour",
        new_col: str = "Date",
    ) -> None:
        self.date_col = date_col
        self.hour_col = hour_col
        self.new_col = new_col

    def fit(self, X: pd.DataFrame, y=None) -> "DateHourJoiner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Join Date and Hour into a DateTime column, inserted as the first column."""
        X = X.copy()
        if self.new_col not in X.columns:
            X.insert(
                0, self.new_col, X[self.date_col] + pd.to_timedelta(X[self.hour_col], unit="h")
            )
        return X


# ---------------------------------------------------------------------------
# TimePeriodTransformer
# ---------------------------------------------------------------------------


class TimePeriodTransformer(BaseEstimator, TransformerMixin):
    """Divide day into Dawn, Morning, Afternoon, and Evening."""

    def __init__(self, date_column: str, hour_column: str) -> None:
        self.date_column = date_column
        self.hour_column = hour_column

    def fit(self, X: pd.DataFrame, y=None) -> "TimePeriodTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        hours = (
            X[self.hour_column] if self.hour_column in X.columns else X[self.date_column].dt.hour
        )
        X["Time_Period"] = np.select(
            [
                (0 <= hours) & (hours < 6),
                (6 <= hours) & (hours < 12),
                (12 <= hours) & (hours < 18),
                (18 <= hours) & (hours < 24),
            ],
            ["Dawn", "Morning", "Afternoon", "Evening"],
            default="Invalid time",
        )
        return X


# ---------------------------------------------------------------------------
# WeekdayWeekStatusTransformer
# ---------------------------------------------------------------------------


class WeekdayWeekStatusTransformer(BaseEstimator, TransformerMixin):
    """Add Weekday, DayNumberOnWeek, and WeekStatus columns."""

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "WeekdayWeekStatusTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X["Date"]):
            X["Date"] = pd.to_datetime(X["Date"], dayfirst=True)

        weekdays = np.where(X["Date"].dt.dayofweek < 5, "Weekday", "Weekend")
        day_numbers = X["Date"].dt.weekday + 1
        week_status = np.where(weekdays == "Weekend", "Weekend", "Weekday")

        X["Weekday"] = X["Date"].dt.day_name()
        X["DayNumberOnWeek"] = day_numbers
        X["WeekStatus"] = week_status

        return X


# ---------------------------------------------------------------------------
# RushHourTransformer
# ---------------------------------------------------------------------------


class RushHourTransformer(BaseEstimator, TransformerMixin):
    """Identify commuting windows on operating, non-holiday weekdays.

    Two complementary columns are produced. ``Rush_Hour`` preserves the
    original binary representation, while ``Rush_Period`` separates morning
    and evening peaks so their distinct demand levels are not collapsed into
    a single category.
    """

    def __init__(
        self,
        hour_col: str = "Hour",
        week_status_col: str = "WeekStatus",
        functioning_day_col: str = "Functioning Day",
        holiday_col: str = "Holiday",
    ) -> None:
        self.hour_col = hour_col
        self.week_status_col = week_status_col
        self.functioning_day_col = functioning_day_col
        self.holiday_col = holiday_col

    def fit(self, X: pd.DataFrame, y=None) -> "RushHourTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        is_weekday = X[self.week_status_col] == "Weekday"
        is_functioning_day = X[self.functioning_day_col] == "Yes"
        not_holiday = X[self.holiday_col] == "No Holiday"
        rush_hour_morning = (X[self.hour_col] >= 7) & (X[self.hour_col] <= 9)
        rush_hour_evening = (X[self.hour_col] >= 16) & (X[self.hour_col] <= 19)
        eligible_day = is_weekday & is_functioning_day & not_holiday

        X["Rush_Hour"] = (eligible_day & (rush_hour_morning | rush_hour_evening)).map(
            {True: "Rush", False: "No Rush"}
        )
        X["Rush_Period"] = np.select(
            [eligible_day & rush_hour_morning, eligible_day & rush_hour_evening],
            ["Morning Rush", "Evening Rush"],
            default="Non-Rush",
        )

        return X


# ---------------------------------------------------------------------------
# AsCategoricalTransformer
# ---------------------------------------------------------------------------


class AsCategoricalTransformer(BaseEstimator, TransformerMixin):
    """Convert object-dtype columns to category dtype."""

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "AsCategoricalTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        X_copy = pd.concat(
            [
                X_copy.select_dtypes([], ["object"]),
                X_copy.select_dtypes(["object"]).apply(pd.Series.astype, dtype="category"),
            ],
            axis=1,
        )
        return X_copy


# ---------------------------------------------------------------------------
# RainfallCategorizer
# ---------------------------------------------------------------------------


class RainfallCategorizer(BaseEstimator, TransformerMixin):
    """Categorize rainfall into No Rain / Light / Moderate / Heavy bins."""

    DEFAULT_BINS = [-float("inf"), 0, 2.5, 7.6, float("inf")]
    DEFAULT_LABELS = ["No Rain", "Light Rain", "Moderate Rain", "Heavy Rain"]

    def __init__(
        self,
        rainfall_col: str = "Rainfall(mm)",
        bins=None,
        labels=None,
        output_col: str = "Rainfall Cat",
        drop_original: bool = True,
    ) -> None:
        self.rainfall_col = rainfall_col
        self.bins = bins
        self.labels = labels
        self.output_col = output_col
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None) -> "RainfallCategorizer":
        return self

    def _resolved_bins_and_labels(self):
        bins = self.DEFAULT_BINS if self.bins is None else self.bins
        labels = self.DEFAULT_LABELS if self.labels is None else self.labels

        if len(labels) != len(bins) - 1:
            raise ValueError("Rainfall labels must have exactly len(bins) - 1 entries.")

        return bins, labels

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        bins, labels = self._resolved_bins_and_labels()
        X = X.copy()

        X[self.output_col] = pd.cut(
            X[self.rainfall_col],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        X[self.output_col] = X[self.output_col].astype("category")

        if self.drop_original:
            X.drop(self.rainfall_col, axis=1, inplace=True)

        return X


# ---------------------------------------------------------------------------
# SnowfallCategorizer
# ---------------------------------------------------------------------------


class SnowfallCategorizer(BaseEstimator, TransformerMixin):
    """Categorize snowfall into No Snow / Light / Moderate / Heavy bins."""

    def __init__(self, snowfall_col: str = "Snowfall (cm)") -> None:
        self.snowfall_col = snowfall_col

    def fit(self, X: pd.DataFrame, y=None) -> "SnowfallCategorizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        bins = [-float("inf"), 0, 1, 4, float("inf")]
        labels = ["No Snow", "Light Snow", "Moderate Snow", "Heavy Snow"]

        X["Snowfall Cat"] = pd.cut(
            X[self.snowfall_col], bins=bins, labels=labels, include_lowest=True
        )
        X["Snowfall Cat"] = X["Snowfall Cat"].astype("category")
        X.drop(self.snowfall_col, axis=1, inplace=True)
        return X


# ---------------------------------------------------------------------------
# SunshineCategorizer (v4)
# ---------------------------------------------------------------------------


class SunshineCategorizer(BaseEstimator, TransformerMixin):
    """Categorize hourly sunshine duration into No / Low / Moderate / Full bins.

    New in v4 (``Sunshine (hr)`` is not present in the v3 dataset). The source
    column is per-hour sunshine fraction in [0, 1]; it is heavily bimodal
    (~65% at 0, ~24% at 1). Unlike Rainfall/Snowfall, the original numeric
    column is kept for numeric EDA. If the source column is absent (e.g. the
    v3 dataset or synthetic test fixtures), the transformer is a no-op.
    """

    DEFAULT_BINS = [-float("inf"), 0, 0.3, 0.7, float("inf")]
    DEFAULT_LABELS = ["No Sun", "Low Sun", "Moderate Sun", "Full Sun"]

    def __init__(
        self,
        sunshine_col: str = "Sunshine (hr)",
        output_col: str = "Sunshine Cat",
        bins=None,
        labels=None,
    ) -> None:
        self.sunshine_col = sunshine_col
        self.output_col = output_col
        self.bins = bins
        self.labels = labels

    def fit(self, X: pd.DataFrame, y=None) -> "SunshineCategorizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.sunshine_col not in X.columns:
            return X

        bins = self.DEFAULT_BINS if self.bins is None else self.bins
        labels = self.DEFAULT_LABELS if self.labels is None else self.labels
        if len(labels) != len(bins) - 1:
            raise ValueError("Sunshine labels must have exactly len(bins) - 1 entries.")

        X = X.copy()
        X[self.output_col] = pd.cut(
            X[self.sunshine_col], bins=bins, labels=labels, include_lowest=True
        ).astype("category")
        return X


# ---------------------------------------------------------------------------
# CloudCoverCategorizer (v4)
# ---------------------------------------------------------------------------


class CloudCoverCategorizer(BaseEstimator, TransformerMixin):
    """Categorize cloud cover (0-10 scale) into Clear / Partly Cloudy / Overcast.

    New in v4 (``Cloud Cover (oktas)`` is not present in the v3 dataset). The
    original numeric column is kept for numeric EDA; missing values (~6.8% in
    the source) stay NaN in the category. No-op if the source column is absent.
    """

    DEFAULT_BINS = [-float("inf"), 2, 7, float("inf")]
    DEFAULT_LABELS = ["Clear", "Partly Cloudy", "Overcast"]

    def __init__(
        self,
        cloud_col: str = "Cloud Cover (oktas)",
        output_col: str = "Cloud Cover Cat",
        bins=None,
        labels=None,
    ) -> None:
        self.cloud_col = cloud_col
        self.output_col = output_col
        self.bins = bins
        self.labels = labels

    def fit(self, X: pd.DataFrame, y=None) -> "CloudCoverCategorizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.cloud_col not in X.columns:
            return X

        bins = self.DEFAULT_BINS if self.bins is None else self.bins
        labels = self.DEFAULT_LABELS if self.labels is None else self.labels
        if len(labels) != len(bins) - 1:
            raise ValueError("Cloud cover labels must have exactly len(bins) - 1 entries.")

        X = X.copy()
        X[self.output_col] = pd.cut(
            X[self.cloud_col], bins=bins, labels=labels, include_lowest=True
        ).astype("category")
        return X


# ---------------------------------------------------------------------------
# TargetAsFloatTransformer
# ---------------------------------------------------------------------------


class TargetAsFloatTransformer(BaseEstimator, TransformerMixin):
    """Cast 'Rented Bike Count' to float (side-effect preserved per §4)."""

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "TargetAsFloatTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.astype({"Rented Bike Count": "float"})
        return X


# ---------------------------------------------------------------------------
# DateAsColumnsTransformer
# ---------------------------------------------------------------------------


class DateAsColumnsTransformer(BaseEstimator, TransformerMixin):
    """Extract Month and Year from the Date column as separate integer columns."""

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "DateAsColumnsTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Month"] = X["Date"].dt.month
        X["Year"] = X["Date"].dt.year
        return X


# ---------------------------------------------------------------------------
# DropDateTransformer
# ---------------------------------------------------------------------------


class DropDateTransformer(BaseEstimator, TransformerMixin):
    """Remove the raw DateTime column (in-place drop, §4 preserved).

    Keeps Date (the DateHourJoiner-combined date+hour timestamp) for
    downstream time-series analysis — e.g. seasonal decomposition in the
    notebook indexes on ``transformed_df1['Date']``.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "DropDateTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=["DateTime"])


# ---------------------------------------------------------------------------
# ElapsedHoursTransformer (v4 model-selection only — not part of
# build_preprocessing_pipeline(), so notebooks 01-03 are unaffected)
# ---------------------------------------------------------------------------


class ElapsedHoursTransformer(BaseEstimator, TransformerMixin):
    """Derive an elapsed-time-since-reference feature from the timestamp alone.

    ``fit`` records the earliest timestamp seen in the training fold as
    ``reference_``; ``transform`` returns hours elapsed since that anchor.
    Because the anchor comes only from ``fit`` (never recomputed on the rows
    passed to ``transform``), a CV fold's test block is expressed relative to
    its own training window's start — the same fit/transform discipline as a
    ``StandardScaler`` — and the feature never reads the target, so it carries
    no leakage risk under temporal CV.
    """

    def __init__(self, date_col: str = "Date", output_col: str = "Elapsed_Hours") -> None:
        self.date_col = date_col
        self.output_col = output_col

    def fit(self, X: pd.DataFrame, y=None) -> "ElapsedHoursTransformer":
        self.reference_ = pd.to_datetime(X[self.date_col]).min()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        elapsed = pd.to_datetime(X[self.date_col]) - self.reference_
        X[self.output_col] = elapsed / pd.Timedelta(hours=1)
        return X


# ---------------------------------------------------------------------------
# Notebook 06 experimental features
# ---------------------------------------------------------------------------


class HourOfWeekTransformer(BaseEstimator, TransformerMixin):
    """Derive a 0..167 hour-of-week category from timestamp and hour columns.

    Monday at 00:00 maps to 0 and Sunday at 23:00 maps to 167. The transformer
    is target-free, clone-compatible and never mutates the input frame.
    """

    def __init__(
        self,
        date_col: str = "Date",
        raw_date_col: str = "DateTime",
        hour_col: str = "Hour",
        output_col: str = "HourOfWeek",
        as_category: bool = True,
    ) -> None:
        self.date_col = date_col
        self.raw_date_col = raw_date_col
        self.hour_col = hour_col
        self.output_col = output_col
        self.as_category = as_category

    def fit(self, X: pd.DataFrame, y=None) -> "HourOfWeekTransformer":
        return self

    def _timestamps(self, X: pd.DataFrame) -> pd.Series:
        if self.date_col in X.columns:
            return pd.to_datetime(X[self.date_col], errors="raise")
        if self.raw_date_col in X.columns:
            return pd.to_datetime(X[self.raw_date_col], errors="raise")
        if isinstance(X.index, pd.DatetimeIndex):
            return pd.Series(X.index, index=X.index)
        raise KeyError(
            f"HourOfWeekTransformer requires '{self.date_col}', "
            f"'{self.raw_date_col}', or a DatetimeIndex."
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        timestamps = self._timestamps(X_out)
        if self.hour_col in X_out.columns:
            hours = pd.to_numeric(X_out[self.hour_col], errors="raise").astype(int)
        else:
            hours = timestamps.dt.hour.astype(int)
        values = timestamps.dt.dayofweek.astype(int) * 24 + hours
        if not values.between(0, 167).all():
            raise ValueError("HourOfWeek must be between 0 and 167.")
        values = values.astype(int)
        if self.as_category:
            X_out[self.output_col] = pd.Categorical(
                values,
                categories=list(range(168)),
                ordered=True,
            )
        else:
            X_out[self.output_col] = values
        return X_out


class SelectiveWeatherInteractionTransformer(BaseEstimator, TransformerMixin):
    """Create a small, target-free interaction set for residual experiments.

    The interactions are deliberately categorical and selective. They describe
    operationally plausible combinations instead of expanding every pair of
    weather and calendar variables.
    """

    def __init__(
        self,
        hour_of_week_col: str = "HourOfWeek",
        rainfall_col: str = "Rainfall Cat",
        rush_period_col: str = "Rush_Period",
        temperature_col: str = "Temperature(C)",
        snowfall_col: str = "Snowfall Cat",
        cloud_col: str = "Cloud Cover Cat",
    ) -> None:
        self.hour_of_week_col = hour_of_week_col
        self.rainfall_col = rainfall_col
        self.rush_period_col = rush_period_col
        self.temperature_col = temperature_col
        self.snowfall_col = snowfall_col
        self.cloud_col = cloud_col

    def fit(self, X: pd.DataFrame, y=None) -> "SelectiveWeatherInteractionTransformer":
        return self

    @staticmethod
    def _as_text(series: pd.Series) -> pd.Series:
        return series.astype("object").where(series.notna(), "Missing").astype(str)

    def _temperature_band(self, X: pd.DataFrame) -> pd.Series:
        if self.temperature_col not in X.columns:
            return pd.Series("Missing", index=X.index)
        return (
            pd.cut(
                pd.to_numeric(X[self.temperature_col], errors="coerce"),
                bins=[-float("inf"), 0.0, 10.0, 20.0, 30.0, float("inf")],
                labels=["freezing", "cold", "mild", "warm", "hot"],
                include_lowest=True,
            )
            .astype("object")
            .where(lambda s: s.notna(), "Missing")
        )

    def _weather_regime(self, X: pd.DataFrame) -> pd.Series:
        rain = (
            self._as_text(X[self.rainfall_col])
            if self.rainfall_col in X.columns
            else pd.Series("Missing", index=X.index)
        )
        snow = (
            self._as_text(X[self.snowfall_col])
            if self.snowfall_col in X.columns
            else pd.Series("Missing", index=X.index)
        )
        cloud = (
            self._as_text(X[self.cloud_col])
            if self.cloud_col in X.columns
            else pd.Series("Missing", index=X.index)
        )
        return pd.Series(
            np.select(
                [
                    snow.ne("No Snow") & snow.ne("Missing"),
                    rain.ne("No Rain") & rain.ne("Missing"),
                    cloud.eq("Overcast"),
                    cloud.eq("Clear"),
                ],
                ["snow", "rain", "overcast", "clear"],
                default="other",
            ),
            index=X.index,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        hour = self._as_text(X_out[self.hour_of_week_col])
        rain = (
            self._as_text(X_out[self.rainfall_col])
            if self.rainfall_col in X_out.columns
            else pd.Series("Missing", index=X_out.index)
        )
        rush = (
            self._as_text(X_out[self.rush_period_col])
            if self.rush_period_col in X_out.columns
            else pd.Series("Missing", index=X_out.index)
        )
        temp_band = self._temperature_band(X_out).astype(str)
        weather = self._weather_regime(X_out).astype(str)

        X_out["Temperature_Band"] = pd.Categorical(temp_band)
        X_out["Weather_Regime"] = pd.Categorical(weather)
        X_out["HourOfWeek_Rainfall_Cat"] = pd.Categorical(hour + "__" + rain)
        X_out["HourOfWeek_Temperature_Band"] = pd.Categorical(hour + "__" + temp_band)
        X_out["Rush_Period_Rainfall_Cat"] = pd.Categorical(rush + "__" + rain)
        X_out["Rush_Period_Weather_Regime"] = pd.Categorical(rush + "__" + weather)
        return X_out


EXPERIMENTAL_CATEGORICAL_FEATURES = [
    "HourOfWeek",
    "Temperature_Band",
    "Weather_Regime",
    "HourOfWeek_Rainfall_Cat",
    "HourOfWeek_Temperature_Band",
    "Rush_Period_Rainfall_Cat",
    "Rush_Period_Weather_Regime",
]


def build_residual_uncertainty_feature_pipeline():
    """Return the target-free experimental feature block used only by notebook 06."""
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("hour_of_week", HourOfWeekTransformer()),
            ("weather_interactions", SelectiveWeatherInteractionTransformer()),
        ]
    )


# ---------------------------------------------------------------------------
# Full preprocessing pipeline factory (convenience)
# ---------------------------------------------------------------------------


def build_preprocessing_pipeline():
    """Return the full EDA-derived feature engineering pipeline used in the notebook."""
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("make_index", DateHourJoiner()),
            ("time_period", TimePeriodTransformer(date_column="Date", hour_column="Hour")),
            ("week_status", WeekdayWeekStatusTransformer()),
            ("rush_hour", RushHourTransformer()),
            ("as_categorical", AsCategoricalTransformer()),
            ("rainfall_categorizer", RainfallCategorizer(rainfall_col="Rainfall(mm)")),
            ("snowfall_categorizer", SnowfallCategorizer(snowfall_col="Snowfall (cm)")),
            # v4-only weather categories (no-op on the v3 schema / synthetic fixtures)
            ("sunshine_categorizer", SunshineCategorizer()),
            ("cloud_cover_categorizer", CloudCoverCategorizer()),
            ("time_transformer", DateAsColumnsTransformer()),
            ("no_date", DropDateTransformer()),
        ]
    )
