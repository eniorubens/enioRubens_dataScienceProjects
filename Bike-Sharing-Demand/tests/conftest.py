"""pytest fixtures shared across test modules."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


@pytest.fixture
def seoul_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame matching the Seoul Bike schema.

    Mirrors the columns produced by read_data() after rename.
    Uses 500 rows so rolling windows (max 168 h) never fail on size.
    """
    rng = np.random.default_rng(42)
    n = 500

    dates = pd.date_range("2018-01-01", periods=n, freq="h")
    hours = dates.hour
    months = dates.month

    df = pd.DataFrame(
        {
            "DateTime": dates,
            "Date": dates.strftime("%d/%m/%Y"),
            "Hour": hours,
            "Rented Bike Count": rng.integers(0, 1000, size=n).astype(float),
            "Temperature(C)": rng.uniform(-10, 35, size=n),
            "Humidity(%)": rng.integers(10, 100, size=n).astype(float),
            "Wind speed (m/s)": rng.uniform(0, 8, size=n),
            "Visibility (10m)": rng.integers(100, 2000, size=n).astype(float),
            "Dew point temperature(C)": rng.uniform(-20, 25, size=n),
            "Ground Temp(C)": rng.uniform(-15, 45, size=n),
            "Solar Radiation (MJ/m2)": rng.uniform(0, 3.5, size=n),
            "Rainfall(mm)": np.where(rng.random(n) < 0.1, rng.uniform(0.1, 20, n), 0.0),
            "Snowfall (cm)": np.where(rng.random(n) < 0.05, rng.uniform(0.1, 5, n), 0.0),
            "Seasons": np.where(
                months.isin([12, 1, 2]),
                "Winter",
                np.where(
                    months.isin([3, 4, 5]),
                    "Spring",
                    np.where(months.isin([6, 7, 8]), "Summer", "Autumn"),
                ),
            ),
            "Holiday": np.where(rng.random(n) < 0.05, "Holiday", "No Holiday"),
            "Functioning Day": np.where(rng.random(n) < 0.02, "No", "Yes"),
        }
    )
    return df


@pytest.fixture(scope="session")
def raw_v4_df() -> pd.DataFrame:
    """Daily-frequency synthetic frame in the v4 raw loader shape.

    Spans Sep/2015-Nov/2024 so that every meteorological test year of
    ``ExpandingMeteorologicalYearSplit`` plus the sealed holdout are present,
    and carries every raw column the shared feature engineering pipeline and
    the v4 candidate feature lists expect. ``Sunshine (hr)`` and ``Cloud Cover
    (oktas)`` deliberately contain missing values, mirroring the real v4
    dataset, so the branch-imputation tests exercise a real NaN path.
    """
    dates = pd.date_range("2015-09-01", "2024-11-30", freq="1D")
    rng = np.random.default_rng(0)
    n = len(dates)
    months = dates.month
    season = np.select(
        [months.isin([12, 1, 2]), months.isin([3, 4, 5]), months.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        default="Autumn",
    )
    sunshine = rng.uniform(0, 1, n)
    sunshine[rng.random(n) < 0.08] = np.nan
    cloud = rng.uniform(0, 10, n)
    cloud[rng.random(n) < 0.08] = np.nan
    return pd.DataFrame(
        {
            "DateTime": dates.normalize(),
            "Hour": rng.integers(0, 24, n),
            "Rented Bike Count": rng.integers(0, 1000, n).astype(float),
            "Temperature(C)": rng.uniform(-15, 35, n),
            "Dew point temperature(C)": rng.uniform(-20, 25, n),
            "Ground Temp(C)": rng.uniform(-15, 45, n),
            "Humidity(%)": rng.integers(10, 100, n).astype(float),
            "Solar Radiation (MJ/m2)": rng.uniform(0, 3.5, n),
            "Wind speed (m/s)": rng.uniform(0, 8, n),
            "Visibility (10m)": rng.integers(100, 2000, n).astype(float),
            "Sunshine (hr)": sunshine,
            "Cloud Cover (oktas)": cloud,
            "Rainfall(mm)": np.where(rng.random(n) < 0.1, rng.uniform(0.1, 20, n), 0.0),
            "Snowfall (cm)": np.where(rng.random(n) < 0.05, rng.uniform(0.1, 5, n), 0.0),
            "Seasons": season,
            "Holiday": np.where(rng.random(n) < 0.05, "Holiday", "No Holiday"),
            "Functioning Day": "Yes",
        }
    )


@pytest.fixture(scope="session")
def dev_split_v4(raw_v4_df):
    """``(X_dev, y_dev, splitter)`` for the synthetic v4 frame, holdout sealed."""
    from src.cv import ExpandingMeteorologicalYearSplit, split_dev_holdout

    X_dev, y_dev, _ = split_dev_holdout(raw_v4_df)
    return X_dev, y_dev, ExpandingMeteorologicalYearSplit()


@pytest.fixture
def preprocessed_df(seoul_df) -> pd.DataFrame:
    """Preprocessed DataFrame (full pipeline applied, keeps all columns)."""
    from src.feature_engineering import build_preprocessing_pipeline

    pipe = build_preprocessing_pipeline()
    return pipe.fit_transform(seoul_df)
