"""Tests for the leakage-safe robust trend/residual decomposition."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.trend import RobustTrendResidualRegressor


def _synthetic_growth_frame():
    timestamps = pd.date_range("2017-01-01", "2021-12-31 18:00", freq="6h")
    elapsed_years = (timestamps - timestamps.min()).total_seconds() / (365.2425 * 24 * 3600)
    seasonal = 0.35 * np.sin(2 * np.pi * timestamps.dayofyear.to_numpy() / 365.2425)
    log_demand = 5.0 + 0.32 * np.asarray(elapsed_years) + seasonal
    demand = np.asarray(np.expm1(log_demand))
    pandemic = timestamps.year.to_numpy() == 2020
    demand[pandemic] *= 0.18
    X = pd.DataFrame({"Date": timestamps})
    return X, pd.Series(demand, name="Rented Bike Count")


def test_low_demand_months_are_removed_only_from_the_trend_fit():
    X, y = _synthetic_growth_frame()
    model = RobustTrendResidualRegressor(DummyRegressor(strategy="mean"))
    model.fit(X, y)

    assert model.n_months_excluded_ > 0
    assert hasattr(model, "estimator_")


def test_baseline_extrapolates_beyond_the_training_period():
    X, y = _synthetic_growth_frame()
    train = X["Date"] < "2021-01-01"
    model = RobustTrendResidualRegressor(DummyRegressor(strategy="mean"))
    model.fit(X.loc[train], y.loc[train])

    future = pd.DataFrame({"Date": pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"])})
    baseline = model.predict_log_baseline(future)
    assert np.all(np.diff(baseline) > 0)


def test_predictions_are_reconstructed_on_the_non_negative_original_scale():
    X, y = _synthetic_growth_frame()
    train = X["Date"] < "2021-01-01"
    model = RobustTrendResidualRegressor(DummyRegressor(strategy="mean"))
    model.fit(X.loc[train], y.loc[train])

    prediction = model.predict(X.loc[~train])
    assert prediction.shape == (int((~train).sum()),)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()
    assert prediction.mean() > y.loc[train].mean()


def test_fitted_wrapper_is_picklable():
    X, y = _synthetic_growth_frame()
    model = RobustTrendResidualRegressor(DummyRegressor(strategy="mean")).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(X.head(20)), model.predict(X.head(20)))
