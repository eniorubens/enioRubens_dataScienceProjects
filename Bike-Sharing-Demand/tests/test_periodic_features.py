"""Tests for src/periodic_features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.periodic_features import (
    CosTransformer,
    DebugTransformer,
    PeriodicSplineTransformer,
    SinTransformer,
)


@pytest.fixture
def hour_series() -> pd.DataFrame:
    return pd.DataFrame({"Hour": np.arange(0, 24, dtype=float)})


@pytest.fixture
def month_series() -> pd.DataFrame:
    return pd.DataFrame({"Month": np.arange(1, 13, dtype=float)})


class TestSinTransformer:
    def test_shape_preserved(self, hour_series):
        out = SinTransformer(period=24).fit_transform(hour_series)
        assert out.shape == hour_series.shape

    def test_range(self, hour_series):
        out = SinTransformer(period=24).fit_transform(hour_series)
        assert np.all(out >= -1) and np.all(out <= 1)

    def test_deterministic(self, hour_series):
        t1 = SinTransformer(period=24).fit_transform(hour_series)
        t2 = SinTransformer(period=24).fit_transform(hour_series)
        np.testing.assert_array_equal(t1, t2)

    def test_period_affects_output(self, hour_series):
        out12 = SinTransformer(period=12).fit_transform(hour_series)
        out24 = SinTransformer(period=24).fit_transform(hour_series)
        assert not np.allclose(out12, out24)


class TestCosTransformer:
    def test_shape_preserved(self, hour_series):
        out = CosTransformer(period=24).fit_transform(hour_series)
        assert out.shape == hour_series.shape

    def test_range(self, hour_series):
        out = CosTransformer(period=24).fit_transform(hour_series)
        assert np.all(out >= -1) and np.all(out <= 1)

    def test_hour_zero_is_one(self):
        df = pd.DataFrame({"x": [0.0]})
        out = CosTransformer(period=24).fit_transform(df)
        val = out.iloc[0, 0] if hasattr(out, "iloc") else out[0, 0]
        np.testing.assert_allclose(val, 1.0, atol=1e-12)


class TestPeriodicSplineTransformer:
    def test_output_columns(self, hour_series):
        out = PeriodicSplineTransformer(period=24, n_splines=12).fit_transform(hour_series)
        assert out.shape[0] == 24
        assert out.shape[1] > 1

    def test_deterministic(self, hour_series):
        t1 = PeriodicSplineTransformer(24, 12).fit_transform(hour_series)
        t2 = PeriodicSplineTransformer(24, 12).fit_transform(hour_series)
        np.testing.assert_array_equal(t1, t2)

    def test_month_spline(self, month_series):
        out = PeriodicSplineTransformer(period=12, n_splines=6).fit_transform(month_series)
        assert out.shape[0] == 12


class TestDebugTransformer:
    def test_passthrough(self, hour_series):
        out = DebugTransformer().fit_transform(hour_series)
        pd.testing.assert_frame_equal(out, hour_series)

    def test_returns_same_object(self, hour_series):
        transformer = DebugTransformer()
        out = transformer.fit_transform(hour_series)
        assert out is hour_series
