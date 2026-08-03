"""Tests for src/seasonal.py — meteorological year and demand de-trending."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.seasonal import demand_index_moving_average, meteorological_year


class TestMeteorologicalYear:
    def test_december_shifts_to_next_year(self):
        dates = pd.to_datetime(["2017-12-15", "2018-01-10", "2018-02-20", "2018-11-30"])
        met = meteorological_year(pd.Series(dates))
        assert met.tolist() == [2018, 2018, 2018, 2018]

    def test_non_december_unchanged(self):
        dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-06-15", "2020-11-30"]))
        assert meteorological_year(dates).tolist() == [2020, 2020, 2020]

    def test_winter_shares_one_meteorological_year(self):
        # Dec 2019 + Jan/Feb 2020 must all map to met-year 2020.
        dates = pd.Series(pd.to_datetime(["2019-12-31", "2020-01-01", "2020-02-28"]))
        assert set(meteorological_year(dates)) == {2020}


class TestDemandIndexMovingAverage:
    @staticmethod
    def _synthetic(years=3, seasonal_amp=0.6, growth=2.0):
        """Hourly series with a yearly sinusoidal season and a growth trend."""
        idx = pd.date_range("2016-01-01", periods=years * 8760, freq="h")
        t = np.arange(len(idx))
        doy = idx.dayofyear.to_numpy()
        season = 1 + seasonal_amp * np.sin(2 * np.pi * (doy - 80) / 365.0)
        trend = growth ** (t / len(idx))  # multiplicative growth across the span
        demand = 100 * trend * season
        return pd.DataFrame({"Date": idx, "Rented Bike Count": demand})

    def test_removes_growth_trend(self):
        df = self._synthetic()
        idx = demand_index_moving_average(df)
        valid = idx.notna()
        t = np.arange(len(df))[valid.to_numpy()]
        # correlation of the index with time is ~0 once the trend is removed.
        assert abs(np.corrcoef(t, idx[valid])[0, 1]) < 0.1

    def test_preserves_seasonal_signal(self):
        df = self._synthetic(seasonal_amp=0.6)
        idx = demand_index_moving_average(df)
        month = df["Date"].dt.month
        monthly = idx.groupby(month.to_numpy()).mean()
        # a 365-day baseline keeps the seasonal cycle: summer clearly > winter.
        assert monthly.max() - monthly.min() > 0.3

    def test_index_defined_in_the_interior(self):
        df = self._synthetic()
        idx = demand_index_moving_average(df)
        assert idx.notna().mean() > 0.6
        # centred window => the middle year is fully covered.
        mid = slice(8760, 2 * 8760)
        assert idx.iloc[mid].notna().all()

    def test_anomaly_mask_excluded_from_baseline(self):
        df = self._synthetic()
        # Force a low-demand anomalous block and confirm it does not crash and
        # the masked rows still receive an index.
        anom = pd.Series(False, index=df.index)
        anom.iloc[10000:11000] = True
        df.loc[df.index[10000:11000], "Rented Bike Count"] *= 0.05
        idx = demand_index_moving_average(df, anomaly_mask=anom)
        assert idx.iloc[10000:11000].notna().any()

    def test_aligned_to_input_index(self):
        df = self._synthetic(years=2)
        idx = demand_index_moving_average(df)
        assert idx.index.equals(df.index)
