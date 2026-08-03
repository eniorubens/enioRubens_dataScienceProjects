"""Test for leave_one_year_out_report in src/cv.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cv import LeaveOneYearOut, leave_one_year_out_report


def _three_year_X() -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", "2021-12-31 23:00", freq="h")
    X = pd.DataFrame({"feature": np.arange(len(dates))}, index=pd.DatetimeIndex(dates))
    return X


def test_report_shape_and_localized_columns():
    X = _three_year_X()
    loyo = LeaveOneYearOut(gap=48)
    years = pd.Series(X.index.year).reset_index(drop=True)
    report = leave_one_year_out_report(X, loyo, years)

    assert len(report) == loyo.get_n_splits(X)
    assert "Ano meteorológico testado" in report.columns
    assert "Treino contém anos futuros" in report.columns
    # First (earliest) test year must see later years in training.
    assert bool(report.iloc[0]["Treino contém anos futuros"]) is True
    # Train/test row counts are positive for every fold.
    assert (report["Linhas de treino"] > 0).all()
    assert (report["Linhas de teste"] > 0).all()
