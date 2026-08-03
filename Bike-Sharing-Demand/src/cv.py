"""Cross-validation utilities.

Faithful port of notebook cells:
  [69] TimeSeriesSplit import
  [70] LeaveOneSeasonOut (BaseCrossValidator with gap=48)
  [71] make_ts_cv (TimeSeriesSplit with gap=48, max_train_size=6000, test_size=1000)
  [73] train/holdout temporal split logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit

from src.i18n import resolve_lang as _resolve_lang
from src.seasonal import meteorological_year


# ---------------------------------------------------------------------------
# LeaveOneSeasonOut
# ---------------------------------------------------------------------------


class LeaveOneSeasonOut(BaseCrossValidator):
    """Leave-One-Season-Out CV for seasonal time series (Seoul).

    Each fold: train on 3 seasons, test on 1.
    Requires either a DatetimeIndex or a 'Month' integer column on X.
    Applies a gap (hours) at season boundaries to prevent neighbor leakage.
    """

    SEASONS = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Fall": [9, 10, 11],
    }

    def __init__(self, gap: int = 48) -> None:
        self.gap = gap

    @staticmethod
    def _get_months(X: pd.DataFrame) -> pd.Series:
        if isinstance(X.index, pd.DatetimeIndex):
            return X.index.month
        if "Month" in X.columns:
            return X["Month"]
        raise ValueError(
            "LeaveOneSeasonOut requires X to have a DatetimeIndex or a 'Month' column."
        )

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        months = self._get_months(X)
        for season_months in self.SEASONS.values():
            test_mask = months.isin(season_months)
            train_mask = ~test_mask
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]
            if len(test_idx) and len(train_idx):
                t_min, t_max = test_idx.min(), test_idx.max()
                train_idx = train_idx[
                    (train_idx < t_min - self.gap) | (train_idx > t_max + self.gap)
                ]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.SEASONS)


# ---------------------------------------------------------------------------
# LeaveOneYearOut (v4)
# ---------------------------------------------------------------------------


class LeaveOneYearOut(BaseCrossValidator):
    """Leave-One-Year-Out CV for the 2015-2024 multi-year dataset.

    With ~9 complete years (and therefore ~9 winters) available, holding out a
    whole calendar year is the natural upgrade over ``LeaveOneSeasonOut``: it
    tests generalization to an unseen demand *regime* (fleet size, ridership
    level, weather realization) rather than to an unseen season, and it does
    not conflate "never saw a winter" with "never saw this year's scale".

    Each fold: train on all other years, test on one year. Requires either a
    DatetimeIndex or a 'Year' integer column on X. A gap (hours) is trimmed
    from the training indices around the test year's boundaries to prevent
    neighbor leakage, mirroring ``LeaveOneSeasonOut``.

    Parameters
    ----------
    gap:
        Hours removed from training on each side of the test block.
    min_test_hours:
        Years with fewer in-sample hours than this are skipped as test folds
        (default 4380 ≈ half a year, which excludes the partial 2015 that
        starts in September) — they still participate in training.
    meteorological:
        When True, group by *meteorological year* (Dec→Nov) instead of the
        calendar year: December is assigned to the following year's fold so a
        winter (Dec-Jan-Feb) is never split across two folds. The calendar-year
        boundary (Jan 1) cuts the middle of winter, which is arbitrary for a
        weather-driven series; the meteorological year keeps each season intact.
        Requires a DatetimeIndex or both 'Year' and 'Month' columns.
    """

    def __init__(
        self, gap: int = 48, min_test_hours: int = 4380, meteorological: bool = False
    ) -> None:
        self.gap = gap
        self.min_test_hours = min_test_hours
        self.meteorological = meteorological

    def _get_years(self, X: pd.DataFrame) -> pd.Series:
        if isinstance(X.index, pd.DatetimeIndex):
            year = pd.Series(X.index.year, index=X.index)
            month = pd.Series(X.index.month, index=X.index)
        elif "Year" in X.columns:
            year = X["Year"]
            month = X["Month"] if "Month" in X.columns else None
        else:
            raise ValueError(
                "LeaveOneYearOut requires X to have a DatetimeIndex or a 'Year' column."
            )
        if self.meteorological:
            if month is None:
                raise ValueError(
                    "meteorological=True requires a DatetimeIndex or a 'Month' column "
                    "so December can be shifted into the next year's fold."
                )
            # December belongs to the following meteorological year (Dec-Nov).
            return year + (np.asarray(month) == 12).astype(int)
        return year

    def _test_years(self, X: pd.DataFrame) -> list:
        counts = self._get_years(X).value_counts()
        return sorted(year for year, n in counts.items() if n >= self.min_test_hours)

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        years = np.asarray(self._get_years(X))
        for test_year in self._test_years(X):
            test_mask = years == test_year
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            if len(test_idx) and len(train_idx):
                t_min, t_max = test_idx.min(), test_idx.max()
                train_idx = train_idx[
                    (train_idx < t_min - self.gap) | (train_idx > t_max + self.gap)
                ]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            raise ValueError(
                "LeaveOneYearOut.get_n_splits needs X: the number of folds "
                "depends on how many years clear min_test_hours."
            )
        return len(self._test_years(X))


# ---------------------------------------------------------------------------
# Inner-CV factory (TimeSeriesSplit)
# ---------------------------------------------------------------------------


def make_ts_cv(
    n_splits: int = 5,
    gap: int = 48,
    max_train_size: int = 6000,
    test_size: int = 1000,
) -> TimeSeriesSplit:
    """Return the TimeSeriesSplit used for inner hyperparameter-tuning CV.

    Port of notebook cell [71]:
      ts_cv = TimeSeriesSplit(n_splits=5, gap=48, max_train_size=6000, test_size=1000)

    Parameters
    ----------
    n_splits:
        Number of CV splits.
    gap:
        Number of samples dropped between each training set and the
        following test set (48 hours to avoid temporal leakage at boundaries).
    max_train_size:
        Maximum number of samples in the training window.
    test_size:
        Number of samples in each test window.
    """
    return TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        max_train_size=max_train_size,
        test_size=test_size,
    )


# ---------------------------------------------------------------------------
# Temporal holdout split
# ---------------------------------------------------------------------------


def make_temporal_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    holdout_size: int = 1000,
    holdout_gap: int = 48,
):
    """Create training and holdout windows preserving temporal order.

    Port of notebook cell [73].

    Parameters
    ----------
    X:
        Feature matrix (full dataset, post-feature-engineering).
    y:
        Target series (already normalised by max_label or raw).
    holdout_size:
        Number of rows reserved as the final temporal holdout.
    holdout_gap:
        Number of rows between the end of the training window and the
        start of the holdout (48-hour buffer to prevent boundary leakage).

    Returns
    -------
    X_train_opt, X_holdout, y_train_opt, y_holdout, train_end, holdout_start
    """
    holdout_start = len(X) - holdout_size
    train_end = holdout_start - holdout_gap

    X_train_opt = X.iloc[:train_end].copy()
    X_holdout = X.iloc[holdout_start:].copy()
    y_train_opt = y.iloc[:train_end].copy()
    y_holdout = y.iloc[holdout_start:].copy()

    return X_train_opt, X_holdout, y_train_opt, y_holdout, train_end, holdout_start


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def leave_one_year_out_report(
    X: pd.DataFrame,
    loyo: LeaveOneYearOut,
    year_values,
    lang=None,
) -> pd.DataFrame:
    """Summarize each Leave-One-Year-Out fold as a localized report row.

    Parameters
    ----------
    X:
        Feature matrix the splitter runs on.
    loyo:
        A configured :class:`LeaveOneYearOut` instance.
    year_values:
        Per-row (meteorological) year, in the same positional order as ``X``
        (e.g. ``meteorological_year(timestamp).reset_index(drop=True)``).
    lang:
        Optional ``LangMap``; column labels are localized (canonical PT).
    """
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "test_year": "Ano meteorológico testado",
            "train_rows": "Linhas de treino",
            "test_rows": "Linhas de teste",
            "first_train_year": "Primeiro ano no treino",
            "last_train_year": "Último ano no treino",
            "has_future": "Treino contém anos futuros",
        }
    )
    yv = np.asarray(year_values)
    rows = []
    for train_idx, test_idx in loyo.split(X):
        test_year = int(yv[test_idx[0]])
        train_years = sorted(pd.unique(yv[train_idx]))
        rows.append(
            {
                labels["test_year"]: test_year,
                labels["train_rows"]: len(train_idx),
                labels["test_rows"]: len(test_idx),
                labels["first_train_year"]: int(min(train_years)),
                labels["last_train_year"]: int(max(train_years)),
                labels["has_future"]: any(year > test_year for year in train_years),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ExpandingMeteorologicalYearSplit (v4 model-selection CV)
# ---------------------------------------------------------------------------

_MONTH_TO_SEASON = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


def _resolve_timestamps(
    X: pd.DataFrame,
    date_col: str = "Date",
    raw_date_col: str = "DateTime",
    hour_col: str = "Hour",
) -> pd.Series:
    """Resolve hour-precise timestamps for ``X``, working both before and
    after feature engineering has run.

    Preference order: a ``DatetimeIndex``; ``date_col`` (the hour-precise
    ``Date`` column produced by ``DateHourJoiner`` once feature engineering
    has run); otherwise ``raw_date_col``+``hour_col`` combined (the raw
    loader schema — a midnight-normalised ``DateTime`` plus a separate
    ``Hour`` column, present *before* feature engineering). The 48-hour gap
    used by ``ExpandingMeteorologicalYearSplit`` needs this hour precision:
    a date-only comparison could be off by up to 23 hours.
    """
    if isinstance(X.index, pd.DatetimeIndex):
        return pd.Series(X.index, index=X.index)
    if date_col in X.columns:
        return pd.to_datetime(X[date_col])
    if raw_date_col in X.columns and hour_col in X.columns:
        return pd.to_datetime(X[raw_date_col]) + pd.to_timedelta(X[hour_col], unit="h")
    raise ValueError(
        f"Could not resolve timestamps: need a DatetimeIndex, a '{date_col}' column, "
        f"or both '{raw_date_col}' and '{hour_col}' columns."
    )


class ExpandingMeteorologicalYearSplit(BaseCrossValidator):
    """Forward CV over meteorological years, optionally bounded to recent history.

    Each fold trains on every row strictly before ``test_start - gap`` and
    tests on one full meteorological year (e.g. test_year=2019 covers
    Dec/2018-Nov/2019, via :func:`src.seasonal.meteorological_year`). Later
    folds' training windows are supersets of earlier ones plus more history —
    this is what makes the CV "expanding" rather than leave-one-year-out: a
    year that was a test block in an earlier fold legitimately becomes part
    of training once a later fold's test block has moved past it.

    The gap is validated against real timestamps, not row counts: the hourly
    grid has ~390 missing stamps (mostly 2015-2017), so trimming by row
    count could silently leave less (or more) than ``gap`` hours of real
    buffer around a hole.

    Parameters
    ----------
    test_years:
        Meteorological test years, in order. Defaults to the five folds
        specified for notebook 04 (2019-2023) — the sealed final holdout
        (Dec/2023-Nov/2024, meteorological year 2024) is deliberately not in
        this list, so it can never become a test (or train) block here even
        if a caller forgot to seal it out of ``X`` first.
    gap:
        Hours required between the last training timestamp and the first
        test timestamp.
    max_train_years:
        When ``None``, every earlier observation is retained (expanding
        window).  A positive integer limits training to that many complete
        meteorological years before each test block (rolling window).  Test
        boundaries and their four-season coverage are unchanged.
    date_col, raw_date_col, hour_col:
        Columns used to resolve a timestamp — see :func:`_resolve_timestamps`.
        Works both on the raw (pre-feature-engineering) schema, which only has
        ``raw_date_col``+``hour_col``, and on the engineered schema, which has
        ``date_col`` already hour-precise.
    """

    def __init__(
        self,
        test_years: Sequence[int] = (2019, 2020, 2021, 2022, 2023),
        gap: int = 48,
        date_col: str = "Date",
        raw_date_col: str = "DateTime",
        hour_col: str = "Hour",
        max_train_years: Optional[int] = None,
    ) -> None:
        if max_train_years is not None and max_train_years < 1:
            raise ValueError("max_train_years must be a positive integer or None.")
        self.test_years = tuple(test_years)
        self.gap = gap
        self.date_col = date_col
        self.raw_date_col = raw_date_col
        self.hour_col = hour_col
        self.max_train_years = max_train_years

    def _get_timestamps(self, X: pd.DataFrame) -> pd.Series:
        return _resolve_timestamps(X, self.date_col, self.raw_date_col, self.hour_col)

    def _available_test_years(self, X: pd.DataFrame) -> list:
        met_year = meteorological_year(self._get_timestamps(X).reset_index(drop=True))
        present = set(pd.unique(met_year))
        return [year for year in self.test_years if year in present]

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        timestamps = self._get_timestamps(X).reset_index(drop=True)
        met_year = meteorological_year(timestamps)
        gap_delta = pd.Timedelta(hours=self.gap)

        for test_year in self._available_test_years(X):
            test_mask = (met_year == test_year).to_numpy()
            test_idx = np.where(test_mask)[0]
            test_start = timestamps.iloc[test_idx].min()
            train_mask = (timestamps < (test_start - gap_delta)).to_numpy()
            if self.max_train_years is not None:
                oldest_train_year = test_year - self.max_train_years
                train_mask &= (met_year >= oldest_train_year).to_numpy()
            train_idx = np.where(train_mask)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            raise ValueError(
                "ExpandingMeteorologicalYearSplit.get_n_splits needs X: the number "
                "of folds depends on which test_years are present in the data."
            )
        return len(self._available_test_years(X))


def expanding_meteorological_year_report(
    X: pd.DataFrame,
    splitter: ExpandingMeteorologicalYearSplit,
    lang=None,
) -> pd.DataFrame:
    """Summarize each ``ExpandingMeteorologicalYearSplit`` fold as a localized report row.

    One row per fold: train/test start and end timestamps, row counts, the
    *real* gap in hours (measured between the last training timestamp and
    the first test timestamp, not inferred from row positions), the set of
    seasons present in the test block, and a boolean confirming training
    never contains a meteorological year at or after the test year.
    """
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "fold": "Fold",
            "train_start": "Início do treino",
            "train_end": "Fim do treino",
            "test_start": "Início do teste",
            "test_end": "Fim do teste",
            "train_rows": "Linhas de treino",
            "test_rows": "Linhas de teste",
            "gap_hours": "Gap real (horas)",
            "seasons": "Estações no teste",
            "has_future": "Treino contém anos futuros",
        }
    )
    timestamps = splitter._get_timestamps(X).reset_index(drop=True)
    seasons_by_row = timestamps.dt.month.map(_MONTH_TO_SEASON)
    met_year = meteorological_year(timestamps)

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        train_ts = timestamps.iloc[train_idx]
        test_ts = timestamps.iloc[test_idx]
        test_year = int(met_year.iloc[test_idx[0]])
        gap_hours = (test_ts.min() - train_ts.max()) / pd.Timedelta(hours=1)
        has_future = bool((met_year.iloc[train_idx] >= test_year).any())
        rows.append(
            {
                labels["fold"]: fold_idx,
                labels["train_start"]: train_ts.min(),
                labels["train_end"]: train_ts.max(),
                labels["test_start"]: test_ts.min(),
                labels["test_end"]: test_ts.max(),
                labels["train_rows"]: len(train_idx),
                labels["test_rows"]: len(test_idx),
                labels["gap_hours"]: gap_hours,
                labels["seasons"]: ", ".join(sorted(set(seasons_by_row.iloc[test_idx]))),
                labels["has_future"]: has_future,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final holdout sealing (by date, not by size)
# ---------------------------------------------------------------------------


@dataclass
class HoldoutSummary:
    """Metadata-only summary of the sealed final holdout — no row data.

    Deliberately carries no ``X``/``y``: any code holding only a
    ``HoldoutSummary`` cannot compute a holdout metric, prediction, or
    feature-importance value, because the rows themselves were never handed
    to it.

    Attributes
    ----------
    start, end, n_rows:
        The sealed window and how many rows fell inside it.
    dev_start, dev_end, n_dev_rows:
        The development window that was actually returned — strictly earlier
        than ``start``.
    n_post_holdout_rows, post_holdout_start, post_holdout_end:
        Rows *later* than the holdout, which exist in the v4 dataset because
        the source runs through 31/12/2024 while the holdout closes on
        30/11/2024. They belong to neither split and are dropped from this
        workflow; they are counted here so the discard is auditable instead
        of silent.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int
    sealed: bool = True
    dev_start: Optional[pd.Timestamp] = None
    dev_end: Optional[pd.Timestamp] = None
    n_dev_rows: int = 0
    n_post_holdout_rows: int = 0
    post_holdout_start: Optional[pd.Timestamp] = None
    post_holdout_end: Optional[pd.Timestamp] = None


def split_dev_holdout(
    df: pd.DataFrame,
    target: str = "Rented Bike Count",
    date_col: str = "Date",
    raw_date_col: str = "DateTime",
    hour_col: str = "Hour",
    holdout_start: str = "2023-12-01",
    holdout_end: str = "2024-11-30",
) -> Tuple[pd.DataFrame, pd.Series, HoldoutSummary]:
    """Seal the final temporal holdout by date, returning only the development split.

    Meant to run on the raw loader output (``DateTime``+``Hour``, before
    ``build_preprocessing_pipeline()``), per "o dataframe bruto deverá ser
    separado em X e y antes do ajuste do pipeline" — but resolves timestamps
    via :func:`_resolve_timestamps`, so it also works on an already-engineered
    frame that has ``date_col`` instead.

    Development-selection code (``ExpandingMeteorologicalYearSplit``,
    ``TemporalRegressionOptimizer``, ...) receives ``X_dev``/``y_dev`` only.
    The holdout rows are sliced locally to compute ``HoldoutSummary`` and
    then discarded — they are never returned, so no notebook variable can
    carry them into a modeling call downstream. ``holdout_end`` is inclusive
    through the end of that calendar day.

    Development is the *strictly earlier* side of the split, never the
    complement of the holdout mask. The v4 source runs to 31/12/2024 while
    the holdout closes on 30/11/2024, so a ``~holdout_mask`` selection would
    quietly return the 744 hours of December/2024 — data recorded *after* the
    period the final model is judged on — as if it were training history.
    Those rows are dropped from this workflow and only counted in
    :class:`HoldoutSummary`.
    """
    dates = _resolve_timestamps(df, date_col, raw_date_col, hour_col)
    start = pd.Timestamp(holdout_start)
    end_inclusive = pd.Timestamp(holdout_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    dev_mask = dates < start
    holdout_mask = (dates >= start) & (dates <= end_inclusive)
    post_holdout_mask = dates > end_inclusive

    dev_dates = dates[dev_mask]
    post_dates = dates[post_holdout_mask]

    summary = HoldoutSummary(
        start=start,
        end=pd.Timestamp(holdout_end),
        n_rows=int(holdout_mask.sum()),
        dev_start=dev_dates.min() if not dev_dates.empty else None,
        dev_end=dev_dates.max() if not dev_dates.empty else None,
        n_dev_rows=int(dev_mask.sum()),
        n_post_holdout_rows=int(post_holdout_mask.sum()),
        post_holdout_start=post_dates.min() if not post_dates.empty else None,
        post_holdout_end=post_dates.max() if not post_dates.empty else None,
    )

    dev_df = df.loc[dev_mask.to_numpy()].copy()
    y_dev = dev_df[target].copy()
    X_dev = dev_df.drop(columns=[target])

    dev_timestamps = dates[dev_mask.to_numpy()]
    if not dev_timestamps.empty:
        assert dev_timestamps.max() < start, (
            f"Development data reaches {dev_timestamps.max()}, which is not strictly "
            f"earlier than the holdout start {start}."
        )
    assert not bool(
        (dev_timestamps >= start).any()
    ), "Holdout rows leaked into the development split."
    assert not bool(
        (dev_timestamps > end_inclusive).any()
    ), "Post-holdout rows leaked into the development split."
    assert len(X_dev) == summary.n_dev_rows, (
        f"Development row count mismatch: {len(X_dev)} returned but "
        f"{summary.n_dev_rows} counted."
    )

    return X_dev, y_dev, summary
