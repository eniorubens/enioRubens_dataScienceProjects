"""Leakage-safe temporal trend decomposition for demand forecasting.

The EDA notebooks may estimate a retrospective moving baseline because their
purpose is descriptive.  A forecasting pipeline cannot do that: the demand in
the validation or holdout period does not exist when the baseline is fitted.

``RobustTrendResidualRegressor`` therefore learns an extrapolatable log-linear
trend from the training target only.  Monthly log-demand averages are used so
that every month has equal influence, and a Huber fit followed by a MAD-based
refit prevents an exceptional low-demand regime from defining the long-term
growth slope.  The wrapped estimator receives the remaining log residual and
continues to own all weather, calendar, encoding and feature-selection choices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
from sklearn.utils.validation import check_is_fitted

from src.cv import _resolve_timestamps


class RobustTrendResidualRegressor(RegressorMixin, BaseEstimator):
    """Model log-demand as a robust temporal trend plus a learned residual.

    Parameters
    ----------
    estimator:
        Unfitted estimator or pipeline used to model the log residual.
    epsilon:
        Huber contamination threshold used by the monthly trend regression.
    alpha:
        L2 regularisation applied to the trend slope.
    anomaly_z:
        Negative monthly residuals below ``anomaly_z`` robust standard
        deviations are excluded from a second trend fit.  They remain present
        when the wrapped residual estimator is trained.
    min_months:
        Minimum number of monthly aggregates needed for the robust trend.
        Shorter samples fall back to a constant log baseline.
    extrapolation_damping:
        Fraction of the fitted slope retained beyond the last training
        timestamp. ``0`` carries the last trend level forward; ``1`` performs
        full log-linear extrapolation.
    """

    def __init__(
        self,
        estimator: Any,
        epsilon: float = 1.35,
        alpha: float = 1e-4,
        anomaly_z: float = 2.5,
        min_months: int = 6,
        extrapolation_damping: float = 0.5,
    ) -> None:
        self.estimator = estimator
        self.epsilon = epsilon
        self.alpha = alpha
        self.anomaly_z = anomaly_z
        self.min_months = min_months
        self.extrapolation_damping = extrapolation_damping

    @staticmethod
    def _as_series(y, index) -> pd.Series:
        values = np.asarray(y, dtype=float).reshape(-1)
        if len(values) != len(index):
            raise ValueError("X and y must contain the same number of rows.")
        if np.any(values < 0):
            raise ValueError("RobustTrendResidualRegressor requires a non-negative target.")
        return pd.Series(values, index=index, name="target")

    def _trend_design_from_timestamps(self, timestamps: pd.Series) -> np.ndarray:
        origin = pd.Timestamp(self.time_origin_)
        hours = (timestamps - origin).dt.total_seconds().to_numpy() / 3600.0
        years = hours / (24.0 * 365.2425)
        month_angle = 2.0 * np.pi * (timestamps.dt.month.to_numpy() - 1.0) / 12.0
        return np.column_stack([years, np.sin(month_angle), np.cos(month_angle)])

    def _trend_design(self, X: pd.DataFrame) -> np.ndarray:
        timestamps = _resolve_timestamps(X).reset_index(drop=True)
        return self._trend_design_from_timestamps(timestamps)

    def _monthly_trend_frame(self, X: pd.DataFrame, y) -> pd.DataFrame:
        timestamps = _resolve_timestamps(X).reset_index(drop=True)
        target = self._as_series(y, timestamps.index)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "log_target": np.log1p(target.to_numpy()),
            }
        )
        frame["month"] = frame["timestamp"].dt.to_period("M")
        monthly = (
            frame.groupby("month", observed=True)
            .agg(timestamp=("timestamp", "mean"), log_target=("log_target", "mean"))
            .reset_index(drop=True)
        )
        return monthly

    def fit_trend(self, X: pd.DataFrame, y):
        """Fit only the train-derived trend and return ``self``.

        This public step is also used by the temporal early-stopping carve so
        that its validation tail receives a baseline learned exclusively from
        the earlier fit portion.
        """

        if not 0.0 <= self.extrapolation_damping <= 1.0:
            raise ValueError("extrapolation_damping must be between 0 and 1.")
        timestamps = _resolve_timestamps(X)
        self.time_origin_ = pd.Timestamp(timestamps.min())
        self.last_training_timestamp_ = pd.Timestamp(timestamps.max())
        monthly = self._monthly_trend_frame(X, y)
        self.n_months_total_ = int(len(monthly))

        if len(monthly) < self.min_months:
            self.trend_model_ = None
            self.constant_log_baseline_ = float(monthly["log_target"].median())
            self.n_months_excluded_ = 0
            return self

        month_x = self._trend_design_from_timestamps(monthly["timestamp"])
        month_y = monthly["log_target"].to_numpy()

        # A trailing abnormal year can occupy roughly 20-25% of the available
        # months and rotate a plain least-squares or Huber slope downwards.
        # Theil-Sen supplies a high-breakdown initial line; Huber is retained
        # for the efficient refit after low outliers have been identified.
        initial = TheilSenRegressor(random_state=42)
        initial.fit(month_x, month_y)
        residual = month_y - initial.predict(month_x)
        residual_median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - residual_median)))
        robust_scale = 1.4826 * mad

        inlier = np.ones(len(monthly), dtype=bool)
        if robust_scale > np.finfo(float).eps:
            lower_limit = residual_median - self.anomaly_z * robust_scale
            inlier = residual >= lower_limit
            if int(inlier.sum()) < self.min_months:
                inlier = np.ones(len(monthly), dtype=bool)

        self.trend_model_ = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha)
        self.trend_model_.fit(month_x[inlier], month_y[inlier])
        self.constant_log_baseline_ = None
        self.n_months_excluded_ = int((~inlier).sum())
        self.trend_slope_ = float(self.trend_model_.coef_[0])
        return self

    def predict_log_baseline(self, X: pd.DataFrame) -> np.ndarray:
        """Return the train-fitted log baseline for arbitrary timestamps."""

        check_is_fitted(self, "time_origin_")
        if self.trend_model_ is None:
            return np.full(len(X), self.constant_log_baseline_, dtype=float)
        design = self._trend_design(X)
        prediction = np.asarray(self.trend_model_.predict(design), dtype=float)
        timestamps = _resolve_timestamps(X).reset_index(drop=True)
        future_years = np.maximum(
            (timestamps - self.last_training_timestamp_).dt.total_seconds().to_numpy()
            / (3600.0 * 24.0 * 365.2425),
            0.0,
        )
        prediction -= (
            (1.0 - float(self.extrapolation_damping))
            * float(self.trend_model_.coef_[0])
            * future_years
        )
        return prediction

    def transform_target(self, X: pd.DataFrame, y) -> np.ndarray:
        """Convert demand into the log residual around the fitted baseline."""

        target = self._as_series(y, X.index).to_numpy()
        return np.log1p(target) - self.predict_log_baseline(X)

    def fit(self, X: pd.DataFrame, y):
        """Fit the robust trend, then the dynamic estimator on log residuals."""

        self.fit_trend(X, y)
        residual = self.transform_target(X, y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Reconstruct non-negative demand predictions on the original scale."""

        check_is_fitted(self, "estimator_")
        residual = np.asarray(self.estimator_.predict(X), dtype=float)
        prediction = np.expm1(self.predict_log_baseline(X) + residual)
        return np.clip(prediction, 0.0, None)
