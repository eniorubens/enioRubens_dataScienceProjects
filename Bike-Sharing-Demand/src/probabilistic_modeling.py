"""Probabilistic adapters for residual-demand experiments.

The confirmed v4 champion models demand as a robust log trend plus a learned
residual. CatBoost's ``RMSEWithUncertainty`` predicts a residual location and a
residual variance, so this module keeps the existing one-dimensional
``predict`` contract while exposing an explicit distribution method for
notebook 06.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.trend import RobustTrendResidualRegressor

DEFAULT_INTERVAL_COVERAGES = (0.5, 0.8, 0.9, 0.95)
_VARIANCE_FLOOR = 1e-9
_NORMAL = NormalDist()


def _z_for_coverage(coverage: float) -> float:
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("Interval coverage must be between 0 and 1.")
    return float(_NORMAL.inv_cdf((1.0 + float(coverage)) / 2.0))


def lognormal_demand_distribution(
    log_location: Sequence[float],
    log_variance: Sequence[float],
    coverages: Sequence[float] = DEFAULT_INTERVAL_COVERAGES,
) -> pd.DataFrame:
    """Convert log-demand Normal parameters into demand-scale summaries."""
    location = np.asarray(log_location, dtype=float).reshape(-1)
    variance = np.asarray(log_variance, dtype=float).reshape(-1)
    if location.shape != variance.shape:
        raise ValueError("log_location and log_variance must have identical shape.")
    if not np.all(np.isfinite(location)):
        raise ValueError("log_location contains non-finite values.")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("log_variance must contain finite non-negative values.")

    variance = np.maximum(variance, _VARIANCE_FLOOR)
    sigma = np.sqrt(variance)
    frame = pd.DataFrame(
        {
            "log_location": location,
            "log_variance": variance,
            "log_sigma": sigma,
            "demand_median": np.clip(np.expm1(location), 0.0, None),
            "demand_mean": np.clip(np.expm1(location + variance / 2.0), 0.0, None),
            "demand_std": np.sqrt((np.exp(variance) - 1.0) * np.exp(2.0 * location + variance)),
        }
    )
    for coverage in coverages:
        z_value = _z_for_coverage(float(coverage))
        suffix = int(round(float(coverage) * 100))
        frame[f"lower_{suffix}"] = np.clip(np.expm1(location - z_value * sigma), 0.0, None)
        frame[f"upper_{suffix}"] = np.clip(np.expm1(location + z_value * sigma), 0.0, None)
    return frame


class CatBoostResidualUncertaintyRegressor(BaseEstimator, RegressorMixin):
    """CatBoost uncertainty adapter with a one-dimensional ``predict`` method."""

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        random_strength: int = 1,
        bagging_temperature: float = 0.0,
        l2_leaf_reg: float = 3.0,
        border_count: int = 254,
        loss_function: str = "RMSEWithUncertainty",
        eval_metric: str = "RMSEWithUncertainty",
        allow_writing_files: bool = False,
        verbose: bool = False,
        task_type: str = "CPU",
        random_state: int = 42,
        variance_floor: float = _VARIANCE_FLOOR,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.allow_writing_files = allow_writing_files
        self.verbose = verbose
        self.task_type = task_type
        self.random_state = random_state
        self.variance_floor = variance_floor

    def _catboost_params(self) -> Dict[str, Any]:
        return {
            "iterations": int(self.iterations),
            "depth": int(self.depth),
            "learning_rate": float(self.learning_rate),
            "random_strength": int(self.random_strength),
            "bagging_temperature": float(self.bagging_temperature),
            "l2_leaf_reg": float(self.l2_leaf_reg),
            "border_count": int(self.border_count),
            "loss_function": "RMSEWithUncertainty",
            "eval_metric": "RMSEWithUncertainty",
            "allow_writing_files": bool(self.allow_writing_files),
            "verbose": bool(self.verbose),
            "task_type": self.task_type,
            "random_state": int(self.random_state),
        }

    def fit(self, X, y):
        target = np.asarray(y, dtype=float).reshape(-1)
        self.model_ = CatBoostRegressor(**self._catboost_params())
        self.model_.fit(X, target, verbose=bool(self.verbose))
        return self

    def _raw_prediction(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        raw = np.asarray(self.model_.predict(X), dtype=float)
        if raw.ndim == 1:
            raw = np.column_stack([raw, np.full(raw.shape[0], np.log(self.variance_floor))])
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError(
                "CatBoost RMSEWithUncertainty must return two columns: "
                "location and log-variance."
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("CatBoost uncertainty prediction contains non-finite values.")
        return raw

    def predict(self, X) -> np.ndarray:
        return self._raw_prediction(X)[:, 0]

    def predict_distribution(self, X) -> pd.DataFrame:
        raw = self._raw_prediction(X)
        variance = np.maximum(np.exp(raw[:, 1]), float(self.variance_floor))
        return pd.DataFrame(
            {
                "log_residual_location": raw[:, 0],
                "log_residual_variance": variance,
                "log_residual_sigma": np.sqrt(variance),
            }
        )


class RobustTrendProbabilisticRegressor(RobustTrendResidualRegressor):
    """Robust trend wrapper with explicit lognormal demand distributions."""

    def _fitted_regressor_step(self):
        check_is_fitted(self, "estimator_")
        core = self.estimator_
        try:
            return core.named_steps["regressor"]
        except (AttributeError, KeyError) as exc:
            raise ValueError("The wrapped estimator must expose a 'regressor' step.") from exc

    def _transform_features_for_regressor(self, X: pd.DataFrame):
        core = self.estimator_
        transformer = Pipeline(core.steps[:-1])
        return transformer.transform(X)

    def predict_distribution(
        self,
        X: pd.DataFrame,
        coverages: Sequence[float] = DEFAULT_INTERVAL_COVERAGES,
    ) -> pd.DataFrame:
        regressor_step = self._fitted_regressor_step()
        residual_regressor = getattr(regressor_step, "regressor_", None)
        if residual_regressor is None or not hasattr(residual_regressor, "predict_distribution"):
            raise ValueError("The fitted residual regressor does not expose predict_distribution.")

        transformed = self._transform_features_for_regressor(X)
        residual = residual_regressor.predict_distribution(transformed).reset_index(drop=True)
        baseline = self.predict_log_baseline(X)
        log_location = baseline + residual["log_residual_location"].to_numpy(dtype=float)
        log_variance = residual["log_residual_variance"].to_numpy(dtype=float)
        demand = lognormal_demand_distribution(log_location, log_variance, coverages=coverages)
        return pd.concat([residual, demand], axis=1)


def as_probabilistic_trend_regressor(pipeline) -> RobustTrendProbabilisticRegressor:
    """Convert an unfitted robust-trend pipeline into the probabilistic subclass."""
    if not isinstance(pipeline, RobustTrendResidualRegressor):
        raise TypeError("Probabilistic experiments require robust_trend_residual target strategy.")
    return RobustTrendProbabilisticRegressor(
        estimator=clone(pipeline.estimator),
        epsilon=pipeline.epsilon,
        alpha=pipeline.alpha,
        anomaly_z=pipeline.anomaly_z,
        min_months=pipeline.min_months,
        extrapolation_damping=pipeline.extrapolation_damping,
    )
