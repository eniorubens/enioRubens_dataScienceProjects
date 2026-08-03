"""Normal-operations regime definition for notebook 04 model selection.

The source dataset is never mutated or truncated.  A calendar-defined
operational-disruption interval is excluded from model fitting and from the
primary selection metric, while its meteorological-year fold remains available
as a stress diagnostic.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

REGIME_POLICY_NORMAL_OPERATIONS = "normal_operations"
DEFAULT_EXCLUSION_START = "2020-01-01 00:00:00"
DEFAULT_EXCLUSION_END = "2020-12-31 23:59:59"
DEFAULT_STRESS_TEST_YEARS: Tuple[int, ...] = (2020,)
DEFAULT_SELECTION_TEST_YEARS: Tuple[int, ...] = (2019, 2021, 2022, 2023)
DEFAULT_SELECTION_FOLD_WEIGHTS: Tuple[float, ...] = (1.0, 1.0, 2.0, 3.0)


def observation_timestamps(X: pd.DataFrame) -> pd.Series:
    """Return one hourly timestamp per row without changing ``X``."""
    if "DateTime" not in X.columns:
        raise KeyError("The regime policy requires the 'DateTime' column.")
    timestamps = pd.to_datetime(X["DateTime"], errors="raise")
    if "Hour" in X.columns and bool((timestamps.dt.hour == 0).all()):
        timestamps = timestamps + pd.to_timedelta(X["Hour"], unit="h")
    return timestamps


def normal_operations_mask(
    X: pd.DataFrame,
    exclusion_start: str = DEFAULT_EXCLUSION_START,
    exclusion_end: str = DEFAULT_EXCLUSION_END,
) -> np.ndarray:
    """Rows eligible for normal-regime fitting or primary scoring."""
    timestamps = observation_timestamps(X)
    start = pd.Timestamp(exclusion_start)
    end = pd.Timestamp(exclusion_end)
    if end < start:
        raise ValueError("exclusion_end must not precede exclusion_start.")
    return (~timestamps.between(start, end, inclusive="both")).to_numpy(dtype=bool)


def regime_fingerprint(
    X: pd.DataFrame,
    *,
    policy: str,
    exclusion_start: str,
    exclusion_end: str,
    selection_test_years: Sequence[int],
    stress_test_years: Sequence[int],
) -> str:
    """Hash the declared regime and the exact rows it admits."""
    mask = normal_operations_mask(X, exclusion_start, exclusion_end)
    payload = {
        "policy": policy,
        "exclusion_start": str(pd.Timestamp(exclusion_start)),
        "exclusion_end": str(pd.Timestamp(exclusion_end)),
        "selection_test_years": list(selection_test_years),
        "stress_test_years": list(stress_test_years),
        "n_rows": int(len(mask)),
        "n_eligible": int(mask.sum()),
    }
    hasher = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    hasher.update(mask.tobytes())
    return hasher.hexdigest()[:16]


# Best configurations from the previous all-observed protocol are hypotheses,
# not inherited winners. They are evaluated first and count toward each new
# study's budget.
NORMAL_OPERATIONS_SEED_TRIALS: Mapping[str, Tuple[Dict[str, Any], ...]] = {
    "Ridge": (
        {
            "modeler_name": "Time_steps_as_categories",
            "encoder": "MEstimateEncoder",
            "alpha": 6.053701076380985,
            "selector": "SequentialFeatureSelector",
            "sfs_n_features": 12,
            "target_strategy": "robust_trend_residual",
            "trend_extrapolation_damping": 0.0,
        },
    ),
    "HistGradientBoostingRegressor": (
        {
            "modeler_name": "Periodic_Spline",
            "encoder": "MeanEncoder",
            "learning_rate": 0.08332902192727393,
            "max_iter": 372,
            "max_leaf_nodes": 52,
            "max_depth": 5,
            "min_samples_leaf": 47,
            "l2_regularization": 0.0011031298085724627,
            "loss_function": "squared_error",
            "selector": "NoSelector",
            "target_strategy": "direct",
        },
    ),
    "XGBRegressor": (
        {
            "modeler_name": "linear_modeling",
            "encoder": "CountFrequencyEncoder",
            "loss_function": "reg:squarederror",
            "boosting_budget_strategy": "temporal_early_stopping",
            "max_depth": 5,
            "learning_rate": 0.025102291222963046,
            "subsample": 0.8155028085257771,
            "colsample_bytree": 0.5892053931536445,
            "gamma": 2.790870764911474,
            "min_child_weight": 18.974064559031465,
            "reg_alpha": 0.2735519562189285,
            "reg_lambda": 1.9252503267345995,
            "selector": "NoSelector",
            "target_strategy": "direct",
        },
    ),
    "LGBMRegressor": (
        {
            "modeler_name": "linear_modeling",
            "encoder": "CountFrequencyEncoder",
            "loss_function": "regression",
            "boosting_budget_strategy": "temporal_early_stopping",
            "num_leaves": 176,
            "learning_rate": 0.03151534038014746,
            "min_child_samples": 46,
            "subsample": 0.6478117388261758,
            "colsample_bytree": 0.5091350242254455,
            "reg_alpha": 0.014021704269111986,
            "reg_lambda": 0.004213552008850135,
            "max_depth": 4,
            "selector": "NoSelector",
            "target_strategy": "direct",
        },
    ),
    "CatBoostRegressor": (
        {
            "modeler_name": "Periodic_Spline",
            "encoder": "OrdinalEncoder",
            "loss_function": "RMSE",
            "boosting_budget_strategy": "fixed_iterations",
            "fixed_iterations": 283,
            "depth": 10,
            "learning_rate": 0.10909945890629895,
            "random_strength": 52,
            "bagging_temperature": 0.7637155891395317,
            "l2_leaf_reg": 0.17643779852577682,
            "border_count": 89,
            "selector": "NoSelector",
            "target_strategy": "direct",
            "target_transform": "none",
        },
        {
            "modeler_name": "Periodic_Spline",
            "encoder": "MeanEncoder",
            "loss_function": "MAE",
            "boosting_budget_strategy": "temporal_early_stopping",
            "depth": 7,
            "learning_rate": 0.24016668051866366,
            "random_strength": 84,
            "bagging_temperature": 0.8283541122091502,
            "l2_leaf_reg": 0.9084331558692036,
            "border_count": 107,
            "selector": "NoSelector",
            "target_strategy": "robust_trend_residual",
            "trend_extrapolation_damping": 0.0,
        },
    ),
}
