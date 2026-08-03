"""Focused CatBoost refinement and v4-configuration ablation.

The functions in this module deliberately receive development-only data.  They
cannot access the sealed holdout and reuse the same fold evaluator and recency
weights as the main temporal optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone

from src.modeling_pipeline import (
    BOOSTING_CEILING,
    SEARCH_PROFILE_BROAD,
    TARGET_STRATEGY_DIRECT,
    build_dynamic_pipeline,
)
from src.temporal_optimizer import (
    aggregate_iteration_budget,
    selection_fold_metrics,
    set_iteration_budget,
    summarize_cv_fold_metrics,
    temporal_cv_fold_results,
)

if TYPE_CHECKING:
    from src.model_selection_workflow import DevelopmentData


# Exact winning CatBoost configuration retained from the pre-v5 experiment.
# It is replayed under the v5 fold windows; no score from the old protocol is
# carried forward.
CATBOOST_V4_RAW_PARAMS: Dict[str, Any] = {
    "modeler_name": "Periodic_Spline",
    "encoder": "OrdinalEncoder",
    "standardizer": "StandardScaler",
    "iterations": 283,
    "depth": 10,
    "learning_rate": 0.10909945890629895,
    "random_strength": 52,
    "bagging_temperature": 0.7637155891395317,
    "l2_leaf_reg": 0.17643779852577682,
    "border_count": 89,
    "selector": "NoSelector",
    "target_transform": "none",
}


# These configurations are evaluated first and count toward the 100-trial
# budget. The first adapts the old raw-target winner to the refined search
# contract; the second anchors the best robust-trend configuration observed in
# the preceding v5 run.
CATBOOST_FOCUSED_SEED_TRIALS: Tuple[Dict[str, Any], ...] = (
    {
        "modeler_name": "Periodic_Spline",
        "encoder": "OrdinalEncoder",
        "loss_function": "RMSE",
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
)


@dataclass(frozen=True)
class CatBoostAblationResults:
    """Development-only comparison of the old configuration under current CV."""

    summary: pd.DataFrame
    fold_metrics: pd.DataFrame


def build_catboost_v4_raw_pipeline(
    development: "DevelopmentData",
):
    """Rebuild the old raw-target pipeline without fitting it."""
    return build_dynamic_pipeline(
        optuna.trial.FixedTrial(CATBOOST_V4_RAW_PARAMS),
        "CatBoostRegressor",
        numeric_features=development.config.numeric_features,
        categorical_features=development.config.categorical_features,
        search_profile=SEARCH_PROFILE_BROAD,
        target_strategy=TARGET_STRATEGY_DIRECT,
    )


def _resolved_fold_weights(development: "DevelopmentData", n_folds: int) -> np.ndarray:
    configured = development.config.fold_weights
    weights = np.ones(n_folds, dtype=float) if configured is None else np.asarray(configured)
    if weights.shape != (n_folds,):
        raise ValueError(
            f"The CatBoost ablation requires {n_folds} fold weights; got {len(weights)}."
        )
    return weights


def evaluate_catboost_v4_ablation(
    development: "DevelopmentData",
) -> CatBoostAblationResults:
    """Evaluate the old CatBoost configuration twice under the current CV.

    The fixed 283-iteration variant reproduces the old fitted configuration.
    The temporal-early-stopping variant keeps all other choices fixed, raises
    the ceiling, and lets each current training fold determine its tree budget.
    """
    base_pipeline, _ = build_catboost_v4_raw_pipeline(development)
    variants = (
        ("v4_raw_fixed_283", clone(base_pipeline), False),
        ("v4_raw_temporal_early_stopping", clone(base_pipeline), True),
    )

    summaries = []
    fold_frames = []
    for variant, pipeline, uses_early_stopping in variants:
        if uses_early_stopping:
            set_iteration_budget(pipeline, BOOSTING_CEILING)

        folds = temporal_cv_fold_results(
            pipeline,
            development.X_dev,
            development.y_dev,
            development.splitter,
            early_stopping=uses_early_stopping,
            train_eligible_mask=development.train_eligible_mask,
            score_eligible_mask=development.score_eligible_mask,
            selection_test_years=development.config.selection_test_years,
        )
        frame = pd.DataFrame(folds)
        frame.insert(0, "variant", variant)
        fold_frames.append(frame)

        selected_frame = selection_fold_metrics(frame)
        weights = _resolved_fold_weights(development, len(selected_frame))
        summary = summarize_cv_fold_metrics(selected_frame, weights)
        best_iterations = selected_frame["best_iteration"].tolist()
        summary.update(
            {
                "variant": variant,
                "early_stopping": uses_early_stopping,
                "configured_iterations": (BOOSTING_CEILING if uses_early_stopping else 283),
                "final_n_estimators": (
                    aggregate_iteration_budget(best_iterations) if uses_early_stopping else 283
                ),
            }
        )
        summaries.append(summary)

    return CatBoostAblationResults(
        summary=pd.DataFrame(summaries),
        fold_metrics=pd.concat(fold_frames, ignore_index=True),
    )
