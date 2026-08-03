"""Focused tests for the confirmatory CatBoost stage."""

from __future__ import annotations

import pandas as pd
import pytest

import src.catboost_refinement as refinement
import src.model_selection_workflow as workflow
from src.catboost_refinement import (
    CATBOOST_FOCUSED_SEED_TRIALS,
    build_catboost_v4_raw_pipeline,
    evaluate_catboost_v4_ablation,
)
from src.model_selection_workflow import ModelSelectionConfig, prepare_development_data
from src.temporal_optimizer import summarize_cv_fold_metrics


def _development(monkeypatch, raw_v4_df):
    monkeypatch.setattr(workflow, "read_data", lambda: raw_v4_df)
    config = ModelSelectionConfig(
        estimators=("CatBoostRegressor",),
        fold_weights=(1.0, 1.0, 2.0, 3.0),
    )
    return prepare_development_data(config)


def _fake_current_cv_folds(best_iteration=None):
    return [
        {
            "fold": index + 1,
            "test_year": 2019 + index,
            "fold_role": "stress" if index == 1 else "selection",
            "n_train": 1000 + index,
            "n_train_excluded": 0,
            "n_test": 100,
            "n_selection_test": 0 if index == 1 else 100,
            "best_iteration": best_iteration,
            "iteration_ceiling": 2000 if best_iteration is not None else None,
            "best_iteration_cap_hit": False,
            "mae": float(100 + 10 * index),
            "rmse": float(150 + 10 * index),
            "r2": float(-0.2 + 0.2 * index),
            "wape": float(0.1 + 0.01 * index),
            "mean_bias": float((-1) ** index * (10 + index)),
            "selection_mae": (float("nan") if index == 1 else float(100 + 10 * index)),
            "selection_rmse": (float("nan") if index == 1 else float(150 + 10 * index)),
            "selection_r2": (float("nan") if index == 1 else float(-0.2 + 0.2 * index)),
            "selection_wape": (float("nan") if index == 1 else float(0.1 + 0.01 * index)),
            "selection_mean_bias": (
                float("nan") if index == 1 else float((-1) ** index * (10 + index))
            ),
            "train_r2": 0.9,
        }
        for index in range(5)
    ]


def test_old_configuration_rebuilds_with_raw_target(monkeypatch, raw_v4_df):
    _, spec = build_catboost_v4_raw_pipeline(_development(monkeypatch, raw_v4_df))

    assert spec.estimator == "CatBoostRegressor"
    assert spec.modeler_name == "Periodic_Spline"
    assert spec.encoder == "OrdinalEncoder"
    assert spec.target_transform == "none"


def test_both_seed_hypotheses_are_auditable():
    raw, robust = CATBOOST_FOCUSED_SEED_TRIALS

    assert raw["target_strategy"] == "direct"
    assert raw["target_transform"] == "none"
    assert robust["target_strategy"] == "robust_trend_residual"
    assert robust["trend_extrapolation_damping"] == 0.0


def test_ablation_reuses_current_folds_and_reports_robust_metrics(
    monkeypatch,
    raw_v4_df,
):
    calls = []

    def fake_fold_results(*args, early_stopping=False, **kwargs):
        calls.append(early_stopping)
        return _fake_current_cv_folds(350 if early_stopping else None)

    monkeypatch.setattr(refinement, "temporal_cv_fold_results", fake_fold_results)
    results = evaluate_catboost_v4_ablation(_development(monkeypatch, raw_v4_df))

    assert calls == [False, True]
    assert list(results.summary["variant"]) == [
        "v4_raw_fixed_283",
        "v4_raw_temporal_early_stopping",
    ]
    assert set(("cv_r2_median", "cv_r2_weighted", "cv_mean_abs_fold_bias")).issubset(
        results.summary.columns
    )
    assert len(results.fold_metrics) == 10
    assert results.summary.loc[0, "final_n_estimators"] == 283
    assert results.summary.loc[1, "final_n_estimators"] == 350


def test_fold_summary_does_not_cancel_opposite_biases():
    frame = pd.DataFrame(_fake_current_cv_folds())
    summary = summarize_cv_fold_metrics(frame, [1, 1, 1, 2, 3])

    assert summary["cv_r2_median"] == pytest.approx(0.2)
    assert summary["cv_r2_weighted"] == pytest.approx(0.325)
    assert summary["cv_mean_abs_fold_bias"] == pytest.approx(12.0)
    assert abs(summary["cv_mean_bias"]) < summary["cv_mean_abs_fold_bias"]
