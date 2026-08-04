"""Read-only replay contract for notebook 04 model-selection reports."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.model_selection_workflow import (
    EstimatorOutcome,
    ModelSelectionConfig,
    ModelSelectionResults,
    StudyResultSummary,
    load_model_selection_results,
    save_model_selection_results,
)
from src.modeling_pipeline import PipelineSpec
from src.temporal_optimizer import FoldEvaluation


def _outcome(estimator, run_id, best_value, baseline=False):
    spec = PipelineSpec(
        estimator=estimator,
        family="baseline" if baseline else "linear",
        modeler_name="linear_modeling",
        encoder="OrdinalEncoder",
        scaler="StandardScaler",
        selector="NoSelector",
        target_transform="direct",
        n_features_selected=3,
    )
    evaluation = FoldEvaluation(
        best_params={"alpha": 1.0},
        spec=spec,
        fold_metrics=pd.DataFrame({"fold": [1], "selection_mae": [best_value], "r2": [0.5]}),
        seasonal_metrics=pd.DataFrame(
            {"fold": [1], "season": ["Winter"], "n": [10], "mae": [best_value]}
        ),
        extreme_metrics=pd.DataFrame(
            {"fold": [1], "band": ["cold"], "n": [2], "mae": [best_value]}
        ),
        fitted_pipeline=object(),
        trials_completed=2,
        cv_metrics={"cv_mae_mean": best_value, "cv_mae_weighted": best_value},
    )
    return EstimatorOutcome(
        estimator=estimator,
        study=StudyResultSummary(best_value, 2),
        evaluation=evaluation,
        run_id=run_id,
        is_baseline=baseline,
        trials_planned=2,
        termination_reason="trial_limit",
        cv_fingerprint="cv-1",
    )


@pytest.fixture
def snapshot_case(tmp_path):
    config = ModelSelectionConfig(
        run_mode="smoke",
        estimators=("Ridge",),
        candidates_root=tmp_path,
    )
    development = SimpleNamespace(
        fingerprint="data-1",
        regime_fingerprint="regime-1",
    )
    outcomes = [
        _outcome("DummyRegressor", "baseline-run", 10.0, baseline=True),
        _outcome("Ridge", "champion-run", 5.0),
    ]
    selection = {
        "run_mode": "smoke",
        "dataset_fingerprint": "data-1",
        "regime_policy": config.regime_policy,
        "regime_fingerprint": "regime-1",
        "champion": {
            "run_id": "champion-run",
            "estimator": "Ridge",
            "model_uri": "runs:/champion-run/model",
        },
        "challengers": [],
    }
    results = ModelSelectionResults(
        config=config,
        development=development,
        outcomes=outcomes,
        selection=selection,
        manifest_path=tmp_path / "candidates_v4_smoke" / "candidates_manifest_smoke.json",
        experiment_name="test-experiment",
        experiment_id="1",
        tracker=object(),
    )
    return config, development, results


def test_snapshot_round_trip_reconstructs_reports_without_model(snapshot_case):
    config, development, original = snapshot_case
    path = save_model_selection_results(original)

    replay = load_model_selection_results(config, development, path)

    assert replay.best_outcome.estimator == "Ridge"
    assert replay.best_outcome.cv_mae_selection == pytest.approx(5.0)
    assert replay.best_outcome.evaluation.fitted_pipeline is None
    assert len(replay.best_outcome.study.trials) == 2
    pd.testing.assert_frame_equal(
        replay.best_outcome.evaluation.fold_metrics,
        original.best_outcome.evaluation.fold_metrics,
        check_dtype=False,
    )


def test_snapshot_fails_closed_on_dataset_drift(snapshot_case):
    config, development, original = snapshot_case
    path = save_model_selection_results(original)
    drifted = SimpleNamespace(
        fingerprint="different-data",
        regime_fingerprint=development.regime_fingerprint,
    )

    with pytest.raises(ValueError, match="dataset_fingerprint mismatch"):
        load_model_selection_results(config, drifted, path)
