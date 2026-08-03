"""Tests for src/model_selection_workflow.py — the layer notebook 04 calls.

The end-to-end test runs the real workflow (baseline plus one candidate, two
trials each) against the synthetic v4 frame, an isolated MLflow tracking URI
and temporary study/candidate directories. It never touches the project's
``mlruns/``, ``optuna_studies/`` or ``dataset/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import src.model_selection_workflow as wf
from src.model_selection_workflow import (
    BASELINE_ESTIMATOR,
    ModelSelectionConfig,
    prepare_development_data,
    run_model_selection,
    sample_dynamic_pipeline,
)
from src.temporal_optimizer import MAX_SMOKE_TRIALS


# ---------------------------------------------------------------------------
# Declarative configuration
# ---------------------------------------------------------------------------


class TestModelSelectionConfig:
    def test_run_mode_is_validated(self):
        with pytest.raises(ValueError, match="run_mode"):
            ModelSelectionConfig(run_mode="quick")

    def test_smoke_caps_the_trial_budget_even_when_more_is_requested(self):
        config = ModelSelectionConfig(run_mode="smoke", trials_per_estimator=500)
        assert config.resolved_trials == MAX_SMOKE_TRIALS
        assert config.is_smoke is True
        assert config.freezes_definitive_candidates is False

    def test_full_mode_keeps_the_requested_budget(self):
        config = ModelSelectionConfig(run_mode="full", trials_per_estimator=120)
        assert config.resolved_trials == 120
        assert config.freezes_definitive_candidates is True

    def test_full_defaults_are_400_trials_and_four_hours(self):
        """Whichever limit arrives first ends the study; both are declared."""
        config = ModelSelectionConfig(run_mode="full")
        assert config.resolved_trials == 400
        assert config.resolved_study_timeout == 14_400.0

    def test_the_trial_timeout_is_configurable(self):
        config = ModelSelectionConfig(run_mode="full", trial_timeout=95.0)
        assert config.resolved_trial_timeout == 95.0
        assert ModelSelectionConfig(run_mode="full").resolved_trial_timeout == 1_800.0

    def test_mode_defaults_differ(self):
        """What separates smoke from full is the trial count and the study
        budget, not the per-trial clock: a smoke trial draws from the same
        search space and may legitimately be an expensive member of it."""
        smoke = ModelSelectionConfig(run_mode="smoke")
        full = ModelSelectionConfig(run_mode="full")
        assert smoke.resolved_trials < full.resolved_trials
        assert smoke.resolved_study_timeout < full.resolved_study_timeout
        assert smoke.resolved_trial_timeout <= full.resolved_trial_timeout

    def test_optional_estimators_are_opt_in(self):
        base = ModelSelectionConfig()
        assert "RandomForestRegressor" not in base.candidate_estimators
        assert "CatBoostRegressor" not in base.candidate_estimators

        extended = ModelSelectionConfig(include_random_forest=True, include_catboost=True)
        assert "RandomForestRegressor" in extended.candidate_estimators
        assert "CatBoostRegressor" in extended.candidate_estimators

    def test_the_baseline_is_never_a_candidate(self):
        config = ModelSelectionConfig(estimators=("DummyRegressor", "Ridge"))
        assert BASELINE_ESTIMATOR not in config.candidate_estimators
        assert config.candidate_estimators == ["Ridge"]

    def test_seeded_trials_must_belong_to_this_run(self):
        with pytest.raises(ValueError, match="outside this run"):
            ModelSelectionConfig(
                estimators=("Ridge",),
                enqueued_trials_by_estimator={"CatBoostRegressor": ({"depth": 8},)},
            )


class TestFullRunSourceGate:
    def test_full_refuses_dirty_source_before_opening_tracking(self, monkeypatch):
        def refuse():
            raise RuntimeError("requires committed model-producing source")

        monkeypatch.setattr(wf, "require_clean_git_source", refuse)
        with pytest.raises(RuntimeError, match="requires committed"):
            run_model_selection(ModelSelectionConfig(run_mode="full"), development=object())


# ---------------------------------------------------------------------------
# Development data and the sealed holdout
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_read_data(monkeypatch, raw_v4_df):
    monkeypatch.setattr(wf, "read_data", lambda: raw_v4_df)
    return raw_v4_df


class TestPrepareDevelopmentData:
    def test_holdout_is_sealed_and_never_materialised(self, patched_read_data):
        development = prepare_development_data(ModelSelectionConfig())
        assert development.holdout.sealed is True
        assert development.holdout.n_rows > 0
        assert not hasattr(development, "X_holdout")
        assert not hasattr(development, "y_holdout")
        timestamps = pd.to_datetime(development.X_dev["DateTime"])
        assert timestamps.max() < pd.Timestamp(development.holdout.start)

    def test_the_holdout_report_states_all_three_windows(self, patched_read_data):
        from src import model_selection_reports as reports

        development = prepare_development_data(ModelSelectionConfig())
        table = reports.holdout_seal_report(development)
        items = list(table[table.columns[0]])
        assert "Período de desenvolvimento" in items
        assert "Período do holdout" in items
        assert "Linhas pós-holdout excluídas" in items

    def test_target_is_removed_from_the_features(self, patched_read_data):
        development = prepare_development_data(ModelSelectionConfig())
        assert "Rented Bike Count" not in development.X_dev.columns
        assert development.y_dev.name == "Rented Bike Count"

    def test_five_expanding_folds_are_available(self, patched_read_data):
        development = prepare_development_data(ModelSelectionConfig())
        assert development.n_folds == 5

    def test_sample_pipeline_is_built_without_fitting(self, patched_read_data):
        development = prepare_development_data(ModelSelectionConfig())
        pipeline, spec = sample_dynamic_pipeline(development, estimator="Ridge")
        assert [name for name, _ in pipeline.steps][-1] == "regressor"
        assert spec.modeler_name == "Periodic_Spline"
        assert spec.n_features_selected is None

    def test_sample_pipeline_resolves_auto_target_strategy(self, patched_read_data):
        config = ModelSelectionConfig(
            search_profile="refined",
            target_strategy="auto",
        )
        development = prepare_development_data(config)

        pipeline, spec = sample_dynamic_pipeline(development, estimator="Ridge")

        assert [name for name, _ in pipeline.steps][-1] == "regressor"
        assert spec.target_transform == "log1p"

    def test_sample_pipeline_accepts_robust_trend_override(self, patched_read_data):
        config = ModelSelectionConfig(target_strategy="auto")
        development = prepare_development_data(config)

        _, spec = sample_dynamic_pipeline(
            development,
            estimator="Ridge",
            params={"target_strategy": "robust_trend_residual"},
        )

        assert spec.target_transform == "robust_trend_residual"

    def test_normal_operations_mask_is_calendar_defined(self, patched_read_data):
        from src.normal_operations import observation_timestamps

        development = prepare_development_data(ModelSelectionConfig())
        timestamps = observation_timestamps(development.X_dev)
        in_2020 = timestamps.between(
            pd.Timestamp("2020-01-01 00:00:00"),
            pd.Timestamp("2020-12-31 23:59:59"),
            inclusive="both",
        ).to_numpy()

        assert in_2020.any()
        assert not development.train_eligible_mask[in_2020].any()
        assert development.train_eligible_mask[~in_2020].all()
        assert development.score_eligible_mask.tolist() == (
            development.train_eligible_mask.tolist()
        )

    def test_fold_roles_and_boundary_rows_are_explicit(self, patched_read_data):
        development = prepare_development_data(ModelSelectionConfig())
        rows = []
        for fold_index, (train_idx, test_idx) in enumerate(
            development.splitter.split(development.X_dev, development.y_dev)
        ):
            test_year = development.config.test_years[fold_index]
            rows.append(
                {
                    "test_year": test_year,
                    "train_excluded": int((~development.train_eligible_mask[train_idx]).sum()),
                    "score_eligible": int(development.score_eligible_mask[test_idx].sum()),
                    "n_test": len(test_idx),
                }
            )

        by_year = {row["test_year"]: row for row in rows}
        assert by_year[2020]["score_eligible"] < by_year[2020]["n_test"]
        assert by_year[2021]["score_eligible"] < by_year[2021]["n_test"]
        assert by_year[2022]["train_excluded"] > 0
        assert development.config.selection_test_years == (2019, 2021, 2022, 2023)
        assert development.config.stress_test_years == (2020,)


# ---------------------------------------------------------------------------
# End-to-end smoke selection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory, request):
    """Run the real workflow once (baseline + Ridge, two trials each)."""
    raw = request.getfixturevalue("raw_v4_df")
    original_read_data = wf.read_data
    wf.read_data = lambda: raw
    try:
        tmp = tmp_path_factory.mktemp("smoke_workflow")
        config = ModelSelectionConfig(
            run_mode="smoke",
            estimators=("Ridge",),
            trial_timeout=180.0,
            study_timeout=900.0,
            tracking_uri=f"file:{(tmp / 'mlruns').as_posix()}",
            studies_dir=tmp / "studies",
            invalid_configs_path=tmp / "invalid.csv",
            candidates_root=tmp / "dataset",
        )
        development = prepare_development_data(config)
        results = run_model_selection(config, development)
        yield config, development, results, tmp
    finally:
        wf.read_data = original_read_data


class TestRunModelSelection:
    def test_baseline_and_candidates_all_produced_runs(self, smoke_run):
        _, _, results, _ = smoke_run
        estimators = [outcome.estimator for outcome in results.outcomes]
        assert estimators == [BASELINE_ESTIMATOR, "Ridge"]
        assert all(outcome.run_id for outcome in results.outcomes)
        assert len(set(outcome.run_id for outcome in results.outcomes)) == 2

    def test_every_outcome_records_its_dynamic_pipeline(self, smoke_run):
        _, _, results, _ = smoke_run
        for outcome in results.outcomes:
            spec = outcome.spec
            assert spec.modeler_name
            assert spec.encoder
            assert spec.selector
            assert spec.target_transform
            assert spec.n_features_selected and spec.n_features_selected > 0

    def test_smoke_runs_at_most_two_trials_per_estimator(self, smoke_run):
        _, _, results, _ = smoke_run
        for outcome in results.outcomes:
            assert len(outcome.study.trials) <= MAX_SMOKE_TRIALS

    def test_champion_comes_from_this_run_and_matches_a_known_run_id(self, smoke_run):
        _, development, results, _ = smoke_run
        assert results.champion["run_id"] in results.outcomes_by_run_id
        assert results.champion["run_mode"] == "smoke"
        assert results.champion["dataset_fingerprint"] == development.fingerprint

    def test_smoke_writes_only_a_provisional_manifest(self, smoke_run):
        config, _, results, tmp = smoke_run
        manifest = json.loads(results.manifest_path.read_text(encoding="utf-8"))
        assert manifest["provisional"] is True
        assert manifest["run_mode"] == "smoke"
        assert manifest["regime_policy"] == "normal_operations"
        assert manifest["regime_fingerprint"]
        assert results.is_provisional is True
        definitive = tmp / "dataset" / "candidates_v4" / "candidates_manifest.json"
        assert not definitive.exists()

    def test_manifest_describes_the_whole_winning_combination(self, smoke_run):
        _, _, results, _ = smoke_run
        manifest = json.loads(results.manifest_path.read_text(encoding="utf-8"))
        champion = manifest["champion"]
        assert champion["modeler_name"]
        assert champion["best_params"]["modeler_name"] == champion["modeler_name"]
        assert champion["model_uri"].startswith("runs:/")
        assert champion["artifact_path"].endswith(".pkl.gz")

    def test_the_frozen_artifact_belongs_to_the_champion_run(self, smoke_run):
        import gzip
        import pickle

        _, _, results, _ = smoke_run
        manifest = json.loads(results.manifest_path.read_text(encoding="utf-8"))
        with gzip.open(manifest["champion"]["artifact_path"], "rb") as handle:
            restored = pickle.load(handle)
        expected = results.best_outcome.evaluation.fitted_pipeline
        restored_regressor = restored.named_steps["regressor"].regressor
        expected_regressor = expected.named_steps["regressor"].regressor
        assert isinstance(restored_regressor, type(expected_regressor))
        assert manifest["champion"]["run_id"] == results.best_outcome.run_id

    def test_no_holdout_metric_reached_mlflow(self, smoke_run):
        import mlflow

        _, _, results, _ = smoke_run
        runs = mlflow.search_runs(experiment_names=[results.experiment_name])
        holdout_columns = [c for c in runs.columns if c.startswith("metrics.") and "holdout" in c]
        assert holdout_columns == []

    def test_re_running_does_not_grow_the_smoke_studies(self, smoke_run):
        config, development, _, _ = smoke_run
        repeated = run_model_selection(config, development)
        for outcome in repeated.outcomes:
            assert len(outcome.study.trials) <= MAX_SMOKE_TRIALS

    def test_the_optuna_metric_equals_the_weighted_reported_folds(self, smoke_run):
        """The search, the diagnostics and the artifact share one protocol."""
        config, _, results, _ = smoke_run
        for outcome in results.outcomes:
            selection_mae = outcome.evaluation.fold_metrics["selection_mae"].dropna().to_numpy()
            fold_mean = float(selection_mae.mean())
            fold_weighted = float(np.average(selection_mae, weights=config.fold_weights))
            assert fold_weighted == pytest.approx(outcome.study.best_value, rel=1e-6)
            assert outcome.evaluation.cv_metrics["cv_mae_mean"] == pytest.approx(fold_mean)

    def test_every_run_records_how_its_study_ended(self, smoke_run):
        _, _, results, _ = smoke_run
        for outcome in results.outcomes:
            assert outcome.termination_reason in ("trial_limit", "study_timeout")
            assert outcome.trials_planned > 0

    def test_the_frozen_pipeline_is_stamped_with_its_own_run(self, smoke_run):
        from src.tracking import pipeline_provenance

        _, _, results, _ = smoke_run
        for outcome in results.outcomes:
            stamp = pipeline_provenance(outcome.evaluation.fitted_pipeline)
            assert stamp["source_run_id"] == outcome.run_id


# ---------------------------------------------------------------------------
# Reports render from the results object
# ---------------------------------------------------------------------------


class TestReports:
    def test_comparison_report_carries_the_pipeline_columns(self, smoke_run):
        from src import model_selection_reports as reports

        _, _, results, _ = smoke_run
        table = reports.comparison_report(results)
        assert len(table) == len(results.outcomes)
        assert "Representação" in table.columns
        assert "MAE médio normal (CV)" in table.columns
        assert "R² mediano normal (CV)" in table.columns
        assert "R² ponderado normal (CV)" in table.columns
        assert "Média do |viés| por fold" in table.columns

    def test_search_space_report_shows_a_per_family_menu(self, smoke_run):
        from src import model_selection_reports as reports

        config, _, _, _ = smoke_run
        table = reports.search_space_report(config)
        spaces = set(table["Espaço de representação (modeler_name)"])
        assert len(spaces) > 1  # baseline and linear cannot share a menu

    def test_diagnostics_and_handoff_reports_are_non_empty(self, smoke_run):
        from src import model_selection_reports as reports

        _, _, results, _ = smoke_run
        assert not reports.fold_metrics_report(results).empty
        assert not reports.condition_metrics_report(results).empty
        assert not reports.selection_report(results).empty
        assert not reports.handoff_report(results).empty
        assert not reports.fold_audit_report(results.development).empty

    def test_comparison_chart_is_returned_closed(self, smoke_run):
        import matplotlib.pyplot as plt

        from src import model_selection_reports as reports

        _, _, results, _ = smoke_run
        figure = reports.plot_comparison(results)
        assert figure.axes
        assert not plt.fignum_exists(figure.number)

    def test_fold_audit_chart_is_returned_closed(self, smoke_run):
        import matplotlib.pyplot as plt

        from src import model_selection_reports as reports

        _, development, _, _ = smoke_run
        figure = reports.plot_fold_audit(development)
        assert figure.axes
        assert not plt.fignum_exists(figure.number)
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert "Treino elegível" in labels
        assert "Teste de seleção" in labels
        assert "Teste de estresse" in labels
        assert "Teste excluído do score" in labels
