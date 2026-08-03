"""Tests for v4 tracking, fail-closed champion selection and candidate freezing.

Every test points MLflow at an isolated, per-test file-based tracking URI
(``tmp_path``) — never the real project ``mlruns/`` directory — so running the
suite can neither pollute nor be polluted by real experiment history.
"""

from __future__ import annotations

import gzip
import json
import pickle

import mlflow
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import src.temporal_optimizer as topt
import src.tracking as tracking
from src.environment import environment_fingerprint
from src.mlflow_integration import ExperimentConfigV4, MLflowTracker
from src.tracking import (
    EXPERIMENT_NAME_V4,
    PROVENANCE_ATTRIBUTE,
    log_temporal_model_run,
    params_hash,
    pipeline_provenance,
    setup_mlflow,
)

FINGERPRINT = "fp_v4_test"


@pytest.fixture(autouse=True)
def stable_clean_source_state(monkeypatch):
    """Tracking tests model committed production code, independent of the
    developer's working-tree state while the test itself is being edited."""
    state = {
        "git_commit": "abc1234",
        "git_source_dirty": "false",
        "git_source_status_hash": "clean",
        "git_source_fingerprint": "source-fingerprint-test",
    }
    monkeypatch.setattr(tracking, "describe_git_source_state", lambda: dict(state))
    monkeypatch.setattr(topt, "describe_git_source_state", lambda: dict(state))


@pytest.fixture
def mlflow_tmp_uri(tmp_path) -> str:
    return f"file:{(tmp_path / 'mlruns').as_posix()}"


@pytest.fixture
def tracker(mlflow_tmp_uri) -> MLflowTracker:
    tracker = MLflowTracker(ExperimentConfigV4(tracking_uri=mlflow_tmp_uri))
    tracker.setup_experiment()
    return tracker


def _fitted_pipeline(alpha: float) -> Pipeline:
    """Minimal pipeline with the step layout the freezing validation expects."""
    pipeline = Pipeline([("regressor", TransformedTargetRegressor(regressor=Ridge(alpha=alpha)))])
    return pipeline.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), pd.Series([1.0, 2.0, 3.0]))


def _log_run(
    tmp_path,
    estimator="Ridge",
    mae=100.0,
    alpha=1.0,
    run_mode="full",
    modeler_name="Periodic_Spline",
    fingerprint=FINGERPRINT,
    code_version=topt.CODE_VERSION,
    model=None,
):
    """Log one v4-shaped run and return ``(run_id, params, pipeline)``."""
    pipeline = model if model is not None else _fitted_pipeline(alpha)
    params = {"alpha": alpha, "modeler_name": modeler_name, "encoder": "MeanEncoder"}
    run_id = log_temporal_model_run(
        estimator_name=estimator,
        params=params,
        cv_metrics={"cv_mae_mean": mae, "cv_rmse_mean": mae * 1.2, "cv_r2_mean": 0.5},
        model_object=pipeline,
        input_example=pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        dataset_fingerprint=fingerprint,
        code_version=code_version,
        run_mode=run_mode,
        pipeline_spec={
            "modeler_name": modeler_name,
            "encoder": "MeanEncoder",
            "scaler": "StandardScaler",
            "selector": "SelectKBest",
            "target_transform": "log1p",
            "regime_policy": "all_observed",
            "regime_fingerprint": "all_observed",
        },
        n_features_selected=12,
        trials_planned=2,
        trials_completed=2,
        artifacts_dir=tmp_path / f"artifacts_{estimator}_{alpha}_{run_mode}",
    )
    return run_id, params, pipeline


class TestSetupMlflowActivatesGivenExperiment:
    """Regression test: setup_mlflow() used to always activate its own
    hardcoded v3 experiment, ignoring any custom name, so a caller's runs
    silently landed in v3."""

    def test_setup_mlflow_activates_the_passed_experiment_name(self, mlflow_tmp_uri):
        setup_mlflow(mlflow_tmp_uri, EXPERIMENT_NAME_V4)
        with mlflow.start_run() as run:
            experiment = mlflow.get_experiment(run.info.experiment_id)
            assert experiment.name == EXPERIMENT_NAME_V4
            assert experiment.name != "bike_sharing_demand_v3"

    def test_mlflow_tracker_setup_experiment_uses_its_own_config(self, tracker):
        with mlflow.start_run() as run:
            experiment = mlflow.get_experiment(run.info.experiment_id)
            assert experiment.name == "bike_sharing_demand_v4_model_selection"


class TestLogTemporalModelRun:
    def test_no_holdout_metric_is_ever_logged(self, tracker, tmp_path):
        run_id, _, _ = _log_run(tmp_path)
        run = mlflow.get_run(run_id)
        assert not any("holdout" in key.lower() for key in run.data.metrics)
        assert run.data.metrics["cv_mae_mean"] == pytest.approx(100.0)
        assert run.data.tags["final_holdout"] == "2023-12-01/2024-11-30"
        assert run.data.tags["cv_strategy"] == "ForwardMeteorologicalYearSplit"

    def test_provenance_tags_are_all_written(self, tracker, tmp_path):
        run_id, params, _ = _log_run(tmp_path)
        tags = mlflow.get_run(run_id).data.tags
        for tag in topt.REQUIRED_SELECTION_TAGS:
            assert tags[tag]
        assert tags["run_mode"] == "full"
        assert tags["cv_strategy_version"] == topt.CV_STRATEGY_VERSION
        assert tags["code_version"] == topt.CODE_VERSION
        assert tags["params_hash"] == params_hash(params)

    def test_pipeline_spec_is_recorded_as_tags_and_params(self, tracker, tmp_path):
        """(d) modeler_name must be visible in best_params and in MLflow."""
        run_id, _, _ = _log_run(tmp_path, modeler_name="Sin_Cos")
        run = mlflow.get_run(run_id)
        assert run.data.tags["modeler_name"] == "Sin_Cos"
        assert run.data.params["modeler_name"] == "Sin_Cos"
        assert run.data.tags["selector"] == "SelectKBest"
        assert run.data.tags["target_transform"] == "log1p"
        assert run.data.metrics["n_features_selected"] == pytest.approx(12.0)
        assert run.data.metrics["trials_completed"] == pytest.approx(2.0)

    def test_model_logged_with_signature_and_input_example(self, tracker, tmp_path):
        run_id, _, _ = _log_run(tmp_path)
        client = mlflow.tracking.MlflowClient()
        assert "model" in [f.path for f in client.list_artifacts(run_id)]
        assert mlflow.models.get_model_info(f"runs:/{run_id}/model").signature is not None

    def test_pipeline_html_repr_artifact_survives_unicode_diagram_arrows(self, tracker, tmp_path):
        """Regression test: sklearn.utils.estimator_html_repr() embeds Unicode
        collapse/expand arrows (U+25B8) even for a single estimator. Writing
        that HTML without encoding="utf-8" raises UnicodeEncodeError on Windows
        (cp1252 default) — caught by the same try/except as
        mlflow.sklearn.log_model(), so the model itself could still log fine
        while this artifact silently never got written."""
        run_id, _, _ = _log_run(tmp_path)
        client = mlflow.tracking.MlflowClient()
        manifest_files = [f.path for f in client.list_artifacts(run_id, "manifests")]
        assert any(path.endswith("_pipeline_repr.html") for path in manifest_files)


class TestGetBestModelByCV:
    """The auxiliary read API must be exactly as strict as the authoritative one.

    A permissive convenience query is not a convenience: it is a second, weaker
    route to the wrong run. Every filter
    ``select_champion_and_challengers`` applies is asserted here too.
    """

    def test_rejects_a_holdout_metric_name(self, tracker):
        with pytest.raises(ValueError, match="holdout"):
            tracker.get_best_model_by_cv(FINGERPRINT, metric_name="holdout_rmse")

    def test_selects_lowest_cv_mae_mean(self, tracker, tmp_path):
        _log_run(tmp_path, estimator="Ridge", mae=200.0)
        _log_run(tmp_path, estimator="XGBRegressor", mae=150.0)
        best = tracker.get_best_model_by_cv(FINGERPRINT)
        assert best["estimator"] == "XGBRegressor"
        assert best["metric_value"] == pytest.approx(150.0)

    def test_filters_by_dataset_fingerprint(self, tracker, tmp_path):
        _log_run(tmp_path, estimator="Ridge", mae=50.0, fingerprint="stale_fp")
        _log_run(tmp_path, estimator="XGBRegressor", mae=150.0, fingerprint="fresh_fp")
        best = tracker.get_best_model_by_cv("fresh_fp")
        assert best["estimator"] == "XGBRegressor"

    def test_a_smoke_run_can_never_win_the_full_query(self, tracker, tmp_path):
        _log_run(tmp_path, estimator="LGBMRegressor", mae=1.0, run_mode="smoke")
        keeper, _, _ = _log_run(tmp_path, estimator="Ridge", mae=500.0, run_mode="full")
        best = tracker.get_best_model_by_cv(FINGERPRINT, run_mode="full")
        assert best["run_id"] == keeper
        assert best["run_mode"] == "full"

    def test_a_superseded_code_version_is_excluded(self, tracker, tmp_path):
        _log_run(tmp_path, estimator="LGBMRegressor", mae=1.0, code_version="temporal_optimizer_v2")
        keeper, _, _ = _log_run(tmp_path, estimator="Ridge", mae=500.0)
        best = tracker.get_best_model_by_cv(FINGERPRINT)
        assert best["run_id"] == keeper
        assert best["code_version"] == topt.CODE_VERSION

    def test_a_run_from_another_environment_is_excluded(self, tracker, tmp_path):
        """The exact regression this filter exists for: a run produced under a
        different conda environment must lose even with a far better metric."""
        foreign, _, _ = _log_run(tmp_path, estimator="LGBMRegressor", mae=1.0)
        mlflow.tracking.MlflowClient().set_tag(
            foreign, "environment_fingerprint", "foreign_environment"
        )
        keeper, _, _ = _log_run(tmp_path, estimator="Ridge", mae=500.0)
        best = tracker.get_best_model_by_cv(FINGERPRINT)
        assert best["run_id"] == keeper
        assert best["environment_name"] == "Bike-Sharing"
        assert best["environment_fingerprint"] == environment_fingerprint()

    def test_an_unverified_artifact_is_excluded(self, tracker, tmp_path):
        unverified, _, _ = _log_run(tmp_path, estimator="LGBMRegressor", mae=1.0)
        mlflow.tracking.MlflowClient().set_tag(unverified, "model_artifact_verified", "false")
        keeper, _, _ = _log_run(tmp_path, estimator="Ridge", mae=500.0)
        assert tracker.get_best_model_by_cv(FINGERPRINT)["run_id"] == keeper

    def test_a_missing_provenance_tag_raises_instead_of_being_skipped(self, tracker):
        with mlflow.start_run():
            mlflow.log_metric("cv_mae_mean", 1.0)
        with pytest.raises(ValueError, match="fail-closed"):
            tracker.get_best_model_by_cv(FINGERPRINT)


# ---------------------------------------------------------------------------
# (k)/(m) Fail-closed champion selection
# ---------------------------------------------------------------------------


class TestFailClosedSelection:
    def test_smoke_runs_are_never_selected_as_a_full_champion(self, tracker, tmp_path):
        """The smoke run has the better metric; it must still lose, because it
        is not a full run."""
        _log_run(tmp_path, estimator="LGBMRegressor", mae=10.0, run_mode="smoke")
        full_run_id, _, _ = _log_run(tmp_path, estimator="Ridge", mae=500.0, run_mode="full")

        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT, run_mode="full")
        assert selection["champion"]["run_id"] == full_run_id
        assert selection["champion"]["run_mode"] == "full"
        assert all(c["run_mode"] == "full" for c in selection["challengers"])

    def test_smoke_selection_only_sees_smoke_runs(self, tracker, tmp_path):
        smoke_run_id, _, _ = _log_run(
            tmp_path, estimator="LGBMRegressor", mae=10.0, run_mode="smoke"
        )
        _log_run(tmp_path, estimator="Ridge", mae=5.0, run_mode="full")
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT, run_mode="smoke")
        assert selection["champion"]["run_id"] == smoke_run_id

    def test_a_different_dataset_fingerprint_is_excluded(self, tracker, tmp_path):
        _log_run(tmp_path, estimator="LGBMRegressor", mae=10.0, fingerprint="other_data")
        keeper, _, _ = _log_run(tmp_path, estimator="Ridge", mae=500.0)
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        assert selection["champion"]["run_id"] == keeper

    def test_a_different_code_version_is_excluded(self, tracker, tmp_path):
        _log_run(
            tmp_path, estimator="LGBMRegressor", mae=10.0, code_version="temporal_optimizer_v0"
        )
        keeper, _, _ = _log_run(tmp_path, estimator="Ridge", mae=500.0)
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        assert selection["champion"]["run_id"] == keeper

    def test_a_run_without_the_provenance_tags_makes_selection_fail(self, tracker, tmp_path):
        """A run logged outside log_temporal_model_run carries none of the
        required tags; selection must refuse to rank rather than guess."""
        with mlflow.start_run():
            mlflow.log_metric("cv_mae_mean", 1.0)
        with pytest.raises(ValueError, match="provenance"):
            topt.select_champion_and_challengers(tracker, FINGERPRINT)

    def test_runs_with_a_null_required_tag_are_dropped(self, tracker, tmp_path):
        keeper, _, _ = _log_run(tmp_path, estimator="Ridge", mae=500.0)
        with mlflow.start_run():
            mlflow.set_tag("estimator", "Ghost")
            mlflow.log_metric("cv_mae_mean", 1.0)
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        assert selection["champion"]["run_id"] == keeper

    def test_no_matching_run_raises_instead_of_returning_nothing(self, tracker, tmp_path):
        _log_run(tmp_path, estimator="Ridge", mae=500.0)
        with pytest.raises(RuntimeError, match="matches"):
            topt.select_champion_and_challengers(tracker, "a_fingerprint_that_never_ran")

    def test_a_holdout_metric_name_is_rejected(self, tracker, tmp_path):
        _log_run(tmp_path)
        with pytest.raises(ValueError, match="holdout"):
            topt.select_champion_and_challengers(tracker, FINGERPRINT, metric_name="holdout_mae")

    def test_selection_carries_the_full_pipeline_description(self, tracker, tmp_path):
        _log_run(tmp_path, estimator="Ridge", mae=120.0, modeler_name="Normalizers")
        champion = topt.select_champion_and_challengers(tracker, FINGERPRINT)["champion"]
        assert champion["modeler_name"] == "Normalizers"
        assert champion["encoder"] == "MeanEncoder"
        assert champion["selector"] == "SelectKBest"
        assert champion["target_transform"] == "log1p"
        assert champion["best_params"]["modeler_name"] == "Normalizers"
        assert champion["model_uri"] == f"runs:/{champion['run_id']}/model"


class TestFreezingRefusesASystematicallyTruncatedChampion:
    """A champion whose folds mostly stopped at their boosting ceiling has a
    tree count that records the limit, not the data. Freezing it would hand
    notebook 05 a model whose size nobody chose."""

    @staticmethod
    def _truncated(candidate, n_cap_hit=3, n_with_budget=5):
        candidate = dict(candidate)
        candidate["n_folds_cap_hit"] = n_cap_hit
        candidate["n_folds_with_budget"] = n_with_budget
        candidate["iteration_ceiling"] = "2000"
        candidate["final_n_estimators"] = "1900"
        return candidate

    def _selection(self, tracker, tmp_path, run_mode="full"):
        run_id, _, pipeline = _log_run(tmp_path, estimator="Ridge", mae=100.0, run_mode=run_mode)
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT, run_mode=run_mode)
        return selection, run_id, pipeline

    def test_a_full_champion_with_systematic_truncation_is_refused(self, tracker, tmp_path):
        selection, run_id, pipeline = self._selection(tracker, tmp_path)
        selection["champion"] = self._truncated(selection["champion"])
        with pytest.raises(RuntimeError, match="systematic boosting truncation"):
            topt.freeze_candidates(
                selection,
                {run_id: pipeline},
                run_mode="full",
                candidates_root=tmp_path / "candidates",
            )

    def test_a_full_challenger_with_systematic_truncation_is_also_refused(self, tracker, tmp_path):
        selection, run_id, pipeline = self._selection(tracker, tmp_path)
        challenger = self._truncated(selection["champion"])
        challenger["run_id"] = "truncated-challenger"
        challenger["estimator"] = "XGBRegressor"
        selection["challengers"] = [challenger]

        with pytest.raises(RuntimeError, match="challenger XGBRegressor"):
            topt.freeze_candidates(
                selection,
                {run_id: pipeline},
                run_mode="full",
                candidates_root=tmp_path / "candidates",
            )

    def test_the_override_exists_for_an_investigated_case(self, tracker, tmp_path):
        selection, run_id, pipeline = self._selection(tracker, tmp_path)
        selection["champion"] = self._truncated(selection["champion"])
        manifest_path = topt.freeze_candidates(
            selection,
            {run_id: pipeline},
            run_mode="full",
            candidates_root=tmp_path / "candidates",
            allow_boosting_truncation=True,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "systematic" in manifest["boosting_truncation"]

    def test_an_isolated_truncated_fold_does_not_block_freezing(self, tracker, tmp_path):
        selection, run_id, pipeline = self._selection(tracker, tmp_path)
        selection["champion"] = self._truncated(selection["champion"], n_cap_hit=1)
        manifest_path = topt.freeze_candidates(
            selection,
            {run_id: pipeline},
            run_mode="full",
            candidates_root=tmp_path / "candidates",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["boosting_truncation"] is None

    def test_a_smoke_run_is_never_blocked_by_it(self, tracker, tmp_path):
        """Smoke artifacts are provisional by construction; the guard protects
        the definitive hand-off, not the infrastructure check."""
        selection, run_id, pipeline = self._selection(tracker, tmp_path, run_mode="smoke")
        selection["champion"] = self._truncated(selection["champion"])
        assert topt.freeze_candidates(
            selection,
            {run_id: pipeline},
            run_mode="smoke",
            candidates_root=tmp_path / "candidates",
        ).exists()


# ---------------------------------------------------------------------------
# (l) Freezing persists exactly the pipeline of the selected run
# ---------------------------------------------------------------------------


class TestFreezeCandidates:
    def test_robust_trend_wrapper_is_frozen_with_its_dynamic_estimator(self, tmp_path):
        core_pipeline = _fitted_pipeline(alpha=0.5)
        pipeline = topt.RobustTrendResidualRegressor(estimator=core_pipeline)
        pipeline.estimator_ = core_pipeline
        run_id = "robust-trend-run"
        tracking.stamp_pipeline_provenance(
            pipeline,
            source_run_id=run_id,
            best_params={},
            pipeline_spec={},
            code_version=topt.CODE_VERSION,
            dataset_fingerprint=FINGERPRINT,
        )

        manifest_path = topt.freeze_candidates(
            {
                "run_mode": "smoke",
                "champion": {"run_id": run_id, "estimator": "Ridge"},
                "challengers": [],
            },
            {run_id: pipeline},
            run_mode="smoke",
            candidates_root=tmp_path / "candidates",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with gzip.open(manifest["champion"]["artifact_path"], "rb") as handle:
            restored = pickle.load(handle)

        assert isinstance(restored, topt.RobustTrendResidualRegressor)
        assert type(restored.estimator_.named_steps["regressor"].regressor).__name__ == "Ridge"

    def test_robust_trend_wrapper_keeps_the_estimator_class_guard(self):
        core_pipeline = _fitted_pipeline(alpha=0.5)
        pipeline = topt.RobustTrendResidualRegressor(estimator=core_pipeline)
        pipeline.estimator_ = core_pipeline

        with pytest.raises(ValueError, match="Pipeline/run mismatch"):
            topt._validate_candidate_pipeline(
                {
                    "run_id": "mismatched-robust-trend-run",
                    "estimator": "RandomForestRegressor",
                },
                pipeline,
            )

    def test_persists_the_pipeline_of_the_selected_run_not_of_the_estimator_name(
        self, tracker, tmp_path
    ):
        """Two runs of the same estimator differ only in a hyperparameter. The
        frozen artifact must be the winner's, which is exactly what a lookup
        keyed on estimator name cannot guarantee."""
        loser_id, _, loser_pipeline = _log_run(tmp_path, estimator="Ridge", mae=900.0, alpha=99.0)
        winner_id, _, winner_pipeline = _log_run(tmp_path, estimator="Ridge", mae=100.0, alpha=0.5)

        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        assert selection["champion"]["run_id"] == winner_id

        manifest_path = topt.freeze_candidates(
            selection,
            {winner_id: winner_pipeline, loser_id: loser_pipeline},
            run_mode="full",
            candidates_root=tmp_path / "candidates",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with gzip.open(manifest["champion"]["artifact_path"], "rb") as handle:
            restored = pickle.load(handle)

        assert restored.named_steps["regressor"].regressor.alpha == pytest.approx(0.5)
        assert manifest["champion"]["run_id"] == winner_id

    def test_manifest_carries_every_required_field(self, tracker, tmp_path):
        run_id, _, pipeline = _log_run(tmp_path, estimator="Ridge", mae=100.0)
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        manifest_path = topt.freeze_candidates(
            selection, {run_id: pipeline}, run_mode="full", candidates_root=tmp_path / "candidates"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        champion = manifest["champion"]
        for field in (
            "run_id",
            "estimator",
            "run_mode",
            "cv_strategy_version",
            "code_version",
            "dataset_fingerprint",
            "best_params",
            "modeler_name",
            "encoder",
            "scaler",
            "selector",
            "target_transform",
            "model_uri",
            "artifact_path",
            "cv_mae_mean",
        ):
            assert field in champion, field
        assert manifest["provisional"] is False
        assert manifest_path.name == "candidates_manifest.json"

    def test_smoke_freezing_is_provisional_and_kept_apart(self, tracker, tmp_path):
        run_id, _, pipeline = _log_run(tmp_path, estimator="Ridge", mae=100.0, run_mode="smoke")
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT, run_mode="smoke")
        manifest_path = topt.freeze_candidates(
            selection, {run_id: pipeline}, run_mode="smoke", candidates_root=tmp_path / "candidates"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["provisional"] is True
        assert manifest["run_mode"] == "smoke"
        assert manifest_path.name == "candidates_manifest_smoke.json"
        assert manifest_path.parent.name == "candidates_v4_smoke"
        assert "provisional_" in manifest["champion"]["artifact_path"]
        definitive = tmp_path / "candidates" / "candidates_v4" / "candidates_manifest.json"
        assert not definitive.exists()

    def test_freezing_refuses_a_run_mode_mismatch(self, tracker, tmp_path):
        run_id, _, pipeline = _log_run(tmp_path, estimator="Ridge", mae=100.0, run_mode="smoke")
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT, run_mode="smoke")
        with pytest.raises(ValueError, match="Refusing to freeze"):
            topt.freeze_candidates(
                selection,
                {run_id: pipeline},
                run_mode="full",
                candidates_root=tmp_path / "candidates",
            )

    def test_a_mismatched_pipeline_is_rejected(self, tracker, tmp_path):
        from sklearn.ensemble import RandomForestRegressor

        run_id, _, _ = _log_run(tmp_path, estimator="Ridge", mae=100.0)
        wrong = Pipeline(
            [("regressor", TransformedTargetRegressor(regressor=RandomForestRegressor()))]
        )
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        with pytest.raises(ValueError, match="Pipeline/run mismatch"):
            topt.freeze_candidates(
                selection,
                {run_id: wrong},
                run_mode="full",
                candidates_root=tmp_path / "candidates",
            )

    def test_a_pipeline_missing_from_memory_is_loaded_from_its_own_run(self, tracker, tmp_path):
        """A champion selected from an earlier session has no in-memory
        pipeline; it must come from that run's own artifact, never from
        another run of the same estimator."""
        run_id, _, _ = _log_run(tmp_path, estimator="Ridge", mae=100.0, alpha=7.5)
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        manifest_path = topt.freeze_candidates(
            selection, {}, run_mode="full", candidates_root=tmp_path / "candidates"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["champion"]["pipeline_source"] == "mlflow"
        with gzip.open(manifest["champion"]["artifact_path"], "rb") as handle:
            restored = pickle.load(handle)
        assert restored.named_steps["regressor"].regressor.alpha == pytest.approx(7.5)
        assert manifest["champion"]["run_id"] == run_id


class TestPipelineProvenanceStamp:
    """Freezing must verify the object it received, not the run's own metadata.

    Re-hashing ``candidate["best_params"]`` only shows the run's metadata is
    self-consistent; it says nothing about the pipeline handed over under that
    ``run_id``. The stamp written onto the artifact at logging time is what
    closes that gap.
    """

    def test_logging_stamps_the_pipeline_with_its_run_identity(self, tracker, tmp_path):
        run_id, params, pipeline = _log_run(tmp_path, estimator="Ridge", alpha=1.0)
        provenance = pipeline_provenance(pipeline)
        assert provenance is not None
        assert provenance["source_run_id"] == run_id
        assert provenance["best_params_hash"] == params_hash(params)
        assert provenance["code_version"] == topt.CODE_VERSION
        assert provenance["dataset_fingerprint"] == FINGERPRINT
        assert provenance["pipeline_spec_hash"]

    def test_the_stamp_survives_a_pickle_round_trip(self, tracker, tmp_path):
        _, _, pipeline = _log_run(tmp_path, estimator="Ridge", alpha=1.0)
        restored = pickle.loads(pickle.dumps(pipeline))
        assert pipeline_provenance(restored) == pipeline_provenance(pipeline)

    def test_the_stamp_reaches_the_mlflow_artifact(self, tracker, tmp_path):
        run_id, _, _ = _log_run(tmp_path, estimator="Ridge", alpha=3.0)
        loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        assert pipeline_provenance(loaded)["source_run_id"] == run_id

    def test_a_same_class_impostor_under_the_right_run_id_is_rejected(self, tracker, tmp_path):
        """The scenario the requirement names: the run is Ridge alpha=1 and the
        mapping offers Ridge alpha=999 under that very ``run_id``."""
        run_id, _, _ = _log_run(tmp_path, estimator="Ridge", mae=100.0, alpha=1.0)
        impostor = _fitted_pipeline(999.0)
        assert impostor.named_steps["regressor"].regressor.alpha == 999.0

        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        assert selection["champion"]["run_id"] == run_id
        with pytest.raises(ValueError, match="no provenance stamp"):
            topt.freeze_candidates(
                selection,
                {run_id: impostor},
                run_mode="full",
                candidates_root=tmp_path / "candidates",
            )

    def test_a_pipeline_stamped_by_another_run_is_rejected(self, tracker, tmp_path):
        """Even a properly stamped pipeline is refused under the wrong run_id."""
        other_id, _, other_pipeline = _log_run(tmp_path, estimator="Ridge", mae=900.0, alpha=42.0)
        winner_id, _, _ = _log_run(tmp_path, estimator="Ridge", mae=100.0, alpha=1.0)

        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        assert selection["champion"]["run_id"] == winner_id
        assert pipeline_provenance(other_pipeline)["source_run_id"] == other_id
        with pytest.raises(ValueError, match="provenance mismatch"):
            topt.freeze_candidates(
                selection,
                {winner_id: other_pipeline},
                run_mode="full",
                candidates_root=tmp_path / "candidates",
            )

    def test_a_tampered_parameter_hash_is_rejected(self, tracker, tmp_path):
        run_id, _, pipeline = _log_run(tmp_path, estimator="Ridge", mae=100.0, alpha=1.0)
        stamp = dict(pipeline_provenance(pipeline))
        stamp["best_params_hash"] = "deadbeefdeadbeef"
        setattr(pipeline, PROVENANCE_ATTRIBUTE, stamp)

        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        with pytest.raises(ValueError, match="provenance mismatch"):
            topt.freeze_candidates(
                selection,
                {run_id: pipeline},
                run_mode="full",
                candidates_root=tmp_path / "candidates",
            )

    def test_a_correctly_stamped_pipeline_is_accepted(self, tracker, tmp_path):
        run_id, _, pipeline = _log_run(tmp_path, estimator="Ridge", mae=100.0, alpha=1.0)
        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        manifest_path = topt.freeze_candidates(
            selection, {run_id: pipeline}, run_mode="full", candidates_root=tmp_path / "candidates"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["champion"]["pipeline_source"] == "memory"


class TestModelLoggingIsFailClosed:
    """A run whose model never reached the store cannot become a champion."""

    def _break_log_model(self, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("artifact store unavailable")

        monkeypatch.setattr(mlflow.sklearn, "log_model", explode)

    def test_full_mode_raises_instead_of_warning(self, tracker, tmp_path, monkeypatch):
        self._break_log_model(monkeypatch)
        with pytest.raises(RuntimeError, match="cannot be frozen as a definitive candidate"):
            _log_run(tmp_path, estimator="Ridge", run_mode="full")

    def test_smoke_mode_records_the_failure_and_continues(self, tracker, tmp_path, monkeypatch):
        self._break_log_model(monkeypatch)
        run_id, _, _ = _log_run(tmp_path, estimator="Ridge", run_mode="smoke")
        assert mlflow.get_run(run_id).data.tags["model_logged"] == "false"

    def test_a_successful_run_is_tagged_as_logged(self, tracker, tmp_path):
        run_id, _, _ = _log_run(tmp_path, estimator="Ridge")
        assert mlflow.get_run(run_id).data.tags["model_logged"] == "true"

    def test_full_selection_skips_a_run_without_a_stored_model(self, tracker, tmp_path):
        good_id, _, good_pipeline = _log_run(tmp_path, estimator="Ridge", mae=500.0, alpha=2.0)
        # A better metric, but its model was never stored.
        with mlflow.start_run() as run:
            mlflow.set_tags(
                {
                    "estimator": "Ridge",
                    "dataset_fingerprint": FINGERPRINT,
                    "cv_strategy_version": topt.CV_STRATEGY_VERSION,
                    "code_version": topt.CODE_VERSION,
                    "run_mode": "full",
                    "model_logged": "false",
                }
            )
            mlflow.log_metric("cv_mae_mean", 1.0)
            unstored_id = run.info.run_id

        selection = topt.select_champion_and_challengers(tracker, FINGERPRINT)
        assert selection["champion"]["run_id"] == good_id
        assert unstored_id not in [c["run_id"] for c in selection["challengers"]]
        assert good_pipeline is not None
