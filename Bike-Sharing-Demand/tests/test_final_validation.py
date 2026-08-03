"""Tests for the one-shot final-holdout validation (:mod:`src.final_validation`).

Nothing here touches the real holdout, the real dataset target or the user's
real MLflow runs. The frozen candidates are fitted on synthetic development data,
the manifest contract is exercised with synthetic dictionaries built from the
module's own constants, and the holdout-sealing logic is tested on a synthetic
frame engineered to have exactly the same 8,784-hour window as the real one.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from src import final_validation as fv
from src.final_validation import (
    CONFIRMED,
    EXPECTED_DATASET_FINGERPRINT,
    EXPECTED_ENVIRONMENT_FINGERPRINT,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_REGIME_FINGERPRINT,
    NOT_CONFIRMED,
    PREREGISTERED_CHAMPION_RUN_ID,
    CandidateHoldoutEvaluation,
    FinalEvaluationData,
    FinalValidationConfig,
    FinalValidationPlan,
    RunRecord,
)
from src.modeling_pipeline import build_dynamic_pipeline
from src.temporal_optimizer import CODE_VERSION, CV_STRATEGY_VERSION
from src.tracking import params_hash, spec_hash, stamp_pipeline_provenance

_FIXED_TRIAL = {
    "modeler_name": "Periodic_Spline",
    "encoder": "OrdinalEncoder",
    "standardizer": "StandardScaler",
    # RandomForest
    "n_estimators": 50,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    # HistGradientBoosting
    "learning_rate": 0.1,
    "max_iter": 60,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.01,
    # CatBoost
    "iterations": 120,
    "depth": 6,
    "random_strength": 5,
    "bagging_temperature": 0.5,
    "l2_leaf_reg": 3.0,
    "border_count": 64,
    "selector": "NoSelector",
    "trend_extrapolation_damping": 0.0,
}

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


def make_raw(start: str, periods: int, freq: str = "h", seed: int = 0) -> pd.DataFrame:
    """A synthetic frame in the ``read_data`` output shape."""
    dates = pd.date_range(start, periods=periods, freq=freq)
    rng = np.random.default_rng(seed)
    n = len(dates)
    season = pd.Series(dates.month).map(_MONTH_TO_SEASON).to_numpy()
    return pd.DataFrame(
        {
            "DateTime": dates.normalize(),
            "Hour": dates.hour.astype(np.int64),
            "Rented Bike Count": rng.integers(0, 1500, n).astype(float),
            "Temperature(C)": rng.uniform(-12, 36, n),
            "Humidity(%)": rng.integers(10, 100, n).astype(float),
            "Wind speed (m/s)": rng.uniform(0, 8, n),
            "Dew point temperature(C)": rng.uniform(-20, 25, n),
            "Solar Radiation (MJ/m2)": rng.uniform(0, 3.5, n),
            "Rainfall(mm)": np.zeros(n),
            "Snowfall (cm)": np.zeros(n),
            "Sunshine (hr)": rng.uniform(0, 1, n),
            "Cloud Cover (oktas)": rng.uniform(0, 10, n),
            "Ground Temp(C)": rng.uniform(-15, 45, n),
            "Visibility (10m)": rng.integers(100, 2000, n).astype(float),
            "Seasons": season,
            "Holiday": np.where(rng.random(n) < 0.05, "Holiday", "No Holiday"),
            "Functioning Day": "Yes",
        }
    )


def fit_pipeline(estimator: str, dev: pd.DataFrame):
    """Fit one dynamic pipeline (robust-trend + periodic spline) on synthetic data."""
    trial = optuna.trial.FixedTrial(_FIXED_TRIAL)
    pipeline, spec = build_dynamic_pipeline(
        trial, estimator, target_strategy="robust_trend_residual"
    )
    pipeline.fit(dev.drop(columns=["Rented Bike Count"]), dev["Rented Bike Count"])
    return pipeline, spec


def make_entry(role: str, run_id: str, estimator: str, spec, artifact_path: Path) -> dict:
    """A manifest entry whose hashes match a stamped pipeline."""
    best_params = {"seed": run_id}
    return {
        "role": role,
        "run_id": run_id,
        "estimator": estimator,
        "params_hash": params_hash(best_params),
        "pipeline_spec_hash": spec_hash(spec.as_tags()),
        "code_version": CODE_VERSION,
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "environment_name": "Bike-Sharing",
        "environment_fingerprint": EXPECTED_ENVIRONMENT_FINGERPRINT,
        "regime_policy": "normal_operations",
        "regime_fingerprint": EXPECTED_REGIME_FINGERPRINT,
        "cv_strategy_version": CV_STRATEGY_VERSION,
        "run_mode": "full",
        "model_logged": "true",
        "model_artifact_verified": "true",
        "cv_mae_mean": 800.0,
        "cv_rmse_mean": 1200.0,
        "cv_r2_mean": 0.83,
        "cv_wape_mean": 0.23,
        "cv_mean_bias": -50.0,
        "cv_mae_weighted": 810.0,
        "artifact_path": str(artifact_path),
        "_best_params": best_params,
    }


@pytest.fixture(scope="module")
def synthetic_dev() -> pd.DataFrame:
    """Multi-year synthetic development data so the robust trend can be fitted."""
    return make_raw("2019-01-01", 3000, seed=1)


@pytest.fixture(scope="module")
def frozen_dir(tmp_path_factory, synthetic_dev):
    """Fit, stamp and pickle three candidates; return their directory and entries."""
    directory = tmp_path_factory.mktemp("frozen_candidates")
    specs = [
        ("champion", "CatBoostRegressor", "run_champion"),
        ("challenger", "HistGradientBoostingRegressor", "run_challenger_1"),
        ("challenger", "RandomForestRegressor", "run_challenger_2"),
    ]
    entries = []
    for role, estimator, run_id in specs:
        pipeline, spec = fit_pipeline(estimator, synthetic_dev)
        path = directory / f"{role}_{estimator}_{run_id}.pkl.gz"
        entry = make_entry(role, run_id, estimator, spec, path)
        stamp_pipeline_provenance(
            pipeline,
            run_id,
            entry["_best_params"],
            spec.as_tags(),
            CODE_VERSION,
            EXPECTED_DATASET_FINGERPRINT,
        )
        with gzip.open(path, "wb") as handle:
            pickle.dump(pipeline, handle)
        entries.append(entry)
    return directory, entries


@pytest.fixture(scope="module")
def manifest(frozen_dir) -> dict:
    """A synthetic definitive manifest wrapping the three fitted candidates."""
    _, entries = frozen_dir
    return {
        "run_mode": "full",
        "provisional": False,
        "champion": entries[0],
        "challengers": entries[1:],
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "regime_fingerprint": EXPECTED_REGIME_FINGERPRINT,
    }


@pytest.fixture(scope="module")
def candidates(manifest):
    """Loaded, provenance-verified candidates."""
    return fv.load_frozen_candidates(manifest)


@pytest.fixture(scope="module")
def holdout_raw() -> pd.DataFrame:
    """A small synthetic holdout frame (not the real holdout)."""
    return make_raw("2024-01-01", 600, seed=7)


@pytest.fixture(scope="module")
def eval_data(holdout_raw) -> FinalEvaluationData:
    """A FinalEvaluationData built directly from the synthetic holdout frame."""
    target = "Rented Bike Count"
    y = holdout_raw[target].reset_index(drop=True)
    X = holdout_raw.drop(columns=[target]).reset_index(drop=True)
    ts = (
        pd.to_datetime(holdout_raw["DateTime"]) + pd.to_timedelta(holdout_raw["Hour"], unit="h")
    ).reset_index(drop=True)
    return FinalEvaluationData(
        X_holdout=X,
        y_holdout=y,
        timestamps=ts,
        dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT,
        holdout_fingerprint="synthetic",
        regime_fingerprint=EXPECTED_REGIME_FINGERPRINT,
        dev_start=ts.min(),
        dev_end=ts.max(),
        n_dev_rows=3000,
        holdout_start=ts.min(),
        holdout_end=ts.max(),
        n_holdout_rows=len(X),
        n_post_holdout_rows=0,
        post_holdout_start=None,
        post_holdout_end=None,
        environment={},
    )


@pytest.fixture(scope="module")
def evaluations(candidates, eval_data):
    """One holdout evaluation per candidate."""
    config = FinalValidationConfig(log_to_mlflow=False)
    return [fv.evaluate_candidate(c, eval_data, config.error_quantiles) for c in candidates]


def valid_manifest_dict() -> dict:
    """A manifest matching the fixed notebook-05 contract, built from constants."""
    return {
        "run_mode": "full",
        "provisional": False,
        "selection_metric": "cv_mae_weighted",
        "cv_strategy_version": CV_STRATEGY_VERSION,
        "code_version": CODE_VERSION,
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "environment_name": "Bike-Sharing",
        "environment_fingerprint": EXPECTED_ENVIRONMENT_FINGERPRINT,
        "regime_policy": "normal_operations",
        "regime_fingerprint": EXPECTED_REGIME_FINGERPRINT,
        "champion": {"run_id": PREREGISTERED_CHAMPION_RUN_ID, "estimator": "CatBoostRegressor"},
        "challengers": [
            {
                "run_id": "381e7061d8ef440e917c792ab38e59f3",
                "estimator": "HistGradientBoostingRegressor",
            },
            {"run_id": "030ef18ea01d46caacbeb2bc0765da61", "estimator": "RandomForestRegressor"},
        ],
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_default_tracking_uri_points_to_project_mlruns(self):
        config = FinalValidationConfig()

        assert config.tracking_uri == fv.DEFAULT_TRACKING_URI
        assert config.tracking_uri.startswith("file:")
        expected_mlruns = fv.DEFAULT_CANDIDATES_ROOT.parents[2] / "mlruns"
        assert Path(config.tracking_uri.removeprefix("file:")) == expected_mlruns


# ---------------------------------------------------------------------------
# Manifest audit
# ---------------------------------------------------------------------------


class TestManifestAudit:
    def test_a_valid_manifest_is_accepted(self):
        fv.audit_manifest(valid_manifest_dict())

    def test_a_provisional_manifest_is_rejected(self):
        manifest = valid_manifest_dict()
        manifest["provisional"] = True
        with pytest.raises(ValueError):
            fv.audit_manifest(manifest)

    def test_a_non_full_run_mode_is_rejected(self):
        manifest = valid_manifest_dict()
        manifest["run_mode"] = "smoke"
        with pytest.raises(ValueError):
            fv.audit_manifest(manifest)

    def test_a_divergent_dataset_fingerprint_is_rejected(self):
        manifest = valid_manifest_dict()
        manifest["dataset_fingerprint"] = "0000000000000000"
        with pytest.raises(ValueError):
            fv.audit_manifest(manifest)

    def test_a_divergent_regime_fingerprint_is_rejected(self):
        manifest = valid_manifest_dict()
        manifest["regime_fingerprint"] = "0000000000000000"
        with pytest.raises(ValueError):
            fv.audit_manifest(manifest)

    def test_a_divergent_champion_run_id_is_rejected(self):
        manifest = valid_manifest_dict()
        manifest["champion"]["run_id"] = "deadbeefdeadbeef"
        with pytest.raises(ValueError):
            fv.audit_manifest(manifest)

    def test_the_real_definitive_manifest_matches_the_contract(self):
        path = fv.DEFAULT_MANIFEST_PATH
        if not path.exists():
            pytest.skip("real candidate manifest is not present in this checkout")
        fv.audit_manifest(fv.load_manifest(path))


# ---------------------------------------------------------------------------
# Source-run audit (read-only, injectable)
# ---------------------------------------------------------------------------


def _good_run_record(entry: dict) -> RunRecord:
    tags = {
        tag: str(entry[tag])
        for tag in (
            "estimator",
            "dataset_fingerprint",
            "cv_strategy_version",
            "code_version",
            "run_mode",
            "environment_name",
            "environment_fingerprint",
            "regime_policy",
            "regime_fingerprint",
            "params_hash",
            "pipeline_spec_hash",
        )
    }
    tags.update(
        {"model_logged": "true", "model_artifact_verified": "true", "git_source_dirty": "false"}
    )
    return RunRecord(run_id=entry["run_id"], status="FINISHED", tags=tags)


class TestSourceRunAudit:
    def test_a_sound_run_record_has_no_problems(self, manifest):
        entry = manifest["champion"]
        assert fv.verify_run_record(entry, _good_run_record(entry)) == []

    def test_a_non_finished_run_is_rejected(self, manifest):
        entry = manifest["champion"]
        record = _good_run_record(entry)
        record.status = "RUNNING"
        assert any("FINISHED" in problem for problem in fv.verify_run_record(entry, record))

    def test_a_divergent_run_id_is_rejected(self, manifest):
        entry = manifest["champion"]
        record = _good_run_record(entry)
        record.run_id = "other"
        assert any("run_id" in problem for problem in fv.verify_run_record(entry, record))

    def test_an_unverified_artifact_is_rejected(self, manifest):
        entry = manifest["champion"]
        record = _good_run_record(entry)
        record.tags["model_artifact_verified"] = "false"
        assert any("model_artifact_verified" in p for p in fv.verify_run_record(entry, record))

    def test_a_dirty_source_run_is_rejected(self, manifest):
        entry = manifest["champion"]
        record = _good_run_record(entry)
        record.tags["git_source_dirty"] = "true"
        assert any("git_source_dirty" in p for p in fv.verify_run_record(entry, record))

    def test_audit_source_runs_accepts_sound_records(self, manifest):
        records = fv.audit_source_runs(
            manifest,
            fetch=lambda run_id: _good_run_record(
                next(e for _, e in fv.manifest_entries(manifest) if e["run_id"] == run_id)
            ),
        )
        assert set(records) == {"run_champion", "run_challenger_1", "run_challenger_2"}

    def test_audit_source_runs_rejects_an_unfinished_run(self, manifest):
        def fetch(run_id):
            entry = next(e for _, e in fv.manifest_entries(manifest) if e["run_id"] == run_id)
            record = _good_run_record(entry)
            if run_id == "run_challenger_1":
                record.status = "FAILED"
            return record

        with pytest.raises(ValueError):
            fv.audit_source_runs(manifest, fetch=fetch)

    def test_audit_source_runs_uses_project_tracking_uri_by_default(self, manifest, monkeypatch):
        seen_tracking_uris = []

        def fetch(run_id, tracking_uri):
            seen_tracking_uris.append(tracking_uri)
            entry = next(e for _, e in fv.manifest_entries(manifest) if e["run_id"] == run_id)
            return _good_run_record(entry)

        monkeypatch.setattr(fv, "_fetch_run_record", fetch)

        fv.audit_source_runs(manifest)

        assert seen_tracking_uris == [fv.DEFAULT_TRACKING_URI] * 3


# ---------------------------------------------------------------------------
# Frozen-candidate loading and provenance
# ---------------------------------------------------------------------------


class TestFrozenCandidateLoading:
    def test_valid_candidates_load_and_verify(self, candidates):
        assert [c.role for c in candidates] == ["champion", "challenger", "challenger"]
        assert candidates[0].run_id == "run_champion"
        assert all(c.artifact_sha256 for c in candidates)

    def test_a_missing_artifact_is_rejected(self, manifest, tmp_path):
        entry = dict(manifest["champion"])
        entry["artifact_path"] = str(tmp_path / "absent.pkl.gz")
        with pytest.raises(FileNotFoundError):
            fv.load_frozen_candidate("champion", entry)

    def test_a_divergent_provenance_is_rejected(self, manifest):
        entry = dict(manifest["champion"])
        entry["params_hash"] = "0000000000000000"
        with pytest.raises(ValueError):
            fv.load_frozen_candidate("champion", entry)

    def test_a_mismatched_estimator_is_rejected(self, manifest):
        entry = dict(manifest["challengers"][0])
        entry["estimator"] = "RandomForestRegressor"  # the artifact is an HGB
        with pytest.raises(ValueError):
            fv.load_frozen_candidate("challenger", entry)


# ---------------------------------------------------------------------------
# Holdout sealing (synthetic frame)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_window_raw() -> pd.DataFrame:
    """A synthetic frame with complete hourly coverage over the holdout window."""
    return make_raw("2023-11-15", periods=24 * 400, seed=3)


class TestHoldoutSealing:
    def test_the_window_is_exact(self, full_window_raw):
        sealed = fv.seal_holdout(full_window_raw, "Rented Bike Count", "2023-12-01", "2024-11-30")
        assert sealed.timestamps.min() == pd.Timestamp("2023-12-01 00:00:00")
        assert sealed.timestamps.max() == pd.Timestamp("2024-11-30 23:00:00")

    def test_the_holdout_has_exactly_8784_rows(self, full_window_raw):
        sealed = fv.seal_holdout(full_window_raw, "Rented Bike Count", "2023-12-01", "2024-11-30")
        assert sealed.n_holdout_rows == EXPECTED_HOLDOUT_ROWS == 8784

    def test_the_post_holdout_tail_is_discarded(self, full_window_raw):
        sealed = fv.seal_holdout(full_window_raw, "Rented Bike Count", "2023-12-01", "2024-11-30")
        assert sealed.n_post_holdout_rows > 0
        assert bool((sealed.timestamps <= pd.Timestamp("2024-11-30 23:00:00")).all())

    def test_features_target_and_timestamps_stay_aligned(self, full_window_raw):
        sealed = fv.seal_holdout(full_window_raw, "Rented Bike Count", "2023-12-01", "2024-11-30")
        assert len(sealed.X_holdout) == len(sealed.y_holdout) == len(sealed.timestamps)
        assert "Rented Bike Count" not in sealed.X_holdout.columns

    def test_a_short_holdout_is_rejected(self):
        short = make_raw("2023-12-01", periods=24 * 30, seed=4)  # only one month
        with pytest.raises(ValueError):
            fv.seal_holdout(short, "Rented Bike Count", "2023-12-01", "2024-11-30")


# ---------------------------------------------------------------------------
# Predictions and metrics
# ---------------------------------------------------------------------------


class TestPredictionsAndMetrics:
    def test_predictions_are_finite_non_negative_and_aligned(self, candidates, eval_data):
        for candidate in candidates:
            predictions = fv.predict_holdout(candidate, eval_data.X_holdout)
            assert predictions.shape == (len(eval_data.X_holdout),)
            assert np.all(np.isfinite(predictions))
            assert np.all(predictions >= 0)

    def test_metrics_are_numerically_correct(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 1000, 200)
        y_pred = y_true + rng.normal(0, 50, 200)
        metrics = fv.holdout_metrics(y_true, y_pred)
        assert metrics["holdout_mae"] == pytest.approx(np.mean(np.abs(y_pred - y_true)))
        assert metrics["holdout_rmse"] == pytest.approx(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        assert metrics["holdout_median_abs_error"] == pytest.approx(
            np.median(np.abs(y_pred - y_true))
        )
        assert metrics["holdout_mean_abs_residual"] == pytest.approx(metrics["holdout_mae"])

    def test_wape_is_defined_when_the_target_has_zeros(self):
        y_true = np.array([0.0, 0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 0.0, 90.0, 210.0])
        expected = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
        assert fv.weighted_absolute_percentage_error(y_true, y_pred) == pytest.approx(expected)

    def test_wape_is_defined_when_every_target_is_zero(self):
        zeros = np.zeros(5)
        assert fv.weighted_absolute_percentage_error(zeros, zeros) == 0.0
        assert fv.weighted_absolute_percentage_error(zeros, np.ones(5)) == float("inf")

    def test_the_bias_sign_follows_the_over_estimation_convention(self):
        y_true = np.array([100.0, 100.0, 100.0])
        over = np.array([120.0, 130.0, 140.0])
        under = np.array([80.0, 70.0, 60.0])
        assert fv.holdout_metrics(y_true, over)["holdout_mean_bias"] > 0
        assert fv.holdout_metrics(y_true, under)["holdout_mean_bias"] < 0

    def test_residuals_follow_the_documented_convention(self, candidates, eval_data):
        evaluation = fv.evaluate_candidate(candidates[0], eval_data)
        expected = eval_data.y_holdout.to_numpy() - evaluation.predictions
        np.testing.assert_allclose(evaluation.residuals.to_numpy(), expected)


# ---------------------------------------------------------------------------
# Formal heteroscedasticity diagnostics
# ---------------------------------------------------------------------------


def _diagnostic_evaluation(
    residuals: np.ndarray,
    predictions: np.ndarray,
    role: str = "champion",
    estimator: str = "SyntheticRegressor",
) -> CandidateHoldoutEvaluation:
    timestamps = pd.date_range("2024-01-01", periods=len(residuals), freq="h")
    return CandidateHoldoutEvaluation(
        role=role,
        run_id=f"run_{estimator}",
        estimator=estimator,
        predictions=np.asarray(predictions, dtype=float),
        residuals=pd.Series(np.asarray(residuals, dtype=float), index=timestamps),
        metrics={},
        cv_metrics={},
    )


def _diagnostic_results(
    evaluations: list[CandidateHoldoutEvaluation],
    config: FinalValidationConfig | None = None,
) -> fv.FinalValidationResults:
    config = config or FinalValidationConfig(log_to_mlflow=False)
    return fv.FinalValidationResults(
        config=config,
        manifest={},
        candidates=[],
        data=None,
        evaluations=evaluations,
        comparison=pd.DataFrame(),
        confirmation={"decision": CONFIRMED},
        segmented={},
        predictions=pd.DataFrame(),
        manifest_fingerprint="synthetic",
        final_manifest_path=Path("synthetic.json"),
    )


def _diagnostic_results_with_predictions(
    residuals: np.ndarray,
    predictions: np.ndarray,
    estimator: str = "SyntheticRegressor",
) -> fv.FinalValidationResults:
    evaluation = _diagnostic_evaluation(residuals, predictions, estimator=estimator)
    result = _diagnostic_results([evaluation])
    timestamps = pd.DatetimeIndex(evaluation.residuals.index)
    y_true = np.asarray(predictions, dtype=float) + np.asarray(residuals, dtype=float)
    result.predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y_true,
            f"pred_{estimator}": predictions,
            f"residual_{estimator}": residuals,
        }
    )
    return result


def _strong_arch_residuals(n: int, lag: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=n)
    residuals = np.zeros(n, dtype=float)
    residuals[:lag] = innovations[:lag] * 0.3
    for index in range(lag, n):
        sigma = np.sqrt(0.05 + 0.9 * residuals[index - lag] ** 2)
        residuals[index] = sigma * innovations[index]
    return residuals


class TestHeteroscedasticityDiagnostics:
    def test_schema_holm_and_strong_variance_pattern_are_reported(self):
        n = 1_000
        rng = np.random.default_rng(11)
        predictions = np.linspace(0.0, 1.0, n)
        residuals = rng.normal(scale=0.15 + 4.0 * predictions**2, size=n)
        results = _diagnostic_results([_diagnostic_evaluation(residuals, predictions)])

        frame = fv.heteroscedasticity_diagnostics(results)

        expected_columns = {
            "role",
            "estimator",
            "test",
            "null_hypothesis",
            "statistic",
            "p_value",
            "adjusted_p_value",
            "alpha",
            "evidence_of_heteroscedasticity",
            "n_observations",
            "diagnostic_specification",
            "limitations",
            "status",
            "reason",
        }
        assert set(frame.columns) == expected_columns
        assert set(frame["test"]) == {
            "Breusch-Pagan (Koenker)",
            "White",
            "Goldfeld-Quandt",
            "Engle ARCH (lag 24)",
            "Engle ARCH (lag 168)",
        }
        applicable = frame[frame["status"] == "ok"]
        assert np.all(np.isfinite(applicable["statistic"]))
        assert np.all(np.isfinite(applicable["p_value"]))
        assert np.all(applicable["adjusted_p_value"] >= applicable["p_value"])
        assert (
            applicable["evidence_of_heteroscedasticity"]
            .eq(applicable["adjusted_p_value"] < applicable["alpha"])
            .all()
        )
        assert frame.loc[
            frame["test"].isin(["Breusch-Pagan (Koenker)", "White"]),
            "evidence_of_heteroscedasticity",
        ].all()

    def test_engle_arch_detects_daily_and_weekly_conditional_heteroscedasticity(self):
        n = 1_500
        residuals = _strong_arch_residuals(n=n, lag=24, seed=23)
        predictions = np.linspace(500.0, 900.0, n)
        results = _diagnostic_results([_diagnostic_evaluation(residuals, predictions)])

        frame = fv.heteroscedasticity_diagnostics(results)
        arch = frame[frame["test"].str.startswith("Engle ARCH")]

        assert set(arch["test"]) == {"Engle ARCH (lag 24)", "Engle ARCH (lag 168)"}
        assert arch["status"].eq("ok").all()
        assert arch["evidence_of_heteroscedasticity"].all()
        assert arch["diagnostic_specification"].str.contains("temporal order").all()

    def test_arch_uses_the_existing_temporal_order(self, monkeypatch):
        captured = []

        def fake_arch(resid, nlags=None, store=False, ddof=0):
            captured.append((np.asarray(resid).copy(), nlags, store, ddof))
            return 1.0, 0.5, 1.0, 0.5

        monkeypatch.setattr(fv, "het_arch", fake_arch)
        residuals = np.arange(600, dtype=float)
        predictions = residuals[::-1] + 10.0
        results = _diagnostic_results([_diagnostic_evaluation(residuals, predictions)])

        fv.heteroscedasticity_diagnostics(results)

        assert {item[1] for item in captured} == {24, 168}
        for resid, _, _, ddof in captured:
            np.testing.assert_array_equal(resid, residuals)
            assert ddof == 0

    def test_invalid_inputs_return_explicit_not_applicable_rows(self):
        predictions = np.linspace(0.0, 1.0, 100)
        residuals = np.ones(100)
        residuals[10] = np.nan
        results = _diagnostic_results([_diagnostic_evaluation(residuals, predictions)])

        frame = fv.heteroscedasticity_diagnostics(results)

        assert frame["status"].eq("not_applicable").all()
        assert frame["reason"].str.contains("non-finite").all()
        assert frame["p_value"].isna().all()

    def test_insufficient_samples_and_length_mismatch_are_reported(self):
        short = _diagnostic_results(
            [_diagnostic_evaluation(np.arange(20, dtype=float), np.arange(20, dtype=float))]
        )
        short_frame = fv.heteroscedasticity_diagnostics(short)
        assert short_frame["status"].eq("not_applicable").all()
        assert short_frame["reason"].str.contains("insufficient observations").all()

        mismatched = _diagnostic_evaluation(
            np.arange(100, dtype=float), np.arange(101, dtype=float)
        )
        mismatch_frame = fv.heteroscedasticity_diagnostics(_diagnostic_results([mismatched]))
        assert mismatch_frame["status"].eq("not_applicable").all()
        assert mismatch_frame["reason"].str.contains("different lengths").all()

    def test_diagnostics_do_not_touch_models_or_materialize_holdout(self, monkeypatch):
        class ExplodingModel:
            def fit(self, *args, **kwargs):
                raise AssertionError("fit was called")

            def predict(self, *args, **kwargs):
                raise AssertionError("predict was called")

        def explode_materialize(*args, **kwargs):
            raise AssertionError("holdout was materialized")

        monkeypatch.setattr(fv, "materialize_final_holdout", explode_materialize)
        n = 600
        results = _diagnostic_results(
            [_diagnostic_evaluation(np.sin(np.arange(n)), np.linspace(0.0, 1.0, n))]
        )
        results.candidates.append(ExplodingModel())

        frame = fv.heteroscedasticity_diagnostics(results)

        assert not frame.empty


# ---------------------------------------------------------------------------
# Champion residual post-holdout triage
# ---------------------------------------------------------------------------


class TestChampionResidualTriage:
    def test_diagnostic_frame_preserves_bias_residual_and_temporal_alignment(self):
        residuals = np.array([-2.0, 0.0, 3.0, 4.0])
        predictions = np.array([10.0, 20.0, 30.0, 40.0])
        results = _diagnostic_results_with_predictions(residuals, predictions)

        frame = fv.champion_residual_diagnostic_frame(results, n_deciles=4)

        np.testing.assert_allclose(frame["y_true"], predictions + residuals)
        np.testing.assert_allclose(frame["residual"], residuals)
        np.testing.assert_allclose(frame["bias"], -residuals)
        assert frame["timestamp"].is_monotonic_increasing
        assert set(frame["season"]) == {"Winter"}
        assert frame.loc[0, "hour_of_week"] == frame.loc[0, "weekday"] * 24 + frame.loc[0, "hour"]

    def test_diagnostic_frame_is_deterministic_and_does_not_mutate_inputs(self):
        residuals = np.linspace(-5.0, 5.0, 40)
        predictions = np.repeat(100.0, 40)
        results = _diagnostic_results_with_predictions(residuals, predictions)
        original_residuals = results.champion_evaluation.residuals.copy(deep=True)
        original_predictions = results.champion_evaluation.predictions.copy()

        first = fv.champion_residual_diagnostic_frame(results)
        second = fv.champion_residual_diagnostic_frame(results)

        pd.testing.assert_frame_equal(first, second)
        assert first["predicted_demand_decile"].eq("D01").all()
        pd.testing.assert_series_equal(results.champion_evaluation.residuals, original_residuals)
        np.testing.assert_array_equal(results.champion_evaluation.predictions, original_predictions)

    def test_error_profiles_have_stable_schema_and_no_local_r2(self):
        n = 240
        residuals = np.sin(np.arange(n) / 6.0) * 20.0
        predictions = np.linspace(200.0, 600.0, n)
        results = _diagnostic_results_with_predictions(residuals, predictions)

        profiles = fv.champion_error_profiles(results)
        profile = profiles["hour"]

        expected = {
            "view",
            "segment",
            "n",
            "observed_mean",
            "predicted_mean",
            "bias_mean",
            "residual_mean",
            "mae",
            "rmse",
            "residual_std",
            "overestimation_share",
            "underestimation_share",
        }
        assert set(profile.columns) == expected
        assert "r2" not in profile.columns
        assert profile["n"].sum() == n

    def test_calendar_demeaning_removes_a_synthetic_hour_of_week_pattern(self):
        n = 24 * 28
        timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
        hour_of_week = timestamps.weekday * 24 + timestamps.hour
        residuals = hour_of_week.astype(float) * 0.25
        predictions = np.repeat(500.0, n)
        results = _diagnostic_results_with_predictions(residuals, predictions)

        transformed = fv.champion_residual_transformation_frame(results)
        calendar = transformed[transformed["residual_version"] == "calendar_demeaned"]

        means = calendar.groupby("hour_of_week", observed=True)["diagnostic_residual"].mean()
        assert np.abs(means.to_numpy()).max() < 1e-10

    def test_level_standardization_reduces_pure_predicted_level_scale(self):
        n = 24 * 80
        rng = np.random.default_rng(42)
        predictions = np.linspace(100.0, 1_000.0, n)
        scale = 0.2 + predictions / predictions.max() * 6.0
        residuals = rng.normal(scale=scale, size=n)
        results = _diagnostic_results_with_predictions(residuals, predictions)

        transformed = fv.champion_residual_transformation_frame(results)

        def ratio(version: str) -> float:
            subset = transformed[transformed["residual_version"] == version]
            std = subset.groupby("predicted_demand_decile", observed=True)[
                "diagnostic_residual"
            ].std()
            return float(std.max() / std.min())

        assert ratio("level_standardized") < ratio("raw")

    def test_arch_persistence_is_detectable_after_descriptive_standardization(self):
        n = 1_500
        residuals = _strong_arch_residuals(n=n, lag=24, seed=37)
        predictions = np.linspace(400.0, 900.0, n)
        results = _diagnostic_results_with_predictions(residuals, predictions)

        diagnostics = fv.champion_residual_transformation_diagnostics(results)
        standardized = diagnostics[
            (diagnostics["residual_version"] == "level_standardized")
            & (diagnostics["arch_lag"] == 24)
        ].iloc[0]

        assert standardized["status"] == "ok"
        assert standardized["evidence_of_arch"]
        assert standardized["arch_statistic_per_observation"] > 0.0

    def test_residual_triage_uses_only_stored_results(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("workflow function was called")

        monkeypatch.setattr(fv, "materialize_final_holdout", explode)
        monkeypatch.setattr(fv, "run_final_validation", explode)
        n = 600
        residuals = np.sin(np.arange(n) / 12.0)
        predictions = np.linspace(100.0, 700.0, n)
        results = _diagnostic_results_with_predictions(residuals, predictions)

        assert not fv.champion_residual_triage(results).empty
        assert not fv.champion_error_profiles(results)["month"].empty
        assert not fv.champion_rolling_residual_diagnostics(results).empty
        assert not fv.champion_residual_transformation_diagnostics(results).empty


# ---------------------------------------------------------------------------
# No fit is ever called
# ---------------------------------------------------------------------------


class TestNoFitIsCalled:
    def test_prediction_never_triggers_fit(self, candidates, eval_data):
        candidate = candidates[0]

        def exploding_fit(*args, **kwargs):
            raise AssertionError("fit was called on a frozen candidate")

        candidate.pipeline.fit = exploding_fit
        try:
            evaluation = fv.evaluate_candidate(candidate, eval_data)
            assert evaluation.predictions.shape == (len(eval_data.X_holdout),)
        finally:
            del candidate.pipeline.fit


# ---------------------------------------------------------------------------
# Confirmation rule
# ---------------------------------------------------------------------------


def _fake_eval(role: str, estimator: str, mae: float, r2: float) -> CandidateHoldoutEvaluation:
    residuals = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])
    return CandidateHoldoutEvaluation(
        role=role,
        run_id=f"run_{estimator}",
        estimator=estimator,
        predictions=np.zeros(1),
        residuals=residuals,
        metrics={"holdout_mae": mae, "holdout_r2": r2},
        cv_metrics={},
    )


class TestConfirmationRule:
    def test_the_champion_is_confirmed_at_the_boundary(self):
        best_mae, best_r2 = 100.0, 0.90
        evals = [
            _fake_eval("champion", "CatBoostRegressor", 1.05 * best_mae, best_r2 - 0.02),
            _fake_eval("challenger", "A", best_mae, best_r2),
            _fake_eval("challenger", "B", 120.0, 0.80),
        ]
        decision = fv.decide_confirmation(evals)
        assert decision["decision"] == CONFIRMED
        assert decision["mae_condition_met"] and decision["r2_condition_met"]

    def test_the_champion_is_not_confirmed_just_past_the_mae_boundary(self):
        best_mae, best_r2 = 100.0, 0.90
        evals = [
            _fake_eval("champion", "CatBoostRegressor", 1.05 * best_mae + 0.1, best_r2),
            _fake_eval("challenger", "A", best_mae, best_r2),
            _fake_eval("challenger", "B", 120.0, 0.80),
        ]
        decision = fv.decide_confirmation(evals)
        assert decision["decision"] == NOT_CONFIRMED
        assert decision["best_holdout_estimator"] == "A"

    def test_the_champion_is_not_confirmed_just_past_the_r2_boundary(self):
        best_mae, best_r2 = 100.0, 0.90
        evals = [
            _fake_eval("champion", "CatBoostRegressor", best_mae, best_r2 - 0.02001),
            _fake_eval("challenger", "A", best_mae, best_r2),
            _fake_eval("challenger", "B", 120.0, 0.80),
        ]
        decision = fv.decide_confirmation(evals)
        assert decision["decision"] == NOT_CONFIRMED
        assert decision["independent_window_required_for_switch"]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


class TestSegmentation:
    def test_every_segmentation_preserves_the_row_count(self, candidates, eval_data, evaluations):
        config = FinalValidationConfig(log_to_mlflow=False)
        engineered = fv.engineered_holdout_frame(candidates[0], eval_data.X_holdout)
        segmented = fv.segmented_metrics(
            evaluations, engineered, eval_data.y_holdout.to_numpy(), config
        )
        assert "season" in segmented and "temperature_band" in segmented
        for frame in segmented.values():
            for estimator in frame["estimator"].unique():
                total = int(frame[frame["estimator"] == estimator]["n"].sum())
                assert total == len(eval_data.X_holdout)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shap_context(candidates, eval_data):
    """Engineered frame and deterministic SHAP sample shared by the SHAP tests."""
    config = FinalValidationConfig(log_to_mlflow=False, shap_max_sample=120)
    engineered = fv.engineered_holdout_frame(candidates[0], eval_data.X_holdout)
    sample = fv.select_shap_sample(eval_data.timestamps, engineered, config.shap_max_sample)
    explanations = [
        fv.explain_candidate(candidate, eval_data, engineered, sample, config)
        for candidate in candidates
    ]
    return config, engineered, sample, explanations


class TestShap:
    def test_the_sample_is_deterministic(self, eval_data, candidates):
        engineered = fv.engineered_holdout_frame(candidates[0], eval_data.X_holdout)
        first = fv.select_shap_sample(eval_data.timestamps, engineered, 120)
        second = fv.select_shap_sample(eval_data.timestamps, engineered, 120)
        np.testing.assert_array_equal(first, second)

    def test_the_three_candidates_share_the_same_rows(self, shap_context):
        _, _, sample, explanations = shap_context
        for explanation in explanations:
            np.testing.assert_array_equal(explanation.sample_positions, sample)

    def test_the_names_have_the_width_of_the_matrix(self, shap_context):
        _, _, _, explanations = shap_context
        for explanation in explanations:
            assert len(explanation.feature_names) == explanation.matrix.shape[1]
            assert len(explanation.feature_sources) == explanation.matrix.shape[1]

    def test_grouping_preserves_the_per_row_sum(self, shap_context):
        _, _, _, explanations = shap_context
        explanation = explanations[0]
        sources = np.array(explanation.feature_sources)
        grouped = explanation.grouped_importance.set_index("feature")["mean_abs_shap"]
        # A cyclic feature really does expand into several columns here, so this
        # is not a degenerate one-to-one grouping.
        assert (pd.Series(explanation.feature_sources).value_counts() > 1).any()
        for source in set(explanation.feature_sources):
            columns = np.where(sources == source)[0]
            # Summing per observation before taking the absolute value is the
            # invariant: it must equal the grouped importance the module reports.
            expected = float(np.mean(np.abs(explanation.shap_values[:, columns].sum(axis=1))))
            assert grouped.loc[source] == pytest.approx(expected)

    def test_additivity_is_verified(self, shap_context):
        config, _, _, explanations = shap_context
        for explanation in explanations:
            assert explanation.additivity_max_error < config.shap_atol * 100 + 1e-6

    def test_the_trend_residual_reconstruction_is_verified(self, shap_context):
        config, _, _, explanations = shap_context
        for explanation in explanations:
            assert explanation.reconstruction_max_error < 1e-4

    def test_a_non_identity_target_transformer_is_rejected(self):
        from sklearn.preprocessing import FunctionTransformer

        with pytest.raises(ValueError):
            fv.assert_identity_target_transformer(FunctionTransformer(func=np.log1p))

    def test_an_identity_target_transformer_is_accepted(self):
        from sklearn.preprocessing import FunctionTransformer

        fv.assert_identity_target_transformer(FunctionTransformer())

    def test_champion_local_examples_span_median_under_and_over(self, candidates, eval_data):
        evaluation = fv.evaluate_candidate(candidates[0], eval_data)
        sample = np.arange(len(eval_data.X_holdout))
        examples = fv.local_example_positions(evaluation, sample)
        assert set(examples) == {
            "median_abs_error",
            "largest_underestimation",
            "largest_overestimation",
        }


# ---------------------------------------------------------------------------
# Persistence, idempotency and experiment isolation
# ---------------------------------------------------------------------------


class TestPersistenceAndIdempotency:
    def test_partial_artifacts_are_not_overwritten(self, tmp_path):
        config = FinalValidationConfig(runtime_root=tmp_path / "run", log_to_mlflow=False)
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "holdout_predictions.csv").write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            fv._guard_partial_artifacts(config)

    def test_an_idempotent_rerun_reuses_the_complete_result(
        self, monkeypatch, tmp_path, candidates, manifest, eval_data
    ):
        config = FinalValidationConfig(runtime_root=tmp_path / "run", log_to_mlflow=False)
        plan = FinalValidationPlan(config, manifest, candidates, fv.manifest_fingerprint(manifest))
        monkeypatch.setattr(fv, "materialize_final_holdout", lambda cfg: eval_data)

        first = fv.run_final_validation(config, plan=plan)
        assert not first.loaded_from_cache
        assert config.final_manifest_path.exists()

        second = fv.run_final_validation(config, plan=plan)
        assert second.loaded_from_cache
        assert second.decision == first.decision
        assert list(second.comparison["estimator"]) == list(first.comparison["estimator"])

    def test_an_incompatible_cache_is_refused(
        self, monkeypatch, tmp_path, candidates, manifest, eval_data
    ):
        config = FinalValidationConfig(runtime_root=tmp_path / "run", log_to_mlflow=False)
        plan = FinalValidationPlan(config, manifest, candidates, fv.manifest_fingerprint(manifest))
        monkeypatch.setattr(fv, "materialize_final_holdout", lambda cfg: eval_data)
        fv.run_final_validation(config, plan=plan)

        import json

        cached = json.loads(config.final_manifest_path.read_text(encoding="utf-8"))
        cached["dataset_fingerprint"] = "0000000000000000"
        config.final_manifest_path.write_text(json.dumps(cached), encoding="utf-8")
        with pytest.raises(ValueError):
            fv.run_final_validation(config, plan=plan)

    def test_the_final_experiment_is_separate_from_selection(self):
        from src.tracking import EXPERIMENT_NAME_V4

        assert fv.FINAL_EXPERIMENT_NAME == "bike_sharing_demand_v4_final_validation"
        assert fv.FINAL_EXPERIMENT_NAME != EXPERIMENT_NAME_V4
