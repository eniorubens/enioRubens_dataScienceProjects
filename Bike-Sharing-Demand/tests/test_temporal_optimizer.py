"""Tests for src/temporal_optimizer.py — the temporal orchestration layer.

The dynamic pipeline contract itself is covered by
``tests/test_modeling_pipeline.py``; what is asserted here is the discipline
this module adds on top of it: no holdout can reach the optimizer, the dataset
fingerprint reacts to any change in the data, smoke studies are capped in
total rather than per call, and the blocklist isolates pipeline *shapes*
rather than numeric hyperparameters.

Most tests avoid real subprocess-guarded cross_validate calls (one process
spawn per trial) by calling leaf functions with ``optuna.trial.FixedTrial`` or
by monkeypatching the timeout helpers with fast fakes. The few that do run a
real ``optimize()`` always use ``trials<=2``, mirroring the convention in
``tests/test_optimizer.py``.
"""

from __future__ import annotations

import pickle

import numpy as np
import optuna
import pandas as pd
import pytest

import src.temporal_optimizer as topt
from src.modeling_pipeline import PipelineSpec
from src.temporal_optimizer import (
    MAX_SMOKE_TRIALS,
    RUN_MODE_FULL,
    RUN_MODE_SMOKE,
    TemporalRegressionOptimizer,
    dataset_fingerprint,
)


def _fixed(params: dict) -> optuna.trial.FixedTrial:
    return optuna.trial.FixedTrial(params)


def _ridge_trial(**overrides) -> optuna.trial.FixedTrial:
    params = {
        "modeler_name": "linear_modeling",
        "encoder": "OrdinalEncoder",
        "alpha": 1.0,
        "selector": "NoSelector",
        "target_transform": "none",
    }
    params.update(overrides)
    return _fixed(params)


def _spec(**overrides) -> PipelineSpec:
    payload = {
        "estimator": "Ridge",
        "family": "linear",
        "modeler_name": "linear_modeling",
        "encoder": "OrdinalEncoder",
        "scaler": "StandardScaler",
        "selector": "NoSelector",
        "target_transform": "none",
    }
    payload.update(overrides)
    return PipelineSpec(**payload)


@pytest.fixture
def dev_split(dev_split_v4):
    return dev_split_v4


def _fake_folds(
    mae=(10.0, 12.0),
    rmse=(15.0, 16.0),
    r2=(-5.0, -3.0),
    train_r2=(0.8, 0.9),
    best_iteration=(None, None),
    wape=(0.1, 0.2),
    mean_bias=(-3.0, 1.0),
):
    """Fold rows shaped exactly like ``temporal_cv_fold_results``' output."""
    return [
        {
            "fold": index + 1,
            "test_year": 2019 + index,
            "fold_role": "selection",
            "n_train": 100,
            "n_train_excluded": 0,
            "n_test": 20,
            "n_selection_test": 20,
            "best_iteration": best_iteration[index],
            "iteration_ceiling": None,
            "best_iteration_cap_hit": False,
            "mae": mae[index],
            "rmse": rmse[index],
            "r2": r2[index],
            "wape": wape[index],
            "mean_bias": mean_bias[index],
            "selection_mae": mae[index],
            "selection_rmse": rmse[index],
            "selection_r2": r2[index],
            "selection_wape": wape[index],
            "selection_mean_bias": mean_bias[index],
            "train_r2": train_r2[index],
        }
        for index in range(len(mae))
    ]


# ---------------------------------------------------------------------------
# (h) The holdout cannot reach the optimizer, by API rather than by convention
# ---------------------------------------------------------------------------


class TestNoHoldoutParameter:
    def test_constructor_rejects_x_holdout_kwarg(self, dev_split):
        X_dev, y_dev, splitter = dev_split
        with pytest.raises(TypeError):
            TemporalRegressionOptimizer(
                "Ridge", X_dev, y_dev, splitter, X_holdout=X_dev, y_holdout=y_dev
            )

    def test_no_holdout_named_parameter_exists_at_all(self):
        import inspect

        parameters = inspect.signature(TemporalRegressionOptimizer.__init__).parameters
        assert not any("holdout" in name.lower() for name in parameters)

    def test_no_module_level_data_globals(self):
        for name in ("max_label", "X_train_opt", "y_train_opt", "ts_cv", "X_holdout", "y_holdout"):
            assert not hasattr(topt, name)

    def test_split_dev_holdout_never_returns_holdout_rows(self, raw_v4_df):
        from src.cv import split_dev_holdout

        X_dev, y_dev, summary = split_dev_holdout(raw_v4_df)
        assert summary.n_rows > 0 and summary.sealed
        assert not hasattr(summary, "X")
        assert not hasattr(summary, "y")
        timestamps = pd.to_datetime(X_dev["DateTime"])
        assert timestamps.max() < pd.Timestamp("2023-12-01")


# ---------------------------------------------------------------------------
# (i) Dataset fingerprint reacts to any change in the development data
# ---------------------------------------------------------------------------


class TestDatasetFingerprint:
    def test_is_stable_for_identical_data(self, dev_split):
        X_dev, y_dev, _ = dev_split
        assert dataset_fingerprint(X_dev, y_dev) == dataset_fingerprint(X_dev, y_dev)

    def test_changing_one_feature_value_changes_the_fingerprint(self, dev_split):
        """Shape, column names and the target all stay identical — only a
        single cell of X moves."""
        X_dev, y_dev, _ = dev_split
        mutated = X_dev.copy()
        column = "Temperature(C)"
        mutated.iloc[0, mutated.columns.get_loc(column)] += 0.001

        assert mutated.shape == X_dev.shape
        assert list(mutated.columns) == list(X_dev.columns)
        assert dataset_fingerprint(mutated, y_dev) != dataset_fingerprint(X_dev, y_dev)

    def test_changing_the_target_changes_the_fingerprint(self, dev_split):
        X_dev, y_dev, _ = dev_split
        assert dataset_fingerprint(X_dev, y_dev + 1) != dataset_fingerprint(X_dev, y_dev)

    def test_reordering_rows_changes_the_fingerprint(self, dev_split):
        X_dev, y_dev, _ = dev_split
        reordered = X_dev.iloc[::-1]
        assert dataset_fingerprint(reordered, y_dev.iloc[::-1]) != dataset_fingerprint(X_dev, y_dev)

    def test_changing_a_dtype_changes_the_fingerprint(self, dev_split):
        X_dev, y_dev, _ = dev_split
        retyped = X_dev.copy()
        retyped["Humidity(%)"] = retyped["Humidity(%)"].astype("float32")
        assert dataset_fingerprint(retyped, y_dev) != dataset_fingerprint(X_dev, y_dev)


# ---------------------------------------------------------------------------
# Metrics stay in the original units and keep their sign
# ---------------------------------------------------------------------------


class TestMetricsScaleAndSign:
    def test_metrics_are_in_original_bike_count_units(self, dev_split):
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer("Ridge", X_dev, y_dev, splitter, trials=1)
        pipeline, _ = optimizer.build_pipeline(_ridge_trial())
        train_idx, test_idx = next(splitter.split(X_dev))
        pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        y_pred = pipeline.predict(X_dev.iloc[test_idx])

        # Target is uniform on [0, 1000); a sane fit's MAE lands in the
        # hundreds, not as a 0-1 fraction or a x100 percentage artefact.
        mae = float(np.mean(np.abs(y_dev.iloc[test_idx].to_numpy() - y_pred)))
        assert 1.0 < mae < 1000.0

    def test_negative_r2_is_not_converted_to_positive(self, dev_split, monkeypatch, tmp_path):
        X_dev, y_dev, splitter = dev_split
        monkeypatch.setattr(topt, "temporal_cv_with_timeout", lambda *a, **k: (_fake_folds(), None))
        optimizer = TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            trials=1,
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        trial = _ridge_trial()
        mae = optimizer.objective(trial)
        assert mae == pytest.approx(11.0)
        assert trial.user_attrs["cv_r2_mean"] == pytest.approx(-4.0)
        assert trial.user_attrs["cv_r2_mean"] < 0

    def test_pipeline_choices_are_recorded_as_trial_attributes(
        self, dev_split, monkeypatch, tmp_path
    ):
        X_dev, y_dev, splitter = dev_split
        monkeypatch.setattr(topt, "temporal_cv_with_timeout", lambda *a, **k: (_fake_folds(), None))
        optimizer = TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            trials=1,
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        trial = _ridge_trial(modeler_name="Sin_Cos", encoder="MeanEncoder")
        optimizer.objective(trial)
        assert trial.user_attrs["modeler_name"] == "Sin_Cos"
        assert trial.user_attrs["encoder"] == "MeanEncoder"
        assert trial.user_attrs["selector"] == "NoSelector"


# ---------------------------------------------------------------------------
# One fold protocol: search, diagnostics and artifact cannot diverge
# ---------------------------------------------------------------------------


def _booster_trial(estimator: str) -> optuna.trial.FixedTrial:
    params = {
        "modeler_name": "linear_modeling",
        "encoder": "OrdinalEncoder",
        "standardizer": "StandardScaler",
        "selector": "NoSelector",
        "target_transform": "none",
    }
    if estimator == "XGBRegressor":
        params.update(
            {
                # n_estimators is deliberately absent: it is pinned to
                # BOOSTING_CEILING and no longer drawn by the search.
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0.0,
                "min_child_weight": 1.0,
                "reg_alpha": 0.01,
                "reg_lambda": 0.1,
            }
        )
    else:
        params.update(
            {
                "num_leaves": 31,
                "learning_rate": 0.1,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.01,
                "reg_lambda": 0.01,
                "max_depth": -1,
            }
        )
    return _fixed(params)


class TestSingleFoldProtocol:
    """The objective and the refit must run the *same* procedure.

    Before this was unified, XGBoost and LightGBM were scored through an
    early-stopping carve during the search and then refit with their full
    sampled ``n_estimators`` — so the reported MAE described a model that was
    never the one persisted.
    """

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor"])
    def test_boosters_route_through_the_early_stopping_branch(
        self, dev_split, monkeypatch, tmp_path, estimator
    ):
        X_dev, y_dev, splitter = dev_split
        seen = {}

        def fake_cv(pipeline, X, y, cv, timeout, early_stopping=False, **kwargs):
            seen["early_stopping"] = early_stopping
            return _fake_folds(), None

        monkeypatch.setattr(topt, "temporal_cv_with_timeout", fake_cv)
        optimizer = TemporalRegressionOptimizer(
            estimator,
            X_dev,
            y_dev,
            splitter,
            trials=1,
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        assert optimizer.uses_early_stopping is True
        optimizer.objective(_booster_trial(estimator))
        assert seen["early_stopping"] is True

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor"])
    def test_refined_fixed_budget_bypasses_early_stopping(
        self, dev_split, monkeypatch, tmp_path, estimator
    ):
        X_dev, y_dev, splitter = dev_split
        seen = {}

        def fake_cv(pipeline, X, y, cv, timeout, early_stopping=False, **kwargs):
            seen["early_stopping"] = early_stopping
            return _fake_folds(), None

        params = {
            "modeler_name": "linear_modeling",
            "encoder": "OrdinalEncoder",
            "standardizer": "StandardScaler",
            "selector": "NoSelector",
            "boosting_budget_strategy": "fixed_iterations",
            "fixed_iterations": 283,
            "target_strategy": "direct",
        }
        if estimator == "XGBRegressor":
            params.update(
                {
                    "modeler_name": "linear_modeling",
                    "encoder": "JamesSteinEncoder",
                    "loss_function": "reg:squarederror",
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "gamma": 0.0,
                    "min_child_weight": 1.0,
                    "reg_alpha": 0.01,
                    "reg_lambda": 0.1,
                }
            )
        else:
            params.update(
                {
                    "modeler_name": "linear_modeling",
                    "encoder": "CountFrequencyEncoder",
                    "loss_function": "regression",
                    "num_leaves": 31,
                    "learning_rate": 0.1,
                    "min_child_samples": 20,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.01,
                    "reg_lambda": 0.01,
                    "max_depth": -1,
                }
            )

        monkeypatch.setattr(topt, "temporal_cv_with_timeout", fake_cv)
        optimizer = TemporalRegressionOptimizer(
            estimator,
            X_dev,
            y_dev,
            splitter,
            trials=1,
            search_profile="refined",
            target_strategy="auto",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        optimizer.objective(_fixed(params))
        assert seen["early_stopping"] is False

    def test_ridge_does_not_use_the_early_stopping_branch(self, dev_split, monkeypatch, tmp_path):
        X_dev, y_dev, splitter = dev_split
        seen = {}

        def fake_cv(pipeline, X, y, cv, timeout, early_stopping=False, **kwargs):
            seen["early_stopping"] = early_stopping
            return _fake_folds(), None

        monkeypatch.setattr(topt, "temporal_cv_with_timeout", fake_cv)
        optimizer = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, trials=1, invalid_configs_path=tmp_path / "invalid.csv"
        )
        assert optimizer.uses_early_stopping is False
        optimizer.objective(_ridge_trial())
        assert seen["early_stopping"] is False

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor", "Ridge"])
    def test_objective_and_evaluate_best_call_the_same_function(self, estimator):
        """Neither path may have a private evaluation route of its own."""
        import inspect

        objective_source = inspect.getsource(TemporalRegressionOptimizer.objective)
        evaluate_source = inspect.getsource(TemporalRegressionOptimizer.evaluate_best)
        assert "temporal_cv_with_timeout" in objective_source
        assert "temporal_cv_fold_results" in evaluate_source
        # temporal_cv_with_timeout is a thin subprocess wrapper around
        # temporal_cv_fold_results, so both routes execute the same body.
        wrapper_source = inspect.getsource(topt._run_temporal_cv_in_subprocess)
        assert "temporal_cv_fold_results" in wrapper_source

    def test_the_objective_value_is_the_mean_of_the_reported_folds(
        self, dev_split, monkeypatch, tmp_path
    ):
        X_dev, y_dev, splitter = dev_split
        monkeypatch.setattr(
            topt,
            "temporal_cv_with_timeout",
            lambda *a, **k: (_fake_folds(mae=(10.0, 12.0)), None),
        )
        optimizer = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, trials=1, invalid_configs_path=tmp_path / "invalid.csv"
        )
        trial = _ridge_trial()
        value = optimizer.objective(trial)
        assert value == pytest.approx(11.0)
        assert trial.user_attrs["fold_mae"] == [10.0, 12.0]

    def test_hgb_internal_early_stopping_stays_disabled(self):
        from src.modeling_pipeline import get_parameters

        params = get_parameters(
            "HistGradientBoostingRegressor",
            _fixed(
                {
                    "learning_rate": 0.1,
                    "max_iter": 100,
                    "max_leaf_nodes": 31,
                    "max_depth": 5,
                    "min_samples_leaf": 20,
                    "l2_regularization": 0.01,
                }
            ),
        )
        assert params["early_stopping"] is False
        assert "validation_fraction" not in params


# ---------------------------------------------------------------------------
# Invalid-config blocklist: keyed on shape, isolated by provenance
# ---------------------------------------------------------------------------


class TestBlocklistIsolation:
    def _optimizer(self, dev_split, tmp_path, **kwargs):
        X_dev, y_dev, splitter = dev_split
        return TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            trials=1,
            invalid_configs_path=tmp_path / "invalid.csv",
            **kwargs,
        )

    def test_signature_carries_every_provenance_key(self, dev_split, tmp_path):
        optimizer = self._optimizer(dev_split, tmp_path)
        signature = optimizer._config_signature(_spec())
        assert signature["dataset_fingerprint"] == optimizer.dataset_fingerprint
        assert signature["cv_version"] == topt.CV_STRATEGY_VERSION
        assert signature["code_version"] == topt.CODE_VERSION
        assert signature["run_mode"] == optimizer.run_mode
        assert signature["modeler_name"] == "linear_modeling"

    def test_numeric_only_failure_does_not_blocklist_a_whole_structure(self, dev_split, tmp_path):
        """The blocklist is keyed on pipeline shape, so a failure recorded for
        one (modeler, encoder, selector) triple must not be found under
        another."""
        optimizer = self._optimizer(dev_split, tmp_path)
        optimizer._register_invalid_config(_spec(), reason="boom")
        assert optimizer._was_invalid_config(_spec()) is True
        assert optimizer._was_invalid_config(_spec(encoder="MeanEncoder")) is False
        assert optimizer._was_invalid_config(_spec(selector="SelectKBest")) is False
        assert optimizer._was_invalid_config(_spec(modeler_name="Sin_Cos")) is False

    def test_high_dimensional_no_selector_combination_is_rejected(
        self, dev_split, tmp_path, monkeypatch
    ):
        called = {"cv": 0}
        monkeypatch.setattr(
            topt,
            "temporal_cv_with_timeout",
            lambda *a, **k: (called.update(cv=called["cv"] + 1), (_fake_folds(), None))[1],
        )
        optimizer = self._optimizer(dev_split, tmp_path)
        trial = _ridge_trial(modeler_name="Polynomial", normalizer="MinMaxScaler")
        assert optimizer.objective(trial) == float("inf")
        assert "NoSelector is not allowed" in trial.user_attrs["failed_reason"]
        assert called["cv"] == 0


# ---------------------------------------------------------------------------
# (j) Run modes: separate studies, and a smoke ceiling that survives re-runs
# ---------------------------------------------------------------------------


class TestRunModes:
    def _kwargs(self, tmp_path):
        return dict(
            trial_timeout=180,
            study_timeout=900,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )

    def test_run_mode_is_validated(self, dev_split):
        X_dev, y_dev, splitter = dev_split
        with pytest.raises(ValueError, match="run_mode"):
            TemporalRegressionOptimizer("Ridge", X_dev, y_dev, splitter, run_mode="quick")

    def test_smoke_and_full_use_different_studies_and_storages(self, dev_split, tmp_path):
        X_dev, y_dev, splitter = dev_split
        smoke = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, run_mode=RUN_MODE_SMOKE, **self._kwargs(tmp_path)
        )
        full = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, run_mode=RUN_MODE_FULL, **self._kwargs(tmp_path)
        )
        assert smoke.study_name != full.study_name
        assert smoke.storage_url != full.storage_url
        assert RUN_MODE_SMOKE in smoke.study_name

    def test_the_study_identity_includes_the_environment(self, dev_split, tmp_path):
        """A study built under one numerical stack must never absorb trials
        measured under another; the environment is part of its name."""
        from src.environment import environment_fingerprint

        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, **self._kwargs(tmp_path)
        )
        assert environment_fingerprint() in optimizer.study_name
        assert optimizer.environment_fingerprint == environment_fingerprint()

    def test_the_study_identity_includes_the_current_code_version(self, dev_split, tmp_path):
        """The bump that invalidated every run made under the previous version
        has to be visible in the study name, or a stale study would be resumed."""
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, **self._kwargs(tmp_path)
        )
        assert topt.CODE_VERSION == "temporal_optimizer_v7"
        assert topt.CODE_VERSION in optimizer.study_name

    def test_fold_weights_are_part_of_the_study_identity(self, dev_split, tmp_path):
        X_dev, y_dev, splitter = dev_split
        equal = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, **self._kwargs(tmp_path)
        )
        recent = TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            fold_weights=(1.0, 1.0, 1.0, 2.0, 3.0),
            **self._kwargs(tmp_path),
        )
        assert equal.cv_fingerprint != recent.cv_fingerprint
        assert equal.study_name != recent.study_name

    def test_smoke_trials_are_capped_at_construction(self, dev_split, tmp_path):
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=500,
            **self._kwargs(tmp_path),
        )
        assert optimizer.trials == MAX_SMOKE_TRIALS

    def test_smoke_total_stays_capped_across_re_executions(self, dev_split, tmp_path):
        """Re-running the notebook must top the study up to the ceiling, not
        append another batch of trials to it."""
        X_dev, y_dev, splitter = dev_split
        kwargs = self._kwargs(tmp_path)

        first = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, run_mode=RUN_MODE_SMOKE, trials=2, **kwargs
        )
        study_one = first.optimize()
        assert len(study_one.trials) == MAX_SMOKE_TRIALS

        second = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, run_mode=RUN_MODE_SMOKE, trials=2, **kwargs
        )
        assert second.remaining_trials(study_one) == 0
        study_two = second.optimize()
        assert len(study_two.trials) == MAX_SMOKE_TRIALS
        assert first.study_name == second.study_name

    def test_partial_study_is_topped_up_rather_than_restarted(self, dev_split, tmp_path):
        X_dev, y_dev, splitter = dev_split
        kwargs = self._kwargs(tmp_path)
        first = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, run_mode=RUN_MODE_SMOKE, trials=1, **kwargs
        )
        first.optimize()
        second = TemporalRegressionOptimizer(
            "Ridge", X_dev, y_dev, splitter, run_mode=RUN_MODE_SMOKE, trials=2, **kwargs
        )
        study = second.optimize()
        assert len(study.trials) == 2

    def test_waiting_seed_trials_do_not_pretend_to_be_completed(self, dev_split, tmp_path):
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=2,
            **self._kwargs(tmp_path),
        )
        study = optuna.create_study(direction="minimize")
        study.enqueue_trial({"alpha": 1.0})
        study.enqueue_trial({"alpha": 2.0})

        assert [trial.state.name for trial in study.trials] == ["WAITING", "WAITING"]
        assert optimizer.remaining_trials(study) == 2

        study.optimize(lambda trial: 1.0, n_trials=1)
        assert optimizer.remaining_trials(study) == 1


# ---------------------------------------------------------------------------
# Study health and artifact reusability
# ---------------------------------------------------------------------------


class TestStudyLifecycle:
    def test_all_trials_returning_inf_raises(self, dev_split, monkeypatch, tmp_path):
        X_dev, y_dev, splitter = dev_split
        monkeypatch.setattr(
            topt, "temporal_cv_with_timeout", lambda *a, **k: (None, "synthetic failure")
        )
        optimizer = TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=1,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        with pytest.raises(RuntimeError, match="no valid configuration"):
            optimizer.optimize()

    def test_winning_pipeline_is_picklable_and_reusable(self, dev_split, tmp_path):
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=2,
            trial_timeout=180,
            study_timeout=900,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        study = optimizer.optimize()
        evaluation = optimizer.evaluate_best(study)

        reloaded = pickle.loads(pickle.dumps(evaluation.fitted_pipeline))
        predictions = reloaded.predict(X_dev.iloc[:5])
        assert len(predictions) == 5
        assert not evaluation.fold_metrics.empty
        assert evaluation.spec.modeler_name in optimizer.modeler_space
        assert evaluation.spec.n_features_selected is not None
        assert evaluation.cv_metrics["cv_mae_mean"] == pytest.approx(study.best_value)

    def test_best_params_carry_the_dynamic_choices(self, dev_split, tmp_path):
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=2,
            trial_timeout=180,
            study_timeout=900,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        study = optimizer.optimize()
        assert "modeler_name" in study.best_params
        assert "encoder" in study.best_params
        assert "selector" in study.best_params
        assert "target_transform" in study.best_params


# ---------------------------------------------------------------------------
# Boosting budget: discovered per fold, aggregated once, carried by the artifact
# ---------------------------------------------------------------------------


class TestIterationBudget:
    def test_aggregation_rule_is_the_documented_median(self):
        from src.temporal_optimizer import ITERATION_AGGREGATION, aggregate_iteration_budget

        assert ITERATION_AGGREGATION == "median"
        assert aggregate_iteration_budget([10, 20, 30]) == 20
        assert aggregate_iteration_budget([10, 20, 30, 1000]) == 25

    def test_folds_without_a_budget_leave_the_sampled_value_alone(self):
        from src.temporal_optimizer import aggregate_iteration_budget

        assert aggregate_iteration_budget([None, None]) is None
        assert aggregate_iteration_budget([]) is None

    def test_setting_a_budget_is_a_no_op_for_estimators_without_one(self, dev_split):
        from src.temporal_optimizer import set_iteration_budget

        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer("Ridge", X_dev, y_dev, splitter, trials=1)
        pipeline, _ = optimizer.build_pipeline(_ridge_trial())
        assert set_iteration_budget(pipeline, 42) is None

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor"])
    def test_setting_a_budget_pins_the_boosting_estimator(self, dev_split, estimator):
        from src.temporal_optimizer import set_iteration_budget

        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(estimator, X_dev, y_dev, splitter, trials=1)
        pipeline, _ = optimizer.build_pipeline(_booster_trial(estimator))
        assert set_iteration_budget(pipeline, 42) == 42
        assert pipeline.named_steps["regressor"].regressor.n_estimators == 42

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor", "Ridge"])
    def test_search_metric_equals_the_mean_of_the_reported_folds(
        self, dev_split, tmp_path, estimator
    ):
        """The requirement in one assertion, run end to end for real."""
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            estimator,
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=2,
            trial_timeout=600,
            study_timeout=1800,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        study = optimizer.optimize()
        evaluation = optimizer.evaluate_best(study)
        assert evaluation.fold_metrics["mae"].mean() == pytest.approx(study.best_value, rel=1e-6)

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor"])
    def test_the_persisted_pipeline_carries_the_aggregated_budget(
        self, dev_split, tmp_path, estimator
    ):
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            estimator,
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=2,
            trial_timeout=600,
            study_timeout=1800,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        study = optimizer.optimize()
        evaluation = optimizer.evaluate_best(study)

        assert len(evaluation.best_iterations_by_fold) == splitter.get_n_splits(X_dev)
        assert evaluation.final_n_estimators is not None
        assert evaluation.iteration_aggregation == "median"
        regressor = evaluation.fitted_pipeline.named_steps["regressor"].regressor
        assert regressor.n_estimators == evaluation.final_n_estimators
        # And it is the folds' aggregate, not the raw sampled value.
        from src.temporal_optimizer import aggregate_iteration_budget

        assert evaluation.final_n_estimators == aggregate_iteration_budget(
            evaluation.best_iterations_by_fold
        )

    def test_a_non_boosting_estimator_reports_no_budget(self, dev_split, tmp_path):
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            "Ridge",
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=2,
            trial_timeout=600,
            study_timeout=1800,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        evaluation = optimizer.evaluate_best(optimizer.optimize())
        assert evaluation.final_n_estimators is None
        assert set(evaluation.best_iterations_by_fold) == {None}


# ---------------------------------------------------------------------------
# The boosting ceiling and the detection of a truncated fit
# ---------------------------------------------------------------------------


class TestTheBoostingCeilingIsHighAndFixed:
    """LightGBM never sampled ``n_estimators`` at all, so it ran at the library
    default of 100 and a fold stopped at 98 — a number that says where the
    budget ended, not where the validation loss stopped improving. XGBoost had
    the same defect in a subtler form: whatever the search drew became the
    ceiling early stopping could not pass."""

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor"])
    def test_the_ceiling_is_pinned_high_for_both_boosters(self, estimator):
        from src.modeling_pipeline import BOOSTING_CEILING, get_parameters

        params = get_parameters(estimator, _booster_trial(estimator))
        assert params["n_estimators"] == BOOSTING_CEILING
        assert BOOSTING_CEILING >= 2000

    def test_lightgbm_is_no_longer_silently_capped_at_one_hundred(self):
        from lightgbm import LGBMRegressor

        from src.modeling_pipeline import get_parameters

        params = get_parameters("LGBMRegressor", _booster_trial("LGBMRegressor"))
        assert params["n_estimators"] != LGBMRegressor().n_estimators
        assert params["n_estimators"] > 100

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor"])
    def test_the_search_no_longer_draws_the_ceiling(self, dev_split, tmp_path, estimator):
        """A sampled ceiling is a sampled truncation; the dimension is gone."""
        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(
            estimator,
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=2,
            trial_timeout=600,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        study = optimizer.optimize()
        assert "n_estimators" not in study.best_params

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor"])
    def test_the_pipeline_reports_the_ceiling_it_will_run_under(self, dev_split, estimator):
        from src.modeling_pipeline import BOOSTING_CEILING
        from src.temporal_optimizer import iteration_ceiling

        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer(estimator, X_dev, y_dev, splitter, trials=1)
        pipeline, _ = optimizer.build_pipeline(_booster_trial(estimator))
        assert iteration_ceiling(pipeline) == BOOSTING_CEILING

    def test_an_estimator_without_a_boosting_budget_reports_no_ceiling(self, dev_split):
        from src.temporal_optimizer import iteration_ceiling

        X_dev, y_dev, splitter = dev_split
        optimizer = TemporalRegressionOptimizer("Ridge", X_dev, y_dev, splitter, trials=1)
        pipeline, _ = optimizer.build_pipeline(_ridge_trial())
        assert iteration_ceiling(pipeline) is None


class TestCapHitDetection:
    def test_a_budget_at_ninety_per_cent_of_the_ceiling_counts_as_a_hit(self):
        from src.modeling_pipeline import BOOSTING_CAP_RATIO
        from src.temporal_optimizer import hit_iteration_ceiling

        assert BOOSTING_CAP_RATIO == 0.9
        assert hit_iteration_ceiling(98, 100) is True
        assert hit_iteration_ceiling(90, 100) is True
        assert hit_iteration_ceiling(89, 100) is False
        assert hit_iteration_ceiling(1800, 2000) is True

    def test_a_missing_budget_or_ceiling_is_never_a_hit(self):
        from src.temporal_optimizer import hit_iteration_ceiling

        assert hit_iteration_ceiling(None, 2000) is False
        assert hit_iteration_ceiling(1900, None) is False
        assert hit_iteration_ceiling(0, 2000) is False

    def test_the_summary_counts_folds_and_flags_systematic_truncation(self):
        from src.temporal_optimizer import summarize_iteration_truncation

        folds = [
            {"best_iteration": 1900, "iteration_ceiling": 2000, "best_iteration_cap_hit": True},
            {"best_iteration": 1950, "iteration_ceiling": 2000, "best_iteration_cap_hit": True},
            {"best_iteration": 120, "iteration_ceiling": 2000, "best_iteration_cap_hit": False},
        ]
        summary = summarize_iteration_truncation(folds)
        assert summary["n_folds_cap_hit"] == 2
        assert summary["n_folds_with_budget"] == 3
        assert summary["iteration_ceiling"] == 2000
        assert summary["cap_hits_by_fold"] == [True, True, False]
        assert summary["systematic"] is True

    def test_one_isolated_fold_at_the_ceiling_is_not_systematic(self):
        """An isolated fold reaching the ceiling is a plausible property of that
        year's data; half of them reaching it says the ceiling is choosing the
        model size."""
        from src.temporal_optimizer import summarize_iteration_truncation

        folds = [
            {"best_iteration": 1900, "iteration_ceiling": 2000, "best_iteration_cap_hit": True},
            {"best_iteration": 100, "iteration_ceiling": 2000, "best_iteration_cap_hit": False},
            {"best_iteration": 150, "iteration_ceiling": 2000, "best_iteration_cap_hit": False},
        ]
        assert summarize_iteration_truncation(folds)["systematic"] is False

    def test_folds_without_any_budget_are_not_truncated(self):
        from src.temporal_optimizer import summarize_iteration_truncation

        summary = summarize_iteration_truncation(_fake_folds())
        assert summary["n_folds_with_budget"] == 0
        assert summary["systematic"] is False

    @pytest.mark.parametrize("estimator", ["XGBRegressor", "LGBMRegressor"])
    def test_a_real_run_records_the_ceiling_per_fold(self, dev_split, tmp_path, estimator):
        X_dev, y_dev, splitter = dev_split
        from src.modeling_pipeline import BOOSTING_CEILING

        optimizer = TemporalRegressionOptimizer(
            estimator,
            X_dev,
            y_dev,
            splitter,
            run_mode=RUN_MODE_SMOKE,
            trials=2,
            trial_timeout=600,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        evaluation = optimizer.evaluate_best(optimizer.optimize())
        assert evaluation.iteration_ceiling == BOOSTING_CEILING
        assert len(evaluation.cap_hits_by_fold) == splitter.get_n_splits(X_dev)
        assert evaluation.n_folds_with_budget == splitter.get_n_splits(X_dev)
        assert set(evaluation.fold_metrics["iteration_ceiling"]) == {BOOSTING_CEILING}


# ---------------------------------------------------------------------------
# Termination reason: which limit actually stopped the study
# ---------------------------------------------------------------------------


class TestTerminationReason:
    def _optimizer(self, dev_split, tmp_path, **kwargs):
        X_dev, y_dev, splitter = dev_split
        params = dict(
            run_mode=RUN_MODE_SMOKE,
            trials=2,
            trial_timeout=600,
            study_timeout=1800,
            studies_dir=tmp_path / "studies",
            invalid_configs_path=tmp_path / "invalid.csv",
        )
        params.update(kwargs)
        return TemporalRegressionOptimizer("Ridge", X_dev, y_dev, splitter, **params)

    def test_starts_unset(self, dev_split, tmp_path):
        assert self._optimizer(dev_split, tmp_path).termination_reason is None

    def test_exhausting_the_trial_budget_reports_the_trial_limit(self, dev_split, tmp_path):
        optimizer = self._optimizer(dev_split, tmp_path)
        optimizer.optimize()
        assert optimizer.termination_reason == topt.TERMINATION_TRIAL_LIMIT

    def test_running_out_of_clock_reports_the_study_timeout(self, dev_split, tmp_path, monkeypatch):
        """A study that returns short of its trial count was stopped by time."""
        optimizer = self._optimizer(dev_split, tmp_path, trials=2)

        class _ShortStudy:
            trials: list = []
            user_attrs: dict = {}

            def optimize(self, *args, **kwargs):
                self.trials = [object()]

            def set_user_attr(self, key, value):
                self.user_attrs[key] = value

        short = _ShortStudy()
        monkeypatch.setattr(topt.optuna, "create_study", lambda **kwargs: short)
        monkeypatch.setattr(topt.TemporalRegressionOptimizer, "_log_study_health", lambda s, x: {})
        optimizer.optimize()
        assert optimizer.termination_reason == topt.TERMINATION_STUDY_TIMEOUT
        assert short.user_attrs[topt.STUDY_TIMEOUT_REACHED_ATTR] is True

    def test_a_timed_out_persistent_study_is_not_resumed(self, dev_split, tmp_path, monkeypatch):
        """Re-running the notebook cannot grant a second four-hour budget."""
        optimizer = self._optimizer(dev_split, tmp_path, trials=2)

        class _TimedOutStudy:
            trials = [object()]
            user_attrs = {
                topt.STUDY_ELAPSED_SECONDS_ATTR: 1800.0,
                topt.STUDY_TIMEOUT_REACHED_ATTR: True,
            }

            def optimize(self, *args, **kwargs):
                raise AssertionError("A timed-out study must not start another trial")

        timed_out = _TimedOutStudy()
        monkeypatch.setattr(topt.optuna, "create_study", lambda **kwargs: timed_out)
        monkeypatch.setattr(topt.TemporalRegressionOptimizer, "_log_study_health", lambda s, x: {})

        optimizer.optimize()

        assert optimizer.elapsed_seconds == 0.0
        assert optimizer.cumulative_elapsed_seconds == 1800.0
        assert optimizer.termination_reason == topt.TERMINATION_STUDY_TIMEOUT

    def test_a_larger_cumulative_budget_can_resume_a_timed_out_study(
        self, dev_split, tmp_path, monkeypatch
    ):
        optimizer = self._optimizer(dev_split, tmp_path, trials=2, study_timeout=3600.0)

        class _ExtendedStudy:
            def __init__(self):
                self.trials = [object()]
                self.user_attrs = {
                    topt.STUDY_ELAPSED_SECONDS_ATTR: 1800.0,
                    topt.STUDY_TIMEOUT_REACHED_ATTR: True,
                }
                self.received_timeout = None

            def optimize(self, *args, **kwargs):
                self.received_timeout = kwargs["timeout"]
                self.trials.append(object())

            def set_user_attr(self, key, value):
                self.user_attrs[key] = value

        extended = _ExtendedStudy()
        monkeypatch.setattr(topt.optuna, "create_study", lambda **kwargs: extended)
        monkeypatch.setattr(topt.TemporalRegressionOptimizer, "_log_study_health", lambda s, x: {})

        optimizer.optimize()

        assert extended.received_timeout == pytest.approx(1800.0, abs=1.0)
        assert optimizer.termination_reason == topt.TERMINATION_TRIAL_LIMIT

    def test_current_trial_timeout_is_clipped_to_the_study_deadline(
        self, dev_split, tmp_path, monkeypatch
    ):
        """A trial already in progress cannot consume a fresh 30-minute budget
        after the cumulative study deadline."""
        optimizer = self._optimizer(dev_split, tmp_path, trial_timeout=600.0)
        captured = {}

        def fast_cv(*args, **kwargs):
            captured["timeout"] = kwargs["timeout"]
            return _fake_folds(), None

        monkeypatch.setattr(topt, "temporal_cv_with_timeout", fast_cv)
        optimizer._search_deadline = topt.time.monotonic() + 5.0

        value = optimizer.objective(_ridge_trial())

        assert np.isfinite(value)
        assert 0.0 < captured["timeout"] <= 5.0

    def test_an_already_complete_study_reports_the_trial_limit(self, dev_split, tmp_path):
        first = self._optimizer(dev_split, tmp_path)
        first.optimize()
        second = self._optimizer(dev_split, tmp_path)
        second.optimize()
        assert second.elapsed_seconds == 0.0
        assert second.termination_reason == topt.TERMINATION_TRIAL_LIMIT
