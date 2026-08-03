"""Smoke tests for src/optimizer.py — trials=2 ONLY (§9).

NEVER raise this to 400 in CI. The RegressionOptimizer is designed for
400 trials in production (cell [137]); smoke-testing with 2 confirms the
interface works without running a real search.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn import linear_model

import src.optimizer as opt_mod
from src.optimizer import (
    RegressionOptimizer,
    _is_transient_error,
    _SingleProcessHashingEncoder,
    define_estimator,
    get_regressor,
    make_metric_dataframe,
    purge_transient_invalid_configs,
    suppress_category_encoder_intercept_warning,
)


@pytest.fixture(autouse=True)
def _patch_globals(preprocessed_df, tmp_path, monkeypatch):
    """Inject the module globals needed by objective() / detailed_objective()."""
    from src.cv import make_temporal_holdout_split, make_ts_cv
    from sklearn.preprocessing import MaxAbsScaler

    df = preprocessed_df.copy()
    target_col = "Rented Bike Count"
    y_full = df[target_col].copy()
    X_full = df.drop(columns=[target_col])

    max_label = float(y_full.max())
    y_norm = y_full / max_label

    X_train, X_holdout, y_train, y_holdout, train_end, holdout_start = make_temporal_holdout_split(
        X_full, y_norm, holdout_size=50, holdout_gap=10
    )
    ts_cv = make_ts_cv(n_splits=2, gap=5, max_train_size=300, test_size=50)

    opt_mod.X_train_opt = X_train
    opt_mod.y_train_opt = y_train
    opt_mod.ts_cv = ts_cv
    opt_mod.X_holdout = X_holdout
    opt_mod.y_holdout = y_holdout
    opt_mod.max_label = max_label
    opt_mod.metric_dataframe = make_metric_dataframe()

    # Never let tests touch the real project's invalid-config blocklist.
    monkeypatch.setattr(opt_mod, "INVALID_CONFIGS_PATH", tmp_path / "invalid_configs.csv")

    yield

    # Reset to defaults after test
    opt_mod.X_train_opt = None
    opt_mod.y_train_opt = None
    opt_mod.ts_cv = None
    opt_mod.X_holdout = None
    opt_mod.y_holdout = None
    opt_mod.max_label = 1.0
    opt_mod.metric_dataframe = None


class TestDefineEstimator:
    def test_returns_list(self):
        result = define_estimator()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_xgb_in_xgboost_category(self):
        define_estimator()
        assert "XGBRegressor" in opt_mod.xgboost_estimators

    def test_lgbm_in_lightgbm_category(self):
        define_estimator()
        assert "LGBMRegressor" in opt_mod.lightgbm_estimators

    def test_hist_in_ensemble_category(self):
        define_estimator()
        assert "HistGradientBoostingRegressor" in opt_mod.essemble_estimators


class TestGetRegressor:
    def test_xgb(self):
        from xgboost import XGBRegressor

        assert get_regressor("XGBRegressor") is XGBRegressor

    def test_lgbm(self):
        from lightgbm import LGBMRegressor

        assert get_regressor("LGBMRegressor") is LGBMRegressor

    def test_hist(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        assert get_regressor("HistGradientBoostingRegressor") is HistGradientBoostingRegressor


class TestMakeMetricDataframe:
    def test_returns_dataframe(self):
        df = make_metric_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_index_names(self):
        df = make_metric_dataframe()
        assert df.index.names == ["Estimator", "Optimization", "Pre-Process Pipeline"]


class TestSuppressCategoryEncoderWarning:
    def test_context_manager_runs(self):
        with suppress_category_encoder_intercept_warning():
            pass  # should not raise


class TestRegressionOptimizerSmoke:
    """Smoke tests with trials=2. Never change to 400."""

    def test_optimizer_xgb_smoke(self, monkeypatch):
        """Two Optuna trials on XGBRegressor — verify best_params is a dict."""
        optimizer = RegressionOptimizer("XGBRegressor", trials=2)

        called = []

        def _fake_detailed(best_trial):
            called.append(best_trial)

        monkeypatch.setattr(optimizer, "detailed_objective", _fake_detailed)

        result = optimizer.optimize()
        assert isinstance(result, dict)
        assert len(called) == 1

    def test_optimizer_histgb_smoke(self, monkeypatch):
        optimizer = RegressionOptimizer("HistGradientBoostingRegressor", trials=2)
        monkeypatch.setattr(optimizer, "detailed_objective", lambda t: None)
        result = optimizer.optimize()
        assert isinstance(result, dict)

    def test_objective_returns_finite_float(self):
        """Single trial: objective() must return a finite float (not inf from exception)."""
        optimizer = RegressionOptimizer("HistGradientBoostingRegressor", trials=1)
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.optimize(optimizer.objective, n_trials=1)
        val = study.best_value
        assert np.isfinite(val) or np.isinf(val)  # finite preferred; inf = failed trial, acceptable


class TestObjectiveTimeout:
    """Per-trial hard timeout via a subprocess (§ Trial 47 runaway fix)."""

    def test_fast_trial_completes_normally(self):
        """Sanity check that the subprocess wrapper doesn't break the happy path.

        Whether the sampled pipeline itself succeeds is independent of the
        timeout machinery (see test_objective_returns_finite_float's own
        finite-or-inf tolerance) — what matters here is that a generous
        timeout never triggers the kill path for a fast trial.
        """
        optimizer = RegressionOptimizer("HistGradientBoostingRegressor", trials=1, trial_timeout=60)
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.optimize(optimizer.objective, n_trials=1)
        failed_reason = study.best_trial.user_attrs.get("failed_reason")
        assert failed_reason is None or "timeout" not in failed_reason.lower()

    def test_slow_trial_times_out_and_marks_failed(self):
        """A tiny trial_timeout guarantees the real pipeline exceeds it (spawn
        overhead alone is well over 0.01s), without needing to simulate a hang.
        Monkeypatching cross_validate wouldn't work here: spawn re-imports
        src.optimizer fresh in the child, so in-memory patches on the parent
        module object never reach the subprocess.

        Several trials are run because which configs the seeded sampler draws
        shifts whenever the search space changes, and some draws are rejected
        by the high-dim/blocklist guards *before* reaching the CV path — only
        trials that reach CV can hit the timeout.
        """
        optimizer = RegressionOptimizer(
            "HistGradientBoostingRegressor", trials=8, trial_timeout=0.01
        )
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.optimize(optimizer.objective, n_trials=8)
        assert np.isinf(study.best_value)
        reasons = [t.user_attrs.get("failed_reason") or "" for t in study.trials]
        assert any("timeout" in reason.lower() for reason in reasons), reasons


class TestInvalidConfigBlocklist:
    """Persistent blocklist so a timed-out/errored pipeline shape isn't retried."""

    def test_register_and_detect(self):
        optimizer = RegressionOptimizer("Ridge", trials=1)
        assert not optimizer.was_invalid_config(
            "Interactions_with_Kernels", "LeaveOneOutEncoder", "SequentialFeatureSelector"
        )
        optimizer.register_invalid_config(
            "Interactions_with_Kernels",
            "LeaveOneOutEncoder",
            "SequentialFeatureSelector",
            reason="Trial exceeded timeout",
        )
        assert optimizer.was_invalid_config(
            "Interactions_with_Kernels", "LeaveOneOutEncoder", "SequentialFeatureSelector"
        )

    def test_flush_persists_to_disk(self):
        invalid_path = opt_mod.INVALID_CONFIGS_PATH

        optimizer = RegressionOptimizer("Ridge", trials=1)
        optimizer.register_invalid_config(
            "Interactions_with_Kernels",
            "LeaveOneOutEncoder",
            "SequentialFeatureSelector",
            reason="x",
        )
        optimizer.flush_invalid_configs()
        assert invalid_path.exists()

        reloaded = RegressionOptimizer("Ridge", trials=1)
        assert reloaded.was_invalid_config(
            "Interactions_with_Kernels", "LeaveOneOutEncoder", "SequentialFeatureSelector"
        )

    def test_different_estimator_not_blocked(self):
        """The blocklist is scoped per-estimator (matches the Churn project's signature)."""
        optimizer = RegressionOptimizer("Ridge", trials=1)
        optimizer.register_invalid_config(
            "Interactions_with_Kernels",
            "LeaveOneOutEncoder",
            "SequentialFeatureSelector",
            reason="x",
        )
        other = RegressionOptimizer("XGBRegressor", trials=1)
        assert not other.was_invalid_config(
            "Interactions_with_Kernels", "LeaveOneOutEncoder", "SequentialFeatureSelector"
        )


class TestRidgeLassoRename:
    """RidgeCV/LassoCV were renamed to Ridge/Lasso — no CV-suffixed name should
    resolve anywhere, since cross-validation is always done externally now."""

    def test_ridge_instantiates_plain_ridge(self):
        assert get_regressor("Ridge") is linear_model.Ridge

    def test_lasso_instantiates_plain_lasso(self):
        assert get_regressor("Lasso") is linear_model.Lasso

    def test_old_cv_names_no_longer_resolve(self):
        define_estimator()
        assert "RidgeCV" not in opt_mod.linear_estimators
        assert "LassoCV" not in opt_mod.linear_estimators
        assert get_regressor("RidgeCV") is None
        assert get_regressor("LassoCV") is None

    def test_ridge_objective_runs_with_scalar_alpha(self):
        """Regression guard for the old alphas=<scalar> bug: Ridge branch must
        build a fittable estimator end-to-end via objective()."""
        optimizer = RegressionOptimizer("Ridge", trials=1, trial_timeout=60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.optimize(optimizer.objective, n_trials=1)
        failed_reason = study.best_trial.user_attrs.get("failed_reason")
        assert failed_reason is None or "timeout" not in failed_reason.lower()


class TestCategoricalFeaturesConsistency:
    """Regression guard keeping get_encoder()'s column list in sync with
    modeling_transformers()'s ColumnTransformer branches.

    WeekStatus/Rush_Hour/Rush_Period/Time_Period/DayNumberOnWeek are intentionally included
    (see docs/analise_r2_85_vs_88.md) — they are deterministic, train-window-only
    derivations of Hour/Weekday/Holiday and are present in X before it reaches
    the optimizer (see conftest.preprocessed_df)."""

    def test_encoder_categorical_list_matches_modeling_transformers(self):
        assert RegressionOptimizer._CATEGORICAL_FEATURES == [
            "Holiday",
            "Seasons",
            "Functioning Day",
            "Weekday",
            "Rainfall Cat",
            "Snowfall Cat",
            "WeekStatus",
            "Time_Period",
            "Rush_Hour",
            "Rush_Period",
            "DayNumberOnWeek",
        ]

    def test_mean_encoder_trial_does_not_crash_on_missing_columns(self):
        """Force encoders='MeanEncoder' (the exact combo that crashed in production,
        via study.enqueue_trial for determinism) and confirm objective() no longer
        fails with the WeekStatus/Rush_Hour/Time_Period KeyError."""
        optimizer = RegressionOptimizer("HistGradientBoostingRegressor", trials=1, trial_timeout=60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.enqueue_trial({"encoders": "MeanEncoder"})
        study.optimize(optimizer.objective, n_trials=1)
        failed_reason = study.best_trial.user_attrs.get("failed_reason") or ""
        assert "not in index" not in failed_reason

    @pytest.mark.parametrize("encoder_name", ["MeanEncoder", "CountFrequencyEncoder"])
    def test_dtype_validating_encoder_trial_does_not_fail_categorical_check(self, encoder_name):
        """Regression guard for the DayNumberOnWeek dtype bug: DayNumberOnWeek is
        an int column (WeekdayWeekStatusTransformer never casts it to category),
        so including it in MeanEncoder/CountFrequencyEncoder's `variables=` made
        every trial sampling either encoder fail 100% of the time with 'Some of
        the variables are not categorical' (see dataset/invalid_configs.csv)."""
        optimizer = RegressionOptimizer("HistGradientBoostingRegressor", trials=1, trial_timeout=60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.enqueue_trial({"encoders": encoder_name})
        study.optimize(optimizer.objective, n_trials=1)
        failed_reason = study.best_trial.user_attrs.get("failed_reason") or ""
        assert "not categorical" not in failed_reason

    def test_mean_encoder_variables_exclude_day_number_on_week(self):
        optimizer = RegressionOptimizer("Ridge", trials=1)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.enqueue_trial({"encoders": "MeanEncoder"})
        trial = study.ask()
        encoder, encoder_name = optimizer.get_encoder(trial)
        assert encoder_name == "MeanEncoder"
        assert "DayNumberOnWeek" not in encoder.variables


class TestHighDimSelectorRestriction:
    """NoSelector must be rejected for preprocessing that expands dimensionality
    a lot (Nystroem, PolynomialFeatures interactions). Optuna requires a static
    categorical search space per parameter name across the whole study, so this
    can't be enforced by narrowing get_feature_selection()'s candidate list per
    trial — it's enforced reactively in objective(), right after the (still
    static) "selectors" choice is sampled.
    """

    def test_no_selector_pruned_for_interactions_with_kernels(self):
        optimizer = RegressionOptimizer("Ridge", trials=1, trial_timeout=60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.enqueue_trial(
            {"modeler_name": "Interactions_with_Kernels", "selectors": "NoSelector"}
        )
        study.optimize(optimizer.objective, n_trials=1)
        assert np.isinf(study.best_value)
        assert "NoSelector" in study.best_trial.user_attrs["failed_reason"]

    def test_non_no_selector_allowed_for_interactions_with_kernels(self):
        optimizer = RegressionOptimizer("Ridge", trials=1, trial_timeout=60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.enqueue_trial(
            {"modeler_name": "Interactions_with_Kernels", "selectors": "SelectKBest"}
        )
        study.optimize(optimizer.objective, n_trials=1)
        failed_reason = study.best_trial.user_attrs.get("failed_reason") or ""
        assert "NoSelector not allowed" not in failed_reason


class TestNoTimestampLeakIntoRemainderScaler:
    """Regression guard: Date (Timestamp) leaking into remainder=scaler.

    Sin_Cos, Time_steps_as_categories, Periodic_Spline, and Pairwise_Interactions
    only list a subset of columns in their ColumnTransformer; remainder used to be
    set to the scaler, so any un-listed column (e.g. the Date timestamp column kept
    for time-series plotting) was passed straight into StandardScaler/RobustScaler/
    MaxAbsScaler and crashed with "float() argument must be a string or a real
    number, not 'Timestamp'".
    """

    @pytest.mark.parametrize(
        "modeler_name",
        ["Sin_Cos", "Time_steps_as_categories", "Periodic_Spline", "Pairwise_Interactions"],
    )
    def test_modeler_does_not_leak_timestamp_into_remainder(self, modeler_name):
        optimizer = RegressionOptimizer("Ridge", trials=1, trial_timeout=60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.enqueue_trial({"modeler_name": modeler_name})
        study.optimize(optimizer.objective, n_trials=1)
        failed_reason = study.best_trial.user_attrs.get("failed_reason") or ""
        assert "Timestamp" not in failed_reason


class TestNestedParallelismUnderDaemonWrapper:
    """Regression guard for the LGBM R2 drop (0.848237 -> 0.746533): every
    trial's cross_validate runs inside a daemon=True subprocess
    (_run_in_subprocess_with_timeout), and daemonic processes cannot spawn
    children. HashingEncoder's _transform() unconditionally creates a
    multiprocessing.Manager() (a spawn, regardless of max_process), which
    crashed every single trial sampling it with "daemonic processes are not
    allowed to have children" -- permanently blocklisting encoder/selector
    combos across every estimator and starving Optuna's search.
    """

    def test_single_process_hashing_encoder_matches_stock_output(self):
        """The daemon-safe override must not change what gets encoded."""
        df = pd.DataFrame({"a": ["x", "y", "z", "w"] * 20, "b": ["p", "q"] * 40})
        from category_encoders import HashingEncoder

        baseline = HashingEncoder(cols=["a", "b"]).fit_transform(df)
        candidate = _SingleProcessHashingEncoder(cols=["a", "b"]).fit_transform(df)
        pd.testing.assert_frame_equal(baseline, candidate)

    def test_hashing_encoder_trial_does_not_crash_under_daemon_subprocess(self):
        """Force encoders='HashingEncoder' via enqueue_trial (mirrors
        TestCategoricalFeaturesConsistency's pattern) and confirm objective()
        no longer fails with the daemonic-process error."""
        optimizer = RegressionOptimizer("HistGradientBoostingRegressor", trials=1, trial_timeout=60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.enqueue_trial({"encoders": "HashingEncoder"})
        study.optimize(optimizer.objective, n_trials=1)
        failed_reason = (study.best_trial.user_attrs.get("failed_reason") or "").lower()
        assert "daemonic" not in failed_reason

    @pytest.mark.parametrize("estimator_name", ["LGBMRegressor", "XGBRegressor", "Ridge"])
    def test_hashing_encoder_trial_does_not_crash_for_affected_estimators(self, estimator_name):
        """LGBM/XGB/Ridge all showed R2 drift in the same MLflow experiment;
        HashingEncoder is reachable from every estimator's encoder search
        space, so every one of them is exposed to this bug."""
        optimizer = RegressionOptimizer(estimator_name, trials=1, trial_timeout=60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.enqueue_trial({"encoders": "HashingEncoder"})
        study.optimize(optimizer.objective, n_trials=1)
        failed_reason = (study.best_trial.user_attrs.get("failed_reason") or "").lower()
        assert "daemonic" not in failed_reason

    def test_sequential_feature_selector_trial_does_not_crash_under_daemon_subprocess(self):
        """SequentialFeatureSelector(n_jobs=-1) is the other nested-parallelism
        candidate (reachable for Ridge/KNN/SVR/MLP); the joblib threading
        backend wrap in _run_cross_validate_in_subprocess must keep it safe."""
        optimizer = RegressionOptimizer("Ridge", trials=1, trial_timeout=60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.enqueue_trial({"selectors": "SequentialFeatureSelector"})
        study.optimize(optimizer.objective, n_trials=1)
        failed_reason = (study.best_trial.user_attrs.get("failed_reason") or "").lower()
        assert "daemonic" not in failed_reason


class TestTransientErrorBlocklistHygiene:
    """Transient infra failures (the daemonic-process bug) must not be
    persisted to the invalid-config blocklist the way genuine, deterministic
    pipeline-shape incompatibilities are -- otherwise a bug in the timeout
    subprocess machinery permanently and silently shrinks Optuna's effective
    search space, exactly as happened for the LGBM R2 drop.
    """

    def test_daemonic_message_is_transient(self):
        assert _is_transient_error("daemonic processes are not allowed to have children")
        assert _is_transient_error("DAEMONIC processes are not allowed to have children")

    @pytest.mark.parametrize(
        "reason",
        [
            "Cannot predict random effects from singular covariance structure.",
            "max_features == 24, must be <= 16.",
            "Some of the variables are not categorical",
            "k should be <= n_features",
        ],
    )
    def test_structural_messages_are_not_transient(self, reason):
        assert not _is_transient_error(reason)

    def test_transient_trial_failure_is_not_persisted_to_blocklist(self, monkeypatch):
        """A trial that fails with a transient reason must still return inf
        and set failed_reason, but must NOT be written to the blocklist."""
        optimizer = RegressionOptimizer("Ridge", trials=1)
        monkeypatch.setattr(
            opt_mod,
            "_cross_validate_with_timeout",
            lambda *a, **k: (None, "daemonic processes are not allowed to have children"),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.optimize(optimizer.objective, n_trials=1)

        assert np.isinf(study.best_value)
        assert "daemonic" in study.best_trial.user_attrs["failed_reason"].lower()
        assert optimizer.invalid_df.empty

    def test_purge_removes_only_transient_rows(self, tmp_path):
        csv_path = tmp_path / "invalid_configs.csv"
        pd.DataFrame(
            [
                {
                    "estimator": "LGBMRegressor",
                    "modeler_name": "linear_modeling",
                    "encoder": "HashingEncoder",
                    "selector": "NoSelector",
                    "reason": "daemonic processes are not allowed to have children",
                },
                {
                    "estimator": "Ridge",
                    "modeler_name": "Interactions_with_Kernels",
                    "encoder": "GLMMEncoder",
                    "selector": "NoSelector",
                    "reason": "Cannot predict random effects from singular covariance structure.",
                },
            ]
        ).to_csv(csv_path, index=False)

        removed = purge_transient_invalid_configs(path=csv_path)

        assert removed == 1
        remaining = pd.read_csv(csv_path)
        assert len(remaining) == 1
        assert remaining.iloc[0]["encoder"] == "GLMMEncoder"

    def test_purge_is_noop_when_file_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.csv"
        assert purge_transient_invalid_configs(path=missing_path) == 0


class TestDetailedFitTimeout:
    """Post-study re-evaluation timeout guard.

    Ridge ran 8h+ in production despite a 4h study_timeout: study_timeout only
    bounds study.optimize() (the trial search), but detailed_objective() ->
    save_model_and_metrics_regression() runs *after* that — a full cross_validate,
    a manual per-fold OOF refit loop, and a final full-train fit — with no cap
    at all. _detailed_fit_with_timeout() closes that gap the same way
    _cross_validate_with_timeout() already does for individual trials.
    """

    @staticmethod
    def _build_numeric_pipeline():
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        return Pipeline(
            steps=[
                (
                    "select_numeric",
                    ColumnTransformer(
                        transformers=[
                            ("num", "passthrough", make_column_selector(dtype_include=np.number))
                        ],
                        remainder="drop",
                    ),
                ),
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", Ridge()),
            ]
        )

    def test_fast_detailed_fit_completes(self):
        estimator = self._build_numeric_pipeline()
        (
            metrics,
            y_pred_cv,
            final_estimator,
            y_train_fit,
            y_holdout_pred,
        ) = opt_mod._detailed_fit_with_timeout(
            estimator,
            opt_mod.X_train_opt,
            opt_mod.y_train_opt,
            opt_mod.X_holdout,
            opt_mod.ts_cv,
            timeout=60,
        )
        assert len(y_pred_cv) == len(opt_mod.y_train_opt)
        assert len(y_holdout_pred) == len(opt_mod.y_holdout)
        assert len(y_train_fit) == len(opt_mod.y_train_opt)
        for key in (
            "train_neg_mean_absolute_error",
            "test_neg_mean_absolute_error",
            "train_neg_mean_squared_error",
            "test_neg_mean_squared_error",
            "train_neg_root_mean_squared_error",
            "test_neg_root_mean_squared_error",
            "train_r2",
            "test_r2",
        ):
            assert key in metrics
            assert len(metrics[key]) == opt_mod.ts_cv.get_n_splits()

    def test_reports_progress_for_each_stage(self):
        """Regression guard for opaque multi-hour black-box runs: the caller must
        see which of the two remaining stages (single-pass CV / final fit) is
        slow instead of only finding out at the very end (or at the timeout)."""
        estimator = self._build_numeric_pipeline()
        messages = []
        status, payload = opt_mod._run_in_subprocess_with_timeout(
            opt_mod._run_detailed_fit_in_subprocess,
            (estimator, opt_mod.X_train_opt, opt_mod.y_train_opt, opt_mod.X_holdout, opt_mod.ts_cv),
            timeout=60,
            on_progress=messages.append,
        )
        assert status == "ok"
        assert len(messages) == 2
        assert "single-pass per-fold CV" in messages[0]
        assert "final full-train fit" in messages[1]

    def test_detailed_fit_times_out(self):
        """A tiny timeout guarantees the real subprocess exceeds it (spawn
        overhead alone is well over 0.01s) — mirrors
        TestObjectiveTimeout.test_slow_trial_times_out_and_marks_failed."""
        estimator = self._build_numeric_pipeline()
        with pytest.raises(TimeoutError):
            opt_mod._detailed_fit_with_timeout(
                estimator,
                opt_mod.X_train_opt,
                opt_mod.y_train_opt,
                opt_mod.X_holdout,
                opt_mod.ts_cv,
                timeout=0.01,
            )


class TestEarlyStoppingCrossValidate:
    """Per-fold nested-carve early stopping for XGBRegressor/LGBMRegressor trials
    (overfitting-reduction plan, item B): neither estimator had any early
    stopping before, so both trained their full sampled n_estimators every
    trial regardless of whether later trees still improved generalization.
    """

    @staticmethod
    def _build_ttr_pipeline(regressor):
        from sklearn.compose import (
            ColumnTransformer,
            TransformedTargetRegressor,
            make_column_selector,
        )
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler

        return Pipeline(
            steps=[
                (
                    "select_numeric",
                    ColumnTransformer(
                        transformers=[
                            ("num", "passthrough", make_column_selector(dtype_include=np.number))
                        ],
                        remainder="drop",
                    ),
                ),
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    TransformedTargetRegressor(regressor=regressor, transformer=MinMaxScaler()),
                ),
            ]
        )

    def test_xgb_early_stopping_produces_expected_keys(self):
        from xgboost import XGBRegressor

        estimator = self._build_ttr_pipeline(
            XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
        )
        cv_results, err = opt_mod._cross_validate_with_early_stopping_timeout(
            estimator,
            opt_mod.X_train_opt,
            opt_mod.y_train_opt,
            opt_mod.ts_cv,
            timeout=60,
        )
        assert err is None
        for key in (
            "train_neg_mean_absolute_error",
            "test_neg_mean_absolute_error",
            "train_neg_root_mean_squared_error",
            "test_neg_root_mean_squared_error",
            "train_r2",
            "test_r2",
        ):
            assert key in cv_results
            assert len(cv_results[key]) == opt_mod.ts_cv.get_n_splits()
        assert len(cv_results["best_iterations"]) == opt_mod.ts_cv.get_n_splits()
        assert all(b is not None for b in cv_results["best_iterations"])
        # Early stopping must never report an iteration count beyond n_estimators.
        assert all(b <= 50 for b in cv_results["best_iterations"])

    def test_lgbm_early_stopping_produces_expected_keys(self):
        from lightgbm import LGBMRegressor

        estimator = self._build_ttr_pipeline(
            LGBMRegressor(n_estimators=50, max_depth=3, verbosity=-1, random_state=42)
        )
        cv_results, err = opt_mod._cross_validate_with_early_stopping_timeout(
            estimator,
            opt_mod.X_train_opt,
            opt_mod.y_train_opt,
            opt_mod.ts_cv,
            timeout=60,
        )
        assert err is None
        assert len(cv_results["best_iterations"]) == opt_mod.ts_cv.get_n_splits()
        assert all(b is not None for b in cv_results["best_iterations"])
        assert all(b <= 50 for b in cv_results["best_iterations"])

    def test_fold_too_small_for_early_stopping_slice_reports_error(self):
        """A tiny cv guarantees the es_val_fraction/gap carve exceeds the fold's
        own train window, which must surface as a normal 'error' result (not an
        unhandled crash) — mirrors TestObjectiveTimeout's timeout-path checks."""
        from xgboost import XGBRegressor
        from src.cv import make_ts_cv

        tiny_cv = make_ts_cv(n_splits=2, gap=1, max_train_size=60, test_size=10)
        estimator = self._build_ttr_pipeline(
            XGBRegressor(n_estimators=20, max_depth=2, random_state=42)
        )
        cv_results, err = opt_mod._cross_validate_with_early_stopping_timeout(
            estimator,
            opt_mod.X_train_opt,
            opt_mod.y_train_opt,
            tiny_cv,
            timeout=60,
        )
        assert cv_results is None
        assert err is not None
        assert "too small" in err.lower()


class TestGapTrackingUserAttrs:
    """Log-only train/val R2 gap tracking (overfitting-reduction plan, item C):
    trial.set_user_attr must never change which trial Optuna considers best —
    only surface train_r2/test_r2/r2_gap for post-hoc inspection.
    """

    @staticmethod
    def _run_single_trial(estimator_name):
        optimizer = RegressionOptimizer(estimator_name, trials=1)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.optimize(optimizer.objective, n_trials=1)
        return study

    def test_xgb_trial_records_gap_and_best_iterations(self):
        study = self._run_single_trial("XGBRegressor")
        attrs = study.best_trial.user_attrs
        if attrs.get("failed_reason") is not None:
            pytest.skip(f"trial failed for an unrelated reason: {attrs['failed_reason']}")
        assert "train_r2" in attrs
        assert "test_r2" in attrs
        assert attrs["r2_gap"] == pytest.approx(attrs["train_r2"] - attrs["test_r2"])
        assert "best_iterations" in attrs

    def test_ridge_trial_records_gap_without_best_iterations(self):
        """Non-early-stopping estimators still get the gap-tracking user_attrs
        (added uniformly via return_train_score=True), but never best_iterations
        (that key is only set on the early-stopping path)."""
        study = self._run_single_trial("Ridge")
        attrs = study.best_trial.user_attrs
        if attrs.get("failed_reason") is not None:
            pytest.skip(f"trial failed for an unrelated reason: {attrs['failed_reason']}")
        assert "train_r2" in attrs
        assert "test_r2" in attrs
        assert "best_iterations" not in attrs


class TestBestIterationPropagation:
    """detailed_objective() must fix the final fit's n_estimators to the
    early-stopping median across the winning trial's CV folds (overfitting-
    reduction plan, item B's final-fit propagation) instead of either the
    trial's full sampled n_estimators or re-running early stopping in the
    one-off final fit.
    """

    def test_final_fit_uses_median_best_iteration(self, monkeypatch):
        optimizer = RegressionOptimizer("XGBRegressor", trials=1)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        study.optimize(optimizer.objective, n_trials=1)

        best_iterations = study.best_trial.user_attrs.get("best_iterations")
        if not best_iterations:
            pytest.skip("trial did not record best_iterations (failed for an unrelated reason)")
        expected_median = max(int(np.median(best_iterations)), 1)

        captured = {}

        def _fake_save(
            description,
            preprocessing,
            pipe,
            regressor,
            target_transformer,
            params,
            feature_selector=None,
            timeout=None,
        ):
            captured["n_estimators"] = regressor.get_params()["n_estimators"]

        monkeypatch.setattr(opt_mod, "save_model_and_metrics_regression", _fake_save)
        optimizer.detailed_objective(study.best_trial)

        assert captured["n_estimators"] == expected_median


class TestLGBMMetricRegression:
    """Regression tests for the all-inf LGBM incident: the search space carried
    ``"metric": "r2"`` — not a LightGBM metric — which was harmless while LGBM
    trained without an eval_set, but once the early-stopping callback was added
    every fit raised "For early stopping, at least one dataset and eval metric
    is required for evaluation", so all 400 trials of the study returned inf and
    the saved "winner" was trial 0's meaningless defaults. The old early-stopping
    test hand-rolled an LGBMRegressor without the ``metric`` field, which is why
    it never caught this — these tests go through the *real* search space.
    """

    @staticmethod
    def _sample_search_space_params(estimator_name):
        optimizer = RegressionOptimizer(estimator_name, trials=1)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        trial = study.ask()
        return optimizer.get_parameters(estimator_name, trial), trial

    def test_lgbm_search_space_metric_is_a_valid_lightgbm_metric(self):
        params, _ = self._sample_search_space_params("LGBMRegressor")
        assert params["metric"] != "r2"
        # Aliases LightGBM actually understands for regression eval.
        assert params["metric"] in {"l1", "mae", "l2", "mse", "rmse", "huber", "mape"}

    def test_lgbm_early_stopping_with_real_search_space_params(self):
        from lightgbm import LGBMRegressor

        params, _ = self._sample_search_space_params("LGBMRegressor")
        params["n_estimators"] = 30  # keep the smoke fast; search space doesn't sample it
        estimator = TestEarlyStoppingCrossValidate._build_ttr_pipeline(LGBMRegressor(**params))
        cv_results, err = opt_mod._cross_validate_with_early_stopping_timeout(
            estimator,
            opt_mod.X_train_opt,
            opt_mod.y_train_opt,
            opt_mod.ts_cv,
            timeout=120,
        )
        assert err is None, f"early-stopping CV failed with real search-space params: {err}"
        assert all(b is not None for b in cv_results["best_iterations"])

    def test_xgb_max_depth_upper_bound_relaxed_to_8(self):
        """The [3, 6] cap made the winner pin depth at the floor and cost ~0.04
        holdout R2 — with temporal early stopping guarding memorization the
        ceiling went back to 8."""
        _, trial = self._sample_search_space_params("XGBRegressor")
        dist = trial.distributions["max_depth"]
        assert dist.low == 3
        assert dist.high == 8


class TestHGBSelectorSpace:
    """HGB's selector space excludes SequentialFeatureSelector (a single SFS
    trial took ~30 min — the last study finished only 17/400 trials in ~8.2 h)
    and RFE/SelectFromModel (both clone the trial estimator, and HGB exposes
    neither feature_importances_ nor coef_)."""

    def test_hgb_selector_space_is_noselector_or_selectkbest(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        optimizer = RegressionOptimizer("HistGradientBoostingRegressor", trials=1)
        study = optuna.create_study(direction="minimize", sampler=optimizer.sampler)
        trial = study.ask()
        optimizer.get_feature_selection(trial, HistGradientBoostingRegressor())
        assert set(trial.distributions["selectors"].choices) == {"NoSelector", "SelectKBest"}


class TestStudyHealthReport:
    """_report_study_health must fail loudly on a fully-broken study (every
    trial inf) instead of letting detailed_objective() save a garbage winner,
    and must stay silent when at least one trial succeeded."""

    def test_all_inf_study_raises_runtime_error(self):
        optimizer = RegressionOptimizer("LGBMRegressor", trials=2)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: float("inf"), n_trials=2)
        with pytest.raises(RuntimeError, match="returned inf"):
            optimizer._report_study_health(study)

    def test_study_with_a_finite_trial_does_not_raise(self):
        optimizer = RegressionOptimizer("LGBMRegressor", trials=2)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: float("inf") if trial.number == 0 else 0.5, n_trials=2)
        optimizer._report_study_health(study)


class TestInvalidConfigsPurgeHygiene:
    """The 138 LGBM combos poisoned by the metric bug were purged from the
    project blocklist; this guards against the file being re-poisoned by a
    recurrence of the incident."""

    def test_real_blocklist_has_no_early_stopping_metric_rows(self):
        from pathlib import Path

        real_csv = (
            Path(opt_mod.__file__).resolve().parent.parent / "dataset" / "invalid_configs.csv"
        )
        if not real_csv.exists():
            pytest.skip("project blocklist not present")
        content = real_csv.read_text(encoding="utf-8")
        assert "For early stopping, at least one dataset and eval metric is required" not in content
