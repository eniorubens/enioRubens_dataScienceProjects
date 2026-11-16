"""Optuna studies with preprocessing and feature selection in the search space."""

from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from optuna.trial import TrialState
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from .modeling import calibrated_classifier, classification_metrics, make_preprocessor

SUPPORTED_MODELS = {
    "regression": {"ElasticNet", "Ridge", "XGBoost", "LightGBM", "CatBoost"},
    "classification": {"LogisticRegression", "XGBoost", "LightGBM", "CatBoost"},
}
# Only linear models receive a scaler; tree ensembles are scale-invariant.
LINEAR_MODELS = frozenset({"ElasticNet", "Ridge", "LogisticRegression"})


def _progress_callback(total_trials: int, initial_completed: int = 0):
    """Create a text progress callback for an Optuna study.

    Parameters
    ----------
    total_trials:
        Number of trials requested for the study.
    initial_completed:
        Number of terminal trials already present when the current run starts.

    Returns
    -------
    callable
        Callback compatible with ``Study.optimize``.
    """

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        total_finished = sum(
            item.state in {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
            for item in study.trials
        )
        finished = max(total_finished - initial_completed, 0)
        width = 24
        filled = int(width * finished / max(total_trials, 1))
        bar = "=" * filled + "." * (width - filled)
        try:
            best_value = f"{study.best_value:.5f}"
        except ValueError:
            best_value = "n/a"
        print(
            f"[Optuna] {study.study_name} [{bar}] {finished}/{total_trials} "
            f"trial={trial.number} state={trial.state.name.lower()} best={best_value}",
            flush=True,
        )

    return callback


def _preprocessor(trial: optuna.Trial, model_name: str):
    """Build the shared preprocessor, searching a scaler only for linear models."""
    scaler_name = (
        trial.suggest_categorical("preprocessing_scaler", ["standard", "robust"])
        if model_name in LINEAR_MODELS
        else None
    )
    return make_preprocessor(scaler_name)


def _selector(trial: optuna.Trial, task: str) -> object:
    selector_name = trial.suggest_categorical(
        "feature_selector", ["passthrough", "percentile"]
    )
    if selector_name == "passthrough":
        return "passthrough"
    score_function = f_regression if task == "regression" else f_classif
    percentile = trial.suggest_int("selector_percentile", 50, 100, step=10)
    return SelectPercentile(score_func=score_function, percentile=percentile)


def _regression_estimator(trial: optuna.Trial, model_name: str) -> object:
    if model_name == "ElasticNet":
        return ElasticNet(
            alpha=trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.05, 0.95),
            max_iter=300_000,
            tol=1e-4,
        )
    if model_name == "Ridge":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100.0, log=True))
    if model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 150, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=1,
        )
    if model_name == "LightGBM":
        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 150, 500, step=50),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            bagging_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    if model_name == "CatBoost":
        return CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 150, 500, step=50),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            loss_function="RMSE",
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"unsupported regression model: {model_name}")


def _classification_estimator(trial: optuna.Trial, model_name: str) -> object:
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100.0, log=True),
            solver="liblinear",
            max_iter=2_000,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 150, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 150, 500, step=50),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            bagging_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    if model_name == "CatBoost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 150, 500, step=50),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"unsupported classification model: {model_name}")


def build_optuna_pipeline(
    trial: optuna.Trial,
    task: str,
    model_name: str,
) -> Pipeline:
    """Build a leakage-aware preprocessing, selection, and model pipeline.

    Classification estimators are unweighted and wrapped in isotonic
    ``CalibratedClassifierCV`` (``cv=3``), matching the tournament policy.
    """
    if task not in SUPPORTED_MODELS or model_name not in SUPPORTED_MODELS[task]:
        raise ValueError(f"unsupported task/model combination: {task}/{model_name}")
    if task == "regression":
        estimator = _regression_estimator(trial, model_name)
    else:
        # Single imbalance policy: no class weights, isotonic calibration.
        estimator = calibrated_classifier(_classification_estimator(trial, model_name))
    return Pipeline(
        [
            ("preprocessor", _preprocessor(trial, model_name)),
            ("selector", _selector(trial, task)),
            ("model", estimator),
        ]
    )


def run_optuna_study(
    splits: dict[str, pd.DataFrame],
    task: str,
    model_name: str,
    n_trials: int = 30,
    storage_path: str | Path | None = None,
    study_name: str | None = None,
    progress: bool = True,
    target_trials: int | None = None,
    folds: list[dict[str, pd.DataFrame]] | None = None,
    timeout_seconds: float | None = None,
) -> optuna.Study:
    """Optimize one model on train/validation with optional SQLite persistence.

    Parameters
    ----------
    splits:
        Chronological train, validation, and test dataframes. The test frame is
        accepted for a common split contract but is not used during tuning.
    task:
        Either ``"regression"`` or ``"classification"``.
    model_name:
        Estimator name supported for the selected task.
    n_trials:
        Number of Optuna trials to execute or resume.
    storage_path:
        Optional SQLite path used to persist the study.
    study_name:
        Optional stable study name for persistence and resumption.
    progress:
        Whether to print a terminal-safe progress gauge after every trial.
    target_trials:
        Optional total number of terminal trials desired for the persisted
        study. When provided, only the remaining trials are executed. This
        makes notebook reruns resumable without exceeding the target.
    folds:
        Optional expanding-window folds. When omitted, the train/validation
        pair in ``splits`` is used as one development fold.
    timeout_seconds:
        Optional wall-clock limit forwarded to ``Study.optimize``. Trials
        already running are allowed to finish; ``None`` means no limit.
    """
    if task not in SUPPORTED_MODELS or model_name not in SUPPORTED_MODELS[task]:
        raise ValueError(f"unsupported task/model combination: {task}/{model_name}")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if target_trials is not None and target_trials < 1:
        raise ValueError("target_trials must be positive")
    target = "delivery_delay_days" if task == "regression" else "sla_breach"
    direction = "minimize" if task == "regression" else "maximize"
    study_name = study_name or f"{task}_{model_name}"
    storage = None
    if storage_path is not None:
        database_path = Path(storage_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{database_path.resolve().as_posix()}"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )

    def objective(trial: optuna.Trial) -> float:
        development_folds = folds or [
            {"train": splits["train"], "validation": splits["validation"]}
        ]
        fold_scores = []
        for fold_index, fold in enumerate(development_folds):
            pipeline = build_optuna_pipeline(trial, task, model_name)
            X_train = fold["train"]
            X_validation = fold["validation"]
            y_train = fold["train"][target]
            y_validation = fold["validation"][target]
            if task == "classification":
                y_train = y_train.astype(int)
            pipeline.fit(X_train, y_train)
            if task == "regression":
                fold_score = mean_squared_error(
                    y_validation,
                    pipeline.predict(X_validation),
                ) ** 0.5
            else:
                fold_score = classification_metrics(
                    y_validation.astype(int),
                    pipeline.predict_proba(X_validation)[:, 1],
                )["average_precision"]
            fold_scores.append(fold_score)
            trial.report(float(sum(fold_scores) / len(fold_scores)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(sum(fold_scores) / len(fold_scores))

    initial_completed = sum(
        item.state in {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
        for item in study.trials
    )
    requested_trials = (
        max(target_trials - initial_completed, 0)
        if target_trials is not None
        else n_trials
    )
    if requested_trials == 0:
        print(
            f"[Optuna] {study.study_name} already has {initial_completed}/"
            f"{target_trials} terminal trials.",
            flush=True,
        )
        return study
    callbacks = (
        [_progress_callback(requested_trials, initial_completed)] if progress else None
    )
    study.optimize(
        objective,
        n_trials=requested_trials,
        timeout=timeout_seconds,
        callbacks=callbacks,
    )
    return study
