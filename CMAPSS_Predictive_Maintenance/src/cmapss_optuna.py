"""Otimização Optuna com validação por motor para o baseline de RUL."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


FINALIST_NAMES = (
    "CatBoost",
    "Extra Trees",
    "HistGradientBoosting",
    "XGBoost",
)

SEARCH_VERSIONS = ("v1", "v2")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _format_hyperparameters(params: dict) -> str:
    """Converte os parâmetros do melhor trial em uma string legível."""
    formatted = []
    for name, value in params.items():
        if isinstance(value, float):
            value_text = f"{value:.6g}"
        else:
            value_text = str(value)
        formatted.append(f"{name}={value_text}")
    return " | ".join(formatted)


def _suggest_feature_selection(
    trial: optuna.Trial,
    n_features: int,
) -> tuple[str, object]:
    selection = trial.suggest_categorical(
        "feature_selection",
        ["passthrough", "select_k_best"],
    )
    if selection == "passthrough":
        return selection, "passthrough"
    k_features = trial.suggest_int(
        "k_features",
        low=min(5, n_features),
        high=n_features,
    )
    return selection, SelectKBest(score_func=f_regression, k=k_features)


def build_optuna_pipeline(
    trial: optuna.Trial,
    model_name: str,
    n_features: int,
    random_state: int = 42,
    search_version: str = "v1",
) -> Pipeline:
    """Monta um pipeline específico para o estimador e o trial atuais."""
    if search_version not in SEARCH_VERSIONS:
        raise ValueError(f"Versão de busca não suportada: {search_version}")

    _, selector = _suggest_feature_selection(trial, n_features)

    if model_name == "CatBoost":
        estimator = CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 200, 600),
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),
            l2_leaf_reg=trial.suggest_float(
                "l2_leaf_reg", 1.0, 30.0, log=True
            ),
            random_strength=trial.suggest_float(
                "random_strength", 0.0, 2.0
            ),
            bagging_temperature=trial.suggest_float(
                "bagging_temperature", 0.0, 2.0
            ),
            loss_function="RMSE",
            random_seed=random_state,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    elif model_name == "Extra Trees":
        estimator = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 150, 500),
            max_depth=trial.suggest_int("max_depth", 4, 24),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
            random_state=random_state,
            n_jobs=1,
        )
    elif model_name == "HistGradientBoosting":
        if search_version == "v2":
            max_iter_bounds = (100, 800)
            max_leaf_nodes_bounds = (3, 31)
            min_samples_leaf_bounds = (10, 160)
        else:
            max_iter_bounds = (100, 500)
            max_leaf_nodes_bounds = (15, 63)
            min_samples_leaf_bounds = (10, 80)
        estimator = HistGradientBoostingRegressor(
            max_iter=trial.suggest_int("max_iter", *max_iter_bounds),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),
            max_leaf_nodes=trial.suggest_int(
                "max_leaf_nodes", *max_leaf_nodes_bounds
            ),
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf", *min_samples_leaf_bounds
            ),
            l2_regularization=trial.suggest_float(
                "l2_regularization", 1e-6, 10.0, log=True
            ),
            random_state=random_state,
        )
    elif model_name == "XGBoost":
        if search_version == "v2":
            n_estimators_bounds = (150, 900)
            gamma_bounds = (0.0, 1.0)
        else:
            n_estimators_bounds = (150, 600)
            gamma_bounds = (0.0, 0.5)
        estimator = XGBRegressor(
            n_estimators=trial.suggest_int(
                "n_estimators", *n_estimators_bounds
            ),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            gamma=trial.suggest_float("gamma", *gamma_bounds),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=1,
            random_state=random_state,
            verbosity=0,
        )
    else:
        raise ValueError(f"Finalista não suportado: {model_name}")

    return Pipeline([
        ("feature_selection", selector),
        ("estimator", estimator),
    ])


def _macro_motor_mae(
    df: pd.DataFrame,
    validation_index: np.ndarray,
    target: str,
    prediction: np.ndarray,
) -> float:
    validation = df.iloc[validation_index][["unit"]].copy()
    error = np.abs(df.iloc[validation_index][target].to_numpy() - prediction)
    validation["absolute_error"] = error
    return float(validation.groupby("unit")["absolute_error"].mean().mean())


def optimize_model(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    n_trials: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
    storage_dir: str | Path | None = None,
    search_version: str = "v1",
    study_id: str = "fd001",
    show_progress_bar: bool = False,
    log_trials: bool = False,
) -> optuna.Study:
    """Executa ou retoma um estudo Optuna com folds por motor."""
    if model_name not in FINALIST_NAMES:
        raise ValueError(f"Finalista não suportado: {model_name}")

    splitter = GroupKFold(n_splits=n_splits)
    X = df[features]
    y = df["rul"]
    groups = df["unit"]
    splits = list(splitter.split(X, y, groups))

    def objective(trial: optuna.Trial) -> float:
        pipeline = build_optuna_pipeline(
            trial,
            model_name=model_name,
            n_features=len(features),
            random_state=random_state,
            search_version=search_version,
        )
        fold_scores: list[float] = []
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            fitted = clone(pipeline)
            fitted.fit(X.iloc[train_index], y.iloc[train_index])
            prediction = fitted.predict(X.iloc[validation_index])
            score = _macro_motor_mae(
                df,
                validation_index,
                "rul",
                prediction,
            )
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    if search_version not in SEARCH_VERSIONS:
        raise ValueError(f"Versão de busca não suportada: {search_version}")
    study_name = f"{_slugify(study_id)}_{_slugify(model_name)}_{search_version}"
    storage = None
    if storage_dir is not None:
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{(storage_path / (study_name + '.db')).as_posix()}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.set_user_attr("search_version", search_version)
    remaining_trials = max(0, n_trials - len(study.trials))
    if remaining_trials:
        def report_trial(study_: optuna.Study, trial: optuna.Trial) -> None:
            if not log_trials:
                return
            value = "n/a" if trial.value is None else f"{trial.value:.4f}"
            print(
                f"[{model_name}] trial={trial.number} "
                f"state={trial.state.name} value={value}",
                flush=True,
            )

        study.optimize(
            objective,
            n_trials=remaining_trials,
            show_progress_bar=show_progress_bar,
            callbacks=[report_trial],
        )
    return study


def study_summary(studies: dict[str, optuna.Study]) -> pd.DataFrame:
    """Resume os melhores resultados de vários estudos."""
    rows = []
    for model_name, study in studies.items():
        hyperparameters = _format_hyperparameters(study.best_params)
        rows.append({
            "modelo": model_name,
            "versao_busca": study.user_attrs.get("search_version", "v1"),
            "mae_cv_motor": float(study.best_value),
            "trials": len(study.trials),
            "melhor_trial": int(study.best_trial.number),
            "status": study.best_trial.state.name,
            "hiperparametros": hyperparameters,
            "melhores_parametros": hyperparameters,
        })
    return pd.DataFrame(rows).sort_values("mae_cv_motor").reset_index(drop=True)


def build_pipeline_from_params(
    model_name: str,
    params: dict,
    n_features: int,
    random_state: int = 42,
    search_version: str = "v1",
) -> Pipeline:
    """Reconstrói o pipeline usando os parâmetros do melhor trial."""
    fixed_trial = optuna.trial.FixedTrial(params)
    return build_optuna_pipeline(
        fixed_trial,
        model_name=model_name,
        n_features=n_features,
        random_state=random_state,
        search_version=search_version,
    )
