"""Funções reutilizáveis para o primeiro baseline de RUL do NASA C-MAPSS."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from cmapss_eda import SENSOR_COLUMNS, feature_columns


def add_rul_label(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona o RUL de cada ciclo usando o último ciclo observado no motor."""
    result = df.copy()
    terminal_cycle = result.groupby("unit")["cycle"].transform("max")
    result["rul"] = terminal_cycle - result["cycle"]
    return result


def split_by_unit(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide trajetórias inteiras entre treino e validação."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, validation_index = next(
        splitter.split(df, groups=df["unit"])
    )
    return (
        df.iloc[train_index].reset_index(drop=True),
        df.iloc[validation_index].reset_index(drop=True),
    )


def candidate_rul_features(include_settings: bool = True) -> list[str]:
    """Retorna as variáveis candidatas, incluindo o ciclo de operação."""
    return ["cycle", *feature_columns(include_settings=include_settings)]


def nonconstant_features(
    df: pd.DataFrame,
    candidates: list[str] | None = None,
) -> list[str]:
    """Mantém apenas variáveis com mais de um valor no conjunto de referência."""
    if candidates is None:
        candidates = candidate_rul_features()
    return [column for column in candidates if df[column].nunique(dropna=False) > 1]


def make_random_forest_regressor(random_state: int = 42) -> RandomForestRegressor:
    """Cria o modelo de referência usado no primeiro baseline."""
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def evaluate_rul(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula erros em ciclos de operação."""
    return {
        "mae_ciclos": float(mean_absolute_error(y_true, y_pred)),
        "rmse_ciclos": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
    }


def median_rul_prediction(y_train: pd.Series, size: int) -> np.ndarray:
    """Cria a previsão ingênua que repete a mediana do RUL de treinamento."""
    return np.full(size, float(y_train.median()))


def screening_estimators(random_state: int = 42) -> dict[str, object]:
    """Cria pipelines iniciais para a triagem de famílias de regressão."""
    return {
        "Mediana": DummyRegressor(strategy="median"),
        "Ridge + StandardScaler": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "ElasticNet + StandardScaler": make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                max_iter=10_000,
                random_state=random_state,
            ),
        ),
        "SGD + StandardScaler": make_pipeline(
            StandardScaler(),
            SGDRegressor(
                loss="squared_error",
                penalty="elasticnet",
                alpha=0.0001,
                l1_ratio=0.15,
                max_iter=2_000,
                tol=1e-3,
                random_state=random_state,
            ),
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=random_state,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=1,
            random_state=random_state,
            verbosity=0,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=random_state,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def group_cv_screening(
    df: pd.DataFrame,
    features: list[str],
    estimators: dict[str, object],
    target: str = "rul",
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia estimadores com folds inteiros e métricas médias por motor."""
    splitter = GroupKFold(n_splits=n_splits)
    X = df[features]
    y = df[target]
    groups = df["unit"]
    fold_records: list[dict[str, float | int | str]] = []

    for model_name, estimator in estimators.items():
        for fold, (train_idx, validation_idx) in enumerate(
            splitter.split(X, y, groups),
            start=1,
        ):
            fitted = clone(estimator)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            prediction = fitted.predict(X.iloc[validation_idx])
            validation = df.iloc[validation_idx][["unit"]].copy()
            errors = y.iloc[validation_idx].to_numpy() - prediction
            validation["absolute_error"] = np.abs(errors)
            validation["squared_error"] = errors**2
            unit_mae = validation.groupby("unit")["absolute_error"].mean()
            unit_rmse = np.sqrt(
                validation.groupby("unit")["squared_error"].mean()
            )
            fold_records.append({
                "modelo": model_name,
                "fold": fold,
                "mae_por_motor": float(unit_mae.mean()),
                "rmse_por_motor": float(unit_rmse.mean()),
                "pior_mae_motor": float(unit_mae.max()),
                "mae_por_linha": float(np.mean(np.abs(errors))),
            })

    fold_results = pd.DataFrame(fold_records)
    summary = (
        fold_results.groupby("modelo")
        .agg(
            mae_medio_motor=("mae_por_motor", "mean"),
            mae_desvio_motor=("mae_por_motor", "std"),
            rmse_medio_motor=("rmse_por_motor", "mean"),
            rmse_desvio_motor=("rmse_por_motor", "std"),
            pior_mae_medio=("pior_mae_motor", "mean"),
        )
        .sort_values("mae_medio_motor")
        .reset_index()
    )
    summary.insert(0, "ordem_mae", range(1, len(summary) + 1))
    return fold_results, summary
