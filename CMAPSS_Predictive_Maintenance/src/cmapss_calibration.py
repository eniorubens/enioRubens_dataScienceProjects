"""Calibração pós-modelo para previsões de RUL.

A calibração implementada aqui corrige um deslocamento sistemático das
previsões. O valor aprendido é a média de ``previsão - valor real`` em
previsões fora da amostra. A correção é subtraída das previsões futuras.

Para que a avaliação permaneça honesta, a função principal usa uma partição
externa por motor e aprende a correção em previsões fora da amostra produzidas
por uma partição interna, também separada por motor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class CalibrationResult:
    """Resultado das previsões externas e das correções aprendidas."""

    predictions: pd.DataFrame
    summary: pd.DataFrame


def signed_bias(y_true, y_pred) -> float:
    """Retorna o erro médio assinado: previsão menos valor real."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.mean(predicted - actual))


def apply_bias_calibration(y_pred, bias: float) -> np.ndarray:
    """Aplica uma correção aditiva, reduzindo o viés estimado."""

    return np.asarray(y_pred, dtype=float) - float(bias)


def _inner_oof_bias(estimator, X, y, groups, n_splits: int) -> float:
    """Estima o viés com previsões fora da amostra do subconjunto de treino."""

    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=float)

    for inner_train, inner_valid in splitter.split(X, y, groups):
        fitted = clone(estimator)
        fitted.fit(X.iloc[inner_train], y.iloc[inner_train])
        oof[inner_valid] = fitted.predict(X.iloc[inner_valid])

    if np.isnan(oof).any():
        raise RuntimeError("A validação interna não produziu todas as previsões.")

    return signed_bias(y, oof)


def nested_group_calibration(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    estimator,
    n_splits: int = 5,
    inner_splits: int = 4,
) -> CalibrationResult:
    """Avalia um modelo com calibração aprendida sem vazamento.

    Em cada partição externa, o modelo é treinado no conjunto de treino. Antes
    disso, o viés é estimado por previsões fora da amostra em partições
    internas do próprio conjunto de treino. A correção é aplicada somente à
    partição externa de validação.
    """

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y e groups precisam ter o mesmo número de linhas.")

    outer = GroupKFold(n_splits=n_splits)
    rows: list[pd.DataFrame] = []

    for fold, (train_idx, valid_idx) in enumerate(outer.split(X, y, groups), start=1):
        train_groups = groups.iloc[train_idx]
        unique_train_groups = train_groups.nunique()
        usable_inner_splits = min(inner_splits, unique_train_groups)
        if usable_inner_splits < 2:
            raise ValueError("São necessários pelo menos dois motores no treino interno.")

        bias = _inner_oof_bias(
            estimator,
            X.iloc[train_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
            train_groups.reset_index(drop=True),
            n_splits=usable_inner_splits,
        )

        fitted = clone(estimator)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        raw = fitted.predict(X.iloc[valid_idx])
        calibrated = apply_bias_calibration(raw, bias)

        prediction_frame = pd.DataFrame(
            {
                "index": valid_idx,
                "fold": fold,
                "group": groups.iloc[valid_idx].to_numpy(),
                "y_true": y.iloc[valid_idx].to_numpy(),
                "y_predito": raw,
                "y_calibrado": calibrated,
                "bias_aprendido": bias,
            }
        )
        if "cycle" in X.columns:
            prediction_frame["cycle"] = X.iloc[valid_idx]["cycle"].to_numpy()
        rows.append(prediction_frame)

    predictions = pd.concat(rows, ignore_index=True).sort_values("index")
    summary = calibration_summary(predictions)
    return CalibrationResult(predictions=predictions, summary=summary)


def calibration_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Resume as métricas por motor antes e depois da calibração."""

    required = {"y_true", "y_predito", "y_calibrado"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Colunas ausentes para o resumo: {sorted(missing)}")

    data = predictions.copy()
    group_column = "group" if "group" in data.columns else None

    if group_column is None:
        data["group"] = 0
        group_column = "group"

    data["erro_original"] = data["y_predito"] - data["y_true"]
    data["erro_calibrado"] = data["y_calibrado"] - data["y_true"]

    grouped = data.groupby(group_column, sort=False).agg(
        erro_original=("erro_original", "mean"),
        erro_calibrado=("erro_calibrado", "mean"),
        mae_original=("erro_original", lambda values: np.mean(np.abs(values))),
        mae_calibrado=("erro_calibrado", lambda values: np.mean(np.abs(values))),
        rmse_original=("erro_original", lambda values: np.sqrt(np.mean(values**2))),
        rmse_calibrado=("erro_calibrado", lambda values: np.sqrt(np.mean(values**2))),
    )

    rows = []
    for label, error_column, mae_column, rmse_column in (
        ("Original", "erro_original", "mae_original", "rmse_original"),
        ("Calibrado", "erro_calibrado", "mae_calibrado", "rmse_calibrado"),
    ):
        motor_errors = grouped[error_column]
        rows.append(
            {
                "configuracao": label,
                "mae": float(grouped[mae_column].mean()),
                "rmse": float(grouped[rmse_column].mean()),
                "erro_medio": float(motor_errors.mean()),
                "superestimacao_percentual": float(np.mean(motor_errors > 0) * 100),
                "subestimacao_percentual": float(np.mean(motor_errors < 0) * 100),
                "motores": int(len(grouped)),
            }
        )

    return pd.DataFrame(rows)
