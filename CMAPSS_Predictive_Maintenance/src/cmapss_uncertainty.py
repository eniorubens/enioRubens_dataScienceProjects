from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_groupwise_interval(
    oof_predictions: pd.DataFrame,
    coverage: float = 0.90,
) -> tuple[float, pd.DataFrame]:
    """Calibra uma largura de intervalo usando resíduos balanceados por motor."""
    if not 0 < coverage < 1:
        raise ValueError("coverage deve estar entre 0 e 1.")

    data = oof_predictions.copy()
    data["erro_absoluto"] = (data["rul"] - data["rul_predito"]).abs()
    motor_scores = (
        data.groupby("unit")["erro_absoluto"]
        .quantile(coverage)
        .rename("score_motor")
        .reset_index()
    )
    half_width = float(np.quantile(motor_scores["score_motor"], coverage))
    return half_width, motor_scores


def add_prediction_interval(
    predictions: pd.DataFrame,
    half_width: float,
    prediction_column: str = "rul_predito",
) -> pd.DataFrame:
    """Adiciona limites inferior e superior ao quadro de previsões."""
    result = predictions.copy()
    result["rul_inferior"] = (
        result[prediction_column] - half_width
    ).clip(lower=0)
    result["rul_superior"] = result[prediction_column] + half_width
    return result


def evaluate_interval_coverage(
    predictions_with_interval: pd.DataFrame,
    actual_column: str = "rul",
) -> dict[str, float]:
    """Resume cobertura e largura média do intervalo."""
    data = predictions_with_interval
    covered = data[actual_column].between(
        data["rul_inferior"], data["rul_superior"]
    )
    return {
        "cobertura_observada": float(covered.mean()),
        "largura_media": float(
            (data["rul_superior"] - data["rul_inferior"]).mean()
        ),
        "observacoes": float(len(data)),
    }
