from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold


def generate_oof_rul_predictions(
    df: pd.DataFrame,
    features: list[str],
    model,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Gera previsões out-of-fold, mantendo cada motor em um único grupo."""
    predictions = np.full(len(df), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    X = df[features]
    y = df["rul"]
    groups = df["unit"]

    for train_index, validation_index in splitter.split(X, y, groups):
        fitted = clone(model)
        fitted.fit(X.iloc[train_index], y.iloc[train_index])
        predictions[validation_index] = fitted.predict(X.iloc[validation_index])

    if np.isnan(predictions).any():
        raise ValueError("Existem observações sem previsão out-of-fold.")

    result = df[["unit", "cycle", "rul"]].copy()
    result["rul_predito"] = predictions
    return result.sort_values(["unit", "cycle"]).reset_index(drop=True)


@dataclass(frozen=True)
class WarningEvent:
    unit: int
    failure_cycle: int
    alert_cycle: int | None
    classification: str
    lead_time: float | None


def _first_confirmed_alert(
    unit_df: pd.DataFrame,
    threshold: float,
    confirmation_cycles: int,
) -> int | None:
    alert_candidate = (unit_df["rul_predito"].to_numpy() <= threshold)
    streak = 0
    for position, candidate in enumerate(alert_candidate):
        streak = streak + 1 if candidate else 0
        if streak >= confirmation_cycles:
            return int(unit_df.iloc[position]["cycle"])
    return None


def evaluate_warning_policy(
    predictions: pd.DataFrame,
    threshold: float,
    warning_horizon: int,
    confirmation_cycles: int = 3,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Classifica alertas como oportunos, prematuros ou não detectados."""
    events: list[WarningEvent] = []
    for unit, unit_df in predictions.groupby("unit", sort=True):
        unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
        failure_cycle = int(unit_df["cycle"].max())
        horizon_start = failure_cycle - warning_horizon
        alert_cycle = _first_confirmed_alert(
            unit_df,
            threshold=threshold,
            confirmation_cycles=confirmation_cycles,
        )

        if alert_cycle is None:
            classification = "falha_nao_detectada"
            lead_time = None
        elif alert_cycle < horizon_start:
            classification = "alerta_prematuro"
            lead_time = float(failure_cycle - alert_cycle)
        else:
            classification = "alerta_oportuno"
            lead_time = float(failure_cycle - alert_cycle)

        events.append(WarningEvent(
            unit=int(unit),
            failure_cycle=failure_cycle,
            alert_cycle=alert_cycle,
            classification=classification,
            lead_time=lead_time,
        ))

    events_df = pd.DataFrame([event.__dict__ for event in events])
    counts = events_df["classification"].value_counts()
    n_units = len(events_df)
    timely = events_df.loc[
        events_df["classification"] == "alerta_oportuno", "lead_time"
    ]
    metrics = {
        "motores": float(n_units),
        "alertas_oportunos": float(counts.get("alerta_oportuno", 0)),
        "alertas_prematuros": float(counts.get("alerta_prematuro", 0)),
        "falhas_nao_detectadas": float(counts.get("falha_nao_detectada", 0)),
        "taxa_alerta_oportuna": float(counts.get("alerta_oportuno", 0) / n_units),
        "taxa_alerta_prematuro": float(counts.get("alerta_prematuro", 0) / n_units),
        "taxa_falha_nao_detectada": float(
            counts.get("falha_nao_detectada", 0) / n_units
        ),
        "lead_time_medio_oportuno": float(timely.mean()) if not timely.empty else np.nan,
        "lead_time_mediano_oportuno": float(timely.median()) if not timely.empty else np.nan,
    }
    return events_df, metrics


def evaluate_warning_grid(
    predictions: pd.DataFrame,
    thresholds: list[float],
    warning_horizon: int,
    confirmation_cycles: int = 3,
) -> pd.DataFrame:
    """Avalia vários limiares de alerta sob o mesmo horizonte operacional."""
    rows = []
    for threshold in thresholds:
        _, metrics = evaluate_warning_policy(
            predictions,
            threshold=threshold,
            warning_horizon=warning_horizon,
            confirmation_cycles=confirmation_cycles,
        )
        rows.append({
            "limiar_alerta": threshold,
            "horizonte_alerta": warning_horizon,
            "ciclos_confirmacao": confirmation_cycles,
            **metrics,
        })
    return pd.DataFrame(rows)
