"""Diagnóstico do erro por faixa de RUL e estágio da trajetória."""

from __future__ import annotations

import numpy as np
import pandas as pd


RUL_BINS = [-np.inf, 30, 60, 100, np.inf]
RUL_LABELS = ["Até 30 ciclos", "31 a 60 ciclos", "61 a 100 ciclos", "Acima de 100 ciclos"]

STAGE_BINS = [0, 0.25, 0.50, 0.75, 1.0]
STAGE_LABELS = ["Início", "Primeira metade", "Segunda metade", "Final"]

CYCLE_BINS = [-np.inf, 50, 100, 150, 200, 250, np.inf]
CYCLE_LABELS = [
    "Até 50 ciclos",
    "51 a 100 ciclos",
    "101 a 150 ciclos",
    "151 a 200 ciclos",
    "201 a 250 ciclos",
    "Acima de 250 ciclos",
]


def enrich_error_profile(
    predictions: pd.DataFrame,
    rul_column: str = "y_true",
    prediction_column: str = "y_predito",
    group_column: str = "group",
    cycle_column: str = "cycle",
) -> pd.DataFrame:
    """Adiciona erro, faixa de RUL e estágio relativo da trajetória."""

    required = {rul_column, prediction_column, group_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Colunas ausentes para o perfil de erro: {sorted(missing)}")

    profile = predictions.copy()
    profile["erro"] = profile[prediction_column] - profile[rul_column]
    profile["faixa_rul"] = pd.cut(
        profile[rul_column], bins=RUL_BINS, labels=RUL_LABELS, include_lowest=True
    )

    if cycle_column in profile.columns:
        max_cycle = profile.groupby(group_column, observed=True)[cycle_column].transform("max")
        profile["estagio_proporcional"] = profile[cycle_column] / max_cycle
        profile["estagio_ciclo"] = pd.cut(
            profile["estagio_proporcional"],
            bins=STAGE_BINS,
            labels=STAGE_LABELS,
            include_lowest=True,
        )
    else:
        profile["estagio_proporcional"] = np.nan
        profile["estagio_ciclo"] = pd.Series(pd.NA, index=profile.index, dtype="object")

    return profile


def summarize_error_profile(
    predictions: pd.DataFrame,
    category_column: str,
    group_column: str = "group",
) -> pd.DataFrame:
    """Calcula métricas médias por motor dentro de cada categoria."""

    required = {category_column, group_column, "erro"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no resumo do erro: {sorted(missing)}")

    per_group = (
        predictions.dropna(subset=[category_column])
        .groupby([category_column, group_column], observed=True)
        .agg(
            erro_medio=("erro", "mean"),
            mae=("erro", lambda values: np.mean(np.abs(values))),
            rmse=("erro", lambda values: np.sqrt(np.mean(values**2))),
        )
        .reset_index()
    )

    summary = (
        per_group.groupby(category_column, observed=True)
        .agg(
            erro_medio_motor=("erro_medio", "mean"),
            mae_medio_motor=("mae", "mean"),
            rmse_medio_motor=("rmse", "mean"),
            superestimacao_percentual=("erro_medio", lambda values: np.mean(values > 0) * 100),
            subestimacao_percentual=("erro_medio", lambda values: np.mean(values < 0) * 100),
            motores=(group_column, "nunique"),
        )
        .reset_index()
    )

    return summary


def enrich_observable_error_profile(
    predictions: pd.DataFrame,
    prediction_column: str = "y_predito",
    cycle_column: str = "cycle",
) -> pd.DataFrame:
    """Adiciona faixas baseadas apenas em variáveis disponíveis na previsão.

    O RUL previsto é usado para organizar o diagnóstico depois que a previsão
    foi produzida. O RUL real permanece apenas na coluna de erro, para medir o
    desempenho; ele não participa da criação das faixas observáveis.
    """

    required = {prediction_column, cycle_column, "erro"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no perfil observável: {sorted(missing)}")

    profile = predictions.copy()
    profile["faixa_cycle"] = pd.cut(
        profile[cycle_column], bins=CYCLE_BINS, labels=CYCLE_LABELS, include_lowest=True
    )
    profile["faixa_rul_predito"] = pd.cut(
        profile[prediction_column],
        bins=RUL_BINS,
        labels=RUL_LABELS,
        include_lowest=True,
    )
    return profile
