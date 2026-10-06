"""Construção de atributos temporais sem utilizar observações futuras."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def add_causal_temporal_features(
    df: pd.DataFrame,
    sensors: Iterable[str],
    windows: Iterable[int] = (5, 10, 20, 40, 80),
    group_column: str = "unit",
    time_column: str = "cycle",
) -> pd.DataFrame:
    """Adiciona estatísticas históricas calculadas até o ciclo atual.

    A observação do ciclo atual é conhecida no momento da previsão e, por isso,
    participa das janelas. Nenhuma linha posterior é consultada. O cálculo é
    reiniciado para cada motor e o resultado retorna à ordem original.
    """

    required = {group_column, time_column}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes para atributos temporais: {sorted(missing)}")

    sensors = list(sensors)
    windows = tuple(sorted(set(int(window) for window in windows)))
    if not sensors:
        raise ValueError("Pelo menos um sensor deve ser informado.")
    if not windows or any(window < 2 for window in windows):
        raise ValueError("As janelas temporais precisam ser inteiros maiores ou iguais a 2.")

    missing_sensors = set(sensors).difference(df.columns)
    if missing_sensors:
        raise KeyError(f"Sensores ausentes: {sorted(missing_sensors)}")

    result = df.copy()
    result["__ordem_original"] = range(len(result))
    result = result.sort_values([group_column, time_column, "__ordem_original"])

    temporal_columns: dict[str, pd.Series] = {}
    for sensor in sensors:
        grouped_sensor = result.groupby(group_column, sort=False)[sensor]
        temporal_columns[f"{sensor}_delta"] = grouped_sensor.diff().fillna(0.0)

        for window in windows:
            temporal_columns[f"{sensor}_media_{window}"] = grouped_sensor.transform(
                lambda values, size=window: values.rolling(size, min_periods=1).mean()
            )
            temporal_columns[f"{sensor}_desvio_{window}"] = grouped_sensor.transform(
                lambda values, size=window: values.rolling(size, min_periods=1).std(ddof=0)
            ).fillna(0.0)

    temporal_frame = pd.DataFrame(temporal_columns, index=result.index)
    result = pd.concat([result, temporal_frame], axis=1)
    result = result.sort_values("__ordem_original").drop(columns="__ordem_original")
    return result.reset_index(drop=True)


def temporal_feature_names(
    sensors: Iterable[str],
    windows: Iterable[int] = (5, 10, 20, 40, 80),
) -> list[str]:
    """Retorna os nomes dos atributos criados pela função temporal."""

    names: list[str] = []
    for sensor in sensors:
        names.append(f"{sensor}_delta")
        for window in sorted(set(int(value) for value in windows)):
            names.extend([f"{sensor}_media_{window}", f"{sensor}_desvio_{window}"])
    return names


def add_causal_temporal_quantiles(
    df: pd.DataFrame,
    sensors: Iterable[str],
    windows: Iterable[int] = (10, 20, 40, 80),
    quantiles: Iterable[float] = (0.25, 0.50, 0.75),
    group_column: str = "unit",
    time_column: str = "cycle",
) -> pd.DataFrame:
    """Adiciona quartis móveis usando somente o presente e o passado."""

    required = {group_column, time_column}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes para quartis temporais: {sorted(missing)}")

    sensors = list(sensors)
    windows = tuple(sorted(set(int(window) for window in windows)))
    quantiles = tuple(float(value) for value in quantiles)
    if not sensors or not windows or not quantiles:
        raise ValueError("Sensores, janelas e quantis não podem ser vazios.")
    if any(window < 2 for window in windows):
        raise ValueError("As janelas temporais precisam ser maiores ou iguais a 2.")
    if any(value <= 0 or value >= 1 for value in quantiles):
        raise ValueError("Os quantis devem estar estritamente entre 0 e 1.")

    missing_sensors = set(sensors).difference(df.columns)
    if missing_sensors:
        raise KeyError(f"Sensores ausentes: {sorted(missing_sensors)}")

    result = df.copy()
    result["__ordem_original"] = range(len(result))
    result = result.sort_values([group_column, time_column, "__ordem_original"])

    quantile_columns: dict[str, pd.Series] = {}
    for sensor in sensors:
        grouped_sensor = result.groupby(group_column, sort=False)[sensor]
        for window in windows:
            for quantile in quantiles:
                quantile_label = f"q{int(quantile * 100):02d}"
                quantile_columns[f"{sensor}_{quantile_label}_{window}"] = grouped_sensor.transform(
                    lambda values, size=window, q=quantile: values.rolling(
                        size, min_periods=1
                    ).quantile(q)
                )

    quantile_frame = pd.DataFrame(quantile_columns, index=result.index)
    result = pd.concat([result, quantile_frame], axis=1)
    result = result.sort_values("__ordem_original").drop(columns="__ordem_original")
    return result.reset_index(drop=True)


def temporal_quantile_feature_names(
    sensors: Iterable[str],
    windows: Iterable[int] = (10, 20, 40, 80),
    quantiles: Iterable[float] = (0.25, 0.50, 0.75),
) -> list[str]:
    """Retorna os nomes dos atributos de quartis móveis."""

    names: list[str] = []
    for sensor in sensors:
        for window in sorted(set(int(value) for value in windows)):
            for quantile in quantiles:
                quantile_label = f"q{int(float(quantile) * 100):02d}"
                names.append(f"{sensor}_{quantile_label}_{window}")
    return names
