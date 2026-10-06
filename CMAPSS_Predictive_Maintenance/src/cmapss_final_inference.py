"""Contrato de entrada e inferência do Extra Trees final FD001."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from cmapss_temporal_features import add_causal_temporal_features


def load_final_bundle(path):
    """Carrega apenas o artefato local confiável produzido pelo projeto."""
    bundle = joblib.load(Path(path))
    if bundle.get("schema_version") != 1 or bundle.get("dataset") != "FD001":
        raise ValueError("Artefato incompatível com o contrato FD001.")
    return bundle


def prepare_history(history, bundle):
    """Requer histórico contínuo desde o ciclo 1; descarta colunas extras."""
    required = ["unit", *bundle["base_features"]]
    missing = sorted(set(required) - set(history.columns))
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    if history.empty or not history.columns.is_unique:
        raise ValueError("Histórico vazio ou com nomes de colunas duplicados.")
    frame = history[required].copy()
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in required):
        raise ValueError("Todas as colunas obrigatórias devem ser numéricas.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Valores ausentes ou não finitos no histórico.")
    for name in ["unit", "cycle"]:
        if (frame[name] < 1).any() or (frame[name] % 1 != 0).any():
            raise ValueError(f"{name} deve conter inteiros positivos.")
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError("Ciclo duplicado por motor.")
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    for _, group in frame.groupby("unit"):
        if group.cycle.iloc[0] != 1 or not group.cycle.diff().dropna().eq(1).all():
            raise ValueError("Forneça ciclos consecutivos desde o ciclo 1 de cada motor.")
    return add_causal_temporal_features(
        frame, sensors=bundle["temporal_sources"], windows=bundle["windows"])


def predict_history(history, bundle, latest_only=True):
    """Prevê com a mesma engenharia e ordem de atributos usadas no treino."""
    frame = prepare_history(history, bundle)
    if latest_only:
        frame = frame.groupby("unit", sort=True).tail(1)
    result = frame[["unit", "cycle"]].copy()
    result["rul_predito"] = bundle["pipeline"].predict(frame[bundle["features"]])
    return result.reset_index(drop=True)

