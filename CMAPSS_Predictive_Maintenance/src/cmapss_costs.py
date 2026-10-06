from __future__ import annotations

from typing import Mapping

import pandas as pd

from cmapss_early_warning import evaluate_warning_policy


def evaluate_policy_cost(
    events: pd.DataFrame,
    costs: Mapping[str, float],
    scenario: str,
    threshold: float,
    warning_horizon: int,
    confirmation_cycles: int,
) -> dict[str, float | str]:
    """Calcula o custo de uma política de alerta em um cenário informado."""
    required = {
        "alerta_oportuno",
        "alerta_prematuro",
        "falha_nao_detectada",
    }
    missing = required.difference(costs)
    if missing:
        raise ValueError(f"Custos ausentes: {sorted(missing)}")

    counts = events["classification"].value_counts()
    timely = int(counts.get("alerta_oportuno", 0))
    premature = int(counts.get("alerta_prematuro", 0))
    missed = int(counts.get("falha_nao_detectada", 0))
    n_units = len(events)
    total_cost = (
        timely * costs["alerta_oportuno"]
        + premature * costs["alerta_prematuro"]
        + missed * costs["falha_nao_detectada"]
    )
    return {
        "cenario": scenario,
        "limiar_alerta": threshold,
        "horizonte_alerta": warning_horizon,
        "ciclos_confirmacao": confirmation_cycles,
        "alertas_oportunos": timely,
        "alertas_prematuros": premature,
        "falhas_nao_detectadas": missed,
        "custo_alerta_oportuno": costs["alerta_oportuno"],
        "custo_alerta_prematuro": costs["alerta_prematuro"],
        "custo_falha_nao_detectada": costs["falha_nao_detectada"],
        "custo_total": float(total_cost),
        "custo_medio_motor": float(total_cost / n_units),
    }


def evaluate_cost_grid(
    predictions: pd.DataFrame,
    thresholds: list[float],
    warning_horizon: int,
    confirmation_cycles: int,
    scenarios: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Compara limiares em vários cenários de custo."""
    rows = []
    for threshold in thresholds:
        events, _ = evaluate_warning_policy(
            predictions,
            threshold=threshold,
            warning_horizon=warning_horizon,
            confirmation_cycles=confirmation_cycles,
        )
        for scenario, costs in scenarios.items():
            rows.append(evaluate_policy_cost(
                events,
                costs=costs,
                scenario=scenario,
                threshold=threshold,
                warning_horizon=warning_horizon,
                confirmation_cycles=confirmation_cycles,
            ))
    return pd.DataFrame(rows)
