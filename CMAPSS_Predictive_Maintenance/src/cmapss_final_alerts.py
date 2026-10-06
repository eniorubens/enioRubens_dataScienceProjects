"""Avaliação descritiva de alertas do modelo final sobre trajetórias OOF."""
import pandas as pd


def evaluate_final_alerts(predictions, thresholds=(10, 20, 30, 40, 50),
                          horizon=30, minimum_lead=5, confirmation=3):
    """Conta o primeiro alerta confirmado por motor, incluindo alertas tardios."""
    if not 0 <= minimum_lead <= horizon or confirmation < 1:
        raise ValueError("Horizonte, antecedência ou confirmação inválidos.")
    if predictions.duplicated(["unit", "cycle"]).any():
        raise ValueError("Ciclos duplicados por motor.")
    rows = []
    for threshold in thresholds:
        for unit, group in predictions.groupby("unit"):
            group = group.sort_values("cycle")
            failure = group.cycle + group.rul_real
            if failure.max() - failure.min() > 1e-6:
                raise ValueError("RUL não define uma falha consistente por motor.")
            failure_cycle = float(failure.iloc[0])
            streak, previous, alert = 0, None, None
            for cycle, prediction in group[["cycle", "rul_predito"]].itertuples(index=False, name=None):
                if previous is not None and cycle != previous + 1:
                    streak = 0
                streak = streak + 1 if prediction <= threshold else 0
                previous = cycle
                if streak >= confirmation:
                    alert = int(cycle)
                    break
            lead = failure_cycle - alert if alert is not None else float("nan")
            status = ("Não detectada" if alert is None else
                      "Prematuro" if lead > horizon else
                      "Tardio" if lead < minimum_lead else "Oportuno")
            rows.append(dict(unit=unit, limiar=threshold, ciclo_alerta=alert,
                             antecedencia=lead, resultado=status))
    events = pd.DataFrame(rows)
    summary = events.groupby(["limiar", "resultado"]).size().unstack(fill_value=0)
    summary = summary.reindex(columns=["Prematuro", "Oportuno", "Tardio", "Não detectada"], fill_value=0)
    summary["motores"] = summary.sum(axis=1)
    return events, summary.reset_index()

