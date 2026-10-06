"""Faixas empíricas de erro com motores distintos para calibração e avaliação."""
import numpy as np
import pandas as pd


def evaluate_oof_intervals(predictions, coverage=0.90, random_state=42):
    """Divide cada fold OOF ao meio por motor; balanceia a calibração por motor.

    Os modelos OOF foram otimizados no desenvolvimento: a cobertura é exploratória.
    Não oferece garantia conformal nem intervalo validado do modelo refitado.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage deve estar entre zero e um.")
    required = ["fold", "unit", "cycle", "rul_real", "rul_predito"]
    data = predictions[required].copy()
    if data.isna().any().any() or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Previsões devem ser finitas e completas.")
    if data.duplicated(["unit", "cycle"]).any() or data.groupby("unit").fold.nunique().max() != 1:
        raise ValueError("Cada ciclo deve ser único e cada motor pertencer a um fold.")
    if (data.rul_real < 0).any():
        raise ValueError("RUL real deve ser não negativo.")
    rng = np.random.default_rng(random_state)
    frames, widths, assignments = [], [], []
    for fold, group in data.groupby("fold", sort=True):
        units = rng.permutation(np.sort(group.unit.unique()))
        if len(units) < 4:
            raise ValueError("São necessários pelo menos quatro motores por fold.")
        calibration_units = units[:len(units) // 2]
        evaluation_units = units[len(units) // 2:]
        calibration = group[group.unit.isin(calibration_units)].copy()
        evaluation = group[group.unit.isin(evaluation_units)].copy()
        calibration["score"] = (calibration.rul_real - calibration.rul_predito).abs()
        calibration["weight"] = 1 / calibration.groupby("unit").unit.transform("size")
        ordered = calibration.sort_values("score")
        cumulative = ordered.weight.cumsum() / ordered.weight.sum()
        position = min(np.searchsorted(cumulative.to_numpy(), coverage), len(ordered) - 1)
        width = float(ordered.score.iloc[position])
        evaluation["meia_largura"] = width
        evaluation["rul_inferior"] = (evaluation.rul_predito - width).clip(lower=0)
        evaluation["rul_superior"] = (evaluation.rul_predito + width).clip(lower=0)
        evaluation["coberto"] = evaluation.rul_real.between(evaluation.rul_inferior, evaluation.rul_superior)
        evaluation["largura"] = evaluation.rul_superior - evaluation.rul_inferior
        evaluation["abaixo"] = evaluation.rul_real < evaluation.rul_inferior
        evaluation["acima"] = evaluation.rul_real > evaluation.rul_superior
        frames.append(evaluation)
        widths.append(dict(fold=fold, meia_largura=width, motores_calibracao=len(calibration_units),
                           motores_avaliacao=len(evaluation_units), referencia=coverage))
        assignments.extend(dict(fold=fold, unit=int(u), papel=role)
                           for role, subset in [("Calibração", calibration_units), ("Avaliação", evaluation_units)]
                           for u in subset)
    intervals = pd.concat(frames, ignore_index=True).sort_values(["unit", "cycle"])
    per_engine = intervals.groupby("unit").agg(
        cobertura=("coberto", "mean"), largura_media=("largura", "mean"),
        cobertura_trajetoria=("coberto", "all"), abaixo=("abaixo", "mean"), acima=("acima", "mean"))
    return intervals, pd.DataFrame(widths), per_engine.reset_index(), pd.DataFrame(assignments)


def interval_summary_by_predicted_rul(intervals):
    """Agrega primeiro por motor dentro de cada faixa de RUL previsto."""
    data = intervals.copy()
    labels = ["Até 30 ciclos", "31 a 60 ciclos", "61 a 100 ciclos", "Acima de 100 ciclos"]
    data["faixa"] = pd.cut(data.rul_predito, [-np.inf, 30, 60, 100, np.inf], labels=labels)
    engine = data.groupby(["faixa", "unit"], observed=True).agg(
        cobertura=("coberto", "mean"), largura_media=("largura", "mean"))
    return engine.groupby("faixa", observed=True).agg(
        cobertura=("cobertura", "mean"), largura_media=("largura_media", "mean"),
        motores=("cobertura", "size")).reindex(labels).reset_index()

