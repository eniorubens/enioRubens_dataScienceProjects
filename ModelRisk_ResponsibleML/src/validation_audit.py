"""Calibração, limiares e auditoria de grupos no split de desenvolvimento.

As funções deste módulo usam somente a partição oficial de treino, separada em
ajuste e calibração/validação. O teste oficial permanece intocado para a etapa
final e não participa de seleção de modelo ou limiar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from adult_data import AUDIT_COLUMNS, COLUMNS, TARGET, load_adult
from baseline_models import (
    CATEGORICAL_COLUMNS,
    PREDICTOR_COLUMNS,
    RANDOM_STATE,
    _predict_probabilities,
    make_models,
    make_preprocessor,
)


DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.10, 0.91, 0.05), 2))
LOW_SUPPORT_THRESHOLD = 30


def _model_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Retorna preditores, alvo e atributos de auditoria do treino oficial."""
    official_train = frame.loc[frame["split"] == "train_official"].copy()
    y = (official_train[TARGET] == ">50K").astype(int)
    x = official_train[PREDICTOR_COLUMNS].copy()
    x[CATEGORICAL_COLUMNS] = x[CATEGORICAL_COLUMNS].astype(object)
    x[CATEGORICAL_COLUMNS] = x[CATEGORICAL_COLUMNS].mask(
        x[CATEGORICAL_COLUMNS].isna(), np.nan
    )
    audit = official_train[AUDIT_COLUMNS].copy()
    return x, y, audit


def _safe_rate(numerator: int, denominator: int) -> float:
    """Calcula uma taxa sem produzir divisão inválida para grupos vazios."""
    return float(numerator / denominator) if denominator else float("nan")


def _observation_keys(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Cria chaves estáveis para diagnosticar duplicatas entre partições."""
    keyed = x.astype("string").fillna("(missing)").copy()
    keyed["__target"] = y.astype("string").to_numpy()
    return pd.util.hash_pandas_object(keyed, index=False)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Calcula intervalo de Wilson para uma proporção binomial."""
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def _classification_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    """Calcula métricas globais para um limiar explicitamente informado."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "selection_rate": float(predictions.mean()),
    }


def group_metric_rows(
    y_true: pd.Series,
    probabilities: np.ndarray,
    audit: pd.DataFrame,
    model: str,
    threshold: float,
) -> list[dict]:
    """Calcula taxas de decisão e erro com intervalos de Wilson por grupo."""
    predictions = (probabilities >= threshold).astype(int)
    work = audit.copy()
    work["sex_race"] = work["sex"].astype(str) + " × " + work["race"].astype(str)
    work["target"] = y_true.to_numpy()
    work["prediction"] = predictions
    rows = []
    for audit_column in ["sex", "race", "sex_race"]:
        for group, group_frame in work.groupby(audit_column, dropna=False):
            target = group_frame["target"].to_numpy()
            predicted = group_frame["prediction"].to_numpy()
            total = int(len(group_frame))
            positives = int(target.sum())
            negatives = total - positives
            predicted_positives = int(predicted.sum())
            true_positives = int(((target == 1) & (predicted == 1)).sum())
            false_positives = int(((target == 0) & (predicted == 1)).sum())
            false_negatives = int(((target == 1) & (predicted == 0)).sum())
            rates = {
                "selection_rate": (predicted_positives, total),
                "tpr": (true_positives, positives),
                "fpr": (false_positives, negatives),
                "fnr": (false_negatives, positives),
                "ppv": (true_positives, predicted_positives),
            }
            for metric, (successes, denominator) in rates.items():
                low, high = wilson_interval(successes, denominator)
                rows.append(
                    {
                        "model": model,
                        "threshold": threshold,
                        "audit_column": audit_column,
                        "group": str(group),
                        "count": total,
                        "denominator": denominator,
                        "metric": metric,
                        "estimate": _safe_rate(successes, denominator),
                        "ci_low": low,
                        "ci_high": high,
                        "low_support": denominator < LOW_SUPPORT_THRESHOLD,
                    }
                )
    return rows


def run_validation_audit(
    frame: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, pd.DataFrame | dict]:
    """Ajusta candidatos e retorna tabelas de validação sem usar o teste."""
    x, y, audit = _model_frame(frame)
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    audit_cal = audit.loc[x_cal.index]
    fit_keys = _observation_keys(x_fit, y_fit)
    cal_keys = _observation_keys(x_cal, y_cal)
    fit_key_set = set(fit_keys.tolist())
    cal_key_set = set(cal_keys.tolist())
    duplicate_key_overlap = fit_key_set.intersection(cal_key_set)
    global_rows = []
    threshold_rows = []
    group_rows = []
    calibration_rows = []
    for model_name, estimator in make_models().items():
        pipeline = Pipeline(
            steps=[("preprocess", make_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(x_fit, y_fit)
        probabilities = _predict_probabilities(pipeline, x_cal)
        for threshold in thresholds:
            metrics = _classification_metrics(y_cal, probabilities, threshold)
            threshold_rows.append({"model": model_name, **metrics})
        global_rows.append(
            {
                "model": model_name,
                **_classification_metrics(y_cal, probabilities, threshold=0.5),
            }
        )
        for threshold in thresholds:
            group_rows.extend(
                group_metric_rows(
                    y_cal, probabilities, audit_cal, model_name, threshold=threshold
                )
            )
        bins = pd.qcut(probabilities, q=10, duplicates="drop")
        calibration = (
            pd.DataFrame(
                {
                    "model": model_name,
                    "bin": bins.astype(str),
                    "probability": probabilities,
                    "observed": y_cal.to_numpy(),
                }
            )
            .groupby(["model", "bin"], observed=False)
            .agg(
                count=("observed", "size"),
                mean_predicted_probability=("probability", "mean"),
                observed_rate=("observed", "mean"),
            )
            .reset_index()
        )
        calibration["n_bins"] = int(calibration["bin"].nunique())
        calibration_rows.append(calibration)
    official_train = frame.loc[frame["split"] == "train_official"]
    contract = {
        "official_train_rows": int(len(x)),
        "fit_rows": int(len(x_fit)),
        "calibration_rows": int(len(x_cal)),
        "validation_rows": int(len(x_cal)),
        "official_test_rows_reserved": int((frame["split"] == "test_official").sum()),
        "audit_columns": AUDIT_COLUMNS,
        "predictors": PREDICTOR_COLUMNS,
        "thresholds_evaluated": list(thresholds),
        "thresholds_not_selected": True,
        "calibration_is_diagnostic_only": True,
        "group_metrics_thresholds": list(thresholds),
        "low_support_denominator_threshold": LOW_SUPPORT_THRESHOLD,
        "duplicate_policy": "source rows retained; exact predictor-signature overlap reported as a limitation",
        "duplicate_policy_status": "adopted",
        "duplicate_identity": "exact_model_predictor_signature_with_missing_marker; target_excluded",
        "source_rows_removed": 0,
        "duplicate_keys_crossing_fit_validation": int(len(duplicate_key_overlap)),
        "fit_rows_with_key_also_in_validation": int(fit_keys.isin(duplicate_key_overlap).sum()),
        "validation_rows_with_key_also_in_fit": int(cal_keys.isin(duplicate_key_overlap).sum()),
        "protected_attributes_excluded_but_proxies_retained": True,
        "official_train_duplicate_rows_without_split": int(
            official_train.drop(columns=["split"]).duplicated().sum()
        ),
        "calibration_positive_rate": float(y_cal.mean()),
        "random_state": RANDOM_STATE,
        "test_used": False,
    }
    return {
        "global_metrics": pd.DataFrame(global_rows),
        "threshold_metrics": pd.DataFrame(threshold_rows),
        "group_metrics": pd.DataFrame(group_rows),
        "calibration_bins": pd.concat(calibration_rows, ignore_index=True),
        "contract": contract,
    }


def write_validation_outputs(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Executa a auditoria de validação e salva tabelas e contrato."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "global_metrics": output_dir / "validation_global_metrics.csv",
        "threshold_metrics": output_dir / "validation_threshold_metrics.csv",
        "group_metrics": output_dir / "validation_group_metrics.csv",
        "calibration_bins": output_dir / "validation_calibration_bins.csv",
        "contract": output_dir / "validation_split_contract.json",
    }
    results = run_validation_audit(frame)
    for key in ["global_metrics", "threshold_metrics", "group_metrics", "calibration_bins"]:
        results[key].to_csv(outputs[key], index=False, encoding="utf-8")
    outputs["contract"].write_text(
        json.dumps(results["contract"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return outputs


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = load_adult(root / "data" / "raw" / "adult_uci")
    paths = write_validation_outputs(data, root / "data" / "processed" / "validation")
    for name, path in paths.items():
        print(f"{name}: {path}")
