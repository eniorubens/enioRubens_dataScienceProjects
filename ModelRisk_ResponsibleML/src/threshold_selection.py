"""Protocolo group-aware para seleção de modelo e threshold.

Este módulo separa seleção de modelo, calibração e escolha de threshold no
treino oficial. O teste oficial permanece reservado para avaliação final.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from adult_data import TARGET, load_adult
from baseline_models import (
    CATEGORICAL_COLUMNS,
    PREDICTOR_COLUMNS,
    RANDOM_STATE,
    _predict_probabilities,
    make_models,
    make_preprocessor,
)
from validation_audit import group_metric_rows


THRESHOLDS = tuple(np.round(np.arange(0.10, 0.901, 0.01), 2))
RECALL_FLOOR = 0.79
PRECISION_FLOOR = 0.64


def model_frame(
    frame: pd.DataFrame, split_name: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepara uma partição para o modelo, mantendo split explícito."""
    subset = frame.loc[frame["split"] == split_name].copy()
    y = (subset[TARGET] == ">50K").astype(int)
    x = subset[PREDICTOR_COLUMNS].copy()
    x[CATEGORICAL_COLUMNS] = x[CATEGORICAL_COLUMNS].astype(object)
    x[CATEGORICAL_COLUMNS] = x[CATEGORICAL_COLUMNS].mask(
        x[CATEGORICAL_COLUMNS].isna(), np.nan
    )
    audit = subset[["sex", "race"]].copy()
    return x, y, audit


def development_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepara apenas o treino oficial para desenvolvimento."""
    return model_frame(frame, "train_official")


def model_group_signature(x: pd.DataFrame) -> pd.Series:
    """Agrupa linhas com a mesma assinatura dos preditores do modelo."""
    normalized = x.astype("string").fillna("(missing)")
    return pd.util.hash_pandas_object(normalized, index=False)


def official_split_overlap_diagnostic(frame: pd.DataFrame) -> dict[str, int]:
    """Conta assinaturas de preditores repetidas entre treino e teste.

    Este diagnóstico lê o teste apenas para documentar uma limitação de
    independência. Ele não participa de ajuste, seleção, calibração ou métricas.
    """
    train_x, train_y, _ = model_frame(frame, "train_official")
    test_x, test_y, _ = model_frame(frame, "test_official")
    train_table = pd.DataFrame(
        {"key": model_group_signature(train_x).to_numpy(), "target": train_y.to_numpy()}
    )
    test_table = pd.DataFrame(
        {"key": model_group_signature(test_x).to_numpy(), "target": test_y.to_numpy()}
    )
    train_targets = train_table.groupby("key")["target"].agg(lambda values: frozenset(values))
    test_targets = test_table.groupby("key")["target"].agg(lambda values: frozenset(values))
    common_keys = train_targets.index.intersection(test_targets.index)
    discordant = sum(
        train_targets.loc[key] != test_targets.loc[key] for key in common_keys
    )
    return {
        "cross_official_split_predictor_signature_keys": int(len(common_keys)),
        "cross_official_split_train_rows": int(train_table["key"].isin(common_keys).sum()),
        "cross_official_split_test_rows": int(test_table["key"].isin(common_keys).sum()),
        "cross_official_split_target_discordant_keys": int(discordant),
    }


def make_group_aware_partitions(frame: pd.DataFrame) -> dict:
    """Divide treino em fit, calibração e seleção sem cruzar grupos."""
    x, y, audit = development_frame(frame)
    groups = model_group_signature(x)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    fold_for_row = np.full(len(x), -1, dtype=int)
    for fold, (_, validation_index) in enumerate(splitter.split(x, y, groups)):
        fold_for_row[validation_index] = fold
    fit_mask = ~np.isin(fold_for_row, [0, 1])
    calibration_mask = fold_for_row == 0
    selection_mask = fold_for_row == 1
    if (fold_for_row < 0).any():
        raise RuntimeError("Some development rows were not assigned to a fold")
    return {
        "x_fit": x.loc[fit_mask],
        "y_fit": y.loc[fit_mask],
        "audit_fit": audit.loc[fit_mask],
        "x_calibration": x.loc[calibration_mask],
        "y_calibration": y.loc[calibration_mask],
        "audit_calibration": audit.loc[calibration_mask],
        "x_selection": x.loc[selection_mask],
        "y_selection": y.loc[selection_mask],
        "audit_selection": audit.loc[selection_mask],
        "group_signature": groups,
        "fold_for_row": fold_for_row,
    }


def classification_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Calcula métricas globais para um threshold."""
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


def build_model_pipeline(
    name: str, model_overrides: dict[str, object] | None = None
) -> Pipeline:
    """Cria um pipeline candidato com pré-processamento encapsulado."""
    models = make_models()
    estimator = models[name]
    if model_overrides and name == "hist_gradient_boosting":
        estimator.set_params(**model_overrides)
    return Pipeline(
        steps=[("preprocess", make_preprocessor()), ("model", estimator)]
    )


def _oof_model_selection(
    partitions: dict, tuned_model_params: dict[str, object] | None = None
) -> pd.DataFrame:
    """Compara modelos por previsões OOF somente dentro do bloco de ajuste."""
    x_fit = partitions["x_fit"]
    y_fit = partitions["y_fit"]
    groups_fit = model_group_signature(x_fit)
    splitter = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE + 1
    )
    rows = []
    for name in make_models():
        probabilities = np.full(len(x_fit), np.nan)
        for train_index, validation_index in splitter.split(x_fit, y_fit, groups_fit):
            pipeline = build_model_pipeline(name, tuned_model_params)
            pipeline.fit(x_fit.iloc[train_index], y_fit.iloc[train_index])
            probabilities[validation_index] = _predict_probabilities(
                pipeline, x_fit.iloc[validation_index]
            )
        metrics = classification_metrics(y_fit, probabilities, threshold=0.5)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows).sort_values(
        ["pr_auc", "roc_auc", "brier"], ascending=[False, False, True]
    )


class SigmoidProbabilityCalibrator:
    """Calibrador Platt simples ajustado fora do treino do modelo base."""

    def __init__(self) -> None:
        self._model = LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, target: pd.Series) -> "SigmoidProbabilityCalibrator":
        """Ajusta a transformação sigmoide no bloco de calibração."""
        self._model.fit(self._logit(probabilities), target)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Retorna probabilidades calibradas."""
        return self._model.predict_proba(self._logit(probabilities))[:, 1]


def select_threshold(
    metrics: pd.DataFrame,
    recall_floor: float = RECALL_FLOOR,
    precision_floor: float = PRECISION_FLOOR,
) -> tuple[float, str]:
    """Escolhe o menor threshold que atende os pisos do cenário de risco."""
    required = {"threshold", "precision", "recall_tpr", "f1"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing threshold-selection columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError("Cannot select a threshold from an empty table")
    numeric = metrics[list(required)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Threshold-selection metrics must be finite")
    if not (0 <= recall_floor <= 1 and 0 <= precision_floor <= 1):
        raise ValueError("Risk floors must be between zero and one")
    eligible = metrics.loc[
        (metrics["recall_tpr"] >= recall_floor)
        & (metrics["precision"] >= precision_floor)
    ].sort_values("threshold")
    if not eligible.empty:
        return float(eligible.iloc[0]["threshold"]), "risk_floors_satisfied"
    fallback = metrics.sort_values(
        ["f1", "recall_tpr", "precision"], ascending=[False, False, False]
    ).iloc[0]
    return float(fallback["threshold"]), "fallback_max_f1_no_threshold_met_floors"


def run_threshold_selection(
    frame: pd.DataFrame,
    tuned_model_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Executa seleção group-aware sem consultar o teste oficial."""
    partitions = make_group_aware_partitions(frame)
    model_metrics = _oof_model_selection(partitions, tuned_model_params)
    selected_model = str(model_metrics.iloc[0]["model"])
    pipeline = build_model_pipeline(selected_model, tuned_model_params)
    pipeline.fit(partitions["x_fit"], partitions["y_fit"])
    calibration_raw = _predict_probabilities(
        pipeline, partitions["x_calibration"]
    )
    selection_raw = _predict_probabilities(pipeline, partitions["x_selection"])
    calibrator = SigmoidProbabilityCalibrator().fit(
        calibration_raw, partitions["y_calibration"]
    )
    selection_probabilities = calibrator.predict(selection_raw)
    threshold_metrics = pd.DataFrame(
        [
            {
                "model": selected_model,
                **classification_metrics(
                    partitions["y_selection"], selection_probabilities, threshold
                ),
            }
            for threshold in THRESHOLDS
        ]
    )
    selected_threshold, selection_reason = select_threshold(threshold_metrics)
    overlap_diagnostic = official_split_overlap_diagnostic(frame)
    group_metrics = pd.DataFrame(
        group_metric_rows(
            partitions["y_selection"],
            selection_probabilities,
            partitions["audit_selection"],
            selected_model,
            selected_threshold,
        )
    )
    contract = {
        "protocol_version": "mvp-1",
        "official_train_rows": int(len(partitions["y_fit"]) + len(partitions["y_calibration"]) + len(partitions["y_selection"])),
        "fit_rows": int(len(partitions["y_fit"])),
        "calibration_rows": int(len(partitions["y_calibration"])),
        "selection_rows": int(len(partitions["y_selection"])),
        "official_test_rows_reserved": int((frame["split"] == "test_official").sum()),
        "group_aware_split": True,
        "group_signature": "model_predictor_columns_with_missing_marker",
        "duplicate_policy": (
            "exact_predictor_signature_groups_adopted_across_development_blocks; "
            "source_rows_retained; official_train_test_overlap_reported_as_diagnostic"
        ),
        "duplicate_policy_status": "adopted",
        "duplicate_identity": "exact_model_predictor_signature_with_missing_marker; target_excluded",
        "source_rows_removed": 0,
        "official_test_overlap_diagnostic_only": True,
        **overlap_diagnostic,
        "random_state": RANDOM_STATE,
        "threshold_grid": list(THRESHOLDS),
        "risk_scenario": "recall_priority_with_precision_floor",
        "recall_floor": RECALL_FLOOR,
        "precision_floor": PRECISION_FLOOR,
        "threshold_rule": (
            "lowest_threshold_meeting_recall_and_precision_floors; "
            "otherwise_max_f1"
        ),
        "model_selection_rule": "oof_pr_auc_then_roc_auc_then_brier",
        "tuning_used": tuned_model_params is not None,
        "tuned_model_params": tuned_model_params,
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "selection_reason": selection_reason,
        "risk_gate_passed": selection_reason == "risk_floors_satisfied",
        "decision_status": (
            "provisional_fallback_requires_risk_review"
            if selection_reason != "risk_floors_satisfied"
            else "provisional_risk_floors_satisfied"
        ),
        "selection_metrics_are_development_only": True,
        "calibrator": "sigmoid_platt_on_calibration_holdout",
        "protected_attributes_excluded_but_proxies_retained": True,
        "fnlwgt_role": "predictor_not_sample_weight",
        "test_used": False,
        "final_test_evaluation_pending": True,
        "final_model_refit_policy_pending": True,
        "optuna_used": False,
    }
    return {
        "model_metrics": model_metrics,
        "threshold_metrics": threshold_metrics,
        "group_metrics": group_metrics,
        "contract": contract,
    }


def write_threshold_selection_outputs(
    frame: pd.DataFrame,
    output_dir: str | Path,
    tuned_model_params: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Salva as evidências da seleção de desenvolvimento."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_threshold_selection(frame, tuned_model_params)
    outputs = {
        "model_metrics": output_dir / "selection_model_metrics.csv",
        "threshold_metrics": output_dir / "selection_threshold_metrics.csv",
        "group_metrics": output_dir / "selection_group_metrics.csv",
        "contract": output_dir / "threshold_selection_contract.json",
    }
    results["model_metrics"].to_csv(outputs["model_metrics"], index=False, encoding="utf-8")
    results["threshold_metrics"].to_csv(
        outputs["threshold_metrics"], index=False, encoding="utf-8"
    )
    results["group_metrics"].to_csv(outputs["group_metrics"], index=False, encoding="utf-8")
    outputs["contract"].write_text(
        json.dumps(results["contract"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return outputs


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = load_adult(root / "data" / "raw" / "adult_uci")
    tuning_contract_path = (
        root / "data" / "processed" / "tuning" / "hgb_tuning_contract.json"
    )
    tuned_model_params = None
    if tuning_contract_path.exists():
        tuned_model_params = json.loads(
            tuning_contract_path.read_text(encoding="utf-8")
        )["selected_params"]
    paths = write_threshold_selection_outputs(
        data,
        root / "data" / "processed" / "selection",
        tuned_model_params=tuned_model_params,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
