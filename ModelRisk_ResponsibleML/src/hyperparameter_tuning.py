"""Ajuste group-aware de hiperparâmetros do HistGradientBoosting.

O módulo usa somente previsões OOF dentro do bloco ``fit`` do treino oficial.
Não escolhe threshold, não calibra probabilidades e não consulta o teste oficial.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from adult_data import load_adult
from baseline_models import RANDOM_STATE, make_preprocessor
from threshold_selection import make_group_aware_partitions, model_group_signature


HGB_TUNING_CANDIDATES = (
    {
        "candidate_id": "baseline_fixed",
        "params": {
            "max_iter": 200,
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        },
    },
    {
        "candidate_id": "conservative_leaves",
        "params": {
            "max_iter": 250,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 30,
        },
    },
    {
        "candidate_id": "deeper_regularized",
        "params": {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
        },
    },
    {
        "candidate_id": "faster_medium",
        "params": {
            "max_iter": 150,
            "learning_rate": 0.10,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        },
    },
    {
        "candidate_id": "wide_faster",
        "params": {
            "max_iter": 200,
            "learning_rate": 0.12,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
        },
    },
    {
        "candidate_id": "shallow_regularized",
        "params": {
            "max_iter": 300,
            "learning_rate": 0.08,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 50,
        },
    },
)


def _fit_pipeline(params: dict[str, int | float]) -> Pipeline:
    estimator = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        **params,
    )
    return Pipeline(
        steps=[("preprocess", make_preprocessor()), ("model", estimator)]
    )


def _oof_probabilities(
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    groups_fit: pd.Series,
    params: dict[str, int | float],
) -> pd.Series:
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE + 2,
    )
    probabilities = pd.Series(index=x_fit.index, dtype=float)
    for train_index, validation_index in splitter.split(x_fit, y_fit, groups_fit):
        pipeline = _fit_pipeline(params)
        pipeline.fit(x_fit.iloc[train_index], y_fit.iloc[train_index])
        probabilities.iloc[validation_index] = pipeline.predict_proba(
            x_fit.iloc[validation_index]
        )[:, 1]
    return probabilities


def run_hgb_tuning(frame: pd.DataFrame) -> dict[str, object]:
    """Compara configurações HGB por métricas OOF sem usar threshold."""
    partitions = make_group_aware_partitions(frame)
    x_fit = partitions["x_fit"]
    y_fit = partitions["y_fit"]
    groups_fit = model_group_signature(x_fit)
    rows = []
    for candidate in HGB_TUNING_CANDIDATES:
        probabilities = _oof_probabilities(
            x_fit,
            y_fit,
            groups_fit,
            candidate["params"],
        )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                **candidate["params"],
                "roc_auc": float(roc_auc_score(y_fit, probabilities)),
                "pr_auc": float(average_precision_score(y_fit, probabilities)),
                "brier": float(brier_score_loss(y_fit, probabilities)),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(
        ["pr_auc", "roc_auc", "brier"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    winner = metrics.iloc[0]
    selected_params = {
        key: int(winner[key]) if key in {"max_iter", "max_leaf_nodes", "min_samples_leaf"}
        else float(winner[key])
        for key in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf")
    }
    contract = {
        "protocol_version": "hgb-tuning-mvp-1",
        "tuned_model": "hist_gradient_boosting",
        "search_method": "explicit_candidate_grid",
        "candidate_count": len(HGB_TUNING_CANDIDATES),
        "candidate_ids": [candidate["candidate_id"] for candidate in HGB_TUNING_CANDIDATES],
        "objective_order": ["pr_auc", "roc_auc", "brier"],
        "fit_rows": int(len(y_fit)),
        "oof_folds": 3,
        "group_aware": True,
        "random_state": RANDOM_STATE + 2,
        "threshold_used": False,
        "calibration_used": False,
        "official_test_rows_reserved": int((frame["split"] == "test_official").sum()),
        "test_used": False,
        "selected_candidate_id": str(winner["candidate_id"]),
        "selected_params": selected_params,
    }
    return {"metrics": metrics, "contract": contract}


def write_hgb_tuning_outputs(
    frame: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    """Salva as métricas OOF e o contrato do ajuste HGB."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_hgb_tuning(frame)
    outputs = {
        "metrics": output_dir / "hgb_tuning_metrics.csv",
        "contract": output_dir / "hgb_tuning_contract.json",
    }
    results["metrics"].to_csv(outputs["metrics"], index=False, encoding="utf-8")
    outputs["contract"].write_text(
        json.dumps(results["contract"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return outputs


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = load_adult(root / "data" / "raw" / "adult_uci")
    paths = write_hgb_tuning_outputs(
        data, root / "data" / "processed" / "tuning"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
