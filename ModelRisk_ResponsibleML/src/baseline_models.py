"""Baseline reproduzível para comparação inicial de modelos no UCI Adult."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from adult_data import AUDIT_COLUMNS, COLUMNS, TARGET, load_adult


RANDOM_STATE = 42
PREDICTOR_COLUMNS = [column for column in COLUMNS[:-1] if column not in AUDIT_COLUMNS]
NUMERIC_COLUMNS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLUMNS = [column for column in PREDICTOR_COLUMNS if column not in NUMERIC_COLUMNS]


def make_preprocessor() -> ColumnTransformer:
    """Cria pré-processamento dentro do pipeline para evitar vazamento."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, object]:
    """Retorna modelos candidatos com configurações transparentes."""
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }


def _evaluate(y_true: pd.Series, probabilities, threshold: float = 0.5) -> dict[str, float]:
    """Calcula métricas na validação sem escolher o limiar com o teste."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def _predict_probabilities(pipeline: Pipeline, x: pd.DataFrame, batch_size: int = 100) -> np.ndarray:
    """Calcula probabilidades em lotes para reduzir instabilidade nativa no Windows."""
    batches = [
        pipeline.predict_proba(x.iloc[start : start + batch_size])[:, 1]
        for start in range(0, len(x), batch_size)
    ]
    return np.concatenate(batches)


def run_baseline(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ajusta candidatos no treino e avalia somente na calibração/validação."""
    official_train = frame.loc[frame["split"] == "train_official"].copy()
    y = (official_train[TARGET] == ">50K").astype(int)
    x = official_train[PREDICTOR_COLUMNS].copy()
    # O pandas StringDtype usa pd.NA, que não é aceito pelo SimpleImputer
    # em algumas operações internas do scikit-learn. Normalizamos somente a
    # cópia usada no modelo para manter a representação da EDA intacta.
    x[CATEGORICAL_COLUMNS] = x[CATEGORICAL_COLUMNS].astype(object)
    x[CATEGORICAL_COLUMNS] = x[CATEGORICAL_COLUMNS].mask(
        x[CATEGORICAL_COLUMNS].isna(), np.nan
    )
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    rows = []
    for name, estimator in make_models().items():
        pipeline = Pipeline(
            steps=[("preprocess", make_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(x_fit, y_fit)
        probabilities = _predict_probabilities(pipeline, x_cal)
        metrics = _evaluate(y_cal, probabilities)
        rows.append({"model": name, **metrics})
    split_contract = {
        "official_train_rows": int(len(official_train)),
        "fit_rows": int(len(x_fit)),
        "calibration_rows": int(len(x_cal)),
        "validation_rows": int(len(x_cal)),
        "official_test_rows_reserved": int((frame["split"] == "test_official").sum()),
        "predictors": PREDICTOR_COLUMNS,
        "audit_columns_excluded_from_predictors": AUDIT_COLUMNS,
        "random_state": RANDOM_STATE,
        "selection_threshold": 0.5,
        "test_used": False,
        "duplicate_policy": "source rows retained; exact predictor-signature overlap reported; baseline holdout is diagnostic",
        "duplicate_policy_status": "adopted",
        "duplicate_identity": "exact_model_predictor_signature_with_missing_marker; target_excluded",
        "source_rows_removed": 0,
        "holdout_role": "validation holdout; no probability calibrator fitted",
        "protected_attributes_excluded_but_proxies_retained": True,
        "preprocessing_contract": {
            "numeric_missing": "median fitted on fit partition",
            "categorical_missing": "most_frequent fitted on fit partition",
            "categorical_encoding": "one-hot with handle_unknown=ignore",
            "rare_categories": "not collapsed in this baseline; sensitivity analysis pending",
            "education_and_education_num": "both retained; redundancy sensitivity analysis pending",
            "fnlwgt": "retained as predictor, not sample weight; sensitivity analysis pending",
        },
    }
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False), split_contract


def write_baseline_outputs(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Executa o baseline e salva comparação e contrato da divisão."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison, contract = run_baseline(frame)
    comparison_path = output_dir / "baseline_model_comparison.csv"
    contract_path = output_dir / "baseline_split_contract.json"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"comparison": comparison_path, "contract": contract_path}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = load_adult(root / "data" / "raw" / "adult_uci")
    paths = write_baseline_outputs(data, root / "data" / "processed" / "baseline")
    for name, path in paths.items():
        print(f"{name}: {path}")
