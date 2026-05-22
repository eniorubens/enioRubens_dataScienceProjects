"""Train the winning pipeline and save to models/churn_pipeline.pkl.gz."""
from __future__ import annotations

import gzip
import pickle
from datetime import datetime, timezone
from pathlib import Path

import category_encoders as ce
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, Normalizer

from churn_project.data import compute_class_ratio, read_telecom_data, split_telecom_dataset

MODEL_PATH = Path("models/churn_pipeline.pkl.gz")
DATA_PATH = Path("dataset/telco.csv")
RANDOM_STATE = 738
THRESHOLD = 0.52


def build_pipeline() -> Pipeline:
    brfc = BalancedRandomForestClassifier(
        n_estimators=306,
        max_depth=9,
        max_features="log2",
        min_samples_leaf=2,
        min_samples_split=6,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("encoder",    ce.SumEncoder()),
        ("normalizer", Normalizer()),
        ("scaler",     MaxAbsScaler()),
        ("selector",   SelectFromModel(brfc, threshold="mean")),
        ("clf",        BalancedRandomForestClassifier(
            n_estimators=306,
            max_depth=9,
            max_features="log2",
            min_samples_leaf=2,
            min_samples_split=6,
            random_state=RANDOM_STATE,
        )),
    ])


def compute_column_defaults(df) -> dict:
    defaults = {}
    for col in df.columns:
        if df[col].dtype == object:
            defaults[col] = df[col].mode()[0]
        else:
            defaults[col] = float(df[col].median())
    return defaults


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = read_telecom_data(str(DATA_PATH))
    splits = split_telecom_dataset(df, random_state=RANDOM_STATE)

    X_train = splits["train_features"]
    y_train = splits["train_labels"]
    X_test  = splits["test_features"]
    y_test  = splits["test_labels"]

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Class ratio (neg/pos): {compute_class_ratio(y_train):.2f}")

    print("\nBuilding and training pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    preds_test = (proba_test >= THRESHOLD).astype(int)

    recall_macro   = recall_score(y_test, preds_test, average="macro")
    roc_auc        = roc_auc_score(y_test, proba_test)
    balanced_acc   = balanced_accuracy_score(y_test, preds_test)

    print(f"\n--- Test Metrics (threshold={THRESHOLD}) ---")
    print(f"Recall Macro:       {recall_macro:.4f}")
    print(f"ROC AUC:            {roc_auc:.4f}")
    print(f"Balanced Accuracy:  {balanced_acc:.4f}")

    artifact = {
        "pipeline":        pipeline,
        "feature_columns": list(X_train.columns),
        "column_defaults": compute_column_defaults(X_train),
        "threshold":       THRESHOLD,
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "feature_count":   int(X_train.shape[1]),
        "churn_rate":      float(y_train.mean()),
        "estimator_name":  "BalancedRandomForestClassifier",
        "test_metrics": {
            "recall_macro":      round(recall_macro, 4),
            "roc_auc":           round(roc_auc, 4),
            "balanced_accuracy": round(balanced_acc, 4),
        },
    }

    with gzip.open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nArtifact saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
