"""Compare the churn champion with a logistic regression baseline on the hold-out set.

Reproduces the "Overall metrics" table in the README: both models are trained on the
same stratified 80/20 split (seed 42) used by train_model.py and scored on the same
test customers. Run from this folder with the dataset at dataset/telco.csv:

    python evaluate_baseline.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_project.data import read_telecom_data, split_telecom_dataset
from train_model import DATA_PATH, SPLIT_STATE, THRESHOLD, build_pipeline

BASELINE_THRESHOLD = 0.50
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


def build_baseline(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def holdout_metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    auc = roc_auc_score(y, scores)
    preds = (scores >= threshold).astype(int)
    top = y[np.argsort(-scores)[: int(np.ceil(0.1 * len(y)))]]
    return {
        "recall_macro": recall_score(y, preds, average="macro"),
        "precision_macro": precision_score(y, preds, average="macro"),
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_2samp(scores[y == 1], scores[y == 0]).statistic,
        "pr_auc": average_precision_score(y, scores),
        "brier": brier_score_loss(y, scores),
        "mean_score": float(scores.mean()),
        "base_rate": float(y.mean()),
        "top10_precision": float(top.mean()),
        "top10_lift": float(top.mean() / y.mean()),
        "top10_capture": float(top.sum() / y.sum()),
    }


def bootstrap_auc_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main() -> None:
    splits = split_telecom_dataset(read_telecom_data(str(DATA_PATH)), random_state=SPLIT_STATE)
    X_train, X_test = splits["train_features"], splits["test_features"]
    y_train = np.asarray(splits["train_labels"]).astype(int)
    y_test = np.asarray(splits["test_labels"]).astype(int)

    champion = build_pipeline().fit(X_train, y_train)
    numeric = [c for c in X_train.columns if X_train[c].dtype.kind in "if"]
    categorical = [c for c in X_train.columns if c not in numeric]
    baseline = build_baseline(numeric, categorical).fit(X_train, y_train)

    s_champion = champion.predict_proba(X_test)[:, 1]
    s_baseline = baseline.predict_proba(X_test)[:, 1]

    result = {
        "n_test": int(len(y_test)),
        "n_churners": int(y_test.sum()),
        "champion_brf": holdout_metrics(s_champion, y_test, THRESHOLD),
        "baseline_lr": holdout_metrics(s_baseline, y_test, BASELINE_THRESHOLD),
        "roc_auc_diff_ci95": bootstrap_auc_diff(y_test, s_champion, s_baseline),
    }
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
