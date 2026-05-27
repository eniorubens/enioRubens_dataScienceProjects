"""
prediction.py
-------------
Supervised repurchase-probability pipeline: factory functions and evaluation.

The pipeline is:
    KMeansClusterAdder  →  ColumnTransformer (scale + one-hot)  →  classifier

Keeping K-Means inside the pipeline ensures cluster labels are derived
only from training data, eliminating look-ahead leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import N_CLUSTERS, RANDOM_STATE
from src.segmentation import KMeansClusterAdder


# ── Encoder / preprocessor factories ──────────────────────────────────────

def make_one_hot_encoder() -> OneHotEncoder:
    """Create a ``OneHotEncoder`` compatible with scikit-learn ≥ 1.0 and older."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)  # type: ignore[call-arg]


def make_preprocessor(
    numeric_features: list[str],
    cluster_features: list[str],
    log_features: list[str],
    selected_k: int = N_CLUSTERS,
) -> Pipeline:
    """Build the leakage-free preprocessing pipeline.

    Parameters
    ----------
    numeric_features : list[str]
        Columns to scale with ``StandardScaler``.
    cluster_features : list[str]
        RFM columns fed to ``KMeansClusterAdder``.
    log_features : list[str]
        Subset of *cluster_features* to log-transform before K-Means.
    selected_k : int
        Number of clusters.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps: ``cluster`` → ``columns``.
    """
    cluster_step = KMeansClusterAdder(
        cluster_features=cluster_features,
        log_features=log_features,
        n_clusters=selected_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    columns_step = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", make_one_hot_encoder(), ["CountryMode", "PredictiveCluster"]),
        ]
    )
    return Pipeline([("cluster", cluster_step), ("columns", columns_step)])


def make_model_specs(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Return the candidate estimator dictionary.

    Returns
    -------
    dict[str, estimator]
        Keys are model names; values are unfitted sklearn estimators.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def make_repurchase_pipeline(
    estimator: Any,
    numeric_features: list[str],
    cluster_features: list[str],
    log_features: list[str],
    selected_k: int = N_CLUSTERS,
) -> Pipeline:
    """Combine the preprocessing pipeline with a supervised *estimator*.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
    numeric_features, cluster_features, log_features, selected_k :
        Forwarded to :func:`make_preprocessor`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps: ``preprocess`` → ``model``.
    """
    return Pipeline([
        ("preprocess", make_preprocessor(
            numeric_features=numeric_features,
            cluster_features=cluster_features,
            log_features=log_features,
            selected_k=selected_k,
        )),
        ("model", estimator),
    ])


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_classifier(
    model_name: str,
    dataset_name: str,
    estimator: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    threshold: float = 0.50,
) -> dict[str, float | str]:
    """Compute classification and calibration metrics for one split.

    Parameters
    ----------
    model_name : str
    dataset_name : str
        E.g. ``"Validation"``, ``"Hold-out"``.
    estimator : fitted pipeline
    X_eval : pd.DataFrame
    y_eval : pd.Series
    threshold : float
        Decision threshold for binary predictions.

    Returns
    -------
    dict
        Keys: ``Model``, ``Sample``, ``ROC_AUC``, ``Recall``,
        ``Precision``, ``Accuracy``, ``BrierScore``.
    """
    y_prob = estimator.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_eval, y_prob) if y_eval.nunique() == 2 else np.nan

    return {
        "Model":       model_name,
        "Sample":      dataset_name,
        "ROC_AUC":     auc,
        "Recall":      recall_score(y_eval, y_pred, zero_division=0),
        "Precision":   precision_score(y_eval, y_pred, zero_division=0),
        "Accuracy":    accuracy_score(y_eval, y_pred),
        "BrierScore":  brier_score_loss(y_eval, y_prob),
    }


def calibrate(pipeline: Pipeline, X_cal: pd.DataFrame, y_cal: pd.Series) -> Pipeline:
    """Wrap the final estimator step with ``CalibratedClassifierCV``.

    Calibration is applied post-fit using the supplied calibration set,
    leaving the preprocessing steps unchanged.

    Parameters
    ----------
    pipeline : fitted Pipeline
        Must have a ``"model"`` step.
    X_cal, y_cal : calibration split (not seen during training).

    Returns
    -------
    Pipeline
        New pipeline with calibrated model step.
    """
    base_estimator = pipeline.named_steps["model"]
    calibrated = CalibratedClassifierCV(
        estimator=base_estimator,
        method="isotonic",
        cv="prefit",
    )
    # Transform X_cal through the preprocessing steps before fitting calibrator
    preprocess = pipeline.named_steps["preprocess"]
    X_transformed = preprocess.transform(X_cal)
    calibrated.fit(X_transformed, y_cal)

    # Rebuild pipeline with calibrated model
    new_pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", calibrated),
    ])
    return new_pipeline
