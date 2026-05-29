"""
scoring.py
----------
Metric helpers, business score and threshold utilities.
"""
from __future__ import annotations

from json import dumps

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Scoring dictionaries
# ---------------------------------------------------------------------------

def get_classification_scoring() -> dict:
    """
    Return the standard scoring dictionary for cross-validation.

    Returns
    -------
    dict
        Mapping of scorer name → sklearn scorer.
    """
    return {
        "roc_auc": make_scorer(roc_auc_score, average="macro"),
        "precision_macro": make_scorer(precision_score, average="macro"),
        "recall_macro": make_scorer(recall_score, average="macro"),
        "f1_macro": make_scorer(f1_score, average="macro"),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }


def get_objective_scoring() -> dict:
    """
    Return the lightweight scoring dictionary used inside the Optuna objective.

    Only recall_macro is computed to keep trial evaluation fast.

    Returns
    -------
    dict
        Mapping of scorer name → sklearn scorer.
    """
    return {
        "recall_macro": make_scorer(
            recall_score,
            average="macro",
            zero_division=0,
        )
    }


# ---------------------------------------------------------------------------
# Business score
# ---------------------------------------------------------------------------

def build_business_weight_config(
    w_recall: float,
    w_precision: float,
    w_time: float,
    time_reference: float | None,
) -> str:
    """
    Serialise business-score weight configuration as a JSON string.

    Parameters
    ----------
    w_recall : float
        Weight assigned to recall.
    w_precision : float
        Weight assigned to precision.
    w_time : float
        Weight assigned to computational cost.
    time_reference : float | None
        Reference time used for time-penalty scaling.

    Returns
    -------
    str
        JSON string with weight configuration.
    """
    return dumps(
        {
            "w_recall": w_recall,
            "w_precision": w_precision,
            "w_time": w_time,
            "time_reference": time_reference,
        }
    )


def compute_business_score(
    recall: float,
    precision: float,
    fit_time: float,
    score_time: float,
    w_recall: float = 0.7,
    w_precision: float = 0.3,
    w_time: float = 0.0,
    normalize_weights: bool = True,
    time_reference: float | None = None,
    epsilon: float = 1e-9,
) -> float:
    """
    Compute a customisable business score combining predictive performance
    and computational cost.

    Parameters
    ----------
    recall : float
        Recall macro score.
    precision : float
        Precision macro score.
    fit_time : float
        Average model fitting time in seconds.
    score_time : float
        Average model scoring time in seconds.
    w_recall : float, default=0.7
        Weight assigned to recall.
    w_precision : float, default=0.3
        Weight assigned to precision.
    w_time : float, default=0.0
        Weight assigned to computational cost.
        Use ``0`` when machine cost is not relevant.
    normalize_weights : bool, default=True
        Whether to normalise weights so they sum to 1.
    time_reference : float | None, default=None
        Reference time used to scale the penalty.
        If ``None``, raw total time is used.
    epsilon : float, default=1e-9
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Business-oriented composite score.
    """
    total_time = fit_time + score_time

    if normalize_weights:
        weight_sum = w_recall + w_precision + w_time
        if weight_sum <= 0:
            raise ValueError("At least one weight must be greater than zero.")
        w_recall /= weight_sum
        w_precision /= weight_sum
        w_time /= weight_sum

    if w_time == 0:
        time_penalty = 0.0
    elif time_reference is not None:
        if time_reference <= 0:
            raise ValueError("time_reference must be greater than zero.")
        time_penalty = total_time / (time_reference + epsilon)
    else:
        time_penalty = total_time

    return (w_recall * recall) + (w_precision * precision) - (w_time * time_penalty)


# ---------------------------------------------------------------------------
# Threshold utilities
# ---------------------------------------------------------------------------

def get_positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return positive-class probability scores from a fitted classifier.

    Uses ``predict_proba`` when available; otherwise uses
    ``decision_function`` and rescales to ``[0, 1]``.

    Parameters
    ----------
    model : fitted estimator
        A fitted sklearn-compatible classifier.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        1-D array of positive-class scores.

    Raises
    ------
    AttributeError
        If the model implements neither ``predict_proba`` nor
        ``decision_function``.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        s_min, s_max = np.min(scores), np.max(scores)
        if s_max == s_min:
            return np.full_like(scores, fill_value=0.5, dtype=float)
        return (scores - s_min) / (s_max - s_min)

    raise AttributeError(
        "The model must implement either predict_proba or decision_function."
    )


def predict_with_threshold(
    model,
    X: pd.DataFrame,
    threshold: float,
) -> np.ndarray:
    """
    Predict binary labels using a custom decision threshold.

    Parameters
    ----------
    model : fitted estimator
        A fitted sklearn-compatible classifier.
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        Decision threshold in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Binary prediction array.
    """
    return (get_positive_scores(model, X) >= threshold).astype(int)
