"""Retention model: logistic regression with and without conversion.

Only the published scenario is implemented: a class-balanced logistic
regression, without resampling, trained on a stratified 80/20 split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from marketing_campaign.config import RANDOM_STATE

CATEGORICAL_CANDIDATES = [
    "marketing_channel",
    "variant",
    "language_displayed",
    "language_preferred",
    "age_group",
]
NUMERIC_CANDIDATES = ["is_correct_lang_flag", "month"]
_FLAG_MAPPING = {
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
    "yes": 1,
    "no": 0,
    "sim": 1,
    "nao": 0,
    "y": 1,
    "n": 0,
}


@dataclass(frozen=True)
class RetentionModel:
    """A fitted retention pipeline and its hold-out evaluation.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Preprocessing plus logistic regression, fitted on the train split.
    metrics : dict
        Output of :func:`calculate_retention_metrics` on the test split.
    categorical_features : list of str
        Categorical columns used.
    numeric_features : list of str
        Numeric columns used.
    y_test : pd.Series
        True retention of the test users.
    y_pred : np.ndarray
        Predicted class of the test users.
    y_proba : np.ndarray
        Predicted probability of being retained.
    X_test : pd.DataFrame
        Features of the test users.
    """

    pipeline: Pipeline
    metrics: dict[str, Any]
    categorical_features: list[str]
    numeric_features: list[str]
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray
    X_test: pd.DataFrame


def to_binary_flag(series: pd.Series, default: int = 0) -> pd.Series:
    """Convert boolean, textual or numeric values into 0/1 flags.

    Parameters
    ----------
    series : pd.Series
        Values such as ``True``, ``"Yes"``, ``"no"`` or ``1``.
    default : int, default=0
        Value used when the input cannot be interpreted.

    Returns
    -------
    pd.Series
        Integer flags clipped to 0 and 1.
    """
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(_FLAG_MAPPING)
    numeric = pd.to_numeric(series, errors="coerce")
    result = mapped.fillna(numeric).fillna(default).astype(int)
    return result.clip(lower=0, upper=1)


def prepare_retention_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create the modelling flags that may be missing from a frame.

    Parameters
    ----------
    df : pd.DataFrame
        User-level base.

    Returns
    -------
    pd.DataFrame
        A copy with ``converted_flag``, ``retained_flag``,
        ``is_correct_lang_flag`` and ``month`` filled in when absent.
    """
    frame = df.copy().replace({pd.NA: np.nan})
    if "converted_flag" not in frame.columns and "converted" in frame:
        frame["converted_flag"] = to_binary_flag(frame["converted"])
    if "retained_flag" not in frame.columns and "is_retained" in frame:
        frame["retained_flag"] = to_binary_flag(frame["is_retained"])
    if "is_correct_lang_flag" not in frame.columns:
        if "is_correct_lang" in frame.columns:
            frame["is_correct_lang_flag"] = to_binary_flag(
                frame["is_correct_lang"]
            )
        elif {"language_displayed", "language_preferred"}.issubset(
            frame.columns
        ):
            frame["is_correct_lang_flag"] = (
                frame["language_displayed"].astype(str)
                == frame["language_preferred"].astype(str)
            ).astype(int)
    if "month" not in frame.columns and "date_served" in frame.columns:
        frame["month"] = pd.to_datetime(
            frame["date_served"], errors="coerce"
        ).dt.month
    return frame


def get_feature_sets(
    df: pd.DataFrame, include_converted: bool
) -> tuple[list[str], list[str]]:
    """List the categorical and numeric features available in a frame.

    Parameters
    ----------
    df : pd.DataFrame
        Frame prepared with :func:`prepare_retention_model_frame`.
    include_converted : bool
        Whether ``converted_flag`` is offered to the model.

    Returns
    -------
    tuple of (list of str, list of str)
        Categorical and numeric feature names present in ``df``.
    """
    numeric = list(NUMERIC_CANDIDATES)
    if include_converted:
        numeric.append("converted_flag")
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    return categorical, [c for c in numeric if c in df.columns]


def make_preprocessor(
    categorical_features: list[str], numeric_features: list[str]
) -> ColumnTransformer:
    """Build the preprocessing shared by the retention estimators.

    Categorical columns are imputed with ``"Unknown"`` and one-hot
    encoded; numeric columns are imputed with the median and scaled.

    Parameters
    ----------
    categorical_features : list of str
        Categorical columns.
    numeric_features : list of str
        Numeric columns.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted preprocessing transformer.
    """
    transformers = []
    if categorical_features:
        categorical = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="Unknown"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical, categorical_features))
    if numeric_features:
        numeric = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric, numeric_features))
    return ColumnTransformer(transformers=transformers)


def calculate_retention_metrics(
    y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, Any]:
    """Compute overall metrics and those of the "not retained" class.

    Parameters
    ----------
    y_test : pd.Series
        True retention (1 retained, 0 not retained).
    y_pred : np.ndarray
        Predicted class.
    y_proba : np.ndarray
        Predicted probability of the retained class.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1 and ROC AUC for the retained
        class, precision, recall and F1 for the not-retained class, and
        the counts of false positives and false negatives. Rates are
        rounded to four decimals.
    """
    truth = np.asarray(y_test)
    pred = np.asarray(y_pred)
    try:
        roc_auc = roc_auc_score(truth, np.asarray(y_proba))
    except ValueError:
        roc_auc = np.nan

    def rounded(metric, **kwargs) -> float:
        """Return ``metric`` on the test split, rounded to four places."""
        return round(metric(truth, pred, zero_division=0, **kwargs), 4)

    return {
        "accuracy": round(accuracy_score(truth, pred), 4),
        "precision": rounded(precision_score),
        "recall": rounded(recall_score),
        "f1": rounded(f1_score),
        "roc_auc": round(float(roc_auc), 4) if pd.notna(roc_auc) else np.nan,
        "precision_not_retained": rounded(precision_score, pos_label=0),
        "recall_not_retained": rounded(recall_score, pos_label=0),
        "f1_not_retained": rounded(f1_score, pos_label=0),
        "false_positives": int(((truth == 0) & (pred == 1)).sum()),
        "false_negatives": int(((truth == 1) & (pred == 0)).sum()),
    }


def build_retention_model(
    df: pd.DataFrame,
    include_converted: bool = True,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> RetentionModel:
    """Train and evaluate the class-balanced logistic regression.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical user-level base with ``retained_flag``.
    include_converted : bool, default=True
        Whether the conversion flag is used as a feature.
    test_size : float, default=0.2
        Share of users held out for evaluation.
    random_state : int, default=42
        Seed of the stratified split.

    Returns
    -------
    RetentionModel
        The fitted pipeline with its test-split predictions and metrics.

    Raises
    ------
    ValueError
        If there is no retention target, no feature, or only one class.
    """
    frame = prepare_retention_model_frame(df)
    if "retained_flag" not in frame.columns:
        raise ValueError(
            "The retained_flag or is_retained column is required."
        )

    categorical, numeric = get_feature_sets(frame, include_converted)
    if not categorical + numeric:
        raise ValueError("No available features for retention modeling.")

    target = to_binary_flag(frame["retained_flag"])
    if target.nunique() < 2:
        raise ValueError("The target must contain both classes 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        frame[categorical + numeric].copy(),
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(categorical, numeric)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", solver="lbfgs"
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return RetentionModel(
        pipeline=pipeline,
        metrics=calculate_retention_metrics(y_test, y_pred, y_proba),
        categorical_features=categorical,
        numeric_features=numeric,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        X_test=X_test,
    )


def run_retention_experiments(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Compare the model with and without the conversion flag.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical user-level base.
    random_state : int, default=42
        Seed of the split.
    test_size : float, default=0.2
        Share of users held out.

    Returns
    -------
    pd.DataFrame
        One row per scenario with its metrics, ranked by ROC AUC, then
        recall and F1 of the not-retained class (best first).
    """
    rows = []
    for include_converted in (True, False):
        model = build_retention_model(
            df,
            include_converted=include_converted,
            test_size=test_size,
            random_state=random_state,
        )
        rows.append(
            {
                "experiment": make_experiment_label(
                    "LogisticRegression", include_converted, None
                ),
                "estimator_name": "LogisticRegression",
                "include_converted": include_converted,
                "resampling_strategy": "None",
                **model.metrics,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["roc_auc", "recall_not_retained", "f1_not_retained"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def make_experiment_label(
    estimator_name: str,
    include_converted: bool,
    resampling_strategy: str | None,
) -> str:
    """Create a readable label for a retention scenario.

    Parameters
    ----------
    estimator_name : str
        Name of the estimator.
    include_converted : bool
        Whether the conversion flag is a feature.
    resampling_strategy : str or None
        ``"smote"`` or ``None``.

    Returns
    -------
    str
        Text such as ``"LogisticRegression | with converted | without
        SMOTE"``.
    """
    converted = "with converted" if include_converted else "without converted"
    resampling = "SMOTE" if resampling_strategy == "smote" else "without SMOTE"
    return f"{estimator_name} | {converted} | {resampling}"


METRIC_LABELS = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-Score", "f1"),
    ("ROC AUC", "roc_auc"),
    ("Precision - Not Retained", "precision_not_retained"),
    ("Recall - Not Retained", "recall_not_retained"),
    ("F1 - Not Retained", "f1_not_retained"),
    ("False Positives", "false_positives"),
    ("False Negatives", "false_negatives"),
]


def make_metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Format the retention metrics as a two-column table.

    Parameters
    ----------
    metrics : dict
        Output of :func:`calculate_retention_metrics`.

    Returns
    -------
    pd.DataFrame
        Columns ``Metric`` and ``Value`` with the ten metrics, in the
        order shown in the report.
    """
    return pd.DataFrame(
        {
            "Metric": [label for label, _ in METRIC_LABELS],
            "Value": [metrics[key] for _, key in METRIC_LABELS],
        }
    )
