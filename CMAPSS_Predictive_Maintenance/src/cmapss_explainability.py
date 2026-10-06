from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold


def permutation_importance_by_engine(
    df: pd.DataFrame,
    features: list[str],
    model,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula importância por permutação em validações separadas por motor."""
    splitter = GroupKFold(n_splits=n_splits)
    records = []
    X = df[features]
    y = df["rul"]
    groups = df["unit"]

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        fitted = clone(model)
        fitted.fit(X.iloc[train_index], y.iloc[train_index])
        result = permutation_importance(
            fitted,
            X.iloc[validation_index],
            y.iloc[validation_index],
            scoring="neg_mean_absolute_error",
            n_repeats=n_repeats,
            random_state=random_state + fold,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            features,
            result.importances_mean,
            result.importances_std,
        ):
            records.append({
                "fold": fold,
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            })

    raw = pd.DataFrame(records)
    summary = (
        raw.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std_between_folds=("importance_mean", "std"),
            repeats_std_mean=("importance_std", "mean"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary["importance_std_between_folds"] = summary[
        "importance_std_between_folds"
    ].fillna(0.0)
    return summary
