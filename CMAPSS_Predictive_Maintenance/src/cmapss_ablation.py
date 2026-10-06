from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold


def evaluate_grouped_model(
    df: pd.DataFrame,
    features: list[str],
    model,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia um pipeline com validação cruzada por motor."""
    predictions = np.full(len(df), np.nan, dtype=float)
    folds = np.full(len(df), -1, dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    X = df[features]
    y = df["rul"]
    groups = df["unit"]

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        fitted = clone(model)
        fitted.fit(X.iloc[train_index], y.iloc[train_index])
        predictions[validation_index] = fitted.predict(X.iloc[validation_index])
        folds[validation_index] = fold

    result = df[["unit", "cycle", "rul"]].copy()
    result["rul_predito"] = predictions
    result["erro"] = result["rul_predito"] - result["rul"]
    result["erro_absoluto"] = result["erro"].abs()
    result["fold"] = folds

    by_unit = (
        result.groupby("unit", as_index=False)
        .agg(
            mae_motor=("erro_absoluto", "mean"),
            rmse_motor=("erro", lambda values: float(np.sqrt(np.mean(values**2)))),
            erro_medio_motor=("erro", "mean"),
        )
    )
    summary = pd.DataFrame([{
        "mae_medio_motor": float(by_unit["mae_motor"].mean()),
        "rmse_medio_motor": float(by_unit["rmse_motor"].mean()),
        "erro_medio_motor": float(by_unit["erro_medio_motor"].mean()),
        "percentual_superestimacao": float((result["erro"] > 0).mean() * 100),
        "percentual_subestimacao": float((result["erro"] < 0).mean() * 100),
        "motores": int(by_unit["unit"].nunique()),
    }])
    return summary, result
