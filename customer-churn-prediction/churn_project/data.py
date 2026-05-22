from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path='../dataset/telco.csv'):
    df = pd.read_csv(path)
    return df


def read_telecom_data(filepath: str = "../dataset/telecom.csv") -> pd.DataFrame:
    """
    Read the telecom churn dataset.

    Parameters
    ----------
    filepath : str, default="../dataset/telecom.csv"
        Path to the dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(filepath)


def split_telecom_dataset(
    df: pd.DataFrame,
    target_col: str = "Churn",
    id_col: str = "customerID",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_cols: list[str] | None = None
) -> dict[str, Any]:
    """
    Split the telecom dataset into train and test partitions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, default="Churn"
        Target column name.
    id_col : str, default="customerID"
        Identifier column name.
    test_size : float, default=0.2
        Proportion of the dataset assigned to test set.
    random_state : int, default=42
        Random seed for reproducibility.
    drop_cols : list[str] | None, default=None
        Additional columns to drop before building X.

    Returns
    -------
    dict[str, Any]
        Dictionary containing split datasets and derived objects.
    """
    if drop_cols is None:
        drop_cols = []

    data = df.copy(deep=True)

    if 'TotalCharges' in data.columns and data['TotalCharges'].dtype == object:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data = data.dropna(subset=['TotalCharges'])

    target = data[target_col].map({"No": 0, "Yes": 1}).values

    feature_drop_cols = [target_col, id_col] + drop_cols
    features = data.drop(columns=feature_drop_cols, errors="ignore")

    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        stratify=data[target_col],
        random_state=random_state
    )

    train_features = train_data.drop(
        columns=[target_col, id_col] + drop_cols,
        errors="ignore"
    )
    test_features = test_data.drop(
        columns=[target_col, id_col] + drop_cols,
        errors="ignore"
    )

    train_labels = train_data[target_col].map({"No": 0, "Yes": 1}).values
    test_labels = test_data[target_col].map({"No": 0, "Yes": 1}).values

    return {
        "data": data,
        "X": features,
        "y": target,
        "train_data": train_data,
        "test_data": test_data,
        "train_features": train_features,
        "test_features": test_features,
        "train_labels": train_labels,
        "test_labels": test_labels,
    }


def compute_class_ratio(labels: np.ndarray) -> float:
    """
    Compute negative-to-positive class ratio.

    Parameters
    ----------
    labels : np.ndarray
        Binary target array encoded as 0/1.

    Returns
    -------
    float
        Ratio of negatives to positives.
    """
    labels = np.asarray(labels)

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0:
        raise ValueError("Positive class count is zero; ratio cannot be computed.")

    return float(n_neg) / float(n_pos)
