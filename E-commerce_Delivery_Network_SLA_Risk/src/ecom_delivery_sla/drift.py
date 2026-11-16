"""Population-drift diagnostics for temporal development partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    expected: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate PSI using quantile bins learned from the expected sample.

    Zero proportions are clipped to a small positive value so the logarithm is
    finite. The function is descriptive and does not choose a model or target.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    expected_values = pd.Series(expected).dropna().astype(float).to_numpy()
    actual_values = pd.Series(actual).dropna().astype(float).to_numpy()
    if expected_values.size == 0 or actual_values.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    expected_counts, _ = np.histogram(expected_values, bins=edges)
    actual_counts, _ = np.histogram(actual_values, bins=edges)
    expected_share = np.clip(expected_counts / expected_counts.sum(), 1e-6, None)
    actual_share = np.clip(actual_counts / actual_counts.sum(), 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def feature_psi_table(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Return PSI by numeric feature for two temporal populations."""
    selected = columns or expected.select_dtypes(include="number").columns.tolist()
    rows = [
        {
            "feature": column,
            "psi": population_stability_index(expected[column], actual[column], bins=bins),
        }
        for column in selected
        if column in expected and column in actual
    ]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
