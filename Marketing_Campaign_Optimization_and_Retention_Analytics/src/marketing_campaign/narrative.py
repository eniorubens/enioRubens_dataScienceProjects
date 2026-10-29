"""Small helpers used to write the narrative cells of the notebooks.

They read values from the result tables whatever the language of their
headers (English or Portuguese).
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

TRUE_TOKENS = {"1", "true", "yes", "y", "correct", "right", "certo", "sim"}
FALSE_TOKENS = {"0", "false", "no", "n", "wrong", "incorrect", "errado", "nao"}
METRIC_HEADERS = {"metric", "metrica"}
VALUE_HEADERS = {"value", "valor"}
NOT_RETAINED_MARKERS = ("not retained", "nao retido")


def safe_float(value: object, default: float = 0.0) -> float:
    """Convert a value to float, falling back to a default.

    Parameters
    ----------
    value : object
        Anything that may hold a number.
    default : float, default=0.0
        Returned when ``value`` is missing or not numeric.

    Returns
    -------
    float
        The converted value or ``default``.
    """
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def normalize_language_flag(value: object) -> float:
    """Read a correct-language flag as 1, 0 or ``NaN``.

    Parameters
    ----------
    value : object
        Flag such as ``True``, ``1``, ``"Yes"``, ``"sim"`` or ``"No"``.

    Returns
    -------
    float
        ``1`` for a correct language, ``0`` for a wrong one and ``NaN``
        when the value is missing or not recognised.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (0, 1) else np.nan
    if isinstance(value, (float, np.floating)):
        as_int = int(value)
        exact = float(as_int) == float(value)
        return as_int if as_int in (0, 1) and exact else np.nan

    text = str(value).strip().lower()
    if text in TRUE_TOKENS:
        return 1
    if text in FALSE_TOKENS:
        return 0
    return np.nan


def normalize_text(value: object) -> str:
    """Lower-case a text and strip its accents.

    Parameters
    ----------
    value : object
        Any value; it is converted with ``str``.

    Returns
    -------
    str
        ASCII, lower-case and stripped text.
    """
    text = unicodedata.normalize("NFKD", str(value))
    return text.encode("ascii", "ignore").decode("ascii").lower().strip()


def metric_column_name(df: pd.DataFrame) -> str:
    """Find the column that names the metrics in a metrics table.

    Parameters
    ----------
    df : pd.DataFrame
        Table with a ``Metric`` or ``Métrica`` column.

    Returns
    -------
    str
        The name of that column, or the first column when none matches.
    """
    for column in df.columns:
        if normalize_text(column) in METRIC_HEADERS:
            return column
    return df.columns[0]


def get_metric_value(
    metrics_df: pd.DataFrame, metric_name: str, default: float = 0
) -> float:
    """Read one metric from the retention metrics table.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of
        :func:`marketing_campaign.retention_model.make_metrics_table`,
        possibly localised.
    metric_name : str
        Name of the metric, compared without accents or case.
    default : float, default=0
        Returned when the metric is not found.

    Returns
    -------
    float
        The metric value. A name containing "recall" also matches the
        recall of the not-retained class when there is no exact match.
    """
    metric_col = metric_column_name(metrics_df)
    value_col = next(
        (c for c in metrics_df.columns if normalize_text(c) in VALUE_HEADERS),
        metrics_df.columns[-1],
    )
    names = metrics_df[metric_col].astype(str).map(normalize_text)
    target = normalize_text(metric_name)
    values = metrics_df.loc[names.eq(target), value_col]

    if values.empty and "recall" in target:
        not_retained = names.apply(
            lambda name: any(m in name for m in NOT_RETAINED_MARKERS)
        )
        values = metrics_df.loc[
            names.str.contains("recall", na=False) & not_retained,
            value_col,
        ]
    if values.empty:
        return default
    return float(values.iloc[0])
