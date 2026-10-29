"""Language impact and error analysis of the retention model."""
from __future__ import annotations

import numpy as np
import pandas as pd

LANGUAGE_FLAGS = {
    True: 1,
    False: 0,
    "True": 1,
    "False": 0,
    "true": 1,
    "false": 0,
    "Yes": 1,
    "No": 0,
    "yes": 1,
    "no": 0,
    1: 1,
    0: 0,
}
LANGUAGE_LABELS = {1: "Correct Language", 0: "Wrong Language"}
MISSING_LABEL = "Missing"
ERROR_COLUMNS = [
    "marketing_channel",
    "variant",
    "age_group",
    "language_displayed",
    "language_preferred",
    "is_correct_lang_flag",
]
BIAS_OUTPUT_COLUMNS = [
    "category",
    "base_count",
    "base_pct",
    "false_positive_count",
    "false_positive_pct",
    "diff_pp",
    "interpretation",
]
OVER_REPRESENTED = "Over-represented in false positives"
UNDER_REPRESENTED = "Under-represented in false positives"
PROPORTIONAL = "Proportional distribution"


def analyze_language_performance(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compare conversion and retention by correct or wrong language.

    Parameters
    ----------
    df : pd.DataFrame
        User-level base with ``user_id``, ``converted_flag``,
        ``retained_flag`` and ``is_correct_lang``.

    Returns
    -------
    pd.DataFrame or None
        One row per language flag with users, conversions, retained
        users among those converted, the conversion rate over all users
        and the retention rate over converted users (both in percent),
        and a ``label``. ``None`` when a required column is missing.
    """
    required = [
        "is_correct_lang",
        "user_id",
        "converted_flag",
        "retained_flag",
    ]
    if any(column not in df.columns for column in required):
        return None

    data = df.copy()
    for flag in ["converted_flag", "retained_flag"]:
        data[flag] = (
            pd.to_numeric(data[flag], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=1)
            .astype(int)
        )
    data["is_correct_lang"] = data["is_correct_lang"].map(LANGUAGE_FLAGS)

    summary = (
        data.groupby("is_correct_lang", dropna=False)
        .agg(
            user_id=("user_id", "nunique"),
            converted_flag=("converted_flag", "sum"),
        )
        .reset_index()
    )
    retained = (
        data[data["converted_flag"] == 1]
        .groupby("is_correct_lang", dropna=False)
        .agg(retained_among_converted=("retained_flag", "sum"))
        .reset_index()
    )
    summary = summary.merge(retained, on="is_correct_lang", how="left")
    summary["retained_among_converted"] = (
        summary["retained_among_converted"].fillna(0).astype(int)
    )
    summary["conversion_rate_%"] = (
        summary["converted_flag"]
        / summary["user_id"].replace(0, np.nan)
        * 100
    ).round(2).fillna(0)
    summary["retention_rate_%"] = (
        summary["retained_among_converted"]
        / summary["converted_flag"].replace(0, np.nan)
        * 100
    ).round(2).fillna(0)
    summary["label"] = (
        summary["is_correct_lang"].map(LANGUAGE_LABELS).fillna("Not informed")
    )
    return summary


def classify_errors(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the test set into false positives and false negatives.

    Parameters
    ----------
    X_test : pd.DataFrame
        Test features; its index is kept in ``original_index``.
    y_test : pd.Series
        True labels (1 = retained).
    y_pred : np.ndarray
        Predicted labels.
    y_pred_proba : np.ndarray
        Predicted probability of retention.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        The test set with ``y_true``, ``y_pred`` and ``prob_retention``,
        its false positives (``y_true`` 0, ``y_pred`` 1) and its false
        negatives (``y_true`` 1, ``y_pred`` 0).
    """
    errors = X_test.copy().reset_index(drop=True)
    errors.insert(0, "original_index", X_test.index.to_numpy())
    errors["y_true"] = np.asarray(y_test)
    errors["y_pred"] = np.asarray(y_pred)
    errors["prob_retention"] = np.asarray(y_pred_proba)

    false_positives = errors[
        (errors["y_true"] == 0) & (errors["y_pred"] == 1)
    ].copy()
    false_negatives = errors[
        (errors["y_true"] == 1) & (errors["y_pred"] == 0)
    ].copy()
    return errors, false_positives, false_negatives


def error_distribution(
    errors: pd.DataFrame, column: str
) -> pd.DataFrame | None:
    """Count how often each category of ``column`` appears in the errors.

    Parameters
    ----------
    errors : pd.DataFrame
        Misclassified rows, for example the false positives.
    column : str
        Categorical column to tabulate.

    Returns
    -------
    pd.DataFrame or None
        Columns ``column``, ``quantity`` and ``percentage`` (of the
        errors). Missing values count as ``"Missing"``. ``None`` when
        ``column`` is absent.
    """
    if column not in errors.columns:
        return None
    values = errors[column].astype("object")
    values = values.where(values.notna(), MISSING_LABEL)
    table = (
        values.value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="quantity")
    )
    table["percentage"] = (table["quantity"] / len(errors) * 100).round(2)
    return table


def compare_false_positives(
    full_df: pd.DataFrame,
    false_positive_df: pd.DataFrame,
    column: str,
    threshold_pp: float = 10.0,
) -> pd.DataFrame:
    """Compare the mix of a variable in the false positives with the base.

    Parameters
    ----------
    full_df : pd.DataFrame
        Complete dataset, the reference for the natural distribution.
    false_positive_df : pd.DataFrame
        Only the false positives of the retention model.
    column : str
        Categorical variable to compare; it must exist in both frames.
    threshold_pp : float, default=10.0
        Difference in percentage points from which a category counts as
        over- or under-represented.

    Returns
    -------
    pd.DataFrame
        One row per category with the counts and percentages in the base
        and in the false positives, ``diff_pp`` (false-positive share
        minus base share) and a textual ``interpretation``, sorted by
        ``diff_pp``, largest first.
    """
    if full_df.empty and false_positive_df.empty:
        return pd.DataFrame(columns=BIAS_OUTPUT_COLUMNS)
    base_values = full_df[column].astype("object")
    base_values = base_values.where(base_values.notna(), MISSING_LABEL)
    error_values = false_positive_df[column].astype("object")
    error_values = error_values.where(error_values.notna(), MISSING_LABEL)

    base_total = len(full_df)
    error_total = len(false_positive_df)
    base_denominator = np.nan if base_total == 0 else base_total
    error_denominator = np.nan if error_total == 0 else error_total

    comparison = (
        pd.concat(
            [
                base_values.value_counts(dropna=False).rename("base_count"),
                error_values.value_counts(dropna=False).rename(
                    "false_positive_count"
                ),
            ],
            axis=1,
        )
        .fillna(0)
        .rename_axis("category")
        .reset_index()
    )
    counts = ["base_count", "false_positive_count"]
    comparison[counts] = comparison[counts].astype(int)
    comparison["base_pct"] = (
        comparison["base_count"] / base_denominator * 100
    ).round(2).fillna(0)
    comparison["false_positive_pct"] = (
        comparison["false_positive_count"] / error_denominator * 100
    ).round(2).fillna(0)
    comparison["diff_pp"] = (
        comparison["false_positive_pct"] - comparison["base_pct"]
    ).round(2)
    comparison["interpretation"] = np.select(
        [
            comparison["diff_pp"] >= threshold_pp,
            comparison["diff_pp"] <= -threshold_pp,
        ],
        [OVER_REPRESENTED, UNDER_REPRESENTED],
        default=PROPORTIONAL,
    )
    comparison = comparison.sort_values("diff_pp", ascending=False)
    return comparison.reset_index(drop=True)[BIAS_OUTPUT_COLUMNS]


def analyze_false_positive_bias(
    full_df: pd.DataFrame,
    false_positive_df: pd.DataFrame,
    columns: list[str],
    threshold_pp: float = 10.0,
) -> dict[str, pd.DataFrame]:
    """Compare the false positives with the full dataset, per variable.

    Parameters
    ----------
    full_df : pd.DataFrame
        Complete dataset used as the reference distribution.
    false_positive_df : pd.DataFrame
        Only the false positives of the retention model.
    columns : list of str
        Categorical variables to compare. Those missing from either
        frame are ignored.
    threshold_pp : float, default=10.0
        Difference in percentage points that flags a category.

    Returns
    -------
    dict of str to pd.DataFrame
        The output of :func:`compare_false_positives` for each valid
        variable. Empty when there is nothing to compare.
    """
    valid = [
        column
        for column in columns
        if column in full_df.columns and column in false_positive_df.columns
    ]
    return {
        column: compare_false_positives(
            full_df, false_positive_df, column, threshold_pp
        )
        for column in valid
    }
