"""Loading, cleaning and user-level consolidation of the campaign data.

The raw file has one row per campaign exposure. The analysis works on a
canonical base with one row per *stable* user, that is, a user shown a
single variant, so the A/B attribution is not ambiguous.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from marketing_campaign.config import DEFAULT_DATASET_PATH

STRING_COLUMNS = [
    "marketing_channel",
    "variant",
    "language_displayed",
    "language_preferred",
    "age_group",
    "subscribing_channel",
]
DATE_COLUMNS = ["date_served", "date_subscribed", "date_canceled"]
BLANK_VALUES = ["nan", "None", "NaN", ""]


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the marketing campaign dataset.

    Parameters
    ----------
    path : str or Path, optional
        CSV file to read. Defaults to ``dataset/marketing_new.csv``
        inside the project folder.

    Returns
    -------
    pd.DataFrame
        Raw campaign exposures, indexed by the first CSV column.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    csv_path = Path(path) if path else DEFAULT_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path, index_col=0)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw exposures and derive the analysis flags.

    Duplicated ``user_id`` and ``date_served`` pairs are dropped, dates
    are parsed, text columns are stripped and blanks become missing, and
    the binary ``converted_flag`` and ``retained_flag`` columns, the
    calendar fields and ``is_correct_lang`` are derived.

    Parameters
    ----------
    df : pd.DataFrame
        Raw exposures as returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned exposures, one row per user and date served.
    """
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=["user_id", "date_served"])

    for column in DATE_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    for column in STRING_COLUMNS:
        if column in cleaned.columns:
            text = cleaned[column].astype(str).str.strip()
            cleaned[column] = text.mask(text.isin(BLANK_VALUES), np.nan)

    cleaned["converted_flag"] = _flag(cleaned, "converted")
    cleaned["retained_flag"] = _flag(cleaned, "is_retained")

    if "date_served" in cleaned.columns:
        served = cleaned["date_served"]
        cleaned["day_of_week"] = served.dt.day_name()
        cleaned["month"] = served.dt.month
        cleaned["campaign_period"] = served.dt.to_period("M").astype(str)

    if "is_correct_lang" not in cleaned.columns:
        if {"language_displayed", "language_preferred"}.issubset(
            cleaned.columns
        ):
            same = cleaned["language_displayed"] == cleaned[
                "language_preferred"
            ]
            cleaned["is_correct_lang"] = same.astype(int)
        else:
            cleaned["is_correct_lang"] = 1
    return cleaned


def _flag(df: pd.DataFrame, column: str) -> pd.Series | int:
    """Turn a boolean-like column into a 0/1 integer flag.

    Parameters
    ----------
    df : pd.DataFrame
        Frame that may hold ``column``.
    column : str
        Name of the boolean-like column.

    Returns
    -------
    pd.Series or int
        The column as 0/1 integers (missing values count as 0), or ``0``
        when the column does not exist.
    """
    if column not in df.columns:
        return 0
    return df[column].astype("boolean").fillna(False).astype(int)


def aggregate_stable_users(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Consolidate exposures into one row per user with a single variant.

    Users exposed to more than one variant are excluded. Conversion and
    retention become "ever" flags, aggregated by the maximum.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned exposures from :func:`clean_data`.

    Returns
    -------
    tuple of (pd.DataFrame, int)
        The canonical user-level base and the number of users who
        switched variants and were excluded.
    """
    cardinality = df.groupby("user_id")["variant"].nunique(dropna=False)
    stable_ids = cardinality[cardinality == 1].index
    switcher_count = int((cardinality > 1).sum())

    stable = (
        df[df["user_id"].isin(stable_ids)]
        .sort_values(["user_id", "date_served"])
        .copy()
    )
    users = stable.groupby("user_id", as_index=False).agg(
        date_served=("date_served", "first"),
        marketing_channel=("marketing_channel", "first"),
        variant=("variant", "first"),
        language_displayed=("language_displayed", "first"),
        language_preferred=("language_preferred", "first"),
        age_group=("age_group", "first"),
        date_subscribed=("date_subscribed", "min"),
        date_canceled=("date_canceled", "max"),
        subscribing_channel=("subscribing_channel", "first"),
        converted_flag=("converted_flag", "max"),
        retained_flag=("retained_flag", "max"),
        is_correct_lang=("is_correct_lang", "first"),
    )
    users["converted"] = users["converted_flag"].astype(bool)
    users["is_retained"] = users["retained_flag"].astype(bool)
    users["day_of_week"] = users["date_served"].dt.day_name()
    users["month"] = users["date_served"].dt.month
    users["campaign_period"] = (
        users["date_served"].dt.to_period("M").astype(str)
    )
    return users, switcher_count


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the share of missing values of every column.

    Parameters
    ----------
    df : pd.DataFrame
        Any table.

    Returns
    -------
    pd.DataFrame
        Columns ``Column`` and ``% Missing``, sorted from the most to the
        least incomplete column.
    """
    missing_percent = df.isnull().sum() / len(df) * 100
    return pd.DataFrame(
        {
            "Column": missing_percent.index,
            "% Missing": missing_percent.values,
        }
    ).sort_values("% Missing", ascending=False)
