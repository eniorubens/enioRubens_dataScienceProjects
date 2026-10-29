"""Loading and basket construction for the Groceries dataset.

The raw file has one row per purchased item (``Member_number``, ``Date``,
``itemDescription``). A basket (transaction) is the set of items bought by
one customer on one date.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "dataset" / "Groceries_dataset.csv"
)
BASKET_KEYS = ["Member_number", "Date"]
ITEM_COLUMN = "itemDescription"


@dataclass(frozen=True)
class DatasetSummary:
    """Headline counts of a collection of baskets.

    Attributes
    ----------
    occurrences : int
        Total number of item records (rows of the raw file).
    transactions : int
        Number of baskets (customer x date).
    items : int
        Number of distinct products in the catalogue.
    """

    occurrences: int
    transactions: int
    items: int


def load_groceries(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw Groceries dataset.

    Parameters
    ----------
    path : str or Path, optional
        CSV file to read. Defaults to ``dataset/Groceries_dataset.csv``
        inside the project folder.

    Returns
    -------
    pd.DataFrame
        One row per purchased item, with the columns ``Member_number``,
        ``Date`` and ``itemDescription``.

    Raises
    ------
    ValueError
        If any of the required columns is missing from the file.
    """
    records = pd.read_csv(path or DEFAULT_DATASET_PATH)
    missing = [
        column
        for column in [*BASKET_KEYS, ITEM_COLUMN]
        if column not in records.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return records


def build_baskets(records: pd.DataFrame) -> list[list[str]]:
    """Group item records into baskets by customer and date.

    Parameters
    ----------
    records : pd.DataFrame
        Raw records as returned by :func:`load_groceries`.

    Returns
    -------
    list of list of str
        One list of item names per ``Member_number`` x ``Date`` pair,
        ordered by the grouping keys. Repeated purchases of the same item
        in a basket are kept.
    """
    grouped = records.groupby(BASKET_KEYS)[ITEM_COLUMN].apply(list)
    return grouped.reset_index()[ITEM_COLUMN].tolist()


def co_count(items: Iterable[str], baskets: Sequence[Sequence[str]]) -> int:
    """Count the baskets that contain every one of the given items.

    Parameters
    ----------
    items : iterable of str
        Products that must all be present in the basket.
    baskets : sequence of sequence of str
        Baskets as returned by :func:`build_baskets`.

    Returns
    -------
    int
        Number of baskets in which the items co-occur.
    """
    target = set(items)
    return sum(1 for basket in baskets if target.issubset(basket))


def summarize_baskets(baskets: Sequence[Sequence[str]]) -> DatasetSummary:
    """Compute the occurrence, transaction and catalogue counts.

    Parameters
    ----------
    baskets : sequence of sequence of str
        Baskets as returned by :func:`build_baskets`.

    Returns
    -------
    DatasetSummary
        Total item occurrences, number of baskets and distinct items.
    """
    flat = [item for basket in baskets for item in basket]
    return DatasetSummary(
        occurrences=len(flat),
        transactions=len(baskets),
        items=len(set(flat)),
    )
