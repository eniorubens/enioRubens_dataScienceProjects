"""Tests for dataset loading and basket construction."""
import pandas as pd
import pytest

from cross_sell_rules import data


def test_baskets_are_grouped_by_customer_and_date():
    """Rows sharing customer and date form one basket."""
    records = pd.DataFrame(
        {
            "Member_number": [1, 1, 1, 2],
            "Date": ["01-01-2015", "01-01-2015", "02-01-2015", "01-01-2015"],
            "itemDescription": ["milk", "bread", "milk", "beer"],
        }
    )

    baskets = data.build_baskets(records)

    assert baskets == [["milk", "bread"], ["milk"], ["beer"]]


def test_load_groceries_rejects_a_file_without_required_columns(tmp_path):
    """A CSV missing a required column is refused with a clear error."""
    broken = tmp_path / "broken.csv"
    pd.DataFrame({"Member_number": [1], "Date": ["x"]}).to_csv(
        broken, index=False
    )

    with pytest.raises(ValueError, match="itemDescription"):
        data.load_groceries(broken)


def test_co_count_counts_co_occurrence_not_identical_baskets():
    """Items count together even when the basket holds other products."""
    baskets = [["a", "b", "c"], ["a", "b"], ["a"], ["b"]]

    assert data.co_count(["a", "b"], baskets) == 2
    assert data.co_count(["c"], baskets) == 1


def test_summarize_baskets_counts_occurrences_baskets_and_items():
    """The summary reports records, baskets and distinct products."""
    summary = data.summarize_baskets([["a", "b"], ["a"], ["c", "a"]])

    assert summary == data.DatasetSummary(
        occurrences=5, transactions=3, items=3
    )


def test_real_dataset_matches_the_reported_counts(baskets):
    """The Groceries file yields the counts quoted in the report."""
    summary = data.summarize_baskets(baskets)

    assert summary.occurrences == 38765
    assert summary.transactions == 14963
    assert summary.items == 167


def test_real_dataset_co_occurrence_counts(baskets):
    """The co-occurrence counts shown in the notebook are reproduced."""
    assert data.co_count(["sausage", "whole milk"], baskets) == 134
    assert data.co_count(["whole milk", "pastry"], baskets) == 97
    assert data.co_count(["sausage", "hygiene articles"], baskets) == 13
