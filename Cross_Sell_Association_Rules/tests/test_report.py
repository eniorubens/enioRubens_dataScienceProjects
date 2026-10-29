"""Tests that protect the numbers published in the executive report."""
import pytest

from cross_sell_rules import compute_report_figures


@pytest.fixture(scope="module")
def figures():
    """Return the report figures recomputed from the raw dataset."""
    return compute_report_figures()


def test_dataset_counts(figures):
    """Occurrences, baskets and catalogue size match the report."""
    assert figures["occurrences"] == 38765
    assert figures["transactions"] == 14963
    assert figures["items"] == 167


def test_popular_pair_metrics(figures):
    """Supports and metrics of whole milk and other vegetables."""
    assert figures["support_whole_milk"] == pytest.approx(0.1579, abs=5e-5)
    assert figures["support_other_vegetables"] == pytest.approx(
        0.1221, abs=5e-5
    )
    assert figures["support_pair"] == pytest.approx(0.0148, abs=5e-5)
    assert figures["confidence_milk_to_vegetables"] == pytest.approx(
        0.09, abs=0.005
    )
    assert figures["confidence_vegetables_to_milk"] == pytest.approx(
        0.12, abs=0.005
    )
    assert figures["lift"] == pytest.approx(0.77, abs=0.005)
    assert figures["conviction"] == pytest.approx(0.97, abs=0.005)
    assert figures["zhang"] == pytest.approx(-0.2625, abs=5e-5)


def test_mining_counts(figures):
    """Itemset, rule and candidate counts match the report."""
    assert figures["itemsets_support_003"] == 216
    assert figures["itemsets_support_001"] == 750
    assert figures["rules_from_itemsets_003"] == 210
    assert figures["rules_from_itemsets_001"] == 648
    assert figures["cross_sell_candidates"] == 16
