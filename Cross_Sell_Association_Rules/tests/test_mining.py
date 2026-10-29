"""Tests for itemset mining and cross-sell candidate selection."""
import pandas as pd
import pytest

from cross_sell_rules import metrics, mining

HIGHLIGHTED_RULES = {
    ("sausage", "beverages"): 1.54,
    ("napkins", "pastry"): 1.52,
    ("curd", "sausage"): 1.45,
    ("brown bread", "canned beer"): 1.36,
}


def test_encode_baskets_builds_a_boolean_matrix():
    """Each basket becomes a row and each product a boolean column."""
    onehot = mining.encode_baskets([["a", "b"], ["b"], ["c", "a"]])

    assert list(onehot.columns) == ["a", "b", "c"]
    assert onehot.to_numpy().tolist() == [
        [True, True, False],
        [False, True, False],
        [True, False, True],
    ]


def test_item_support_is_sorted_and_joint_support_requires_all_items():
    """Supports are shares of baskets, largest first."""
    onehot = mining.encode_baskets([["a", "b"], ["b"], ["c", "a"], ["b"]])

    assert mining.item_support(onehot).to_dict() == {
        "b": 0.75,
        "a": 0.5,
        "c": 0.25,
    }
    assert list(mining.item_support(onehot).index)[0] == "b"
    assert mining.joint_support(onehot, ["a", "b"]) == pytest.approx(0.25)


def test_format_itemset_sorts_names():
    """Itemsets are rendered alphabetically, whatever the input order."""
    assert mining.format_itemset(frozenset({"pastry", "milk"})) == (
        "milk, pastry"
    )


def test_selection_applies_every_filter():
    """Lift, item support and Zhang each remove the rules they fail."""
    rules = pd.DataFrame(
        {
            "antecedents": [frozenset({"a"})] * 4,
            "consequents": [frozenset({"b"})] * 4,
            "antecedent support": [0.1, 0.1, 0.001, 0.1],
            "consequent support": [0.1, 0.1, 0.1, 0.001],
            "lift": [1.5, 0.9, 1.5, 1.5],
            "zhangs_metric": [0.3, 0.3, 0.3, 0.3],
        }
    )
    rules.loc[3, "zhangs_metric"] = 0.1

    selected = mining.select_cross_sell_candidates(rules)

    assert selected.index.tolist() == [0]
    assert selected.loc[0, "antecedents"] == "a"
    assert selected.loc[0, "consequents"] == "b"


def test_selection_bounds_follow_the_notebook():
    """Lift is inclusive; Zhang and item support are exclusive."""
    rules = pd.DataFrame(
        {
            "antecedents": [frozenset({"a"}), frozenset({"a"})],
            "consequents": [frozenset({"b"}), frozenset({"b"})],
            "antecedent support": [0.1, 0.002],
            "consequent support": [0.1, 0.1],
            "lift": [1.0, 1.5],
            "zhangs_metric": [0.25, 0.25],
        }
    )

    assert mining.select_cross_sell_candidates(rules).index.tolist() == [0]
    assert mining.select_cross_sell_candidates(
        rules, min_zhang=0.25
    ).empty


@pytest.fixture(scope="module")
def candidates(onehot):
    """Return the cross-sell candidates mined from the real dataset."""
    itemsets = mining.find_frequent_itemsets(onehot, 0.0015, max_len=2)
    rules = mining.generate_rules(itemsets, len(onehot), "lift", 1.0)
    return mining.select_cross_sell_candidates(rules)


def test_candidates_satisfy_the_documented_filters(candidates):
    """Every candidate has lift >= 1 and Zhang > 0.2."""
    assert len(candidates) == 16
    assert (candidates["lift"] >= 1.0).all()
    assert (candidates["zhangs_metric"] > 0.2).all()


def test_highlighted_rules_are_among_the_candidates(candidates):
    """The four rules featured in the report appear with their lifts."""
    found = {
        (row.antecedents, row.consequents): row.lift
        for row in candidates.itertuples()
    }

    for pair, expected_lift in HIGHLIGHTED_RULES.items():
        assert found[pair] == pytest.approx(expected_lift, abs=0.005)


def test_popular_pair_is_not_a_candidate(candidates):
    """Whole milk and other vegetables fail the lift filter."""
    pairs = set(zip(candidates["antecedents"], candidates["consequents"]))

    assert ("whole milk", "other vegetables") not in pairs
    assert ("other vegetables", "whole milk") not in pairs


def test_mlxtend_columns_agree_with_the_metric_functions(candidates):
    """The library's lift, conviction and Zhang match our formulas."""
    for _, row in candidates.iterrows():
        args = (
            row["antecedent support"],
            row["consequent support"],
            row["support"],
        )
        assert metrics.lift(*args) == pytest.approx(row["lift"])
        assert metrics.conviction(*args) == pytest.approx(row["conviction"])
        assert metrics.zhang(*args) == pytest.approx(row["zhangs_metric"])
