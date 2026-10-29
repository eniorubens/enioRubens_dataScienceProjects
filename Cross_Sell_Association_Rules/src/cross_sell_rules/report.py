"""Reproduction of the figures quoted in the executive report."""
from __future__ import annotations

from pathlib import Path

from cross_sell_rules import data, metrics, mining

POPULAR_PAIR = ("whole milk", "other vegetables")


def compute_report_figures(
    path: str | Path | None = None,
) -> dict[str, float | int]:
    """Recompute the headline numbers of the analysis from the raw file.

    The thresholds are the ones used in the notebooks: Apriori at 0.003
    and 0.001 for the itemset counts, rules at support 0.0015 for the
    rule counts, and lift >= 1, Zhang > 0.2 and item support > 0.002
    for the cross-sell candidates.

    Parameters
    ----------
    path : str or Path, optional
        Groceries CSV file. Defaults to the dataset shipped with the
        project.

    Returns
    -------
    dict
        Counts and metrics keyed by name, for example ``transactions``,
        ``lift`` and ``zhang`` for the pair whole milk and other
        vegetables.
    """
    baskets = data.build_baskets(data.load_groceries(path))
    summary = data.summarize_baskets(baskets)
    onehot = mining.encode_baskets(baskets)
    n_transactions = summary.transactions

    first, second = POPULAR_PAIR
    support_first = float(onehot[first].mean())
    support_second = float(onehot[second].mean())
    support_pair = mining.joint_support(onehot, POPULAR_PAIR)

    itemsets_003 = mining.find_frequent_itemsets(onehot, 0.003, max_len=3)
    itemsets_001 = mining.find_frequent_itemsets(onehot, 0.001, max_len=3)
    rules_003 = mining.generate_rules(
        itemsets_003, n_transactions, "support", 0.0015
    )
    rules_001 = mining.generate_rules(
        itemsets_001, n_transactions, "support", 0.0015
    )

    pair_itemsets = mining.find_frequent_itemsets(onehot, 0.0015, max_len=2)
    pair_rules = mining.generate_rules(
        pair_itemsets, n_transactions, "lift", 1.0
    )
    candidates = mining.select_cross_sell_candidates(pair_rules)

    return {
        "occurrences": summary.occurrences,
        "transactions": summary.transactions,
        "items": summary.items,
        "support_whole_milk": support_first,
        "support_other_vegetables": support_second,
        "support_pair": support_pair,
        "confidence_milk_to_vegetables": metrics.confidence(
            support_first, support_pair
        ),
        "confidence_vegetables_to_milk": metrics.confidence(
            support_second, support_pair
        ),
        "lift": metrics.lift(support_first, support_second, support_pair),
        "conviction": metrics.conviction(
            support_first, support_second, support_pair
        ),
        "zhang": metrics.zhang(support_first, support_second, support_pair),
        "itemsets_support_003": len(itemsets_003),
        "itemsets_support_001": len(itemsets_001),
        "rules_from_itemsets_003": len(rules_003),
        "rules_from_itemsets_001": len(rules_001),
        "cross_sell_candidates": len(candidates),
    }
