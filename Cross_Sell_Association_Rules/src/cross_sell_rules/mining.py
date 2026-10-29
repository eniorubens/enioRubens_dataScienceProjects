"""Frequent itemset mining and rule selection with mlxtend."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from inspect import signature

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def encode_baskets(baskets: Sequence[Sequence[str]]) -> pd.DataFrame:
    """One-hot encode baskets into a boolean transaction matrix.

    Parameters
    ----------
    baskets : sequence of sequence of str
        Baskets as returned by :func:`cross_sell_rules.data.build_baskets`.

    Returns
    -------
    pd.DataFrame
        One row per basket and one boolean column per product, set to
        ``True`` when the product is in the basket.
    """
    encoder = TransactionEncoder().fit(baskets)
    return pd.DataFrame(encoder.transform(baskets), columns=encoder.columns_)


def item_support(onehot: pd.DataFrame) -> pd.Series:
    """Compute the support of every product.

    Parameters
    ----------
    onehot : pd.DataFrame
        Transaction matrix from :func:`encode_baskets`.

    Returns
    -------
    pd.Series
        Share of baskets containing each product, largest first.
    """
    return onehot.mean().sort_values(ascending=False)


def joint_support(onehot: pd.DataFrame, items: Iterable[str]) -> float:
    """Compute the support of several products bought together.

    Parameters
    ----------
    onehot : pd.DataFrame
        Transaction matrix from :func:`encode_baskets`.
    items : iterable of str
        Products that must all be in the basket.

    Returns
    -------
    float
        Share of baskets containing all the given products.
    """
    return float(onehot[list(items)].all(axis=1).mean())


def find_frequent_itemsets(
    onehot: pd.DataFrame, min_support: float, max_len: int
) -> pd.DataFrame:
    """Mine frequent itemsets with the Apriori algorithm.

    Parameters
    ----------
    onehot : pd.DataFrame
        Transaction matrix from :func:`encode_baskets`.
    min_support : float
        Minimum support for an itemset to be kept.
    max_len : int
        Maximum number of products in an itemset.

    Returns
    -------
    pd.DataFrame
        Columns ``support`` and ``itemsets`` (a frozenset of names).
    """
    return apriori(
        onehot, min_support=min_support, max_len=max_len, use_colnames=True
    )


def generate_rules(
    itemsets: pd.DataFrame,
    n_transactions: int,
    metric: str,
    min_threshold: float,
) -> pd.DataFrame:
    """Derive association rules from frequent itemsets.

    Parameters
    ----------
    itemsets : pd.DataFrame
        Output of :func:`find_frequent_itemsets`.
    n_transactions : int
        Number of baskets the itemsets were mined from. It is forwarded
        to mlxtend releases whose ``association_rules`` requires it.
    metric : str
        Metric the threshold applies to (for example ``"lift"``).
    min_threshold : float
        Minimum value of ``metric`` for a rule to be kept.

    Returns
    -------
    pd.DataFrame
        One row per rule with support, confidence, lift, conviction and
        Zhang's metric, among others. Rows are sorted by antecedent and
        then consequent, and re-indexed, because the order in which mlxtend
        returns the rules changes between runs (it depends on the hash
        seed of the Python process).
    """
    options = {}
    if "num_itemsets" in signature(association_rules).parameters:
        options["num_itemsets"] = n_transactions
    rules = association_rules(
        itemsets, metric=metric, min_threshold=min_threshold, **options
    )
    if rules.empty:
        return rules
    key = (
        rules["antecedents"].apply(format_itemset)
        + " -> "
        + rules["consequents"].apply(format_itemset)
    )
    return rules.iloc[key.argsort(kind="stable")].reset_index(drop=True)


def select_cross_sell_candidates(
    rules: pd.DataFrame,
    min_lift: float = 1.0,
    min_item_support: float = 0.002,
    min_zhang: float = 0.2,
) -> pd.DataFrame:
    """Keep the rules that qualify as cross-sell candidates.

    A rule qualifies when its lift reaches ``min_lift``, both the
    antecedent and the consequent are above ``min_item_support`` and
    Zhang's metric is above ``min_zhang``.

    Parameters
    ----------
    rules : pd.DataFrame
        Output of :func:`generate_rules`.
    min_lift : float, default=1.0
        Minimum lift, inclusive.
    min_item_support : float, default=0.002
        Lower bound, exclusive, for the antecedent and consequent supports.
    min_zhang : float, default=0.2
        Lower bound, exclusive, for Zhang's metric.

    Returns
    -------
    pd.DataFrame
        The qualifying rules, with ``antecedents`` and ``consequents``
        rendered as comma-separated, alphabetically sorted strings.
    """
    selected = rules[
        (rules["lift"] >= min_lift)
        & (rules["antecedent support"] > min_item_support)
        & (rules["consequent support"] > min_item_support)
        & (rules["zhangs_metric"] > min_zhang)
    ].copy()
    for column in ("antecedents", "consequents"):
        selected[column] = selected[column].apply(format_itemset)
    return selected


def format_itemset(itemset: Iterable[str]) -> str:
    """Render an itemset as a sorted, comma-separated string.

    Parameters
    ----------
    itemset : iterable of str
        Product names, for example a frozenset from mlxtend.

    Returns
    -------
    str
        Names in alphabetical order joined by ``", "``.
    """
    return ", ".join(sorted(itemset))


def rules_to_coordinates(rules: pd.DataFrame) -> pd.DataFrame:
    """Prepare rules for a parallel-coordinates plot.

    Each rule is described by its first antecedent and first consequent
    product, so it is meant for rules with a single item on each side.

    Parameters
    ----------
    rules : pd.DataFrame
        Rules from :func:`generate_rules`, with the ``antecedents`` and
        ``consequents`` columns holding sets of product names.

    Returns
    -------
    pd.DataFrame
        Columns ``antecedent``, ``consequent`` and ``rule`` (the index of
        the rule in ``rules``). The input is not modified.
    """
    return pd.DataFrame(
        {
            "antecedent": rules["antecedents"].apply(lambda s: list(s)[0]),
            "consequent": rules["consequents"].apply(lambda s: list(s)[0]),
            "rule": rules.index,
        }
    )
