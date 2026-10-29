"""Association metrics computed from supports.

Every function takes the supports of the antecedent ``A``, of the
consequent ``B`` and of their joint occurrence ``A and B``, all as
fractions of the transactions in ``[0, 1]``.
"""
from __future__ import annotations

import math


def _validate_supports(*supports: float) -> None:
    """Check that every support is a fraction between 0 and 1.

    Parameters
    ----------
    *supports : float
        Supports to validate.

    Raises
    ------
    ValueError
        If any support lies outside ``[0, 1]``.
    """
    for value in supports:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Support must be in [0, 1], got {value}.")


def confidence(support_a: float, support_ab: float) -> float:
    """Return the confidence of the rule ``A -> B``.

    Parameters
    ----------
    support_a : float
        Support of the antecedent.
    support_ab : float
        Support of ``A and B``.

    Returns
    -------
    float
        ``P(B | A)``, the share of baskets with ``A`` that also hold ``B``.

    Raises
    ------
    ValueError
        If a support is out of range or ``support_a`` is zero.
    """
    _validate_supports(support_a, support_ab)
    if support_a == 0:
        raise ValueError("The antecedent support must be greater than zero.")
    return support_ab / support_a


def lift(support_a: float, support_b: float, support_ab: float) -> float:
    """Return the lift of the rule ``A -> B``.

    Parameters
    ----------
    support_a : float
        Support of the antecedent.
    support_b : float
        Support of the consequent.
    support_ab : float
        Support of ``A and B``.

    Returns
    -------
    float
        Observed co-occurrence divided by the one expected under
        independence. Values above 1 indicate a positive association.

    Raises
    ------
    ValueError
        If a support is out of range or either marginal support is zero.
    """
    _validate_supports(support_a, support_b, support_ab)
    if support_a == 0 or support_b == 0:
        raise ValueError("Marginal supports must be greater than zero.")
    return support_ab / (support_a * support_b)


def conviction(
    support_a: float, support_b: float, support_ab: float
) -> float:
    """Return the conviction of the rule ``A -> B``.

    Parameters
    ----------
    support_a : float
        Support of the antecedent.
    support_b : float
        Support of the consequent.
    support_ab : float
        Support of ``A and B``.

    Returns
    -------
    float
        Ratio between the expected and the observed frequency of ``A``
        appearing without ``B``. Values below 1 mean that ``A`` barely
        predicts ``B``. It is ``math.inf`` when ``B`` always follows ``A``.

    Raises
    ------
    ValueError
        If any support is out of range.
    """
    _validate_supports(support_a, support_b, support_ab)
    support_a_not_b = support_a - support_ab
    if support_a_not_b <= 0:
        return math.inf
    return support_a * (1.0 - support_b) / support_a_not_b


def zhang(support_a: float, support_b: float, support_ab: float) -> float:
    """Return Zhang's metric for the rule ``A -> B``.

    Parameters
    ----------
    support_a : float
        Support of the antecedent.
    support_b : float
        Support of the consequent.
    support_ab : float
        Support of ``A and B``.

    Returns
    -------
    float
        Association strength on a scale from -1 (dissociation) to 1
        (perfect association). It is 0 when the denominator vanishes.

    Raises
    ------
    ValueError
        If any support is out of range.
    """
    _validate_supports(support_a, support_b, support_ab)
    numerator = support_ab - support_a * support_b
    denominator = max(
        support_ab * (1.0 - support_a),
        support_a * (support_b - support_ab),
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator
