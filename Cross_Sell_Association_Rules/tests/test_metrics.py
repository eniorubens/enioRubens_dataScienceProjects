"""Tests for the association metrics."""
import math

import pytest

from cross_sell_rules import metrics


def test_independent_items_have_unit_lift_and_zero_zhang():
    """Independence gives lift 1, conviction 1 and Zhang 0."""
    assert metrics.lift(0.4, 0.5, 0.2) == pytest.approx(1.0)
    assert metrics.conviction(0.4, 0.5, 0.2) == pytest.approx(1.0)
    assert metrics.zhang(0.4, 0.5, 0.2) == pytest.approx(0.0)


def test_perfectly_associated_items():
    """Items that always appear together reach Zhang 1 and infinite
    conviction."""
    assert metrics.zhang(0.5, 0.5, 0.5) == pytest.approx(1.0)
    assert metrics.conviction(0.5, 0.5, 0.5) == math.inf
    assert metrics.lift(0.5, 0.5, 0.5) == pytest.approx(2.0)


def test_mutually_exclusive_items_have_zero_lift_and_negative_zhang():
    """Items that never co-occur are fully dissociated."""
    assert metrics.lift(0.3, 0.3, 0.0) == 0.0
    assert metrics.zhang(0.3, 0.3, 0.0) == pytest.approx(-1.0)


def test_zhang_is_zero_when_the_denominator_vanishes():
    """A degenerate rule does not raise a division error."""
    assert metrics.zhang(1.0, 1.0, 1.0) == 0.0


def test_confidence_is_the_conditional_frequency():
    """Confidence equals joint support over antecedent support."""
    assert metrics.confidence(0.25, 0.1) == pytest.approx(0.4)


@pytest.mark.parametrize(
    "call",
    [
        lambda: metrics.lift(1.2, 0.5, 0.2),
        lambda: metrics.lift(0.0, 0.5, 0.0),
        lambda: metrics.confidence(0.0, 0.0),
        lambda: metrics.zhang(0.5, -0.1, 0.0),
        lambda: metrics.conviction(0.5, 0.5, 2.0),
    ],
)
def test_invalid_supports_are_rejected(call):
    """Supports outside [0, 1] or zero marginals raise ValueError."""
    with pytest.raises(ValueError):
        call()


def test_popular_pair_values_from_the_notebook():
    """The whole milk and other vegetables figures are reproduced."""
    support_milk, support_veg, support_pair = 0.157923, 0.122101, 0.014837

    assert metrics.lift(support_milk, support_veg, support_pair) == (
        pytest.approx(0.77, abs=0.005)
    )
    assert metrics.conviction(support_milk, support_veg, support_pair) == (
        pytest.approx(0.97, abs=0.005)
    )
    assert metrics.zhang(support_milk, support_veg, support_pair) == (
        pytest.approx(-0.2625, abs=0.0005)
    )
