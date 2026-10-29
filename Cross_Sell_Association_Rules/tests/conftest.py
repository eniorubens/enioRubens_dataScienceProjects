"""Shared fixtures: the real Groceries dataset, loaded once per session."""
import pytest

from cross_sell_rules import data, mining


@pytest.fixture(scope="session")
def records():
    """Return the raw Groceries records."""
    return data.load_groceries()


@pytest.fixture(scope="session")
def baskets(records):
    """Return the baskets built from the raw records."""
    return data.build_baskets(records)


@pytest.fixture(scope="session")
def onehot(baskets):
    """Return the one-hot transaction matrix of the real dataset."""
    return mining.encode_baskets(baskets)
