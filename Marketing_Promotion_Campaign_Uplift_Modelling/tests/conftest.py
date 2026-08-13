import matplotlib

matplotlib.use("Agg")

import pytest

from src.data import load_hillstrom


@pytest.fixture(scope="session")
def df():
    """Dataset Hillstrom completo, carregado uma única vez por sessão de testes."""
    return load_hillstrom()
