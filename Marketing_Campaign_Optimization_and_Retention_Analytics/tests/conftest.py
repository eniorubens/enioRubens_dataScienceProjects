"""Shared fixtures: the real campaign data, processed once per session."""
import pytest

from marketing_campaign.pipeline import run_pipeline


@pytest.fixture(scope="session")
def result():
    """Return the full pipeline result on the real dataset."""
    return run_pipeline()
