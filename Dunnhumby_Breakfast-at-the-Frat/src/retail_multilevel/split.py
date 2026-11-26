"""Temporal split (D11): the last 52 weeks (one full seasonal cycle) are the test period.

Model selection (step-up, LR tests, ICCs) uses the training weeks only; the test weeks
are opened once, for the out-of-sample comparison of OLS, multilevel and GPBoost.
"""

from __future__ import annotations

import pandas as pd

from retail_multilevel.cleaning import N_WEEKS_TOTAL

TEST_WEEKS = 52
FIRST_TEST_WEEK = N_WEEKS_TOTAL - TEST_WEEKS  # WEEK index 104 = 2011-01-12


def is_test(panel: pd.DataFrame, first_test_week: int = FIRST_TEST_WEEK) -> pd.Series:
    """Boolean mask of the test weeks."""
    return panel["WEEK"] >= first_test_week


def model_sample(panel: pd.DataFrame, part: str = "train", first_test_week: int = FIRST_TEST_WEEK) -> pd.DataFrame:
    """Return the modelling rows (``IN_MODEL``) of ``part`` ('train', 'test' or 'all')."""
    if part not in ("train", "test", "all"):
        raise ValueError(f"part must be 'train', 'test' or 'all', got {part!r}")
    df = panel.loc[panel["IN_MODEL"]]
    if part == "all":
        return df
    test = is_test(df, first_test_week)
    return df.loc[test if part == "test" else ~test]
