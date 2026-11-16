"""Tests for scripts/audit_unknown_categories.py on synthetic partitions (no raw data needed)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_unknown_categories.py"
_spec = importlib.util.spec_from_file_location("audit_unknown_categories", SCRIPT)
audit_unknown = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_unknown)


def test_unknown_values_are_counted_and_compared_with_the_dropped_reference():
    train = pd.DataFrame({"seller_state": ["AM", "SP", "SP", "MG", None], "sla_breach": [0, 0, 1, 0, 0]})
    evaluation = pd.DataFrame(
        {
            "seller_state": ["SP", "MA", "PI", "MA", None],
            "sla_breach": [0, 1, 0, 0, 1],
            "seller_cold_start": [0, 1, 0, 1, 0],
        }
    )
    before = evaluation.copy()

    result = audit_unknown.unknown_category_stats(train, evaluation, ["seller_state"], "fold x")
    row = result.iloc[0]

    pd.testing.assert_frame_equal(evaluation, before)
    assert row["unknown_rows"] == 3  # MA, PI, MA; the missing value is imputed with the train mode (SP)
    assert row["unknown_values"] == "MA, PI"
    assert row["reference_category"] == "AM"  # first sorted category, dropped by OneHotEncoder(drop="first")
    assert row["reference_rows_train"] == 1 and row["reference_rows_evaluation"] == 0
    assert math.isclose(row["breach_rate_unknown"], 1 / 3)
    assert math.isclose(row["seller_cold_start_share_unknown"], 2 / 3)
    assert math.isnan(row["edge_cold_start_share_unknown"])


def test_no_unknown_values_gives_zero_and_nan_rates():
    train = pd.DataFrame({"customer_state": ["SP", "RJ"], "sla_breach": [0, 1]})
    evaluation = pd.DataFrame({"customer_state": ["RJ", "SP"], "sla_breach": [1, 0]})
    row = audit_unknown.unknown_category_stats(train, evaluation, ["customer_state"], "F1 test").iloc[0]
    assert row["unknown_rows"] == 0 and row["unknown_share"] == 0
    assert math.isnan(row["breach_rate_unknown"])
