"""Targeted tests for the S5 estimator benchmark."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from consumer_complaint_intelligence.s5 import (
    CRITICAL_CLASS,
    _check_reference,
    load_s5_config,
    run_s5_smoke,
    validate_scientific_cache,
)
from consumer_complaint_intelligence.s5_reporting import build_s5_report_tables


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "s5_estimator_benchmark.json"


def _frame() -> pa.Table:
    """Build a small development-only cache covering every class."""

    rows = []
    identifier = 1
    for label in (
        "credit_reporting",
        "debt_collection",
        "mortgage",
        "deposit_accounts",
        "cards_prepaid",
        "money_services",
        "student_loan",
        "consumer_lending",
        CRITICAL_CLASS,
    ):
        for partition in ("train", "validation"):
            for repeat in range(3):
                rows.append({
                    "Complaint ID": identifier,
                    "received_date": "2024-01-01",
                    "product_family": label,
                    "normalized_group_hash": f"hash-{identifier}",
                    "normalized_length": identifier,
                    "partition_name": partition,
                    "narrative": f"complaint {label} reference token {repeat}",
                })
                identifier += 1
    return pa.Table.from_pylist(rows)


class S5Tests(unittest.TestCase):
    """Verify S5 configuration, boundary, smoke execution, and reporting."""

    def test_config_is_frozen_and_has_deferred_candidate(self) -> None:
        """Load exactly three local candidates and one deferred candidate."""

        config = load_s5_config(CONFIG)
        self.assertEqual(config.status, "FROZEN_FOR_S5_DEVELOPMENT")
        self.assertEqual(config.representation.max_features, 40000)
        self.assertEqual(len(config.candidates), 3)
        self.assertIn("logistic_regression_saga", config.deferred_candidates)

    def test_cache_rejects_sealed_partition(self) -> None:
        """Reject test rows before any estimator can be fitted."""

        table = _frame()
        values = [value.as_py() for value in table["partition_name"]]
        values[0] = "test"
        table = table.set_column(5, "partition_name", pa.array(values))
        with self.assertRaises(ValueError):
            validate_scientific_cache(table)

    def test_smoke_is_diagnostic_and_uses_one_vectorizer(self) -> None:
        """Run the deterministic cap without touching sealed data."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s5_results.json"
            pq.write_table(_frame(), cache)
            result = run_s5_smoke(cache, artifact, CONFIG, max_per_class=2)
            self.assertTrue(result["complete"])
            self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
            self.assertEqual(result["claim_boundary"], "NO_TEST_STRESS_OR_MONITOR_ACCESS")
            self.assertEqual(result["vectorizer_fit_count"], 1)
            self.assertEqual(len(result["candidates"]), 3)
            self.assertTrue(all("runtime_seconds" in item for item in result["candidates"]))
            self.assertTrue(all("convergence_warnings" in item for item in result["candidates"]))

    def test_reporting_does_not_invent_recommendation_for_smoke(self) -> None:
        """Keep smoke output diagnostic-only in framework-neutral tables."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s5_results.json"
            pq.write_table(_frame(), cache)
            result = run_s5_smoke(cache, artifact, CONFIG, max_per_class=2)
            tables = build_s5_report_tables(result)
            self.assertEqual(tables.selection_status, "DIAGNOSTIC_ONLY")
            self.assertIsNone(tables.recommended_candidate)
            self.assertEqual(tables.candidate_summary.height, 3)
            self.assertEqual(tables.reference_parity.height, 4)
            self.assertEqual(
                tables.reference_parity.columns,
                [
                    "metric",
                    "reference",
                    "actual",
                    "delta",
                    "tolerance",
                    "passed",
                    "status",
                ],
            )
            self.assertFalse(tables.reference_parity["passed"].any())
            self.assertEqual(
                tables.reference_parity["status"].to_list(),
                ["NOT_CHECKED_SMOKE"] * 4,
            )
            self.assertEqual(tables.reference_parity["reference"].null_count(), 4)
            self.assertEqual(tables.reference_parity["actual"].null_count(), 4)
            self.assertEqual(tables.deferred_candidates.height, 1)

    def test_reference_check_applies_frozen_tolerance(self) -> None:
        """Block parity when any required metric exceeds the tolerance."""

        config = load_s5_config(CONFIG)
        result = {
            "name": "sgd_log_loss_balanced_reference",
            "metrics": {
                "macro_f1": 0.7,
                "per_class": {
                    CRITICAL_CLASS: {
                        "precision": 0.2,
                        "recall": 0.3,
                        "f1": 0.24,
                    }
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "s4_results.json"
            reference.write_text(json.dumps({
                "candidates": [{
                    "name": "word_balanced_reference",
                    "metrics": {
                        "macro_f1": 0.7,
                        CRITICAL_CLASS: {
                            "precision": 0.2,
                            "recall": 0.3,
                            "f1": 0.240002,
                        },
                    },
                }],
            }), encoding="utf-8")
            checked = _check_reference([result], reference, config, smoke=False)
        self.assertEqual(checked["status"], "FAILED")
        self.assertFalse(checked["passed"])
        self.assertGreater(abs(checked["deltas"]["critical_f1"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
