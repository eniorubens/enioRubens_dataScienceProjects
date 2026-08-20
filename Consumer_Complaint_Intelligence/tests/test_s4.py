"""Synthetic tests for the S4 representation challenge."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from consumer_complaint_intelligence.s4 import _diagnostics
from consumer_complaint_intelligence.s4 import _metrics_from_matrix
from consumer_complaint_intelligence.s4 import load_s4_config
from consumer_complaint_intelligence.s4 import run_s4
from consumer_complaint_intelligence.s4 import sqrt_balanced_weights
from consumer_complaint_intelligence.s4 import validate_scientific_cache
from consumer_complaint_intelligence.temporal_split import MODELED_FAMILIES
from consumer_complaint_intelligence.s4_reporting import (
    build_s4_report_tables,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "s4_experiment.json"


def _frame() -> pa.Table:
    """Build a small nine-class train/validation scientific cache."""

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
        "debt_credit_management",
    ):
        for partition in ("train", "validation"):
            for repeat in range(2):
                rows.append(
                    {
                        "Complaint ID": identifier,
                        "received_date": "2024-01-01",
                        "product_family": label,
                        "normalized_group_hash": f"hash-{identifier}",
                        "normalized_length": identifier,
                        "partition_name": partition,
                        "narrative": f"{label} complaint {partition} {repeat}",
                    }
                )
                identifier += 1
    return pa.Table.from_pylist(rows)


class S4Tests(unittest.TestCase):
    """Verify S4 boundaries, weighting, diagnostics, and reporting."""

    def test_sqrt_weights_have_sample_mean_one(self) -> None:
        """Normalize square-root balanced weights to a sample mean of one."""

        labels = ["credit_reporting"] * 3 + ["mortgage"] + [
            "debt_credit_management"
        ]
        weights = sqrt_balanced_weights(labels, [
            "credit_reporting",
            "mortgage",
            "debt_credit_management",
        ])
        mean = np.mean([weights[label] for label in labels])
        self.assertAlmostEqual(float(mean), 1.0)
        self.assertGreater(weights["debt_credit_management"], weights["credit_reporting"])

    def test_config_manifest_is_frozen_and_complete(self) -> None:
        """Load the four frozen candidates and their gates."""

        config = load_s4_config(CONFIG)
        self.assertEqual(config.status, "FROZEN_FOR_S4_DEVELOPMENT")
        self.assertEqual(len(config.candidates), 4)
        self.assertEqual(config.gates.critical_f1_min, 0.2715)

    def test_scientific_cache_rejects_sealed_partition(self) -> None:
        """Reject a cache row from any sealed partition."""

        table = _frame().set_column(
            5,
            "partition_name",
            pa.array(["test"] + [
                value.as_py() for value in _frame()["partition_name"][1:]
            ]),
        )
        with self.assertRaises(ValueError):
            validate_scientific_cache(table)

    def test_diagnostics_are_exact_for_critical_class(self) -> None:
        """Report critical false negatives, false positives, and top errors."""

        matrix = np.zeros((9, 9), dtype=np.int64)
        critical = MODELED_FAMILIES.index("debt_credit_management")
        matrix[critical, 0] = 3
        matrix[critical, 1] = 2
        matrix[0, critical] = 4
        matrix[1, critical] = 1
        matrix[2, 0] = 5
        diagnostics = _diagnostics(matrix)
        self.assertEqual(diagnostics["critical_false_negatives"][0]["count"], 3)
        self.assertEqual(diagnostics["critical_false_positives"][0]["count"], 4)
        self.assertEqual(diagnostics["top_confusions"][0]["count"], 5)

    def test_metrics_use_fixed_class_order(self) -> None:
        """Calculate macro and per-class metrics from a fixed matrix."""

        matrix = np.eye(9, dtype=np.int64)
        metrics = _metrics_from_matrix(matrix)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["per_class"]["mortgage"]["support"], 1)

    def test_run_s4_fits_only_train_and_writes_cache(self) -> None:
        """Run the synthetic challenge without accessing sealed partitions."""

        frame = _frame()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s4_results.json"
            pq.write_table(frame, cache)
            with patch(
                "consumer_complaint_intelligence.s4.read_scientific_frame",
                return_value=frame,
            ) as reader:
                result = run_s4(cache, artifact, CONFIG, smoke_max_per_class=2)
            reader.assert_called_once_with(cache)
            self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
            self.assertEqual(
                result["claim_boundary"], "NO_TEST_STRESS_OR_MONITOR_ACCESS"
            )
            self.assertTrue(artifact.exists())

    def test_representation_fit_is_shared_between_weight_schemes(self) -> None:
        """Fit exactly one vectorizer for each of the two representations."""

        frame = _frame()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s4_results.json"
            pq.write_table(frame, cache)
            with patch(
                "consumer_complaint_intelligence.s4._vectorizer",
                wraps=__import__(
                    "consumer_complaint_intelligence.s4",
                    fromlist=["_vectorizer"],
                )._vectorizer,
            ) as vectorizer_factory:
                result = run_s4(cache, artifact, CONFIG, smoke_max_per_class=2)
            self.assertEqual(vectorizer_factory.call_count, 2)
            self.assertEqual(result["vectorizer_fit_count"], 2)
            self.assertEqual(len(result["representation_fit_events"]), 2)

    def test_interrupted_run_is_incomplete_and_not_a_cache_hit(self) -> None:
        """Persist an error artifact and rerun after an interrupted fit."""

        frame = _frame()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s4_results.json"
            pq.write_table(frame, cache)
            with patch(
                "consumer_complaint_intelligence.s4._evaluate",
                side_effect=RuntimeError("synthetic interruption"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    run_s4(cache, artifact, CONFIG, smoke_max_per_class=2)
            interrupted = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertFalse(interrupted["complete"])
            self.assertEqual(interrupted["status"], "ERROR")
            with patch(
                "consumer_complaint_intelligence.s4._evaluate",
                wraps=__import__(
                    "consumer_complaint_intelligence.s4",
                    fromlist=["_evaluate"],
                )._evaluate,
            ) as evaluator:
                completed = run_s4(cache, artifact, CONFIG, smoke_max_per_class=2)
            self.assertTrue(completed["complete"])
            self.assertTrue(evaluator.called)

    def test_run_s4_reuses_matching_artifact(self) -> None:
        """Reuse a matching diagnostic cache without rereading the frame."""

        frame = _frame()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s4_results.json"
            pq.write_table(frame, cache)
            with patch(
                "consumer_complaint_intelligence.s4.read_scientific_frame",
                return_value=frame,
            ) as reader:
                first = run_s4(cache, artifact, CONFIG, smoke_max_per_class=2)
                second = run_s4(cache, artifact, CONFIG, smoke_max_per_class=2)
            self.assertEqual(first, second)
            reader.assert_called_once()

    def test_reporting_returns_three_polars_tables(self) -> None:
        """Build framework-neutral tables from a minimal complete payload."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s4_results.json"
            pq.write_table(_frame(), cache)
            result = run_s4(cache, artifact, CONFIG, smoke_max_per_class=2)
            tables = build_s4_report_tables(result)
            self.assertEqual(tables.candidate_summary.height, 4)
            self.assertGreater(tables.per_class.height, 0)
            self.assertIn("diagnostic", tables.critical_confusions.columns)

    def test_reporting_keeps_no_eligible_selection_null(self) -> None:
        """Keep diagnostic focus separate from a null recommendation."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s4_results.json"
            pq.write_table(_frame(), cache)
            result = run_s4(cache, artifact, CONFIG, smoke_max_per_class=2)
            result["selection_status"] = "NO_ELIGIBLE_CHALLENGER"
            result["recommended_candidate"] = None
            for candidate in result["candidates"]:
                candidate["eligible"] = False
            tables = build_s4_report_tables(result)
            self.assertEqual(result["selection_status"], "NO_ELIGIBLE_CHALLENGER")
            self.assertIsNone(result["recommended_candidate"])
            self.assertEqual(
                tables.selection_status,
                "NO_ELIGIBLE_CHALLENGER",
            )
            self.assertIsNotNone(tables.diagnostic_focus_candidate)
            self.assertIn("recommended", tables.candidate_summary.columns)
            self.assertIn("diagnostic_focus", tables.candidate_summary.columns)
            self.assertNotIn("selected", tables.candidate_summary.columns)
            self.assertEqual(tables.candidate_summary["recommended"].sum(), 0)
            self.assertEqual(
                tables.candidate_summary["diagnostic_focus"].sum(),
                1,
            )

    def test_artifact_is_json_serializable(self) -> None:
        """Ensure cached output contains no estimator or non-JSON object."""

        frame = _frame()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s4_results.json"
            pq.write_table(frame, cache)
            run_s4(cache, artifact, CONFIG, smoke_max_per_class=2)
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertNotIn("estimator", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
