"""Targeted tests for the S6 calibrated classical benchmark."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from consumer_complaint_intelligence.s6 import (
    CRITICAL_CLASS,
    MODELED_FAMILIES,
    S6GateConfig,
    _indices_for_scope,
    _margin_predictions,
    _scores_in_family_order,
    load_s6_config,
    run_s6_smoke,
    search_thresholds_exact,
    validate_scientific_cache,
)
from consumer_complaint_intelligence.s6_reporting import build_s6_report_tables


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "s6_calibrated_classical.json"


def _frame() -> pa.Table:
    """Build a development-only cache covering all S6 date scopes."""

    rows = []
    identifier = 1
    dates = {
        "inner_fit": ("train", "2024-01-01"),
        "inner_calibration": ("train", "2024-05-15"),
        "outer": ("validation", "2024-08-15"),
    }
    for label in MODELED_FAMILIES:
        for scope, (partition, received_date) in dates.items():
            for repeat in range(3):
                rows.append({
                    "Complaint ID": identifier,
                    "received_date": received_date,
                    "product_family": label,
                    "normalized_group_hash": f"hash-{identifier}",
                    "normalized_length": identifier,
                    "partition_name": partition,
                    "narrative": f"complaint {label} scope {scope} token {repeat}",
                })
                identifier += 1
    return pa.Table.from_pylist(rows)


class S6Tests(unittest.TestCase):
    """Verify S6 boundaries, thresholding, smoke, and reporting contracts."""

    def test_config_is_frozen_with_five_candidates(self) -> None:
        """Load the exact five candidates and frozen representation."""

        config = load_s6_config(CONFIG)
        self.assertEqual(config.status, "FROZEN_FOR_S6_DEVELOPMENT")
        self.assertEqual(config.approved_on, "2026-08-16")
        self.assertEqual(len(config.candidates), 5)
        self.assertEqual(config.representation.max_features, 40000)

    def test_cache_rejects_sealed_partition(self) -> None:
        """Reject sealed rows before the S6 model-selection flow."""

        table = _frame()
        values = [value.as_py() for value in table["partition_name"]]
        values[0] = "test"
        table = table.set_column(5, "partition_name", pa.array(values))
        with self.assertRaises(ValueError):
            validate_scientific_cache(table)

    def test_inner_date_boundaries_are_disjoint(self) -> None:
        """Keep fit, calibration, and outer dates in their approved scopes."""

        table = _frame()
        fit = _indices_for_scope(
            table, "train", "2023-08-01", "2024-04-30", None, 42
        )
        calibration = _indices_for_scope(
            table, "train", "2024-05-01", "2024-06-30", None, 42
        )
        outer = _indices_for_scope(
            table, "validation", "2024-07-01", "2024-12-31", None, 42
        )
        self.assertTrue(set(fit).isdisjoint(calibration))
        self.assertTrue(set(calibration).isdisjoint(outer))
        self.assertEqual(len(fit), 27)
        self.assertEqual(len(calibration), 27)
        self.assertEqual(len(outer), 27)

    def test_score_columns_are_reordered_by_estimator_classes(self) -> None:
        """Ensure the critical column is found by class label, not position."""

        class FakeEstimator:
            """Expose classes in reverse order for the reordering assertion."""

            classes_ = np.asarray(tuple(reversed(MODELED_FAMILIES)))

        reversed_scores = np.arange(len(MODELED_FAMILIES), dtype=float).reshape(1, -1)
        reordered = _scores_in_family_order(FakeEstimator(), reversed_scores)
        expected = reversed_scores[:, ::-1]
        np.testing.assert_array_equal(reordered, expected)
        critical_position = MODELED_FAMILIES.index(CRITICAL_CLASS)
        self.assertEqual(
            reordered[0, critical_position], expected[0, critical_position]
        )

    def test_threshold_search_matches_brute_force_selection(self) -> None:
        """Compare incremental threshold selection with a direct enumeration."""

        labels = [
            MODELED_FAMILIES[index % len(MODELED_FAMILIES)]
            for index in range(18)
        ]
        scores = np.arange(18 * len(MODELED_FAMILIES), dtype=float).reshape(
            18, len(MODELED_FAMILIES)
        )
        scores[:, MODELED_FAMILIES.index(CRITICAL_CLASS)] -= 20.0
        gates = S6GateConfig(0.69, 0.2715, 0.2)
        result = search_thresholds_exact(labels, scores, gates)
        margins = _margin_predictions(scores, 0.0)[1]
        thresholds = [
            float(np.nextafter(np.max(margins), np.inf)),
            *[float(value) for value in np.unique(margins)],
        ]
        brute = []
        for threshold in thresholds:
            predictions, _ = _margin_predictions(scores, threshold)
            matrix = np.zeros((len(MODELED_FAMILIES), len(MODELED_FAMILIES)), dtype=int)
            positions = {label: index for index, label in enumerate(MODELED_FAMILIES)}
            for actual, predicted in zip(labels, predictions):
                matrix[positions[actual], positions[str(predicted)]] += 1
            support = matrix.sum(axis=1)
            predicted_count = matrix.sum(axis=0)
            true_positive = np.diag(matrix)
            precision = np.divide(
                true_positive,
                predicted_count,
                out=np.zeros(len(MODELED_FAMILIES)),
                where=predicted_count != 0,
            )
            recall = np.divide(
                true_positive,
                support,
                out=np.zeros(len(MODELED_FAMILIES)),
                where=support != 0,
            )
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros(len(MODELED_FAMILIES)),
                where=(precision + recall) != 0,
            )
            critical = MODELED_FAMILIES.index(CRITICAL_CLASS)
            checks = (
                float(f1.mean()) >= 0.69,
                float(f1[critical]) >= 0.2715,
                float(precision[critical]) >= 0.2,
            )
            brute.append((
                int(all(checks)),
                int(sum(checks)),
                float(f1[critical]),
                float(f1.mean()),
                float(precision[critical]),
                -abs(threshold),
                -threshold,
                threshold,
            ))
        expected = max(brute)
        selected = result["selected"]
        selected_metrics = selected["metrics"]
        selected_critical = selected_metrics["per_class"][CRITICAL_CLASS]
        actual = (
            int(selected["gates"]["eligible"]),
            selected["gates"]["gate_count"],
            selected_critical["f1"],
            selected_metrics["macro_f1"],
            selected_critical["precision"],
            -abs(selected["threshold"]),
            -selected["threshold"],
            selected["threshold"],
        )
        self.assertEqual(actual, expected)
        self.assertTrue(np.isfinite(result["selected"]["threshold"]))

    def test_smoke_uses_one_vectorizer_and_one_outer_candidate(self) -> None:
        """Run diagnostic smoke with five candidates and one outer call."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s6_results.json"
            pq.write_table(_frame(), cache)
            with patch(
                "consumer_complaint_intelligence.s6._evaluate_outer",
                wraps=__import__(
                    "consumer_complaint_intelligence.s6",
                    fromlist=["_evaluate_outer"],
                )._evaluate_outer,
            ) as evaluate_outer:
                result = run_s6_smoke(cache, artifact, CONFIG, max_per_class=2)
            self.assertTrue(result["complete"])
            self.assertEqual(result["selection_status"], "DIAGNOSTIC_ONLY")
            self.assertIsNone(result["recommended_candidate"])
            self.assertEqual(result["vectorizer_fit_count"], 1)
            self.assertEqual(len(result["candidates"]), 5)
            self.assertEqual(evaluate_outer.call_count, 1)
            self.assertTrue(all(
                "baseline_threshold_zero" in item["calibration"]
                for item in result["candidates"]
            ))
            tables = build_s6_report_tables(result)
            self.assertEqual(tables.calibration_summary.height, 5)
            self.assertEqual(tables.outer_summary.height, 1)
            self.assertEqual(tables.statuses.height, 1)
            self.assertEqual(
                tables.statuses.columns,
                [
                    "run_mode",
                    "selection_status",
                    "recommended_candidate",
                    "diagnostic_focus",
                    "outer_evaluated_candidate",
                ],
            )
            self.assertIsNone(tables.recommended_candidate)

    def test_smoke_artifact_is_cached(self) -> None:
        """Reuse a complete smoke artifact with the same signature."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s6_results.json"
            pq.write_table(_frame(), cache)
            first = run_s6_smoke(cache, artifact, CONFIG, max_per_class=2)
            with patch("consumer_complaint_intelligence.s6._fit_scores") as fit:
                second = run_s6_smoke(cache, artifact, CONFIG, max_per_class=2)
            self.assertEqual(first["signature"], second["signature"])
            fit.assert_not_called()

    def test_resume_reuses_prefix_and_refits_only_best_completed(self) -> None:
        """Resume after a third-candidate failure without duplicate work."""

        quality = {
            "ridge_balanced": 1.0,
            "logistic_regression_saga_balanced": 5.0,
            "linear_svc_c_0_3_balanced": 2.0,
            "linear_svc_c_1_balanced": 3.0,
            "linear_svc_c_3_balanced": 4.0,
        }
        first_calls: list[str] = []
        resume_calls: list[str] = []

        class FakeEstimator:
            """Provide deterministic scores without fitting a real estimator."""

            def __init__(self, score: float) -> None:
                self.score = score
                self.classes_ = np.asarray(MODELED_FAMILIES)

            def decision_function(self, matrix: object) -> np.ndarray:
                """Return one deterministic score row per transformed document."""

                rows = getattr(matrix, "shape")[0]
                values = np.zeros(len(MODELED_FAMILIES), dtype=float)
                values[0] = self.score
                return np.tile(values, (rows, 1))

        def fake_fit_first(
            candidate: object, *args: object
        ) -> tuple[object, np.ndarray, list]:
            """Fail on the third candidate after two persisted results."""

            name = candidate.name
            first_calls.append(name)
            if len(first_calls) == 3:
                raise RuntimeError("synthetic third-candidate failure")
            score = quality[name]
            matrix = np.zeros((args[1].shape[0], len(MODELED_FAMILIES)))
            matrix[:, 0] = score
            return FakeEstimator(score), matrix, []

        def fake_fit_resume(
            candidate: object, *args: object
        ) -> tuple[object, np.ndarray, list]:
            """Refit the stored winner and fit only the pending suffix."""

            name = candidate.name
            resume_calls.append(name)
            score = quality[name]
            matrix = np.zeros((args[1].shape[0], len(MODELED_FAMILIES)))
            matrix[:, 0] = score
            return FakeEstimator(score), matrix, []

        def fake_search(labels: object, scores: np.ndarray, gates: object) -> dict:
            """Return compact deterministic calibration evidence per candidate."""

            score = float(scores[0, 0])
            metrics = {
                "macro_f1": score,
                "weighted_f1": score,
                "balanced_accuracy": score,
                "per_class": {
                    CRITICAL_CLASS: {
                        "precision": score,
                        "recall": score,
                        "f1": score,
                        "support": len(labels),
                    }
                },
                "row_count": len(labels),
                "confusion_matrix": [],
            }
            selected = {
                "threshold": 0.0,
                "metrics": metrics,
                "gates": {
                    "eligible": False,
                    "gate_count": 0,
                },
            }
            return {
                "baseline_threshold_zero": selected,
                "selected": selected,
                "threshold_count": 1,
            }

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "s6_results.json"
            pq.write_table(_frame(), cache)
            with patch(
                "consumer_complaint_intelligence.s6._fit_scores",
                side_effect=fake_fit_first,
            ), patch(
                "consumer_complaint_intelligence.s6.search_thresholds_exact",
                side_effect=fake_search,
            ):
                with self.assertRaises(RuntimeError):
                    run_s6_smoke(cache, artifact, CONFIG, max_per_class=2)
            interrupted = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(len(interrupted["candidates"]), 2)
            self.assertEqual(interrupted["execution_attempts"], 1)
            self.assertEqual(interrupted["vectorizer_fit_count"], 1)

            with patch(
                "consumer_complaint_intelligence.s6._fit_scores",
                side_effect=fake_fit_resume,
            ), patch(
                "consumer_complaint_intelligence.s6.search_thresholds_exact",
                side_effect=fake_search,
            ) as search, patch(
                "consumer_complaint_intelligence.s6._evaluate_outer",
                wraps=__import__(
                    "consumer_complaint_intelligence.s6",
                    fromlist=["_evaluate_outer"],
                )._evaluate_outer,
            ) as evaluate_outer:
                result = run_s6_smoke(cache, artifact, CONFIG, max_per_class=2)

        self.assertEqual(first_calls[:2], [
            "ridge_balanced",
            "logistic_regression_saga_balanced",
        ])
        self.assertEqual(resume_calls, [
            "logistic_regression_saga_balanced",
            "linear_svc_c_0_3_balanced",
            "linear_svc_c_1_balanced",
            "linear_svc_c_3_balanced",
        ])
        self.assertEqual(result["execution_attempts"], 2)
        self.assertEqual(result["vectorizer_fit_count"], 2)
        self.assertEqual(len(result["candidates"]), 5)
        self.assertEqual(
            len({candidate["name"] for candidate in result["candidates"]}), 5
        )
        self.assertEqual(evaluate_outer.call_count, 1)
        self.assertEqual(search.call_count, 3)


if __name__ == "__main__":
    unittest.main()
