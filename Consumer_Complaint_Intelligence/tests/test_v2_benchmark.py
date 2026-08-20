"""Synthetic tests for the V2 classical benchmark runner."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from consumer_complaint_intelligence import v2_benchmark
from consumer_complaint_intelligence.s6 import CRITICAL_CLASS, MODELED_FAMILIES


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "v2_development_protocol.json"


class _FakePrediction:
    """Minimal S7 prediction test double."""

    def __init__(self, label: str) -> None:
        self.label = label


class _FakeBatch:
    """Minimal S7 batch test double."""

    def __init__(self, labels: list[str]) -> None:
        self.predictions = tuple(_FakePrediction(label) for label in labels)


class _FakePredictor:
    """Record bounded fallback prediction calls."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(self, texts, *, input_language: str):
        self.batch_sizes.append(len(texts))
        return _FakeBatch([MODELED_FAMILIES[0]] * len(texts))


def _rows() -> pa.Table:
    """Build a tiny three-window cache with the exact S3 schema."""

    rows = []
    identifier = 1
    windows = (
        ("2023-08-01", "train"),
        ("2024-07-01", "validation"),
        ("2024-10-01", "validation"),
    )
    for window_index, (received, partition) in enumerate(windows):
        for label_index, label in enumerate(MODELED_FAMILIES):
            rows.append(
                {
                    "Complaint ID": identifier,
                    "received_date": received,
                    "product_family": label,
                    "normalized_group_hash": f"group-{window_index}-{label_index}",
                    "normalized_length": 10,
                    "partition_name": partition,
                    "narrative": f"{label} complaint token {window_index}",
                }
            )
            identifier += 1
    return pa.Table.from_pylist(rows)


def _fake_candidate(
    candidate_id: str,
    *,
    passed_margins: bool,
    effective_overrides: int,
    critical_f1_vs_fallback: float,
    critical_f1: float = 0.5,
    macro_f1: float = 0.75,
    critical_precision: float = 0.4,
    runtime_seconds: float = 1.0,
) -> dict:
    """Build one minimal fake candidate for direct `_select_candidate` tests."""

    return {
        "candidate_id": candidate_id,
        "runtime_seconds": runtime_seconds,
        "outer": {
            "metrics": {
                "critical_f1": critical_f1,
                "macro_f1": macro_f1,
                "critical_precision": critical_precision,
            },
            "gates": {},
            "safety": {
                "passed": passed_margins,
                "gate_count": 3 if passed_margins else 2,
            },
            "override_decisions": max(effective_overrides, 0),
            "effective_overrides": effective_overrides,
            "critical_f1_vs_fallback": critical_f1_vs_fallback,
        },
    }


class V2BenchmarkTests(unittest.TestCase):
    """Verify catalog, OOF determinism, privacy, and artifact guards."""

    def test_modes_and_partition_boundary(self) -> None:
        """Expose disabled and smoke modes and reject sealed cache rows."""

        self.assertEqual(v2_benchmark.run_v2_benchmark()["status"], "DISABLED")
        with patch.object(
            v2_benchmark, "_run_candidates", side_effect=AssertionError
        ):
            result = v2_benchmark.run_v2_benchmark("disabled")
        self.assertEqual(result["status"], "DISABLED")
        with self.assertRaises(ValueError):
            v2_benchmark.run_v2_benchmark("sealed")

    def test_catalog_is_exactly_30_and_unique(self) -> None:
        """Keep the two representations and frozen balance grid complete."""

        catalog = v2_benchmark.candidate_catalog()
        self.assertEqual(len(catalog), 30)
        self.assertEqual(len({item["candidate_id"] for item in catalog}), 30)
        self.assertEqual(
            {item["representation"] for item in catalog},
            set(v2_benchmark.REPRESENTATIONS),
        )
        self.assertEqual(
            sum(item["balance_strategy"] == "hard_negative" for item in catalog),
            6,
        )

    def test_hard_negative_oof_is_deterministic_and_keeps_positives(self) -> None:
        """Use the same three-fold OOF pool on repeated calls."""

        texts = [f"critical complaint {i}" for i in range(6)]
        labels = [CRITICAL_CLASS] * 6
        for i in range(120):
            texts.append(f"ordinary complaint topic {i % 10} marker {i}")
            labels.append(MODELED_FAMILIES[0])
        first = v2_benchmark.generate_hard_negative_indices(texts, labels)
        second = v2_benchmark.generate_hard_negative_indices(texts, labels)
        self.assertEqual(first, second)
        self.assertTrue(set(range(6)).issubset(first))
        self.assertEqual(len(first), 6 + 60 + 30)
        self.assertLess(len(first), len(texts))

    def test_complete_stale_marker_is_rejected(self) -> None:
        """Never accept a complete artifact under a changed run signature."""

        protocol = v2_benchmark.load_v2_protocol(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            payload = v2_benchmark._base_result(
                "old-signature", protocol, diagnostic_only=False
            )
            payload["complete"] = True
            v2_benchmark._write_json_atomic(path, payload)
            with self.assertRaises(ValueError):
                v2_benchmark._resume_marker(
                    path, "new-signature", protocol
                )

    def test_prediction_batches_are_bounded(self) -> None:
        """Send fallback calls in the configured bounded batch size."""

        predictor = _FakePredictor()
        scope = v2_benchmark._Scope(
            tuple(f"text {i}" for i in range(7)),
            tuple(MODELED_FAMILIES[0] for _ in range(7)),
            tuple((str(i), 1) for i in range(7)),
        )
        labels = v2_benchmark._fallback_labels(predictor, scope, 3)
        self.assertEqual(len(labels), 7)
        self.assertEqual(predictor.batch_sizes, [3, 3, 1])

    def test_smoke_is_diagnostic_and_private(self) -> None:
        """Run all candidates synthetically without producing project files."""

        result = v2_benchmark.run_v2_benchmark_smoke()
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["candidates"]), 30)
        serialized = json.dumps(result).lower()
        for token in ("complaint marker", "decision_function", "indices", "models"):
            self.assertNotIn(token, serialized)

    def test_full_cache_rejects_sealed_rows_before_predictor(self) -> None:
        """Reject a cache partition boundary before loading the fallback."""

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scientific.parquet"
            table = _rows()
            values = table["partition_name"].to_pylist()
            values[-1] = "test"
            table = table.set_column(
                table.schema.get_field_index("partition_name"),
                "partition_name",
                pa.array(values),
            )
            pq.write_table(table, path)
            with self.assertRaises(ValueError):
                v2_benchmark._read_development_cache(
                    path, v2_benchmark.load_v2_protocol(CONFIG), 4
                )

    def test_manifest_tamper_and_cache_republication(self) -> None:
        """Reject a changed artifact hash and validate portable manifests."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol = directory / "v2.json"
            protocol.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            cache = directory / "scientific.parquet"
            pq.write_table(_rows(), cache)
            artifact = directory / "result.json"
            artifact.write_text(
                json.dumps({"schema_version": v2_benchmark.RESULT_SCHEMA}),
                encoding="utf-8",
            )
            manifest = directory / "manifest.json"
            payload = {
                "schema_version": v2_benchmark.MANIFEST_SCHEMA,
                "complete": True,
                "diagnostic_only": False,
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                v2_benchmark.validate_v2_manifest(manifest, artifact, protocol)

    def test_no_selected_when_safety_margin_fails(self) -> None:
        """Keep selection null when every candidate misses one safety gate."""

        fake = [
            _fake_candidate(
                "candidate",
                passed_margins=False,
                effective_overrides=0,
                critical_f1_vs_fallback=0.0,
                critical_f1=0.1,
                macro_f1=0.71,
                critical_precision=0.3,
            )
        ]
        selection = v2_benchmark._select_candidate(fake)
        self.assertIsNone(selection["selected_candidate"])
        self.assertEqual(selection["margin_eligible_count"], 0)
        self.assertEqual(selection["effective_eligible_count"], 0)
        self.assertEqual(selection["fallback_beating_eligible_count"], 0)
        self.assertTrue(selection["degenerate_null"])
        self.assertEqual(
            selection["selection_blocked_reason"],
            "no_candidate_passed_safety_margins",
        )

    def test_zero_effective_overrides_candidate_is_rejected_despite_margins(
        self,
    ) -> None:
        """Reject a fallback clone even though it clears all three margins."""

        fake = [
            _fake_candidate(
                "clone",
                passed_margins=True,
                effective_overrides=0,
                critical_f1_vs_fallback=0.0,
            )
        ]
        selection = v2_benchmark._select_candidate(fake)
        self.assertIsNone(selection["selected_candidate"])
        self.assertEqual(selection["margin_eligible_count"], 1)
        self.assertEqual(selection["effective_eligible_count"], 0)
        self.assertEqual(selection["fallback_beating_eligible_count"], 0)
        self.assertTrue(selection["degenerate_null"])
        self.assertEqual(
            selection["selection_blocked_reason"],
            "all_candidates_zero_effective_overrides",
        )

    def test_degenerate_null_true_when_every_candidate_ties_fallback(
        self,
    ) -> None:
        """Flag the run as a degenerate null when no candidate ever overrides."""

        fake = [
            _fake_candidate(
                "clone_a",
                passed_margins=True,
                effective_overrides=0,
                critical_f1_vs_fallback=0.0,
            ),
            _fake_candidate(
                "clone_b",
                passed_margins=False,
                effective_overrides=0,
                critical_f1_vs_fallback=0.0,
                critical_f1=0.05,
            ),
        ]
        selection = v2_benchmark._select_candidate(fake)
        self.assertIsNone(selection["selected_candidate"])
        self.assertTrue(selection["degenerate_null"])
        self.assertEqual(
            selection["selection_blocked_reason"],
            "all_candidates_zero_effective_overrides",
        )

    def test_candidate_not_beating_fallback_baseline_is_rejected(self) -> None:
        """Reject a real override that is still no better than the fallback."""

        fake = [
            _fake_candidate(
                "worse_than_fallback",
                passed_margins=True,
                effective_overrides=5,
                critical_f1_vs_fallback=-0.01,
            )
        ]
        selection = v2_benchmark._select_candidate(fake)
        self.assertIsNone(selection["selected_candidate"])
        self.assertEqual(selection["margin_eligible_count"], 1)
        self.assertEqual(selection["effective_eligible_count"], 1)
        self.assertEqual(selection["fallback_beating_eligible_count"], 0)
        self.assertFalse(selection["degenerate_null"])
        self.assertEqual(
            selection["selection_blocked_reason"],
            "no_candidate_beat_the_fallback_baseline",
        )

    def test_healthy_candidate_with_overrides_and_fallback_beat_is_selected(
        self,
    ) -> None:
        """Select normally when a candidate truly overrides and beats fallback."""

        fake = [
            _fake_candidate(
                "clone",
                passed_margins=True,
                effective_overrides=0,
                critical_f1_vs_fallback=0.0,
                critical_f1=0.30,
            ),
            _fake_candidate(
                "winner",
                passed_margins=True,
                effective_overrides=12,
                critical_f1_vs_fallback=0.05,
                critical_f1=0.35,
            ),
        ]
        selection = v2_benchmark._select_candidate(fake)
        self.assertEqual(selection["selected_candidate"], "winner")
        self.assertEqual(selection["margin_eligible_count"], 2)
        self.assertEqual(selection["effective_eligible_count"], 1)
        self.assertEqual(selection["fallback_beating_eligible_count"], 1)
        self.assertFalse(selection["degenerate_null"])
        self.assertIsNone(selection["selection_blocked_reason"])

    def test_run_candidates_flags_the_synthetic_perfect_fallback_as_degenerate(
        self,
    ) -> None:
        """Reproduce the reported incident on the synthetic smoke fixture.

        The synthetic fallback is a perfect oracle (it equals the true
        label), so no override can ever improve on it. Every one of the 30
        real candidates must therefore report zero effective overrides, and
        the run must be flagged as a degenerate null rather than silently
        report a winner.
        """

        protocol = v2_benchmark.load_v2_protocol(CONFIG)
        scopes, fallback = v2_benchmark._synthetic_scopes()
        results, evidence, extra = v2_benchmark._run_candidates(
            scopes, fallback, protocol, batch_size=4096
        )
        self.assertEqual(len(results), 30)
        self.assertTrue(
            all(item["outer"]["effective_overrides"] == 0 for item in results)
        )
        fallback_outer_f1 = extra["fallback_baseline"]["outer_evaluation"][
            "critical_f1"
        ]
        self.assertEqual(fallback_outer_f1, 1.0)
        for item in results:
            expected_delta = item["outer"]["metrics"]["critical_f1"] - (
                fallback_outer_f1
            )
            self.assertEqual(item["outer"]["critical_f1_vs_fallback"], expected_delta)
        self.assertEqual(evidence["margin_eligible_count"], 30)
        self.assertEqual(evidence["effective_eligible_count"], 0)
        self.assertEqual(evidence["fallback_beating_eligible_count"], 0)
        self.assertIsNone(evidence["selected_candidate"])
        self.assertTrue(extra["degenerate_null"])
        self.assertEqual(
            extra["selection_blocked_reason"],
            "all_candidates_zero_effective_overrides",
        )

    def test_smoke_result_reports_degenerate_null_and_fallback_baseline(
        self,
    ) -> None:
        """Surface the synthetic degenerate null through the public schema."""

        result = v2_benchmark.run_v2_benchmark_smoke()
        self.assertIsNone(result["selected"])
        self.assertTrue(result["degenerate_null"])
        self.assertEqual(
            result["selection_blocked_reason"],
            "all_candidates_zero_effective_overrides",
        )
        self.assertEqual(
            set(result["fallback_baseline"]),
            {"inner_calibration", "outer_evaluation"},
        )
        self.assertEqual(result["hard_negative"]["margin_eligible_count"], 30)
        self.assertEqual(result["hard_negative"]["effective_eligible_count"], 0)
        self.assertEqual(
            result["hard_negative"]["fallback_beating_eligible_count"], 0
        )


if __name__ == "__main__":
    unittest.main()
