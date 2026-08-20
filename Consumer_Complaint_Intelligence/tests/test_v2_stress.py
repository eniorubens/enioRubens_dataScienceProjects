"""Synthetic-only tests for the V2.1-C confirmatory stress boundary."""

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from consumer_complaint_intelligence import s8, v2_package, v2_stress
from consumer_complaint_intelligence.contracts import Prediction, PredictionBatch
from consumer_complaint_intelligence.v2_detector import (
    HARD_NEGATIVE,
    WORD_CHAR_TFIDF_ALIAS,
    build_estimator,
    build_vectorizer,
    combine_detector_with_fallback,
)
from consumer_complaint_intelligence.v2_stress import MODELED_FAMILIES


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "v2_stress_protocol.json"
CRITICAL_CLASS = v2_stress.CRITICAL_CLASS
PRIVATE_INDEX = ROOT / "temp" / "s2" / "modeling_index.parquet"


def _index_table() -> pa.Table:
    """Create a hash-only synthetic index with clean and ambiguous groups."""

    rows = [
        {
            "Complaint ID": 1,
            "received_date": "2024-01-01",
            "product_family": MODELED_FAMILIES[0],
            "normalized_group_hash": "seen",
            "normalized_length": 10,
        },
        {
            "Complaint ID": 20,
            "received_date": "2025-07-02",
            "product_family": MODELED_FAMILIES[1],
            "normalized_group_hash": "clean",
            "normalized_length": 11,
        },
        {
            "Complaint ID": 21,
            "received_date": "2025-07-03",
            "product_family": MODELED_FAMILIES[1],
            "normalized_group_hash": "clean",
            "normalized_length": 11,
        },
        {
            "Complaint ID": 30,
            "received_date": "2025-07-04",
            "product_family": MODELED_FAMILIES[2],
            "normalized_group_hash": "ambiguous",
            "normalized_length": 12,
        },
        {
            "Complaint ID": 31,
            "received_date": "2025-07-05",
            "product_family": MODELED_FAMILIES[3],
            "normalized_group_hash": "ambiguous",
            "normalized_length": 12,
        },
        {
            "Complaint ID": 40,
            "received_date": "2025-07-06",
            "product_family": CRITICAL_CLASS,
            "normalized_group_hash": "critical",
            "normalized_length": 13,
        },
    ]
    return pa.Table.from_pylist(rows)


class _S8LikeSource:
    """Expose ``.source`` with S8's field names for the equivalence test."""

    def __init__(self, start: str, end: str) -> None:
        self._source = {"test_start": start, "test_end": end}

    @property
    def source(self):
        """Return the S8-shaped scope window mapping."""

        return self._source


class _V2StressLikeSource:
    """Expose ``.source`` with V2 stress's field names for the same test."""

    def __init__(self, start: str, end: str) -> None:
        self._source = {
            "stress_start": start,
            "stress_end": end,
            "text_column": "Consumer complaint narrative",
        }

    @property
    def source(self):
        """Return the V2-stress-shaped scope window mapping."""

        return self._source


class V2StressConfigTests(unittest.TestCase):
    """Verify the frozen V2.1-C protocol loads strictly."""

    def test_config_freezes_hashes_and_scientific_contract(self) -> None:
        """Require the exact window, threshold, gate, and token metadata."""

        config = v2_stress.load_v2_stress_config(CONFIG)
        self.assertEqual(v2_stress.V2_STRESS_CODE_SCHEMA, "v2-stress-runtime-v1")
        self.assertEqual(config.payload["status"], "FROZEN_FOR_CONFIRMATORY_TEST")
        self.assertEqual(
            config.stress_scope, {"start": "2025-07-01", "end": "2025-12-31"}
        )
        self.assertEqual(config.payload["model"]["threshold"], -0.13949530151425016)
        self.assertEqual(
            config.payload["access"]["unlock_sha256"], v2_stress.UNLOCK_SHA256
        )
        self.assertEqual(config.payload["gates"]["required_gate_count"], 4)
        self.assertEqual(config.payload["gates"]["paired_gain_strict"], True)

    def test_rejects_other_window(self) -> None:
        """Reject a config whose stress window is not 2025-07-01..2025-12-31."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["stress_scope"] = {"start": "2025-01-01", "end": "2025-06-30"}
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.json"
            changed.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                v2_stress.load_v2_stress_config(changed)

    def test_rejects_source_window_drifting_from_stress_scope(self) -> None:
        """Reject a config whose source dates disagree with stress_scope."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["source"]["stress_start"] = "2025-01-01"
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.json"
            changed.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                v2_stress.load_v2_stress_config(changed)

    def test_rejects_other_partition_name(self) -> None:
        """Reject a config naming any partition other than stress."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["confirmatory_partition"] = "test"
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.json"
            changed.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                v2_stress.load_v2_stress_config(changed)

    def test_rejects_widened_sealed_boundary(self) -> None:
        """Reject a config that no longer keeps monitor sealed."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["remaining_sealed"] = ["stress", "monitor"]
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.json"
            changed.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                v2_stress.load_v2_stress_config(changed)

    def test_module_exposes_no_test_or_monitor_selection(self) -> None:
        """Prove the module's constants cannot ever select test or monitor."""

        self.assertEqual(v2_stress.STRESS_PARTITION, "stress")
        self.assertEqual(v2_stress.SEALED_PARTITIONS, ("monitor",))
        self.assertFalse(hasattr(v2_stress, "TEST_PARTITION"))
        config = v2_stress.load_v2_stress_config(CONFIG)
        self.assertNotIn("test_start", config.source)
        self.assertNotIn("test_end", config.source)


class TokenGuardTests(unittest.TestCase):
    """Verify the confirmatory unlock guard never opens without a token."""

    @unittest.skipUnless(
        PRIVATE_INDEX.exists(),
        "requires the undistributed S2 modeling index",
    )
    def test_no_token_raises_and_touches_no_duckdb(self) -> None:
        """Reject full mode before any DuckDB access when token is absent."""

        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "v2_stress_results.json"
            manifest_path = Path(temporary) / "v2_stress_manifest.json"
            with patch.dict(os.environ, {}, clear=True), patch.object(
                v2_stress.duckdb, "connect", side_effect=AssertionError("raw accessed")
            ):
                with self.assertRaises(PermissionError):
                    v2_stress.run_v2_stress(
                        project_root=ROOT,
                        config_path=CONFIG,
                        result_path=result_path,
                        manifest_path=manifest_path,
                        run_mode="full",
                    )
            self.assertFalse(result_path.exists())

    def test_wrong_token_raises_permission_error(self) -> None:
        """Reject a token whose digest does not match the frozen contract."""

        config = v2_stress.load_v2_stress_config(CONFIG)
        with patch.dict(os.environ, {v2_stress.STRESS_UNLOCK_ENV: "wrong-token"}):
            with self.assertRaises(PermissionError):
                v2_stress.require_confirmatory_unlock(config)

    def test_disabled_mode_never_reads_environment(self) -> None:
        """Keep the default disabled mode a pure no-op."""

        result = v2_stress.run_v2_stress(project_root=ROOT, run_mode="disabled")
        self.assertEqual(result["status"], "DISABLED")
        self.assertFalse(result["deploy"])


class ScopeEquivalenceTests(unittest.TestCase):
    """Prove the stress scope SQL did not drift from S8's definition."""

    def test_scope_counts_match_s8_on_same_window(self) -> None:
        """Return identical counts from v2_stress and s8 on the same window."""

        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "index.parquet"
            pq.write_table(_index_table(), index_path)

            v2c_source = _V2StressLikeSource("2025-07-01", "2025-12-31")
            s8_source = _S8LikeSource("2025-07-01", "2025-12-31")

            connection = v2_stress.duckdb.connect()
            try:
                v2c_counts = v2_stress._scope_counts(connection, index_path, v2c_source)
            finally:
                connection.close()

            connection = s8.duckdb.connect()
            try:
                s8_counts = s8._scope_counts(connection, index_path, s8_source)
            finally:
                connection.close()

        self.assertEqual(v2c_counts, s8_counts)
        self.assertEqual(v2c_counts["novel_unique_groups"], 3)
        self.assertEqual(v2c_counts["clean_unique_groups"], 2)
        self.assertEqual(v2c_counts["primary_representatives"], 2)
        self.assertEqual(v2c_counts["operational_lines"], 3)


class CombinationEquivalenceTests(unittest.TestCase):
    """Prove the two-arm split reproduces V2Predictor's frozen labels."""

    def test_combine_matches_v2_predictor_predict(self) -> None:
        """Reproduce V2Predictor.predict exactly via the raw building blocks."""

        fallback_label = "credit_reporting"
        fallback_model_version = "consumer-complaint-classifier-s7"
        texts = tuple(
            f"debt collection notice number {i} about an unpaid loan"
            for i in range(6)
        ) + tuple(
            f"mortgage escrow account statement number {i} with a fee"
            for i in range(6)
        )
        targets = (1,) * 6 + (0,) * 6

        vectorizer = build_vectorizer(WORD_CHAR_TFIDF_ALIAS)
        matrix = vectorizer.fit_transform(texts)
        estimator = build_estimator(1.0, HARD_NEGATIVE)
        estimator.fit(matrix, np.asarray(targets, dtype=np.int8))
        bundle = v2_package.V2ModelBundle(
            vectorizer=vectorizer, estimator=estimator, threshold=0.0
        )

        class _StubFallback:
            model_version = fallback_model_version
            input_language = "en-US"

            def predict(self, batch_texts, *, input_language="en-US"):
                """Return one constant fallback label per narrative."""

                return PredictionBatch(predictions=tuple(
                    Prediction(
                        label=fallback_label,
                        score=0.0,
                        model_version=fallback_model_version,
                    )
                    for _ in batch_texts
                ))

        base_predictor = v2_package.V2Predictor(bundle, _StubFallback())
        margins = base_predictor.decision_margins(list(texts))
        threshold = float(np.median(margins))
        thresholded_bundle = dataclasses.replace(bundle, threshold=threshold)
        predictor = v2_package.V2Predictor(thresholded_bundle, _StubFallback())

        batch = predictor.predict(list(texts))
        expected_labels = [item.label for item in batch.predictions]

        fallback_labels = [fallback_label for _ in texts]
        decisions = margins >= threshold
        manual_labels = list(
            combine_detector_with_fallback(decisions, fallback_labels)
        )

        self.assertEqual(manual_labels, expected_labels)
        self.assertIn(CRITICAL_CLASS, manual_labels)
        self.assertIn(fallback_label, manual_labels)


class _FakePrediction:
    """Stand in for one S7 ``Prediction`` carrying only a label."""

    def __init__(self, label: str) -> None:
        self.label = label


class _FakeS7Predictor:
    """Return the fallback label embedded in each synthetic narrative."""

    model_version = "consumer-complaint-classifier-s7"

    def predict(self, texts, *, input_language):
        """Return a synthetic prediction batch under en-US."""

        return SimpleNamespace(
            predictions=[_FakePrediction(text.split("|")[1]) for text in texts]
        )


class _FakeV2Predictor:
    """Return a high or low margin from the level embedded in each text."""

    model_version = "consumer-complaint-detector-v2"

    def decision_margins(self, texts, *, input_language):
        """Return +1.0 for HIGH-tagged texts and -1.0 otherwise."""

        return np.array(
            [1.0 if text.split("|")[2] == "HIGH" else -1.0 for text in texts]
        )


class TwoArmScoringTests(unittest.TestCase):
    """Verify the joint accumulator and its override counters."""

    def _score(self, threshold: float) -> tuple[np.ndarray, int, int, int]:
        """Score one tiny synthetic batch through both arms in one pass."""

        combos = [
            (MODELED_FAMILIES[0], MODELED_FAMILIES[1], "HIGH"),
            (MODELED_FAMILIES[2], MODELED_FAMILIES[3], "LOW"),
            (CRITICAL_CLASS, MODELED_FAMILIES[0], "LOW"),
            (CRITICAL_CLASS, CRITICAL_CLASS, "HIGH"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            index_rows = []
            raw_rows = []
            for identifier, (truth, s7_label, level) in enumerate(combos, 1):
                index_rows.append({
                    "Complaint ID": identifier,
                    "received_date": "2025-07-01",
                    "product_family": truth,
                    "normalized_group_hash": f"g{identifier}",
                    "normalized_length": identifier,
                })
                raw_rows.append({
                    "Complaint ID": identifier,
                    "Consumer complaint narrative": f"{truth}|{s7_label}|{level}",
                })
            index_path = directory / "index.parquet"
            raw_path = directory / "raw.parquet"
            pq.write_table(pa.Table.from_pylist(index_rows), index_path)
            pq.write_table(pa.Table.from_pylist(raw_rows), raw_path)
            connection = v2_stress.duckdb.connect()
            try:
                source = _V2StressLikeSource("2025-07-01", "2025-12-31")
                v2_stress._scope_counts(connection, index_path, source)
                joint, override, effective, rows = v2_stress._score_scope_two_arms(
                    connection,
                    raw_path,
                    "v2c_primary",
                    _FakeV2Predictor(),
                    _FakeS7Predictor(),
                    threshold,
                    source,
                )
            finally:
                connection.close()
        return joint, override, effective, rows

    def test_joint_shape_and_row_count(self) -> None:
        """Accumulate exactly one joint cell increment per scored row."""

        joint, _, _, rows = self._score(0.0)
        size = len(MODELED_FAMILIES)
        self.assertEqual(joint.shape, (size, size, size))
        self.assertEqual(rows, 4)
        self.assertEqual(int(joint.sum()), 4)

    def test_marginals_reproduce_both_confusion_matrices(self) -> None:
        """Marginalize the joint table into both arms' confusion matrices."""

        joint, _, _, _ = self._score(0.0)
        v2_confusion = joint.sum(axis=2)
        s7_confusion = joint.sum(axis=1)
        positions = {label: i for i, label in enumerate(MODELED_FAMILIES)}
        # Row 1: truth=fam[0], s7=fam[1], HIGH -> v2 overrides to critical.
        self.assertEqual(
            v2_confusion[positions[MODELED_FAMILIES[0]], positions[CRITICAL_CLASS]],
            1,
        )
        self.assertEqual(
            s7_confusion[
                positions[MODELED_FAMILIES[0]], positions[MODELED_FAMILIES[1]]
            ],
            1,
        )
        # Row 3: truth=critical, s7=fam[0], LOW -> v2 keeps the s7 fallback.
        self.assertEqual(
            v2_confusion[positions[CRITICAL_CLASS], positions[MODELED_FAMILIES[0]]],
            1,
        )
        self.assertEqual(int(v2_confusion.sum()), 4)
        self.assertEqual(int(s7_confusion.sum()), 4)

    def test_override_counters_effective_excludes_already_critical(self) -> None:
        """Keep effective overrides a subset that excludes already-critical rows."""

        _, override, effective, _ = self._score(0.0)
        # Two HIGH rows raise the raw override count; one of them (row 4) was
        # already predicted critical by S7, so it is not an effective flip.
        self.assertEqual(override, 2)
        self.assertEqual(effective, 1)
        self.assertLessEqual(effective, override)

    def test_higher_threshold_yields_fewer_overrides(self) -> None:
        """Raise the threshold above every margin to suppress all overrides."""

        _, override, effective, _ = self._score(10.0)
        self.assertEqual(override, 0)
        self.assertEqual(effective, 0)


class GatesTests(unittest.TestCase):
    """Verify the four simultaneous V2.1-C gates."""

    LIMITS = {
        "macro_f1_min": 0.69,
        "critical_f1_min": 0.2715,
        "critical_precision_min": 0.20,
        "paired_critical_f1_gain_min": 0.0,
    }
    PASSING_METRICS = {
        "macro_f1": 0.9,
        "critical_f1": 0.9,
        "critical_precision": 0.9,
    }

    def test_four_of_four_confirms(self) -> None:
        """Pass all four gates when every metric clears its limit."""

        gates = v2_stress.evaluate_gates(self.PASSING_METRICS, 0.1, self.LIMITS)
        self.assertTrue(gates["passed"])
        self.assertEqual(gates["passed_count"], 4)
        self.assertEqual(gates["required_gate_count"], 4)

    def test_each_single_absolute_failure_blocks_confirmation(self) -> None:
        """Fail confirmation when exactly one absolute gate misses its floor."""

        for metric in ("macro_f1", "critical_f1", "critical_precision"):
            metrics = dict(self.PASSING_METRICS)
            metrics[metric] = 0.0
            gates = v2_stress.evaluate_gates(metrics, 0.1, self.LIMITS)
            self.assertFalse(gates["passed"], metric)
            self.assertEqual(gates["passed_count"], 3, metric)

    def test_strict_paired_gate_rejects_exact_zero_gain(self) -> None:
        """Reject a paired gain of exactly zero under the strict gate."""

        gates = v2_stress.evaluate_gates(self.PASSING_METRICS, 0.0, self.LIMITS)
        self.assertFalse(gates["passed"])
        self.assertEqual(gates["passed_count"], 3)
        paired = next(
            item for item in gates["results"]
            if item["name"] == "paired_critical_f1_gain"
        )
        self.assertTrue(paired["strict"])
        self.assertFalse(paired["passed"])

    def test_tiny_positive_paired_gain_passes(self) -> None:
        """Accept a paired gain that is strictly, even minimally, positive."""

        gates = v2_stress.evaluate_gates(self.PASSING_METRICS, 1e-9, self.LIMITS)
        paired = next(
            item for item in gates["results"]
            if item["name"] == "paired_critical_f1_gain"
        )
        self.assertTrue(paired["passed"])
        self.assertTrue(gates["passed"])


class BootstrapTests(unittest.TestCase):
    """Verify the paired joint-resample bootstrap is deterministic."""

    def _joint(self) -> np.ndarray:
        """Build a small non-trivial (9, 9, 9) joint table."""

        size = len(MODELED_FAMILIES)
        joint = np.zeros((size, size, size), dtype=np.int64)
        for index in range(size):
            joint[index, index, index] = 8
        joint[0, 1, 0] = 2
        return joint

    def test_deterministic_under_fixed_seed(self) -> None:
        """Reproduce the identical interval from the identical seed."""

        joint = self._joint()
        first = v2_stress.paired_bootstrap_interval(joint, replicates=50, seed=42)
        second = v2_stress.paired_bootstrap_interval(joint, replicates=50, seed=42)
        self.assertEqual(first, second)
        self.assertTrue(first["diagnostic_only"])
        self.assertLessEqual(first["lower"], first["upper"])

    def test_rejects_invalid_parameters(self) -> None:
        """Reject a non-positive replicate count or an out-of-range level."""

        joint = self._joint()
        with self.assertRaises(ValueError):
            v2_stress.paired_bootstrap_interval(joint, replicates=0)
        with self.assertRaises(ValueError):
            v2_stress.paired_bootstrap_interval(joint, confidence_level=1.5)

    def test_per_arm_intervals_reuse_s8_bootstrap_unchanged(self) -> None:
        """Feed each arm's confusion matrix into s8's own bootstrap helper."""

        joint = self._joint()
        v2_confusion = joint.sum(axis=2)
        s7_confusion = joint.sum(axis=1)
        v2_ci = s8.bootstrap_confidence_intervals(v2_confusion, replicates=20)
        s7_ci = s8.bootstrap_confidence_intervals(s7_confusion, replicates=20)
        self.assertTrue(v2_ci["diagnostic_only"])
        self.assertTrue(s7_ci["diagnostic_only"])
        for metric in ("macro_f1", "critical_f1", "critical_precision"):
            self.assertLessEqual(
                v2_ci["intervals"][metric]["lower"],
                v2_ci["intervals"][metric]["upper"],
            )


class PrivacyTests(unittest.TestCase):
    """Verify the aggregate-only privacy boundary of the persisted result."""

    def test_rejects_forbidden_keys_at_any_depth(self) -> None:
        """Reject narrative, identifier, score, and margin keys recursively."""

        for payload in (
            {"score": [0.1]},
            {"texts": ["secret"]},
            {"Complaint ID": [1]},
            {"margin": 0.05},
            {"margins": [0.05, -0.1]},
            {"a": {"b": {"c": {"individual_score": 1}}}},
            {"a": [{"narrative": "secret text"}]},
        ):
            with self.assertRaises(ValueError):
                v2_stress._check_result_privacy(payload)

    def test_allows_stress_count_evidence_keys(self) -> None:
        """Keep aggregate count keys that merely contain the word text."""

        payload = {
            "s2_evidence": {
                "evidence": {
                    "stress_all_text": 100,
                    "stress_novel_text": 50,
                    "test_all_text": 10,
                    "novel_text": 5,
                }
            }
        }
        v2_stress._check_result_privacy(payload)

    def test_rejects_input_text_leaking_through_a_string_value(self) -> None:
        """Reject a persisted string containing forbidden input text."""

        with self.assertRaises(ValueError):
            v2_stress._check_result_privacy(
                {"note": "contains secret-narrative-token"},
                ["secret-narrative-token"],
            )

    def test_atomic_write_leaves_no_temp_file(self) -> None:
        """Use the atomic JSON helper and clean up its temporary file."""

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            v2_stress._write_json_atomic(path, {"complete": True})
            self.assertEqual(json.loads(path.read_text())["complete"], True)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])


def _synthetic_complete_result(config: v2_stress.V2StressConfig) -> dict:
    """Build a complete aggregate result without dataset access."""

    signature = v2_stress._signature(
        CONFIG, config.source, config.v2_freeze, config.s7_freeze
    )
    size = len(MODELED_FAMILIES)
    joint = np.zeros((size, size, size), dtype=np.int64)
    for index in range(size):
        joint[index, index, index] = 5
    result = v2_stress._base_result(
        signature,
        config,
        {"s2": config.payload["s2_evidence"], "v2": {}, "s7": {}},
    )
    result["scope_counts"] = {"test_all_text": 45}
    result["primary"] = v2_stress._view_block("scientific", joint, 45, 3, 2)
    result["operational_secondary"] = v2_stress._view_block(
        "operational", joint, 45, 3, 2
    )
    result["gates"] = v2_stress.evaluate_gates(
        result["primary"]["arms"]["v2_combined"]["metrics"],
        result["primary"]["paired"]["critical_f1_gain"],
        config.gates,
    )
    result["bootstrap"] = {
        "replicates": 20,
        "seed": 42,
        "confidence_level": 0.95,
        "diagnostic_only": True,
        "v2_combined": {"macro_f1": [0.0, 1.0]},
        "s7_fallback_alone": {"macro_f1": [0.0, 1.0]},
        "paired_critical_f1_gain": [0.0, 0.0],
    }
    result["expectation"] = dict(config.payload["expectation"])
    result["expectation"]["observed_paired_gain"] = 0.0
    result["expectation"]["agrees_in_sign"] = False
    result["confirmed"] = bool(result["gates"]["passed"])
    result["status"] = "CONFIRMED" if result["confirmed"] else "NOT_CONFIRMED"
    result["complete"] = True
    return result


class ManifestTests(unittest.TestCase):
    """Verify result/manifest round-trips and the pre-full publication state."""

    def test_real_result_and_manifest_share_publication_state(self) -> None:
        """Require the real result and manifest to be absent or present together."""

        manifest = ROOT / "config" / "v2_stress_results.json"
        result = ROOT / "temp" / "v2" / "v2_stress_results.json"
        self.assertEqual(manifest.exists(), result.exists())

    def test_synthetic_round_trip_and_hash_tamper_detection(self) -> None:
        """Publish a synthetic manifest, validate it, then detect tampering."""

        config = v2_stress.load_v2_stress_config(CONFIG)
        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as temporary:
            directory = Path(temporary)
            result_path = directory / "v2_stress_results.json"
            manifest_path = directory / "v2_stress_manifest.json"
            result = _synthetic_complete_result(config)
            result_path.write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8"
            )
            v2_stress._publish_manifest(
                result_path, manifest_path, CONFIG, config, result
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            v2_stress.validate_v2_stress_manifest(manifest, result_path, CONFIG)
            self.assertFalse(manifest["deploy"])
            self.assertEqual(manifest["stage"], "V2.1-C")

            tampered = json.loads(json.dumps(manifest))
            tampered["result"]["sha256"] = "tampered"
            with self.assertRaises(ValueError):
                v2_stress.validate_v2_stress_manifest(tampered, result_path, CONFIG)

            tampered_path = json.loads(json.dumps(manifest))
            tampered_path["protocol"]["path"] = "D:/absolute/config.json"
            with self.assertRaises(ValueError):
                v2_stress.validate_v2_stress_manifest(
                    tampered_path, result_path, CONFIG
                )

            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            repaired = v2_stress._cached_result(
                result_path, manifest_path, result["signature"], config_path=CONFIG
            )
            self.assertEqual(repaired, result)
            repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            v2_stress.validate_v2_stress_manifest(
                repaired_manifest, result_path, CONFIG
            )

    def test_incomplete_result_never_publishes_a_manifest(self) -> None:
        """Never publish a manifest while the run remains incomplete."""

        config = v2_stress.load_v2_stress_config(CONFIG)
        signature = v2_stress._signature(
            CONFIG, config.source, config.v2_freeze, config.s7_freeze
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as temporary:
            directory = Path(temporary)
            result_path = directory / "v2_stress_results.json"
            manifest_path = directory / "v2_stress_manifest.json"
            result = v2_stress._base_result(
                signature,
                config,
                {"s2": config.payload["s2_evidence"], "v2": {}, "s7": {}},
            )
            v2_stress._write_json_atomic(result_path, result)
            self.assertFalse(manifest_path.exists())
            cached = v2_stress._cached_result(
                result_path, manifest_path, signature, config_path=CONFIG
            )
            self.assertIsNone(cached)
            self.assertFalse(manifest_path.exists())


class SmokeModeTests(unittest.TestCase):
    """Verify smoke mode validates real frozen artifacts without a token."""

    @unittest.skipUnless(
        PRIVATE_INDEX.exists(),
        "requires the undistributed S2 modeling index",
    )
    def test_smoke_validates_without_touching_sealed_data_or_a_token(self) -> None:
        """Validate config, hashes, manifests, and predictors, token-free."""

        with patch.dict(os.environ, {}, clear=True):
            result = v2_stress.run_v2_stress_smoke(ROOT)
        self.assertTrue(result["complete"])
        self.assertFalse(result["stress_opened"])
        self.assertFalse(result["deploy"])
        self.assertTrue(all(
            value in (True, "PASS")
            for key, value in result["checks"].items()
            if key != "s2_candidate_status"
        ))

    @unittest.skipUnless(
        PRIVATE_INDEX.exists(),
        "requires the undistributed S2 modeling index",
    )
    def test_run_v2_stress_smoke_mode_matches_direct_call(self) -> None:
        """Route run_v2_stress(run_mode='smoke') to the same diagnostic path."""

        result = v2_stress.run_v2_stress(project_root=ROOT, run_mode="smoke")
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertFalse(result["stress_opened"])


class FullRunnerSyntheticTests(unittest.TestCase):
    """Exercise the full two-arm runner without touching real project data."""

    def _staged_config(self, directory: Path) -> Path:
        """Write a config copy pointing at synthetic files in ``directory``."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["paths"].update({
            "source_path": "raw.parquet",
            "index_path": "index.parquet",
            "s2_report": "s2.json",
            "s3_protocol": "s3_protocol.json",
            "v2_config": "v2_config.json",
            "v2_manifest": "v2_manifest.json",
            "v2_result": "v2_result.json",
            "v2_bundle": "v2_bundle.joblib",
            "s7_config": "s7_config.json",
            "s7_manifest": "s7_manifest.json",
            "s7_result": "s7_result.json",
            "s7_bundle": "s7_bundle.joblib",
        })
        config_path = directory / "config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        return config_path

    def test_full_runner_confirms_when_v2_corrects_the_critical_class(self) -> None:
        """Run end to end and confirm when V2 fixes S7's one critical miss.

        ``require_confirmatory_unlock`` is patched to a no-op instead of
        supplying a real token: the frozen protocol pins its real
        ``access.unlock_sha256`` exactly (see ``V2StressConfig.validate_shape``),
        and per ADR-014 the real plaintext deliberately never appears in this
        repository, so no test can reconstruct it. The token guard itself is
        exercised directly and unpatched in ``TokenGuardTests`` above; this
        test's purpose is the two-arm scoring pipeline, not the guard.
        """

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = self._staged_config(directory)
            index_rows = []
            raw_rows = []
            wrong_family = next(
                family for family in MODELED_FAMILIES if family != CRITICAL_CLASS
            )
            for identifier, family in enumerate(MODELED_FAMILIES, 100):
                if family == CRITICAL_CLASS:
                    s7_label, level = wrong_family, "HIGH"
                else:
                    s7_label, level = family, "LOW"
                index_rows.append({
                    "Complaint ID": identifier,
                    "received_date": "2025-07-15",
                    "product_family": family,
                    "normalized_group_hash": f"group-{identifier}",
                    "normalized_length": identifier,
                })
                raw_rows.append({
                    "Complaint ID": identifier,
                    "Consumer complaint narrative": f"{family}|{s7_label}|{level}",
                })
            pq.write_table(
                pa.Table.from_pylist(index_rows), directory / "index.parquet"
            )
            pq.write_table(
                pa.Table.from_pylist(raw_rows), directory / "raw.parquet"
            )
            with patch.object(
                v2_stress,
                "validate_frozen_metadata",
                return_value={"s2": {"candidate_status": "PASS"}, "v2": {}, "s7": {}},
            ), patch.object(
                v2_stress, "load_v2_predictor", return_value=_FakeV2Predictor()
            ), patch.object(
                v2_stress, "load_s7_predictor", return_value=_FakeS7Predictor()
            ), patch.object(
                v2_stress, "require_confirmatory_unlock", return_value=None
            ):
                result = v2_stress.run_v2_stress(
                    project_root=directory,
                    config_path=config_path,
                    result_path=directory / "v2_stress_results.json",
                    manifest_path=directory / "v2_stress_manifest.json",
                    run_mode="full",
                )
            self.assertTrue(result["complete"])
            self.assertFalse(result["deploy"])
            self.assertTrue(result["stress_opened"])
            self.assertEqual(result["gates"]["required_gate_count"], 4)
            self.assertTrue(result["confirmed"])
            self.assertEqual(result["status"], "CONFIRMED")
            primary = result["primary"]
            self.assertEqual(primary["rows"], 9)
            v2_metrics = primary["arms"]["v2_combined"]["metrics"]
            s7_metrics = primary["arms"]["s7_fallback_alone"]["metrics"]
            self.assertEqual(v2_metrics["critical_f1"], 1.0)
            self.assertEqual(s7_metrics["critical_f1"], 0.0)
            self.assertGreater(primary["paired"]["critical_f1_gain"], 0.0)
            self.assertEqual(primary["override"]["override_decisions"], 1)
            self.assertEqual(primary["override"]["effective_overrides"], 1)
            self.assertEqual(
                result["operational_secondary"]["view"], "operational"
            )
            self.assertIn("paired_critical_f1_gain", result["bootstrap"])
            self.assertTrue(result["bootstrap"]["diagnostic_only"])
            self.assertTrue(result["expectation"]["agrees_in_sign"])

            # A second run with the identical inputs must hit the cache and
            # publish the manifest without touching DuckDB again.
            with patch.object(
                v2_stress.duckdb, "connect", side_effect=AssertionError("reopened")
            ):
                cached = v2_stress.run_v2_stress(
                    project_root=directory,
                    config_path=config_path,
                    result_path=directory / "v2_stress_results.json",
                    manifest_path=directory / "v2_stress_manifest.json",
                    run_mode="full",
                )
            self.assertEqual(cached, result)
            manifest_file = directory / "v2_stress_manifest.json"
            self.assertTrue(manifest_file.exists())
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "CONFIRMED")
            self.assertFalse(manifest["deploy"])


if __name__ == "__main__":
    unittest.main()
