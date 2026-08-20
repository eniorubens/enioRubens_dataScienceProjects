"""Synthetic-only tests for the S8 confirmatory boundary."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from consumer_complaint_intelligence import s8
from consumer_complaint_intelligence.s8 import MODELED_FAMILIES


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "s8_confirmatory_protocol.json"


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
            "received_date": "2025-01-02",
            "product_family": MODELED_FAMILIES[1],
            "normalized_group_hash": "clean",
            "normalized_length": 11,
        },
        {
            "Complaint ID": 21,
            "received_date": "2025-01-03",
            "product_family": MODELED_FAMILIES[1],
            "normalized_group_hash": "clean",
            "normalized_length": 11,
        },
        {
            "Complaint ID": 30,
            "received_date": "2025-01-04",
            "product_family": MODELED_FAMILIES[2],
            "normalized_group_hash": "ambiguous",
            "normalized_length": 12,
        },
        {
            "Complaint ID": 31,
            "received_date": "2025-01-05",
            "product_family": MODELED_FAMILIES[3],
            "normalized_group_hash": "ambiguous",
            "normalized_length": 12,
        },
        {
            "Complaint ID": 40,
            "received_date": "2025-01-06",
            "product_family": "debt_credit_management",
            "normalized_group_hash": "critical",
            "normalized_length": 13,
        },
    ]
    return pa.Table.from_pylist(rows)


class S8SyntheticTests(unittest.TestCase):
    """Verify S8 contracts without opening the real test partition."""

    def test_config_freezes_hashes_and_scientific_contract(self) -> None:
        """Require the exact S7, source, index, token, and gate metadata."""

        config = s8.load_s8_config(CONFIG)
        self.assertEqual(s8.S8_CODE_SCHEMA, "s8-runtime-v2")
        self.assertEqual(config.payload["status"], "FROZEN_FOR_CONFIRMATORY_TEST")
        self.assertEqual(config.source["raw_size_bytes"], 821595288)
        self.assertEqual(
            config.source["index_sha256"],
            "EDBE3C38225DA1B380E5651436FF9ABEE6591BE14A1390A988FC24B2F7D8F1A9",
        )
        self.assertEqual(config.payload["access"]["unlock_sha256"], s8.UNLOCK_SHA256)
        self.assertEqual(config.payload["s2_evidence"]["test_all_text"], 695184)
        self.assertEqual(config.payload["s2_evidence"]["novel_text"], 566040)

    @unittest.skipUnless(
        (ROOT / "temp" / "s2" / "modeling_index.parquet").exists(),
        "requires the undistributed S2 modeling index",
    )
    def test_metadata_validation_does_not_hash_raw_by_default(self) -> None:
        """Validate S7 and S2 hashes without accessing raw source metadata."""

        config = s8.load_s8_config(CONFIG)
        with patch.object(s8, "_file_signature",
                          wraps=s8._file_signature) as signature:
            result = s8.validate_frozen_metadata(config, ROOT)
        names = [call.args[0].name for call in signature.call_args_list]
        self.assertNotIn("complaints.parquet", names)
        self.assertEqual(result["s2"]["candidate_status"], "PASS")

    @unittest.skipUnless(
        (ROOT / "temp" / "s2" / "modeling_index.parquet").exists(),
        "requires the undistributed S2 modeling index",
    )
    def test_token_guard_proves_zero_duckdb_access_without_token(self) -> None:
        """Reject full mode before any raw or index query when token is absent."""

        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "s8_results.json"
            manifest_path = Path(temporary) / "s8_manifest.json"
            with patch.dict(os.environ, {}, clear=True), patch.object(
                s8.duckdb, "connect", side_effect=AssertionError("raw accessed")
            ):
                with self.assertRaises(PermissionError):
                    s8.run_s8(
                        project_root=ROOT,
                        config_path=CONFIG,
                        result_path=result_path,
                        manifest_path=manifest_path,
                        run_mode="full",
                    )
            self.assertFalse(result_path.exists())

    def test_sealed_boundary_and_resume_contract_are_hard_guards(self) -> None:
        """Reject sealed edits and partial results with changed signatures."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["remaining_sealed"] = ["stress"]
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.json"
            changed.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                s8.load_s8_config(changed)

        config = s8.load_s8_config(CONFIG)
        signature = s8._signature(CONFIG, config.source, config.s7_freeze)
        with tempfile.TemporaryDirectory() as temporary:
            partial = Path(temporary) / "partial.json"
            partial.write_text(
                json.dumps({
                    "complete": False,
                    "signature": "changed",
                    "code_schema": s8.S8_CODE_SCHEMA,
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                s8._resume_attempts(partial, signature)

    def test_error_marker_is_atomic_and_resume_preserves_opened_at(self) -> None:
        """Persist only an ERROR marker and preserve its first timestamp."""

        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as temporary:
            directory = Path(temporary)
            result_path = directory / "s8_results.json"
            manifest_path = directory / "s8_manifest.json"
            with patch.object(
                s8,
                "validate_frozen_metadata",
                return_value={"s2": {"candidate_status": "PASS"}},
            ), patch.object(
                s8,
                "load_s7_predictor",
                return_value=object(),
            ), patch.object(
                s8,
                "_scope_counts",
                side_effect=RuntimeError("synthetic query failure"),
            ), patch.dict(
                os.environ, {s8.S8_UNLOCK_ENV: "S8_CONFIRM_TEST_2025_H1_ONCE"}
            ):
                with self.assertRaises(RuntimeError):
                    s8.run_s8(
                        project_root=ROOT,
                        config_path=CONFIG,
                        result_path=result_path,
                        manifest_path=manifest_path,
                        run_mode="full",
                    )
            error = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(error["status"], "ERROR")
            self.assertFalse(error["complete"])
            self.assertIsNone(error["primary"])
            self.assertIsNone(error["operational_secondary"])
            attempts, opened_at = s8._resume_attempts(
                result_path, error["signature"]
            )
            self.assertEqual(attempts, 2)
            self.assertEqual(opened_at, error["opened_at"])

    def test_s7_bundle_invariants_are_revalidated(self) -> None:
        """Load only the already-frozen S7 bundle before any test access."""

        predictor = __import__(
            "consumer_complaint_intelligence.s7",
            fromlist=["load_s7_predictor"],
        ).load_s7_predictor(
            ROOT / "artifacts" / "s7" / "consumer_complaint_classifier_s7.joblib",
            ROOT / "config" / "s7_results.json",
            ROOT / "temp" / "s7" / "s7_results.json",
        )
        self.assertEqual(predictor.input_language, "en-US")
        self.assertEqual(predictor._bundle.classes, tuple(MODELED_FAMILIES))

    def test_result_and_manifest_share_publication_state(self) -> None:
        """Require result and manifest to be absent or published together."""

        manifest = ROOT / "config" / "s8_results.json"
        result = ROOT / "temp" / "s8" / "s8_results.json"
        self.assertEqual(manifest.exists(), result.exists())

    def test_scope_query_enforces_novelty_ambiguity_and_minimum_id(self) -> None:
        """Keep only clean novel groups and the smallest representative ID."""

        config = s8.load_s8_config(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "index.parquet"
            pq.write_table(_index_table(), index_path)
            connection = s8.duckdb.connect()
            try:
                counts = s8._scope_counts(connection, index_path, config)
                minimum = connection.execute(
                    'SELECT "Complaint ID" FROM s8_primary ORDER BY 1'
                ).fetchall()
            finally:
                connection.close()
        self.assertEqual(counts["novel_unique_groups"], 3)
        self.assertEqual(counts["clean_unique_groups"], 2)
        self.assertEqual(counts["ambiguous_unique_groups"], 1)
        self.assertEqual(counts["test_all_text"], 5)
        self.assertEqual(counts["all_test_unique_groups"], 3)
        self.assertEqual(counts["seen_previously_test_lines"], 0)
        self.assertEqual(counts["seen_previously_test_groups"], 0)
        self.assertEqual(counts["prior_modeled_lines"], 1)
        self.assertEqual(counts["prior_modeled_groups"], 1)
        self.assertEqual(counts["primary_representatives"], 2)
        self.assertEqual(counts["operational_lines"], 3)
        self.assertEqual([row[0] for row in minimum], [20, 40])

    def test_synthetic_raw_join_rejects_empty_narrative(self) -> None:
        """Reject an empty narrative instead of silently changing support."""

        config = s8.load_s8_config(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            index_path = directory / "index.parquet"
            raw_path = directory / "raw.parquet"
            pq.write_table(_index_table(), index_path)
            raw_rows = [
                {
                    "Complaint ID": 20,
                    "Consumer complaint narrative": "",
                },
                {
                    "Complaint ID": 40,
                    "Consumer complaint narrative": "critical complaint",
                },
            ]
            pq.write_table(pa.Table.from_pylist(raw_rows), raw_path)
            connection = s8.duckdb.connect()
            try:
                s8._scope_counts(connection, index_path, config)
                with self.assertRaises(ValueError):
                    list(s8._iter_raw_batches(
                        connection, raw_path, "s8_primary", config
                    ))
            finally:
                connection.close()

    def test_synthetic_raw_join_rejects_missing_complaint_id(self) -> None:
        """Reject a raw join that omits a scoped synthetic Complaint ID."""

        config = s8.load_s8_config(CONFIG)

        class FakePredictor:
            """Return one valid label for the single joined row."""

            def predict(self, texts, *, input_language):
                """Return a synthetic prediction batch."""

                return SimpleNamespace(predictions=[
                    SimpleNamespace(label=MODELED_FAMILIES[1])
                    for _ in texts
                ])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            index_path = directory / "index.parquet"
            raw_path = directory / "raw.parquet"
            pq.write_table(_index_table(), index_path)
            pq.write_table(pa.Table.from_pylist([{
                "Complaint ID": 20,
                "Consumer complaint narrative": "synthetic complaint",
            }]), raw_path)
            connection = s8.duckdb.connect()
            try:
                counts = s8._scope_counts(connection, index_path, config)
                _, rows = s8._score_scope(
                    connection,
                    raw_path,
                    "s8_primary",
                    FakePredictor(),
                    config,
                )
                with self.assertRaises(ValueError):
                    s8._validate_scope_row_count(
                        "s8_primary", rows, counts["primary_representatives"]
                    )
            finally:
                connection.close()

    def test_full_runner_completes_only_with_synthetic_paths(self) -> None:
        """Exercise streaming aggregation without using the project dataset."""

        config_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        config_payload["paths"].update({
            "source_path": "raw.parquet",
            "index_path": "index.parquet",
            "s2_report": "s2.json",
            "s7_config": "s7_config.json",
            "s7_manifest": "s7_manifest.json",
            "s7_result": "s7_result.json",
            "s7_bundle": "s7_bundle.joblib",
        })

        class FakePredictor:
            """Return the family embedded in each synthetic narrative."""

            def predict(self, texts, *, input_language):
                """Return perfect synthetic predictions under en-US."""

                self.last_language = input_language
                return SimpleNamespace(predictions=[
                    SimpleNamespace(label=text.split("|")[0]) for text in texts
                ])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "config.json"
            config_path.write_text(
                json.dumps(config_payload), encoding="utf-8"
            )
            index_rows = []
            raw_rows = []
            for identifier, family in enumerate(MODELED_FAMILIES, 100):
                index_rows.append({
                    "Complaint ID": identifier,
                    "received_date": "2025-01-01",
                    "product_family": family,
                    "normalized_group_hash": f"group-{identifier}",
                    "normalized_length": identifier,
                })
                raw_rows.append({
                    "Complaint ID": identifier,
                    "Consumer complaint narrative": f"{family}|synthetic",
                })
            pq.write_table(
                pa.Table.from_pylist(index_rows), directory / "index.parquet"
            )
            pq.write_table(pa.Table.from_pylist(raw_rows), directory / "raw.parquet")
            (directory / "s2.json").write_text("{}", encoding="utf-8")
            with patch.object(s8, "validate_frozen_metadata", return_value={
                "s2": {"candidate_status": "PASS"}
            }), patch.object(
                s8, "load_s7_predictor", return_value=FakePredictor()
            ), patch.dict(
                os.environ, {s8.S8_UNLOCK_ENV: "S8_CONFIRM_TEST_2025_H1_ONCE"}
            ):
                result = s8.run_s8(
                    project_root=directory,
                    config_path=config_path,
                    result_path=directory / "s8_results.json",
                    manifest_path=directory / "s8_manifest.json",
                    run_mode="full",
                )
        self.assertTrue(result["complete"])
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["primary"]["gates"]["gate_count"], 3)
        self.assertFalse(result["deploy"])

    def test_threshold_and_class_order_are_exact(self) -> None:
        """Apply the frozen critical margin with the fixed class order."""

        critical = MODELED_FAMILIES.index(s8.CRITICAL_CLASS)
        scores = [[0.0] * len(MODELED_FAMILIES) for _ in range(2)]
        scores[0][critical] = 0.3
        scores[0][0] = 0.1
        scores[1][critical] = 0.2
        scores[1][0] = 0.1
        labels = s8.apply_threshold(scores, 0.1135351095114484)
        self.assertEqual(labels[0], s8.CRITICAL_CLASS)
        self.assertEqual(labels[1], MODELED_FAMILIES[0])

    def test_gates_are_simultaneous_and_operational_is_excluded(self) -> None:
        """Require all three primary gates and ignore secondary metrics."""

        metrics = {
            "macro_f1": 0.70,
            "critical_f1": 0.28,
            "critical_precision": 0.21,
        }
        gates = s8.evaluate_gates(
            metrics,
            {"macro_f1_min": 0.69, "critical_f1_min": 0.2715,
             "critical_precision_min": 0.2},
        )
        self.assertTrue(gates["passed"])
        self.assertEqual(gates["gate_count"], 3)

    def test_bootstrap_is_deterministic_and_diagnostic(self) -> None:
        """Keep the stratified confidence interval reproducible and non-gating."""

        matrix = [[5 if row == col else 0 for col in range(9)] for row in range(9)]
        first = s8.bootstrap_confidence_intervals(matrix, replicates=20)
        second = s8.bootstrap_confidence_intervals(matrix, replicates=20)
        self.assertEqual(first, second)
        self.assertTrue(first["diagnostic_only"])

    def test_result_privacy_and_atomic_write(self) -> None:
        """Reject individual data and use the atomic JSON helper."""

        for payload in (
            {"score": [0.1]},
            {"texts": ["secret"]},
            {"Complaint ID": [1]},
        ):
            with self.assertRaises(ValueError):
                s8._check_result_privacy(payload)
        with self.assertRaises(ValueError):
            s8._check_result_privacy({"value": "allowed-key"}, ["allowed-key"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            s8._write_json_atomic(path, {"complete": True})
            self.assertEqual(json.loads(path.read_text())["complete"], True)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_cache_hit_does_not_require_token_or_dataset(self) -> None:
        """Repair a missing manifest without opening raw or index files."""

        config = s8.load_s8_config(CONFIG)
        signature = s8._signature(CONFIG, config.source, config.s7_freeze)
        matrix = __import__("numpy").eye(len(MODELED_FAMILIES), dtype=int)
        metrics = s8.metrics_from_confusion(matrix)
        gates = s8.evaluate_gates(metrics, config.gates)
        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as temporary:
            directory = Path(temporary)
            result = s8._base_result(
                signature, config, {"s2": config.payload["s2_evidence"]}
            )
            result["scope_counts"] = {"test_all_text": 9}
            result["primary"] = {
                "view": "scientific_primary",
                "metrics": metrics,
                "support_all_nine_classes": True,
                "support_by_class": {
                    label: 1 for label in MODELED_FAMILIES
                },
                "gates": gates,
                "confidence_intervals": s8.bootstrap_confidence_intervals(
                    matrix, replicates=2
                ),
            }
            result["operational_secondary"] = {
                "view": "operational_secondary",
                "metrics": metrics,
                "excluded_from_decision": True,
            }
            result["confirmed"] = True
            result["status"] = "CONFIRMED_FOR_STRESS_EVALUATION"
            result["decision"] = {
                "scientific_view": "primary",
                "gate_count": 3,
                "required_gate_count": 3,
                "status": result["status"],
                "deploy": False,
            }
            result["complete"] = True
            result_path = directory / "s8_results.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            manifest_path = directory / "manifest.json"
            with patch.object(s8.duckdb, "connect",
                              side_effect=AssertionError("dataset accessed")):
                cached = s8._cached_result(
                    result_path,
                    manifest_path,
                    signature,
                    config_path=CONFIG,
                )
            self.assertTrue(manifest_path.exists())
        self.assertEqual(cached, result)

    def test_synthetic_smoke_is_not_confirmatory_test_opening(self) -> None:
        """Keep smoke diagnostic and never persist an S8 final artifact."""

        result = s8.run_s8_smoke()
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertFalse(result["test_opened"])
        self.assertFalse(result["deploy"])


if __name__ == "__main__":
    unittest.main()
