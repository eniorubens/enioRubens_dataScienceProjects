"""Targeted tests for the S7 frozen package and serving contract."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from consumer_complaint_intelligence.s6 import (
    CRITICAL_CLASS,
    MODELED_FAMILIES,
    _margin_predictions,
    search_thresholds_exact,
    validate_scientific_cache,
)
from consumer_complaint_intelligence.s7 import (
    DEFAULT_BUNDLE,
    INPUT_LANGUAGE,
    MODEL_VERSION,
    S7_CODE_SCHEMA,
    S7ModelBundle,
    S7Predictor,
    _fit_and_calibrate,
    _dump_joblib_atomic,
    _file_signature,
    _signature,
    _write_json_atomic,
    _publish_manifest,
    _scope_indices,
    load_s7_config,
    load_s7_predictor,
    run_s7_smoke,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "s7_frozen_package.json"


def _frame() -> pa.Table:
    """Build a small development-only table for S7 tests."""

    rows = []
    identifier = 1
    scopes = {
        "fit": ("train", "2024-01-01"),
        "calibration": ("validation", "2024-08-15"),
    }
    for label in MODELED_FAMILIES:
        for scope, (partition, received_date) in scopes.items():
            for repeat in range(4):
                rows.append({
                    "Complaint ID": identifier,
                    "received_date": received_date,
                    "product_family": label,
                    "normalized_group_hash": f"hash-{identifier}",
                    "normalized_length": identifier,
                    "partition_name": partition,
                    "narrative": (
                        f"complaint {label} scope {scope} token {repeat}"
                    ),
                })
                identifier += 1
    return pa.Table.from_pylist(rows)


def _fitted_bundle() -> S7ModelBundle:
    """Fit a compact valid bundle using the frozen parameters."""

    config = load_s7_config(CONFIG)
    frame = _frame()
    fit_indices = _scope_indices(frame, config.fit_scope, None, 42)
    calibration_indices = _scope_indices(frame, config.calibration_scope, None, 42)
    bundle, _ = _fit_and_calibrate(
        frame, config, fit_indices, calibration_indices
    )
    return bundle


class S7Tests(unittest.TestCase):
    """Verify S7 freezing, calibration, serialization, and serving safeguards."""

    def test_config_is_frozen_for_one_candidate(self) -> None:
        """Require the approved date, language, scopes, and exact estimator."""

        config = load_s7_config(CONFIG)
        self.assertEqual(config.status, "FROZEN_FOR_FINAL_FIT")
        self.assertEqual(config.approved_on, "2026-08-16")
        self.assertEqual(config.input_language, INPUT_LANGUAGE)
        self.assertEqual(config.estimator["C"], 0.3)
        self.assertEqual(config.fit_scope["partition"], "train")
        self.assertEqual(config.calibration_scope["partition"], "validation")
        self.assertEqual(config.scientific_cache, "temp/s3/scientific.parquet")
        self.assertEqual(config.fit_partition, "train")
        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.gates["critical_f1_min"], 0.2715)
        self.assertEqual(
            config.threshold_policy["calibration_source"], "validation_only"
        )
        self.assertGreater(config.run_defaults["batch_size"], 0)

    def test_adultered_config_is_rejected(self) -> None:
        """Reject a frozen config whose scientific contract was changed."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["random_state"] = 7
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_s7_config(path)

    def test_scope_indices_are_disjoint_and_sealed_rejected(self) -> None:
        """Keep fit and validation scopes separate and reject sealed data."""

        config = load_s7_config(CONFIG)
        frame = _frame()
        fit = _scope_indices(frame, config.fit_scope, None, 42)
        calibration = _scope_indices(frame, config.calibration_scope, None, 42)
        self.assertTrue(set(fit).isdisjoint(calibration))
        sealed = frame.set_column(
            5,
            "partition_name",
            pa.array([
                "test" if index == 0 else value.as_py()
                for index, value in enumerate(frame["partition_name"])
            ]),
        )
        with self.assertRaises(ValueError):
            validate_scientific_cache(sealed)

    def test_threshold_search_is_the_s6_exact_rule(self) -> None:
        """Use the same exact threshold implementation and margin rule as S6."""

        labels = [
            MODELED_FAMILIES[index % len(MODELED_FAMILIES)]
            for index in range(27)
        ]
        scores = np.zeros((len(labels), len(MODELED_FAMILIES)))
        critical_index = MODELED_FAMILIES.index(CRITICAL_CLASS)
        scores[:, critical_index] = np.linspace(-1.0, 1.0, len(labels))
        result = search_thresholds_exact(
            labels,
            scores,
            __import__(
                "consumer_complaint_intelligence.s6",
                fromlist=["S6GateConfig"],
            ).S6GateConfig(0.69, 0.2715, 0.2),
        )
        _, margins = _margin_predictions(scores, 0.0)
        self.assertTrue(np.isfinite(result["selected"]["threshold"]))
        self.assertEqual(result["threshold_count"], len(np.unique(margins)) + 1)

    def test_bundle_class_order_and_critical_override(self) -> None:
        """Reorder score columns and apply the critical margin threshold."""

        bundle = _fitted_bundle()
        self.assertEqual(bundle.classes, tuple(MODELED_FAMILIES))
        predictor = S7Predictor(bundle)
        result = predictor.predict(("complaint debt_credit_management",))
        self.assertEqual(
            result.predictions[0].metadata["score_kind"], "critical_margin"
        )
        self.assertEqual(
            result.predictions[0].metadata["threshold"], bundle.threshold
        )

    def test_critical_override_switches_the_predicted_class(self) -> None:
        """Prove the threshold changes both critical and non-critical outputs."""

        class FakeVectorizer:
            """Expose frozen parameters without transforming real text."""

            def get_params(self) -> dict[str, object]:
                """Return the frozen vectorizer parameters."""

                return {
                    "analyzer": "word",
                    "ngram_range": (1, 2),
                    "max_features": 40000,
                    "min_df": 2,
                    "max_df": 0.98,
                    "sublinear_tf": True,
                    "dtype": np.float32,
                }

            def transform(self, values: tuple[str, ...]) -> tuple[str, ...]:
                """Return an opaque matrix accepted by the fake estimator."""

                return values

        class FakeEstimator:
            """Return two score rows with opposite threshold outcomes."""

            classes_ = np.asarray(MODELED_FAMILIES)

            def get_params(self) -> dict[str, object]:
                """Return the frozen LinearSVC parameters."""

                return {
                    "C": 0.3,
                    "class_weight": "balanced",
                    "tol": 0.0001,
                    "max_iter": 5000,
                    "dual": "auto",
                    "random_state": 42,
                }

            def decision_function(self, values: tuple[str, ...]) -> np.ndarray:
                """Return a critical row followed by a non-critical row."""

                scores = np.zeros((len(values), len(MODELED_FAMILIES)))
                critical = MODELED_FAMILIES.index(CRITICAL_CLASS)
                scores[0, critical] = 1.0
                scores[1, 0] = 0.7
                return scores

        bundle = S7ModelBundle(
            vectorizer=FakeVectorizer(),
            estimator=FakeEstimator(),
            threshold=0.5,
            classes=tuple(MODELED_FAMILIES),
            critical_class=CRITICAL_CLASS,
            model_version=MODEL_VERSION,
            input_language=INPUT_LANGUAGE,
        )
        predictions = S7Predictor(bundle).predict(("a", "b")).predictions
        self.assertEqual(predictions[0].label, CRITICAL_CLASS)
        self.assertEqual(predictions[1].label, MODELED_FAMILIES[0])

    def test_predictor_validates_batch_and_language(self) -> None:
        """Reject empty narratives, empty batches, and non-English contracts."""

        predictor = S7Predictor(_fitted_bundle())
        with self.assertRaises(ValueError):
            predictor.predict(())
        with self.assertRaises(ValueError):
            predictor.predict(("",))
        with self.assertRaises(ValueError):
            predictor.predict(("texto",), input_language="pt-BR")

    def test_round_trip_preserves_contract_without_probability_language(self) -> None:
        """Keep the serialized predictor framework-neutral and margin-based."""

        bundle = _fitted_bundle()
        predictor = S7Predictor(bundle)
        before = predictor.predict(("complaint credit reporting",)).to_dict()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.joblib"
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            after = S7Predictor(loaded).predict(("complaint credit reporting",))
        self.assertEqual(before, after.to_dict())
        self.assertNotIn("probability", str(after.to_dict()).lower())
        self.assertNotIn("confidence", str(after.to_dict()).lower())

    def test_loader_rejects_adultered_bundle_hash(self) -> None:
        """Reject a bundle whose bytes no longer match the public manifest."""

        bundle = _fitted_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle_path = directory / "bundle.joblib"
            result_path = directory / "s7_results.json"
            manifest_path = directory / "s7_manifest.json"
            joblib.dump(bundle, bundle_path)
            result = {
                "schema_version": "s7-results-v1",
                "code_schema": S7_CODE_SCHEMA,
                "complete": True,
                "status": "packaged_for_confirmation",
                "development_only": True,
                "deploy": False,
                "confirmatory": False,
                "sealed": False,
                "sealed_partitions": ["test", "stress", "monitor"],
                "model_version": MODEL_VERSION,
                "bundle": {},
                "fit_scope": {},
                "calibration_scope": {},
                "validation_role": "FINAL_CALIBRATION_ONLY",
                "validation_independence": (
                    "NOT_INDEPENDENT_EVIDENCE_AFTER_FINAL_CALIBRATION"
                ),
                "validation_reuse_note_pt_br": "A validation foi reutilizada.",
                "validation_reuse_note_en_us": "Validation was reused.",
                "runtime_seconds": 0.1,
                "calibration": {
                    "selected": {
                        "threshold": bundle.threshold,
                        "gates": {"eligible": True},
                    }
                },
                "calibration_gate_passed": True,
                "input_language": INPUT_LANGUAGE,
            }
            bundle_digest = __import__(
                "consumer_complaint_intelligence.s7",
                fromlist=["_file_signature"],
            )._file_signature(bundle_path)
            result["bundle"] = bundle_digest
            result_path.write_text(
                __import__("json").dumps(result), encoding="utf-8"
            )
            _publish_manifest(
                result_path,
                bundle_path,
                CONFIG,
                manifest_path,
                result,
            )
            bundle_path.write_bytes(bundle_path.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                load_s7_predictor(bundle_path, manifest_path, result_path)

    def test_joblib_write_is_atomic_and_batch_size_is_validated(self) -> None:
        """Keep temporary joblib files out of the final artifact directory."""

        bundle = _fitted_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle_path = directory / "bundle.joblib"
            _dump_joblib_atomic(bundle, bundle_path)
            self.assertTrue(bundle_path.exists())
            self.assertEqual(list(directory.glob("*.tmp")), [])
            cache = directory / "scientific.parquet"
            artifact = directory / "smoke.json"
            pq.write_table(_frame(), cache)
            with self.assertRaises(ValueError):
                run_s7_smoke(cache, artifact, CONFIG, batch_size=0)

    def test_full_cache_republishes_missing_or_stale_manifest(self) -> None:
        """Reuse valid full evidence only after repairing its public manifest."""

        bundle = _fitted_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            result_path = directory / "s7_results.json"
            bundle_path = directory / "bundle.joblib"
            manifest_path = directory / "manifest.json"
            pq.write_table(_frame(), cache)
            _dump_joblib_atomic(bundle, bundle_path)
            signature = _signature(cache, CONFIG, "full", None, 4096)
            bundle_signature = _file_signature(bundle_path)
            cached = {
                "schema_version": "s7-results-v1",
                "code_schema": S7_CODE_SCHEMA,
                "complete": True,
                "signature": signature,
                "status": "packaged_for_confirmation",
                "development_only": True,
                "deploy": False,
                "confirmatory": False,
                "sealed": False,
                "sealed_partitions": ["test", "stress", "monitor"],
                "scientific_cache": "temp/s3/scientific.parquet",
                "fit_partition": "train",
                "fit_scope": {"partition": "train"},
                "calibration_scope": {"partition": "validation"},
                "model_version": MODEL_VERSION,
                "input_language": INPUT_LANGUAGE,
                "validation_role": "FINAL_CALIBRATION_ONLY",
                "validation_independence": "NOT_INDEPENDENT_EVIDENCE_AFTER_FINAL_CALIBRATION",
                "validation_reuse_note_pt_br": "A validation foi reutilizada.",
                "validation_reuse_note_en_us": "Validation was reused.",
                "runtime_seconds": 0.1,
                "calibration_gate_passed": True,
                "bundle": bundle_signature,
                "calibration": {
                    "selected": {
                        "threshold": bundle.threshold,
                        "gates": {"eligible": True},
                    }
                },
            }
            _write_json_atomic(result_path, cached)
            manifest_path.write_text('{"stale": true}', encoding="utf-8")
            with patch(
                "consumer_complaint_intelligence.s7._publish_manifest",
                wraps=__import__(
                    "consumer_complaint_intelligence.s7",
                    fromlist=["_publish_manifest"],
                )._publish_manifest,
            ) as publish:
                run_s7 = __import__(
                    "consumer_complaint_intelligence.s7",
                    fromlist=["run_s7"],
                ).run_s7
                run_s7(
                    cache,
                    result_path,
                    CONFIG,
                    bundle_path=bundle_path,
                    manifest_path=manifest_path,
                )
            self.assertEqual(publish.call_count, 1)
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["schema_version"],
                "s7-results-manifest-v1",
            )

            cached["status"] = "calibration_gate_failed"
            cached["calibration_gate_passed"] = False
            cached["calibration"]["selected"]["gates"]["eligible"] = False
            _write_json_atomic(result_path, cached)
            _publish_manifest(
                result_path,
                bundle_path,
                CONFIG,
                manifest_path,
                cached,
            )
            failed_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(failed_manifest["status"], "calibration_gate_failed")
            self.assertIn(
                "does not advance to confirmation",
                failed_manifest["interpretation_en_us"],
            )
            with self.assertRaises(ValueError):
                load_s7_predictor(bundle_path, manifest_path, result_path)

    def test_smoke_never_persists_final_bundle_or_opens_sealed(self) -> None:
        """Keep smoke diagnostic and the final bundle untouched."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "scientific.parquet"
            artifact = directory / "smoke.json"
            bundle = directory / "final.joblib"
            pq.write_table(_frame(), cache)
            result = run_s7_smoke(
                cache,
                artifact,
                CONFIG,
                max_per_class=2,
                bundle_path=bundle,
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
            self.assertFalse(bundle.exists())
            self.assertEqual(
                tuple(result["sealed_partitions"]),
                ("test", "stress", "monitor"),
            )


if __name__ == "__main__":
    unittest.main()
