"""Synthetic, cache-free tests for the frozen V2 package runner.

Every test here runs without the 152 MB scientific parquet cache, without a
GPU, and without network access. Real project files are copied or replaced
by tiny staged fixtures inside temporary directories, and the two heavy
seams -- reading the development cache and loading the frozen S7 fallback --
are monkeypatched rather than widened into production injection points, so
no new code path can reach a sealed partition.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from consumer_complaint_intelligence import v2_benchmark, v2_package
from consumer_complaint_intelligence.contracts import Prediction, PredictionBatch
from consumer_complaint_intelligence.s6 import CRITICAL_CLASS, MODELED_FAMILIES
from consumer_complaint_intelligence.v2_detector import (
    HARD_NEGATIVE,
    WORD_CHAR_TFIDF_ALIAS,
    _metrics_from_confusion,
    build_estimator,
    build_vectorizer,
)
from consumer_complaint_intelligence.v2_protocol import load_v2_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "v2_frozen_package.json"
PROTOCOL_PATH = ROOT / "config" / "v2_development_protocol.json"
CANDIDATE_ID = "word_char_tfidf_union_40000_60000_c_1_hard_negative"
FALLBACK_LABEL = "credit_reporting"
FALLBACK_MODEL_VERSION = "consumer-complaint-classifier-s7"
BUNDLE_TEXTS = tuple(
    f"debt collection notice number {index} about an unpaid loan balance"
    for index in range(6)
) + tuple(
    f"mortgage escrow account statement number {index} with a late fee"
    for index in range(6)
)
BUNDLE_TARGETS = (1,) * 6 + (0,) * 6


def _load_config_payload() -> dict:
    """Read the real frozen package config JSON as a plain dictionary."""

    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> Path:
    """Write one JSON document and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _confusion(diagonal: int, off_diagonal: int) -> list[list[int]]:
    """Build one nine-class confusion matrix with a dominant diagonal."""

    size = len(MODELED_FAMILIES)
    matrix = np.full((size, size), off_diagonal, dtype=np.int64)
    np.fill_diagonal(matrix, diagonal)
    return matrix.astype(int).tolist()


def _observed_evidence() -> dict:
    """Build one canned aggregate refit result with real derived metrics."""

    calibration_matrix = _confusion(100, 1)
    outer_matrix = _confusion(120, 2)
    calibration_metrics = _metrics_from_confusion(np.asarray(calibration_matrix))
    outer_metrics = _metrics_from_confusion(np.asarray(outer_matrix))
    fallback_metrics = _metrics_from_confusion(np.asarray(_confusion(110, 3)))
    return {
        "candidate_id": CANDIDATE_ID,
        "calibration": {
            "threshold": -0.13949530151425016,
            "threshold_count": 117974,
            "metrics": calibration_metrics,
            "gates": {"passed": False, "gate_count": 2},
            "override_decisions": 57,
            "effective_overrides": 16,
        },
        "outer": {
            "metrics": outer_metrics,
            "gates": {"passed": True, "gate_count": 3},
            "safety": {"passed": True, "gate_count": 3},
            "override_decisions": 258,
            "effective_overrides": 82,
            "critical_f1_vs_fallback": 0.047233864926836844,
        },
        "hard_negative": {
            "positive_groups": v2_package.D1_POSITIVE_GROUPS,
            "hard_negative_groups": v2_package.D1_HARD_NEGATIVE_GROUPS,
            "pool_rows": v2_package.D1_POOL_ROWS,
            "pool_signature": "A" * 64,
        },
        "fallback_baseline": {
            "inner_calibration": fallback_metrics,
            "outer_evaluation": fallback_metrics,
        },
    }


def _d1_payload(observed: dict) -> dict:
    """Build one D1 classical artifact reproducing ``observed`` exactly."""

    observed = copy.deepcopy(observed)
    return {
        "schema_version": "v2-classical-benchmark-v1",
        "complete": True,
        "selected": CANDIDATE_ID,
        "hard_negative": {
            "positive_groups": observed["hard_negative"]["positive_groups"],
            "hard_negative_groups": observed["hard_negative"][
                "hard_negative_groups"
            ],
        },
        "candidates": [
            {
                "candidate_id": CANDIDATE_ID,
                "calibration": {
                    "threshold": observed["calibration"]["threshold"],
                    "metrics": observed["calibration"]["metrics"],
                    "override_decisions": observed["calibration"][
                        "override_decisions"
                    ],
                    "effective_overrides": observed["calibration"][
                        "effective_overrides"
                    ],
                },
                "outer": {
                    "metrics": observed["outer"]["metrics"],
                    "override_decisions": observed["outer"][
                        "override_decisions"
                    ],
                    "effective_overrides": observed["outer"][
                        "effective_overrides"
                    ],
                },
            }
        ],
    }


def _expected_from_observed(observed: dict) -> dict:
    """Derive the frozen config's scalar expectations from ``observed``."""

    calibration = observed["calibration"]
    outer_metrics = observed["outer"]["metrics"]
    return {
        "threshold": calibration["threshold"],
        "calibration_override_decisions": calibration["override_decisions"],
        "calibration_effective_overrides": calibration["effective_overrides"],
        "calibration_row_count": calibration["metrics"]["row_count"],
        "outer_override_decisions": observed["outer"]["override_decisions"],
        "outer_effective_overrides": observed["outer"]["effective_overrides"],
        "outer_row_count": outer_metrics["row_count"],
        "outer_critical_f1": outer_metrics["critical_f1"],
        "outer_critical_precision": outer_metrics["critical_precision"],
        "outer_critical_recall": outer_metrics["critical_recall"],
        "outer_macro_f1": outer_metrics["macro_f1"],
        "outer_critical_support": outer_metrics["critical_support"],
    }


def _stage_root(directory: Path, observed: dict, d1_payload: dict) -> Path:
    """Stage a complete, self-consistent project tree in ``directory``.

    Every pinned provenance hash in the staged frozen config is recomputed
    from the tiny fixture actually written, so ``verify_package_provenance``
    passes without any real project artifact.

    Args:
        directory: Temporary directory to populate.
        observed: The canned refit evidence the config must expect.
        d1_payload: The staged D1 classical artifact payload.

    Returns:
        The resolved staged project root.
    """

    root = directory.resolve()
    (root / "config").mkdir(parents=True, exist_ok=True)
    protocol = root / "config" / "v2_development_protocol.json"
    protocol.write_bytes(PROTOCOL_PATH.read_bytes())
    _write_json(root / "temp" / "v2" / "v2_classical_benchmark.json", d1_payload)
    _write_json(root / "config" / "v2_classical_results.json", {"stage": "V2-D1"})
    _write_json(
        root / "temp" / "v2" / "v2_transformer_challenge.json", {"stage": "V2.1-D2"}
    )
    _write_json(root / "config" / "v2_transformer_results.json", {"stage": "D2"})
    _write_json(root / "config" / "s7_results.json", {"stage": "S7"})
    _write_json(root / "temp" / "s7" / "s7_results.json", {"stage": "S7"})
    cache = root / "temp" / "s3" / "scientific.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"staged-development-cache-placeholder")
    bundle = root / "artifacts" / "s7" / "consumer_complaint_classifier_s7.joblib"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"staged-s7-bundle-placeholder")
    payload = _load_config_payload()
    payload["reproduction_gate"]["expected"] = _expected_from_observed(observed)
    payload["protocol"].update(_signature_of(protocol))
    for keys in (
        ("cache",),
        ("d1", "artifact"),
        ("d1", "manifest"),
        ("d2", "artifact"),
        ("d2", "manifest"),
        ("s7", "bundle"),
        ("s7", "manifest"),
        ("s7", "result"),
    ):
        entry = payload["provenance"]
        for key in keys:
            entry = entry[key]
        entry.update(_signature_of(root / entry["path"]))
    _write_json(root / "config" / "v2_frozen_package.json", payload)
    return root


def _signature_of(path: Path) -> dict:
    """Return the uppercase digest and byte size of one staged fixture."""

    return {
        "sha256": v2_package._sha256(path),
        "size_bytes": path.stat().st_size,
    }


class _StubFallback:
    """Stand in for the frozen S7 predictor without loading any joblib."""

    model_version = FALLBACK_MODEL_VERSION
    input_language = "en-US"

    def __init__(self, label: str = FALLBACK_LABEL) -> None:
        """Store the single multiclass label this stub always returns."""

        self._label = label

    def predict(
        self, texts, *, input_language: str = "en-US"
    ) -> PredictionBatch:
        """Return one constant modeled-family label per input narrative."""

        if input_language != "en-US":
            raise ValueError("stub fallback accepts en-US only")
        return PredictionBatch(
            predictions=tuple(
                Prediction(
                    label=self._label,
                    score=0.0,
                    model_version=FALLBACK_MODEL_VERSION,
                )
                for _ in texts
            )
        )


def _build_bundle(threshold: float = 0.0) -> v2_package.V2ModelBundle:
    """Fit one tiny but contract-exact V2 bundle for serving tests."""

    vectorizer = build_vectorizer(WORD_CHAR_TFIDF_ALIAS)
    matrix = vectorizer.fit_transform(BUNDLE_TEXTS)
    estimator = build_estimator(1.0, HARD_NEGATIVE)
    estimator.fit(matrix, np.asarray(BUNDLE_TARGETS, dtype=np.int8))
    return v2_package.V2ModelBundle(
        vectorizer=vectorizer, estimator=estimator, threshold=float(threshold)
    )


_CACHED_BUNDLE: list[v2_package.V2ModelBundle] = []


def _bundle() -> v2_package.V2ModelBundle:
    """Return an independent copy of the cached fitted test bundle."""

    if not _CACHED_BUNDLE:
        _CACHED_BUNDLE.append(_build_bundle())
    return copy.deepcopy(_CACHED_BUNDLE[0])


class PackageConfigTests(unittest.TestCase):
    """Verify the frozen package configuration loads strictly."""

    def test_frozen_config_loads_and_exposes_the_pinned_candidate(self) -> None:
        """Load the real approved contract and expose its key values."""

        config = v2_package.load_v2_package_config(CONFIG_PATH)
        self.assertEqual(config.schema_version, v2_package.V2_PACKAGE_CONFIG_SCHEMA)
        self.assertEqual(config.candidate_id, CANDIDATE_ID)
        self.assertEqual(config.status, "FROZEN_FOR_CONFIRMATION")
        self.assertEqual(config.batch_size, 4096)
        self.assertEqual(
            config.hard_negative_pool["positive_groups"],
            v2_package.D1_POSITIVE_GROUPS,
        )
        self.assertEqual(
            config.hard_negative_pool["hard_negative_groups"],
            v2_package.D1_HARD_NEGATIVE_GROUPS,
        )
        descriptor = config.descriptor()
        self.assertEqual(descriptor["representation"], WORD_CHAR_TFIDF_ALIAS)
        self.assertEqual(descriptor["balance_strategy"], HARD_NEGATIVE)
        self.assertEqual(descriptor["C"], 1.0)
        self.assertIsNone(descriptor["sampling_strategy"])

    def test_rejects_wrong_schema_version(self) -> None:
        """Reject a configuration whose schema_version has drifted."""

        payload = _load_config_payload()
        payload["schema_version"] = "v2-frozen-package-config-v2"
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_json(Path(temporary) / "config.json", payload)
            with self.assertRaises(ValueError):
                v2_package.load_v2_package_config(path)

    def test_rejects_missing_required_key(self) -> None:
        """Reject a configuration missing a required top-level block."""

        payload = _load_config_payload()
        del payload["reproduction_gate"]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_json(Path(temporary) / "config.json", payload)
            with self.assertRaises(ValueError):
                v2_package.load_v2_package_config(path)

    def test_rejects_a_widened_candidate(self) -> None:
        """Reject a configuration that swaps the D1-selected candidate."""

        payload = _load_config_payload()
        payload["candidate"]["balance_strategy"] = "weighted_full"
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_json(Path(temporary) / "config.json", payload)
            with self.assertRaises(ValueError):
                v2_package.load_v2_package_config(path)

    def test_rejects_a_bundle_publishing_mismatch(self) -> None:
        """Reject a configuration allowing a bundle on a failed gate."""

        payload = _load_config_payload()
        payload["reproduction_gate"]["publishes_bundle_on_mismatch"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_json(Path(temporary) / "config.json", payload)
            with self.assertRaises(ValueError):
                v2_package.load_v2_package_config(path)

    def test_pinned_hash_mismatch_raises(self) -> None:
        """Reject a staged tree whose pinned D1 artifact was rewritten."""

        observed = _observed_evidence()
        with tempfile.TemporaryDirectory() as temporary:
            root = _stage_root(Path(temporary), observed, _d1_payload(observed))
            config = v2_package.load_v2_package_config(project_root=root)
            self.assertIn(
                "d1_artifact",
                v2_package.verify_package_provenance(config, project_root=root),
            )
            tampered = root / "temp" / "v2" / "v2_classical_benchmark.json"
            tampered.write_text(
                tampered.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                v2_package.verify_package_provenance(config, project_root=root)


class ReproductionGateTests(unittest.TestCase):
    """Verify the exact reproduction gate accepts and blocks correctly."""

    def setUp(self) -> None:
        """Load the real frozen config and canned refit evidence."""

        self.observed = _observed_evidence()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = _stage_root(
            Path(self.temporary.name), self.observed, _d1_payload(self.observed)
        )
        self.config = v2_package.load_v2_package_config(project_root=self.root)
        self.d1_record = v2_package._read_d1_record(
            self.config, project_root=self.root
        )

    def test_identical_refit_passes_every_check(self) -> None:
        """Pass all named checks when the refit reproduces D1 exactly."""

        gate = v2_package._evaluate_reproduction_gate(
            self.observed, self.d1_record, self.config
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failed_checks"], [])
        self.assertEqual(gate["divergences"], {})
        for name in v2_package._REQUIRED_GATE_CHECKS:
            self.assertTrue(gate["checks"][name], name)
        self.assertTrue(all(gate["checks"].values()))
        self.assertGreater(gate["check_count"], len(v2_package._REQUIRED_GATE_CHECKS))

    def test_threshold_divergence_publishes_scalars(self) -> None:
        """Fail on the threshold alone and publish both scalar values."""

        record = copy.deepcopy(self.d1_record)
        record["calibration"]["threshold"] = -0.2
        gate = v2_package._evaluate_reproduction_gate(
            self.observed, record, self.config
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failed_checks"], ["calibrated_threshold"])
        divergence = gate["divergences"]["calibrated_threshold"]
        self.assertEqual(divergence["expected"], -0.2)
        self.assertEqual(
            divergence["observed"], self.observed["calibration"]["threshold"]
        )

    def test_matrix_divergence_publishes_only_a_difference_summary(self) -> None:
        """Fail on the outer matrix and never echo either matrix."""

        record = copy.deepcopy(self.d1_record)
        record["outer"]["metrics"]["confusion_matrix"][0][1] += 7
        gate = v2_package._evaluate_reproduction_gate(
            self.observed, record, self.config
        )
        self.assertFalse(gate["passed"])
        self.assertIn("outer_confusion_matrix", gate["failed_checks"])
        summary = gate["divergences"]["outer_confusion_matrix"]
        self.assertTrue(summary["comparable"])
        self.assertEqual(summary["mismatched_cells"], 1)
        self.assertEqual(summary["total_absolute_difference"], 7)
        self.assertNotIn("observed", summary)
        self.assertNotIn("expected", summary)

    def test_pool_count_divergence_is_named(self) -> None:
        """Fail on the hard-negative pool counts against the D1 record."""

        record = copy.deepcopy(self.d1_record)
        record["hard_negative"]["hard_negative_groups"] = 14189
        gate = v2_package._evaluate_reproduction_gate(
            self.observed, record, self.config
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failed_checks"], ["hard_negative_pool_counts"])

    def test_config_expectation_divergence_is_named_separately(self) -> None:
        """Fail the config cross-check when the refit misses the contract."""

        observed = copy.deepcopy(self.observed)
        observed["outer"]["effective_overrides"] = 81
        gate = v2_package._evaluate_reproduction_gate(
            observed, self.d1_record, self.config
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["failed_checks"],
            [
                "config_expected_outer_effective_overrides",
                "outer_effective_overrides",
            ],
        )


class ModelBundleTests(unittest.TestCase):
    """Verify the frozen bundle enforces its hard serving invariants."""

    def test_accepts_a_correctly_built_bundle(self) -> None:
        """Accept a bundle fitted from the frozen builders."""

        bundle = _bundle()
        bundle.validate()
        self.assertEqual(bundle.model_version, v2_package.MODEL_VERSION)
        self.assertEqual(bundle.critical_class, CRITICAL_CLASS)
        self.assertEqual(bundle.input_language, "en-US")
        self.assertEqual(bundle.schema_version, v2_package.V2_BUNDLE_SCHEMA)

    def test_rejects_a_wrong_threshold_type(self) -> None:
        """Reject a bundle whose threshold is not a real number."""

        bundle = dataclasses.replace(_bundle(), threshold="-0.139")
        with self.assertRaises(ValueError):
            bundle.validate()

    def test_rejects_a_non_finite_threshold(self) -> None:
        """Reject a bundle whose threshold is not finite."""

        bundle = dataclasses.replace(_bundle(), threshold=float("nan"))
        with self.assertRaises(ValueError):
            bundle.validate()

    def test_rejects_a_tampered_estimator_hyperparameter(self) -> None:
        """Reject a bundle whose LinearSVC regularization was changed."""

        bundle = _bundle()
        bundle.estimator.set_params(C=0.3)
        with self.assertRaises(ValueError):
            bundle.validate()

    def test_rejects_a_tampered_vectorizer_hyperparameter(self) -> None:
        """Reject a bundle whose character branch was narrowed."""

        bundle = _bundle()
        bundle.vectorizer.transformer_list[1][1].set_params(max_features=10)
        with self.assertRaises(ValueError):
            bundle.validate()

    def test_rejects_a_wrong_model_version(self) -> None:
        """Reject a bundle carrying a foreign model version."""

        bundle = dataclasses.replace(_bundle(), model_version="other")
        with self.assertRaises(ValueError):
            bundle.validate()


class PredictorTests(unittest.TestCase):
    """Verify the hierarchical override and the serving input contract."""

    def setUp(self) -> None:
        """Build one predictor whose threshold splits the test batch."""

        self.texts = list(BUNDLE_TEXTS)
        base = v2_package.V2Predictor(_bundle(), _StubFallback())
        self.margins = base.decision_margins(self.texts)
        threshold = float(np.median(self.margins))
        self.bundle = dataclasses.replace(_bundle(), threshold=threshold)
        self.predictor = v2_package.V2Predictor(self.bundle, _StubFallback())

    def test_hierarchy_overrides_only_above_the_threshold(self) -> None:
        """Emit critical above the threshold and the fallback below it."""

        batch = self.predictor.predict(self.texts)
        labels = [item.label for item in batch.predictions]
        threshold = self.predictor.threshold
        expected = [
            CRITICAL_CLASS if margin >= threshold else FALLBACK_LABEL
            for margin in self.margins
        ]
        self.assertEqual(labels, expected)
        self.assertIn(CRITICAL_CLASS, labels)
        self.assertIn(FALLBACK_LABEL, labels)

    def test_predictions_carry_the_margin_and_hierarchy_metadata(self) -> None:
        """Publish the stage-A margin as score plus hierarchy metadata."""

        batch = self.predictor.predict(self.texts)
        for prediction, margin in zip(batch.predictions, self.margins):
            self.assertAlmostEqual(prediction.score, float(margin))
            self.assertEqual(prediction.model_version, v2_package.MODEL_VERSION)
            metadata = prediction.metadata
            self.assertEqual(metadata["score_kind"], "critical_margin")
            self.assertEqual(metadata["threshold"], self.predictor.threshold)
            self.assertEqual(metadata["input_language"], "en-US")
            self.assertEqual(
                metadata["stage"], "hierarchical_critical_override"
            )
            self.assertEqual(
                metadata["fallback_model_version"], FALLBACK_MODEL_VERSION
            )

    def test_rejects_a_bare_string_batch(self) -> None:
        """Reject a single string mistaken for a batch of narratives."""

        with self.assertRaises(ValueError):
            self.predictor.predict("a single complaint narrative")

    def test_rejects_an_empty_batch(self) -> None:
        """Reject an empty batch."""

        with self.assertRaises(ValueError):
            self.predictor.predict([])

    def test_rejects_non_string_items(self) -> None:
        """Reject a batch containing a non-string item."""

        with self.assertRaises(ValueError):
            self.predictor.predict([BUNDLE_TEXTS[0], 42])

    def test_rejects_blank_narratives(self) -> None:
        """Reject a batch containing a blank narrative."""

        with self.assertRaises(ValueError):
            self.predictor.predict([BUNDLE_TEXTS[0], "   "])

    def test_rejects_a_foreign_input_language(self) -> None:
        """Reject any language other than the frozen en-US contract."""

        with self.assertRaises(ValueError):
            self.predictor.predict(self.texts, input_language="pt-BR")

    def test_rejects_a_fallback_without_predict(self) -> None:
        """Reject a stage-B fallback that is not a predictor."""

        with self.assertRaises(ValueError):
            v2_package.V2Predictor(_bundle(), object())


class RefitTests(unittest.TestCase):
    """Exercise the real refit path on tiny synthetic development scopes."""

    def test_refit_produces_aggregate_evidence_and_a_valid_bundle(self) -> None:
        """Fit, calibrate, and evaluate the pinned candidate end to end."""

        protocol = load_v2_protocol(PROTOCOL_PATH)
        config = v2_package.load_v2_package_config(CONFIG_PATH)
        scopes, fallback = v2_benchmark._synthetic_scopes()
        observed, bundle = v2_package._refit_selected_candidate(
            scopes, fallback, protocol, config, batch_size=4096
        )
        bundle.validate()
        self.assertEqual(observed["candidate_id"], CANDIDATE_ID)
        self.assertEqual(
            observed["calibration"]["threshold"], float(bundle.threshold)
        )
        for block in ("calibration", "outer"):
            self.assertIn("confusion_matrix", observed[block]["metrics"])
            self.assertIn("override_decisions", observed[block])
            self.assertIn("effective_overrides", observed[block])
        pool = observed["hard_negative"]
        self.assertEqual(
            pool["pool_rows"],
            pool["positive_groups"] + pool["hard_negative_groups"],
        )
        self.assertEqual(len(pool["pool_signature"]), 64)
        self.assertEqual(
            set(observed["fallback_baseline"]),
            {"inner_calibration", "outer_evaluation"},
        )

    def test_refit_rejects_a_non_positive_batch_size(self) -> None:
        """Reject a non-positive outer evaluation batch size."""

        protocol = load_v2_protocol(PROTOCOL_PATH)
        config = v2_package.load_v2_package_config(CONFIG_PATH)
        scopes, fallback = v2_benchmark._synthetic_scopes()
        with self.assertRaises(ValueError):
            v2_package._refit_selected_candidate(
                scopes, fallback, protocol, config, batch_size=0
            )


class _StagedRun:
    """Own one staged root plus the two monkeypatched heavy seams."""

    def __init__(self, case: unittest.TestCase, d1_mutation=None) -> None:
        """Stage a project tree and patch the cache and fallback seams."""

        self.observed = _observed_evidence()
        d1_payload = _d1_payload(self.observed)
        if d1_mutation is not None:
            d1_mutation(d1_payload)
        temporary = tempfile.TemporaryDirectory()
        case.addCleanup(temporary.cleanup)
        self.root = _stage_root(Path(temporary.name), self.observed, d1_payload)
        scopes, _ = v2_benchmark._synthetic_scopes()
        patches = (
            mock.patch.object(
                v2_package, "load_s7_predictor", lambda *_: _StubFallback()
            ),
            mock.patch.object(
                v2_package,
                "_read_development_cache",
                lambda *_args, **_kwargs: scopes,
            ),
            mock.patch.object(
                v2_package,
                "_refit_selected_candidate",
                lambda *_args, **_kwargs: (
                    copy.deepcopy(self.observed),
                    dataclasses.replace(
                        _bundle(),
                        threshold=self.observed["calibration"]["threshold"],
                    ),
                ),
            ),
        )
        for patch in patches:
            patch.start()
            case.addCleanup(patch.stop)

    @property
    def artifact(self) -> Path:
        """Return the staged aggregate result path."""

        return self.root / "temp" / "v2" / "v2_package.json"

    @property
    def manifest(self) -> Path:
        """Return the staged public manifest path."""

        return self.root / "config" / "v2_results.json"

    @property
    def bundle(self) -> Path:
        """Return the staged joblib bundle path."""

        return (
            self.root
            / "artifacts"
            / "v2"
            / "consumer_complaint_detector_v2.joblib"
        )


class FullRunTests(unittest.TestCase):
    """Verify the freeze publishes only on an exact reproduction."""

    def test_matching_refit_freezes_the_package(self) -> None:
        """Freeze, persist, and validate when every check reproduces D1."""

        staged = _StagedRun(self)
        result = v2_package.run_v2_package("full", project_root=staged.root)
        self.assertEqual(result["outcome"], v2_package.OUTCOME_FROZEN)
        self.assertTrue(result["complete"])
        self.assertTrue(result["frozen"])
        self.assertTrue(result["reproduction_gate"]["passed"])
        self.assertTrue(staged.bundle.is_file())
        self.assertTrue(staged.artifact.is_file())
        self.assertTrue(staged.manifest.is_file())
        self.assertEqual(
            result["sealed_access"],
            {"test": False, "stress": False, "monitor": False},
        )
        self.assertEqual(result["bundle"]["persisted"], True)
        manifest = v2_package.validate_v2_manifest(
            staged.manifest, staged.artifact
        )
        self.assertEqual(manifest["outcome"], v2_package.OUTCOME_FROZEN)
        self.assertEqual(manifest["deployment_authorized"], False)

    def _assert_mismatch(self, staged: _StagedRun, failed_check: str) -> dict:
        """Run the freeze and assert a named, bundle-free divergence."""

        result = v2_package.run_v2_package("full", project_root=staged.root)
        self.assertEqual(result["outcome"], v2_package.OUTCOME_MISMATCH)
        self.assertTrue(result["complete"])
        self.assertFalse(result["frozen"])
        self.assertFalse(staged.bundle.exists())
        self.assertTrue(staged.artifact.is_file())
        self.assertIn(failed_check, result["reproduction_gate"]["failed_checks"])
        self.assertFalse(result["reproduction_gate"]["checks"][failed_check])
        self.assertEqual(result["bundle"]["persisted"], False)
        published = json.loads(staged.artifact.read_text(encoding="utf-8"))
        self.assertIn(
            failed_check, published["reproduction_gate"]["failed_checks"]
        )
        self.assertIsNone(
            json.loads(staged.manifest.read_text(encoding="utf-8"))["bundle"]
        )
        return result

    def test_threshold_mismatch_blocks_the_freeze(self) -> None:
        """Publish REPRODUCTION_MISMATCH when the threshold diverges."""

        def mutate(payload: dict) -> None:
            payload["candidates"][0]["calibration"]["threshold"] = -0.5

        staged = _StagedRun(self, mutate)
        self._assert_mismatch(staged, "calibrated_threshold")

    def test_confusion_matrix_mismatch_blocks_the_freeze(self) -> None:
        """Publish REPRODUCTION_MISMATCH when one outer cell diverges."""

        def mutate(payload: dict) -> None:
            matrix = payload["candidates"][0]["outer"]["metrics"][
                "confusion_matrix"
            ]
            matrix[2][3] += 1

        staged = _StagedRun(self, mutate)
        result = self._assert_mismatch(staged, "outer_confusion_matrix")
        summary = result["reproduction_gate"]["divergences"][
            "outer_confusion_matrix"
        ]
        self.assertEqual(summary["mismatched_cells"], 1)

    def test_override_count_mismatch_blocks_the_freeze(self) -> None:
        """Publish REPRODUCTION_MISMATCH when an override count diverges."""

        def mutate(payload: dict) -> None:
            payload["candidates"][0]["outer"]["effective_overrides"] = 83

        staged = _StagedRun(self, mutate)
        self._assert_mismatch(staged, "outer_effective_overrides")

    def test_rejects_an_unknown_mode(self) -> None:
        """Reject any run mode outside full and smoke."""

        with self.assertRaises(ValueError):
            v2_package.run_v2_package("disabled", project_root=ROOT)


class ManifestRoundTripTests(unittest.TestCase):
    """Verify the published manifest re-validates and detects tampering."""

    def setUp(self) -> None:
        """Publish one frozen package into a staged project tree."""

        self.staged = _StagedRun(self)
        v2_package.run_v2_package("full", project_root=self.staged.root)

    def test_round_trip_validates(self) -> None:
        """Re-validate every recorded hash and cross-check immediately."""

        manifest = v2_package.validate_v2_manifest(
            self.staged.manifest, self.staged.artifact
        )
        self.assertTrue(manifest["frozen"])
        self.assertTrue(manifest["reproduction_gate_passed"])
        self.assertEqual(manifest["candidate_id"], CANDIDATE_ID)

    def test_corrupted_recorded_hash_raises(self) -> None:
        """Reject a manifest whose recorded artifact hash was rewritten."""

        payload = json.loads(self.staged.manifest.read_text(encoding="utf-8"))
        payload["artifact"]["sha256"] = "0" * 64
        _write_json(self.staged.manifest, payload)
        with self.assertRaises(ValueError):
            v2_package.validate_v2_manifest(
                self.staged.manifest, self.staged.artifact
            )

    def test_corrupted_bundle_hash_raises(self) -> None:
        """Reject a manifest whose recorded bundle hash was rewritten."""

        payload = json.loads(self.staged.manifest.read_text(encoding="utf-8"))
        payload["bundle"]["sha256"] = "0" * 64
        _write_json(self.staged.manifest, payload)
        with self.assertRaises(ValueError):
            v2_package.validate_v2_manifest(
                self.staged.manifest, self.staged.artifact
            )

    def test_frozen_manifest_without_a_bundle_raises(self) -> None:
        """Reject a frozen manifest that dropped its bundle record."""

        payload = json.loads(self.staged.manifest.read_text(encoding="utf-8"))
        payload["bundle"] = None
        _write_json(self.staged.manifest, payload)
        with self.assertRaises(ValueError):
            v2_package.validate_v2_manifest(
                self.staged.manifest, self.staged.artifact
            )

    def test_loaded_predictor_applies_the_frozen_hierarchy(self) -> None:
        """Load the published package and serve through the hierarchy."""

        with mock.patch.object(
            v2_package, "load_s7_predictor", lambda *_: _StubFallback()
        ):
            predictor = v2_package.load_v2_predictor(self.staged.root)
        self.assertEqual(predictor.model_version, v2_package.MODEL_VERSION)
        self.assertEqual(predictor.input_language, "en-US")
        batch = predictor.predict(list(BUNDLE_TEXTS))
        self.assertEqual(len(batch.predictions), len(BUNDLE_TEXTS))
        for prediction in batch.predictions:
            self.assertIn(prediction.label, MODELED_FAMILIES)


class PrivacyTests(unittest.TestCase):
    """Verify row-level evidence can never reach a published artifact."""

    def test_forbidden_key_is_rejected_before_writing(self) -> None:
        """Reject an aggregate result carrying a forbidden row-level key."""

        observed = _observed_evidence()
        observed["outer"]["scores"] = [0.1, 0.2, 0.3]
        with tempfile.TemporaryDirectory() as temporary:
            root = _stage_root(
                Path(temporary), _observed_evidence(), _d1_payload(observed)
            )
            config = v2_package.load_v2_package_config(project_root=root)
            protocol = load_v2_protocol(root / "config" / PROTOCOL_PATH.name)
            base = v2_package._base_result(
                config,
                protocol,
                run_mode="full",
                diagnostic_only=False,
                opened_at="2026-08-19T00:00:00Z",
                signature="test",
            )
            with self.assertRaises(ValueError):
                v2_package._complete_result(
                    base,
                    observed,
                    {"passed": True, "failed_checks": []},
                    {},
                    {"persisted": False},
                    FALLBACK_MODEL_VERSION,
                    0.0,
                )

    def test_published_artifact_never_names_a_model_key(self) -> None:
        """Keep the frozen artifact free of the banned model key names."""

        staged = _StagedRun(self)
        result = v2_package.run_v2_package("full", project_root=staged.root)
        serialized = json.dumps(result)
        self.assertNotIn('"model":', serialized)
        self.assertNotIn('"models":', serialized)
        self.assertNotIn('"indices":', serialized)
        self.assertIn("model_spec", result)


class SmokeTests(unittest.TestCase):
    """Verify the diagnostic preflight fits nothing and writes nothing."""

    def test_smoke_is_diagnostic_only_and_writes_nothing(self) -> None:
        """Validate the config, hashes, D1 record, and S7 fallback only."""

        staged = _StagedRun(self)
        result = v2_package.run_v2_package_smoke(staged.root)
        self.assertTrue(result["diagnostic_only"])
        self.assertTrue(result["complete"])
        self.assertFalse(result["frozen"])
        self.assertFalse(result["fitted"])
        self.assertEqual(result["outcome"], "DIAGNOSTIC_ONLY")
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(
            result["d1_reference"]["candidate_id"], CANDIDATE_ID
        )
        self.assertFalse(staged.artifact.exists())
        self.assertFalse(staged.manifest.exists())
        self.assertFalse(staged.bundle.exists())

    def test_smoke_rejects_a_drifted_pinned_artifact(self) -> None:
        """Refuse to report a clean preflight over a rewritten D1 file."""

        staged = _StagedRun(self)
        drifted = staged.root / "temp" / "v2" / "v2_classical_benchmark.json"
        drifted.write_text(
            drifted.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            v2_package.run_v2_package_smoke(staged.root)


class RootResolutionTests(unittest.TestCase):
    """Pin every entry point to an explicit root, never the process cwd."""

    def test_default_root_is_the_package_location(self) -> None:
        """Derive the default root from the module, not the working directory."""

        original = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                resolved = v2_package.default_project_root()
            finally:
                os.chdir(original)
        self.assertEqual(resolved, ROOT.resolve())

    def test_smoke_uses_the_given_root_from_a_foreign_directory(self) -> None:
        """Let a staged kernel run the preflight from any working directory."""

        staged = _StagedRun(self)
        expected = v2_package._sha256(
            staged.root / "config" / "v2_frozen_package.json"
        )
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                result = v2_package.run_v2_package_smoke(staged.root)
            finally:
                os.chdir(original)
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(result["frozen_config"]["sha256"], expected)

    def test_full_run_uses_the_given_root_from_a_foreign_directory(self) -> None:
        """Publish into the staged root even when the cwd is unrelated."""

        staged = _StagedRun(self)
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                result = v2_package.run_v2_package(
                    "full", project_root=staged.root
                )
                stray = list(Path(elsewhere).rglob("*.json"))
            finally:
                os.chdir(original)
        self.assertEqual(result["outcome"], v2_package.OUTCOME_FROZEN)
        self.assertEqual(stray, [])
        self.assertTrue(staged.artifact.is_file())


if __name__ == "__main__":
    unittest.main()
