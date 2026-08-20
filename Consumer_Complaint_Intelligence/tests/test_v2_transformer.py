"""Synthetic, torch-free tests for the D2 transformer challenge runner."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from consumer_complaint_intelligence import v2_benchmark, v2_transformer
from consumer_complaint_intelligence.s6 import CRITICAL_CLASS


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "config" / "v2_d2_execution.json"
PROTOCOL_PATH = ROOT / "config" / "v2_development_protocol.json"


def _load_config_payload() -> dict:
    """Read the frozen D2 execution config JSON as a plain dict."""

    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(directory: Path, payload: dict) -> Path:
    """Write one mutated D2 execution config JSON to ``directory``."""

    path = directory / "v2_d2_execution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stub_scorer_factory(critical_margin: float, non_critical_margin: float):
    """Build a deterministic ``scorer_factory`` for direct pipeline tests."""

    def factory(fit_texts, fit_targets, seed, config):
        del fit_texts, fit_targets, config

        def score(batch_texts):
            offset = float(seed) * 1e-9
            return np.asarray(
                [
                    (critical_margin if CRITICAL_CLASS in text else -1.0)
                    + non_critical_margin
                    + offset
                    for text in batch_texts
                ],
                dtype=np.float64,
            )

        score.resolved_revision = f"stub-{seed}"
        return score

    return factory


def _fake_outer_block(
    *,
    critical_f1: float,
    critical_precision: float,
    effective_overrides: int,
    passed_margins: bool = True,
    gate_count: int = 3,
) -> dict:
    """Build one minimal fake ``outer`` block for decision-rule tests."""

    return {
        "metrics": {
            "critical_f1": critical_f1,
            "critical_precision": critical_precision,
            "macro_f1": 0.75,
        },
        "gates": {"passed": True, "gate_count": 3},
        "safety": {"passed": passed_margins, "gate_count": gate_count},
        "override_decisions": max(effective_overrides, 0),
        "effective_overrides": effective_overrides,
    }


class D2ExecutionConfigTests(unittest.TestCase):
    """Verify the frozen D2 execution config loads strictly."""

    def test_frozen_config_loads_and_exposes_pre_registered_values(self) -> None:
        """Load the real frozen contract and expose its key properties."""

        config = v2_transformer.load_d2_execution_config(CONFIG_PATH)
        self.assertEqual(config.model_id, "distilbert-base-uncased")
        self.assertEqual(config.seeds, (42, 43, 44))
        self.assertEqual(config.max_length, 256)
        self.assertEqual(config.epochs, 3)
        self.assertAlmostEqual(config.learning_rate, 0.00002)
        self.assertEqual(config.train_batch_size, 32)
        self.assertEqual(config.eval_batch_size, 128)
        self.assertAlmostEqual(config.precision_floor, 0.4342857142857143)
        self.assertAlmostEqual(config.displacement_bar, 0.4255986693961106)
        self.assertEqual(
            config.incumbent_candidate_id,
            "word_char_tfidf_union_40000_60000_c_1_hard_negative",
        )
        self.assertIn("sha256", config.signature)
        self.assertIn("size_bytes", config.signature)

    def test_rejects_wrong_schema_version(self) -> None:
        """Reject a config whose schema_version has drifted."""

        payload = _load_config_payload()
        payload["schema_version"] = "v2-d2-execution-config-v2"
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_config(Path(temporary), payload)
            with self.assertRaises(ValueError):
                v2_transformer.load_d2_execution_config(path)

    def test_rejects_wrong_seeds(self) -> None:
        """Reject a config whose seeds are not exactly 42, 43, 44."""

        payload = _load_config_payload()
        payload["seeds"]["values"] = [1, 2, 3]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_config(Path(temporary), payload)
            with self.assertRaises(ValueError):
                v2_transformer.load_d2_execution_config(path)

    def test_rejects_maximum_models_not_one(self) -> None:
        """Reject a config that widens the model search beyond one model."""

        payload = _load_config_payload()
        payload["model"]["maximum_models"] = 2
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_config(Path(temporary), payload)
            with self.assertRaises(ValueError):
                v2_transformer.load_d2_execution_config(path)

    def test_rejects_non_pre_registered_decision_rule(self) -> None:
        """Reject a config whose decision rule is not pre-registered."""

        payload = _load_config_payload()
        payload["decision_rule"]["pre_registered"] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_config(Path(temporary), payload)
            with self.assertRaises(ValueError):
                v2_transformer.load_d2_execution_config(path)


class HardNegativePoolSignatureTests(unittest.TestCase):
    """Verify the hard-negative pool fingerprint is order-independent."""

    def test_signature_is_order_independent_and_stable(self) -> None:
        """Ignore input order and repeat identically across calls."""

        first = v2_transformer.hard_negative_pool_signature([5, 1, 3, 2, 4])
        second = v2_transformer.hard_negative_pool_signature([1, 2, 3, 4, 5])
        third = v2_transformer.hard_negative_pool_signature([4, 3, 2, 1, 5])
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(len(first), 64)
        self.assertEqual(first, first.upper())

    def test_signature_changes_with_the_pool(self) -> None:
        """Produce a different fingerprint for a different index set."""

        base = v2_transformer.hard_negative_pool_signature([1, 2, 3])
        changed = v2_transformer.hard_negative_pool_signature([1, 2, 4])
        self.assertNotEqual(base, changed)


class ReportedSeedSelectionTests(unittest.TestCase):
    """Verify median aggregation never reports the best seed."""

    def test_median_seed_is_reported_never_the_best_or_worst(self) -> None:
        """Report seed 42 when it attains the median of a spread trio."""

        seed_results = [
            {"seed": 42, "outer": {"metrics": {"critical_f1": 0.30}}},
            {"seed": 43, "outer": {"metrics": {"critical_f1": 0.55}}},
            {"seed": 44, "outer": {"metrics": {"critical_f1": 0.10}}},
        ]
        reported, spread = v2_transformer._select_reported_seed(seed_results)
        self.assertEqual(reported["seed"], 42)
        self.assertAlmostEqual(spread["critical_f1_min"], 0.10)
        self.assertAlmostEqual(spread["critical_f1_median"], 0.30)
        self.assertAlmostEqual(spread["critical_f1_max"], 0.55)
        self.assertAlmostEqual(spread["critical_f1_range"], 0.45)

    def test_median_selection_is_order_independent(self) -> None:
        """Reach the same reported seed regardless of input ordering."""

        seed_results = [
            {"seed": 44, "outer": {"metrics": {"critical_f1": 0.10}}},
            {"seed": 42, "outer": {"metrics": {"critical_f1": 0.30}}},
            {"seed": 43, "outer": {"metrics": {"critical_f1": 0.55}}},
        ]
        reported, _ = v2_transformer._select_reported_seed(seed_results)
        self.assertEqual(reported["seed"], 42)


class DecisionRuleTests(unittest.TestCase):
    """Verify the five pre-registered D2 displacement conditions."""

    def setUp(self) -> None:
        """Load the real frozen D2 config used by every decision check."""

        self.config = v2_transformer.load_d2_execution_config(CONFIG_PATH)
        self.fallback_f1 = 0.339665

    def test_blocked_below_displacement_bar(self) -> None:
        """Block the outcome when critical F1 clears the fallback but not
        the displacement bar."""

        outer = _fake_outer_block(
            critical_f1=0.40, critical_precision=0.45, effective_overrides=10
        )
        decision = v2_transformer._evaluate_decision_rule(
            {"outer": outer}, self.fallback_f1, self.config
        )
        self.assertEqual(decision["outcome"], "CLASSICAL_WINNER_STANDS")
        self.assertEqual(
            decision["blocked_reason"], "outer_critical_f1_at_least_displacement_bar"
        )
        self.assertTrue(decision["outer_critical_f1_greater_than_fallback_baseline"])
        self.assertFalse(decision["outer_critical_f1_at_least_displacement_bar"])

    def test_blocked_on_zero_effective_overrides(self) -> None:
        """Block the outcome even when every metric clears its bar."""

        outer = _fake_outer_block(
            critical_f1=0.50,
            critical_precision=0.50,
            effective_overrides=0,
        )
        decision = v2_transformer._evaluate_decision_rule(
            {"outer": outer}, self.fallback_f1, self.config
        )
        self.assertEqual(decision["outcome"], "CLASSICAL_WINNER_STANDS")
        self.assertEqual(
            decision["blocked_reason"], "effective_overrides_greater_than_zero"
        )

    def test_transformer_displaces_classical_when_every_condition_clears(
        self,
    ) -> None:
        """Reach TRANSFORMER_DISPLACES_CLASSICAL when all five conditions hold."""

        outer = _fake_outer_block(
            critical_f1=self.config.displacement_bar + 0.01,
            critical_precision=self.config.precision_floor + 0.01,
            effective_overrides=50,
        )
        decision = v2_transformer._evaluate_decision_rule(
            {"outer": outer}, self.fallback_f1, self.config
        )
        self.assertEqual(decision["outcome"], "TRANSFORMER_DISPLACES_CLASSICAL")
        self.assertIsNone(decision["blocked_reason"])
        self.assertTrue(all(
            decision[name] for name in self.config.decision_conditions
        ))

    def test_blocked_when_safety_margins_incomplete(self) -> None:
        """Block the outcome first on the weakest, earliest-listed condition."""

        outer = _fake_outer_block(
            critical_f1=self.config.displacement_bar + 0.01,
            critical_precision=self.config.precision_floor + 0.01,
            effective_overrides=50,
            passed_margins=False,
            gate_count=2,
        )
        decision = v2_transformer._evaluate_decision_rule(
            {"outer": outer}, self.fallback_f1, self.config
        )
        self.assertEqual(decision["outcome"], "CLASSICAL_WINNER_STANDS")
        self.assertEqual(
            decision["blocked_reason"],
            "passes_three_of_three_development_safety_margins",
        )


class ExecuteSeedsPipelineTests(unittest.TestCase):
    """Exercise the full torch-free pipeline with synthetic scopes."""

    def setUp(self) -> None:
        """Load the real frozen protocol and D2 config for pipeline tests."""

        self.protocol = v2_benchmark.load_v2_protocol(PROTOCOL_PATH)
        self.config = v2_transformer.load_d2_execution_config(CONFIG_PATH)

    def test_execute_seeds_runs_three_seeds_with_a_stub_scorer(self) -> None:
        """Run all three seeds against synthetic scopes with no torch."""

        scopes, fallback = v2_benchmark._synthetic_scopes()
        seed_results, evidence, baseline = v2_transformer._execute_seeds(
            scopes,
            fallback,
            self.protocol,
            self.config,
            batch_size=4096,
            scorer_factory=v2_transformer._synthetic_scorer_factory,
        )
        self.assertEqual(len(seed_results), 3)
        self.assertEqual(
            {item["seed"] for item in seed_results}, {42, 43, 44}
        )
        for item in seed_results:
            self.assertIn("calibration", item)
            self.assertIn("outer", item)
            self.assertEqual(item["resolved_revision"], "synthetic")
        self.assertEqual(len(evidence["pool_signature"]), 64)
        self.assertGreater(evidence["pool_rows"], evidence["positive_groups"])
        self.assertEqual(
            set(baseline), {"inner_calibration", "outer_evaluation"}
        )

    def test_execute_seeds_pool_signature_is_reproducible(self) -> None:
        """Reproduce the identical hard-negative pool across independent runs."""

        scopes, fallback = v2_benchmark._synthetic_scopes()
        _, first_evidence, _ = v2_transformer._execute_seeds(
            scopes,
            fallback,
            self.protocol,
            self.config,
            batch_size=4096,
            scorer_factory=v2_transformer._synthetic_scorer_factory,
        )
        _, second_evidence, _ = v2_transformer._execute_seeds(
            scopes,
            fallback,
            self.protocol,
            self.config,
            batch_size=4096,
            scorer_factory=v2_transformer._synthetic_scorer_factory,
        )
        self.assertEqual(
            first_evidence["pool_signature"], second_evidence["pool_signature"]
        )


class SmokeModeTests(unittest.TestCase):
    """Verify the diagnostic smoke path completes without touching torch."""

    def test_smoke_completes_and_is_diagnostic_only(self) -> None:
        """Run a single-seed synthetic diagnostic and validate its shape."""

        result = v2_transformer.run_v2_transformer_smoke()
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertTrue(result["complete"])
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(len(result["seeds"]), 1)
        self.assertIn("outcome", result["decision"])
        self.assertIn(
            result["decision"]["outcome"],
            {"TRANSFORMER_DISPLACES_CLASSICAL", "CLASSICAL_WINNER_STANDS"},
        )
        self.assertEqual(
            set(result["seed_spread"]),
            {
                "critical_f1_min",
                "critical_f1_median",
                "critical_f1_max",
                "critical_f1_range",
            },
        )
        self.assertEqual(result["model_spec"]["model_id"], "distilbert-base-uncased")

    def test_smoke_result_passes_the_shared_privacy_validator(self) -> None:
        """Keep the smoke artifact free of forbidden aggregate-only keys."""

        result = v2_transformer.run_v2_transformer_smoke()
        v2_transformer._validate_result_privacy(result)
        serialized = json.dumps(result)
        self.assertNotIn('"model":', serialized)
        self.assertNotIn('"models":', serialized)


class ValidateD2ManifestTests(unittest.TestCase):
    """Verify manifest validation rejects a tampered or incomplete pair."""

    def test_rejects_incomplete_manifest_and_artifact(self) -> None:
        """Reject a manifest/artifact pair missing required path fields."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = directory / "result.json"
            artifact.write_text(
                json.dumps({"schema_version": v2_transformer.D2_RESULT_SCHEMA}),
                encoding="utf-8",
            )
            manifest = directory / "manifest.json"
            payload = {
                "schema_version": v2_transformer.D2_MANIFEST_SCHEMA,
                "complete": True,
                "diagnostic_only": False,
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                v2_transformer.validate_d2_manifest(manifest, artifact)

    def test_rejects_wrong_manifest_schema(self) -> None:
        """Reject a manifest whose schema_version has drifted."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = directory / "result.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = directory / "manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": "stale-schema"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                v2_transformer.validate_d2_manifest(manifest, artifact)


if __name__ == "__main__":
    unittest.main()


class SmokeRootResolutionTests(unittest.TestCase):
    """Pin the D2 smoke to an explicit root rather than the process cwd."""

    def test_smoke_uses_the_given_root_from_a_foreign_working_directory(
        self,
    ) -> None:
        """Let a staged caller run the smoke from any working directory."""

        root = Path(__file__).resolve().parents[1]
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                result = v2_transformer.run_v2_transformer_smoke(root)
            finally:
                os.chdir(original)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertTrue(result["diagnostic_only"])

    def test_smoke_without_a_root_fails_outside_the_project(self) -> None:
        """Keep the cwd default from silently resolving to a wrong tree."""

        original = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                with self.assertRaises(ValueError):
                    v2_transformer.run_v2_transformer_smoke()
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
