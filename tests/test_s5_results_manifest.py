"""Validate the S5 result manifest against the real artifacts."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "config" / "s5_results.json"
ARTIFACT = ROOT / "temp" / "s5" / "s5_results.json"
RESOURCE_MONITOR = ROOT / "temp" / "s5" / "s5_resource_monitor.json"


class S5ResultsManifestTests(unittest.TestCase):
    """Verify S5 traceability, metrics, resources, and publication status."""

    def test_manifest_matches_real_s5_artifacts(self) -> None:
        """Compare every published S5 field with the stored artifacts."""

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        monitor = json.loads(RESOURCE_MONITOR.read_text(encoding="utf-8"))
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest().upper()

        self.assertEqual(manifest["schema_version"], "s5-results-manifest-v1")
        self.assertFalse(manifest["confirmatory"])
        self.assertFalse(manifest["deploy_ready"])
        self.assertFalse(manifest["sealed"])
        self.assertEqual(manifest["selection_status"], "NO_ELIGIBLE_ESTIMATOR")
        self.assertIsNone(manifest["recommended_candidate"])
        self.assertEqual(manifest["source_artifact"]["sha256"], digest)
        self.assertEqual(manifest["source_artifact"]["sha256"], "8A41690A7C337EC64DDC288E27A82AF3C0BF87C91FE8BEC48D3FEDDD20AC183C")
        self.assertEqual(manifest["selection_status"], artifact["selection_status"])
        self.assertEqual(manifest["recommended_candidate"], artifact["recommended_candidate"])

        for key in ("status", "peak_process_rss_bytes", "min_system_available_bytes", "samples", "started_at_epoch", "elapsed_seconds", "peak_process_rss_gb", "min_system_available_gb"):
            self.assertEqual(manifest["resource_monitor"][key], monitor[key])

        by_name = {item["name"]: item for item in artifact["candidates"]}
        self.assertEqual(len(manifest["metrics"]), len(by_name))
        for published in manifest["metrics"]:
            actual = by_name[published["candidate"]]
            self.assertEqual(published["estimator"], actual["estimator"])
            for key in ("macro_f1", "weighted_f1", "balanced_accuracy", "runtime_seconds", "train_rows", "validation_rows"):
                self.assertEqual(published[key], actual["metrics"].get(key, actual.get(key)))
            critical = actual["metrics"]["per_class"]["debt_credit_management"]
            self.assertEqual(published["critical_precision"], critical["precision"])
            self.assertEqual(published["critical_recall"], critical["recall"])
            self.assertEqual(published["critical_f1"], critical["f1"])
            self.assertEqual(published["gates"], actual["gates"])
            self.assertEqual(published["eligible"], actual["eligible"])

        self.assertEqual(manifest["parity"], {
            "status": artifact["reference_reproduction"]["status"],
            "reference_artifact": "temp/s4/s4_results.json",
            "candidate": "word_balanced_reference",
            "tolerance": artifact["reference_reproduction"]["tolerance"],
            "passed": artifact["reference_reproduction"]["passed"],
            "actual": artifact["reference_reproduction"]["actual"],
            "reference": artifact["reference_reproduction"]["reference"],
            "deltas": artifact["reference_reproduction"]["deltas"],
        })

    def test_interpretation_is_utf8_pt_br_and_non_promotional(self) -> None:
        """Keep the Portuguese interpretation readable and diagnostic."""

        text = MANIFEST.read_text(encoding="utf-8")
        for value in ("três", "diagnóstico", "não", "seleção", "partição", "precisão", "adiado"):
            self.assertIn(value, text)
        self.assertNotIn("Ã", text)
        self.assertIn("linear_svc_balanced", text)
        self.assertIn("não é seleção nem promoção", text)


if __name__ == "__main__":
    unittest.main()
