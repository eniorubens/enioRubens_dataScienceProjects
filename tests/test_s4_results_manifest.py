"""Validate the versioned manifest for the real S4 development results."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "config" / "s4_results.json"
ARTIFACT = ROOT / "temp" / "s4" / "s4_results.json"


class S4ResultsManifestTests(unittest.TestCase):
    """Verify publication status and real-result traceability."""

    def test_real_manifest_is_non_confirmatory_and_unselected(self) -> None:
        """Keep the real S4 result diagnostic and unpromoted."""

        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "DEVELOPMENT_CHALLENGE_COMPLETE")
        self.assertFalse(payload["confirmatory"])
        self.assertFalse(payload["deploy_ready"])
        self.assertFalse(payload["sealed"])
        self.assertEqual(payload["selection_status"], "NO_ELIGIBLE_CHALLENGER")
        self.assertIsNone(payload["recommended_candidate"])
        self.assertEqual(
            payload["source_artifact"]["path"],
            "temp/s4/s4_results.json",
        )
        self.assertTrue(payload["source_artifact"]["complete"])
        self.assertFalse(payload["source_artifact"]["modeling_recomputed"])
        self.assertEqual(
            payload["source_artifact"]["sha256"],
            "3F6F36DC0B768A36FD650F32D09A58809E3E7676CDA0897D7599BB60614F01A8",
        )
        self.assertEqual(
            payload["metrics"][0]["candidate"],
            "word_balanced_reference",
        )
        self.assertEqual(len(payload["metrics"]), 4)
        artifact_by_name = {
            item["name"]: item for item in artifact["candidates"]
        }
        for metric in payload["metrics"]:
            real_metrics = artifact_by_name[metric["candidate"]]["metrics"]
            critical = real_metrics["per_class"]["debt_credit_management"]
            for key in (
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy",
            ):
                self.assertAlmostEqual(metric[key], real_metrics[key])
            for key, critical_key in (
                ("critical_precision", "precision"),
                ("critical_recall", "recall"),
                ("critical_f1", "f1"),
            ):
                self.assertAlmostEqual(metric[key], critical[critical_key])
            self.assertEqual(metric["gates"], artifact_by_name[metric["candidate"]]["gates"])
            self.assertEqual(metric["eligible"], artifact_by_name[metric["candidate"]]["eligible"])
        self.assertEqual(payload["resources"]["elapsed_seconds"], 2193.0)
        self.assertEqual(payload["resources"]["peak_rss_gb"], 7.24)
        self.assertEqual(payload["resources"]["minimum_system_available_gb"], 0.01)
        self.assertEqual(artifact["selection_status"], payload["selection_status"])
        self.assertIsNone(artifact["recommended_candidate"])

    def test_manifest_contains_visible_pt_br_accents(self) -> None:
        """Keep visible Portuguese interpretation correctly encoded."""

        text = MANIFEST.read_text(encoding="utf-8")
        for value in ("três", "diagnóstico", "seleção", "partição", "precisão"):
            self.assertIn(value, text)
        for value in ("crítica", "não", "promoção", "prático", "disponível"):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
