"""Validate the frozen S6 results manifest against its development evidence."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "config" / "s6_results.json"
ARTIFACT = ROOT / "temp" / "s6" / "s6_results.json"


class S6ResultsManifestTests(unittest.TestCase):
    """Check S6 provenance, statuses, evidence, and Portuguese text."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the manifest, artifact, and resource monitors once."""

        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_hash_and_schema_are_frozen(self) -> None:
        """Match the declared artifact digest and manifest schema."""

        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest().upper()
        self.assertEqual(digest, "050AA8B379A4F2DB95A29662C69AA214CD4154971C2C1534384500967D7211AE")
        self.assertEqual(self.manifest["schema_version"], "s6-results-manifest-v1")
        self.assertEqual(self.manifest["artifact"]["sha256"], digest)

    def test_manifest_matches_artifact_and_monitors(self) -> None:
        """Compare status, candidates, outer evidence, and monitor evidence."""

        artifact = self.manifest["artifact"]
        self.assertEqual(artifact["status"], self.artifact["status"])
        self.assertEqual(artifact["selection_status"], self.artifact["selection_status"])
        self.assertEqual(artifact["recommended_candidate"], self.artifact["recommended_candidate"])
        self.assertFalse(artifact["deploy"])
        self.assertFalse(artifact["sealed"])
        self.assertEqual(len(self.manifest["candidates"]), 5)
        self.assertEqual(self.manifest["execution"]["execution_attempts"], self.artifact["execution_attempts"])
        self.assertEqual(self.manifest["execution"]["vectorizer_fit_count"], self.artifact["vectorizer_fit_count"])
        outer = self.manifest["candidates"][2]["outer"]
        self.assertEqual(outer["row_count"], self.artifact["outer_summary"]["metrics"]["row_count"])
        self.assertAlmostEqual(outer["global_macro_f1"], self.artifact["outer_summary"]["gates"]["values"]["global_macro_f1"])
        self.assertEqual(self.manifest["resource_monitors"][0]["status"], "ABORTED_LOW_MEMORY")
        self.assertEqual(self.manifest["resource_monitors"][1]["status"], "COMPLETE")

    def test_warning_statuses_and_accented_interpretation(self) -> None:
        """Require the SAGA warning, development status, and PT-BR accents."""

        saga = self.manifest["candidates"][1]
        self.assertTrue(saga["warnings"])
        self.assertIn("did not converge", saga["warnings"][0])
        self.assertEqual(self.manifest["execution"]["attempts"][0]["status"], "ABORTED_LOW_MEMORY")
        self.assertEqual(self.manifest["execution"]["attempts"][1]["status"], "COMPLETE")
        text = self.manifest["interpretation_pt_br"]
        for term in ("recomendado", "elegível", "desenvolvimento", "não", "partições", "seladas"):
            self.assertIn(term, text)
        self.assertIn("Nenhuma abertura automática", text)


if __name__ == "__main__":
    unittest.main()
