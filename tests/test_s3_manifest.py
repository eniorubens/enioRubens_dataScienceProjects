"""Validate the versioned S3 development-baseline manifest."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class S3ManifestTests(unittest.TestCase):
    """Check manifest schema and persisted execution values without data access."""

    def test_manifest_declares_development_only_evidence(self) -> None:
        """Require the real baseline values and reject deployment claims."""

        path = ROOT / "config" / "s3_baseline.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], "s3-baseline-manifest-v1")
        self.assertEqual(
            manifest["status"], "DEVELOPMENT_BASELINE_COMPLETE"
        )
        self.assertFalse(manifest["claim_scope"]["confirmatory_result"])
        self.assertFalse(manifest["claim_scope"]["deploy_ready"])
        self.assertEqual(manifest["baseline"]["max_features"], 40000)
        self.assertEqual(manifest["baseline"]["batch_size_rows"], 4096)
        self.assertEqual(manifest["scientific_units"]["train_groups"], 345552)
        self.assertEqual(
            manifest["scientific_units"]["validation_groups"], 245980
        )
        self.assertAlmostEqual(
            manifest["metrics"]["final_operational_all_text_macro_f1"],
            0.6784213061,
        )
        self.assertNotIn("sha256", json.dumps(manifest).lower())


if __name__ == "__main__":
    unittest.main()
