"""Validate the optional S7 public manifest when full packaging exists."""

import json
import unittest
from pathlib import Path

from consumer_complaint_intelligence.s7 import (
    DEFAULT_BUNDLE,
    DEFAULT_RESULT,
    validate_s7_manifest,
)


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "config" / "s7_results.json"


class S7ManifestTests(unittest.TestCase):
    """Keep manifest tests tolerant before the explicitly deferred full run."""

    def test_manifest_is_optional_before_full(self) -> None:
        """Allow the public result to be absent while S7 remains unexecuted."""

        if not MANIFEST.exists():
            self.skipTest("S7 full packaging was explicitly deferred")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        validate_s7_manifest(
            manifest,
            ROOT / DEFAULT_BUNDLE,
            ROOT / DEFAULT_RESULT,
        )
        self.assertEqual(manifest["status"], "packaged_for_confirmation")
        self.assertTrue(manifest["development_only"])
        self.assertFalse(manifest["deploy"])
        self.assertFalse(manifest["confirmatory"])
        self.assertIn("não há", manifest["interpretation_pt_br"])


if __name__ == "__main__":
    unittest.main()
