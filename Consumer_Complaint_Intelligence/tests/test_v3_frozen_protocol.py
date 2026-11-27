"""Tests for the execution-blocking V3-E0 scientific contract."""

import copy
import json
import unittest
from pathlib import Path

from consumer_complaint_intelligence.v3 import protocol as v3_protocol


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "v3" / "development_protocol.json"
MANIFEST = ROOT / "config" / "v3" / "development_protocol_manifest.json"


class V3ProtocolTests(unittest.TestCase):
    """Verify temporal, sampling, representation, and seal boundaries."""

    def setUp(self) -> None:
        """Load a detached protocol payload for mutation tests."""

        self.payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_protocol_loads_and_authorizes_development_only(self) -> None:
        """Authorize the frozen E2/E3 contract without weakening boundaries."""

        protocol = v3_protocol.load_v3_protocol(CONFIG)
        self.assertEqual(protocol.approval_status, "FROZEN_FOR_V3_DEVELOPMENT")
        self.assertTrue(protocol.is_frozen)
        self.assertEqual(tuple(item["fold_id"] for item in protocol.folds), (
            "F1",
            "F2",
            "F3",
        ))
        self.assertEqual(len(protocol.protocol_sha256), 64)
        protocol.validate_for_scientific_execution()

    def test_approval_status_and_schema_must_change_together(self) -> None:
        """Reject a false freeze or downgrade with inconsistent metadata."""

        changed = copy.deepcopy(self.payload)
        changed["schema_version"] = "v3-development-protocol-v1-draft"
        with self.assertRaisesRegex(ValueError, "schema version"):
            v3_protocol.validate_v3_protocol(changed)

    def test_frozen_manifest_pins_file_and_normalized_protocol_hashes(self) -> None:
        """Verify approval, authorized stages, boundaries, and both hashes."""

        manifest = v3_protocol.validate_protocol_manifest(MANIFEST, ROOT)
        self.assertEqual(manifest["approved_by"], "user")
        self.assertEqual(
            manifest["authorized_next_stages"],
            [
                "V3-E2_DEVELOPMENT_EMBEDDING_CACHE",
                "V3-E3_ROLLING_LINEAR_PROBES",
            ],
        )
        self.assertFalse(manifest["deployment_authorized"])

        changed = copy.deepcopy(self.payload)
        changed["approval"] = {
            "status": "PROPOSED_FOR_USER_REVIEW",
            "proposed_on": "2026-08-21",
            "approved_on": None,
            "approved_by": None,
        }
        with self.assertRaisesRegex(ValueError, "schema version"):
            v3_protocol.validate_v3_protocol(changed)

    def test_only_train_and_validation_are_available(self) -> None:
        """Keep consumed and sealed partitions outside development code."""

        for partition in ("train", "validation"):
            self.assertEqual(
                v3_protocol.require_development_partition(partition), partition
            )
        for partition in ("test", "stress", "monitor"):
            with self.assertRaises(ValueError):
                v3_protocol.require_development_partition(partition)

    def test_unknown_top_level_field_is_rejected(self) -> None:
        """Prevent silent expansion of the experimental contract."""

        changed = copy.deepcopy(self.payload)
        changed["new_model_search"] = True
        with self.assertRaisesRegex(ValueError, "top-level"):
            v3_protocol.validate_v3_protocol(changed)

    def test_folds_are_exact_expanding_and_chronological(self) -> None:
        """Reject overlap, reordered folds, and unaudited support changes."""

        overlap = copy.deepcopy(self.payload)
        overlap["rolling_origin"]["folds"][0]["calibration"]["start"] = (
            "2023-12-31"
        )
        with self.assertRaisesRegex(ValueError, "overlaps calibration"):
            v3_protocol.validate_v3_protocol(overlap)

        reordered = copy.deepcopy(self.payload)
        reordered["rolling_origin"]["folds"].reverse()
        with self.assertRaisesRegex(ValueError, "identifiers or ordering"):
            v3_protocol.validate_v3_protocol(reordered)

        changed_support = copy.deepcopy(self.payload)
        changed_support["rolling_origin"]["folds"][2]["fit"]["critical_rows"] = 947
        with self.assertRaisesRegex(ValueError, "differs from audited scope"):
            v3_protocol.validate_v3_protocol(changed_support)

    def test_later_windows_must_keep_natural_prevalence(self) -> None:
        """Forbid resampling calibration or evaluation windows."""

        changed = copy.deepcopy(self.payload)
        changed["sampling"]["resampling_calibration_or_evaluation"] = True
        with self.assertRaisesRegex(ValueError, "later windows"):
            v3_protocol.validate_v3_protocol(changed)

    def test_sampling_ratios_and_compositions_are_frozen_as_a_set(self) -> None:
        """Require 1:5, 1:10, and the V2-comparable 1:15 control."""

        changed = copy.deepcopy(self.payload)
        changed["sampling"]["negative_ratios_per_positive"] = [1, 5, 10, 15]
        with self.assertRaisesRegex(ValueError, "negative ratios"):
            v3_protocol.validate_v3_protocol(changed)

        changed = copy.deepcopy(self.payload)
        changed["sampling"]["negative_composition"]["10"]["hard"] = 8
        with self.assertRaisesRegex(ValueError, "composition differs"):
            v3_protocol.validate_v3_protocol(changed)

    def test_initial_catalog_is_tfidf_control_and_frozen_minilm(self) -> None:
        """Prevent an unregistered model search or early MiniLM fine-tuning."""

        changed = copy.deepcopy(self.payload)
        changed["screening"]["representations"][1]["encoder_trainable"] = True
        with self.assertRaisesRegex(ValueError, "must remain frozen"):
            v3_protocol.validate_v3_protocol(changed)

        changed = copy.deepcopy(self.payload)
        changed["screening"]["representations"].append(
            {"candidate_id": "unregistered_model"}
        )
        with self.assertRaisesRegex(ValueError, "exactly two"):
            v3_protocol.validate_v3_protocol(changed)

    def test_unresolved_revision_blocks_a_false_frozen_status(self) -> None:
        """Require an immutable encoder commit before scientific execution."""

        changed = copy.deepcopy(self.payload)
        changed["approval"] = {
            "status": "FROZEN_FOR_V3_DEVELOPMENT",
            "proposed_on": "2026-08-21",
            "approved_on": "2026-08-21",
            "approved_by": "user",
        }
        changed["screening"]["representations"][1]["revision"] = None
        changed["screening"]["representations"][1]["revision_status"] = (
            "MUST_RESOLVE_BEFORE_PROTOCOL_FREEZE"
        )
        with self.assertRaisesRegex(ValueError, "revision is not frozen"):
            v3_protocol.validate_v3_protocol(changed)

    def test_s7_is_forbidden_in_early_in_sample_folds(self) -> None:
        """Protect F1 and F2 from the frozen fallback's fit-scope leakage."""

        changed = copy.deepcopy(self.payload)
        changed["selection"]["historical_s7_use"]["F1"] = (
            "ALLOWED_FOR_DEVELOPMENT_SYSTEM_COMPARISON"
        )
        with self.assertRaisesRegex(ValueError, "S7 fold boundary"):
            v3_protocol.validate_v3_protocol(changed)

    def test_precision_floor_and_three_fold_requirement_are_explicit(self) -> None:
        """Reject post-hoc changes to stability and operating constraints."""

        changed = copy.deepcopy(self.payload)
        changed["calibration"]["critical_precision_floor"] = 0.2
        with self.assertRaisesRegex(ValueError, "exactly 0.40"):
            v3_protocol.validate_v3_protocol(changed)

        changed = copy.deepcopy(self.payload)
        changed["selection"]["required_fold_count"] = 1
        with self.assertRaisesRegex(ValueError, "all three folds"):
            v3_protocol.validate_v3_protocol(changed)


if __name__ == "__main__":
    unittest.main()
