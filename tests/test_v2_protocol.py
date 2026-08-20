"""Synthetic tests for the frozen V2 development protocol."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from consumer_complaint_intelligence import v2_protocol


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "v2_development_protocol.json"


class V2ProtocolTests(unittest.TestCase):
    """Verify V2 contract, partition, decision, and metric safeguards."""

    def test_frozen_protocol_loads_strictly(self) -> None:
        """Load the exact frozen contract and expose its boundaries."""

        protocol = v2_protocol.load_v2_protocol(CONFIG)
        self.assertEqual(protocol.allowed_partitions, ("train", "validation"))
        self.assertEqual(protocol.critical_class, "debt_credit_management")
        self.assertFalse(protocol.payload["objective"]["deployment_authorized"])

    def test_adultered_config_is_rejected(self) -> None:
        """Reject changed values and unexpected keys in the protocol."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["selection"]["scientific_gates"]["critical_f1_min"] = 0.27
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v2.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                v2_protocol.load_v2_protocol(path)

        extra = json.loads(CONFIG.read_text(encoding="utf-8"))
        extra["unexpected"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v2.json"
            path.write_text(json.dumps(extra), encoding="utf-8")
            with self.assertRaises(ValueError):
                v2_protocol.load_v2_protocol(path)

    def test_partition_guard_accepts_only_development_partitions(self) -> None:
        """Accept train and validation and reject every sealed partition."""

        self.assertEqual(
            v2_protocol.require_development_partition("train"), "train"
        )
        self.assertEqual(
            v2_protocol.require_development_partition("validation"),
            "validation",
        )
        for partition in ("test", "stress", "monitor"):
            with self.assertRaises(ValueError):
                v2_protocol.require_development_partition(partition)

    def test_binary_target_is_deterministic_and_exact(self) -> None:
        """Map only the critical class to one in stable input order."""

        labels = (
            "credit_reporting",
            "debt_credit_management",
            "debt_collection",
        )
        expected = (0, 1, 0)
        self.assertEqual(v2_protocol.make_binary_critical_target(labels), expected)
        self.assertEqual(v2_protocol.make_critical_target(labels), expected)
        with self.assertRaises(ValueError):
            v2_protocol.make_binary_critical_target(("unknown_family",))

    def test_hierarchical_override_and_fallback_are_explicit(self) -> None:
        """Override fallback labels only for true detector decisions."""

        result = v2_protocol.combine_detector_with_fallback(
            (True, False, False),
            ("credit_reporting", "debt_collection", "mortgage"),
        )
        self.assertEqual(
            result,
            ("debt_credit_management", "debt_collection", "mortgage"),
        )
        with self.assertRaises(ValueError):
            v2_protocol.combine_detector_with_fallback(
                (1, False), ("credit_reporting", "mortgage")
            )
        with self.assertRaises(ValueError):
            v2_protocol.combine_detector_with_fallback(
                (False,), ("unknown_family",)
            )

    def test_scientific_gates_use_inclusive_boundaries(self) -> None:
        """Pass values exactly at all three scientific thresholds."""

        protocol = v2_protocol.load_v2_protocol(CONFIG)
        metrics = {
            "macro_f1": 0.69,
            "critical_f1": 0.2715,
            "critical_precision": 0.2,
        }
        result = v2_protocol.calculate_scientific_gates(metrics, protocol)
        self.assertTrue(result["passed"])
        self.assertEqual(result["gate_count"], 3)

        below = copy.deepcopy(metrics)
        below["critical_f1"] -= 1e-12
        result = v2_protocol.calculate_scientific_gates(below, protocol)
        self.assertFalse(result["passed"])
        self.assertEqual(result["gate_count"], 2)
        with self.assertRaises(ValueError):
            v2_protocol.calculate_scientific_gates(
                {**metrics, "macro_f1": 1.01}, protocol
            )

    def test_safety_margins_require_the_stricter_three_limits(self) -> None:
        """Calculate headroom and require every development margin."""

        protocol = v2_protocol.load_v2_protocol(CONFIG)
        metrics = {
            "macro_f1": 0.70,
            "critical_f1": 0.29,
            "critical_precision": 0.22,
        }
        result = v2_protocol.calculate_safety_margins(metrics, protocol)
        self.assertTrue(result["passed"])
        self.assertEqual(result["gate_count"], 3)
        self.assertEqual(result["headroom"]["critical_f1"], 0.0)

        metrics["critical_precision"] = 0.219999
        result = v2_protocol.calculate_safety_margins(metrics, protocol)
        self.assertFalse(result["passed"])
        self.assertEqual(result["gate_count"], 2)

    def test_real_baseline_hashes_and_sizes_validate_without_dataset(self) -> None:
        """Validate frozen S8 and S7 artifacts without reading data files."""

        protocol = v2_protocol.load_v2_protocol(CONFIG)
        signatures = v2_protocol.validate_baseline_artifacts(protocol, ROOT)
        self.assertEqual(
            signatures["manifest"]["sha256"],
            protocol.payload["baseline_s8"]["manifest"]["sha256"],
        )
        self.assertEqual(
            signatures["result"]["size_bytes"],
            protocol.payload["baseline_s8"]["result"]["size_bytes"],
        )
        self.assertEqual(
            signatures["s7_bundle"]["sha256"],
            protocol.payload["architecture"]["stage_b_bundle"]["sha256"],
        )

    def test_hash_validation_is_optional_per_artifact_family(self) -> None:
        """Allow metadata-only checks for either S8 or S7 artifacts."""

        protocol = v2_protocol.load_v2_protocol(CONFIG)
        s8 = v2_protocol.validate_baseline_artifacts(
            protocol, ROOT, validate_s7=False
        )
        s7 = v2_protocol.validate_baseline_artifacts(
            protocol, ROOT, validate_s8=False
        )
        self.assertEqual(set(s8), {"manifest", "result"})
        self.assertEqual(set(s7), {"s7_bundle"})

    def test_expected_protocol_matches_config_file(self) -> None:
        """Keep the hard-coded contract byte-identical to the on-disk file."""

        on_disk = json.loads(CONFIG.read_text(encoding="utf-8"))
        normalized = json.loads(json.dumps(on_disk, sort_keys=True))
        self.assertEqual(normalized, v2_protocol._expected_protocol())

    def test_overlapping_or_unordered_windows_are_rejected(self) -> None:
        """Reject development windows that overlap or lose strict ordering."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))

        touching = copy.deepcopy(payload)
        touching["development_windows"]["inner_calibration"]["start"] = (
            touching["development_windows"]["inner_fit"]["end"]
        )
        with self.assertRaises(ValueError):
            v2_protocol.validate_v2_protocol(touching)

        unordered = copy.deepcopy(payload)
        unordered["development_windows"]["outer_evaluation"]["start"] = (
            "2024-08-01"
        )
        with self.assertRaises(ValueError):
            v2_protocol.validate_v2_protocol(unordered)

    def test_old_calibration_window_intersecting_fit_scope_is_rejected(
        self,
    ) -> None:
        """Reject the D1 root cause: calibrating inside the S7 fit scope."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        regressed = copy.deepcopy(payload)
        regressed["development_windows"] = {
            "inner_fit": {
                "partition": "train",
                "start": "2023-08-01",
                "end": "2024-04-30",
            },
            "inner_calibration": {
                "partition": "train",
                "start": "2024-05-01",
                "end": "2024-06-30",
            },
            "outer_evaluation": {
                "partition": "validation",
                "start": "2024-07-01",
                "end": "2024-12-31",
            },
        }
        with self.assertRaisesRegex(
            ValueError, "inner_calibration window intersects"
        ):
            v2_protocol.validate_v2_protocol(regressed)

        # The frozen V2.1 windows do not reuse the S7 fallback fit scope.
        v2_protocol.validate_v2_protocol(payload)


if __name__ == "__main__":
    unittest.main()
