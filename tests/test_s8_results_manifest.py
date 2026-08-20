"""Synthetic and pre-full tests for the S8 public result manifest."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from consumer_complaint_intelligence import s8
from consumer_complaint_intelligence import s8_reporting
from consumer_complaint_intelligence.s8 import MODELED_FAMILIES


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "s8_confirmatory_protocol.json"
RESULT = ROOT / "temp" / "s8" / "s8_results.json"
MANIFEST = ROOT / "config" / "s8_results.json"


def _synthetic_complete_result(config: s8.S8Config) -> dict[str, object]:
    """Build a complete aggregate result without dataset access."""

    signature = s8._signature(CONFIG, config.source, config.s7_freeze)
    matrix = np.eye(len(MODELED_FAMILIES), dtype=int)
    metrics = s8.metrics_from_confusion(matrix)
    gates = s8.evaluate_gates(metrics, config.gates)
    result = s8._base_result(
        signature, config, {"s2": config.payload["s2_evidence"]}
    )
    result["scope_counts"] = {"test_all_text": 9}
    result["primary"] = {
        "view": "scientific_primary",
        "metrics": metrics,
        "support_all_nine_classes": True,
        "support_by_class": {label: 1 for label in MODELED_FAMILIES},
        "gates": gates,
        "confidence_intervals": s8.bootstrap_confidence_intervals(
            matrix, replicates=2
        ),
    }
    result["operational_secondary"] = {
        "view": "operational_secondary",
        "metrics": metrics,
        "excluded_from_decision": True,
    }
    result["confirmed"] = bool(gates["passed"])
    result["status"] = (
        "CONFIRMED_FOR_STRESS_EVALUATION"
        if result["confirmed"]
        else "NOT_CONFIRMED"
    )
    result["decision"] = {
        "scientific_view": "primary",
        "gate_count": gates["gate_count"],
        "required_gate_count": len(gates["checks"]),
        "status": result["status"],
        "deploy": False,
    }
    result["complete"] = True
    return result


class S8ManifestTests(unittest.TestCase):
    """Keep the manifest optional before full and strict after publication."""

    def test_manifest_is_tolerant_before_full_and_strict_after(self) -> None:
        """Validate both the absent pre-full state and any full artifact."""

        if not RESULT.exists() and not MANIFEST.exists():
            return
        self.assertTrue(RESULT.exists())
        self.assertTrue(MANIFEST.exists())
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        s8.validate_s8_manifest(manifest, RESULT, CONFIG)
        self.assertIn(
            manifest["status"],
            {"CONFIRMED_FOR_STRESS_EVALUATION", "NOT_CONFIRMED"},
        )
        expected_status = (
            "CONFIRMED_FOR_STRESS_EVALUATION"
            if manifest["confirmed"]
            else "NOT_CONFIRMED"
        )
        self.assertEqual(manifest["status"], expected_status)
        self.assertEqual(manifest["decision"]["status"], expected_status)
        self.assertFalse(manifest["deploy"])

    def test_synthetic_manifest_rejects_tamper_and_absolute_paths(self) -> None:
        """Reject altered hashes, gates, and non-portable paths."""

        config = s8.load_s8_config(CONFIG)
        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as temporary:
            directory = Path(temporary)
            result_path = directory / "s8_results.json"
            manifest_path = directory / "s8_manifest.json"
            result = _synthetic_complete_result(config)
            result_path.write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8"
            )
            s8._publish_manifest(
                result_path, manifest_path, CONFIG, config, result
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            s8.validate_s8_manifest(manifest, result_path, CONFIG)
            tables = s8_reporting.load_s8_report_tables(
                result_path,
                manifest_path=manifest_path,
                config_path=CONFIG,
            )
            self.assertEqual(tables.primary_summary.height, 1)

            tampered_hash = json.loads(json.dumps(manifest))
            tampered_hash["result"]["sha256"] = "tampered"
            with self.assertRaises(ValueError):
                s8.validate_s8_manifest(tampered_hash, result_path, CONFIG)

            tampered_path = json.loads(json.dumps(manifest))
            tampered_path["protocol"]["path"] = "D:/absolute/config.json"
            with self.assertRaises(ValueError):
                s8.validate_s8_manifest(tampered_path, result_path, CONFIG)

            tampered_gate = json.loads(json.dumps(manifest))
            tampered_gate["decision"]["gate_count"] = 2
            with self.assertRaises(ValueError):
                s8.validate_s8_manifest(tampered_gate, result_path, CONFIG)

            manifest_path.write_text(
                json.dumps(tampered_hash), encoding="utf-8"
            )
            repaired = s8._cached_result(
                result_path,
                manifest_path,
                result["signature"],
                config_path=CONFIG,
            )
            self.assertEqual(repaired, result)
            repaired_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            s8.validate_s8_manifest(repaired_manifest, result_path, CONFIG)


if __name__ == "__main__":
    unittest.main()
