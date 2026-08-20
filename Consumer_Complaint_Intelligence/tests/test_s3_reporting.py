"""Test the framework-neutral S3 evidence reporting API."""

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from consumer_complaint_intelligence.s3_reporting import (
    S3EvidenceTables,
    build_s3_evidence_tables,
    load_s3_evidence_tables,
)


def _valid_payload() -> dict:
    """Return a small completed S3 artifact payload for tests."""

    per_class = {
        "z_family": {
            "precision": 0.4,
            "recall": 0.5,
            "f1": 0.444,
            "support": 4,
        },
        "debt_credit_management": {
            "precision": 0.6,
            "recall": 0.7,
            "f1": 0.646,
            "support": 7,
        },
    }
    return {
        "complete": True,
        "points": {
            "1.0": {
                "train_groups": 100,
                "sgd_logistic": {
                    "scientific": {
                        "macro_f1": 0.7,
                        "balanced_accuracy": 0.75,
                        "per_class": per_class,
                    },
                    "operational_all_text": {"macro_f1": 0.68},
                },
            },
            "0.25": {
                "train_groups": 25,
                "sgd_logistic": {
                    "scientific": {
                        "macro_f1": 0.6,
                        "balanced_accuracy": 0.65,
                        "per_class": per_class,
                    },
                    "operational_all_text": {},
                },
            },
        },
    }


class S3ReportingTests(unittest.TestCase):
    """Verify S3 evidence validation and table construction."""

    def test_build_returns_sorted_typed_tables(self) -> None:
        """Build the required curve and final per-class columns."""

        evidence = build_s3_evidence_tables(_valid_payload())

        self.assertIsInstance(evidence, S3EvidenceTables)
        self.assertIsInstance(evidence.curve, pl.DataFrame)
        self.assertEqual(
            evidence.curve.columns,
            [
                "fraction",
                "train_groups",
                "macro_f1",
                "balanced_accuracy",
                "debt_credit_management_f1",
                "operational_macro_f1",
            ],
        )
        self.assertEqual(evidence.curve["fraction"].to_list(), [0.25, 1.0])
        self.assertEqual(
            evidence.per_class.columns,
            ["product_family", "precision", "recall", "f1", "support"],
        )
        self.assertEqual(
            evidence.per_class["product_family"].to_list(),
            ["debt_credit_management", "z_family"],
        )

    def test_load_reads_json_artifact(self) -> None:
        """Load a valid JSON artifact without accessing the dataset."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s3_full.json"
            path.write_text(json.dumps(_valid_payload()), encoding="utf-8")

            evidence = load_s3_evidence_tables(path)

        self.assertEqual(evidence.curve.height, 2)
        self.assertEqual(evidence.per_class.height, 2)

    def test_missing_artifact_raises_clear_error(self) -> None:
        """Reject a missing artifact path."""

        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_s3_evidence_tables("missing-s3-artifact.json")

    def test_invalid_json_raises_clear_error(self) -> None:
        """Reject malformed JSON with an artifact-specific error."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{invalid", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid S3 evidence JSON"):
                load_s3_evidence_tables(path)

    def test_incomplete_artifact_is_rejected(self) -> None:
        """Reject artifacts that are not marked complete."""

        payload = _valid_payload()
        payload["complete"] = False

        with self.assertRaisesRegex(ValueError, "complete=True"):
            build_s3_evidence_tables(payload)

    def test_empty_points_are_rejected(self) -> None:
        """Reject completed artifacts without learning-curve points."""

        payload = {"complete": True, "points": {}}

        with self.assertRaisesRegex(ValueError, "non-empty points"):
            build_s3_evidence_tables(payload)

    def test_required_model_section_is_rejected(self) -> None:
        """Reject a point without the required SGD logistic section."""

        payload = _valid_payload()
        del payload["points"]["1.0"]["sgd_logistic"]

        with self.assertRaisesRegex(ValueError, "sgd_logistic"):
            build_s3_evidence_tables(payload)


if __name__ == "__main__":
    unittest.main()
