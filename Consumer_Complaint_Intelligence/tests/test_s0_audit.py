"""Tests for sample and full-corpus S0 audit behavior."""

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from consumer_complaint_intelligence.audit import audit_s0_corpus
from consumer_complaint_intelligence.audit import audit_s0_sample
from consumer_complaint_intelligence.config import S0AuditConfig
from consumer_complaint_intelligence.data import query_parquet
from consumer_complaint_intelligence.data import read_parquet_sample


class S0AuditTests(unittest.TestCase):
    """Exercise sample and aggregate outputs on small synthetic Parquet data."""

    def _write_frame(self, directory: str) -> Path:
        """Write a deterministic corpus fixture and return its path."""

        frame = pl.DataFrame(
            {
                "Consumer complaint narrative": [
                    "Alpha",
                    "Alpha",
                    "alpha",
                    "Beta",
                    "Beta",
                    "Gamma",
                ],
                "Date received": [
                    "2020-01-01",
                    "2020-01-02",
                    "2021-01-01",
                    "2021-01-02",
                    "2022-01-01",
                    "2022-01-02",
                ],
                "Product": ["Card", "Card", "Card", "Loan", "Loan", "Loan"],
                "Issue": [
                    "Billing",
                    "Billing",
                    "Billing",
                    "Payment",
                    "Payment",
                    None,
                ],
                "Complaint ID": [1, 2, 2, 3, 4, 5],
            }
        )
        path = Path(directory) / "sample.parquet"
        frame.write_parquet(path)
        return path

    def test_sample_audit_is_explicitly_bounded(self) -> None:
        """Label head-sample evidence and separate total groups from previews."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_frame(directory)
            report = audit_s0_sample(
                path,
                S0AuditConfig(sample_rows=3, top_k=5),
            )

        self.assertEqual(report["audit"], "S0_bounded_sample")
        self.assertEqual(report["scope"]["evidence"], "sample_only")
        self.assertIn("total_duplicate_groups", report["duplicates"])
        self.assertIn("top_duplicate_groups", report["duplicates"])

    def test_full_audit_aggregates_volume_taxonomy_ids_and_duplicates(self) -> None:
        """Return corpus aggregates without exposing complete narratives."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_frame(directory)
            report = audit_s0_corpus(
                path,
                temp_directory=Path(directory) / "duckdb",
            )

        self.assertEqual(report["audit"], "S0_full_corpus_aggregate")
        self.assertEqual(report["scope"]["evidence"], "complete_parquet_corpus")
        self.assertEqual(report["volume"]["total_rows"], 6)
        self.assertEqual(report["volume"]["narrative_rows"], 6)
        self.assertEqual(len(report["volume"]["coverage_by_year"]), 3)
        self.assertEqual(report["taxonomy"]["product"]["number_years"], 3)
        self.assertEqual(report["taxonomy"]["issue"]["distinct_labels"], 2)
        self.assertEqual(report["taxonomy"]["issue"]["missing_label_rows"], 1)
        self.assertEqual(
            report["taxonomy"]["issue"]["distinct_buckets_including_missing"],
            3,
        )
        self.assertIn(
            {"label": "<NULL>", "is_missing": True},
            [
                {"label": row["label"], "is_missing": row["is_missing"]}
                for row in report["taxonomy"]["issue"]["counts_by_year_label"]
            ],
        )
        self.assertEqual(report["complaint_id"]["distinct_non_null_ids"], 5)
        self.assertEqual(report["complaint_id"]["redundant_non_null_id_rows"], 1)
        duplicates = report["exact_narrative_duplicates"]
        self.assertEqual(duplicates["total_duplicate_groups"], 2)
        self.assertEqual(duplicates["duplicate_group_rows"], 4)
        self.assertEqual(duplicates["redundant_rows"], 2)
        self.assertFalse(report["scope"]["narratives_materialized_in_python"])
        self.assertNotIn("Alpha", str(report))
        json.dumps(report)

    def test_sample_reader_rejects_unbounded_arguments(self) -> None:
        """Reject a non-positive limit before touching the source."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            pl.DataFrame({"value": [1]}).write_parquet(path)
            with self.assertRaises(ValueError):
                read_parquet_sample(path, limit=0)

    def test_duckdb_query_binds_the_parquet_path(self) -> None:
        """Run a bounded DuckDB query without embedding a filesystem path."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            pl.DataFrame({"value": [1, 2, 3]}).write_parquet(path)
            rows = query_parquet(
                path,
                "SELECT count(*) AS rows FROM read_parquet(?)",
            )
        self.assertEqual(rows, [{"rows": 3}])
