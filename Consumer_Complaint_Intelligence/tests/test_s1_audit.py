"""Synthetic tests for S1 taxonomy and duplicate policies."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from consumer_complaint_intelligence.deduplication import audit_deduplication
from consumer_complaint_intelligence.deduplication import _combined_query
from consumer_complaint_intelligence.config import S0AuditConfig
from consumer_complaint_intelligence.s1 import audit_s1_corpus
from consumer_complaint_intelligence.s1 import audit_s1_taxonomy
from consumer_complaint_intelligence.s1 import load_or_run_s1
import consumer_complaint_intelligence.deduplication as deduplication_module
import consumer_complaint_intelligence.s1 as s1_module
from consumer_complaint_intelligence.taxonomy import PRODUCT_FAMILY_REGISTRY
from consumer_complaint_intelligence.taxonomy import TAXONOMY_VERSION
from consumer_complaint_intelligence.taxonomy import build_issue_key
from consumer_complaint_intelligence.taxonomy import map_product


class S1AuditTests(unittest.TestCase):
    """Cover pure mappings and full aggregate behavior on synthetic data."""

    def _write_frame(self, directory: str) -> Path:
        """Write a fixture with cross-year and cross-taxonomy duplicates."""

        frame = pl.DataFrame(
            {
                "Consumer complaint narrative": [
                    "Alpha",
                    " alpha  ",
                    "Alpha",
                    "Beta",
                    "Beta",
                    "Gamma",
                    "Gamma",
                    "",
                ],
                "Date received": [
                    "2020-01-01",
                    "2021-01-02",
                    "2022-01-03",
                    "2021-02-01",
                    "2022-02-02",
                    "2022-03-01",
                    "2023-03-02",
                    "2023-04-01",
                ],
                "Product": [
                    "Credit card",
                    "Prepaid card",
                    "Credit card",
                    "Mortgage",
                    "Debt collection",
                    "Unknown product",
                    "Unknown product",
                    "Credit card",
                ],
                "Issue": [
                    "Billing",
                    "Billing",
                    "Payment",
                    "Payment",
                    "Payment",
                    "X",
                    "Y",
                    "Billing",
                ],
                "Complaint ID": list(range(1, 9)),
            }
        )
        path = Path(directory) / "s1.parquet"
        frame.write_parquet(path)
        return path

    def test_all_registered_products_map_strictly(self) -> None:
        """Map every registered raw Product with the expected version."""

        self.assertEqual(len(PRODUCT_FAMILY_REGISTRY), 21)
        for label in PRODUCT_FAMILY_REGISTRY:
            result = map_product(label)
            self.assertEqual(result.mapping_status, "mapped")
            self.assertEqual(result.taxonomy_version, TAXONOMY_VERSION)

    def test_unknown_product_requires_explicit_audit_mode(self) -> None:
        """Reject unknown labels by default and expose audit unmapped status."""

        with self.assertRaises(ValueError):
            map_product("Unknown product")
        result = map_product("Unknown product", mode="unmapped")
        self.assertEqual(result.mapping_status, "unmapped")
        self.assertIsNone(result.family)

    def test_issue_key_is_hierarchical_and_raw_issue_is_preserved(self) -> None:
        """Build the family-plus-raw-Issue key without global Issue merging."""

        self.assertEqual(
            build_issue_key("credit_reporting", "Payment"),
            "credit_reporting :: Payment",
        )
        self.assertEqual(
            build_issue_key(None, None),
            "<UNMAPPED_FAMILY> :: <NULL>",
        )

    def test_s1_aggregates_and_serializes_without_narratives(self) -> None:
        """Return coverage, conflicts, duplicate levels, and a blocked gate."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_frame(directory)
            taxonomy = audit_s1_taxonomy(
                path,
                temp_directory=Path(directory) / "duckdb",
            )
            duplicates = audit_deduplication(
                path,
                temp_directory=Path(directory) / "duckdb",
            )
            report = audit_s1_corpus(
                path,
                temp_directory=Path(directory) / "duckdb",
            )

        self.assertEqual(taxonomy["taxonomy_version"], TAXONOMY_VERSION)
        self.assertEqual(taxonomy["mapping"]["summary"]["unmapped_rows"], 2)
        self.assertEqual(taxonomy["issue"]["distinct_raw_issue"], 4)
        self.assertGreaterEqual(
            len(taxonomy["issue"]["conflicts_across_families"]),
            1,
        )
        self.assertEqual(
            duplicates["summary_by_method"]["exact"]["groups_cross_date"],
            3,
        )
        self.assertGreater(
            duplicates["summary_by_method"]["normalized"][
                "groups_cross_year"
            ],
            0,
        )
        self.assertFalse(duplicates["scope"]["narratives_materialized_in_python"])
        self.assertEqual(report["gate"]["gate_status"], "BLOCKED")
        self.assertIn("taxonomy", report)
        self.assertNotIn("Alpha", json.dumps(report))
        json.dumps(report)

    def test_top_k_is_applied_independently_per_method(self) -> None:
        """Keep at most top-k exact and normalized groups separately."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_frame(directory)
            report = audit_deduplication(
                path,
                temp_directory=Path(directory) / "duckdb",
                config=S0AuditConfig(top_k=1),
            )

        methods = [row["method"] for row in report["top_groups"]]
        self.assertEqual(methods, ["exact", "normalized"])
        self.assertTrue(
            all(row["method_rank"] == 1 for row in report["top_groups"])
        )

    def test_materialized_dedup_projection_stores_hashes_only(self) -> None:
        """Keep full text outside every materialized deduplication CTE."""

        query = _combined_query(S0AuditConfig(top_k=1))
        prepared_start = query.index("prepared AS MATERIALIZED")
        fingerprints_start = query.index("fingerprints AS MATERIALIZED")
        prepared_sql = query[prepared_start:fingerprints_start]
        fingerprints_end = query.index("groups AS MATERIALIZED")
        fingerprints_sql = query[fingerprints_start:fingerprints_end]

        self.assertIn("exact_hash", prepared_sql)
        self.assertIn("normalized_hash", prepared_sql)
        self.assertIn("exact_length", prepared_sql)
        self.assertIn("normalized_length", prepared_sql)
        self.assertNotIn("AS raw_text", prepared_sql)
        self.assertNotIn("AS normalized_text", prepared_sql)
        self.assertNotIn("md5(", fingerprints_sql)

    def test_cache_hit_and_source_signature_invalidation(self) -> None:
        """Reuse a matching cache and recompute after source metadata changes."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_frame(directory)
            cache = Path(directory) / "s1_report.json"
            config = S0AuditConfig(top_k=2)
            first = load_or_run_s1(path, cache, config=config)
            second = load_or_run_s1(path, cache, config=config)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            third = load_or_run_s1(path, cache, config=config)
            cached_text = cache.read_text(encoding="utf-8")

        self.assertEqual(first["cache"]["status"], "miss")
        self.assertEqual(second["cache"]["status"], "hit")
        self.assertEqual(third["cache"]["status"], "invalidated")
        self.assertEqual(first["report"], second["report"])
        self.assertNotIn("Alpha", cached_text)
        self.assertEqual(
            Path(first["report"]["taxonomy"]["scope"]["temp_directory"])
            .parts[-2:],
            ("temp", "duckdb"),
        )
        self.assertEqual(
            Path(first["report"]["deduplication"]["scope"]["temp_directory"])
            .parts[-2:],
            ("temp", "duckdb"),
        )
        self.assertEqual(
            first["cache"]["schema_version"],
            "s1-report-cache-v2",
        )

    def test_component_cache_survives_deduplication_failure(self) -> None:
        """Reuse taxonomy after deduplication fails during a first run."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_frame(directory)
            cache = Path(directory) / "s1_report.json"
            original = s1_module.audit_deduplication

            def fail_deduplication(*args, **kwargs):
                """Raise a deterministic synthetic component failure."""

                raise RuntimeError("synthetic deduplication failure")

            s1_module.audit_deduplication = fail_deduplication
            try:
                with self.assertRaises(RuntimeError):
                    load_or_run_s1(path, cache)
            finally:
                s1_module.audit_deduplication = original

            taxonomy_cache = cache.with_name("s1_report_taxonomy.json")
            self.assertTrue(taxonomy_cache.exists())
            resumed = load_or_run_s1(path, cache)

        self.assertEqual(
            resumed["cache"]["components"]["taxonomy"]["status"],
            "hit",
        )

    def test_s1_components_use_one_source_call_each(self) -> None:
        """Use one source query call for taxonomy and one for deduplication."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_frame(directory)
            with patch.object(
                s1_module,
                "query_parquet_batch",
                wraps=s1_module.query_parquet_batch,
            ) as taxonomy_query:
                with patch.object(
                    deduplication_module,
                    "query_parquet",
                    wraps=deduplication_module.query_parquet,
                ) as dedup_query:
                    audit_s1_corpus(path, Path(directory) / "duckdb")

        self.assertEqual(taxonomy_query.call_count, 1)
        self.assertEqual(dedup_query.call_count, 1)
