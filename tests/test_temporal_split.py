"""Synthetic tests for S2 temporal protocols and modeling-index leakage."""

import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from consumer_complaint_intelligence.taxonomy import PRODUCT_FAMILY_REGISTRY
from consumer_complaint_intelligence.config import S0AuditConfig
from consumer_complaint_intelligence.temporal_split import DateWindow
from consumer_complaint_intelligence.temporal_split import TemporalCandidate
from consumer_complaint_intelligence.temporal_split import TemporalCriteria
from consumer_complaint_intelligence.temporal_split import audit_temporal_candidates
from consumer_complaint_intelligence.temporal_split import build_candidates
from consumer_complaint_intelligence.temporal_split import (
    build_or_load_modeling_index,
)
from consumer_complaint_intelligence.temporal_split import build_criteria_scenarios
from consumer_complaint_intelligence.temporal_split import _group_identity
from consumer_complaint_intelligence.temporal_split import load_or_run_s2


class S2TemporalSplitTests(unittest.TestCase):
    """Verify S2 contracts without reading the full CFPB corpus."""

    def _write_source(self, directory: Path) -> Path:
        """Write one bounded source with all modeled families and one rare row."""

        labels = {}
        for label, family in PRODUCT_FAMILY_REGISTRY.items():
            labels.setdefault(family, label)
        rows = []
        complaint_id = 1
        for year in range(2015, 2027):
            for family in sorted(labels):
                if family == "other_financial_services":
                    continue
                rows.append(
                    {
                        "Complaint ID": complaint_id,
                        "Date received": f"{year}-01-15",
                        "Product": labels[family],
                        "Issue": "Synthetic issue",
                        "Consumer complaint narrative": (
                            f"unique {family} {year} {complaint_id}"
                        ),
                    }
                )
                complaint_id += 1
        for extra in range(20):
            rows.append(
                {
                    "Complaint ID": complaint_id,
                    "Date received": "2025-02-01",
                    "Product": labels["credit_reporting"],
                    "Issue": "Synthetic issue",
                    "Consumer complaint narrative": f"stress {extra}",
                }
            )
            complaint_id += 1
        rows.extend(
            [
                {
                    "Complaint ID": complaint_id,
                    "Date received": "2023-09-01",
                    "Product": labels["other_financial_services"],
                    "Issue": "Synthetic rare",
                    "Consumer complaint narrative": "rare text",
                },
                {
                    "Complaint ID": complaint_id + 1,
                    "Date received": "2024-02-01",
                    "Product": labels["credit_reporting"],
                    "Issue": "Synthetic issue",
                    "Consumer complaint narrative": " same   repeated ",
                },
                {
                    "Complaint ID": complaint_id + 2,
                    "Date received": "2024-02-02",
                    "Product": labels["credit_reporting"],
                    "Issue": "Synthetic issue",
                    "Consumer complaint narrative": "same repeated",
                },
            ]
        )
        for row in rows:
            row["Alternate narrative"] = row[
                "Consumer complaint narrative"
            ]
        path = directory / "source.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def test_invalid_interval_and_overlap_are_rejected(self) -> None:
        """Reject inverted windows and candidate overlap."""

        with self.assertRaises(ValueError):
            DateWindow("bad", date(2024, 2, 1), date(2024, 1, 1))
        with self.assertRaises(ValueError):
            TemporalCandidate(
                "overlap",
                (
                    DateWindow("train", date(2020, 1, 1), date(2021, 1, 1)),
                    DateWindow("validation", date(2021, 1, 1), date(2021, 2, 1)),
                    DateWindow("test", date(2021, 3, 1), date(2021, 4, 1)),
                ),
            )
        with self.assertRaises(ValueError):
            TemporalCandidate(
                "unordered",
                (
                    DateWindow("train", date(2020, 1, 1), date(2020, 12, 31)),
                    DateWindow("test", date(2021, 3, 1), date(2021, 4, 1)),
                    DateWindow("validation", date(2021, 1, 1), date(2021, 2, 1)),
                ),
            )

    def test_same_hash_with_different_lengths_is_a_distinct_group(self) -> None:
        """Require hash and normalized length for group identity."""

        self.assertNotEqual(_group_identity("same-hash", 10), ("same-hash", 11))

    def test_index_schema_excludes_narrative_and_cache_invalidates(self) -> None:
        """Keep narrative out of the index and refresh after source changes."""

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write_source(directory)
            index = directory / "modeling_index.parquet"
            first = build_or_load_modeling_index(source, index)
            second = build_or_load_modeling_index(source, index)
            self.assertEqual(second["status"], "hit")
            self.assertNotIn("Consumer complaint narrative", first["columns"])
            self.assertNotIn("normalized_text", first["columns"])
            os.utime(source, None)
            third = build_or_load_modeling_index(source, index)
            self.assertEqual(third["status"], "refreshed")
            changed_config = build_or_load_modeling_index(
                source,
                index,
                config=S0AuditConfig(text_column="Alternate narrative"),
            )
            self.assertEqual(changed_config["status"], "refreshed")

    def test_partition_assignment_rare_policy_and_novel_purge(self) -> None:
        """Report rare rows, within-partition repeats, and prior-group purges."""

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write_source(directory)
            index = directory / "modeling_index.parquet"
            build_or_load_modeling_index(source, index)
            criteria = TemporalCriteria(
                min_train_rows=1,
                min_train_unique_groups=1,
                min_validation_novel_unique_groups=1,
                min_test_novel_unique_groups=1,
                max_largest_family_share=0.5,
            )
            report = audit_temporal_candidates(
                index,
                criteria=criteria,
                temp_directory=directory / "duckdb",
            )
            self.assertEqual(
                report["recommendation_status"], "READY_FOR_REVIEW"
            )
            self.assertEqual(
                report["recommended_candidate"], "historical_stress"
            )
            historical = next(
                item
                for item in report["candidates"]
                if item["candidate"]["name"] == "historical_stress"
            )
            test_partition = next(
                item
                for item in historical["partitions"]
                if item["partition"] == "test"
            )
            self.assertGreaterEqual(
                test_partition["repeated_within_partition_rows"], 1
            )
            self.assertGreaterEqual(test_partition["out_of_scope_rare_rows"], 0)
            self.assertTrue(
                historical["criterion_results"][
                    "largest_family_share_limit"
                ]
            )
            self.assertEqual(
                historical["min_novel_unique_groups_test"],
                1,
            )
            self.assertEqual(historical["eligible_class_count"], 9)
            json.dumps(report)

    def test_s2_report_cache_is_serializable_and_incremental(self) -> None:
        """Reuse a matching report cache without model training."""

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write_source(directory)
            index = directory / "modeling_index.parquet"
            report_path = directory / "s2_report.json"
            first = load_or_run_s2(source, index, report_path)
            second = load_or_run_s2(source, index, report_path)
            self.assertEqual(first["cache"]["status"], "refreshed")
            self.assertEqual(second["cache"]["status"], "hit")
            self.assertTrue(second["report"]["status_is_not_sealed"])
            json.dumps(second)

    def test_candidate_catalog_is_explicit_and_ordered(self) -> None:
        """Keep all three requested candidate names and inclusive limits."""

        candidates = build_candidates()
        self.assertEqual(
            [candidate.name for candidate in candidates],
            [
                "historical_stress",
                "post_2023_taxonomy",
                "extended_history",
            ],
        )
        for candidate in candidates:
            self.assertTrue(candidate.to_dict()["limits_are_inclusive"])

    def test_criteria_scenarios_keep_strict_and_pilot_distinct(self) -> None:
        """Expose strict defaults and exploratory pilot thresholds explicitly."""

        scenarios = build_criteria_scenarios()
        self.assertEqual(set(scenarios), {"strict", "pilot"})
        self.assertEqual(scenarios["strict"], TemporalCriteria())
        self.assertEqual(scenarios["pilot"].min_train_rows, 750)
        self.assertEqual(scenarios["pilot"].min_train_unique_groups, 750)
        self.assertEqual(
            scenarios["pilot"].min_validation_novel_unique_groups,
            500,
        )
        self.assertEqual(scenarios["pilot"].min_test_novel_unique_groups, 500)
        self.assertEqual(scenarios["pilot"].max_largest_family_share, 0.80)


if __name__ == "__main__":
    unittest.main()
