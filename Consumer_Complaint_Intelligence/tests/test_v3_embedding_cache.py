"""Tests for the development-only V3-E2 embedding cache contract."""

import copy
import json
import unittest
from pathlib import Path

import duckdb

from consumer_complaint_intelligence.v3 import embedding_cache


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "v3" / "embedding_cache.json"
SOURCE = ROOT / "temp" / "s3" / "scientific.parquet"


class V3EmbeddingCacheTests(unittest.TestCase):
    """Verify E2 partition, model, support, and publication safeguards."""

    def setUp(self) -> None:
        """Load one detached frozen cache configuration."""

        self.payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_config_loads_with_pinned_model_and_batches(self) -> None:
        """Expose the exact E2 shard and CUDA-batch controls."""

        config = embedding_cache.load_embedding_cache_config(CONFIG)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.shard_rows, 32768)
        self.assertEqual(
            config.payload["model"]["revision"],
            embedding_cache.MODEL_REVISION,
        )

    def test_only_train_and_validation_may_enter_the_cache(self) -> None:
        """Reject any attempt to add a consumed or sealed partition."""

        changed = copy.deepcopy(self.payload)
        changed["source"]["allowed_partitions"].append("monitor")
        with self.assertRaisesRegex(ValueError, "train and validation only"):
            embedding_cache.validate_embedding_cache_config(changed)

    def test_individual_cache_and_raw_narrative_publication_are_forbidden(
        self,
    ) -> None:
        """Keep shard-level scientific data local and ignored."""

        for key in ("individual_cache_publication", "raw_narrative_persistence"):
            changed = copy.deepcopy(self.payload)
            changed["output"][key] = True
            with self.assertRaises(ValueError):
                embedding_cache.validate_embedding_cache_config(changed)

    def test_source_query_has_a_fixed_development_filter_and_sort(self) -> None:
        """Keep the streaming query deterministic and development-only."""

        query = embedding_cache._source_query().lower()
        self.assertIn("partition_name in ('train', 'validation')", query)
        self.assertIn("order by", query)
        for forbidden in ("'test'", "'stress'", "'monitor'"):
            self.assertNotIn(forbidden, query)

    @unittest.skipUnless(
        SOURCE.exists(),
        "the local scientific cache is private and is not part of the repository",
    )
    def test_real_source_aggregate_support_and_group_uniqueness_match(self) -> None:
        """Audit metadata only, without loading or persisting narratives."""

        config = embedding_cache.load_embedding_cache_config(CONFIG)
        with duckdb.connect() as connection:
            audit = embedding_cache._audit_source(connection, SOURCE)
        embedding_cache._validate_source_audit(audit, config)
        self.assertEqual(audit["train"]["rows"], 345552)
        self.assertEqual(audit["validation"]["critical_rows"], 892)
        self.assertEqual(audit["train"]["groups"], audit["train"]["rows"])

    def test_support_drift_or_duplicate_groups_are_rejected(self) -> None:
        """Fail rather than silently cache a changed scientific population."""

        config = embedding_cache.load_embedding_cache_config(CONFIG)
        audit = {
            "train": {"rows": 345552, "critical_rows": 946, "groups": 345552},
            "validation": {
                "rows": 245980,
                "critical_rows": 892,
                "groups": 245980,
            },
        }
        changed = copy.deepcopy(audit)
        changed["train"]["groups"] -= 1
        with self.assertRaisesRegex(ValueError, "not unique"):
            embedding_cache._validate_source_audit(changed, config)

        changed = copy.deepcopy(audit)
        changed["validation"]["critical_rows"] += 1
        with self.assertRaisesRegex(ValueError, "critical support"):
            embedding_cache._validate_source_audit(changed, config)

    def test_cli_is_disabled_without_explicit_run_gate(self) -> None:
        """Prevent imports or an empty command from starting E2."""

        with self.assertRaisesRegex(SystemExit, "disabled by default"):
            embedding_cache.main([])


if __name__ == "__main__":
    unittest.main()
