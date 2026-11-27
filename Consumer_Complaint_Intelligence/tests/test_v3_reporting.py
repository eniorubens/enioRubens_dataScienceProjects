"""Tests for the read-only V3-E2 notebook reporting boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from consumer_complaint_intelligence.v3.reporting import (
    audit_v3_e2_cache,
    render_v3_e2_audit,
)


ROOT = Path(__file__).parents[1]
CACHE_ROOT = ROOT / "temp" / "v3" / "embeddings" / "minilm_bc57282_head256"


def _fake_audit() -> dict:
    """Return a compact aggregate payload for renderer tests."""

    summary = {
        "rows": 10,
        "minimum": 1.0,
        "p01": 1.1,
        "median": 2.0,
        "mean": 2.1,
        "p99": 3.0,
        "maximum": 3.1,
        "zero_vectors": 0,
        "non_finite_vectors": 0,
    }
    return {
        "status": "VALID_COMPLETE_DEVELOPMENT_CACHE",
        "model": {
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "bc57282",
            "embedding_dimensions": 384,
            "embedding_dtype": "float16",
        },
        "rows": 20,
        "unique_groups": 20,
        "partition_rows": {"train": 10, "validation": 10},
        "shards": 2,
        "cache_size_bytes": 1024,
        "ordered_key_sha256": "A" * 64,
        "manifest_sha256": "B" * 64,
        "missing_keys": 0,
        "extra_keys": 0,
        "raw_narratives_persisted": False,
        "forbidden_partitions_accessed": False,
        "runtime": {
            "elapsed_seconds": 1.0,
            "rows_per_second": 20.0,
            "batch_size": 128,
            "device": "fake-gpu",
            "peak_reserved_mib": 1.0,
        },
        "norms_by_partition": {"train": summary, "validation": summary},
        "norms_by_class_role": {"critical": summary, "background": summary},
        "observed_unit_norm_embeddings": True,
    }


def test_renderer_is_bilingual_and_does_not_claim_predictive_results() -> None:
    """Localize visible prose while preserving the E2 scientific boundary."""

    pt_report = render_v3_e2_audit(_fake_audit(), "pt-BR")
    en_report = render_v3_e2_audit(_fake_audit(), "en-US")
    assert "AUDITORIA DO CACHE" in pt_report
    assert "Ainda não há métrica preditiva" in pt_report
    assert "camada Normalize interna" in pt_report
    assert "EMBEDDING CACHE AUDIT" in en_report
    assert "No predictive metric" in en_report
    assert "selected threshold" in en_report
    assert "internal Normalize layer" in en_report


def test_renderer_rejects_an_unsupported_language() -> None:
    """Fail instead of silently publishing an unintended localization."""

    with pytest.raises(ValueError, match="language"):
        render_v3_e2_audit(_fake_audit(), "es-ES")


def test_notebook_pair_has_equivalent_structure_and_bounded_code() -> None:
    """Keep the PT-BR and EN-US notebooks thin and structurally equivalent."""

    paths = (
        ROOT / "notebooks" / "pt-BR" / "14_V3_E2_Embedding_Cache_Audit_PT.ipynb",
        ROOT / "notebooks" / "en-US" / "14_V3_E2_Embedding_Cache_Audit_EN.ipynb",
    )
    notebooks = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert [cell["cell_type"] for cell in notebooks[0]["cells"]] == [
        cell["cell_type"] for cell in notebooks[1]["cells"]
    ]
    pt_code = [
        "".join(cell["source"])
        for cell in notebooks[0]["cells"]
        if cell["cell_type"] == "code"
    ]
    en_code = [
        "".join(cell["source"])
        for cell in notebooks[1]["cells"]
        if cell["cell_type"] == "code"
    ]
    assert len(pt_code) == len(en_code) == 2
    assert pt_code[0] == en_code[0]
    assert pt_code[1].replace("pt-BR", "LANG") == en_code[1].replace(
        "en-US", "LANG"
    )
    code = "\n".join(pt_code + en_code)
    assert "audit_v3_e2_cache" in code
    assert "SentenceTransformer" not in code
    assert "LinearSVC" not in code
    assert "read_parquet" not in code
    assert "narrative" not in code.lower()


@pytest.mark.skipif(not CACHE_ROOT.is_dir(), reason="local V3-E2 cache unavailable")
def test_completed_local_cache_passes_the_read_only_audit() -> None:
    """Exercise the complete local cache when the ignored shards are present."""

    audit = audit_v3_e2_cache(ROOT)
    assert audit["status"] == "VALID_COMPLETE_DEVELOPMENT_CACHE"
    assert audit["rows"] == 591_532
    assert audit["unique_groups"] == 591_532
    assert audit["missing_keys"] == 0
    assert audit["extra_keys"] == 0
    assert audit["predictive_metrics_available"] is False
    assert audit["v3_e3_started"] is False
    assert audit["observed_unit_norm_embeddings"] is True
    for summary in audit["norms_by_partition"].values():
        assert summary["zero_vectors"] == 0
        assert summary["non_finite_vectors"] == 0
