"""Read-only aggregate reporting for the completed V3-E2 embedding cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MANIFEST = "temp/v3/v3_e2_embedding_cache_manifest.json"
DEFAULT_CACHE_ROOT = "temp/v3/embeddings/minilm_bc57282_head256"
DEFAULT_SOURCE = "temp/s3/scientific.parquet"
ALLOWED_PARTITIONS = ("train", "validation")
KEY_COLUMNS = (
    "received_date",
    "product_family",
    "normalized_group_hash",
    "normalized_length",
    "partition_name",
)
CRITICAL_CLASS = "debt_credit_management"


def _read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3-E2 evidence: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("V3-E2 evidence must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    """Return an uppercase SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _safe_path(root: Path, relative: str) -> Path:
    """Resolve one project-relative evidence path without allowing escape."""

    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError("V3-E2 reporting paths must be project-relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("V3-E2 reporting path escapes the project root")
    return resolved


def _norm_summary(values: Any) -> dict[str, int | float]:
    """Return aggregate-only L2-norm diagnostics."""

    import numpy as np

    combined = np.concatenate(values)
    return {
        "rows": int(combined.size),
        "minimum": float(np.min(combined)),
        "p01": float(np.quantile(combined, 0.01)),
        "median": float(np.median(combined)),
        "mean": float(np.mean(combined)),
        "p99": float(np.quantile(combined, 0.99)),
        "maximum": float(np.max(combined)),
        "zero_vectors": int(np.count_nonzero(combined == 0.0)),
        "non_finite_vectors": int(np.count_nonzero(~np.isfinite(combined))),
    }


def audit_v3_e2_cache(
    project_root: str | Path = ".",
    *,
    manifest_path: str = DEFAULT_MANIFEST,
    cache_root: str = DEFAULT_CACHE_ROOT,
    source_path: str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    """Validate V3-E2 files and return aggregate diagnostics only."""

    import duckdb
    import numpy as np
    import pyarrow.parquet as pq

    from .protocol import validate_protocol_manifest

    root = Path(project_root).expanduser().resolve()
    manifest_file = _safe_path(root, manifest_path)
    cache_directory = _safe_path(root, cache_root)
    source_file = _safe_path(root, source_path)
    manifest = _read_json(manifest_file)
    protocol = validate_protocol_manifest(project_root=root)

    if manifest.get("stage") != "V3-E2":
        raise ValueError("V3-E2 manifest stage is invalid")
    if manifest.get("status") != "COMPLETE_DEVELOPMENT_CACHE":
        raise ValueError("V3-E2 cache is not complete")
    if manifest.get("complete") is not True:
        raise ValueError("V3-E2 completion flag is invalid")
    if manifest.get("selection_authorized") is not False:
        raise ValueError("V3-E2 may not authorize model selection")
    if tuple(manifest.get("scientific_partitions", ())) != ALLOWED_PARTITIONS:
        raise ValueError("V3-E2 manifest partition boundary is invalid")
    if manifest.get("forbidden_partition_access") is not False:
        raise ValueError("V3-E2 reports forbidden partition access")
    if manifest.get("individual_cache_publication") is not False:
        raise ValueError("V3-E2 individual cache may not be public")
    if manifest.get("raw_narrative_persistence") is not False:
        raise ValueError("V3-E2 may not persist raw narratives")
    if manifest.get("protocol_file_sha256") != protocol["protocol"]["sha256"]:
        raise ValueError("V3-E2 protocol hash differs from the frozen contract")
    if not cache_directory.is_dir():
        raise FileNotFoundError(f"V3-E2 local cache is unavailable: {cache_directory}")
    if list(cache_directory.parent.glob(f".{cache_directory.name}.stage-*")):
        raise ValueError("V3-E2 has an unfinished staging directory")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != manifest.get("shard_count"):
        raise ValueError("V3-E2 shard list differs from the manifest")

    ordered_digest = hashlib.sha256()
    total_rows = 0
    total_bytes = 0
    norm_groups: dict[str, list[Any]] = {
        "train": [],
        "validation": [],
        "critical": [],
        "background": [],
    }
    for expected_index, shard in enumerate(shards):
        if shard.get("shard_index") != expected_index:
            raise ValueError("V3-E2 shard sequence is invalid")
        embedding_file = _safe_path(
            cache_directory,
            str(shard.get("embedding_path", "")),
        )
        keys_file = _safe_path(
            cache_directory,
            str(shard.get("keys_path", "")),
        )
        if _sha256(embedding_file) != shard["embedding"]["sha256"]:
            raise ValueError(f"V3-E2 embedding hash differs: {expected_index}")
        if embedding_file.stat().st_size != shard["embedding"]["size_bytes"]:
            raise ValueError(f"V3-E2 embedding size differs: {expected_index}")
        if _sha256(keys_file) != shard["keys"]["sha256"]:
            raise ValueError(f"V3-E2 key hash differs: {expected_index}")
        if keys_file.stat().st_size != shard["keys"]["size_bytes"]:
            raise ValueError(f"V3-E2 key size differs: {expected_index}")

        embeddings = np.load(embedding_file, mmap_mode="r", allow_pickle=False)
        expected_rows = int(shard["row_count"])
        if embeddings.dtype != np.float16:
            raise ValueError(f"V3-E2 embedding dtype differs: {expected_index}")
        if embeddings.shape != (expected_rows, 384):
            raise ValueError(f"V3-E2 embedding shape differs: {expected_index}")
        if not np.isfinite(embeddings).all():
            raise ValueError(
                f"V3-E2 embedding contains non-finite values: {expected_index}"
            )

        keys = pq.read_table(keys_file)
        if tuple(keys.column_names) != KEY_COLUMNS:
            raise ValueError(f"V3-E2 key schema differs: {expected_index}")
        if keys.num_rows != expected_rows:
            raise ValueError(f"V3-E2 key row count differs: {expected_index}")
        for name in KEY_COLUMNS:
            for value in keys[name].to_pylist():
                ordered_digest.update(str(value).encode("utf-8"))
                ordered_digest.update(b"\x1f")
            ordered_digest.update(b"\x1e")

        norms = np.sqrt(
            np.sum(np.square(np.asarray(embeddings, dtype=np.float32)), axis=1)
        )
        partitions = np.asarray(keys["partition_name"].to_pylist(), dtype=object)
        products = np.asarray(keys["product_family"].to_pylist(), dtype=object)
        if not set(partitions).issubset(ALLOWED_PARTITIONS):
            raise ValueError("V3-E2 key shard contains a forbidden partition")
        for partition in ALLOWED_PARTITIONS:
            selected = norms[partitions == partition]
            if selected.size:
                norm_groups[partition].append(selected)
        norm_groups["critical"].append(norms[products == CRITICAL_CLASS])
        norm_groups["background"].append(norms[products != CRITICAL_CLASS])
        total_rows += expected_rows
        total_bytes += embedding_file.stat().st_size + keys_file.stat().st_size

    if total_rows != manifest.get("row_count"):
        raise ValueError("V3-E2 total rows differ from the manifest")
    if ordered_digest.hexdigest().upper() != manifest.get("ordered_key_sha256"):
        raise ValueError("V3-E2 ordered key hash differs from the manifest")

    cache_glob = str(cache_directory / "keys-*.parquet").replace("\\", "/")
    source_name = str(source_file).replace("\\", "/")
    with duckdb.connect() as connection:
        cache_rows, unique_groups = connection.execute(
            """
            SELECT count(*),
                   count(DISTINCT normalized_group_hash || ':' || normalized_length)
            FROM read_parquet(?)
            """,
            [cache_glob],
        ).fetchone()
        partition_rows = connection.execute(
            """
            SELECT partition_name, count(*)
            FROM read_parquet(?)
            GROUP BY partition_name
            ORDER BY partition_name
            """,
            [cache_glob],
        ).fetchall()
        missing_keys = connection.execute(
            """
            SELECT count(*) FROM (
                SELECT received_date, product_family, normalized_group_hash,
                       normalized_length, partition_name
                FROM read_parquet(?)
                WHERE partition_name IN ('train', 'validation')
                EXCEPT
                SELECT received_date, product_family, normalized_group_hash,
                       normalized_length, partition_name
                FROM read_parquet(?)
            )
            """,
            [source_name, cache_glob],
        ).fetchone()[0]
        extra_keys = connection.execute(
            """
            SELECT count(*) FROM (
                SELECT received_date, product_family, normalized_group_hash,
                       normalized_length, partition_name
                FROM read_parquet(?)
                EXCEPT
                SELECT received_date, product_family, normalized_group_hash,
                       normalized_length, partition_name
                FROM read_parquet(?)
                WHERE partition_name IN ('train', 'validation')
            )
            """,
            [cache_glob, source_name],
        ).fetchone()[0]

    if cache_rows != total_rows or unique_groups != total_rows:
        raise ValueError("V3-E2 cache group identity is not unique")
    if missing_keys or extra_keys:
        raise ValueError("V3-E2 cache keys differ from the development source")

    norms_by_partition = {
        partition: _norm_summary(norm_groups[partition])
        for partition in ALLOWED_PARTITIONS
    }
    norms_by_class_role = {
        role: _norm_summary(norm_groups[role])
        for role in ("critical", "background")
    }
    observed_unit_norm = all(
        summary["minimum"] >= 0.998 and summary["maximum"] <= 1.002
        for summary in norms_by_partition.values()
    )
    return {
        "schema_version": "v3-e2-read-only-audit-v1",
        "stage": "V3-E2",
        "status": "VALID_COMPLETE_DEVELOPMENT_CACHE",
        "predictive_metrics_available": False,
        "v3_e3_started": False,
        "selection_authorized": False,
        "model": dict(manifest["model"]),
        "rows": total_rows,
        "unique_groups": int(unique_groups),
        "partition_rows": {
            str(partition): int(rows) for partition, rows in partition_rows
        },
        "source_audit": dict(manifest["source_audit"]),
        "shards": len(shards),
        "cache_size_bytes": total_bytes,
        "ordered_key_sha256": manifest["ordered_key_sha256"],
        "manifest_sha256": _sha256(manifest_file),
        "protocol_file_sha256": manifest["protocol_file_sha256"],
        "missing_keys": int(missing_keys),
        "extra_keys": int(extra_keys),
        "raw_narratives_persisted": False,
        "forbidden_partitions_accessed": False,
        "runtime": dict(manifest["runtime"]),
        "norms_by_partition": norms_by_partition,
        "norms_by_class_role": norms_by_class_role,
        "observed_unit_norm_embeddings": observed_unit_norm,
    }


def _metric_line(name: str, summary: Mapping[str, Any]) -> str:
    """Render one compact L2-norm summary line."""

    return (
        f"  {name}: n={summary['rows']:,} | min={summary['minimum']:.4f} | "
        f"p01={summary['p01']:.4f} | median={summary['median']:.4f} | "
        f"mean={summary['mean']:.4f} | p99={summary['p99']:.4f} | "
        f"max={summary['maximum']:.4f} | zeros={summary['zero_vectors']}"
    )


def render_v3_e2_audit(
    audit: Mapping[str, Any],
    language: str = "pt-BR",
) -> str:
    """Render the aggregate audit in PT-BR or EN-US."""

    if language not in {"pt-BR", "en-US"}:
        raise ValueError(f"Unsupported V3-E2 report language: {language}")
    runtime = audit["runtime"]
    cache_mib = audit["cache_size_bytes"] / 1024**2
    if language == "pt-BR":
        labels = {
            "title": "V3-E2 — AUDITORIA DO CACHE DE EMBEDDINGS",
            "integrity": "INTEGRIDADE",
            "runtime": "EXECUÇÃO",
            "norms": "NORMAS L2 DOS EMBEDDINGS (DIAGNÓSTICO)",
            "meaning": "INTERPRETAÇÃO",
            "meaning_text": (
                "E2 comprova integridade e viabilidade operacional. Ainda não há "
                "métrica preditiva, threshold escolhido ou candidato selecionado."
            ),
            "normalization_text": (
                "As normas próximas de 1 confirmam a camada Normalize interna do "
                "checkpoint. normalize_embeddings=False apenas evita uma "
                "normalização adicional pela chamada encode."
            ),
        }
    else:
        labels = {
            "title": "V3-E2 — EMBEDDING CACHE AUDIT",
            "integrity": "INTEGRITY",
            "runtime": "EXECUTION",
            "norms": "EMBEDDING L2 NORMS (DIAGNOSTIC)",
            "meaning": "INTERPRETATION",
            "meaning_text": (
                "E2 establishes integrity and operational feasibility. No "
                "predictive metric, selected threshold, or candidate exists yet."
            ),
            "normalization_text": (
                "Norms near 1 confirm the checkpoint's internal Normalize layer. "
                "normalize_embeddings=False only prevents extra normalization in "
                "the encode call."
            ),
        }
    lines = [
        labels["title"],
        "",
        f"status: {audit['status']}",
        f"model: {audit['model']['model_id']}",
        f"revision: {audit['model']['revision']}",
        f"shape: {audit['rows']:,} x {audit['model']['embedding_dimensions']}",
        f"dtype: {audit['model']['embedding_dtype']}",
        "",
        labels["integrity"],
        f"  shards: {audit['shards']}",
        f"  unique_groups: {audit['unique_groups']:,}",
        f"  train_rows: {audit['partition_rows']['train']:,}",
        f"  validation_rows: {audit['partition_rows']['validation']:,}",
        f"  missing_keys: {audit['missing_keys']}",
        f"  extra_keys: {audit['extra_keys']}",
        f"  raw_narratives_persisted: {audit['raw_narratives_persisted']}",
        f"  forbidden_partitions_accessed: {audit['forbidden_partitions_accessed']}",
        f"  cache_size_mib: {cache_mib:.2f}",
        f"  ordered_key_sha256: {audit['ordered_key_sha256']}",
        f"  manifest_sha256: {audit['manifest_sha256']}",
        "",
        labels["runtime"],
        f"  elapsed_seconds: {runtime['elapsed_seconds']:.4f}",
        f"  rows_per_second: {runtime['rows_per_second']:.2f}",
        f"  batch_size: {runtime['batch_size']}",
        f"  device: {runtime['device']}",
        f"  peak_reserved_mib: {runtime['peak_reserved_mib']:.2f}",
        "",
        labels["norms"],
        _metric_line("train", audit["norms_by_partition"]["train"]),
        _metric_line("validation", audit["norms_by_partition"]["validation"]),
        _metric_line("critical", audit["norms_by_class_role"]["critical"]),
        _metric_line("background", audit["norms_by_class_role"]["background"]),
        f"  observed_unit_norm_embeddings: {audit['observed_unit_norm_embeddings']}",
        "",
        labels["meaning"],
        f"  {labels['meaning_text']}",
        f"  {labels['normalization_text']}",
    ]
    return "\n".join(lines)


__all__ = ["audit_v3_e2_cache", "render_v3_e2_audit"]
