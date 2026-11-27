"""V3-E2 deterministic MiniLM cache restricted to development partitions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .protocol import (
    ALLOWED_PARTITIONS,
    CRITICAL_CLASS,
    validate_protocol_manifest,
)


DEFAULT_CONFIG = "config/v3/embedding_cache.json"
CONFIG_SCHEMA = "v3-e2-embedding-cache-v1"
MANIFEST_SCHEMA = "v3-e2-embedding-cache-manifest-v1"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "bc57282bc374d33e0d6c4de27f12dc1c2a87f37a"
KEY_COLUMNS = (
    "received_date",
    "product_family",
    "normalized_group_hash",
    "normalized_length",
    "partition_name",
)


def _exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    """Require one mapping with exactly the expected fields."""

    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"V3-E2 {name} keys differ from the contract")
    return value


def _safe_project_path(root: Path, value: str) -> Path:
    """Resolve one project-relative path without allowing traversal."""

    candidate = Path(value)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError("V3-E2 paths must be project-relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("V3-E2 path escapes the project root")
    return resolved


@dataclass(frozen=True, slots=True)
class EmbeddingCacheConfig:
    """Validated frozen V3-E2 cache configuration."""

    payload: Mapping[str, Any]

    @property
    def batch_size(self) -> int:
        """Return the smoke-qualified CUDA batch size."""

        return int(self.payload["execution"]["batch_size"])

    @property
    def shard_rows(self) -> int:
        """Return the maximum rows written to one shard."""

        return int(self.payload["execution"]["shard_rows"])


def validate_embedding_cache_config(
    payload: Mapping[str, Any],
) -> EmbeddingCacheConfig:
    """Validate the complete V3-E2 data, model, and output boundary."""

    block = _exact_keys(
        payload,
        {
            "schema_version",
            "stage",
            "status",
            "protocol_manifest",
            "source",
            "model",
            "execution",
            "output",
        },
        "top-level",
    )
    if block["schema_version"] != CONFIG_SCHEMA or block["stage"] != "V3-E2":
        raise ValueError("V3-E2 schema or stage is invalid")
    if block["status"] != "FROZEN_FOR_EXECUTION":
        raise ValueError("V3-E2 configuration is not frozen")
    if block["protocol_manifest"] != (
        "config/v3/development_protocol_manifest.json"
    ):
        raise ValueError("V3-E2 protocol manifest path differs")

    source = _exact_keys(
        block["source"],
        {
            "path",
            "allowed_partitions",
            "expected_rows",
            "expected_partition_rows",
            "expected_critical_rows",
            "sort_order",
        },
        "source",
    )
    if tuple(source["allowed_partitions"]) != ALLOWED_PARTITIONS:
        raise ValueError("V3-E2 may cache train and validation only")
    if int(source["expected_rows"]) != 591532:
        raise ValueError("V3-E2 expected row count differs")
    if source["expected_partition_rows"] != {
        "train": 345552,
        "validation": 245980,
    }:
        raise ValueError("V3-E2 partition supports differ")
    if source["expected_critical_rows"] != {
        "train": 946,
        "validation": 892,
    }:
        raise ValueError("V3-E2 critical supports differ")
    if tuple(source["sort_order"]) != (
        "partition_name",
        "received_date",
        "normalized_group_hash",
        "normalized_length",
    ):
        raise ValueError("V3-E2 deterministic sort order differs")

    model = _exact_keys(
        block["model"],
        {
            "model_id",
            "revision",
            "device",
            "torch_dtype",
            "max_seq_length",
            "normalize_embeddings",
            "embedding_dimensions",
        },
        "model",
    )
    if model != {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": "cuda",
        "torch_dtype": "float16",
        "max_seq_length": 256,
        "normalize_embeddings": False,
        "embedding_dimensions": 384,
    }:
        raise ValueError("V3-E2 model differs from the frozen MiniLM")

    execution = _exact_keys(
        block["execution"],
        {
            "default_run_mode",
            "batch_size",
            "shard_rows",
            "random_state",
            "transactional_write",
            "overwrite_existing_cache",
        },
        "execution",
    )
    if execution != {
        "default_run_mode": "disabled",
        "batch_size": 128,
        "shard_rows": 32768,
        "random_state": 42,
        "transactional_write": True,
        "overwrite_existing_cache": False,
    }:
        raise ValueError("V3-E2 execution controls differ")

    output = _exact_keys(
        block["output"],
        {
            "cache_root",
            "model_cache",
            "manifest",
            "embedding_dtype",
            "key_columns",
            "individual_cache_publication",
            "raw_narrative_persistence",
        },
        "output",
    )
    if tuple(output["key_columns"]) != KEY_COLUMNS:
        raise ValueError("V3-E2 key columns differ")
    if output["embedding_dtype"] != "float16":
        raise ValueError("V3-E2 embeddings must be float16")
    if output["individual_cache_publication"] is not False:
        raise ValueError("V3-E2 individual cache may not be published")
    if output["raw_narrative_persistence"] is not False:
        raise ValueError("V3-E2 may not persist raw narratives")
    return EmbeddingCacheConfig(json.loads(json.dumps(block)))


def load_embedding_cache_config(
    path: str | Path = DEFAULT_CONFIG,
) -> EmbeddingCacheConfig:
    """Load one frozen V3-E2 cache configuration."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3-E2 config: {path}") from error
    return validate_embedding_cache_config(payload)


def _file_signature(path: Path) -> dict[str, Any]:
    """Return the size and uppercase SHA-256 of one cache file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "sha256": digest.hexdigest().upper(),
        "size_bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one aggregate manifest atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_encoder(config: EmbeddingCacheConfig, cache_folder: Path) -> tuple[Any, Any]:
    """Load the frozen MiniLM lazily after boundary validation."""

    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("V3-E2 requires CUDA")
    model = config.payload["model"]
    encoder = SentenceTransformer(
        model["model_id"],
        revision=model["revision"],
        cache_folder=str(cache_folder),
        device="cuda",
        backend="torch",
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.float16},
    )
    encoder.max_seq_length = int(model["max_seq_length"])
    return encoder, torch


def _source_query() -> str:
    """Return the fixed development-only, deterministically ordered query."""

    return """
        SELECT
            received_date,
            product_family,
            normalized_group_hash,
            normalized_length,
            partition_name,
            narrative
        FROM read_parquet(?)
        WHERE partition_name IN ('train', 'validation')
        ORDER BY
            partition_name,
            received_date,
            normalized_group_hash,
            normalized_length
    """


def _audit_source(connection: Any, source: Path) -> dict[str, Any]:
    """Audit aggregate supports and group uniqueness before model loading."""

    rows = connection.execute(
        """
        SELECT
            partition_name,
            count(*) AS rows,
            sum(CASE WHEN product_family = ? THEN 1 ELSE 0 END) AS critical,
            count(DISTINCT normalized_group_hash || ':' || normalized_length) AS groups
        FROM read_parquet(?)
        GROUP BY partition_name
        ORDER BY partition_name
        """,
        [CRITICAL_CLASS, str(source)],
    ).fetchall()
    return {
        str(partition): {
            "rows": int(row_count),
            "critical_rows": int(critical),
            "groups": int(groups),
        }
        for partition, row_count, critical, groups in rows
    }


def _validate_source_audit(
    audit: Mapping[str, Mapping[str, int]],
    config: EmbeddingCacheConfig,
) -> None:
    """Reject forbidden partitions, support drift, or duplicate groups."""

    if tuple(sorted(audit)) != ALLOWED_PARTITIONS:
        raise ValueError("V3-E2 source contains unexpected partitions")
    expected_rows = config.payload["source"]["expected_partition_rows"]
    expected_critical = config.payload["source"]["expected_critical_rows"]
    for partition in ALLOWED_PARTITIONS:
        observed = audit[partition]
        if observed["rows"] != int(expected_rows[partition]):
            raise ValueError(f"V3-E2 {partition} row support differs")
        if observed["critical_rows"] != int(expected_critical[partition]):
            raise ValueError(f"V3-E2 {partition} critical support differs")
        if observed["groups"] != observed["rows"]:
            raise ValueError(f"V3-E2 {partition} group identity is not unique")


def _row_signature(batch: Any, digest: Any) -> None:
    """Update an order-sensitive digest from non-narrative scientific keys."""

    for name in KEY_COLUMNS:
        for value in batch.column(batch.schema.get_field_index(name)).to_pylist():
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\x1f")
        digest.update(b"\x1e")


def run_embedding_cache(
    project_root: str | Path = ".",
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    encoder_loader: (
        Callable[[EmbeddingCacheConfig, Path], tuple[Any, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Create the V3-E2 local cache without opening any sealed partition."""

    import duckdb
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(project_root).expanduser().resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_embedding_cache_config(config_file)
    protocol = validate_protocol_manifest(
        config.payload["protocol_manifest"], root
    )
    source = _safe_project_path(root, config.payload["source"]["path"])
    cache_root = _safe_project_path(root, config.payload["output"]["cache_root"])
    model_cache = _safe_project_path(root, config.payload["output"]["model_cache"])
    manifest_path = _safe_project_path(root, config.payload["output"]["manifest"])
    if cache_root.exists():
        raise FileExistsError(f"V3-E2 cache already exists: {cache_root}")
    model_cache.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as connection:
        audit = _audit_source(connection, source)
        _validate_source_audit(audit, config)

    loader = encoder_loader or _load_encoder
    encoder, torch_module = loader(config, model_cache)
    stage = cache_root.parent / f".{cache_root.name}.stage-{uuid.uuid4().hex}"
    if not stage.resolve().is_relative_to((root / "temp/v3").resolve()):
        raise ValueError("V3-E2 staging directory escapes temp/v3")
    stage.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    shards: list[dict[str, Any]] = []
    row_digest = hashlib.sha256()
    total_rows = 0
    try:
        with duckdb.connect() as connection:
            reader = connection.execute(
                _source_query(), [str(source)]
            ).to_arrow_reader(batch_size=config.shard_rows)
            for shard_index, batch in enumerate(reader):
                narrative_index = batch.schema.get_field_index("narrative")
                texts = batch.column(narrative_index).to_pylist()
                embeddings = encoder.encode(
                    texts,
                    batch_size=config.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                embeddings = np.asarray(embeddings, dtype=np.float16)
                if embeddings.shape != (len(texts), 384):
                    raise ValueError("V3-E2 encoder returned an unexpected shape")
                embedding_name = f"embeddings-{shard_index:04d}.npy"
                keys_name = f"keys-{shard_index:04d}.parquet"
                embedding_path = stage / embedding_name
                keys_path = stage / keys_name
                np.save(embedding_path, embeddings, allow_pickle=False)
                key_batch = batch.select(list(KEY_COLUMNS))
                pq.write_table(
                    pa.Table.from_batches([key_batch]),
                    keys_path,
                    compression="zstd",
                )
                _row_signature(key_batch, row_digest)
                row_count = len(texts)
                total_rows += row_count
                shards.append(
                    {
                        "shard_index": shard_index,
                        "row_count": row_count,
                        "embedding_path": embedding_name,
                        "embedding": _file_signature(embedding_path),
                        "keys_path": keys_name,
                        "keys": _file_signature(keys_path),
                    }
                )
                del embeddings, texts, key_batch
        expected_rows = int(config.payload["source"]["expected_rows"])
        if total_rows != expected_rows:
            raise ValueError("V3-E2 streamed row count differs from the contract")
        os.replace(stage, cache_root)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    torch_module.cuda.synchronize()
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": "V3-E2",
        "status": "COMPLETE_DEVELOPMENT_CACHE",
        "complete": True,
        "selection_authorized": False,
        "scientific_partitions": list(ALLOWED_PARTITIONS),
        "forbidden_partition_access": False,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "embedding_dimensions": 384,
            "embedding_dtype": "float16",
        },
        "source_audit": audit,
        "row_count": total_rows,
        "ordered_key_sha256": row_digest.hexdigest().upper(),
        "shard_count": len(shards),
        "shards": shards,
        "runtime": {
            "elapsed_seconds": elapsed,
            "rows_per_second": total_rows / elapsed,
            "batch_size": config.batch_size,
            "device": str(torch_module.cuda.get_device_name(0)),
            "peak_allocated_mib": (
                torch_module.cuda.max_memory_allocated() / 1024**2
            ),
            "peak_reserved_mib": (
                torch_module.cuda.max_memory_reserved() / 1024**2
            ),
            "python": platform.python_version(),
            "sentence_transformers": importlib.metadata.version(
                "sentence-transformers"
            ),
            "torch": importlib.metadata.version("torch"),
        },
        "protocol_file_sha256": protocol["protocol"]["sha256"],
        "individual_cache_publication": False,
        "raw_narrative_persistence": False,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    """Run E2 only when the explicit ``--run`` gate is present."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.run:
        raise SystemExit("V3-E2 is disabled by default; pass --run explicitly")
    result = run_embedding_cache()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
