"""S3 frozen-protocol materialization, scientific sampling, and baselines."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import f1_score

from .temporal_split import FINGERPRINT_VERSION, MODELED_FAMILIES
from .tracking import NullTracker, Tracker


S3_SCHEMA_VERSION = "s3-frozen-protocol-v1"
S3_SCIENTIFIC_CACHE_VERSION = "s3-scientific-cache-v2"
DEVELOPMENT_PARTITIONS = ("train", "validation")
SEALED_PARTITIONS = ("test", "stress", "monitor")
S3_PROTOCOL_BOUNDARY = {
    "development_partitions": list(DEVELOPMENT_PARTITIONS),
    "fit_partition": "train",
    "evaluation_partition": "validation",
    "sealed_partitions": list(SEALED_PARTITIONS),
}
DEFAULT_MAX_LOADED_ROWS = 250_000
DEFAULT_FULL_BATCH_SIZE = 4_096
DEFAULT_FULL_MEMORY_BUDGET_GB = 7.0
DATASET_COLUMNS = (
    "Complaint ID",
    "received_date",
    "product_family",
    "normalized_group_hash",
    "normalized_length",
    "partition_name",
    "narrative",
)
SCIENTIFIC_COLUMNS = DATASET_COLUMNS


@dataclass(frozen=True, slots=True)
class FrozenS3Protocol:
    """Represent the approved S3 protocol and its privacy boundary."""

    protocol_id: str
    windows: Mapping[str, Mapping[str, str]]
    modeled_families: tuple[str, ...]
    rare_family: str
    fingerprint_version: str
    report_sha256: str | None
    index_sha256: str | None
    approval_status: str
    approved_on: str

    @classmethod
    def from_json(cls, path: str | Path) -> "FrozenS3Protocol":
        """Load and validate a frozen protocol configuration."""

        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != S3_SCHEMA_VERSION:
            raise ValueError("Unexpected S3 protocol schema version")
        group = payload.get("group_identity", {})
        approval = payload.get("approval", {})
        evidence = payload.get("s2_evidence", {})
        protocol = cls(
            protocol_id=str(payload["protocol_id"]),
            windows=payload["windows"],
            modeled_families=tuple(payload["modeled_families"]),
            rare_family=str(payload["rare_policy"]["family"]),
            fingerprint_version=str(group["version"]),
            report_sha256=evidence.get("report_sha256"),
            index_sha256=evidence.get("index_sha256"),
            approval_status=str(approval["status"]),
            approved_on=str(approval["approved_on"]),
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        """Validate the approved windows, classes, and sealed partitions."""

        if self.protocol_id != "post_2023_taxonomy":
            raise ValueError("S3 requires the approved post_2023_taxonomy")
        if self.approval_status != "FROZEN_FOR_S3_DEVELOPMENT":
            raise ValueError("S3 protocol is not frozen for development")
        if self.fingerprint_version != FINGERPRINT_VERSION:
            raise ValueError("S3 fingerprint version differs from S2")
        if tuple(sorted(self.modeled_families)) != tuple(
            sorted(MODELED_FAMILIES)
        ):
            raise ValueError("S3 modeled families differ from S2")
        expected = {
            "train": ("2023-08-01", "2024-06-30"),
            "validation": ("2024-07-01", "2024-12-31"),
            "test": ("2025-01-01", "2025-06-30"),
            "stress": ("2025-07-01", "2025-12-31"),
            "monitor": ("2026-01-01", "2026-12-31"),
        }
        actual = {
            name: (window["start"], window["end"])
            for name, window in self.windows.items()
        }
        if actual != expected:
            raise ValueError("S3 windows do not match the approved protocol")

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the frozen protocol."""

        return asdict(self)


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_frozen_report(
    protocol: FrozenS3Protocol,
    report_path: str | Path,
    verify_digest: bool = True,
) -> dict[str, Any]:
    """Validate that the approved S2 pilot report is the selected candidate.

    Args:
        protocol: Frozen S3 protocol to validate.
        report_path: Cached S2 pilot report JSON path.
        verify_digest: Verify the configured digest when one is present.

    Returns:
        The selected candidate report.

    Raises:
        ValueError: If the report is not the approved nine-class pilot.
    """

    path = Path(report_path)
    if verify_digest and protocol.report_sha256:
        if _sha256(path) != protocol.report_sha256.upper():
            raise ValueError("S2 pilot report digest does not match S3 config")
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = payload.get("report", payload)
    candidate = next(
        (
            item
            for item in report.get("candidates", [])
            if item.get("candidate", {}).get("name") == protocol.protocol_id
        ),
        None,
    )
    if candidate is None or candidate.get("candidate_status") != "PASS":
        raise ValueError("Approved S2 candidate is absent or does not pass")
    if candidate.get("eligible_class_count") != len(MODELED_FAMILIES):
        raise ValueError("Approved S2 candidate does not contain nine classes")
    return candidate


def assert_development_partitions(partitions: Iterable[str]) -> None:
    """Reject any request that crosses the S3 development boundary."""

    requested = set(partitions)
    invalid = requested.difference(DEVELOPMENT_PARTITIONS)
    if invalid:
        raise ValueError(
            "S3 cannot read narratives from sealed partitions: "
            f"{sorted(invalid)}"
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalize_protocol_boundary(payload: dict[str, Any]) -> bool:
    """Migrate an S3 artifact to the explicit partition boundary.

    Returns:
        ``True`` when the payload was changed and should be persisted.
    """

    boundary = payload.get("protocol_boundary")
    if boundary == S3_PROTOCOL_BOUNDARY:
        return False
    payload["protocol_boundary"] = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in S3_PROTOCOL_BOUNDARY.items()
    }
    return True


def _dataset_signature(
    source_path: Path,
    index_path: Path,
    protocol: FrozenS3Protocol,
) -> dict[str, Any]:
    """Build the cache signature for the development-only dataset."""

    protocol_payload = json.loads(
        json.dumps(protocol.to_dict(), sort_keys=True)
    )
    return {
        "schema_version": S3_SCHEMA_VERSION,
        "source": {
            "path": str(source_path.resolve()),
            "size": source_path.stat().st_size,
            "mtime_ns": source_path.stat().st_mtime_ns,
        },
        "index": {
            "path": str(index_path.resolve()),
            "size": index_path.stat().st_size,
            "mtime_ns": index_path.stat().st_mtime_ns,
        },
        "protocol": protocol_payload,
        "narrative_boundary": list(DEVELOPMENT_PARTITIONS),
    }


def _validate_development_dataset(path: Path) -> int:
    """Validate the cached dataset schema and development-only partitions."""

    with duckdb.connect() as connection:
        schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
        columns = tuple(row[0] for row in schema)
        if columns != DATASET_COLUMNS:
            raise ValueError(f"Unexpected S3 dataset columns: {columns!r}")
        result = connection.execute(
            """
            SELECT count(*), count(DISTINCT "Complaint ID"),
                count(*) FILTER (
                    WHERE partition_name NOT IN ('train', 'validation')
                )
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
    rows, distinct_ids, sealed_rows = (int(value) for value in result)
    if rows != distinct_ids:
        raise ValueError("S3 development dataset contains duplicate IDs")
    if sealed_rows:
        raise ValueError("S3 development dataset contains a sealed partition")
    return rows


def build_or_load_development_dataset(
    parquet_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    protocol: FrozenS3Protocol,
    metadata_path: str | Path | None = None,
    temp_directory: str | Path | None = None,
    text_column: str = "Consumer complaint narrative",
    memory_limit: str = "2GB",
    threads: int = 2,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Build or load a narrative cache containing only train and validation.

    Args:
        parquet_path: Original processed Parquet source.
        index_path: S2 hash-only modeling index.
        output_path: Development-only Parquet cache destination.
        protocol: Frozen protocol approved before fitting.
        metadata_path: Optional cache metadata destination.
        temp_directory: Optional DuckDB spill directory.
        text_column: Narrative column in the source Parquet.
        memory_limit: DuckDB memory limit.
        threads: DuckDB worker count.
        force_refresh: Ignore matching cache metadata.

    Returns:
        Cache status, signature, schema, and row count.

    Raises:
        ValueError: If the protocol or cache contract is invalid.
    """

    protocol.validate()
    source = Path(parquet_path).expanduser().resolve()
    index = Path(index_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    metadata = Path(metadata_path).expanduser().resolve() if metadata_path else (
        output.with_suffix(".json")
    )
    if not source.exists() or not index.exists():
        raise FileNotFoundError("S3 source and S2 index are both required")
    if threads <= 0:
        raise ValueError("threads must be positive")
    signature = _dataset_signature(source, index, protocol)
    cached = None
    if not force_refresh and output.exists() and metadata.exists():
        try:
            cached = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
    if cached and cached.get("signature") == signature:
        rows = _validate_development_dataset(output)
        return {"status": "hit", "rows": rows, **cached}

    output.parent.mkdir(parents=True, exist_ok=True)
    spill = Path(temp_directory) if temp_directory else output.parent / "duckdb"
    spill.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    partition_sql = (
        "CASE WHEN TRY_CAST(received_date AS DATE) BETWEEN "
        "DATE '2023-08-01' AND DATE '2024-06-30' THEN 'train' "
        "WHEN TRY_CAST(received_date AS DATE) BETWEEN DATE '2024-07-01' "
        "AND DATE '2024-12-31' THEN 'validation' END"
    )
    family_values = ", ".join(f"'{family}'" for family in protocol.modeled_families)
    identifier = '"' + text_column.replace('"', '""') + '"'
    try:
        with duckdb.connect() as connection:
            connection.execute(f"SET memory_limit = '{memory_limit}'")
            connection.execute(f"SET threads = {threads}")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute(f"SET temp_directory = '{spill.as_posix()}'")
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE allowed_index AS
                SELECT "Complaint ID", received_date, product_family,
                    normalized_group_hash, normalized_length,
                    {partition_sql} AS partition_name
                FROM read_parquet(?)
                WHERE product_family IN ({family_values})
                    AND TRY_CAST(received_date AS DATE) BETWEEN DATE '2023-08-01'
                    AND DATE '2024-12-31'
                """,
                [str(index)],
            )
            connection.execute(
                f"""
                COPY (
                    SELECT i."Complaint ID", i.received_date,
                        i.product_family, i.normalized_group_hash,
                        i.normalized_length, i.partition_name,
                        CAST(s.{identifier} AS VARCHAR) AS narrative
                    FROM read_parquet(?) AS s
                    INNER JOIN allowed_index AS i
                        ON s."Complaint ID" = i."Complaint ID"
                    WHERE trim(coalesce(CAST(s.{identifier} AS VARCHAR), '')) <> ''
                    ORDER BY i."Complaint ID"
                ) TO '{temporary.as_posix().replace("'", "''")}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """,
                [str(source)],
            )
        rows = _validate_development_dataset(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    payload = {
        "schema_version": S3_SCHEMA_VERSION,
        "signature": signature,
        "columns": list(DATASET_COLUMNS),
        "rows": rows,
        "contains_sealed_narratives": False,
    }
    _write_json_atomic(metadata, payload)
    return {"status": "refreshed", **payload}


def _scientific_signature(dataset_path: Path) -> dict[str, Any]:
    """Build the invalidation signature for the scientific cache."""

    return {
        "schema_version": S3_SCIENTIFIC_CACHE_VERSION,
        "source": {
            "path": str(dataset_path.resolve()),
            "size": dataset_path.stat().st_size,
            "mtime_ns": dataset_path.stat().st_mtime_ns,
        },
        "group_identity": ["normalized_group_hash", "normalized_length"],
        "ambiguous_label_scope": list(DEVELOPMENT_PARTITIONS),
        "representative": "minimum Complaint ID",
    }


def _validate_scientific_cache(path: Path) -> int:
    """Validate schema, uniqueness, and sealed-partition exclusion."""

    with duckdb.connect() as connection:
        schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
        columns = tuple(row[0] for row in schema)
        if columns != SCIENTIFIC_COLUMNS:
            raise ValueError(
                f"Unexpected scientific cache columns: {columns!r}"
            )
        result = connection.execute(
            """
            SELECT count(*), count(DISTINCT "Complaint ID"),
                count(DISTINCT (normalized_group_hash, normalized_length)),
                count(*) FILTER (
                    WHERE partition_name NOT IN ('train', 'validation')
                )
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
    rows, ids, groups, sealed = (int(value) for value in result)
    if rows != ids or rows != groups:
        raise ValueError(
            "Scientific cache must contain one unique row per group and ID"
        )
    if sealed:
        raise ValueError("Scientific cache contains a sealed partition")
    return rows


def _scientific_summary(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Return audit counts for ambiguity and scientific representatives."""

    base = """
        WITH base AS (
            SELECT "Complaint ID", product_family, partition_name,
                normalized_group_hash, normalized_length
            FROM source_data
        ), labels AS (
            SELECT normalized_group_hash, normalized_length,
                count(DISTINCT product_family) AS label_count
            FROM base
            GROUP BY normalized_group_hash, normalized_length
        )
    """
    totals = connection.execute(
        base
        + """
        SELECT
            (SELECT count(*) FROM base),
            (SELECT count(*) FROM labels),
            (SELECT count(*) FROM labels WHERE label_count > 1),
            (SELECT count(*) FROM base b JOIN labels l USING (
                normalized_group_hash, normalized_length
            ) WHERE l.label_count > 1)
        """
    ).fetchone()
    raw_rows, raw_groups, ambiguous_groups, ambiguous_rows = (
        int(value) for value in totals
    )
    partition_rows = connection.execute(
        base
        + """
        SELECT partition_name, count(*)
        FROM base
        GROUP BY partition_name
        ORDER BY partition_name
        """
    ).fetchall()
    ambiguous_partition_rows = connection.execute(
        base
        + """
        SELECT b.partition_name, count(*)
        FROM base b
        JOIN labels l USING (normalized_group_hash, normalized_length)
        WHERE l.label_count > 1
        GROUP BY b.partition_name
        ORDER BY b.partition_name
        """
    ).fetchall()
    scientific_counts = connection.execute(
        base
        + """
        , clean_train AS (
            SELECT b.*, row_number() OVER (
                PARTITION BY b.normalized_group_hash,
                    b.normalized_length
                ORDER BY b."Complaint ID"
            ) AS representative_rank
            FROM base b
            JOIN labels l USING (normalized_group_hash, normalized_length)
            WHERE l.label_count = 1 AND b.partition_name = 'train'
        ), clean_validation AS (
            SELECT b.*, row_number() OVER (
                PARTITION BY b.normalized_group_hash,
                    b.normalized_length
                ORDER BY b."Complaint ID"
            ) AS representative_rank
            FROM base b
            JOIN labels l USING (normalized_group_hash, normalized_length)
            WHERE l.label_count = 1 AND b.partition_name = 'validation'
        )
        SELECT partition_name, product_family, count(*)
        FROM clean_train
        WHERE representative_rank = 1
        GROUP BY partition_name, product_family
        UNION ALL
        SELECT partition_name, product_family, count(*)
        FROM clean_validation v
        WHERE representative_rank = 1
            AND NOT EXISTS (
                SELECT 1 FROM clean_train t
                WHERE t.normalized_group_hash = v.normalized_group_hash
                    AND t.normalized_length = v.normalized_length
            )
        GROUP BY partition_name, product_family
        ORDER BY partition_name, product_family
        """
    ).fetchall()
    return {
        "raw_rows": raw_rows,
        "raw_groups": raw_groups,
        "ambiguous_groups": ambiguous_groups,
        "ambiguous_rows": ambiguous_rows,
        "raw_rows_by_partition": {
            str(partition): int(count) for partition, count in partition_rows
        },
        "ambiguous_rows_by_partition": {
            str(partition): int(count)
            for partition, count in ambiguous_partition_rows
        },
        "scientific_rows_by_partition_and_class": [
            {
                "partition": str(partition),
                "product_family": str(family),
                "rows": int(count),
            }
            for partition, family, count in scientific_counts
        ],
    }


def build_or_load_scientific_cache(
    development_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    temp_directory: str | Path | None = None,
    memory_limit: str = "4GB",
    threads: int = 1,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Build or load a scientific one-row-per-group development cache.

    Ambiguity is resolved over all train and validation rows before selecting
    representatives. Validation representatives whose group appears in train
    are excluded. The source cache remains the all-text operational view.

    Args:
        development_path: Development-only cache with narratives.
        output_path: Scientific cache destination.
        metadata_path: Optional metadata and audit-summary destination.
        temp_directory: Optional DuckDB spill directory.
        memory_limit: DuckDB memory limit.
        threads: DuckDB worker count.
        force_refresh: Ignore matching cache metadata.

    Returns:
        Cache status, signature, row count, and audit summary.

    Raises:
        FileNotFoundError: If the development cache is absent.
        ValueError: If the cache parameters or source boundary are invalid.
    """

    source = Path(development_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    metadata = Path(metadata_path).expanduser().resolve() if metadata_path else (
        output.with_suffix(".json")
    )
    if not source.exists():
        raise FileNotFoundError(f"S3 development dataset does not exist: {source}")
    if threads <= 0:
        raise ValueError("threads must be positive")
    signature = _scientific_signature(source)
    cached = None
    if not force_refresh and output.exists() and metadata.exists():
        try:
            cached = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
    if cached and cached.get("signature") == signature:
        rows = _validate_scientific_cache(output)
        return {"status": "hit", "rows": rows, **cached}

    output.parent.mkdir(parents=True, exist_ok=True)
    spill = Path(temp_directory) if temp_directory else output.parent / "duckdb"
    spill.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    projection = ", ".join(f'"{column}"' for column in SCIENTIFIC_COLUMNS)
    source_literal = str(source).replace("'", "''")
    try:
        with duckdb.connect() as connection:
            connection.execute(f"SET memory_limit = '{memory_limit}'")
            connection.execute(f"SET threads = {threads}")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute(f"SET temp_directory = '{spill.as_posix()}'")
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP VIEW source_data AS
                SELECT "Complaint ID", received_date, product_family,
                    normalized_group_hash, normalized_length,
                    partition_name, narrative
                FROM read_parquet('{source_literal}')
                WHERE partition_name IN ('train', 'validation')
                """
            )
            summary = _scientific_summary(connection)
            connection.execute(
                f"""
                COPY (
                    WITH labels AS (
                        SELECT normalized_group_hash, normalized_length,
                            count(DISTINCT product_family) AS label_count
                        FROM source_data
                        GROUP BY normalized_group_hash, normalized_length
                    ), clean_train AS (
                        SELECT s.*, row_number() OVER (
                            PARTITION BY s.normalized_group_hash,
                                s.normalized_length
                            ORDER BY s."Complaint ID"
                        ) AS representative_rank
                        FROM source_data s
                        JOIN labels l USING (
                            normalized_group_hash, normalized_length
                        )
                        WHERE l.label_count = 1
                            AND s.partition_name = 'train'
                    ), clean_validation AS (
                        SELECT s.*, row_number() OVER (
                            PARTITION BY s.normalized_group_hash,
                                s.normalized_length
                            ORDER BY s."Complaint ID"
                        ) AS representative_rank
                        FROM source_data s
                        JOIN labels l USING (
                            normalized_group_hash, normalized_length
                        )
                        WHERE l.label_count = 1
                            AND s.partition_name = 'validation'
                    ), ranked AS (
                        SELECT * FROM clean_train
                        WHERE representative_rank = 1
                        UNION ALL
                        SELECT v.* FROM clean_validation v
                        WHERE v.representative_rank = 1
                            AND NOT EXISTS (
                                SELECT 1 FROM clean_train t
                                WHERE t.normalized_group_hash =
                                    v.normalized_group_hash
                                    AND t.normalized_length = v.normalized_length
                            )
                    )
                    SELECT {projection}
                    FROM ranked
                    WHERE representative_rank = 1
                    ORDER BY "Complaint ID"
                ) TO '{temporary.as_posix().replace("'", "''")}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        rows = _validate_scientific_cache(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    payload = {
        "schema_version": S3_SCIENTIFIC_CACHE_VERSION,
        "signature": signature,
        "columns": list(SCIENTIFIC_COLUMNS),
        "rows": rows,
        "summary": summary,
        "contains_sealed_narratives": False,
    }
    _write_json_atomic(metadata, payload)
    return {"status": "refreshed", **payload}


def read_scientific_frame(
    cache_path: str | Path,
) -> Any:
    """Load only the scientific cache as a columnar Arrow table.

    Args:
        cache_path: Scientific cache produced by
            :func:`build_or_load_scientific_cache`.

    Returns:
        A PyArrow table; it contains no row dictionaries.
    """

    path = Path(cache_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Scientific cache does not exist: {path}")
    _validate_scientific_cache(path)
    projection = ", ".join(f'"{column}"' for column in SCIENTIFIC_COLUMNS)
    with duckdb.connect() as connection:
        return connection.execute(
            f"SELECT {projection} "
            "FROM read_parquet(?) ORDER BY \"Complaint ID\"",
            [str(path)],
        ).to_arrow_table()


def iter_operational_validation_batches(
    development_path: str | Path,
    batch_size: int = DEFAULT_FULL_BATCH_SIZE,
) -> Iterable[tuple[list[str], list[str]]]:
    """Yield validation narratives and labels in bounded Arrow batches.

    Only the all-text validation partition is read. The caller must consume
    the iterator without retaining previous batches.

    Args:
        development_path: Development-only cache.
        batch_size: Maximum number of rows yielded per batch.

    Yields:
        Text and label lists for one validation batch.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    path = Path(development_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"S3 development dataset does not exist: {path}")
    assert_development_partitions(("validation",))
    with duckdb.connect() as connection:
        reader = connection.execute(
            """
            SELECT narrative, product_family
            FROM read_parquet(?)
            WHERE partition_name = 'validation'
            ORDER BY "Complaint ID"
            """,
            [str(path)],
        ).to_arrow_reader(batch_size)
        for batch in reader:
            yield (
                [str(value) for value in batch.column("narrative").to_pylist()],
                [str(value)
                 for value in batch.column("product_family").to_pylist()],
            )


def read_development_rows(
    dataset_path: str | Path,
    partitions: Sequence[str] = DEVELOPMENT_PARTITIONS,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Read development rows, optionally sampled by partition and class.

    Args:
        dataset_path: Development-only S3 cache.
        partitions: Allowed development partitions to read.
        max_rows: Optional deterministic upper bound allocated across
            partition and family strata.
    """

    assert_development_partitions(partitions)
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"S3 development dataset does not exist: {path}")
    selected = tuple(partitions)
    placeholders = ", ".join("?" for _ in selected)
    projection = ", ".join(f'"{column}"' for column in DATASET_COLUMNS)
    if max_rows is None:
        query = f"""
            SELECT {projection} FROM read_parquet(?)
            WHERE partition_name IN ({placeholders})
            ORDER BY "Complaint ID"
        """
        parameters: list[Any] = [str(path), *selected]
    else:
        strata = max(1, len(selected) * len(MODELED_FAMILIES))
        per_stratum = max(1, int(np.ceil(max_rows / strata)))
        query = f"""
            WITH ranked AS (
                SELECT {projection},
                    row_number() OVER (
                        PARTITION BY partition_name, product_family
                        ORDER BY "Complaint ID"
                    ) AS s3_rank
                FROM read_parquet(?)
                WHERE partition_name IN ({placeholders})
            )
            SELECT {projection} FROM ranked
            WHERE s3_rank <= ?
            ORDER BY "Complaint ID"
        """
        parameters = [str(path), *selected, per_stratum]
    with duckdb.connect() as connection:
        result = connection.execute(query, parameters)
        names = [item[0] for item in connection.description]
        rows = [dict(zip(names, row)) for row in result.fetchall()]
    if max_rows is not None and len(rows) > max_rows:
        return _stratified_limit_rows(rows, max_rows)
    return rows


def _identity(row: Mapping[str, Any]) -> tuple[str, int]:
    """Return the versioned composite identity for one row."""

    return str(row["normalized_group_hash"]), int(row["normalized_length"])


def prepare_scientific_split(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve labels and select clean one-row-per-group train and validation.

    Groups with multiple modeled families anywhere in development receive the
    technical ``label_ambiguous`` status and never enter scientific samples.
    Validation groups also must be novel relative to the complete train group
    identity set before their deterministic representative is selected.
    """

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_identity(row)].append(row)
    labels = {
        key: {str(item["product_family"]) for item in group}
        for key, group in grouped.items()
    }
    ambiguous = {
        key for key, group_labels in labels.items() if len(group_labels) > 1
    }
    train_ids = {
        _identity(row) for row in rows if row["partition_name"] == "train"
    }
    clean_train: list[Mapping[str, Any]] = []
    clean_validation: list[Mapping[str, Any]] = []
    for key, group in grouped.items():
        if key in ambiguous:
            continue
        representative = min(group, key=lambda item: int(item["Complaint ID"]))
        if representative["partition_name"] == "train":
            clean_train.append(representative)
        if representative["partition_name"] == "validation" and key not in train_ids:
            clean_validation.append(representative)
    clean_train.sort(key=lambda item: int(item["Complaint ID"]))
    clean_validation.sort(key=lambda item: int(item["Complaint ID"]))
    ambiguous_by_partition = {
        partition: sum(
            1
            for key in ambiguous
            if any(
                row["partition_name"] == partition for row in grouped[key]
            )
        )
        for partition in DEVELOPMENT_PARTITIONS
    }
    return {
        "train": [dict(row) for row in clean_train],
        "validation_scientific": [dict(row) for row in clean_validation],
        "validation_all_text": [
            dict(row) for row in rows if row["partition_name"] == "validation"
        ],
        "summary": {
            "groups_total": len(grouped),
            "label_ambiguous_groups": len(ambiguous),
            "label_ambiguous_by_partition": ambiguous_by_partition,
            "label_ambiguous": "excluded_from_scientific_train_and_validation",
            "train_scientific_rows": len(clean_train),
            "validation_scientific_rows": len(clean_validation),
        },
    }


def build_learning_curve_subsets(
    train_rows: Sequence[Mapping[str, Any]],
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    random_state: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic group-stratified subsets containing every class."""

    if not train_rows:
        raise ValueError("Scientific training rows must not be empty")
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_class[str(row["product_family"])].append(row)
    missing = set(MODELED_FAMILIES).difference(by_class)
    if missing:
        raise ValueError(f"Learning curve is missing classes: {sorted(missing)}")
    normalized_fractions = sorted({float(value) for value in fractions})
    if not normalized_fractions or any(
        value <= 0 or value > 1 for value in normalized_fractions
    ):
        raise ValueError("Learning-curve fractions must be in (0, 1]")
    output: dict[str, list[dict[str, Any]]] = {}
    for fraction in normalized_fractions:
        selected: list[dict[str, Any]] = []
        for label in sorted(by_class):
            candidates = sorted(
                by_class[label],
                key=lambda row: hashlib.sha256(
                    f"{random_state}|{label}|{_identity(row)}".encode("utf-8")
                ).hexdigest(),
            )
            count = max(1, int(np.ceil(len(candidates) * fraction)))
            selected.extend(dict(row) for row in candidates[:count])
        selected.sort(key=lambda row: int(row["Complaint ID"]))
        output[f"{fraction:g}"] = selected
    return output


def build_learning_curve_indices(
    frame: Any,
    fractions: Sequence[float],
    random_state: int = 42,
    max_groups_per_class: int | None = None,
) -> dict[str, np.ndarray]:
    """Build curve subsets as row indices over a scientific Arrow table.

    Args:
        frame: Scientific PyArrow table with one row per clean group.
        fractions: Fractions in the open interval ``(0, 1]``.
        random_state: Stable seed used in deterministic group ordering.
        max_groups_per_class: Optional per-class cap for a bounded run.

    Returns:
        Mapping from fraction labels to sorted NumPy row indices.

    Raises:
        ValueError: If a fraction, cap, or modeled class is invalid.
    """

    if max_groups_per_class is not None and max_groups_per_class <= 0:
        raise ValueError("max_groups_per_class must be positive")
    normalized = sorted({float(value) for value in fractions})
    if not normalized or any(value <= 0 or value > 1 for value in normalized):
        raise ValueError("Learning-curve fractions must be in (0, 1]")
    partitions = [str(value.as_py()) for value in frame["partition_name"]]
    train_indices = [
        index for index, partition in enumerate(partitions)
        if partition == "train"
    ]
    labels = [
        str(frame["product_family"][index].as_py()) for index in train_indices
    ]
    groups = [
        str(frame["normalized_group_hash"][index].as_py())
        for index in train_indices
    ]
    lengths = [
        int(frame["normalized_length"][index].as_py())
        for index in train_indices
    ]
    by_class: dict[str, list[int]] = defaultdict(list)
    for local_index, label in enumerate(labels):
        by_class[label].append(local_index)
    missing = set(MODELED_FAMILIES).difference(by_class)
    if missing:
        raise ValueError(f"Learning curve is missing classes: {sorted(missing)}")
    output: dict[str, np.ndarray] = {}
    for fraction in normalized:
        selected: list[int] = []
        for label in sorted(by_class):
            candidates = sorted(
                by_class[label],
                key=lambda index: hashlib.sha256(
                    f"{random_state}|{label}|{groups[index]}|"
                    f"{lengths[index]}".encode("utf-8")
                ).hexdigest(),
            )
            count = max(1, int(np.ceil(len(candidates) * fraction)))
            if max_groups_per_class is not None:
                count = min(count, max_groups_per_class)
            selected.extend(candidates[:count])
        output[f"{fraction:g}"] = np.asarray(
            sorted(train_indices[index] for index in selected), dtype=np.int64
        )
    return output


class _BatchMetricAccumulator:
    """Accumulate multiclass metrics without retaining predictions."""

    def __init__(self, labels: Sequence[str]) -> None:
        """Initialize a confusion matrix for the fixed label order."""

        self.labels = tuple(labels)
        self._positions = {label: index for index, label in enumerate(labels)}
        self._matrix = np.zeros(
            (len(self.labels), len(self.labels)), dtype=np.int64
        )

    def update(
        self,
        y_true: Sequence[str],
        y_pred: Sequence[str],
    ) -> None:
        """Add one prediction batch to the confusion matrix."""

        if len(y_true) != len(y_pred):
            raise ValueError("Batch metric inputs must have equal length")
        for actual, predicted in zip(y_true, y_pred):
            actual_position = self._positions.get(str(actual))
            predicted_position = self._positions.get(str(predicted))
            if actual_position is None or predicted_position is None:
                raise ValueError("Batch contains a label outside the protocol")
            self._matrix[actual_position, predicted_position] += 1

    def result(self) -> dict[str, Any]:
        """Return metrics equivalent to :func:`calculate_metrics`."""

        support = self._matrix.sum(axis=1)
        predicted = self._matrix.sum(axis=0)
        true_positive = np.diag(self._matrix)
        precision = np.divide(
            true_positive,
            predicted,
            out=np.zeros_like(true_positive, dtype=float),
            where=predicted != 0,
        )
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros_like(true_positive, dtype=float),
            where=support != 0,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision, dtype=float),
            where=(precision + recall) != 0,
        )
        total = int(support.sum())
        if total == 0:
            raise ValueError("Cannot calculate metrics for an empty stream")
        return {
            "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.dot(f1, support) / total),
            "balanced_accuracy": float(recall.mean()),
            "per_class": {
                label: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(self.labels)
            },
            "row_count": total,
        }


def estimate_full_memory_bytes(
    scientific_rows: int,
    narrative_bytes: int,
    config: BaselineConfig,
    batch_size: int,
) -> int:
    """Estimate peak bytes for one full scientific TF-IDF point.

    The estimate is deliberately conservative and uses only Parquet metadata
    and configuration. It does not inspect process memory and does not require
    ``psutil``.

    Args:
        scientific_rows: Number of clean train and validation rows.
        narrative_bytes: UTF-8 narrative bytes in those rows.
        config: TF-IDF configuration.
        batch_size: Operational prediction batch size.

    Returns:
        Conservative byte estimate for the largest working point.
    """

    if scientific_rows <= 0 or narrative_bytes < 0 or batch_size <= 0:
        raise ValueError("Memory estimate inputs must be positive")
    assumed_nonzero = min(512, config.max_features)
    sparse_bytes = scientific_rows * assumed_nonzero * 12
    sparse_bytes += (scientific_rows + 1) * 4
    batch_bytes = batch_size * config.max_features * 4
    text_bytes = narrative_bytes * 3
    return int(text_bytes + sparse_bytes + batch_bytes)


def _frame_texts(frame: Any, indices: Sequence[int]) -> list[str]:
    """Extract only the requested text column for one model point."""

    column = frame["narrative"]
    return [str(column[int(index)].as_py()) for index in indices]


def _frame_labels(frame: Any, indices: Sequence[int]) -> list[str]:
    """Extract only the requested labels for one model point."""

    column = frame["product_family"]
    return [str(column[int(index)].as_py()) for index in indices]


def _evaluate_frame_in_batches(
    frame: Any,
    indices: Sequence[int],
    vectorizer: TfidfVectorizer,
    estimator: Any,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate a scientific Arrow table without retaining its predictions."""

    accumulator = _BatchMetricAccumulator(MODELED_FAMILIES)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        texts = _frame_texts(frame, batch_indices)
        labels = _frame_labels(frame, batch_indices)
        accumulator.update(labels, estimator.predict(vectorizer.transform(texts)))
    return accumulator.result()


def _fit_full_point(
    frame: Any,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    config: BaselineConfig,
    batch_size: int,
    include_operational: bool,
    development_path: Path,
) -> dict[str, Any]:
    """Fit one full learning-curve point with bounded evaluation batches."""

    train_text = _frame_texts(frame, train_indices)
    train_labels = _frame_labels(frame, train_indices)
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=config.ngram_range,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train_text)
    estimators = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "sgd_logistic": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=config.max_iter,
            tol=1e-3,
            random_state=config.random_state,
        ),
    }
    result: dict[str, Any] = {}
    for name, estimator in estimators.items():
        estimator.fit(x_train, train_labels)
        result[name] = {
            "scientific": _evaluate_frame_in_batches(
                frame,
                validation_indices,
                vectorizer,
                estimator,
                batch_size,
            )
        }
        if include_operational:
            operational = _BatchMetricAccumulator(MODELED_FAMILIES)
            for texts, labels in iter_operational_validation_batches(
                development_path, batch_size
            ):
                operational.update(
                    labels,
                    estimator.predict(vectorizer.transform(texts)),
                )
            result[name]["operational_all_text"] = operational.result()
    result["train_rows"] = len(train_indices)
    result["train_groups"] = len(train_indices)
    result["vocabulary_size"] = len(vectorizer.vocabulary_)
    return result


def run_s3_full(
    development_path: str | Path,
    artifact_path: str | Path,
    scientific_cache_path: str | Path | None = None,
    config: BaselineConfig | None = None,
    tracker: Tracker | None = None,
    force_refresh: bool = False,
    batch_size: int = DEFAULT_FULL_BATCH_SIZE,
    memory_budget_gb: float | None = DEFAULT_FULL_MEMORY_BUDGET_GB,
    max_groups_per_class: int | None = None,
    memory_limit: str = "4GB",
    threads: int = 1,
) -> dict[str, Any]:
    """Run the full S3 curve through scientific and operational boundaries.

    The scientific cache is built with DuckDB and loaded as Arrow columns.
    The all-text operational validation is streamed only at the final curve
    point. Test, stress, and monitor are never requested by this function.

    Args:
        development_path: Development-only all-text Parquet cache.
        artifact_path: Incremental JSON result artifact.
        scientific_cache_path: Optional scientific cache destination.
        config: Bounded TF-IDF and classifier settings.
        tracker: Optional experiment tracker.
        force_refresh: Rebuild cache and artifact even when signatures match.
        batch_size: Rows per validation and operational prediction batch.
        memory_budget_gb: Conservative local memory budget; ``None`` disables
            the estimate guard explicitly.
        max_groups_per_class: Optional bounded cap for a diagnostic run.
        memory_limit: DuckDB memory limit for cache construction.
        threads: DuckDB worker count.

    Returns:
        Serializable incremental learning-curve results.

    Raises:
        MemoryError: If the conservative estimate exceeds the configured
            budget.
        ValueError: If an input violates the S3 full-run contract.
    """

    settings = config or BaselineConfig()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if memory_budget_gb is not None and memory_budget_gb <= 0:
        raise ValueError("memory_budget_gb must be positive or None")
    dataset = Path(development_path).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    scientific = (
        Path(scientific_cache_path).expanduser().resolve()
        if scientific_cache_path is not None
        else dataset.with_name("scientific.parquet")
    )
    cache = build_or_load_scientific_cache(
        dataset,
        scientific,
        temp_directory=scientific.parent / "duckdb",
        memory_limit=memory_limit,
        threads=threads,
        force_refresh=force_refresh,
    )
    signature = {
        "scientific_cache": cache["signature"],
        "config": settings.to_dict(),
        "batch_size": batch_size,
        "memory_budget_gb": memory_budget_gb,
        "max_groups_per_class": max_groups_per_class,
    }
    if not force_refresh and artifact.exists():
        try:
            cached = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if cached and cached.get("signature") == signature and cached.get(
            "complete"
        ):
            if _normalize_protocol_boundary(cached):
                _write_json_atomic(artifact, cached)
            return cached
    frame = read_scientific_frame(scientific)
    partitions = [str(value.as_py()) for value in frame["partition_name"]]
    train_indices = np.asarray(
        [index for index, value in enumerate(partitions) if value == "train"],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [
            index
            for index, value in enumerate(partitions)
            if value == "validation"
        ],
        dtype=np.int64,
    )
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Scientific cache needs train and validation rows")
    narrative_bytes = sum(
        len(str(value.as_py()).encode("utf-8"))
        for value in frame["narrative"]
    )
    estimate = estimate_full_memory_bytes(
        len(train_indices) + len(validation_indices),
        narrative_bytes,
        settings,
        batch_size,
    )
    budget_bytes = (
        None
        if memory_budget_gb is None
        else int(memory_budget_gb * 1024**3)
    )
    if budget_bytes is not None and estimate > budget_bytes:
        raise MemoryError(
            f"Estimated S3 full peak is {estimate / 1024**3:.2f} GB, "
            f"above the {memory_budget_gb:.2f} GB budget"
        )
    subsets = build_learning_curve_indices(
        frame,
        settings.fractions,
        settings.random_state,
        max_groups_per_class,
    )
    active_tracker = tracker or NullTracker()
    active_tracker.log_params(settings.to_dict())
    result: dict[str, Any] = {
        "schema_version": S3_SCHEMA_VERSION,
        "signature": signature,
        "complete": False,
        "protocol_boundary": S3_PROTOCOL_BOUNDARY.copy(),
        "group_policy": cache["summary"],
        "scientific_cache": {
            "path": str(scientific),
            "rows": cache["rows"],
            "summary": cache["summary"],
        },
        "memory_guard": {
            "estimate_bytes": estimate,
            "budget_bytes": budget_bytes,
            "narrative_bytes": narrative_bytes,
        },
        "operational_policy": "final_fraction_only_streamed_in_batches",
        "points": {},
    }
    _write_json_atomic(artifact, result)
    try:
        final_key = max(subsets, key=float)
        for fraction, indices in subsets.items():
            point = _fit_full_point(
                frame,
                indices,
                validation_indices,
                settings,
                batch_size,
                fraction == final_key,
                dataset,
            )
            result["points"][fraction] = point
            _write_json_atomic(artifact, result)
        result["complete"] = True
        _write_json_atomic(artifact, result)
        active_tracker.log_metrics(
            {
                "learning_curve_points": float(len(result["points"])),
                "final_scientific_macro_f1": float(
                    result["points"][final_key]["sgd_logistic"]["scientific"][
                        "macro_f1"
                    ]
                ),
            }
        )
        active_tracker.log_artifact(artifact)
        return result
    finally:
        active_tracker.close()


def calculate_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = MODELED_FAMILIES,
) -> dict[str, Any]:
    """Calculate primary and secondary classification metrics."""

    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Metric inputs must have equal non-zero length")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "row_count": len(y_true),
    }


def _stratified_limit_rows(
    rows: Sequence[Mapping[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    """Select a deterministic per-class prefix with a global row limit."""

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["product_family"])].append(row)
    ordered = {
        label: sorted(group, key=lambda row: int(row["Complaint ID"]))
        for label, group in sorted(by_class.items())
    }
    if not ordered:
        return []
    selected: list[dict[str, Any]] = []
    cursors = {label: 0 for label in ordered}
    labels = list(ordered)
    while len(selected) < max_rows:
        progressed = False
        for label in labels:
            cursor = cursors[label]
            if cursor >= len(ordered[label]) or len(selected) >= max_rows:
                continue
            selected.append(dict(ordered[label][cursor]))
            cursors[label] += 1
            progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: int(row["Complaint ID"]))
    return selected


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Bound TF-IDF and SGD-logistic resources for an S3 run."""

    max_features: int = 50_000
    min_df: int = 2
    max_df: float = 0.98
    ngram_range: tuple[int, int] = (1, 2)
    max_iter: int = 1_000
    random_state: int = 42
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly baseline parameters."""

        return asdict(self)


def _fit_point(
    train_rows: Sequence[Mapping[str, Any]],
    validation_scientific: Sequence[Mapping[str, Any]],
    validation_all_text: Sequence[Mapping[str, Any]],
    config: BaselineConfig,
) -> dict[str, Any]:
    """Fit both baselines for one learning-curve point."""

    train_text = [str(row["narrative"]) for row in train_rows]
    train_labels = [str(row["product_family"]) for row in train_rows]
    scientific_text = [str(row["narrative"]) for row in validation_scientific]
    scientific_labels = [
        str(row["product_family"]) for row in validation_scientific
    ]
    operational_text = [str(row["narrative"]) for row in validation_all_text]
    operational_labels = [
        str(row["product_family"]) for row in validation_all_text
    ]
    if not scientific_text or not operational_text:
        raise ValueError("Validation requires scientific and all-text rows")
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=config.ngram_range,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train_text)
    x_scientific = vectorizer.transform(scientific_text)
    x_operational = vectorizer.transform(operational_text)
    estimators = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "sgd_logistic": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=config.max_iter,
            tol=1e-3,
            random_state=config.random_state,
        ),
    }
    result: dict[str, Any] = {}
    for name, estimator in estimators.items():
        estimator.fit(x_train, train_labels)
        result[name] = {
            "scientific": calculate_metrics(
                scientific_labels,
                estimator.predict(x_scientific),
            ),
            "operational_all_text": calculate_metrics(
                operational_labels,
                estimator.predict(x_operational),
            ),
        }
    result["train_rows"] = len(train_rows)
    result["train_groups"] = len(train_rows)
    result["vocabulary_size"] = len(vectorizer.vocabulary_)
    return result


def run_s3_baseline(
    dataset_path: str | Path,
    artifact_path: str | Path,
    config: BaselineConfig | None = None,
    tracker: Tracker | None = None,
    force_refresh: bool = False,
    max_groups_per_class: int | None = None,
    max_validation_rows: int | None = None,
    max_loaded_rows: int | None = DEFAULT_MAX_LOADED_ROWS,
) -> dict[str, Any]:
    """Run or load the S3 learning curve without accessing sealed partitions.

    ``SGDClassifier(loss="log_loss")`` is the scalable linear logistic
    regression baseline used here; it is optimized with stochastic gradient
    descent and is intentionally not named ``LogisticRegression``.

    Args:
        dataset_path: Development-only S3 cache.
        artifact_path: Incremental JSON result artifact.
        config: Bounded TF-IDF and classifier settings.
        tracker: Optional implementation of the project tracker protocol.
        force_refresh: Recompute an existing artifact.
        max_groups_per_class: Optional smoke limit applied to each curve point.
        max_validation_rows: Optional smoke limit for validation rows.
        max_loaded_rows: Guard for list-based loading. Set to ``None`` only
            after explicitly accepting the memory cost of a large load.

    Returns:
        Serializable learning-curve results and ambiguity accounting.
    """

    settings = config or BaselineConfig()
    artifact = Path(artifact_path).expanduser().resolve()
    dataset = Path(dataset_path).expanduser().resolve()
    if max_loaded_rows is not None and max_loaded_rows <= 0:
        raise ValueError("max_loaded_rows must be positive or None")
    with duckdb.connect() as connection:
        total_rows = int(
            connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(dataset)]
            ).fetchone()[0]
        )
    bounded_mode = (
        max_groups_per_class is not None or max_validation_rows is not None
    )
    if (
        not bounded_mode
        and max_loaded_rows is not None
        and total_rows > max_loaded_rows
    ):
        raise ValueError(
            f"S3 dataset has {total_rows:,} rows; refusing list loading above "
            f"{max_loaded_rows:,}. Use SQL/Arrow preparation or explicitly "
            "set max_loaded_rows=None."
        )
    rows = read_development_rows(
        dataset,
        max_rows=max_loaded_rows if bounded_mode else None,
    )
    split = prepare_scientific_split(rows)
    validation_scientific = split["validation_scientific"]
    validation_all_text = split["validation_all_text"]
    if max_validation_rows is not None:
        if max_validation_rows <= 0:
            raise ValueError("max_validation_rows must be positive")
        validation_scientific = _stratified_limit_rows(
            validation_scientific, max_validation_rows
        )
        validation_all_text = _stratified_limit_rows(
            validation_all_text, max_validation_rows
        )
    subsets = build_learning_curve_subsets(
        split["train"], settings.fractions, settings.random_state
    )
    if max_groups_per_class is not None:
        if max_groups_per_class <= 0:
            raise ValueError("max_groups_per_class must be positive")
        limited: dict[str, list[dict[str, Any]]] = {}
        for fraction, subset in subsets.items():
            by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in subset:
                by_class[str(row["product_family"])].append(row)
            limited[fraction] = [
                row
                for label in sorted(by_class)
                for row in by_class[label][:max_groups_per_class]
            ]
        subsets = limited
    signature = {
        "dataset": {
            "path": str(Path(dataset_path).resolve()),
            "size": Path(dataset_path).stat().st_size,
            "mtime_ns": Path(dataset_path).stat().st_mtime_ns,
        },
        "config": settings.to_dict(),
        "max_groups_per_class": max_groups_per_class,
        "max_validation_rows": max_validation_rows,
        "max_loaded_rows": max_loaded_rows,
    }
    cached = None
    if not force_refresh and artifact.exists():
        try:
            cached = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
    if cached and cached.get("signature") == signature and cached.get("complete"):
        if _normalize_protocol_boundary(cached):
            _write_json_atomic(artifact, cached)
        return cached
    active_tracker = tracker or NullTracker()
    active_tracker.log_params(settings.to_dict())
    result: dict[str, Any] = {
        "schema_version": S3_SCHEMA_VERSION,
        "signature": signature,
        "complete": False,
        "protocol_boundary": S3_PROTOCOL_BOUNDARY.copy(),
        "group_policy": split["summary"],
        "points": {},
    }
    _write_json_atomic(artifact, result)
    try:
        for fraction, subset in subsets.items():
            point = _fit_point(
                subset,
                validation_scientific,
                validation_all_text,
                settings,
            )
            result["points"][fraction] = point
            _write_json_atomic(artifact, result)
        result["complete"] = True
        _write_json_atomic(artifact, result)
        final_key = max(result["points"], key=float)
        active_tracker.log_metrics(
            {
                "learning_curve_points": float(len(result["points"])),
                "final_scientific_macro_f1": float(
                    result["points"][final_key]["sgd_logistic"]["scientific"][
                        "macro_f1"
                    ]
                ),
            }
        )
        active_tracker.log_artifact(artifact)
        return result
    finally:
        active_tracker.close()


def run_s3_smoke(
    dataset_path: str | Path,
    artifact_path: str | Path,
) -> dict[str, Any]:
    """Run a bounded S3 smoke baseline suitable for tests and notebooks."""

    return run_s3_baseline(
        dataset_path,
        artifact_path,
        config=BaselineConfig(
            max_features=2_000,
            min_df=1,
            max_iter=100,
            fractions=(1.0,),
        ),
        max_groups_per_class=100,
        max_validation_rows=500,
        max_loaded_rows=20_000,
    )
