"""Candidate temporal protocols and S2 modeling-index audits."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import duckdb

from .config import S0AuditConfig
from .taxonomy import TAXONOMY_VERSION, product_family_case_sql


FINGERPRINT_VERSION = (
    "lowercase-trim-whitespace-collapse-md5-length-v1"
)
INDEX_SCHEMA_VERSION = "s2-modeling-index-v1"
REPORT_SCHEMA_VERSION = "s2-temporal-report-v2"
RARE_FAMILY = "other_financial_services"
MODELED_FAMILIES = tuple(
    sorted(
        {
            family
            for family in (
                "credit_reporting",
                "debt_collection",
                "mortgage",
                "deposit_accounts",
                "cards_prepaid",
                "money_services",
                "student_loan",
                "consumer_lending",
                "debt_credit_management",
            )
        }
    )
)
INDEX_COLUMNS = (
    "Complaint ID",
    "received_date",
    "raw_product",
    "product_family",
    "raw_issue",
    "normalized_group_hash",
    "normalized_length",
    "is_modeled_family",
)


def _as_date(value: str | date) -> date:
    """Parse one ISO date value."""

    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid ISO date: {value!r}") from error


def _quote_identifier(name: str) -> str:
    """Quote a DuckDB identifier supplied by configuration."""

    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str | Path) -> str:
    """Quote a DuckDB string literal without relying on interpolation safety."""

    return "'" + str(value).replace("'", "''") + "'"


def _group_identity(hash_value: str, normalized_length: int) -> tuple[str, int]:
    """Return the composite normalized group identity."""

    return hash_value, int(normalized_length)


@dataclass(frozen=True, slots=True)
class DateWindow:
    """Represent an inclusive temporal window."""

    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        """Validate the window name and date order."""

        if not self.name.strip():
            raise ValueError("Window name must not be empty")
        start = _as_date(self.start)
        end = _as_date(self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if start > end:
            raise ValueError("Window start must not be after its end")

    def contains(self, value: date) -> bool:
        """Return whether ``value`` belongs to this inclusive window."""

        current = _as_date(value)
        return self.start <= current <= self.end

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable window representation."""

        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TemporalCandidate:
    """Define a complete candidate protocol without sealing it."""

    name: str
    windows: tuple[DateWindow, ...]

    def __post_init__(self) -> None:
        """Validate required partitions and absence of date overlap."""

        required = {"train", "validation", "test"}
        names = [window.name for window in self.windows]
        if len(names) != len(set(names)):
            raise ValueError("Candidate partition names must be unique")
        if not required.issubset(names):
            missing = sorted(required.difference(names))
            raise ValueError(f"Candidate is missing partitions: {missing}")
        for previous, current in zip(self.windows, self.windows[1:]):
            if current.start < previous.start:
                raise ValueError(
                    "Candidate windows must be in chronological order"
                )
            if current.start <= previous.end:
                raise ValueError(
                    f"Overlapping windows: {previous.name} and {current.name}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate representation."""

        return {
            "name": self.name,
            "windows": [window.to_dict() for window in self.windows],
            "limits_are_inclusive": True,
            "overlap_policy": "forbidden",
        }

    def partition_names(self) -> tuple[str, ...]:
        """Return partitions in chronological order."""

        return tuple(window.name for window in self.windows)


@dataclass(frozen=True, slots=True)
class TemporalCriteria:
    """Configure non-confirmatory support and concentration gates."""

    min_train_rows: int = 2_000
    min_train_unique_groups: int = 1_000
    min_validation_novel_unique_groups: int = 500
    min_test_novel_unique_groups: int = 500
    max_largest_family_share: float = 0.80
    required_mapping_coverage: float = 1.0

    def __post_init__(self) -> None:
        """Validate positive support and bounded share criteria."""

        for name in (
            "min_train_rows",
            "min_train_unique_groups",
            "min_validation_novel_unique_groups",
            "min_test_novel_unique_groups",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.max_largest_family_share <= 1:
            raise ValueError("max_largest_family_share must be in (0, 1]")
        if not 0 < self.required_mapping_coverage <= 1:
            raise ValueError("required_mapping_coverage must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable criteria representation."""

        return asdict(self)


def build_criteria_scenarios() -> dict[str, TemporalCriteria]:
    """Return strict and exploratory pilot S2 criteria scenarios.

    The ``pilot`` scenario is an exploratory sensitivity analysis. It is not a
    retroactive relaxation of the strict gate and does not approve a candidate
    or authorize model training. Any protocol approval must happen explicitly
    before training and remain frozen for the scientific evaluation.

    Returns:
        Mapping with the strict default and the exploratory pilot criteria.
    """

    return {
        "strict": TemporalCriteria(),
        "pilot": TemporalCriteria(
            min_train_rows=750,
            min_train_unique_groups=750,
            min_validation_novel_unique_groups=500,
            min_test_novel_unique_groups=500,
            max_largest_family_share=0.80,
        ),
    }


def build_candidates() -> tuple[TemporalCandidate, ...]:
    """Build the three explicit S2 temporal protocol candidates."""

    def window(name: str, start: str, end: str) -> DateWindow:
        return DateWindow(name, _as_date(start), _as_date(end))

    return (
        TemporalCandidate(
            "historical_stress",
            (
                window("train", "2015-01-01", "2022-12-31"),
                window("validation", "2023-01-01", "2023-12-31"),
                window("test", "2024-01-01", "2024-12-31"),
                window("stress", "2025-01-01", "2025-12-31"),
                window("monitor", "2026-01-01", "2026-12-31"),
            ),
        ),
        TemporalCandidate(
            "post_2023_taxonomy",
            (
                window("train", "2023-08-01", "2024-06-30"),
                window("validation", "2024-07-01", "2024-12-31"),
                window("test", "2025-01-01", "2025-06-30"),
                window("stress", "2025-07-01", "2025-12-31"),
                window("monitor", "2026-01-01", "2026-12-31"),
            ),
        ),
        TemporalCandidate(
            "extended_history",
            (
                window("train", "2015-01-01", "2023-12-31"),
                window("validation", "2024-01-01", "2024-12-31"),
                window("test", "2025-01-01", "2025-12-31"),
                window("monitor", "2026-01-01", "2026-12-31"),
            ),
        ),
    )


def _source_signature(
    path: Path,
    config: S0AuditConfig,
) -> dict[str, Any]:
    """Return source metadata used to invalidate the modeling-index cache."""

    stat = path.stat()
    payload = {
        "resolved_path": str(path.resolve()),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "taxonomy_version": TAXONOMY_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "schema_version": INDEX_SCHEMA_VERSION,
        "source_columns": {
            "text": config.text_column,
            "date": config.date_column,
            "product": config.product_column,
            "issue": config.issue_column,
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "payload": payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON document atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _index_query(config: S0AuditConfig, output: Path) -> str:
    """Build the hash-only DuckDB COPY query for the modeling index."""

    date_column = _quote_identifier(config.date_column)
    product_column = _quote_identifier(config.product_column)
    issue_column = _quote_identifier(config.issue_column)
    text_column = _quote_identifier(config.text_column)
    id_column = _quote_identifier("Complaint ID")
    family_sql = product_family_case_sql(product_column)
    raw_text = f"CAST({text_column} AS VARCHAR)"
    normalized = (
        "regexp_replace(lower(trim(" + raw_text + ")), "
        "'\\s+', ' ', 'g')"
    )
    output_literal = _quote_literal(output)
    return f"""
        COPY (
            WITH source_rows AS (
                SELECT
                    {id_column} AS complaint_id,
                    TRY_CAST({date_column} AS DATE) AS received_date,
                    coalesce(nullif(trim(CAST({product_column} AS VARCHAR)), ''),
                        '<NULL>') AS raw_product,
                    {family_sql} AS product_family,
                    coalesce(nullif(trim(CAST({issue_column} AS VARCHAR)), ''),
                        '<NULL>') AS raw_issue,
                    {normalized} AS normalized_text
                FROM read_parquet(?)
                WHERE trim(coalesce({raw_text}, '')) <> ''
            )
            SELECT
                complaint_id AS "Complaint ID",
                received_date,
                raw_product,
                product_family,
                raw_issue,
                md5(normalized_text) AS normalized_group_hash,
                length(normalized_text)::BIGINT AS normalized_length,
                (
                    product_family IS NOT NULL
                    AND product_family <> '{RARE_FAMILY}'
                ) AS is_modeled_family
            FROM source_rows
        ) TO {output_literal}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """


def _validate_index(path: Path, expected_signature: dict[str, Any]) -> int:
    """Validate schema and unique non-null IDs in one index."""

    if not path.exists():
        raise FileNotFoundError(f"Modeling index does not exist: {path}")
    with duckdb.connect() as connection:
        schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
        columns = tuple(row[0] for row in schema)
        if columns != INDEX_COLUMNS:
            raise ValueError(
                f"Unexpected modeling-index columns: {columns!r}"
            )
        result = connection.execute(
            """
            SELECT
                count(*) AS rows,
                count("Complaint ID") AS non_null_ids,
                count(DISTINCT "Complaint ID") AS distinct_ids
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
    rows, non_null_ids, distinct_ids = (int(value) for value in result)
    if rows != non_null_ids or rows != distinct_ids:
        raise ValueError("Modeling index Complaint ID values are not unique")
    return rows


def build_or_load_modeling_index(
    parquet_path: str | Path,
    index_path: str | Path,
    metadata_path: str | Path | None = None,
    temp_directory: str | Path | None = None,
    config: S0AuditConfig | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load or atomically build the hash-only S2 modeling index.

    Args:
        parquet_path: Processed CFPB Parquet source.
        index_path: Target Parquet index path.
        metadata_path: Optional JSON metadata path beside the index.
        temp_directory: DuckDB spill directory.
        config: Source-column and DuckDB settings.
        force_refresh: Ignore matching cache metadata.

    Returns:
        Serializable index status, signature, schema, and row count.

    Raises:
        FileNotFoundError: If the source Parquet does not exist.
        ValueError: If IDs or output schema fail validation.
    """

    source = Path(parquet_path).expanduser().resolve()
    index = Path(index_path).expanduser().resolve()
    metadata = Path(metadata_path).expanduser().resolve() if metadata_path else (
        index.with_suffix(".json")
    )
    if not source.exists():
        raise FileNotFoundError(f"Parquet source does not exist: {source}")
    audit_config = config or S0AuditConfig()
    signature = _source_signature(source, audit_config)
    cached = None
    if not force_refresh and index.exists() and metadata.exists():
        try:
            cached = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
    cache_hit = (
        cached is not None
        and cached.get("schema_version") == INDEX_SCHEMA_VERSION
        and cached.get("signature") == signature
    )
    if cache_hit:
        rows = _validate_index(index, signature)
        return {
            "status": "hit",
            "path": str(index),
            "metadata_path": str(metadata),
            "schema_version": INDEX_SCHEMA_VERSION,
            "signature": signature,
            "columns": list(INDEX_COLUMNS),
            "rows": rows,
        }

    index.parent.mkdir(parents=True, exist_ok=True)
    temporary = index.with_name(f".{index.name}.{os.getpid()}.tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    spill = Path(temp_directory) if temp_directory else index.parents[1] / "duckdb"
    spill.mkdir(parents=True, exist_ok=True)
    try:
        with duckdb.connect() as connection:
            connection.execute(
                f"SET memory_limit = '{audit_config.duckdb_memory_limit}'"
            )
            connection.execute(f"SET threads = {audit_config.duckdb_threads}")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute(f"SET temp_directory = {_quote_literal(spill)}")
            connection.execute(
                _index_query(audit_config, temporary), [str(source)]
            )
        rows = _validate_index(temporary, signature)
        os.replace(temporary, index)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata_payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "signature": signature,
        "columns": list(INDEX_COLUMNS),
        "rows": rows,
        "contains_narrative": False,
        "contains_normalized_text": False,
        "contains_zip_code": False,
    }
    _write_json_atomic(metadata, metadata_payload)
    return {
        "status": "refreshed",
        "path": str(index),
        "metadata_path": str(metadata),
        **metadata_payload,
    }


def _case_partition_sql(candidate: TemporalCandidate) -> str:
    """Build a date-to-partition CASE expression."""

    clauses = []
    for window in candidate.windows:
        clauses.append(
            "WHEN received_date BETWEEN "
            f"DATE '{window.start.isoformat()}' AND "
            f"DATE '{window.end.isoformat()}' THEN '{window.name}'"
        )
    return "CASE " + " ".join(clauses) + " ELSE NULL END"


def _candidate_report(
    connection: duckdb.DuckDBPyConnection,
    candidate: TemporalCandidate,
    criteria: TemporalCriteria,
) -> dict[str, Any]:
    """Audit one candidate from a prepared DuckDB index relation."""

    partition_sql = _case_partition_sql(candidate)
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE candidate_rows AS
        SELECT *, {partition_sql} AS partition_name
        FROM s2_index
        WHERE ({partition_sql}) IS NOT NULL
        """
    )
    order = {
        name: number
        for number, name in enumerate(candidate.partition_names())
    }
    order_sql = "CASE " + " ".join(
        f"WHEN partition_name = '{name}' THEN {number}"
        for name, number in order.items()
    ) + " ELSE NULL END"
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE candidate_tagged AS
        SELECT *,
            {order_sql} AS partition_order,
            min({order_sql}) OVER (
                PARTITION BY normalized_group_hash, normalized_length
            ) AS first_partition_order
        FROM candidate_rows
        """
    )
    rows = connection.execute(
        """
        SELECT
            partition_name,
            product_family,
            count(*)::BIGINT AS all_text_rows,
            count(
                DISTINCT (normalized_group_hash, normalized_length)
            )::BIGINT AS unique_groups,
            count(*) FILTER (
                WHERE first_partition_order < partition_order
            )::BIGINT AS seen_before_rows,
            count(*) FILTER (
                WHERE first_partition_order = partition_order
            )::BIGINT AS novel_text_rows,
            count(
                DISTINCT (normalized_group_hash, normalized_length)
            ) FILTER (
                WHERE first_partition_order = partition_order
            )::BIGINT AS novel_unique_groups,
            count(*) - count(
                DISTINCT (normalized_group_hash, normalized_length)
            )::BIGINT AS repeated_within_partition_rows
        FROM candidate_tagged
        WHERE product_family IS NOT NULL
        GROUP BY partition_name, product_family
        ORDER BY partition_name, product_family
        """
    ).fetchall()
    by_partition: dict[str, dict[str, Any]] = {}
    for name in candidate.partition_names():
        by_partition[name] = {
            "partition": name,
            "all_text_rows": 0,
            "modeled_text_rows": 0,
            "unique_groups": 0,
            "seen_before_rows": 0,
            "purged_seen_before_rows": 0,
            "novel_text_rows": 0,
            "novel_unique_groups": 0,
            "repeated_within_partition_rows": 0,
            "out_of_scope_rare_rows": 0,
            "support_by_family": [],
        }
    for row in rows:
        (
            partition,
            family,
            all_rows,
            unique_groups,
            seen_before,
            novel_rows,
            novel_groups,
            repeated_rows,
        ) = row
        data = by_partition[partition]
        counts = {
            "family": family,
            "all_text_rows": int(all_rows),
            "unique_groups": int(unique_groups),
            "seen_before_rows": int(seen_before),
            "purged_seen_before_rows": int(seen_before),
            "novel_text_rows": int(novel_rows),
            "novel_unique_groups": int(novel_groups),
            "repeated_within_partition_rows": int(repeated_rows),
        }
        data["all_text_rows"] += int(all_rows)
        if family in MODELED_FAMILIES:
            data["modeled_text_rows"] += int(all_rows)
            data["unique_groups"] += int(unique_groups)
            data["seen_before_rows"] += int(seen_before)
            data["purged_seen_before_rows"] += int(seen_before)
            data["novel_text_rows"] += int(novel_rows)
            data["novel_unique_groups"] += int(novel_groups)
            data["repeated_within_partition_rows"] += int(repeated_rows)
        elif family == RARE_FAMILY:
            data["out_of_scope_rare_rows"] += int(all_rows)
        data["support_by_family"].append(counts)
    for data in by_partition.values():
        data["support_by_family"].sort(key=lambda item: item["family"])
        modeled_rows = data["modeled_text_rows"]
        largest = max(
            (
                item["all_text_rows"]
                for item in data["support_by_family"]
                if item["family"] in MODELED_FAMILIES
            ),
            default=0,
        )
        data["largest_family_share"] = (
            largest / modeled_rows if modeled_rows else 0.0
        )
        data["purge_pct"] = (
            data["purged_seen_before_rows"] / modeled_rows * 100
            if modeled_rows
            else 0.0
        )

    for name in candidate.partition_names():
        data = by_partition[name]
        observed = {item["family"] for item in data["support_by_family"]}
        data["missing_modeled_classes"] = [
            family for family in MODELED_FAMILIES if family not in observed
        ]
        for family in MODELED_FAMILIES:
            if family not in observed:
                data["support_by_family"].append(
                    {
                        "family": family,
                        "all_text_rows": 0,
                        "unique_groups": 0,
                        "seen_before_rows": 0,
                        "purged_seen_before_rows": 0,
                        "novel_text_rows": 0,
                        "novel_unique_groups": 0,
                        "repeated_within_partition_rows": 0,
                    }
                )
        data["support_by_family"].sort(key=lambda item: item["family"])
        data["classes_present"] = [
            family for family in MODELED_FAMILIES
            if family not in data["missing_modeled_classes"]
        ]

    coverage = connection.execute(
        """
        SELECT
            count(*)::BIGINT AS scoped_rows,
            count(*) FILTER (WHERE product_family IS NOT NULL)::BIGINT
                AS mapped_rows,
            count(*) FILTER (WHERE product_family = ?)::BIGINT AS rare_rows
        FROM candidate_rows
        """,
        [RARE_FAMILY],
    ).fetchone()
    scoped_rows, mapped_rows, rare_rows = (int(value) for value in coverage)
    mapping_coverage = mapped_rows / scoped_rows if scoped_rows else 0.0
    eligible = []
    for family in MODELED_FAMILIES:
        train = next(
            item for item in by_partition["train"]["support_by_family"]
            if item["family"] == family
        )
        validation = next(
            item for item in by_partition["validation"]["support_by_family"]
            if item["family"] == family
        )
        test = next(
            item for item in by_partition["test"]["support_by_family"]
            if item["family"] == family
        )
        if (
            train["all_text_rows"] >= criteria.min_train_rows
            and train["unique_groups"] >= criteria.min_train_unique_groups
            and validation["novel_unique_groups"]
            >= criteria.min_validation_novel_unique_groups
            and test["novel_unique_groups"]
            >= criteria.min_test_novel_unique_groups
        ):
            eligible.append(family)

    criterion_results = {
        "mapping_coverage_100_pct": {
            "pass": mapping_coverage >= criteria.required_mapping_coverage,
            "observed": mapping_coverage,
            "required": criteria.required_mapping_coverage,
        },
        "train_min_rows_per_class": {
            "pass": all(
                next(
                    item for item in by_partition["train"]["support_by_family"]
                    if item["family"] == family
                )["all_text_rows"] >= criteria.min_train_rows
                for family in MODELED_FAMILIES
            ),
            "required": criteria.min_train_rows,
        },
        "train_min_unique_groups_per_class": {
            "pass": all(
                next(
                    item for item in by_partition["train"]["support_by_family"]
                    if item["family"] == family
                )["unique_groups"] >= criteria.min_train_unique_groups
                for family in MODELED_FAMILIES
            ),
            "required": criteria.min_train_unique_groups,
        },
        "validation_min_novel_unique_groups_per_class": {
            "pass": all(
                next(
                    item
                    for item in by_partition["validation"]["support_by_family"]
                    if item["family"] == family
                )["novel_unique_groups"]
                >= criteria.min_validation_novel_unique_groups
                for family in MODELED_FAMILIES
            ),
            "required": criteria.min_validation_novel_unique_groups,
        },
        "test_min_novel_unique_groups_per_class": {
            "pass": all(
                next(
                    item for item in by_partition["test"]["support_by_family"]
                    if item["family"] == family
                )["novel_unique_groups"]
                >= criteria.min_test_novel_unique_groups
                for family in MODELED_FAMILIES
            ),
            "required": criteria.min_test_novel_unique_groups,
        },
        "classes_present_train_validation_test": all(
            not by_partition[name]["missing_modeled_classes"]
            for name in ("train", "validation", "test")
        ),
        "largest_family_share_limit": all(
            by_partition[name]["largest_family_share"]
            <= criteria.max_largest_family_share
            for name in ("train", "validation", "test")
        ),
    }
    passed = all(
        value["pass"] if isinstance(value, dict) else value
        for value in criterion_results.values()
    )
    test_data = by_partition["test"]
    return {
        "candidate": candidate.to_dict(),
        "candidate_status": "PASS" if passed else "FAIL",
        "partitions": list(by_partition.values()),
        "mapping": {
            "scoped_rows": scoped_rows,
            "mapped_rows": mapped_rows,
            "coverage": mapping_coverage,
            "rare_rows_out_of_scope": rare_rows,
            "rare_family": RARE_FAMILY,
        },
        "criterion_results": criterion_results,
        "eligible_classes": eligible,
        "eligible_class_count": len(eligible),
        "min_novel_unique_groups_test": min(
            next(
                item for item in test_data["support_by_family"]
                if item["family"] == family
            )["novel_unique_groups"]
            for family in MODELED_FAMILIES
        ),
        "test_purge_pct": test_data["purge_pct"],
        "limitations": [
            "Stress and monitor partitions are diagnostic and do not approve a model.",
            "Novel groups are the scientific unit; all-text metrics remain "
            "operational.",
            "The 2.24pp approximation for n=500 assumes approximately "
            "independent groups.",
            "Group dependence still requires group-weighted or one-observation "
            "per-group metrics.",
        ],
    }


def _report_signature(
    index_signature: dict[str, Any],
    candidates: Iterable[TemporalCandidate],
    criteria: TemporalCriteria,
) -> dict[str, Any]:
    """Build a deterministic signature for the incremental S2 report cache."""

    payload = {
        "index_signature": index_signature,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "criteria": criteria.to_dict(),
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "payload": payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def audit_temporal_candidates(
    index_path: str | Path,
    candidates: Iterable[TemporalCandidate] | None = None,
    criteria: TemporalCriteria | None = None,
    temp_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Audit temporal candidates from the hash-only modeling index.

    Args:
        index_path: S2 modeling-index Parquet path.
        candidates: Candidate protocols, defaulting to the three S2 options.
        criteria: Configurable support and concentration thresholds.
        temp_directory: DuckDB spill directory.

    Returns:
        A serializable comparison report with no model or sealed holdout.

    Raises:
        FileNotFoundError: If the modeling index does not exist.
        ValueError: If the index schema or IDs are invalid.
    """

    index = Path(index_path).expanduser().resolve()
    if not index.exists():
        raise FileNotFoundError(f"Modeling index does not exist: {index}")
    selected = tuple(candidates or build_candidates())
    selected_criteria = criteria or TemporalCriteria()
    _validate_index(index, {})
    spill = Path(temp_directory) if temp_directory else index.parents[1] / "duckdb"
    spill.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit = '2GB'")
        connection.execute("SET threads = 2")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(f"SET temp_directory = {_quote_literal(spill)}")
        connection.execute(
            "CREATE TEMP TABLE s2_index AS SELECT * FROM read_parquet(?)",
            [str(index)],
        )
        reports = [
            _candidate_report(connection, candidate, selected_criteria)
            for candidate in selected
        ]
    passing = [report for report in reports if report["candidate_status"] == "PASS"]
    if passing:
        recommended = sorted(
            passing,
            key=lambda report: (
                -report["eligible_class_count"],
                -report["min_novel_unique_groups_test"],
                report["test_purge_pct"],
                0 if report["candidate"]["name"] == "historical_stress" else 1,
            ),
        )[0]
        recommendation_status = "READY_FOR_REVIEW"
        recommended_name = recommended["candidate"]["name"]
    else:
        recommendation_status = "BLOCKED"
        recommended_name = None
    return {
        "audit": "S2_temporal_protocol_candidates",
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "modeled_families": list(MODELED_FAMILIES),
        "rare_family_policy": {
            "family": RARE_FAMILY,
            "status": "out_of_scope_rare_abstention",
            "kept_in_counts": True,
        },
        "criteria": selected_criteria.to_dict(),
        "candidates": reports,
        "recommendation_status": recommendation_status,
        "recommended_candidate": recommended_name,
        "recommendation_basis": [
            "PASS candidates only",
            "most eligible classes",
            "highest minimum novel unique-group test support",
            "lowest test purge percentage",
            "historical_stress wins the final tie-break",
        ],
        "status_is_not_sealed": True,
        "next_steps": [
            "Review the candidate matrix and taxonomy regime assumptions.",
            "Choose a protocol explicitly before any model training.",
            "Use group-aware or one-observation-per-novel-group scientific metrics.",
            "Keep all-text operational metrics beside novel-group metrics.",
        ],
    }


def load_or_run_s2(
    parquet_path: str | Path,
    index_path: str | Path,
    report_path: str | Path,
    temp_directory: str | Path | None = None,
    criteria: TemporalCriteria | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Build the S2 index and load or cache its temporal audit report.

    This orchestration never trains a model and never marks a holdout sealed.

    Args:
        parquet_path: Processed CFPB Parquet source.
        index_path: S2 modeling-index target path.
        report_path: JSON report cache path.
        temp_directory: DuckDB spill directory.
        criteria: Configurable S2 thresholds.
        force_refresh: Rebuild both caches.

    Returns:
        Serializable cache metadata and S2 report.
    """

    index_result = build_or_load_modeling_index(
        parquet_path,
        index_path,
        temp_directory=temp_directory,
        force_refresh=force_refresh,
    )
    selected_criteria = criteria or TemporalCriteria()
    candidates = build_candidates()
    signature = _report_signature(
        index_result["signature"], candidates, selected_criteria
    )
    report = Path(report_path).expanduser().resolve()
    cached = None
    if not force_refresh and report.exists():
        try:
            cached = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
    if (
        cached is not None
        and cached.get("report_schema_version") == REPORT_SCHEMA_VERSION
        and cached.get("signature") == signature
        and isinstance(cached.get("report"), dict)
    ):
        return {
            "cache": {"status": "hit", "path": str(report)},
            "index": index_result,
            "report": cached["report"],
        }
    result = audit_temporal_candidates(
        index_result["path"],
        candidates=candidates,
        criteria=selected_criteria,
        temp_directory=temp_directory,
    )
    _write_json_atomic(
        report,
        {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "signature": signature,
            "report": result,
        },
    )
    return {
        "cache": {"status": "refreshed", "path": str(report)},
        "index": index_result,
        "report": result,
    }
