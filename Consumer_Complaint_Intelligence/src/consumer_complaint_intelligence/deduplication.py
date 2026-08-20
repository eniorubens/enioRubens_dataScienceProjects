"""Full-corpus duplicate and leakage-risk aggregates."""

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import S0AuditConfig
from .data import query_parquet
from .taxonomy import product_family_case_sql


def _quote_identifier(name: str) -> str:
    """Quote a DuckDB identifier supplied through configuration."""

    return '"' + name.replace('"', '""') + '"'


def _text_sql(column: str) -> str:
    """Return a null-safe text expression for a configured column."""

    return f"CAST({column} AS VARCHAR)"


def _combined_query(config: S0AuditConfig) -> str:
    """Build one query with hash-only materialized deduplication state."""

    text = _quote_identifier(config.text_column)
    date = _quote_identifier(config.date_column)
    product = _quote_identifier(config.product_column)
    issue = _quote_identifier(config.issue_column)
    family = product_family_case_sql(product)
    raw_text = _text_sql(text)
    normalized = (
        "regexp_replace(lower(trim(" + raw_text + ")), "
        "'\\s+', ' ', 'g')"
    )
    return f"""
        WITH text_projection AS (
            SELECT
                EXTRACT(YEAR FROM TRY_CAST({date} AS DATE))::BIGINT AS year,
                TRY_CAST({date} AS DATE) AS received_date,
                coalesce(nullif(trim(CAST({product} AS VARCHAR)), ''),
                    '<NULL>') AS raw_product,
                coalesce({family}, '<UNMAPPED_FAMILY>') AS product_family,
                coalesce(nullif(trim(CAST({issue} AS VARCHAR)), ''),
                    '<NULL>') AS issue,
                {raw_text} AS raw_text,
                {normalized} AS normalized_text
            FROM read_parquet(?)
            WHERE trim(coalesce({raw_text}, '')) <> ''
        ), prepared AS MATERIALIZED (
            SELECT
                md5(raw_text) AS exact_hash,
                length(raw_text)::BIGINT AS exact_length,
                md5(normalized_text) AS normalized_hash,
                length(normalized_text)::BIGINT AS normalized_length,
                year,
                received_date,
                raw_product,
                product_family,
                issue
            FROM text_projection
        ), fingerprints AS MATERIALIZED (
            SELECT
                'exact' AS method,
                exact_hash AS fingerprint,
                exact_length AS fingerprint_length,
                year,
                received_date,
                raw_product,
                product_family,
                issue
            FROM prepared
            UNION ALL
            SELECT
                'normalized' AS method,
                normalized_hash AS fingerprint,
                normalized_length AS fingerprint_length,
                year,
                received_date,
                raw_product,
                product_family,
                issue
            FROM prepared
        ), groups AS MATERIALIZED (
            SELECT
                method,
                fingerprint,
                fingerprint_length,
                count(*)::BIGINT AS group_rows,
                count(DISTINCT year)::BIGINT AS distinct_years,
                count(DISTINCT raw_product)::BIGINT AS distinct_raw_products,
                count(DISTINCT product_family)::BIGINT AS distinct_families,
                count(DISTINCT issue)::BIGINT AS distinct_issues,
                min(year)::BIGINT AS first_year,
                max(year)::BIGINT AS last_year,
                min(received_date) AS first_date,
                max(received_date) AS last_date
            FROM fingerprints
            GROUP BY 1, 2, 3
        ), ranked AS MATERIALIZED (
            SELECT
                groups.*,
                row_number() OVER (
                    PARTITION BY method
                    ORDER BY group_rows DESC, fingerprint
                ) AS method_rank
            FROM groups
            WHERE group_rows > 1
        ), summaries AS (
            SELECT
                method,
                count(*)::BIGINT AS total_duplicate_groups,
                coalesce(sum(group_rows), 0)::BIGINT AS duplicate_group_rows,
                coalesce(sum(group_rows - 1), 0)::BIGINT AS redundant_rows,
                count(*) FILTER (WHERE distinct_years > 1)::BIGINT
                    AS groups_cross_year,
                count(*) FILTER (WHERE distinct_raw_products > 1)::BIGINT
                    AS groups_cross_raw_product,
                count(*) FILTER (WHERE distinct_families > 1)::BIGINT
                    AS groups_cross_family,
                count(*) FILTER (WHERE distinct_issues > 1)::BIGINT
                    AS groups_cross_issue,
                count(*) FILTER (
                    WHERE first_date IS NOT NULL
                        AND last_date IS NOT NULL
                        AND first_date < last_date
                )::BIGINT AS groups_cross_date
            FROM ranked
            GROUP BY method
        )
        SELECT
            'summary' AS row_type,
            method,
            json_object(
                'total_duplicate_groups', total_duplicate_groups,
                'duplicate_group_rows', duplicate_group_rows,
                'redundant_rows', redundant_rows,
                'groups_cross_year', groups_cross_year,
                'groups_cross_raw_product', groups_cross_raw_product,
                'groups_cross_family', groups_cross_family,
                'groups_cross_issue', groups_cross_issue,
                'groups_cross_date', groups_cross_date
            )::VARCHAR AS payload
        FROM summaries
        UNION ALL
        SELECT
            'top' AS row_type,
            method,
            json_object(
                'fingerprint', fingerprint,
                'fingerprint_length', fingerprint_length,
                'group_rows', group_rows,
                'distinct_years', distinct_years,
                'distinct_raw_products', distinct_raw_products,
                'distinct_families', distinct_families,
                'distinct_issues', distinct_issues,
                'first_year', first_year,
                'last_year', last_year,
                'first_date', first_date,
                'last_date', last_date,
                'method_rank', method_rank
            )::VARCHAR AS payload
        FROM ranked
        WHERE method_rank <= {config.top_k}
        ORDER BY row_type, method, payload
    """


def audit_deduplication(
    parquet_path: str | Path,
    temp_directory: str | Path | None = None,
    config: S0AuditConfig | None = None,
) -> dict[str, Any]:
    """Audit exact and normalized narrative groups in one DuckDB scan.

    The materialized CTEs are reused by summary and top-group rows. Python
    receives only aggregate counts, hashes, dates, and categorical metadata.

    Args:
        parquet_path: Processed CFPB Parquet path.
        temp_directory: DuckDB spill directory.
        config: Shared S0 column, memory, thread, and top-k settings.

    Returns:
        JSON-serializable duplicate and leakage-risk evidence.

    Raises:
        FileNotFoundError: If the Parquet path does not exist.
        ValueError: If the configured top-k or thread count is invalid.
    """

    audit_config = config or S0AuditConfig()
    if audit_config.top_k <= 0:
        raise ValueError("top_k must be positive")
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {path}")
    spill = (
        Path(temp_directory)
        if temp_directory
        else path.parents[2] / "temp" / "duckdb"
    )
    spill.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rows = query_parquet(
        path,
        _combined_query(audit_config),
        temp_directory=spill,
        memory_limit=audit_config.duckdb_memory_limit,
        threads=audit_config.duckdb_threads,
    )
    duration = round(perf_counter() - started, 3)
    summary: dict[str, dict[str, Any]] = {}
    top_groups = []
    for row in rows:
        payload = json.loads(str(row["payload"]))
        if row["row_type"] == "summary":
            summary[row["method"]] = payload
        else:
            payload["method"] = row["method"]
            top_groups.append(payload)

    return {
        "policy": {
            "unique_complaint_id": (
                "Integrity check only; it is not text deduplication."
            ),
            "exact_text": "MD5 plus raw text length inside DuckDB.",
            "normalized_text": (
                "Lowercase, trim, and collapse whitespace only."
            ),
            "raw_dataset_preserved": True,
            "modeling_group_id": (
                "Use normalized fingerprint plus normalized length."
            ),
            "is_repeated": "True when the normalized group has more than one row.",
            "future_metrics": ["all_text_operational", "novel_text_purged"],
            "templates_policy": (
                "Near-duplicates beyond this normalization remain a later audit."
            ),
            "collision_risk": (
                "MD5 collisions are theoretical but not impossible; confirm "
                "critical groups with a collision-safe follow-up."
            ),
        },
        "scope": {
            "evidence": "complete_parquet_corpus",
            "narratives_materialized_in_python": False,
            "source": str(path),
            "engine": "DuckDB",
            "memory_limit": audit_config.duckdb_memory_limit,
            "threads": audit_config.duckdb_threads,
            "temp_directory": str(spill),
            "parquet_scans_expected": 1,
        },
        "duration_seconds": duration,
        "summary_by_method": summary,
        "top_groups": top_groups,
        "implication_for_temporal_split": (
            "Never split one normalized fingerprint across train and evaluation. "
            "Publish both operational all-text and novel-text purged metrics."
        ),
    }
