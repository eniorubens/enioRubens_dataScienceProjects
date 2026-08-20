"""Sample and full-corpus S0 audits for CFPB complaint narratives."""

from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl

from .config import S0AuditConfig
from .data import query_parquet, read_parquet_sample


def _as_text(value: Any) -> str | None:
    """Convert a scalar to a JSON-friendly text value."""

    if value is None:
        return None
    return str(value)


def _as_int(value: Any) -> int | None:
    """Convert a numeric scalar to an integer or JSON null."""

    if value is None:
        return None
    return int(value)


def _quote_identifier(name: str) -> str:
    """Quote a DuckDB identifier supplied through audit configuration."""

    return '"' + name.replace('"', '""') + '"'


def _normalize_text_expression(column: str) -> pl.Expr:
    """Build the sample duplicate-normalization expression."""

    return (
        pl.col(column)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _quantile(values: list[int], probability: float) -> int | None:
    """Return a nearest-rank quantile for a non-empty integer list."""

    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(probability * (len(ordered) - 1)))
    return ordered[index]


def audit_narrative_coverage(
    frame: pl.DataFrame,
    text_column: str,
) -> dict[str, Any]:
    """Measure narrative availability and lengths in a bounded frame.

    Args:
        frame: DataFrame limited by the caller.
        text_column: Narrative column name.

    Returns:
        JSON-friendly coverage and length statistics.

    Raises:
        pl.exceptions.ColumnNotFoundError: If ``text_column`` is absent.
    """

    total = frame.height
    text = pl.col(text_column).fill_null("").cast(pl.String)
    non_empty = frame.select(text.str.strip_chars().str.len_chars()).to_series()
    lengths = [int(value) for value in non_empty.to_list() if value > 0]
    coverage = (len(lengths) / total * 100) if total else 0.0
    return {
        "rows": total,
        "narrative_rows": len(lengths),
        "narrative_coverage_pct": round(coverage, 4),
        "narrative_length": {
            "min": min(lengths) if lengths else None,
            "median": _quantile(lengths, 0.5),
            "p90": _quantile(lengths, 0.9),
            "max": max(lengths) if lengths else None,
        },
    }


def audit_taxonomy(
    frame: pl.DataFrame,
    config: S0AuditConfig,
) -> dict[str, Any]:
    """Summarize labels and sampled year coverage in a bounded frame.

    Args:
        frame: DataFrame limited by the caller.
        config: S0 column names and top-label limit.

    Returns:
        JSON-friendly taxonomy summary.

    Raises:
        pl.exceptions.ColumnNotFoundError: If a configured label column is
            absent.
    """

    result: dict[str, Any] = {}
    for name, column in (
        ("product", config.product_column),
        ("issue", config.issue_column),
    ):
        counts = (
            frame.select(pl.col(column).fill_null("<NULL>").alias("label"))
            .group_by("label")
            .len()
            .sort("len", descending=True)
            .head(config.top_k)
        )
        result[name] = {
            "distinct": int(frame.select(pl.col(column).n_unique()).item()),
            "top": [
                {"label": str(row[0]), "rows": int(row[1])}
                for row in counts.iter_rows()
            ],
        }

    years = (
        frame.select(pl.col(config.date_column).cast(pl.String).alias("date"))
        .with_columns(pl.col("date").str.slice(0, 4).alias("year"))
        .group_by("year")
        .len()
        .sort("year")
    )
    result["sampled_years"] = [
        {"year": _as_text(row[0]), "rows": int(row[1])}
        for row in years.iter_rows()
    ]
    return result


def audit_normalized_duplicates_sample(
    frame: pl.DataFrame,
    text_column: str,
    date_column: str,
    top_k: int = 20,
) -> dict[str, Any]:
    """Audit normalized narrative duplicates in a bounded sample only.

    Args:
        frame: DataFrame limited by the caller.
        text_column: Narrative column name.
        date_column: Complaint date column name.
        top_k: Number of duplicate groups retained for inspection.

    Returns:
        JSON-friendly sample duplicate evidence. The total group count is for
        the bounded sample, while ``top_duplicate_groups`` is only a preview.

    Raises:
        pl.exceptions.ColumnNotFoundError: If a configured column is absent.
        ValueError: If ``top_k`` is not positive.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    normalized = "_normalized_narrative"
    prepared = frame.with_columns(
        _normalize_text_expression(text_column).alias(normalized)
    ).filter(pl.col(normalized) != "")
    all_groups = (
        prepared.group_by(normalized)
        .agg(
            pl.len().alias("rows"),
            pl.col(date_column).cast(pl.String).first().alias("first_date"),
        )
        .filter(pl.col("rows") > 1)
    )
    groups = all_groups.sort("rows", descending=True).head(top_k)
    rows_in_top = int(groups.select(pl.col("rows").sum()).item() or 0)
    redundant_in_top = int(
        groups.select((pl.col("rows") - 1).sum()).item() or 0
    )
    return {
        "evidence": "sample_only",
        "normalization": "lowercase_whitespace_trim",
        "non_empty_narratives": prepared.height,
        "total_duplicate_groups": all_groups.height,
        "top_k": top_k,
        "rows_in_top_duplicate_groups": rows_in_top,
        "redundant_rows_in_top_duplicate_groups": redundant_in_top,
        "top_duplicate_groups": [
            {
                "normalized_text_preview": str(row[0])[:160],
                "rows": int(row[1]),
                "first_date": _as_text(row[2]),
            }
            for row in groups.iter_rows()
        ],
    }


def audit_duplicates(
    frame: pl.DataFrame,
    text_column: str,
    date_column: str,
    top_k: int = 20,
) -> dict[str, Any]:
    """Preserve the old name while returning explicitly sample-only evidence.

    Args:
        frame: DataFrame limited by the caller.
        text_column: Narrative column name.
        date_column: Complaint date column name.
        top_k: Number of duplicate groups retained for inspection.

    Returns:
        The sample-only normalized duplicate report.
    """

    return audit_normalized_duplicates_sample(
        frame, text_column, date_column, top_k
    )


def audit_s0_sample(
    parquet_path: str | Path,
    config: S0AuditConfig | None = None,
) -> dict[str, Any]:
    """Run the bounded-head S0 smoke audit, never the corpus audit.

    Args:
        parquet_path: Processed CFPB Parquet path.
        config: Optional bounded-audit configuration.

    Returns:
        Report explicitly labeled as sample-only evidence.

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
        ValueError: If the audit configuration is invalid.
    """

    audit_config = config or S0AuditConfig()
    columns = [
        audit_config.text_column,
        audit_config.date_column,
        audit_config.product_column,
        audit_config.issue_column,
    ]
    frame = read_parquet_sample(
        parquet_path,
        limit=audit_config.sample_rows,
        columns=columns,
    )
    return {
        "audit": "S0_bounded_sample",
        "scope": {
            "mode": "bounded_head_sample",
            "evidence": "sample_only",
            "sampling_method": "first_rows_from_lazy_parquet_scan",
            "rows_scanned": frame.height,
            "max_rows": audit_config.sample_rows,
            "source": str(parquet_path),
        },
        "narrative_coverage": audit_narrative_coverage(
            frame, audit_config.text_column
        ),
        "taxonomy": audit_taxonomy(frame, audit_config),
        "duplicates": audit_normalized_duplicates_sample(
            frame,
            audit_config.text_column,
            audit_config.date_column,
            audit_config.top_k,
        ),
        "decisions_pending": [
            "Consolidate historical product and issue taxonomy.",
            "Use the full-corpus duplicate aggregate before modeling.",
            "Define a temporal split only after taxonomy stability is assessed.",
        ],
        "limitations": [
            "This report is bounded to the configured first rows, not the corpus.",
            "The bounded head is not a random or population-representative sample.",
            "No scientific model training or final split selection occurs in S0.",
        ],
    }


def audit_s0(
    parquet_path: str | Path,
    config: S0AuditConfig | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for the explicitly bounded sample audit.

    Args:
        parquet_path: Processed CFPB Parquet path.
        config: Optional bounded-audit configuration.

    Returns:
        The ``S0_bounded_sample`` report from :func:`audit_s0_sample`.
    """

    return audit_s0_sample(parquet_path, config)


def _timed_query(
    parquet_path: Path,
    query: str,
    temp_directory: Path,
    config: S0AuditConfig,
) -> tuple[list[dict[str, Any]], float]:
    """Execute one bounded-memory DuckDB aggregate and measure its duration."""

    started = perf_counter()
    rows = query_parquet(
        parquet_path,
        query,
        temp_directory=temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )
    return rows, round(perf_counter() - started, 3)


def _taxonomy_corpus_profile(
    rows: list[dict[str, Any]],
    total_rows: int,
) -> dict[str, Any]:
    """Convert year-label aggregate rows into a serializable profile."""

    labels: dict[str, dict[str, Any]] = {}
    counts: list[dict[str, Any]] = []
    years: set[int] = set()
    missing_label_rows = 0
    for row in rows:
        label = str(row["label"])
        year = _as_int(row["year"])
        count = int(row["rows"])
        is_missing = bool(row.get("is_missing", label == "<NULL>"))
        counts.append(
            {
                "year": year,
                "label": label,
                "rows": count,
                "is_missing": is_missing,
            }
        )
        if is_missing:
            missing_label_rows += count
        profile = labels.setdefault(
            label if not is_missing else "<NULL>",
            {"label": label, "total_rows": 0, "years": set()},
        )
        if not is_missing:
            profile["total_rows"] += count
        if year is not None:
            years.add(year)
            if not is_missing:
                profile["years"].add(year)

    labels.pop("<NULL>", None)

    summaries = []
    for profile in labels.values():
        label_years = sorted(profile["years"])
        summaries.append(
            {
                "label": profile["label"],
                "total_rows": profile["total_rows"],
                "first_year": label_years[0] if label_years else None,
                "last_year": label_years[-1] if label_years else None,
                "number_years": len(label_years),
            }
        )
    summaries.sort(key=lambda item: (-item["total_rows"], item["label"]))
    ordered_counts = sorted(
        counts,
        key=lambda item: (
            item["year"] is None,
            item["year"] if item["year"] is not None else 0,
            item["label"],
        ),
    )
    ordered_years = sorted(years)
    return {
        "total_rows": total_rows,
        "distinct_labels": len(labels),
        "missing_label_rows": missing_label_rows,
        "distinct_buckets_including_missing": len(labels)
        + int(missing_label_rows > 0),
        "first_year": ordered_years[0] if ordered_years else None,
        "last_year": ordered_years[-1] if ordered_years else None,
        "number_years": len(ordered_years),
        "label_summaries": summaries,
        "counts_by_year_label": ordered_counts,
    }


def audit_s0_corpus(
    parquet_path: str | Path,
    temp_directory: str | Path | None = None,
    config: S0AuditConfig | None = None,
) -> dict[str, Any]:
    """Run separate DuckDB aggregates over the complete Parquet corpus.

    Narratives are filtered, hashed, grouped, and counted inside DuckDB. No
    narrative text or full-row table is materialized in Python.

    Args:
        parquet_path: Processed CFPB Parquet path.
        temp_directory: DuckDB spill directory, preferably on ``D:``.
        config: Memory, thread, column, and sample settings.

    Returns:
        JSON-serializable corpus-level volume, taxonomy, ID, duplicate, scope,
        and timing results.

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
        ValueError: If the audit configuration is invalid.
        duckdb.Error: If an aggregate query fails.
    """

    audit_config = config or S0AuditConfig()
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {path}")
    spill = Path(temp_directory) if temp_directory else path.parents[2]
    if temp_directory is None:
        spill = spill / "temp" / "duckdb"
    spill.mkdir(parents=True, exist_ok=True)

    text = _quote_identifier(audit_config.text_column)
    date = _quote_identifier(audit_config.date_column)
    product = _quote_identifier(audit_config.product_column)
    issue = _quote_identifier(audit_config.issue_column)
    complaint_id = _quote_identifier("Complaint ID")

    volume_query = f"""
        SELECT
            count(*)::BIGINT AS total_rows,
            count(*) FILTER (
                WHERE trim(coalesce(CAST({text} AS VARCHAR), '')) <> ''
            )::BIGINT AS narrative_rows
        FROM read_parquet(?)
    """
    coverage_query = f"""
        SELECT
            EXTRACT(YEAR FROM TRY_CAST({date} AS DATE))::BIGINT AS year,
            count(*)::BIGINT AS total_rows,
            count(*) FILTER (
                WHERE trim(coalesce(CAST({text} AS VARCHAR), '')) <> ''
            )::BIGINT AS narrative_rows
        FROM read_parquet(?)
        GROUP BY 1
        ORDER BY 1
    """
    taxonomy_template = f"""
        SELECT
            EXTRACT(YEAR FROM TRY_CAST({date} AS DATE))::BIGINT AS year,
            CASE
                WHEN {{column}} IS NULL
                    OR trim(CAST({{column}} AS VARCHAR)) = ''
                    THEN '<NULL>'
                ELSE trim(CAST({{column}} AS VARCHAR))
            END AS label,
            ({{column}} IS NULL OR trim(CAST({{column}} AS VARCHAR)) = '')
                AS is_missing,
            count(*)::BIGINT AS rows
        FROM read_parquet(?)
        GROUP BY 1, 2, 3
        ORDER BY 1, 2
    """
    id_query = f"""
        SELECT
            count(*)::BIGINT AS total_rows,
            count(DISTINCT {complaint_id})::BIGINT AS distinct_non_null_ids,
            count(*) FILTER (WHERE {complaint_id} IS NULL)::BIGINT AS null_id_rows
        FROM read_parquet(?)
    """
    duplicate_query = f"""
        SELECT
            count(*) FILTER (WHERE group_rows > 1)::BIGINT
                AS total_duplicate_groups,
            coalesce(sum(group_rows) FILTER (WHERE group_rows > 1), 0)::BIGINT
                AS duplicate_group_rows,
            coalesce(sum(group_rows - 1) FILTER (WHERE group_rows > 1), 0)::BIGINT
                AS redundant_rows
        FROM (
            SELECT
                length(CAST({text} AS VARCHAR)) AS text_length,
                md5(CAST({text} AS VARCHAR)) AS text_hash,
                count(*)::BIGINT AS group_rows
            FROM read_parquet(?)
            WHERE trim(coalesce(CAST({text} AS VARCHAR), '')) <> ''
            GROUP BY 1, 2
        ) AS narrative_groups
    """

    started = perf_counter()
    volume_rows, volume_seconds = _timed_query(
        path, volume_query, spill, audit_config
    )
    coverage_rows, coverage_seconds = _timed_query(
        path, coverage_query, spill, audit_config
    )
    product_rows, product_seconds = _timed_query(
        path,
        taxonomy_template.format(column=product),
        spill,
        audit_config,
    )
    issue_rows, issue_seconds = _timed_query(
        path,
        taxonomy_template.format(column=issue),
        spill,
        audit_config,
    )
    id_rows, id_seconds = _timed_query(path, id_query, spill, audit_config)
    duplicate_rows, duplicate_seconds = _timed_query(
        path, duplicate_query, spill, audit_config
    )

    volume = volume_rows[0]
    total_rows = int(volume["total_rows"])
    narrative_rows = int(volume["narrative_rows"])
    coverage_by_year = []
    for row in coverage_rows:
        year_total = int(row["total_rows"])
        year_narratives = int(row["narrative_rows"])
        coverage_by_year.append(
            {
                "year": _as_int(row["year"]),
                "total_rows": year_total,
                "narrative_rows": year_narratives,
                "narrative_coverage_pct": round(
                    year_narratives / year_total * 100, 4
                )
                if year_total
                else 0.0,
            }
        )

    ids = id_rows[0]
    distinct_ids = int(ids["distinct_non_null_ids"])
    null_ids = int(ids["null_id_rows"])
    duplicate_summary = duplicate_rows[0]
    return {
        "audit": "S0_full_corpus_aggregate",
        "scope": {
            "mode": "full_corpus_aggregate",
            "evidence": "complete_parquet_corpus",
            "rows_scanned": total_rows,
            "narratives_materialized_in_python": False,
            "source": str(path),
            "engine": "DuckDB",
            "memory_limit": audit_config.duckdb_memory_limit,
            "threads": audit_config.duckdb_threads,
            "temp_directory": str(spill),
        },
        "duration_seconds": round(perf_counter() - started, 3),
        "volume": {
            "total_rows": total_rows,
            "narrative_rows": narrative_rows,
            "narrative_coverage_pct": round(
                narrative_rows / total_rows * 100, 4
            )
            if total_rows
            else 0.0,
            "coverage_by_year": coverage_by_year,
        },
        "taxonomy": {
            "product": _taxonomy_corpus_profile(
                product_rows, total_rows
            ),
            "issue": _taxonomy_corpus_profile(issue_rows, total_rows),
        },
        "complaint_id": {
            "total_rows": int(ids["total_rows"]),
            "distinct_non_null_ids": distinct_ids,
            "null_id_rows": null_ids,
            "redundant_non_null_id_rows": total_rows - distinct_ids - null_ids,
            "is_unique_non_null": total_rows - null_ids == distinct_ids,
        },
        "exact_narrative_duplicates": {
            "method": "md5_plus_text_length_inside_duckdb",
            "total_duplicate_groups": int(
                duplicate_summary["total_duplicate_groups"]
            ),
            "duplicate_group_rows": int(
                duplicate_summary["duplicate_group_rows"]
            ),
            "redundant_rows": int(duplicate_summary["redundant_rows"]),
            "narratives_materialized_in_python": False,
            "collision_risk": (
                "Counts are keyed by MD5 and text length. MD5 collisions are "
                "theoretical but not impossible; confirm critical groups with "
                "a collision-safe follow-up before publication."
            ),
        },
        "query_timings_seconds": {
            "volume": volume_seconds,
            "coverage_by_year": coverage_seconds,
            "product_by_year_label": product_seconds,
            "issue_by_year_label": issue_seconds,
            "complaint_id": id_seconds,
            "exact_narrative_duplicates": duplicate_seconds,
        },
        "limitations": [
            "Taxonomy counts describe historical labels and do not consolidate "
            "renames.",
            "Normalized duplicate detection remains intentionally sample-only.",
            "Hash-key duplicate counts carry the documented MD5 collision risk.",
            "No scientific model training or final split selection occurs in S0.",
        ],
    }
