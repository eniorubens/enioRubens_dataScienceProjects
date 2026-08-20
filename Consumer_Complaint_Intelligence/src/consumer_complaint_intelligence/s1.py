"""S1 taxonomy, deduplication, and report-cache orchestration."""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import S0AuditConfig
from .data import query_parquet_batch
from .deduplication import audit_deduplication
from .taxonomy import TAXONOMY_VERSION, mapping_status_case_sql
from .taxonomy import product_family_case_sql


CACHE_SCHEMA_VERSION = "s1-report-cache-v2"
COMPONENT_CACHE_SCHEMA_VERSION = "s1-component-cache-v1"


SOURCE_URLS = {
    "cfpb_database": (
        "https://www.consumerfinance.gov/data-research/consumer-complaints/"
    ),
    "cfpb_fields": "https://cfpb.github.io/api/ccdb/fields.html",
    "cfpb_august_2023_pdf": (
        "https://files.consumerfinance.gov/f/documents/"
        "cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.pdf"
    ),
    "cfpb_june_2026_notice": (
        "https://www.consumerfinance.gov/about-us/newsroom/"
        "the-cfpb-is-correcting-flaws-to-restore-integrity-and-utility-to-the-"
        "consumer-complaint-system/"
    ),
}


def _quote_identifier(name: str) -> str:
    """Quote a DuckDB identifier supplied through configuration."""

    return '"' + name.replace('"', '""') + '"'


def _prepared_query(config: S0AuditConfig) -> str:
    """Build one narrow materialization without selecting narrative text."""

    date = _quote_identifier(config.date_column)
    product = _quote_identifier(config.product_column)
    issue = _quote_identifier(config.issue_column)
    text = _quote_identifier(config.text_column)
    family = product_family_case_sql(product)
    status = mapping_status_case_sql(product)
    return f"""
        CREATE TEMP TABLE s1_prepared AS
        SELECT
            EXTRACT(YEAR FROM TRY_CAST({date} AS DATE))::BIGINT AS year,
            TRY_CAST({date} AS DATE) AS received_date,
            coalesce(nullif(trim(CAST({product} AS VARCHAR)), ''),
                '<NULL>') AS raw_product,
            {family} AS product_family,
            {status} AS mapping_status,
            coalesce(nullif(trim(CAST({issue} AS VARCHAR)), ''),
                '<NULL>') AS issue,
            trim(coalesce(CAST({text} AS VARCHAR), '')) <> '' AS has_narrative
        FROM read_parquet(?)
    """


def _as_int(value: Any) -> int | None:
    """Convert an aggregate scalar to an integer or JSON null."""

    return None if value is None else int(value)


def _as_date(value: Any) -> str | None:
    """Convert a DuckDB date scalar to a JSON-friendly string."""

    return None if value is None else str(value)


def _default_spill_directory(path: Path) -> Path:
    """Return the project-level DuckDB spill directory for a Parquet path."""

    return path.parents[2] / "temp" / "duckdb"


def _cache_config(config: S0AuditConfig) -> dict[str, Any]:
    """Return configuration fields that can change an S1 report."""

    fields = (
        "top_k",
        "text_column",
        "date_column",
        "product_column",
        "issue_column",
        "duckdb_memory_limit",
        "duckdb_threads",
    )
    return {field: getattr(config, field) for field in fields}


def _cache_signature(
    parquet_path: Path,
    config: S0AuditConfig,
) -> dict[str, Any]:
    """Build a deterministic signature for one source and audit config."""

    stat = parquet_path.stat()
    payload = {
        "resolved_path": str(parquet_path.resolve()),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "taxonomy_version": TAXONOMY_VERSION,
        "config": _cache_config(config),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {"payload": payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def _write_cache(cache_path: Path, envelope: dict[str, Any]) -> None:
    """Write a JSON cache envelope atomically in its target directory."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                envelope,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _component_cache_path(cache_path: Path, component: str) -> Path:
    """Return the sibling cache path for one S1 component."""

    return cache_path.with_name(f"{cache_path.stem}_{component}.json")


def _read_cache(
    cache_path: Path,
    schema_version: str,
    signature: dict[str, Any],
    component: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Read a cache only when schema, signature, and component match."""

    if not cache_path.exists():
        return None, "miss"
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            envelope = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None, "invalid"
    matches = (
        envelope.get("schema_version") == schema_version
        and envelope.get("signature") == signature
        and isinstance(envelope.get("report"), dict)
        and (
            component is None
            or envelope.get("component") == component
        )
    )
    if not matches:
        return None, "invalidated"
    return envelope["report"], "hit"


def _cache_result(
    report: dict[str, Any],
    status: str,
    cache_path: Path,
    signature: dict[str, Any],
    component_statuses: dict[str, str],
) -> dict[str, Any]:
    """Build the stable external result envelope for cached S1 reports."""

    components = {}
    for component, component_status in component_statuses.items():
        component_path = _component_cache_path(cache_path, component)
        components[component] = {
            "status": component_status,
            "path": str(component_path),
            "schema_version": COMPONENT_CACHE_SCHEMA_VERSION,
            "signature": signature,
        }
    return {
        "report": report,
        "cache": {
            "status": status,
            "path": str(cache_path),
            "schema_version": CACHE_SCHEMA_VERSION,
            "signature": signature,
            "components": components,
            "durations_seconds": report["durations_seconds"],
            "parquet_scans_expected": report["parquet_scans_expected"],
        },
    }


def load_or_run_s1(
    parquet_path: str | Path,
    cache_path: str | Path,
    temp_directory: str | Path | None = None,
    config: S0AuditConfig | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load a valid S1 JSON cache or compute and atomically persist it.

    Args:
        parquet_path: Processed CFPB Parquet path.
        cache_path: JSON report cache path.
        temp_directory: DuckDB spill directory, or the project default.
        config: S1 column, top-k, memory, and thread settings.
        force_refresh: Recompute even when a matching cache exists.

    Returns:
        A serializable dictionary with ``report`` and cache status metadata.

    Raises:
        FileNotFoundError: If the Parquet path does not exist.
        OSError: If the cache cannot be read or written.
    """

    audit_config = config or S0AuditConfig()
    path = Path(parquet_path).expanduser().resolve()
    cache = Path(cache_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {path}")
    signature = _cache_signature(path, audit_config)
    final_report, final_status = (None, "refreshed") if force_refresh else (
        _read_cache(cache, CACHE_SCHEMA_VERSION, signature)
    )
    component_names = ("taxonomy", "deduplication")
    component_statuses = {}
    component_reports: dict[str, dict[str, Any]] = {}
    for component in component_names:
        component_path = _component_cache_path(cache, component)
        cached, component_status = (None, "refreshed") if force_refresh else (
            _read_cache(
                component_path,
                COMPONENT_CACHE_SCHEMA_VERSION,
                signature,
                component,
            )
        )
        component_statuses[component] = component_status
        if cached is not None:
            component_reports[component] = cached

    if final_report is not None and not force_refresh:
        return _cache_result(
            final_report,
            "hit",
            cache,
            signature,
            component_statuses,
        )

    def compute_component(component: str) -> dict[str, Any]:
        """Compute and persist one component before continuing."""

        if component == "taxonomy":
            report = audit_s1_taxonomy(path, temp_directory, audit_config)
        else:
            report = audit_deduplication(path, temp_directory, audit_config)
        component_path = _component_cache_path(cache, component)
        _write_cache(
            component_path,
            {
                "schema_version": COMPONENT_CACHE_SCHEMA_VERSION,
                "component": component,
                "signature": signature,
                "report": report,
            },
        )
        return report

    for component in component_names:
        if component not in component_reports:
            component_reports[component] = compute_component(component)

    report = _compose_s1_report(
        component_reports["taxonomy"],
        component_reports["deduplication"],
    )
    _write_cache(
        cache,
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "signature": signature,
            "report": report,
        },
    )
    if force_refresh:
        status = "refreshed"
    elif final_status in {"invalid", "invalidated"}:
        status = final_status
    elif all(value == "hit" for value in component_statuses.values()):
        status = "assembled"
    elif any(value == "hit" for value in component_statuses.values()):
        status = "partial"
    else:
        status = "miss"
    return _cache_result(
        report,
        status,
        cache,
        signature,
        component_statuses,
    )


def _coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert year-label counts into compact JSON-friendly rows."""

    result = []
    for row in rows:
        total = int(row["total_rows"])
        narratives = int(row["narrative_rows"])
        result.append(
            {
                "year": _as_int(row["year"]),
                "label": str(row["label"]),
                "total_rows": total,
                "narrative_rows": narratives,
                "narrative_coverage_pct": round(
                    narratives / total * 100, 4
                )
                if total
                else 0.0,
            }
        )
    return result


def _label_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize first and last years for each raw or stable label."""

    summaries = []
    for row in rows:
        summaries.append(
            {
                "label_level": row["label_level"],
                "label": row["label"],
                "total_rows": int(row["total_rows"]),
                "narrative_rows": int(row["narrative_rows"]),
                "first_year": _as_int(row["first_year"]),
                "last_year": _as_int(row["last_year"]),
            }
        )
    return summaries


def _compose_s1_report(
    taxonomy_report: dict[str, Any],
    dedup_report: dict[str, Any],
    split_cut_date: str | None = None,
    minimum_class_rows: int | None = None,
    minimum_class_years: int | None = None,
) -> dict[str, Any]:
    """Combine component reports while preserving the S1 public shape."""

    taxonomy_duration = float(taxonomy_report.get("duration_seconds", 0.0))
    dedup_duration = float(dedup_report.get("duration_seconds", 0.0))
    taxonomy_scans = int(
        taxonomy_report["scope"].get("parquet_scans_expected", 1)
    )
    dedup_scans = int(
        dedup_report["scope"].get("parquet_scans_expected", 1)
    )
    return {
        "audit": "S1_taxonomy_deduplication",
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy": taxonomy_report,
        "deduplication": dedup_report,
        "decisions": {
            "raw_product": "Preserve original Product and add product_family.",
            "issue": "Preserve raw Issue and use product_family + Issue as key.",
            "normalized_duplicates": (
                "Never split one normalized fingerprint across train/evaluation."
            ),
            "raw_data": "Never delete duplicate rows from the raw dataset.",
        },
        "gate": _gate_report(
            taxonomy_report,
            split_cut_date,
            minimum_class_rows,
            minimum_class_years,
        ),
        "durations_seconds": {
            "taxonomy": taxonomy_duration,
            "deduplication": dedup_duration,
            "total": round(taxonomy_duration + dedup_duration, 3),
        },
        "parquet_scans_expected": {
            "taxonomy": taxonomy_scans,
            "deduplication": dedup_scans,
            "total": taxonomy_scans + dedup_scans,
        },
        "limitations": [
            "The normalized fingerprint does not remove numbers or PII in S1.",
            "Near-duplicate templates remain a later audit, not an automatic purge.",
            "2026 is partial in this snapshot and may reflect process changes.",
            "The June 2026 notice is a review signal, not an exclusion rule.",
            "No model training or final temporal split selection occurs in S1.",
        ],
    }


def audit_s1_taxonomy(
    parquet_path: str | Path,
    temp_directory: str | Path | None = None,
    config: S0AuditConfig | None = None,
) -> dict[str, Any]:
    """Run complete taxonomy aggregates without materializing narratives.

    Args:
        parquet_path: Processed CFPB Parquet path.
        temp_directory: DuckDB spill directory.
        config: Shared S0 column, memory, thread, and top-k settings.

    Returns:
        JSON-serializable raw/family coverage, stability, and issue evidence.

    Raises:
        FileNotFoundError: If the Parquet path does not exist.
    """

    audit_config = config or S0AuditConfig()
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {path}")
    spill = (
        Path(temp_directory)
        if temp_directory
        else _default_spill_directory(path)
    )
    spill.mkdir(parents=True, exist_ok=True)
    setup_query = _prepared_query(audit_config)

    by_year_label_query = """
        SELECT
            year,
            raw_product AS label,
            count(*)::BIGINT AS total_rows,
            count(*) FILTER (WHERE has_narrative)::BIGINT AS narrative_rows
        FROM s1_prepared
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    by_year_family_query = """
        SELECT
            year,
            coalesce(product_family, '<UNMAPPED_FAMILY>') AS label,
            count(*)::BIGINT AS total_rows,
            count(*) FILTER (WHERE has_narrative)::BIGINT AS narrative_rows
        FROM s1_prepared
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    first_last_query = """
        SELECT
            'raw_product' AS label_level,
            raw_product AS label,
            count(*)::BIGINT AS total_rows,
            count(*) FILTER (WHERE has_narrative)::BIGINT AS narrative_rows,
            min(year)::BIGINT AS first_year,
            max(year)::BIGINT AS last_year
        FROM s1_prepared
        GROUP BY 1, 2
        UNION ALL
        SELECT
            'product_family' AS label_level,
            coalesce(product_family, '<UNMAPPED_FAMILY>') AS label,
            count(*)::BIGINT AS total_rows,
            count(*) FILTER (WHERE has_narrative)::BIGINT AS narrative_rows,
            min(year)::BIGINT AS first_year,
            max(year)::BIGINT AS last_year
        FROM s1_prepared
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    mapping_query = """
        SELECT
            mapping_status,
            raw_product,
            count(*)::BIGINT AS total_rows,
            count(*) FILTER (WHERE has_narrative)::BIGINT AS narrative_rows
        FROM s1_prepared
        GROUP BY 1, 2
        ORDER BY mapping_status, total_rows DESC, raw_product
    """
    mapping_total_query = """
        SELECT
            count(*)::BIGINT AS total_rows,
            count(*) FILTER (WHERE mapping_status = 'mapped')::BIGINT
                AS mapped_rows,
            count(*) FILTER (WHERE mapping_status = 'unmapped')::BIGINT
                AS unmapped_rows,
            count(*) FILTER (WHERE has_narrative)::BIGINT AS narrative_rows,
            count(*) FILTER (
                WHERE has_narrative AND mapping_status = 'mapped'
            )::BIGINT AS mapped_narrative_rows,
            count(*) FILTER (
                WHERE has_narrative AND mapping_status = 'unmapped'
            )::BIGINT AS unmapped_narrative_rows
        FROM s1_prepared
    """
    narrative_family_query = """
        SELECT
            coalesce(product_family, '<UNMAPPED_FAMILY>') AS family,
            count(*) FILTER (WHERE has_narrative)::BIGINT AS narrative_rows
        FROM s1_prepared
        GROUP BY 1
        ORDER BY narrative_rows DESC, family
    """
    issue_cardinality_query = """
        SELECT
            count(DISTINCT issue) FILTER (WHERE issue <> '<NULL>')::BIGINT
                AS distinct_issue_raw,
            count(DISTINCT issue)::BIGINT AS distinct_issue_including_missing,
            count(DISTINCT concat(
                coalesce(product_family, '<UNMAPPED_FAMILY>'), ' :: ', issue
            ))::BIGINT AS distinct_family_issue_key
        FROM s1_prepared
    """
    conflict_query = """
        SELECT
            issue,
            count(DISTINCT coalesce(product_family, '<UNMAPPED_FAMILY>'))
                ::BIGINT AS number_families,
            string_agg(
                DISTINCT coalesce(product_family, '<UNMAPPED_FAMILY>'),
                ' | ' ORDER BY coalesce(product_family, '<UNMAPPED_FAMILY>')
            ) AS families,
            count(*)::BIGINT AS total_rows
        FROM s1_prepared
        WHERE issue <> '<NULL>'
        GROUP BY issue
        HAVING count(DISTINCT coalesce(
            product_family, '<UNMAPPED_FAMILY>'
        )) > 1
        ORDER BY number_families DESC, total_rows DESC, issue
    """

    started = perf_counter()
    query_results = query_parquet_batch(
        path,
        setup_query,
        {
            "by_year_raw": by_year_label_query,
            "by_year_family": by_year_family_query,
            "first_last": first_last_query,
            "mapping": mapping_query,
            "mapping_total": mapping_total_query,
            "narrative_family": narrative_family_query,
            "issue_cardinality": issue_cardinality_query,
            "conflicts": conflict_query,
        },
        temp_directory=spill,
        memory_limit=audit_config.duckdb_memory_limit,
        threads=audit_config.duckdb_threads,
    )
    duration = round(perf_counter() - started, 3)
    by_year_raw = query_results["by_year_raw"]
    by_year_family = query_results["by_year_family"]
    first_last = query_results["first_last"]
    mapping_rows = query_results["mapping"]
    mapping_total = query_results["mapping_total"][0]
    narrative_family = query_results["narrative_family"]
    issue_cardinality = query_results["issue_cardinality"][0]
    conflicts = query_results["conflicts"]

    narrative_total = sum(int(row["narrative_rows"]) for row in narrative_family)
    family_distribution = []
    for row in narrative_family:
        narratives = int(row["narrative_rows"])
        family_distribution.append(
            {
                "family": row["family"],
                "narrative_rows": narratives,
                "narrative_share_pct": round(
                    narratives / narrative_total * 100, 4
                )
                if narrative_total
                else 0.0,
            }
        )
    top_shares = [
        item["narrative_share_pct"] for item in family_distribution[:3]
    ]
    mapping_summary = {
        "total_rows": int(mapping_total["total_rows"]),
        "mapped_rows": int(mapping_total["mapped_rows"]),
        "unmapped_rows": int(mapping_total["unmapped_rows"]),
        "mapping_coverage_pct": round(
            int(mapping_total["mapped_rows"])
            / int(mapping_total["total_rows"])
            * 100,
            4,
        ),
        "narrative_rows": int(mapping_total["narrative_rows"]),
        "mapped_narrative_rows": int(mapping_total["mapped_narrative_rows"]),
        "unmapped_narrative_rows": int(
            mapping_total["unmapped_narrative_rows"]
        ),
    }
    conflict_rows = []
    for row in conflicts:
        conflict_rows.append(
            {
                "issue": row["issue"],
                "number_families": int(row["number_families"]),
                "families": row["families"].split(" | "),
                "total_rows": int(row["total_rows"]),
            }
        )
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "product_policy": {
            "raw_product_preserved": True,
            "unknown_mode": "unmapped for audit; strict in modeling code",
            "registered_raw_product_count": 21,
        },
        "coverage_by_year_raw_product": _coverage_rows(by_year_raw),
        "coverage_by_year_product_family": _coverage_rows(by_year_family),
        "first_last_year": _label_summaries(first_last),
        "mapping": {
            "summary": mapping_summary,
            "classes": [
                {
                    "mapping_status": row["mapping_status"],
                    "raw_product": row["raw_product"],
                    "total_rows": int(row["total_rows"]),
                    "narrative_rows": int(row["narrative_rows"]),
                }
                for row in mapping_rows
            ],
        },
        "narrative_distribution_by_family": family_distribution,
        "concentration": {
            "narrative_rows": narrative_total,
            "largest_family": (
                family_distribution[0] if family_distribution else None
            ),
            "largest_family_share_pct": top_shares[0] if top_shares else 0.0,
            "top_3_share_pct": round(sum(top_shares), 4),
        },
        "issue": {
            "distinct_raw_issue": int(issue_cardinality["distinct_issue_raw"]),
            "distinct_raw_issue_including_missing": int(
                issue_cardinality["distinct_issue_including_missing"]
            ),
            "distinct_family_issue_key": int(
                issue_cardinality["distinct_family_issue_key"]
            ),
            "conflicts_across_families": conflict_rows,
            "hierarchical_key": "product_family + Issue",
            "merge_policy": "Preserve raw Issue; no automatic merge in S1.",
        },
        "interpretation": {
            "target_meaning": (
                "Routing to historical form categories, not prevalence of harm."
            ),
            "issue_stability": (
                "Issue values depend on Product and form options changed over time; "
                "use family plus raw Issue as a secondary hierarchical target."
            ),
        },
        "regime_review": {
            "required": True,
            "periods": [2017, 2023, 2025, 2026],
            "credit_reporting_concentration_requires_review": True,
            "year_2026_partial": True,
            "notice_is_not_an_automatic_exclusion_rule": True,
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
        "sources": SOURCE_URLS,
    }


def _gate_report(
    taxonomy_report: dict[str, Any],
    split_cut_date: str | None,
    minimum_class_rows: int | None,
    minimum_class_years: int | None,
) -> dict[str, Any]:
    """Build the conservative S1 gate without inventing split settings."""

    reasons = []
    if split_cut_date is None:
        reasons.append("No temporal split cut date has been configured.")
    if minimum_class_rows is None or minimum_class_years is None:
        reasons.append(
            "Minimum per-class row and year criteria are not both configured."
        )
    if taxonomy_report["mapping"]["summary"]["unmapped_rows"] > 0:
        reasons.append("Unmapped Product labels require an explicit decision.")
    reasons.extend(
        [
            "The 2017 and August 2023 form regimes require review before splitting.",
            "Credit-reporting concentration and the 2025/2026 anomaly require review.",
            "S1 does not select or seal a scientific temporal split.",
        ]
    )
    return {
        "gate_status": "BLOCKED",
        "reasons": reasons,
        "configuration": {
            "split_cut_date": split_cut_date,
            "minimum_class_rows": minimum_class_rows,
            "minimum_class_years": minimum_class_years,
        },
        "next_stage_requirements": [
            "Review 2017 and August 2023 taxonomy regime boundaries.",
            "Quantify credit_reporting concentration by year and narrative coverage.",
            "Investigate 2025/2026 volume and process-integrity anomaly without "
            "automatic exclusion.",
            "Choose a cut date and minimum class criteria explicitly in S2.",
            "Construct group-aware temporal partitions using normalized fingerprints.",
        ],
    }


def audit_s1_corpus(
    parquet_path: str | Path,
    temp_directory: str | Path | None = None,
    config: S0AuditConfig | None = None,
    split_cut_date: str | None = None,
    minimum_class_rows: int | None = None,
    minimum_class_years: int | None = None,
) -> dict[str, Any]:
    """Run the complete S1 audit and return a serializable blocked gate.

    Args:
        parquet_path: Processed CFPB Parquet path.
        temp_directory: DuckDB spill directory.
        config: Shared S0 column, memory, thread, and top-k settings.
        split_cut_date: Deliberately unset until S2 approves it.
        minimum_class_rows: Deliberately unset until S2 defines it.
        minimum_class_years: Deliberately unset until S2 defines it.

    Returns:
        Combined taxonomy, deduplication, decisions, and gate report.

    Raises:
        FileNotFoundError: If the Parquet path does not exist.
        ValueError: If a configured class criterion is not positive.
    """

    for name, value in (
        ("minimum_class_rows", minimum_class_rows),
        ("minimum_class_years", minimum_class_years),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when configured")
    audit_config = config or S0AuditConfig()
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {path}")
    spill = (
        Path(temp_directory)
        if temp_directory
        else _default_spill_directory(path)
    )
    spill.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    taxonomy_report = audit_s1_taxonomy(path, spill, audit_config)
    dedup_report = audit_deduplication(path, spill, audit_config)
    report = _compose_s1_report(
        taxonomy_report,
        dedup_report,
        split_cut_date,
        minimum_class_rows,
        minimum_class_years,
    )
    report["durations_seconds"]["orchestration"] = round(
        perf_counter() - started,
        3,
    )
    return report
