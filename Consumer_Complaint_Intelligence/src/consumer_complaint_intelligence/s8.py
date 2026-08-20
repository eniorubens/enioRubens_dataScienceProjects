"""S8 single-use confirmatory evaluation with sealed-data safeguards."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np

from .s7 import load_s7_predictor
from .temporal_split import MODELED_FAMILIES


S8_SCHEMA_VERSION = "s8-confirmatory-protocol-v1"
S8_RESULT_SCHEMA = "s8-confirmatory-results-v1"
S8_MANIFEST_SCHEMA = "s8-confirmatory-manifest-v1"
S8_CODE_SCHEMA = "s8-runtime-v2"
CRITICAL_CLASS = "debt_credit_management"
INPUT_LANGUAGE = "en-US"
S8_UNLOCK_ENV = "S8_CONFIRMATORY_UNLOCK"
SEALED_PARTITIONS = ("stress", "monitor")
TEST_PARTITION = "test"
DEFAULT_BATCH_SIZE = 4096
DEFAULT_MEMORY_LIMIT = "4GB"
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CONFIDENCE_LEVEL = 0.95
UNLOCK_SHA256 = (
    "FBBB77B06EE1EE6E1E98AE82BC93D410EE32408702C5E7DBD61FC6771551D03C"
)
DEFAULT_CONFIG = Path("config/s8_confirmatory_protocol.json")
DEFAULT_RESULT = Path("temp/s8/s8_results.json")
DEFAULT_MANIFEST = Path("config/s8_results.json")
DEFAULT_BUNDLE = Path("artifacts/s7/consumer_complaint_classifier_s7.joblib")
FORBIDDEN_RESULT_KEY_TERMS = (
    "narrative",
    "text",
    "texts",
    "complaint id",
    "complaint_id",
    "individual score",
    "individual_score",
)


def _sha256(path: Path) -> str:
    """Return one file's uppercase SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _file_signature(path: Path) -> dict[str, Any]:
    """Return the portable size and digest metadata for one file."""

    return {
        "path": str(path).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_expected_signature(
    path: Path,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Validate one file's frozen digest and optional byte size."""

    if not path.exists():
        raise FileNotFoundError(f"Frozen {label} is missing: {path}")
    actual = _file_signature(path)
    if actual["sha256"] != str(expected["sha256"]).upper():
        raise ValueError(f"Frozen {label} hash changed")
    expected_size = expected.get("size_bytes")
    if expected_size is not None and actual["size_bytes"] != int(expected_size):
        raise ValueError(f"Frozen {label} size changed")
    return actual


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object through a same-directory atomic replacement."""

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


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject non-object payloads."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S8 JSON artifact must be an object")
    return payload


def _relative(path: Path, root: Path) -> str:
    """Return a portable path relative to the project root."""

    return path.resolve().relative_to(root.resolve()).as_posix()


def _quote_identifier(identifier: str) -> str:
    """Quote a configured DuckDB identifier."""

    if not identifier or "\x00" in identifier:
        raise ValueError("Invalid DuckDB identifier")
    return '"' + identifier.replace('"', '""') + '"'


class S8Config:
    """Represent the frozen S8 protocol and its data-access boundary."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        """Initialize a validated configuration payload."""

        self.payload = dict(payload)
        self.validate_shape()

    @property
    def source(self) -> Mapping[str, Any]:
        """Return frozen source paths, columns, and hashes."""

        return self.payload["source"]

    @property
    def s7_freeze(self) -> Mapping[str, Any]:
        """Return the four frozen S7 artifact signatures."""

        return self.payload["s7_freeze"]

    @property
    def gates(self) -> Mapping[str, float]:
        """Return the simultaneous scientific gates."""

        return self.payload["gates"]

    @property
    def test_scope(self) -> Mapping[str, str]:
        """Return the single confirmatory test window."""

        return self.payload["test_scope"]

    def validate_shape(self) -> None:
        """Validate immutable protocol values without reading dataset rows."""

        if self.payload.get("schema_version") != S8_SCHEMA_VERSION:
            raise ValueError("Unexpected S8 protocol schema")
        if self.payload.get("status") != "FROZEN_FOR_CONFIRMATORY_TEST":
            raise ValueError("S8 protocol is not frozen")
        if self.payload.get("approved_on") != "2026-08-16":
            raise ValueError("Unexpected S8 approval date")
        if self.payload.get("approved_by") != "user":
            raise ValueError("S8 approval must be explicit user approval")
        if self.payload.get("confirmatory_partition") != TEST_PARTITION:
            raise ValueError("S8 must use the test partition")
        expected_scope = {"start": "2025-01-01", "end": "2025-06-30"}
        if dict(self.test_scope) != expected_scope:
            raise ValueError("S8 test scope differs from the frozen window")
        if tuple(self.payload.get("remaining_sealed", ())) != SEALED_PARTITIONS:
            raise ValueError("S8 sealed boundary is invalid")
        if self.payload.get("model", {}).get("input_language") != INPUT_LANGUAGE:
            raise ValueError("S8 input language must be en-US")
        model = self.payload.get("model", {})
        if tuple(model.get("classes", ())) != tuple(MODELED_FAMILIES):
            raise ValueError("S8 class order differs from MODELED_FAMILIES")
        if model.get("critical_class") != CRITICAL_CLASS:
            raise ValueError("S8 critical class is invalid")
        if float(model.get("threshold")) != 0.1135351095114484:
            raise ValueError("S8 threshold differs from frozen S7")
        if model.get("score_kind") != "critical_margin":
            raise ValueError("S8 score kind is invalid")
        expected_gates = {
            "macro_f1_min": 0.69,
            "critical_f1_min": 0.2715,
            "critical_precision_min": 0.2,
        }
        if dict(self.gates) != expected_gates:
            raise ValueError("S8 scientific gates differ from the contract")
        bootstrap = self.payload.get("bootstrap", {})
        if bootstrap != {
            "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "diagnostic_only": True,
        }:
            raise ValueError("S8 bootstrap configuration is invalid")
        access = self.payload.get("access", {})
        if access.get("batch_size") != DEFAULT_BATCH_SIZE:
            raise ValueError("S8 batch size is invalid")
        if access.get("memory_limit") != DEFAULT_MEMORY_LIMIT:
            raise ValueError("S8 memory limit is invalid")
        if access.get("threads") != 1:
            raise ValueError("S8 must use one DuckDB thread")
        if access.get("unlock_env") != S8_UNLOCK_ENV:
            raise ValueError("S8 unlock environment variable is invalid")
        if access.get("unlock_sha256") != UNLOCK_SHA256:
            raise ValueError("S8 unlock digest is invalid")
        temp_directory = access.get("temp_directory")
        if not isinstance(temp_directory, str) or Path(temp_directory).is_absolute():
            raise ValueError("S8 DuckDB temp directory must be relative")
        if ".." in Path(temp_directory).parts:
            raise ValueError("S8 DuckDB temp directory escapes the project")
        for key in (
            "source_path",
            "index_path",
            "s2_report",
            "s3_protocol",
            "s7_config",
            "s7_manifest",
            "s7_result",
            "s7_bundle",
        ):
            if not isinstance(self.payload.get("paths", {}).get(key), str):
                raise ValueError(f"S8 path is missing: {key}")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached serializable configuration."""

        return json.loads(json.dumps(self.payload))


def load_s8_config(path: str | Path = DEFAULT_CONFIG) -> S8Config:
    """Load and validate the frozen S8 protocol configuration."""

    return S8Config(_read_json(Path(path).expanduser().resolve()))


def validate_frozen_metadata(
    config: S8Config,
    project_root: str | Path,
    *,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Validate frozen metadata and hashes without querying dataset rows.

    Args:
        config: Frozen S8 configuration.
        project_root: Project directory containing the relative artifacts.
        include_raw: Permit raw-source metadata hashing after the unlock guard.

    Returns:
        Validated file signatures and S2 evidence metadata.
    """

    root = Path(project_root).expanduser().resolve()
    signatures: dict[str, Any] = {}
    for key, expected in config.s7_freeze.items():
        path = root / str(expected["path"])
        signatures[key] = _validate_expected_signature(path, expected, f"S7 {key}")
    source = config.source
    source_path = root / config.payload["paths"]["source_path"]
    index_path = root / config.payload["paths"]["index_path"]
    _validate_expected_signature(
        index_path,
        {
            "sha256": source["index_sha256"],
            "size_bytes": source["index_size_bytes"],
        },
        "S2 modeling index",
    )
    s2_report = root / config.payload["paths"]["s2_report"]
    _validate_expected_signature(
        s2_report,
        {
            "sha256": source["s2_report_sha256"],
            "size_bytes": source["s2_report_size_bytes"],
        },
        "S2 pilot report",
    )
    report = _read_json(s2_report)
    s3_protocol = root / config.payload["paths"]["s3_protocol"]
    _validate_expected_signature(
        s3_protocol,
        {
            "sha256": source["s3_protocol_sha256"],
            "size_bytes": source["s3_protocol_size_bytes"],
        },
        "S3 protocol",
    )
    evidence = config.payload["s2_evidence"]
    candidate = next(
        (
            item
            for item in report.get("report", report).get("candidates", [])
            if item.get("candidate", {}).get("name") == "post_2023_taxonomy"
        ),
        None,
    )
    if not candidate or candidate.get("candidate_status") != "PASS":
        raise ValueError("S2 post_2023_taxonomy evidence is not PASS")
    if candidate.get("eligible_class_count") != len(MODELED_FAMILIES):
        raise ValueError("S2 evidence does not cover nine classes")
    if include_raw:
        raw = _validate_expected_signature(
            source_path,
            {
                "sha256": source["raw_sha256"],
                "size_bytes": source["raw_size_bytes"],
            },
            "raw CFPB source",
        )
        signatures["raw_source"] = raw
    return {
        "s7": signatures,
        "s2": {
            "protocol_id": "post_2023_taxonomy",
            "candidate_status": candidate["candidate_status"],
            "eligible_class_count": candidate["eligible_class_count"],
            "s3_protocol_sha256": source["s3_protocol_sha256"],
            "evidence": evidence,
        },
    }


def require_confirmatory_unlock(config: S8Config) -> None:
    """Require the one-time full-run token without exposing its plaintext."""

    token = os.environ.get(S8_UNLOCK_ENV)
    if not token or hashlib.sha256(token.encode("utf-8")).hexdigest().upper() != (
        config.payload["access"]["unlock_sha256"]
    ):
        raise PermissionError("S8 full mode requires the approved unlock token")


def apply_threshold(
    scores: Sequence[Sequence[float]],
    threshold: float,
    classes: Sequence[str] = MODELED_FAMILIES,
) -> list[str]:
    """Apply the frozen critical-margin override to decision scores.

    Args:
        scores: Rows ordered by the estimator class columns.
        threshold: Frozen critical margin threshold.
        classes: Frozen class order.

    Returns:
        One predicted class per score row.
    """

    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(classes):
        raise ValueError("S8 scores do not match the frozen class order")
    critical_index = tuple(classes).index(CRITICAL_CLASS)
    noncritical = np.delete(matrix, critical_index, axis=1)
    noncritical_indices = [i for i in range(len(classes)) if i != critical_index]
    labels: list[str] = []
    for row, alternatives in zip(matrix, noncritical):
        best = noncritical_indices[int(np.argmax(alternatives))]
        margin = float(row[critical_index] - np.max(alternatives))
        labels.append(CRITICAL_CLASS if margin >= threshold else classes[best])
    return labels


def metrics_from_confusion(
    confusion: Sequence[Sequence[int]],
    classes: Sequence[str] = MODELED_FAMILIES,
) -> dict[str, Any]:
    """Calculate aggregate and per-class metrics from a confusion matrix."""

    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)):
        raise ValueError("Confusion matrix shape does not match classes")
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, label in enumerate(classes):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        recalls.append(recall)
    row_count = int(matrix.sum())
    weights = matrix.sum(axis=1)
    weighted_f1 = sum(
        per_class[label]["f1"] * int(weights[i])
        for i, label in enumerate(classes)
    ) / row_count if row_count else 0.0
    critical = per_class[CRITICAL_CLASS]
    return {
        "row_count": row_count,
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(np.mean(recalls)),
        "critical_precision": float(critical["precision"]),
        "critical_recall": float(critical["recall"]),
        "critical_f1": float(critical["f1"]),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_gates(
    metrics: Mapping[str, Any], gates: Mapping[str, float]
) -> dict[str, Any]:
    """Evaluate the three exact and simultaneous scientific gates."""

    checks = {
        "macro_f1": float(metrics["macro_f1"]) >= float(gates["macro_f1_min"]),
        "critical_f1": float(metrics["critical_f1"])
        >= float(gates["critical_f1_min"]),
        "critical_precision": float(metrics["critical_precision"])
        >= float(gates["critical_precision_min"]),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "gate_count": sum(checks.values()),
        "limits": dict(gates),
        "values": {
            key: float(metrics[value])
            for key, value in {
                "macro_f1": "macro_f1",
                "critical_f1": "critical_f1",
                "critical_precision": "critical_precision",
            }.items()
        },
    }


def bootstrap_confidence_intervals(
    confusion: Sequence[Sequence[int]],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Bootstrap metrics by stratified multinomial resampling of rows.

    The procedure is diagnostic only: it never changes gate decisions.
    """

    if replicates <= 0 or not 0 < confidence_level < 1:
        raise ValueError("Invalid S8 bootstrap parameters")
    matrix = np.asarray(confusion, dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = np.empty((replicates, 3), dtype=float)
    for replicate in range(replicates):
        sampled = np.zeros_like(matrix)
        for row_index, counts in enumerate(matrix):
            support = int(counts.sum())
            if support:
                sampled[row_index] = rng.multinomial(support, counts / support)
        metrics = metrics_from_confusion(sampled)
        values[replicate] = [
            metrics["macro_f1"],
            metrics["critical_f1"],
            metrics["critical_precision"],
        ]
    alpha = (1 - confidence_level) / 2
    labels = ("macro_f1", "critical_f1", "critical_precision")
    return {
        "method": "stratified_multinomial_from_confusion_matrix",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "diagnostic_only": True,
        "intervals": {
            label: {
                "lower": float(np.quantile(values[:, index], alpha)),
                "upper": float(np.quantile(values[:, index], 1 - alpha)),
            }
            for index, label in enumerate(labels)
        },
    }


def _scope_counts(
    connection: duckdb.DuckDBPyConnection,
    index_path: Path,
    config: S8Config,
) -> dict[str, Any]:
    """Materialize only hash and label scope state in DuckDB."""

    source = config.source
    families = ", ".join("?" for _ in MODELED_FAMILIES)
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE s8_index AS
        SELECT "Complaint ID", received_date, product_family,
            normalized_group_hash, normalized_length
        FROM read_parquet(?)
        WHERE product_family IN (""" + families + ")",
        [str(index_path), *MODELED_FAMILIES],
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE s8_test_base AS
        SELECT i.*
        FROM s8_index AS i
        WHERE TRY_CAST(i.received_date AS DATE) BETWEEN ? AND ?
          AND NOT EXISTS (
            SELECT 1 FROM s8_index AS prior
            WHERE TRY_CAST(prior.received_date AS DATE) < ?
              AND prior.normalized_group_hash = i.normalized_group_hash
              AND prior.normalized_length = i.normalized_length
          )
        """,
        [
            source["test_start"],
            source["test_end"],
            source["test_start"],
        ],
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE s8_clean_groups AS
        SELECT normalized_group_hash, normalized_length,
            min(product_family) AS product_family
        FROM s8_test_base
        GROUP BY normalized_group_hash, normalized_length
        HAVING count(DISTINCT product_family) = 1
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE s8_primary AS
        SELECT * EXCLUDE (row_number)
        FROM (
            SELECT b.*, row_number() OVER (
                PARTITION BY b.normalized_group_hash, b.normalized_length
                ORDER BY b."Complaint ID"
            ) AS row_number
            FROM s8_test_base AS b
            INNER JOIN s8_clean_groups AS g USING (
                normalized_group_hash, normalized_length
            )
        )
        WHERE row_number = 1
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE s8_operational AS
        SELECT b.* FROM s8_test_base AS b
        INNER JOIN s8_clean_groups AS g USING (
            normalized_group_hash, normalized_length
        )
        """
    )
    result = connection.execute(
        """
        SELECT
            (SELECT count(*) FROM s8_test_base) AS novel_test_lines,
            (SELECT count(*) FROM s8_primary) AS primary_representatives,
            (SELECT count(*) FROM s8_operational) AS operational_lines,
            (SELECT count(DISTINCT (normalized_group_hash, normalized_length))
             FROM s8_test_base) AS novel_unique_groups,
            (SELECT count(*) FROM s8_clean_groups) AS clean_unique_groups,
            (SELECT count(*) FROM s8_clean_groups
             WHERE product_family = ?) AS critical_novel_groups,
            (SELECT count(DISTINCT (normalized_group_hash, normalized_length))
             FROM s8_index
             WHERE TRY_CAST(received_date AS DATE) BETWEEN ? AND ?)
             AS all_test_unique_groups,
            (SELECT count(*) FROM s8_index
             WHERE TRY_CAST(received_date AS DATE) BETWEEN ? AND ?)
             AS test_all_text,
            (SELECT count(*) - count(DISTINCT (
                normalized_group_hash, normalized_length
            )) FROM s8_primary) AS scientific_duplicate_representatives,
            (SELECT count(*) FROM s8_test_base b
             WHERE NOT EXISTS (
                 SELECT 1 FROM s8_clean_groups g
                 WHERE g.normalized_group_hash = b.normalized_group_hash
                   AND g.normalized_length = b.normalized_length
             )) AS ambiguous_test_lines,
            (SELECT count(*) FROM s8_index i
             WHERE TRY_CAST(i.received_date AS DATE) < ?) AS prior_modeled_lines,
            (SELECT count(DISTINCT (i.normalized_group_hash, i.normalized_length))
             FROM s8_index i
             WHERE TRY_CAST(i.received_date AS DATE) < ?) AS prior_modeled_groups
        """,
        [
            CRITICAL_CLASS,
            source["test_start"],
            source["test_end"],
            source["test_start"],
            source["test_end"],
            source["test_start"],
            source["test_start"],
        ],
    ).fetchone()
    names = (
        "novel_test_lines",
        "primary_representatives",
        "operational_lines",
        "novel_unique_groups",
        "clean_unique_groups",
        "critical_novel_groups",
        "all_test_unique_groups",
        "test_all_text",
        "scientific_duplicate_representatives",
        "ambiguous_test_lines",
        "prior_modeled_lines",
        "prior_modeled_groups",
    )
    counts = dict(zip(names, (int(value) for value in result)))
    counts["ambiguous_unique_groups"] = (
        counts["novel_unique_groups"] - counts["clean_unique_groups"]
    )
    counts["seen_previously_test_lines"] = (
        counts["test_all_text"] - counts["novel_test_lines"]
    )
    counts["seen_previously_test_groups"] = (
        counts["all_test_unique_groups"] - counts["novel_unique_groups"]
    )
    return counts


def _iter_raw_batches(
    connection: duckdb.DuckDBPyConnection,
    source_path: Path,
    scope_table: str,
    config: S8Config,
) -> Iterable[tuple[list[str], list[str]]]:
    """Stream raw narratives and labels for one already-defined scope."""

    text_column = _quote_identifier(config.source["text_column"])
    query = (
        "SELECT s.product_family, CAST(r."
        + text_column
        + " AS VARCHAR) AS narrative FROM read_parquet(?) r "
        "INNER JOIN "
        + scope_table
        + ' s ON r."Complaint ID" = s."Complaint ID" '
        'ORDER BY s."Complaint ID"'
    )
    reader = connection.execute(query, [str(source_path)]).to_arrow_reader(
        DEFAULT_BATCH_SIZE
    )
    for batch in reader:
        labels = [str(value) for value in batch.column("product_family").to_pylist()]
        raw_texts = batch.column("narrative").to_pylist()
        if any(value is None or not str(value).strip() for value in raw_texts):
            raise ValueError("S8 raw narrative join returned an empty narrative")
        texts = [str(value) for value in raw_texts]
        yield labels, texts


def _score_scope(
    connection: duckdb.DuckDBPyConnection,
    source_path: Path,
    scope_table: str,
    predictor: Any,
    config: S8Config,
) -> tuple[np.ndarray, int]:
    """Score one scope into an aggregate confusion matrix only."""

    matrix = np.zeros((len(MODELED_FAMILIES), len(MODELED_FAMILIES)), dtype=np.int64)
    positions = {label: index for index, label in enumerate(MODELED_FAMILIES)}
    rows = 0
    for labels, texts in _iter_raw_batches(
        connection, source_path, scope_table, config
    ):
        if len(labels) != len(texts):
            raise ValueError(
                f"S8 {scope_table} batch labels/texts have different lengths"
            )
        unknown_labels = set(labels).difference(positions)
        if unknown_labels:
            raise ValueError(
                f"S8 {scope_table} contains unknown truth labels: "
                f"{sorted(unknown_labels)}"
            )
        predicted = predictor.predict(texts, input_language=INPUT_LANGUAGE)
        predictions = getattr(predicted, "predictions", None)
        if predictions is None or len(predictions) != len(labels):
            raise ValueError(
                f"S8 {scope_table} predictor returned an invalid batch length"
            )
        for truth, item in zip(labels, predictions):
            label = getattr(item, "label", None)
            if label not in positions:
                raise ValueError(
                    f"S8 {scope_table} predictor returned unknown label: {label!r}"
                )
            matrix[positions[truth], positions[label]] += 1
            rows += 1
    return matrix, rows


def _validate_scope_row_count(scope_table: str, observed: int, expected: int) -> None:
    """Reject a raw join that silently loses or duplicates scoped rows."""

    if observed != expected:
        raise ValueError(
            f"S8 {scope_table} raw join rows {observed} differ from expected "
            f"{expected}"
        )


def _check_result_privacy(
    payload: Mapping[str, Any],
    forbidden_texts: Iterable[str] = (),
) -> None:
    """Reject raw text, identifiers, or individual scores recursively."""

    forbidden_values = tuple(value for value in forbidden_texts if value)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower().replace("-", "_")
                score_key = lowered in {"score", "scores"} or (
                    "individual_score" in lowered
                )
                text_key = "text" in lowered and lowered not in {
                    "test_all_text",
                    "novel_text",
                }
                exact_key = lowered in FORBIDDEN_RESULT_KEY_TERMS
                if score_key or text_key or exact_key:
                    raise ValueError(f"S8 result contains forbidden key: {path}.{key}")
                visit(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, str) and any(
            text in value for text in forbidden_values
        ):
            raise ValueError(f"S8 result contains input text at {path}")

    visit(payload, "result")


def _signature(
    config_path: Path,
    source: Mapping[str, Any],
    s7_freeze: Mapping[str, Any],
) -> str:
    """Create a stable run signature from protocol and frozen metadata only."""

    payload = {
        "code_schema": S8_CODE_SCHEMA,
        "config": _file_signature(config_path),
        "source": dict(source),
        "s7_freeze": dict(s7_freeze),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()


def _require_relative_path(value: Any, label: str) -> str:
    """Validate and normalize one project-relative manifest path."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"S8 manifest {label} path is invalid")
    path = Path(value)
    windows_path = path.drive or value.startswith("\\")
    if path.is_absolute() or windows_path or ".." in path.parts:
        raise ValueError(f"S8 manifest {label} path must be relative")
    return value.replace("\\", "/")


def _validate_complete_result(
    payload: Mapping[str, Any],
    signature: str | None = None,
) -> None:
    """Validate a complete aggregate result without accessing dataset files."""

    if payload.get("schema_version") != S8_RESULT_SCHEMA:
        raise ValueError("Unexpected S8 result schema")
    if payload.get("code_schema") != S8_CODE_SCHEMA:
        raise ValueError("S8 result code schema is stale")
    if payload.get("complete") is not True:
        raise ValueError("S8 result is not complete")
    if signature is not None and payload.get("signature") != signature:
        raise ValueError("S8 result signature differs from frozen protocol")
    if payload.get("status") not in {
        "CONFIRMED_FOR_STRESS_EVALUATION",
        "NOT_CONFIRMED",
    }:
        raise ValueError("S8 complete result status is invalid")
    if payload.get("confirmatory") is not True:
        raise ValueError("S8 result confirmatory flag is invalid")
    if payload.get("deploy") is not False:
        raise ValueError("S8 result cannot authorize deployment")
    if payload.get("test_opened") is not True:
        raise ValueError("S8 result must record test_opened=true")
    if tuple(payload.get("remaining_sealed", ())) != SEALED_PARTITIONS:
        raise ValueError("S8 result sealed boundary is invalid")
    model = payload.get("model", {})
    if model.get("input_language") != INPUT_LANGUAGE:
        raise ValueError("S8 result language is invalid")
    if tuple(model.get("classes", ())) != tuple(MODELED_FAMILIES):
        raise ValueError("S8 result class order is invalid")
    if model.get("critical_class") != CRITICAL_CLASS:
        raise ValueError("S8 result critical class is invalid")
    if model.get("score_kind") != "critical_margin":
        raise ValueError("S8 result score kind is invalid")
    primary = payload.get("primary")
    operational = payload.get("operational_secondary")
    if not isinstance(primary, Mapping) or not isinstance(operational, Mapping):
        raise ValueError("S8 result aggregate views are missing")
    if operational.get("excluded_from_decision") is not True:
        raise ValueError("S8 operational view must be diagnostic only")
    gates = primary.get("gates", {})
    metrics = primary.get("metrics", {})
    if gates.get("gate_count") != sum(gates.get("checks", {}).values()):
        raise ValueError("S8 gate count is inconsistent")
    if payload.get("confirmed") is not bool(gates.get("passed")):
        raise ValueError("S8 confirmed flag is inconsistent with gates")
    expected_status = (
        "CONFIRMED_FOR_STRESS_EVALUATION"
        if payload["confirmed"]
        else "NOT_CONFIRMED"
    )
    if payload["status"] != expected_status:
        raise ValueError("S8 status is inconsistent with gates")
    if payload.get("decision", {}).get("status") != payload["status"]:
        raise ValueError("S8 decision status is inconsistent")
    if payload.get("decision", {}).get("gate_count") != gates.get("gate_count"):
        raise ValueError("S8 decision gates are inconsistent")
    if payload.get("decision", {}).get("deploy") is not False:
        raise ValueError("S8 decision cannot authorize deployment")
    if set(metrics.get("per_class", {})) != set(MODELED_FAMILIES):
        raise ValueError("S8 result per-class support is incomplete")
    if primary.get("support_all_nine_classes") is not True:
        raise ValueError("S8 result does not support all nine classes")
    _check_result_privacy(payload)


def validate_s8_manifest(
    manifest: Mapping[str, Any],
    result_path: str | Path,
    config_path: str | Path,
) -> None:
    """Validate the complete portable S8 manifest and aggregate result."""

    result_file = Path(result_path).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    config = load_s8_config(config_file)
    root = config_file.parent.parent.resolve()
    if manifest.get("schema_version") != S8_MANIFEST_SCHEMA:
        raise ValueError("Unexpected S8 manifest schema")
    if manifest.get("stage") != "S8":
        raise ValueError("S8 manifest stage is invalid")
    if manifest.get("status") not in {
        "CONFIRMED_FOR_STRESS_EVALUATION",
        "NOT_CONFIRMED",
    }:
        raise ValueError("S8 manifest status is invalid")
    if manifest.get("confirmatory") is not True:
        raise ValueError("S8 manifest confirmatory flag is invalid")
    if manifest.get("deploy") is not False:
        raise ValueError("S8 manifest cannot authorize deployment")
    if manifest.get("test_opened") is not True:
        raise ValueError("S8 manifest must record test_opened=true")
    if tuple(manifest.get("remaining_sealed", ())) != SEALED_PARTITIONS:
        raise ValueError("S8 manifest sealed boundary is invalid")
    protocol = manifest.get("protocol", {})
    result_meta = manifest.get("result", {})
    if _require_relative_path(protocol.get("path"), "protocol") != _relative(
        config_file, root
    ):
        raise ValueError("S8 manifest protocol path is invalid")
    if _require_relative_path(result_meta.get("path"), "result") != _relative(
        result_file, root
    ):
        raise ValueError("S8 manifest result path is invalid")
    protocol_actual = _file_signature(config_file)
    if protocol.get("sha256") != protocol_actual["sha256"]:
        raise ValueError("S8 manifest protocol hash is invalid")
    if protocol.get("size_bytes") != protocol_actual["size_bytes"]:
        raise ValueError("S8 manifest protocol size is invalid")
    result_actual = _file_signature(result_file)
    if result_meta.get("sha256") != result_actual["sha256"]:
        raise ValueError("S8 manifest result hash is invalid")
    if result_meta.get("size_bytes") != result_actual["size_bytes"]:
        raise ValueError("S8 manifest result size is invalid")
    if manifest.get("s7_freeze") != config.s7_freeze:
        raise ValueError("S8 manifest S7 freeze differs from protocol")
    expected_external = {
        "raw_source": (
            config.payload["paths"]["source_path"],
            config.source["raw_sha256"],
            config.source["raw_size_bytes"],
        ),
        "index": (
            config.payload["paths"]["index_path"],
            config.source["index_sha256"],
            config.source["index_size_bytes"],
        ),
        "s3_protocol": (
            config.payload["paths"]["s3_protocol"],
            config.source["s3_protocol_sha256"],
            config.source["s3_protocol_size_bytes"],
        ),
    }
    for key, (path, digest, size) in expected_external.items():
        item = manifest.get(key, {})
        if _require_relative_path(item.get("path"), key) != path:
            raise ValueError(f"S8 manifest {key} path is invalid")
        if item.get("sha256") != digest or item.get("size_bytes") != size:
            raise ValueError(f"S8 manifest {key} metadata is invalid")
    payload = _read_json(result_file)
    # The historical result signature included the checkout's absolute path.
    # The manifest hashes above are the portable integrity boundary.
    _validate_complete_result(payload)
    if payload["model"]["threshold"] != config.payload["model"]["threshold"]:
        raise ValueError("S8 manifest model threshold differs from protocol")
    if payload["primary"]["gates"]["limits"] != dict(config.gates):
        raise ValueError("S8 manifest gate limits differ from protocol")
    if manifest.get("status") != payload.get("status"):
        raise ValueError("S8 manifest status differs from result")
    if manifest.get("confirmed") != payload.get("confirmed"):
        raise ValueError("S8 manifest decision differs from result")
    if manifest.get("decision") != payload.get("decision"):
        raise ValueError("S8 manifest gates differ from result")
    if manifest.get("opened_at") != payload.get("opened_at"):
        raise ValueError("S8 manifest opened_at differs from result")
    if manifest.get("execution_attempts") != payload.get("execution_attempts"):
        raise ValueError("S8 manifest execution attempts differ from result")


def _cached_result(
    result_path: Path,
    manifest_path: Path,
    signature: str,
    *,
    config_path: Path,
) -> dict[str, Any] | None:
    """Return or repair a complete cache hit without dataset access."""

    if not result_path.exists():
        return None
    payload = _read_json(result_path)
    if payload.get("complete") is not True or payload.get("signature") != signature:
        return None
    config = load_s8_config(config_path)
    _validate_complete_result(payload, signature)
    try:
        manifest = _read_json(manifest_path)
        validate_s8_manifest(manifest, result_path, config_path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        _publish_manifest(result_path, manifest_path, config_path, config, payload)
    return payload


def _resume_attempts(
    result_path: Path, signature: str
) -> tuple[int, str | None]:
    """Validate an incomplete marker and return its next attempt number."""

    if not result_path.exists():
        return 1, None
    payload = _read_json(result_path)
    if payload.get("complete") is True:
        if payload.get("signature") != signature:
            raise ValueError("Complete S8 result has a stale protocol signature")
        return int(payload.get("execution_attempts", 1)), None
    if payload.get("signature") != signature:
        raise ValueError("Incomplete S8 result cannot resume under new hashes")
    if payload.get("code_schema") != S8_CODE_SCHEMA:
        raise ValueError("Incomplete S8 result cannot resume under new code")
    if payload.get("primary") is not None:
        raise ValueError("S8 partial primary metrics must not be persisted")
    if payload.get("operational_secondary") is not None:
        raise ValueError("S8 partial operational metrics must not be persisted")
    opened_at = payload.get("opened_at")
    if not isinstance(opened_at, str) or not opened_at.endswith("Z"):
        raise ValueError("Incomplete S8 result has no valid opened_at")
    return int(payload.get("execution_attempts", 1)) + 1, opened_at


def _base_result(
    signature: str,
    config: S8Config,
    evidence: Mapping[str, Any],
    execution_attempts: int = 1,
    opened_at: str | None = None,
) -> dict[str, Any]:
    """Create the aggregate-only S8 result envelope."""

    return {
        "schema_version": S8_RESULT_SCHEMA,
        "code_schema": S8_CODE_SCHEMA,
        "signature": signature,
        "complete": False,
        "status": "RUNNING",
        "confirmatory": True,
        "confirmed": None,
        "deploy": False,
        "test_opened": True,
        "opened_at": opened_at or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "execution_attempts": execution_attempts,
        "model": {
            "input_language": INPUT_LANGUAGE,
            "classes": list(MODELED_FAMILIES),
            "critical_class": CRITICAL_CLASS,
            "threshold": 0.1135351095114484,
            "score_kind": "critical_margin",
        },
        "test_scope": dict(config.test_scope),
        "remaining_sealed": list(SEALED_PARTITIONS),
        "s2_evidence": evidence["s2"],
        "scope_counts": None,
        "primary": None,
        "operational_secondary": None,
        "decision": None,
    }


def _error_result(result: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    """Create an incomplete error marker without persisting partial metrics."""

    failed = dict(result)
    failed["status"] = "ERROR"
    failed["complete"] = False
    failed["primary"] = None
    failed["operational_secondary"] = None
    failed["scope_counts"] = None
    failed["decision"] = None
    failed["confirmed"] = None
    failed["error"] = {
        "type": type(error).__name__,
        "stage": "confirmatory_execution",
    }
    return failed


def _publish_manifest(
    result_path: Path,
    manifest_path: Path,
    config_path: Path,
    config: S8Config,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the post-full S8 manifest with all frozen digests."""

    root = config_path.parent.parent
    protocol_signature = _file_signature(config_path)
    protocol_signature["path"] = _relative(config_path, root)
    manifest = {
        "schema_version": S8_MANIFEST_SCHEMA,
        "stage": "S8",
        "status": result["status"],
        "confirmatory": True,
        "confirmed": result["confirmed"],
        "deploy": False,
        "test_opened": True,
        "opened_at": result["opened_at"],
        "execution_attempts": result["execution_attempts"],
        "remaining_sealed": list(SEALED_PARTITIONS),
        "protocol": protocol_signature,
        "result": {
            **_file_signature(result_path),
            "path": _relative(result_path, root),
        },
        "s7_freeze": dict(config.s7_freeze),
        "raw_source": {
            "path": config.payload["paths"]["source_path"],
            "sha256": config.source["raw_sha256"],
            "size_bytes": config.source["raw_size_bytes"],
        },
        "index": {
            "path": config.payload["paths"]["index_path"],
            "sha256": config.source["index_sha256"],
            "size_bytes": config.source["index_size_bytes"],
        },
        "s3_protocol": {
            "path": config.payload["paths"]["s3_protocol"],
            "sha256": config.source["s3_protocol_sha256"],
            "size_bytes": config.source["s3_protocol_size_bytes"],
        },
        "decision": result["decision"],
        "scientific_view": "primary_only",
        "operational_view": "secondary_diagnostic_only",
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _run_full(
    project_root: Path,
    config_path: Path,
    result_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run the one-time test evaluation after all guards pass."""

    config = load_s8_config(config_path)
    signature = _signature(config_path, config.source, config.s7_freeze)
    cached = _cached_result(
        result_path, manifest_path, signature, config_path=config_path
    )
    if cached is not None:
        return cached
    execution_attempts, opened_at = _resume_attempts(result_path, signature)
    metadata = validate_frozen_metadata(config, project_root, include_raw=False)
    require_confirmatory_unlock(config)
    result = _base_result(
        signature,
        config,
        metadata,
        execution_attempts=execution_attempts,
        opened_at=opened_at,
    )
    try:
        _write_json_atomic(result_path, result)
        metadata = validate_frozen_metadata(config, project_root, include_raw=True)
        source_path = project_root / config.payload["paths"]["source_path"]
        index_path = project_root / config.payload["paths"]["index_path"]
        bundle_path = project_root / config.payload["paths"]["s7_bundle"]
        s7_manifest = project_root / config.payload["paths"]["s7_manifest"]
        s7_result = project_root / config.payload["paths"]["s7_result"]
        predictor = load_s7_predictor(bundle_path, s7_manifest, s7_result)
        spill_root = (
            project_root / config.payload["access"]["temp_directory"]
        ).resolve()
        spill_root.mkdir(parents=True, exist_ok=True)
        spill_directory = spill_root / (
            f"run-{os.getpid()}-{execution_attempts}"
        )
        spill_directory.mkdir(parents=True, exist_ok=False)
        connection = duckdb.connect()
        try:
            spill_literal = str(spill_directory).replace("'", "''")
            connection.execute(f"SET temp_directory = '{spill_literal}'")
            connection.execute(f"SET memory_limit = '{DEFAULT_MEMORY_LIMIT}'")
            connection.execute("SET threads = 1")
            connection.execute("SET preserve_insertion_order = false")
            counts = _scope_counts(connection, index_path, config)
            primary_matrix, primary_rows = _score_scope(
                connection, source_path, "s8_primary", predictor, config
            )
            operational_matrix, operational_rows = _score_scope(
                connection, source_path, "s8_operational", predictor, config
            )
        finally:
            connection.close()
            if spill_directory.parent.resolve() == spill_root:
                shutil.rmtree(spill_directory, ignore_errors=True)
        _validate_scope_row_count(
            "s8_primary", primary_rows, counts["primary_representatives"]
        )
        _validate_scope_row_count(
            "s8_operational", operational_rows, counts["operational_lines"]
        )
        support = primary_matrix.sum(axis=1)
        if np.any(support <= 0):
            raise ValueError("S8 primary test view does not support all nine classes")
        primary_metrics = metrics_from_confusion(primary_matrix)
        operational_metrics = metrics_from_confusion(operational_matrix)
        primary_gates = evaluate_gates(primary_metrics, config.gates)
        result["scope_counts"] = {
            **counts,
            "primary_scored_rows": primary_rows,
            "operational_scored_rows": operational_rows,
            "s2_difference": {
                "expected": {
                    "test_all_text": config.payload["s2_evidence"][
                        "test_all_text"
                    ],
                    "novel_text": config.payload["s2_evidence"]["novel_text"],
                    "novel_unique_groups": config.payload["s2_evidence"][
                        "novel_unique_groups"
                    ],
                    "critical_novel_groups": config.payload["s2_evidence"][
                        "critical_novel_groups"
                    ],
                },
                "observed": {
                    "test_all_text": counts["test_all_text"],
                    "novel_text": counts["novel_test_lines"],
                    "novel_unique_groups": counts["novel_unique_groups"],
                    "critical_novel_groups": counts["critical_novel_groups"],
                },
            },
        }
        observed = result["scope_counts"]["s2_difference"]["observed"]
        expected = result["scope_counts"]["s2_difference"]["expected"]
        result["scope_counts"]["s2_difference"]["delta"] = {
            key: int(observed[key]) - int(expected[key]) for key in expected
        }
        result["primary"] = {
            "view": "scientific_primary",
            "metrics": primary_metrics,
            "support_all_nine_classes": bool(np.all(support > 0)),
            "support_by_class": {
                label: int(value)
                for label, value in zip(MODELED_FAMILIES, support)
            },
            "gates": primary_gates,
            "confidence_intervals": bootstrap_confidence_intervals(
                primary_matrix,
                replicates=DEFAULT_BOOTSTRAP_REPLICATES,
                seed=DEFAULT_BOOTSTRAP_SEED,
                confidence_level=DEFAULT_CONFIDENCE_LEVEL,
            ),
        }
        result["operational_secondary"] = {
            "view": "operational_secondary",
            "metrics": operational_metrics,
            "excluded_from_decision": True,
        }
        result["confirmed"] = bool(primary_gates["passed"])
        result["status"] = (
            "CONFIRMED_FOR_STRESS_EVALUATION"
            if result["confirmed"]
            else "NOT_CONFIRMED"
        )
        result["decision"] = {
            "scientific_view": "primary",
            "gate_count": primary_gates["gate_count"],
            "required_gate_count": 3,
            "status": result["status"],
            "deploy": False,
            "test_does_not_authorize_deployment": True,
        }
        result["complete"] = True
        _check_result_privacy(result)
    except Exception as error:
        spill_candidate = locals().get("spill_directory")
        if isinstance(spill_candidate, Path) and spill_candidate.exists():
            if spill_candidate.parent.resolve() == (
                project_root / config.payload["access"]["temp_directory"]
            ).resolve():
                shutil.rmtree(spill_candidate, ignore_errors=True)
        failed = _error_result(result, error)
        _check_result_privacy(failed)
        _write_json_atomic(result_path, failed)
        raise
    _write_json_atomic(result_path, result)
    _publish_manifest(result_path, manifest_path, config_path, config, result)
    return result


def run_s8(
    *,
    project_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    result_path: str | Path = DEFAULT_RESULT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    run_mode: str = "disabled",
) -> dict[str, Any]:
    """Run S8 in disabled, synthetic smoke, or guarded full mode.

    ``full`` is the only mode that can access the real index or raw source.
    """

    if run_mode == "disabled":
        return {"status": "DISABLED", "confirmatory": True, "deploy": False}
    if run_mode == "smoke":
        return run_s8_smoke()
    if run_mode != "full":
        raise ValueError("S8 run_mode must be disabled, smoke, or full")
    return _run_full(
        Path(project_root).expanduser().resolve(),
        Path(config_path).expanduser().resolve(),
        Path(result_path).expanduser().resolve(),
        Path(manifest_path).expanduser().resolve(),
    )


def run_s8_smoke() -> dict[str, Any]:
    """Run deterministic metric and guard checks with synthetic aggregates."""

    matrix = np.eye(len(MODELED_FAMILIES), dtype=np.int64) * 5
    matrix[MODELED_FAMILIES.index(CRITICAL_CLASS), 0] = 1
    metrics = metrics_from_confusion(matrix)
    gates = evaluate_gates(
        metrics,
        {
            "macro_f1_min": 0.69,
            "critical_f1_min": 0.2715,
            "critical_precision_min": 0.2,
        },
    )
    return {
        "status": "DIAGNOSTIC_ONLY",
        "complete": True,
        "confirmatory": True,
        "confirmed": bool(gates["passed"]),
        "deploy": False,
        "test_opened": False,
        "primary": {"metrics": metrics, "gates": gates},
        "confidence_intervals": bootstrap_confidence_intervals(
            matrix, replicates=20
        ),
    }
