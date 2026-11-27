"""V3-E3 rolling-origin linear probes on frozen development evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import load_v3_protocol, validate_protocol_manifest
from .reporting import audit_v3_e2_cache


DEFAULT_CONFIG = "config/v3/rolling_probes.json"
RESULT_SCHEMA = "v3-e3-rolling-results-v1"
MANIFEST_SCHEMA = "v3-e3-rolling-results-manifest-v1"
CRITICAL_CLASS = "debt_credit_management"
ALLOWED_PARTITIONS = ("train", "validation")
RATIOS = (5, 10, 15)
REPRESENTATIONS = ("tfidf_word_char", "minilm_frozen")
RANDOM_STATE = 42
PRECISION_FLOOR = 0.4
_FORBIDDEN_RESULT_KEYS = {
    "ids",
    "indices",
    "texts",
    "narratives",
    "scores",
    "probabilities",
    "embeddings",
    "estimator",
}


@dataclass(slots=True)
class DevelopmentData:
    """Hold aligned development rows and frozen embeddings in memory."""

    dates: np.ndarray
    labels: np.ndarray
    group_hashes: np.ndarray
    group_lengths: np.ndarray
    partitions: np.ndarray
    texts: np.ndarray
    embeddings: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3-E3 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("V3-E3 JSON must be an object")
    return payload


def _sha256(path: Path) -> str:
    """Return an uppercase SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _signature(path: Path) -> dict[str, Any]:
    """Return a portable file signature."""

    if not path.is_file():
        raise FileNotFoundError(path)
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _safe_path(root: Path, relative: str) -> Path:
    """Resolve a project-relative path without allowing directory escape."""

    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError("V3-E3 paths must be project-relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("V3-E3 path escapes the project root")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an aggregate result atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_rolling_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the pre-result operational choices for V3-E3."""

    block = json.loads(json.dumps(payload))
    if block.get("schema_version") != "v3-e3-rolling-probes-v1":
        raise ValueError("V3-E3 config schema is invalid")
    if block.get("stage") != "V3-E3":
        raise ValueError("V3-E3 config stage is invalid")
    if block.get("status") != "FROZEN_FOR_EXECUTION":
        raise ValueError("V3-E3 config is not frozen")
    source = block.get("source", {})
    if tuple(source.get("allowed_partitions", ())) != ALLOWED_PARTITIONS:
        raise ValueError("V3-E3 allowed partitions differ")
    if int(source.get("expected_rows", -1)) != 591_532:
        raise ValueError("V3-E3 source support differs")
    if source.get("critical_class") != CRITICAL_CLASS:
        raise ValueError("V3-E3 critical class differs")
    representations = block.get("representations", [])
    if tuple(item.get("candidate_prefix") for item in representations) != (
        REPRESENTATIONS
    ):
        raise ValueError("V3-E3 representation catalog differs")
    mining = block.get("hard_negative_mining", {})
    expected_mining = {
        "scope": "fit_window_only",
        "representation": "v2_word_char_tfidf_union_40000_60000",
        "classifier": "LinearSVC",
        "C": 1.0,
        "class_weight": "balanced",
        "tol": 0.0001,
        "max_iter": 5000,
        "dual": "auto",
        "splitter": "StratifiedKFold",
        "n_splits": 3,
        "shuffle": True,
        "random_state": 42,
        "ranking": (
            "descending_out_of_fold_critical_margin_then_group_identity"
        ),
    }
    if mining != expected_mining:
        raise ValueError("V3-E3 hard-negative mining contract differs")
    sampling = block.get("sampling", {})
    if tuple(sampling.get("negative_ratios", ())) != RATIOS:
        raise ValueError("V3-E3 negative ratios differ")
    expected_composition = {
        "5": {"hard": 3, "background": 2},
        "10": {"hard": 7, "background": 3},
        "15": {"hard": 10, "background": 5},
    }
    if sampling.get("negative_composition") != expected_composition:
        raise ValueError("V3-E3 negative composition differs")
    if tuple(sampling.get("background_strata", ())) != (
        "calendar_month",
        "product_family",
    ):
        raise ValueError("V3-E3 background strata differ")
    if sampling.get("same_group_ids_across_representations") is not True:
        raise ValueError("V3-E3 samples must be shared across representations")
    classifier = block.get("classifier", {})
    if classifier != {
        "class": "LinearSVC",
        "C": 1.0,
        "class_weight": None,
        "tol": 0.0001,
        "max_iter": 5000,
        "dual": "auto",
        "random_state": 42,
    }:
        raise ValueError("V3-E3 candidate classifier differs")
    calibration = block.get("calibration", {})
    if float(calibration.get("critical_precision_floor", -1.0)) != 0.4:
        raise ValueError("V3-E3 precision floor differs")
    if calibration.get("evaluation_oracle_threshold_forbidden") is not True:
        raise ValueError("V3-E3 evaluation oracle must remain forbidden")
    if block.get("execution", {}).get("default_run_mode") != "disabled":
        raise ValueError("V3-E3 must be disabled by default")
    return block


def load_rolling_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the V3-E3 execution contract."""

    return validate_rolling_config(_read_json(Path(path)))


def _validate_privacy(payload: Mapping[str, Any]) -> None:
    """Reject row-level or fitted evidence in an aggregate V3-E3 result."""

    def visit(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in _FORBIDDEN_RESULT_KEYS:
                    raise ValueError(f"Private V3-E3 field at {location}.{key}")
                visit(child, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    visit(payload, "result")


def _load_development_data(root: Path, config: Mapping[str, Any]) -> DevelopmentData:
    """Load the ordered source and align it exactly to the V3-E2 cache."""

    import duckdb
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    source = _safe_path(root, str(config["source"]["path"]))
    embedding_manifest = _read_json(
        _safe_path(root, str(config["embedding_manifest"]))
    )
    cache_root = _safe_path(
        root,
        str(config["representations"][1]["cache_root"]),
    )
    query = """
        SELECT received_date, product_family, normalized_group_hash,
               normalized_length, partition_name, narrative
        FROM read_parquet(?)
        WHERE partition_name IN ('train', 'validation')
        ORDER BY partition_name, received_date,
                 normalized_group_hash, normalized_length
    """
    with duckdb.connect() as connection:
        source_table = connection.execute(query, [str(source)]).fetch_arrow_table()
    if source_table.num_rows != int(config["source"]["expected_rows"]):
        raise ValueError("V3-E3 source support differs from the config")

    key_tables = []
    embedding_blocks = []
    for shard in embedding_manifest["shards"]:
        key_tables.append(pq.read_table(cache_root / shard["keys_path"]))
        embedding_blocks.append(
            np.load(cache_root / shard["embedding_path"], allow_pickle=False)
        )
    key_table = pa.concat_tables(key_tables)
    for name in (
        "received_date",
        "product_family",
        "normalized_group_hash",
        "normalized_length",
        "partition_name",
    ):
        if not bool(pc.all(pc.equal(source_table[name], key_table[name])).as_py()):
            raise ValueError(f"V3-E3 source and embedding key order differ: {name}")
    embeddings = np.concatenate(embedding_blocks, axis=0)
    if embeddings.shape != (source_table.num_rows, 384):
        raise ValueError("V3-E3 embedding support differs from the source")
    if embeddings.dtype != np.float16 or not np.isfinite(embeddings).all():
        raise ValueError("V3-E3 embeddings are invalid")
    values = source_table.to_pydict()
    data = DevelopmentData(
        dates=np.asarray(values["received_date"], dtype="datetime64[D]"),
        labels=np.asarray(values["product_family"], dtype=object),
        group_hashes=np.asarray(values["normalized_group_hash"], dtype=object),
        group_lengths=np.asarray(values["normalized_length"], dtype=np.int64),
        partitions=np.asarray(values["partition_name"], dtype=object),
        texts=np.asarray(values["narrative"], dtype=object),
        embeddings=embeddings,
    )
    groups = {
        (str(group_hash), int(group_length))
        for group_hash, group_length in zip(
            data.group_hashes,
            data.group_lengths,
        )
    }
    if len(groups) != len(data.labels):
        raise ValueError("V3-E3 development group identities are not unique")
    return data


def _fold_indices(
    data: DevelopmentData,
    fold: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return exact indices for one frozen rolling-origin fold."""

    result = {}
    for role in ("fit", "calibration", "evaluation"):
        window = fold[role]
        start = np.datetime64(str(window["start"]), "D")
        end = np.datetime64(str(window["end"]), "D")
        mask = (
            (data.partitions == str(window["partition"]))
            & (data.dates >= start)
            & (data.dates <= end)
        )
        indices = np.flatnonzero(mask)
        critical = int(np.count_nonzero(data.labels[indices] == CRITICAL_CLASS))
        if len(indices) != int(window["rows"]):
            raise ValueError(f"V3-E3 {fold['fold_id']} {role} rows differ")
        if critical != int(window["critical_rows"]):
            raise ValueError(f"V3-E3 {fold['fold_id']} {role} critical rows differ")
        result[role] = indices
    if result["fit"][-1] >= result["calibration"][0]:
        raise ValueError("V3-E3 fit and calibration order overlap")
    if result["calibration"][-1] >= result["evaluation"][0]:
        raise ValueError("V3-E3 calibration and evaluation order overlap")
    return result


def _binary_target(labels: np.ndarray) -> np.ndarray:
    """Return the binary critical target."""

    return np.asarray(labels == CRITICAL_CLASS, dtype=np.int8)


def _candidate_estimator(*, balanced: bool = False) -> Any:
    """Build the frozen candidate or OOF-mining LinearSVC."""

    from sklearn.svm import LinearSVC

    return LinearSVC(
        C=1.0,
        class_weight="balanced" if balanced else None,
        tol=0.0001,
        max_iter=5000,
        dual="auto",
        random_state=RANDOM_STATE,
    )


def _mine_oof_margins(
    data: DevelopmentData,
    fit_indices: np.ndarray,
    fold_id: str,
) -> np.ndarray:
    """Rank fit-window negatives using strictly within-fit OOF margins."""

    from sklearn.model_selection import StratifiedKFold

    from consumer_complaint_intelligence.v2_detector import (
        WORD_CHAR_TFIDF_ALIAS,
        build_vectorizer,
    )

    fit_labels = _binary_target(data.labels[fit_indices])
    splitter = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    margins = np.full(len(fit_indices), np.nan, dtype=np.float64)
    for split_number, (train_local, held_local) in enumerate(
        splitter.split(np.zeros(len(fit_indices)), fit_labels),
        start=1,
    ):
        started = time.perf_counter()
        vectorizer = build_vectorizer(WORD_CHAR_TFIDF_ALIAS)
        train_matrix = vectorizer.fit_transform(
            data.texts[fit_indices[train_local]].tolist()
        )
        estimator = _candidate_estimator(balanced=True)
        estimator.fit(train_matrix, fit_labels[train_local])
        del train_matrix
        gc.collect()
        held_matrix = vectorizer.transform(
            data.texts[fit_indices[held_local]].tolist()
        )
        margins[held_local] = estimator.decision_function(held_matrix)
        print(
            f"V3-E3 {fold_id} hard-negative OOF {split_number}/3 "
            f"completed in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del vectorizer, estimator, held_matrix
        gc.collect()
    if not np.isfinite(margins).all():
        raise ValueError(f"V3-E3 {fold_id} OOF margins are incomplete")
    return margins


def _stable_group_key(
    data: DevelopmentData,
    index: int,
    salt: str,
) -> tuple[str, str, int]:
    """Return a deterministic pseudo-random key without row-position dependence."""

    group_hash = str(data.group_hashes[index])
    group_length = int(data.group_lengths[index])
    digest = hashlib.sha256(
        f"{RANDOM_STATE}|{salt}|{group_hash}|{group_length}".encode("utf-8")
    ).hexdigest()
    return digest, group_hash, group_length


def _allocate_largest_remainder(
    sizes: Mapping[tuple[str, str], int],
    target: int,
) -> dict[tuple[str, str], int]:
    """Allocate a proportional sample exactly through largest remainders."""

    total = sum(sizes.values())
    if target < 0 or target > total:
        raise ValueError("V3-E3 background target exceeds available rows")
    raw = {key: target * value / total for key, value in sizes.items()}
    allocation = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = target - sum(allocation.values())
    order = sorted(
        sizes,
        key=lambda key: (
            -(raw[key] - allocation[key]),
            key[0],
            key[1],
        ),
    )
    for key in order:
        if remaining <= 0:
            break
        if allocation[key] < sizes[key]:
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("V3-E3 background allocation is incomplete")
    return allocation


def _sample_signature(
    data: DevelopmentData,
    positives: Sequence[int],
    hard: Sequence[int],
    background: Sequence[int],
) -> str:
    """Fingerprint a sample without publishing its group identities."""

    digest = hashlib.sha256()
    for role, indices in (
        ("positive", positives),
        ("hard", hard),
        ("background", background),
    ):
        ordered = sorted(
            indices,
            key=lambda index: (
                str(data.group_hashes[index]),
                int(data.group_lengths[index]),
            ),
        )
        for index in ordered:
            digest.update(role.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(str(data.group_hashes[index]).encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(str(int(data.group_lengths[index])).encode("utf-8"))
            digest.update(b"\x1e")
    return digest.hexdigest().upper()


def sample_fit_pool(
    data: DevelopmentData,
    fit_indices: np.ndarray,
    oof_margins: np.ndarray,
    *,
    fold_id: str,
    ratio: int,
    hard_per_positive: int,
    background_per_positive: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select all positives plus exact hard and stratified background negatives."""

    if ratio != hard_per_positive + background_per_positive:
        raise ValueError("V3-E3 negative composition does not equal the ratio")
    fit_target = _binary_target(data.labels[fit_indices])
    positive_local = np.flatnonzero(fit_target == 1)
    negative_local = np.flatnonzero(fit_target == 0)
    positives = [int(fit_indices[index]) for index in positive_local]
    negative_global = [int(fit_indices[index]) for index in negative_local]
    margin_by_global = {
        int(fit_indices[local]): float(oof_margins[local])
        for local in negative_local
    }
    hard_count = len(positives) * hard_per_positive
    hard = sorted(
        negative_global,
        key=lambda index: (
            -margin_by_global[index],
            str(data.group_hashes[index]),
            int(data.group_lengths[index]),
        ),
    )[:hard_count]
    if len(hard) != hard_count:
        raise ValueError("V3-E3 hard-negative pool is undersized")
    hard_set = set(hard)
    remaining = [index for index in negative_global if index not in hard_set]
    strata: dict[tuple[str, str], list[int]] = {}
    for index in remaining:
        month = str(data.dates[index])[:7]
        key = (month, str(data.labels[index]))
        strata.setdefault(key, []).append(index)
    background_count = len(positives) * background_per_positive
    allocation = _allocate_largest_remainder(
        {key: len(values) for key, values in strata.items()},
        background_count,
    )
    background = []
    for key in sorted(strata):
        ordered = sorted(
            strata[key],
            key=lambda index: _stable_group_key(
                data,
                index,
                f"{fold_id}|ratio={ratio}|background",
            ),
        )
        background.extend(ordered[: allocation[key]])
    if len(background) != background_count:
        raise ValueError("V3-E3 background sample is undersized")
    if set(background).intersection(hard_set):
        raise ValueError("V3-E3 hard and background samples overlap")
    selected = np.asarray(sorted(positives + hard + background), dtype=np.int64)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("V3-E3 sampled groups are not unique")
    evidence = {
        "fold_id": fold_id,
        "negative_ratio": ratio,
        "positive_groups": len(positives),
        "hard_negative_groups": len(hard),
        "background_negative_groups": len(background),
        "negative_groups": len(hard) + len(background),
        "sample_rows": len(selected),
        "background_strata": len(strata),
        "sample_signature": _sample_signature(
            data, positives, hard, background
        ),
        "same_sample_for_all_representations": True,
    }
    return selected, evidence


def _score_sparse(
    vectorizer: Any,
    estimator: Any,
    texts: np.ndarray,
    batch: int,
) -> np.ndarray:
    """Score sparse text features in bounded batches."""

    margins = np.empty(len(texts), dtype=np.float64)
    for start in range(0, len(texts), batch):
        end = min(start + batch, len(texts))
        matrix = vectorizer.transform(texts[start:end].tolist())
        margins[start:end] = estimator.decision_function(matrix)
    return margins


def _score_dense(estimator: Any, matrix: np.ndarray, batch: int) -> np.ndarray:
    """Score frozen dense embeddings in bounded float32 batches."""

    margins = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), batch):
        end = min(start + batch, len(matrix))
        values = np.asarray(matrix[start:end], dtype=np.float32)
        margins[start:end] = estimator.decision_function(values)
    return margins


def _fit_candidate(
    data: DevelopmentData,
    sample_indices: np.ndarray,
    calibration_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    representation: str,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one representation-specific probe and return future margins."""

    started = time.perf_counter()
    target = _binary_target(data.labels[sample_indices])
    estimator = _candidate_estimator(balanced=False)
    if representation == "tfidf_word_char":
        from consumer_complaint_intelligence.v2_detector import (
            WORD_CHAR_TFIDF_ALIAS,
            build_vectorizer,
        )

        vectorizer = build_vectorizer(WORD_CHAR_TFIDF_ALIAS)
        fit_matrix = vectorizer.fit_transform(data.texts[sample_indices].tolist())
        estimator.fit(fit_matrix, target)
        del fit_matrix
        calibration_margins = _score_sparse(
            vectorizer,
            estimator,
            data.texts[calibration_indices],
            batch_rows,
        )
        evaluation_margins = _score_sparse(
            vectorizer,
            estimator,
            data.texts[evaluation_indices],
            batch_rows,
        )
        del vectorizer
    elif representation == "minilm_frozen":
        fit_matrix = np.asarray(data.embeddings[sample_indices], dtype=np.float32)
        estimator.fit(fit_matrix, target)
        del fit_matrix
        calibration_margins = _score_dense(
            estimator,
            data.embeddings[calibration_indices],
            batch_rows,
        )
        evaluation_margins = _score_dense(
            estimator,
            data.embeddings[evaluation_indices],
            batch_rows,
        )
    else:
        raise ValueError(f"Unsupported V3-E3 representation: {representation}")
    runtime = {
        "seconds": float(time.perf_counter() - started),
        "linear_svc_n_iter": int(estimator.n_iter_),
    }
    del estimator
    gc.collect()
    return calibration_margins, evaluation_margins, runtime


def calibrate_sigmoid(
    calibration_margins: Sequence[float],
    calibration_target: Sequence[int],
    evaluation_margins: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one deterministic sigmoid on the natural calibration window."""

    from sklearn.linear_model import LogisticRegression

    calibration_scores = np.asarray(calibration_margins, dtype=np.float64)
    evaluation_scores = np.asarray(evaluation_margins, dtype=np.float64)
    target = np.asarray(calibration_target, dtype=np.int8)
    if np.unique(target).size != 2:
        raise ValueError("V3-E3 sigmoid calibration requires both classes")
    calibrator = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    calibrator.fit(calibration_scores.reshape(-1, 1), target)
    calibration_probabilities = calibrator.predict_proba(
        calibration_scores.reshape(-1, 1)
    )[:, 1]
    evaluation_probabilities = calibrator.predict_proba(
        evaluation_scores.reshape(-1, 1)
    )[:, 1]
    return calibration_probabilities, evaluation_probabilities


def select_threshold(
    target: Sequence[int],
    probabilities: Sequence[float],
    precision_floor: float = PRECISION_FLOOR,
) -> dict[str, Any]:
    """Select the calibration-only F1 maximum subject to a precision floor."""

    from sklearn.metrics import precision_recall_curve

    actual = np.asarray(target, dtype=np.int8)
    values = np.asarray(probabilities, dtype=np.float64)
    if actual.ndim != 1 or values.shape != actual.shape or not len(actual):
        raise ValueError("V3-E3 threshold arrays must be non-empty and aligned")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("V3-E3 threshold probabilities are invalid")
    precision, recall, thresholds = precision_recall_curve(actual, values)
    usable_precision = precision[:-1]
    usable_recall = recall[:-1]
    f1 = np.divide(
        2 * usable_precision * usable_recall,
        usable_precision + usable_recall,
        out=np.zeros_like(usable_precision),
        where=(usable_precision + usable_recall) != 0,
    )
    feasible = np.flatnonzero(usable_precision >= float(precision_floor))
    if not len(feasible):
        threshold = float(np.nextafter(np.max(values), np.inf))
        return {
            "threshold": threshold,
            "critical_precision": 0.0,
            "critical_recall": 0.0,
            "critical_f1": 0.0,
            "positive_decision_rate": 0.0,
            "precision_floor_met": False,
            "threshold_candidates": int(len(thresholds)),
        }
    selected = max(
        (int(index) for index in feasible),
        key=lambda index: (
            float(f1[index]),
            float(usable_recall[index]),
            float(usable_precision[index]),
            float(thresholds[index]),
        ),
    )
    decisions = values >= float(thresholds[selected])
    return {
        "threshold": float(thresholds[selected]),
        "critical_precision": float(usable_precision[selected]),
        "critical_recall": float(usable_recall[selected]),
        "critical_f1": float(f1[selected]),
        "positive_decision_rate": float(np.mean(decisions)),
        "precision_floor_met": True,
        "threshold_candidates": int(len(thresholds)),
    }


def _expected_calibration_error(
    target: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    """Calculate equal-width expected calibration error."""

    assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
    total = len(target)
    error = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(
            float(np.mean(probabilities[mask])) - float(np.mean(target[mask]))
        )
    return float(error if total else np.nan)


def evaluate_probabilities(
    target: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    precision_floor: float = PRECISION_FLOOR,
) -> dict[str, Any]:
    """Return aggregate binary, ranking, and calibration metrics."""

    from sklearn.metrics import (
        auc,
        average_precision_score,
        brier_score_loss,
        precision_recall_curve,
    )

    actual = np.asarray(target, dtype=np.int8)
    values = np.asarray(probabilities, dtype=np.float64)
    decisions = values >= float(threshold)
    true_positive = int(np.count_nonzero((actual == 1) & decisions))
    false_positive = int(np.count_nonzero((actual == 0) & decisions))
    false_negative = int(np.count_nonzero((actual == 1) & ~decisions))
    true_negative = int(np.count_nonzero((actual == 0) & ~decisions))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    curve_precision, curve_recall, _ = precision_recall_curve(actual, values)
    pr_auc = auc(curve_recall[::-1], curve_precision[::-1])
    floor_met = precision >= float(precision_floor)
    return {
        "row_count": int(len(actual)),
        "critical_support": int(np.count_nonzero(actual)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "critical_precision": float(precision),
        "critical_recall": float(recall),
        "critical_f1": float(f1),
        "critical_pr_auc": float(pr_auc),
        "critical_average_precision": float(average_precision_score(actual, values)),
        "brier_score": float(brier_score_loss(actual, values)),
        "expected_calibration_error": _expected_calibration_error(actual, values),
        "positive_decision_rate": float(np.mean(decisions)),
        "precision_floor_met": bool(floor_met),
        "critical_recall_at_precision_floor": float(recall if floor_met else 0.0),
    }


def _candidate_id(representation: str, ratio: int) -> str:
    """Return the frozen candidate identifier."""

    return f"{representation}_ratio_1_{ratio}"


def _aggregate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate the three chronological folds for ranking."""

    folds = candidate["folds"]
    recalls = [
        float(fold["evaluation"]["critical_recall_at_precision_floor"])
        for fold in folds
    ]
    pr_auc = [float(fold["evaluation"]["critical_pr_auc"]) for fold in folds]
    f1 = [float(fold["evaluation"]["critical_f1"]) for fold in folds]
    runtime = [float(fold["runtime"]["seconds"]) for fold in folds]
    eligible = len(folds) == 3 and all(
        bool(fold["calibration"]["precision_floor_met"])
        and bool(fold["evaluation"]["precision_floor_met"])
        for fold in folds
    )
    return {
        "eligible": eligible,
        "completed_folds": len(folds),
        "worst_fold_critical_recall_at_precision_floor": min(recalls),
        "median_critical_pr_auc": float(np.median(pr_auc)),
        "median_critical_f1": float(np.median(f1)),
        "critical_f1_between_fold_range": max(f1) - min(f1),
        "runtime_seconds": float(sum(runtime)),
    }


def _system_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Return aggregate multiclass and critical metrics for the F3 system."""

    from sklearn.metrics import f1_score, precision_recall_fscore_support

    from consumer_complaint_intelligence.s6 import MODELED_FAMILIES

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=list(MODELED_FAMILIES),
        average="macro",
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[CRITICAL_CLASS],
        average=None,
        zero_division=0,
    )
    return {
        "row_count": int(len(actual)),
        "macro_f1": float(macro_f1),
        "critical_precision": float(precision[0]),
        "critical_recall": float(recall[0]),
        "critical_f1": float(f1[0]),
        "critical_support": int(support[0]),
    }


def _s7_labels(root: Path, texts: np.ndarray, batch_size: int) -> np.ndarray:
    """Score only the protocol-authorized F3 future window with frozen S7."""

    from consumer_complaint_intelligence.s7 import load_s7_predictor

    predictor = load_s7_predictor(
        root / "artifacts/s7/consumer_complaint_classifier_s7.joblib",
        root / "config/s7_results.json",
        root / "temp/s7/s7_results.json",
    )
    labels = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size].tolist()
        result = predictor.predict(batch, input_language="en-US")
        predictions = getattr(result, "predictions", result)
        labels.extend(str(getattr(item, "label", item)) for item in predictions)
    if len(labels) != len(texts):
        raise ValueError("V3-E3 S7 prediction support differs")
    return np.asarray(labels, dtype=object)


def _compare_with_s7_f3(
    root: Path,
    data: DevelopmentData,
    evaluation_indices: np.ndarray,
    decisions: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    """Compare the selected detector with S7 only on the authorized F3 window."""

    actual = data.labels[evaluation_indices]
    fallback = _s7_labels(root, data.texts[evaluation_indices], batch_size)
    combined = np.where(decisions, CRITICAL_CLASS, fallback)
    selected_metrics = _system_metrics(actual, combined)
    fallback_metrics = _system_metrics(actual, fallback)
    effective_overrides = int(
        np.count_nonzero(decisions & (fallback != CRITICAL_CLASS))
    )
    gain = (
        selected_metrics["critical_f1"] - fallback_metrics["critical_f1"]
    )
    gates = {
        "macro_f1_at_least_0_69": selected_metrics["macro_f1"] >= 0.69,
        "critical_f1_at_least_0_2715": (
            selected_metrics["critical_f1"] >= 0.2715
        ),
        "critical_precision_at_least_0_20": (
            selected_metrics["critical_precision"] >= 0.2
        ),
        "paired_critical_f1_gain_strictly_positive": gain > 0,
    }
    return {
        "fold_id": "F3",
        "s7_use_in_f1_f2": "FORBIDDEN_NOT_EXECUTED",
        "selected_system": selected_metrics,
        "s7_fallback_alone": fallback_metrics,
        "override_decisions": int(np.count_nonzero(decisions)),
        "effective_overrides": effective_overrides,
        "paired_critical_f1_gain_vs_s7": float(gain),
        "gates": gates,
        "gate_count": len(gates),
        "gates_passed": sum(bool(value) for value in gates.values()),
        "eligible_for_development_packaging": all(gates.values()),
    }


def run_rolling_probes(
    project_root: str | Path = ".",
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Execute V3-E3 once and persist aggregate development evidence only."""

    root = Path(project_root).expanduser().resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_rolling_config(config_file)
    protocol_manifest = validate_protocol_manifest(
        config["protocol_manifest"], root
    )
    protocol = load_v3_protocol(root / protocol_manifest["protocol"]["path"])
    protocol.validate_for_scientific_execution()
    e2_audit = audit_v3_e2_cache(root)
    if e2_audit["status"] != "VALID_COMPLETE_DEVELOPMENT_CACHE":
        raise ValueError("V3-E3 requires a valid completed E2 cache")
    result_path = _safe_path(root, config["output"]["result"])
    manifest_path = _safe_path(root, config["output"]["manifest"])
    if result_path.exists() or manifest_path.exists():
        raise FileExistsError("V3-E3 result already exists; overwrite is forbidden")

    started = time.perf_counter()
    data = _load_development_data(root, config)
    candidates = {
        _candidate_id(representation, ratio): {
            "candidate_id": _candidate_id(representation, ratio),
            "representation": representation,
            "negative_ratio": ratio,
            "folds": [],
        }
        for representation in REPRESENTATIONS
        for ratio in RATIOS
    }
    sampling_evidence = []
    f3_decisions: dict[str, np.ndarray] = {}
    f3_evaluation_indices: np.ndarray | None = None
    folds = protocol.payload["rolling_origin"]["folds"]
    batch_rows = int(config["execution"]["transform_batch_rows"])
    composition = config["sampling"]["negative_composition"]
    for fold in folds:
        fold_id = str(fold["fold_id"])
        fold_indices = _fold_indices(data, fold)
        print(
            f"V3-E3 {fold_id} supports: fit={len(fold_indices['fit'])}, "
            f"calibration={len(fold_indices['calibration'])}, "
            f"evaluation={len(fold_indices['evaluation'])}",
            flush=True,
        )
        oof_margins = _mine_oof_margins(data, fold_indices["fit"], fold_id)
        samples = {}
        for ratio in RATIOS:
            counts = composition[str(ratio)]
            sample, evidence = sample_fit_pool(
                data,
                fold_indices["fit"],
                oof_margins,
                fold_id=fold_id,
                ratio=ratio,
                hard_per_positive=int(counts["hard"]),
                background_per_positive=int(counts["background"]),
            )
            samples[ratio] = sample
            sampling_evidence.append(evidence)
        del oof_margins
        for representation in REPRESENTATIONS:
            for ratio in RATIOS:
                candidate_id = _candidate_id(representation, ratio)
                calibration_margins, evaluation_margins, runtime = _fit_candidate(
                    data,
                    samples[ratio],
                    fold_indices["calibration"],
                    fold_indices["evaluation"],
                    representation,
                    batch_rows,
                )
                calibration_target = _binary_target(
                    data.labels[fold_indices["calibration"]]
                )
                evaluation_target = _binary_target(
                    data.labels[fold_indices["evaluation"]]
                )
                calibration_probabilities, evaluation_probabilities = (
                    calibrate_sigmoid(
                        calibration_margins,
                        calibration_target,
                        evaluation_margins,
                    )
                )
                threshold = select_threshold(
                    calibration_target,
                    calibration_probabilities,
                )
                evaluation = evaluate_probabilities(
                    evaluation_target,
                    evaluation_probabilities,
                    threshold["threshold"],
                )
                sample_evidence = next(
                    item
                    for item in sampling_evidence
                    if item["fold_id"] == fold_id
                    and item["negative_ratio"] == ratio
                )
                candidates[candidate_id]["folds"].append(
                    {
                        "fold_id": fold_id,
                        "sampling_signature": sample_evidence["sample_signature"],
                        "calibration": threshold,
                        "evaluation": evaluation,
                        "runtime": runtime,
                    }
                )
                if fold_id == "F3":
                    f3_decisions[candidate_id] = (
                        evaluation_probabilities >= threshold["threshold"]
                    )
                    f3_evaluation_indices = fold_indices["evaluation"]
                print(
                    f"V3-E3 {fold_id} {candidate_id}: "
                    f"eval_f1={evaluation['critical_f1']:.4f}, "
                    f"precision={evaluation['critical_precision']:.4f}, "
                    f"recall={evaluation['critical_recall']:.4f}",
                    flush=True,
                )
                del (
                    calibration_margins,
                    evaluation_margins,
                    calibration_probabilities,
                    evaluation_probabilities,
                )
                gc.collect()
        del samples
        gc.collect()

    for candidate in candidates.values():
        candidate["aggregate"] = _aggregate_candidate(candidate)
    eligible = [
        candidate
        for candidate in candidates.values()
        if candidate["aggregate"]["eligible"]
    ]
    ranking = sorted(
        candidates.values(),
        key=lambda candidate: (
            not candidate["aggregate"]["eligible"],
            -candidate["aggregate"][
                "worst_fold_critical_recall_at_precision_floor"
            ],
            -candidate["aggregate"]["median_critical_pr_auc"],
            -candidate["aggregate"]["median_critical_f1"],
            candidate["aggregate"]["critical_f1_between_fold_range"],
            candidate["aggregate"]["runtime_seconds"],
            candidate["candidate_id"],
        ),
    )
    selected_id = ranking[0]["candidate_id"] if eligible else None
    system_comparison = None
    if selected_id is not None:
        if f3_evaluation_indices is None:
            raise RuntimeError("V3-E3 F3 evaluation indices are unavailable")
        system_comparison = _compare_with_s7_f3(
            root,
            data,
            f3_evaluation_indices,
            f3_decisions[selected_id],
            int(config["execution"]["s7_prediction_batch_rows"]),
        )
    status = (
        "COMPLETE_SELECTED_DEVELOPMENT_CANDIDATE"
        if selected_id is not None
        else "COMPLETE_NO_ELIGIBLE_CANDIDATE"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "stage": "V3-E3",
        "status": status,
        "complete": True,
        "development_only": True,
        "confirmatory": False,
        "deploy": False,
        "selection_authorized": True,
        "allowed_partitions": list(ALLOWED_PARTITIONS),
        "forbidden_partition_access": False,
        "critical_class": CRITICAL_CLASS,
        "protocol_file_sha256": protocol_manifest["protocol"]["sha256"],
        "execution_config": {
            "path": config_file.relative_to(root).as_posix(),
            **_signature(config_file),
        },
        "embedding_manifest": {
            "path": config["embedding_manifest"],
            **_signature(_safe_path(root, config["embedding_manifest"])),
        },
        "sampling": sampling_evidence,
        "candidates": list(candidates.values()),
        "ranking": [
            {
                "rank": index,
                "candidate_id": candidate["candidate_id"],
                **candidate["aggregate"],
            }
            for index, candidate in enumerate(ranking, start=1)
        ],
        "eligible_candidate_count": len(eligible),
        "selected_candidate_id": selected_id,
        "f3_system_comparison": system_comparison,
        "runtime_seconds": float(time.perf_counter() - started),
        "privacy": {
            "individual_samples_persisted": False,
            "individual_scores_persisted": False,
            "fitted_estimators_persisted": False,
            "raw_narratives_persisted": False,
        },
    }
    _validate_privacy(result)
    _atomic_json(result_path, result)
    result_signature = _signature(result_path)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": "V3-E3",
        "status": status,
        "complete": True,
        "development_only": True,
        "confirmatory": False,
        "deploy": False,
        "result": {
            "path": result_path.relative_to(root).as_posix(),
            **result_signature,
        },
        "execution_config": result["execution_config"],
        "protocol_file_sha256": result["protocol_file_sha256"],
        "selected_candidate_id": selected_id,
        "eligible_candidate_count": len(eligible),
        "f3_system_comparison": system_comparison,
        "forbidden_partition_access": False,
        "raw_narratives_persisted": False,
    }
    _validate_privacy(manifest)
    _atomic_json(manifest_path, manifest)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run V3-E3 only through the explicit ``--run`` gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.run:
        raise SystemExit("V3-E3 is disabled by default; pass --run explicitly")
    result = run_rolling_probes()
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "status": result["status"],
                "selected_candidate_id": result["selected_candidate_id"],
                "eligible_candidate_count": result["eligible_candidate_count"],
                "runtime_seconds": result["runtime_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
