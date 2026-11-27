"""Read-only validation and reporting for completed V3-E3 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .protocol import load_v3_protocol, validate_protocol_manifest
from .rolling_probes import (
    ALLOWED_PARTITIONS,
    CRITICAL_CLASS,
    RATIOS,
    REPRESENTATIONS,
    _aggregate_candidate,
    _candidate_id,
    _validate_privacy,
    load_rolling_config,
)


DEFAULT_RESULT = "temp/v3/v3_e3_rolling_results.json"
DEFAULT_MANIFEST = "config/v3/e3_results_manifest.json"
DEFAULT_CONFIG = "config/v3/rolling_probes.json"
EXPECTED_FOLDS = ("F1", "F2", "F3")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3-E3 evidence: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("V3-E3 evidence must be a JSON object")
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
        raise ValueError("V3-E3 reporting paths must be project-relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("V3-E3 reporting path escapes the project root")
    return resolved


def _same_number(left: Any, right: Any) -> bool:
    """Compare serialized numeric evidence with strict floating tolerance."""

    return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))


def _validate_candidate_metrics(candidate: Mapping[str, Any]) -> None:
    """Validate bounded aggregate fold metrics for one candidate."""

    folds = candidate.get("folds")
    if not isinstance(folds, list):
        raise ValueError("V3-E3 candidate folds are invalid")
    if tuple(fold.get("fold_id") for fold in folds) != EXPECTED_FOLDS:
        raise ValueError("V3-E3 candidate fold sequence differs")
    bounded = (
        "critical_precision",
        "critical_recall",
        "critical_f1",
        "critical_pr_auc",
        "critical_average_precision",
        "brier_score",
        "expected_calibration_error",
        "positive_decision_rate",
        "critical_recall_at_precision_floor",
    )
    for fold in folds:
        evaluation = fold.get("evaluation", {})
        for metric in bounded:
            value = float(evaluation.get(metric, -1.0))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"V3-E3 metric is invalid: {metric}")
        calibration = fold.get("calibration", {})
        for metric in (
            "critical_precision",
            "critical_recall",
            "critical_f1",
            "positive_decision_rate",
        ):
            value = float(calibration.get(metric, -1.0))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"V3-E3 calibration metric is invalid: {metric}")
        threshold = float(calibration.get("threshold", np.nan))
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("V3-E3 calibrated threshold is invalid")


def audit_v3_e3_results(
    project_root: str | Path = ".",
    *,
    result_path: str = DEFAULT_RESULT,
    manifest_path: str = DEFAULT_MANIFEST,
    config_path: str = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Validate completed V3-E3 aggregate evidence without reopening data."""

    root = Path(project_root).expanduser().resolve()
    result_file = _safe_path(root, result_path)
    manifest_file = _safe_path(root, manifest_path)
    config_file = _safe_path(root, config_path)
    result = _read_json(result_file)
    manifest = _read_json(manifest_file)
    config = load_rolling_config(config_file)
    protocol_manifest = validate_protocol_manifest(project_root=root)
    protocol = load_v3_protocol(root / protocol_manifest["protocol"]["path"])

    allowed_status = {
        "COMPLETE_SELECTED_DEVELOPMENT_CANDIDATE",
        "COMPLETE_NO_ELIGIBLE_CANDIDATE",
    }
    if result.get("schema_version") != "v3-e3-rolling-results-v1":
        raise ValueError("V3-E3 result schema is invalid")
    if result.get("stage") != "V3-E3" or result.get("status") not in allowed_status:
        raise ValueError("V3-E3 result stage or status is invalid")
    required_flags = {
        "complete": True,
        "development_only": True,
        "confirmatory": False,
        "deploy": False,
        "selection_authorized": True,
        "forbidden_partition_access": False,
    }
    if any(result.get(key) is not value for key, value in required_flags.items()):
        raise ValueError("V3-E3 scientific boundary flags are invalid")
    if tuple(result.get("allowed_partitions", ())) != ALLOWED_PARTITIONS:
        raise ValueError("V3-E3 result partition boundary is invalid")
    if result.get("critical_class") != CRITICAL_CLASS:
        raise ValueError("V3-E3 result critical class differs")
    if result.get("protocol_file_sha256") != protocol_manifest["protocol"]["sha256"]:
        raise ValueError("V3-E3 protocol hash differs")

    result_sha = _sha256(result_file)
    if manifest.get("schema_version") != "v3-e3-rolling-results-manifest-v1":
        raise ValueError("V3-E3 manifest schema is invalid")
    if manifest.get("stage") != "V3-E3" or manifest.get("status") != result["status"]:
        raise ValueError("V3-E3 manifest stage or status differs")
    published = manifest.get("result", {})
    if published.get("path") != result_path:
        raise ValueError("V3-E3 manifest result path differs")
    if published.get("sha256") != result_sha:
        raise ValueError("V3-E3 result hash differs from the manifest")
    if int(published.get("size_bytes", -1)) != result_file.stat().st_size:
        raise ValueError("V3-E3 result size differs from the manifest")
    config_signature = result.get("execution_config", {})
    if config_signature.get("path") != config_path:
        raise ValueError("V3-E3 execution config path differs")
    if config_signature.get("sha256") != _sha256(config_file):
        raise ValueError("V3-E3 execution config hash differs")
    if int(config_signature.get("size_bytes", -1)) != config_file.stat().st_size:
        raise ValueError("V3-E3 execution config size differs")
    if manifest.get("execution_config") != config_signature:
        raise ValueError("V3-E3 config signatures differ")

    expected_ids = tuple(
        _candidate_id(representation, ratio)
        for representation in REPRESENTATIONS
        for ratio in RATIOS
    )
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or tuple(
        candidate.get("candidate_id") for candidate in candidates
    ) != expected_ids:
        raise ValueError("V3-E3 candidate catalog differs")
    fold_payload = protocol.payload["rolling_origin"]["folds"]
    expected_support = {
        str(fold["fold_id"]): {
            "rows": int(fold["evaluation"]["rows"]),
            "critical": int(fold["evaluation"]["critical_rows"]),
        }
        for fold in fold_payload
    }
    sampling = result.get("sampling")
    if not isinstance(sampling, list) or len(sampling) != 9:
        raise ValueError("V3-E3 sampling evidence differs")
    sample_map = {
        (str(item.get("fold_id")), int(item.get("negative_ratio", -1))): item
        for item in sampling
    }
    if len(sample_map) != 9:
        raise ValueError("V3-E3 sampling evidence is not unique")
    composition = config["sampling"]["negative_composition"]
    fit_positives = {
        str(fold["fold_id"]): int(fold["fit"]["critical_rows"])
        for fold in fold_payload
    }
    for fold_id in EXPECTED_FOLDS:
        for ratio in RATIOS:
            sample = sample_map[(fold_id, ratio)]
            positives = fit_positives[fold_id]
            counts = composition[str(ratio)]
            expected = {
                "positive_groups": positives,
                "hard_negative_groups": positives * int(counts["hard"]),
                "background_negative_groups": (
                    positives * int(counts["background"])
                ),
                "negative_groups": positives * ratio,
                "sample_rows": positives * (ratio + 1),
            }
            counts_differ = any(
                int(sample.get(key, -1)) != value
                for key, value in expected.items()
            )
            if counts_differ:
                raise ValueError("V3-E3 sampling counts differ")
            signature = str(sample.get("sample_signature", ""))
            if len(signature) != 64 or signature.upper() != signature:
                raise ValueError("V3-E3 sampling signature is invalid")
            if sample.get("same_sample_for_all_representations") is not True:
                raise ValueError("V3-E3 representation samples differ")

    eligible = []
    for candidate in candidates:
        _validate_candidate_metrics(candidate)
        fold_by_id = {fold["fold_id"]: fold for fold in candidate["folds"]}
        ratio = int(candidate["negative_ratio"])
        for fold_id in EXPECTED_FOLDS:
            fold = fold_by_id[fold_id]
            support = expected_support[fold_id]
            evaluation = fold["evaluation"]
            if int(evaluation["row_count"]) != support["rows"]:
                raise ValueError("V3-E3 evaluation row count differs")
            if int(evaluation["critical_support"]) != support["critical"]:
                raise ValueError("V3-E3 evaluation critical support differs")
            if fold.get("sampling_signature") != sample_map[
                (fold_id, ratio)
            ]["sample_signature"]:
                raise ValueError("V3-E3 candidate sample signature differs")
        aggregate = _aggregate_candidate(candidate)
        observed = candidate.get("aggregate", {})
        for key, value in aggregate.items():
            if isinstance(value, bool):
                if observed.get(key) is not value:
                    raise ValueError("V3-E3 aggregate eligibility differs")
            elif not _same_number(observed.get(key), value):
                raise ValueError(f"V3-E3 aggregate metric differs: {key}")
        if aggregate["eligible"]:
            eligible.append(candidate)

    expected_ranking = sorted(
        candidates,
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
    ranking = result.get("ranking")
    observed_ranking = (
        [item.get("candidate_id") for item in ranking]
        if isinstance(ranking, list)
        else []
    )
    expected_ranking_ids = [item["candidate_id"] for item in expected_ranking]
    if observed_ranking != expected_ranking_ids:
        raise ValueError("V3-E3 ranking differs from the frozen rule")
    selected = expected_ranking[0]["candidate_id"] if eligible else None
    if result.get("selected_candidate_id") != selected:
        raise ValueError("V3-E3 selected candidate differs")
    if int(result.get("eligible_candidate_count", -1)) != len(eligible):
        raise ValueError("V3-E3 eligible candidate count differs")
    if manifest.get("selected_candidate_id") != selected:
        raise ValueError("V3-E3 manifest selection differs")

    comparison = result.get("f3_system_comparison")
    if selected is None:
        if comparison is not None:
            raise ValueError("V3-E3 cannot compare S7 without a selection")
    else:
        if not isinstance(comparison, Mapping) or comparison.get("fold_id") != "F3":
            raise ValueError("V3-E3 S7 comparison scope differs")
        if comparison.get("s7_use_in_f1_f2") != "FORBIDDEN_NOT_EXECUTED":
            raise ValueError("V3-E3 reports forbidden S7 use")
        gates = comparison.get("gates", {})
        if int(comparison.get("gate_count", -1)) != len(gates):
            raise ValueError("V3-E3 S7 gate count differs")
        passed = sum(bool(value) for value in gates.values())
        if int(comparison.get("gates_passed", -1)) != passed:
            raise ValueError("V3-E3 S7 passed-gate count differs")
        if comparison.get("eligible_for_development_packaging") is not all(
            gates.values()
        ):
            raise ValueError("V3-E3 S7 packaging gate differs")

    _validate_privacy(result)
    _validate_privacy(manifest)
    return {
        "schema_version": "v3-e3-read-only-audit-v1",
        "stage": "V3-E3",
        "status": "VALID_COMPLETE_DEVELOPMENT_RESULTS",
        "result_status": result["status"],
        "development_only": True,
        "confirmatory": False,
        "deploy": False,
        "allowed_partitions": list(ALLOWED_PARTITIONS),
        "forbidden_partitions_accessed": False,
        "critical_class": CRITICAL_CLASS,
        "result_sha256": result_sha,
        "manifest_sha256": _sha256(manifest_file),
        "execution_config_sha256": _sha256(config_file),
        "protocol_file_sha256": result["protocol_file_sha256"],
        "runtime_seconds": float(result["runtime_seconds"]),
        "eligible_candidate_count": len(eligible),
        "selected_candidate_id": selected,
        "candidates": candidates,
        "ranking": ranking,
        "f3_system_comparison": comparison,
        "privacy": dict(result["privacy"]),
    }


def _candidate_lines(candidate: Mapping[str, Any]) -> list[str]:
    """Render fold metrics for one candidate."""

    lines = []
    for fold in candidate["folds"]:
        metrics = fold["evaluation"]
        lines.append(
            f"    {fold['fold_id']}: P={metrics['critical_precision']:.4f} | "
            f"R={metrics['critical_recall']:.4f} | "
            f"F1={metrics['critical_f1']:.4f} | "
            f"PR-AUC={metrics['critical_pr_auc']:.4f} | "
            f"floor={metrics['precision_floor_met']}"
        )
    return lines


def render_v3_e3_audit(
    audit: Mapping[str, Any],
    language: str = "pt-BR",
) -> str:
    """Render completed V3-E3 aggregate results in PT-BR or EN-US."""

    if language not in {"pt-BR", "en-US"}:
        raise ValueError(f"Unsupported V3-E3 report language: {language}")
    if language == "pt-BR":
        labels = {
            "title": "V3-E3 — RESULTADOS TEMPORAIS DOS PROBES",
            "summary": "RESUMO",
            "ranking": "RANKING CONGELADO",
            "folds": "MÉTRICAS FUTURAS POR CANDIDATO",
            "s7": "COMPARAÇÃO AUTORIZADA COM S7 — SOMENTE F3",
            "boundary": "LIMITE CIENTÍFICO",
            "boundary_text": (
                "A seleção é apenas de desenvolvimento. Teste, stress, "
                "monitoramento, implantação e alegações confirmatórias continuam "
                "bloqueados."
            ),
        }
    else:
        labels = {
            "title": "V3-E3 — TEMPORAL PROBE RESULTS",
            "summary": "SUMMARY",
            "ranking": "FROZEN RANKING",
            "folds": "FUTURE METRICS BY CANDIDATE",
            "s7": "AUTHORIZED S7 COMPARISON — F3 ONLY",
            "boundary": "SCIENTIFIC BOUNDARY",
            "boundary_text": (
                "Selection is development-only. Test, stress, monitoring, deployment, "
                "and confirmatory claims remain blocked."
            ),
        }
    lines = [
        labels["title"],
        "",
        labels["summary"],
        f"  status: {audit['status']}",
        f"  selected_candidate: {audit['selected_candidate_id']}",
        f"  eligible_candidates: {audit['eligible_candidate_count']}",
        f"  runtime_seconds: {audit['runtime_seconds']:.2f}",
        f"  result_sha256: {audit['result_sha256']}",
        f"  manifest_sha256: {audit['manifest_sha256']}",
        "",
        labels["ranking"],
    ]
    for item in audit["ranking"]:
        lines.append(
            f"  {item['rank']}. {item['candidate_id']} | "
            f"eligible={item['eligible']} | "
            "worst_recall@P>=0.40="
            f"{item['worst_fold_critical_recall_at_precision_floor']:.4f} | "
            f"median_PR-AUC={item['median_critical_pr_auc']:.4f} | "
            f"median_F1={item['median_critical_f1']:.4f}"
        )
    lines.extend(["", labels["folds"]])
    for candidate in audit["candidates"]:
        lines.append(
            f"  {candidate['candidate_id']} | "
            f"eligible={candidate['aggregate']['eligible']}"
        )
        lines.extend(_candidate_lines(candidate))
    comparison = audit["f3_system_comparison"]
    lines.extend(["", labels["s7"]])
    if comparison is None:
        lines.append("  unavailable: no eligible V3-E3 candidate")
    else:
        selected = comparison["selected_system"]
        fallback = comparison["s7_fallback_alone"]
        lines.extend(
            [
                f"  rows: {selected['row_count']:,}",
                f"  hybrid_macro_f1: {selected['macro_f1']:.4f}",
                f"  hybrid_critical_precision: {selected['critical_precision']:.4f}",
                f"  hybrid_critical_recall: {selected['critical_recall']:.4f}",
                f"  hybrid_critical_f1: {selected['critical_f1']:.4f}",
                f"  s7_critical_f1: {fallback['critical_f1']:.4f}",
                "  paired_critical_f1_gain_vs_s7: "
                f"{comparison['paired_critical_f1_gain_vs_s7']:.4f}",
                f"  effective_overrides: {comparison['effective_overrides']:,}",
                f"  gates_passed: {comparison['gates_passed']}/"
                f"{comparison['gate_count']}",
                "  eligible_for_development_packaging: "
                f"{comparison['eligible_for_development_packaging']}",
            ]
        )
    lines.extend(["", labels["boundary"], f"  {labels['boundary_text']}"])
    return "\n".join(lines)


__all__ = ["audit_v3_e3_results", "render_v3_e3_audit"]
