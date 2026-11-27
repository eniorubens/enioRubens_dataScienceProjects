"""Tests for the read-only V3-E3 notebook reporting boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from consumer_complaint_intelligence.v3.e3_reporting import (
    audit_v3_e3_results,
    render_v3_e3_audit,
)


ROOT = Path(__file__).parents[1]
RESULT = ROOT / "temp" / "v3" / "v3_e3_rolling_results.json"


def _fake_audit() -> dict:
    """Return compact aggregate evidence for bilingual renderer tests."""

    evaluation = {
        "critical_precision": 0.5,
        "critical_recall": 0.25,
        "critical_f1": 1 / 3,
        "critical_pr_auc": 0.2,
        "precision_floor_met": True,
    }
    candidate = {
        "candidate_id": "tfidf_word_char_ratio_1_10",
        "aggregate": {"eligible": True},
        "folds": [
            {"fold_id": fold, "evaluation": evaluation}
            for fold in ("F1", "F2", "F3")
        ],
    }
    return {
        "status": "VALID_COMPLETE_DEVELOPMENT_RESULTS",
        "selected_candidate_id": candidate["candidate_id"],
        "eligible_candidate_count": 1,
        "runtime_seconds": 100.0,
        "result_sha256": "A" * 64,
        "manifest_sha256": "B" * 64,
        "ranking": [
            {
                "rank": 1,
                "candidate_id": candidate["candidate_id"],
                "eligible": True,
                "worst_fold_critical_recall_at_precision_floor": 0.25,
                "median_critical_pr_auc": 0.2,
                "median_critical_f1": 1 / 3,
            }
        ],
        "candidates": [candidate],
        "f3_system_comparison": {
            "selected_system": {
                "row_count": 100,
                "macro_f1": 0.72,
                "critical_precision": 0.41,
                "critical_recall": 0.33,
                "critical_f1": 0.37,
            },
            "s7_fallback_alone": {"critical_f1": 0.34},
            "paired_critical_f1_gain_vs_s7": 0.03,
            "effective_overrides": 10,
            "gates_passed": 4,
            "gate_count": 4,
            "eligible_for_development_packaging": True,
        },
    }


def test_renderer_is_bilingual_and_preserves_development_boundary() -> None:
    """Localize visible interpretation without confirmatory overclaiming."""

    pt_report = render_v3_e3_audit(_fake_audit(), "pt-BR")
    en_report = render_v3_e3_audit(_fake_audit(), "en-US")
    assert "RESULTADOS TEMPORAIS" in pt_report
    assert "apenas de desenvolvimento" in pt_report
    assert "TEMPORAL PROBE RESULTS" in en_report
    assert "development-only" in en_report
    assert "gates_passed: 4/4" in pt_report
    assert "tfidf_word_char_ratio_1_10" in en_report


def test_renderer_rejects_unsupported_language() -> None:
    """Fail rather than silently publish an unintended localization."""

    with pytest.raises(ValueError, match="language"):
        render_v3_e3_audit(_fake_audit(), "es-ES")


def test_notebook_pair_has_equivalent_structure_and_bounded_code() -> None:
    """Keep both notebooks thin, equivalent, and read-only."""

    paths = (
        ROOT / "notebooks" / "pt-BR" / "15_V3_E3_Temporal_Probes_PT.ipynb",
        ROOT / "notebooks" / "en-US" / "15_V3_E3_Temporal_Probes_EN.ipynb",
    )
    notebooks = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert [cell["cell_type"] for cell in notebooks[0]["cells"]] == [
        cell["cell_type"] for cell in notebooks[1]["cells"]
    ]
    pt_code = [
        "".join(cell["source"])
        for cell in notebooks[0]["cells"]
        if cell["cell_type"] == "code"
    ]
    en_code = [
        "".join(cell["source"])
        for cell in notebooks[1]["cells"]
        if cell["cell_type"] == "code"
    ]
    assert len(pt_code) == len(en_code) == 2
    assert pt_code[0] == en_code[0]
    assert pt_code[1].replace("pt-BR", "LANG") == en_code[1].replace(
        "en-US", "LANG"
    )
    code = "\n".join(pt_code + en_code)
    assert "audit_v3_e3_results" in code
    assert "LinearSVC" not in code
    assert "fit(" not in code
    assert "predict(" not in code
    assert "read_parquet" not in code
    assert "narrative" not in code.lower()


@pytest.mark.skipif(not RESULT.is_file(), reason="local V3-E3 result unavailable")
def test_completed_local_result_passes_read_only_audit() -> None:
    """Exercise the completed aggregate result when it is available locally."""

    audit = audit_v3_e3_results(ROOT)
    assert audit["status"] == "VALID_COMPLETE_DEVELOPMENT_RESULTS"
    assert audit["selected_candidate_id"] == "tfidf_word_char_ratio_1_10"
    assert audit["eligible_candidate_count"] == 3
    assert audit["f3_system_comparison"]["gates_passed"] == 4
    assert audit["f3_system_comparison"][
        "eligible_for_development_packaging"
    ] is True
    assert audit["development_only"] is True
    assert audit["confirmatory"] is False
    assert audit["deploy"] is False
