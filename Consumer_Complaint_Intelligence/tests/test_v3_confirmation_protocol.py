"""Contract tests for the approved, non-executable V3-C protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config/v3/confirmation_protocol.json"
MANIFEST = ROOT / "config/v3/confirmation_protocol_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_six_methodological_points_are_approved() -> None:
    protocol = _load(CONFIG)
    approval = protocol["approval_record"]
    assert protocol["status"] == "APPROVED_AWAITING_COMPLETE_SOURCE_FREEZE"
    assert protocol["methodological_protocol_frozen"] is True
    assert approval["all_six_points_approved"] is True
    assert [item["id"] for item in approval["points"]] == [1, 2, 3, 4, 5, 6]
    assert all(item["approved"] is True for item in approval["points"])


def test_five_gates_are_frozen_and_all_are_required() -> None:
    protocol = _load(CONFIG)
    block = protocol["confirmatory_gates"]
    assert block["required_gate_count"] == 5
    assert block["all_must_pass"] is True
    assert block["gates"] == {
        "macro_f1_at_least_0_69": {
            "metric": "macro_f1",
            "minimum": 0.69,
            "inclusive": True,
            "origin": "historical_S8_and_V2_1_C",
        },
        "critical_f1_at_least_0_2715": {
            "metric": "critical_f1",
            "minimum": 0.2715,
            "inclusive": True,
            "origin": "historical_S8_and_V2_1_C",
        },
        "critical_precision_at_least_0_20": {
            "metric": "critical_precision",
            "minimum": 0.2,
            "inclusive": True,
            "origin": "historical_S8_and_V2_1_C",
        },
        "paired_critical_f1_gain_strictly_positive": {
            "metric": "paired_critical_f1_gain_vs_s7",
            "minimum": 0.0,
            "inclusive": False,
            "origin": "historical_V2_1_C",
        },
        "system_critical_precision_at_least_0_40": {
            "metric": "critical_precision",
            "minimum": 0.4,
            "inclusive": True,
            "origin": "approved_V3_safety_gate_2026_08_24",
        },
    }
    assert protocol["decision"]["confirmed_if"] == (
        "adequate_scope_and_all_five_gates_pass"
    )


def test_complete_future_source_and_second_approval_remain_required() -> None:
    protocol = _load(CONFIG)
    source = protocol["future_source_freeze"]
    boundary = protocol["execution_boundary"]
    assert protocol["timing"]["window_is_currently_complete"] is False
    assert protocol["timing"]["earliest_source_acquisition"] == "2027-02-01"
    assert source["raw_sha256"].startswith("TBD_")
    assert source["source_identity_approval_required"] is True
    assert protocol["execution_authorized"] is False
    assert boundary["runner_implemented"] is False
    assert boundary["monitor_accessed_by_v3"] is False


def test_manifest_pins_protocol_adr_and_preapproval_archive() -> None:
    manifest = _load(MANIFEST)
    artifacts = [
        manifest["protocol"],
        manifest["adr"],
        manifest["preapproval_archive"]["protocol"],
        manifest["preapproval_archive"]["adr"],
    ]
    for artifact in artifacts:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert _sha256(path) == artifact["sha256"]
    assert manifest["required_gate_count"] == 5
    assert manifest["future_source_identity_frozen"] is False
    assert manifest["execution_authorized"] is False


def test_frozen_model_control_and_history_hashes_are_current() -> None:
    protocol = _load(CONFIG)
    blocks = (
        protocol["frozen_v3_package"],
        protocol["frozen_s7_control"],
    )
    artifacts = [item for block in blocks for item in block.values()]
    artifacts.append(protocol["frozen_history_index"])
    artifacts.append(protocol["current_snapshot_limitation"]["s1_report"])
    missing: list[str] = []
    for artifact in artifacts:
        path = ROOT / artifact["path"]
        if not path.exists():
            missing.append(artifact["path"])
            continue
        assert path.stat().st_size == artifact["size_bytes"]
        assert _sha256(path) == artifact["sha256"]
    if missing:
        pytest.skip(
            "local frozen historical artifacts are not part of the repository: "
            + ", ".join(missing)
        )


def test_no_unlock_or_confirmatory_output_contract_exists() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    protocol = json.loads(text)
    assert "unlock_sha256" not in text
    assert "unlock_env" not in text
    assert protocol["execution_boundary"]["unlock_token_defined"] is False
    assert protocol["execution_boundary"]["outputs_defined"] is False
    assert protocol["confirmatory_claim_authorized"] is False
    assert protocol["decision"]["deployment_authorized"] is False
