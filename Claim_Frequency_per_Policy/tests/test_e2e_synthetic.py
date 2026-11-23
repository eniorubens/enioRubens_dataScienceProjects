"""Pipeline completo (particao, EDA, comparacao) em dados sinteticos."""

import json

import pandas as pd
import pytest

from freq_sinistros import artifacts, design, stage3, stage4
from freq_sinistros.data import load_partitions
from tests.test_stage2 import _toy_project


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    _toy_project(root, n=9000)
    assert design.run_partition(root) == 0
    assert stage3.run_eda(root) == 0
    assert stage3.run_compare(root) == 0
    return root


def test_loader_returns_only_requested_partitions(project):
    parts, provenance = load_partitions(project, keep=("train",))
    assert set(parts) == {"train"}
    assert provenance["sizes"]["train"] == len(parts["train"])
    both, _ = load_partitions(project)
    assert set(both) == {"train", "validation"}
    assert set(parts["train"]["IDpol"]).isdisjoint(both["validation"]["IDpol"])


def test_comparison_outputs_reconcile_and_origin_verifies(project):
    origin = artifacts.verify_origin(project)
    assert origin["all_ok"], origin["checks"]
    out = artifacts.load_outputs(project)
    assert out["reconciliation"]["ok"].all()
    assert out["fits"]["converged"].all()
    config = out["config"]
    assert config["viable_degrees"] == ["100000", "200000", "train_full"]
    assert config["production_eligible"] is False
    assert "test" not in config["provenance"]["sizes"]
    amendment = json.loads(
        (project / "outputs/etapa_03/protocol_amendment.json").read_text())
    assert amendment["production_eligible"] is False


def test_artifact_helpers_on_synthetic_outputs(project):
    out = artifacts.load_outputs(project)
    metrics = artifacts.main_metrics(out, "train_full")
    assert set(metrics["candidate"]) == {"global", "poisson", "nb2"}
    rules = artifacts.rules_table(out)
    assert set(rules["preferred_by_rules"]) <= {"global", "poisson", "nb2"}
    pivot = artifacts.calibration_pivot(out, "exposure_band", "train_full")
    assert {"poisson", "nb2", "global"} <= set(pivot.columns)
    assert len(artifacts.sensitivity_table(out)) > 0
    assert artifacts.reconciliation_summary(out)["all_ok"].all()
    assert len(artifacts.zero_table(out, "train_full")) > 0
    diag = artifacts.zero_table(
        out, "train_full", "validation_free_exposure_diagnostic",
        ("poisson_free_exposure",))
    assert set(diag["candidate"]) == {"poisson_free_exposure"}
    totals = artifacts.population_totals(out)
    main = totals[totals["population"] == "validation"].iloc[0]
    grouped = totals[totals["population"] == "grouped_validation"].iloc[0]
    assert main["records"] == out["config"]["validation"]["rows"]
    assert main["observed_claims"] == out["config"]["validation"][
        "observed_claims"]
    assert grouped["records"] != main["records"]
    assert bool(main["all_ok"]) and bool(grouped["all_ok"])
    sub = totals[totals["population"].str.contains("descriptive")].iloc[0]
    assert sub["records"] < main["records"]
    assert not sub["reconciled_by_segment"]
    sub = artifacts.select_intervals(
        out["intervals"], "train_full", "validation",
        artifacts.KEY_INDICATORS)
    assert len(sub) == len(artifacts.KEY_INDICATORS)
    table = artifacts.provenance_table(
        {"arff_sha256": "a", "assignment_sha256": "b",
         "partition_rule": "r"}, 10, "en_us")
    assert len(table) == 6


def test_eda_tables_and_summary_use_only_train(project):
    summary = json.loads(
        (project / "outputs/etapa_03/eda_summary.json").read_text())
    assert summary["partition_used"] == "train"
    parts, _ = load_partitions(project, keep=("train",))
    assert summary["train_rows"] == len(parts["train"])
    pre = artifacts.coverage_table(summary, "pt_br")
    assert len(pre) == 5
    assert len(artifacts.extremes_table(summary, "en_us")) == 6
    dist = pd.read_csv(
        project / "outputs/etapa_03/eda_claimnb_distribution.csv")
    assert dist["rows"].sum() == summary["train_rows"]


def test_stage4_freeze_verify_and_tamper(project, monkeypatch):
    assert stage4.run_stage4(project) == 0
    out = project / "outputs/etapa_04"
    for name in ("frozen_model.json", "zero_diagnostics.csv",
                 "exposure_band_rates.csv", "model_decision.csv",
                 "zero_gate.json", "stage5_script.json", "decision.json",
                 "reproduction_check.json"):
        assert (out / name).exists()
    zeros = pd.read_csv(out / "zero_reconciliation.csv")
    assert zeros["ok"].all()
    frozen = json.loads((out / "frozen_model.json").read_text())
    assert frozen["production_eligible"] is False
    assert frozen["estimator"]["bonus_malus_included"] is False
    assert not any(c.startswith("BonusMalus")
                   for c in frozen["estimator"]["columns"])

    def forbidden(*args, **kwargs):
        raise AssertionError("reajuste nao permitido na verificacao")

    monkeypatch.setattr("freq_sinistros.models.fit_poisson", forbidden)
    monkeypatch.setattr("freq_sinistros.prep.Preprocessor.fit", forbidden)
    assert stage4.verify_freeze(project) == 0
    result = json.loads((out / "verify_freeze.json").read_text())
    assert result["ok"] and not result["refit_performed"]
    frozen["estimator"]["coefficients"][0] += 0.01
    (out / "frozen_model.json").write_text(json.dumps(frozen))
    assert stage4.verify_freeze(project) == 1
