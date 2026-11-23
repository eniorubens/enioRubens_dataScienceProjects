"""Testes sinteticos da etapa 5: portao, trava, avaliacao e verificacao."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freq_sinistros import artifacts, design, plots, stage3, stage4, stage5
from freq_sinistros.cli import main
from tests.test_stage2 import _toy_project

MODULES = ["freeze", "arff", "profiles"]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("s5")
    _toy_project(root, n=9000)
    assert design.run_partition(root) == 0
    assert stage3.run_eda(root) == 0
    assert stage3.run_compare(root) == 0
    assert stage4.run_stage4(root) == 0
    return root


@pytest.fixture
def project(base, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(base, target, ignore=shutil.ignore_patterns("etapa_05"))
    return target


def approved(root):
    return {
        "frozen_model_sha256": stage5.file_sha(
            root / "outputs/etapa_04/frozen_model.json"),
        "stage5_script_sha256": stage5.file_sha(
            root / "outputs/etapa_04/stage5_script.json"),
        "modules": stage5.module_hashes(stage5.default_package_dir(),
                                        MODULES)}


@pytest.fixture
def no_test_rows(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("linhas de teste lidas antes do portao")

    monkeypatch.setattr(stage5, "_load_test_rows", forbidden)


def test_approved_constants_match_reviewed_values():
    assert stage5.APPROVED["frozen_model_sha256"].startswith("8ac94b7e")
    assert stage5.APPROVED["stage5_script_sha256"].startswith("c0365ad5")
    assert set(stage5.APPROVED["modules"]) == set(MODULES)


def test_gate_passes_without_reading_test(project, no_test_rows):
    gate = stage5.integrity_gate(project, approved(project))
    assert gate.ok and all(gate.checks.values())
    info = json.loads(
        (project / "outputs/etapa_05/integrity_gate.json").read_text())
    assert info["test_rows_read"] == 0
    assert not (project / "outputs/etapa_05" / stage5.LOCK_NAME).exists()


def test_gate_detects_changed_model_and_script(project, no_test_rows):
    ref = approved(project)
    path = project / "outputs/etapa_04/frozen_model.json"
    path.write_text(path.read_text().replace("0.", "0.0", 1))
    with pytest.raises(stage5.IntegrityError, match="frozen_model"):
        stage5.run_official(project, ref)
    assert not (project / "outputs/etapa_05" / stage5.LOCK_NAME).exists()


def test_gate_detects_changed_script(project, no_test_rows):
    ref = approved(project)
    path = project / "outputs/etapa_04/stage5_script.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(stage5.IntegrityError, match="stage5_script"):
        stage5.integrity_gate(project, ref)


def test_gate_detects_changed_frozen_code(project, tmp_path, no_test_rows):
    package = tmp_path / "pkg"
    shutil.copytree(stage5.default_package_dir(), package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    with open(package / "prep.py", "a", encoding="utf-8") as handle:
        handle.write("\n# alteracao\n")
    with pytest.raises(stage5.IntegrityError, match="code_modules"):
        stage5.integrity_gate(project, approved(project), package)


def test_gate_detects_complementary_module_and_data_changes(
        project, no_test_rows):
    wrong = approved(project)
    wrong["modules"]["profiles"] = "0" * 64
    with pytest.raises(stage5.IntegrityError, match="module_profiles"):
        stage5.integrity_gate(project, wrong)
    raw = next((project / "data/raw").glob("*.arff"))
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(stage5.IntegrityError, match="arff_sha256"):
        stage5.integrity_gate(project, approved(project))


def test_check_only_cli_and_exit_codes(project, no_test_rows, monkeypatch):
    monkeypatch.setattr(stage5, "APPROVED", approved(project))
    assert main(["--root", str(project), "check-stage5"]) == 0
    path = project / "outputs/etapa_04/stage5_script.json"
    path.write_text(path.read_text() + " ")
    assert main(["--root", str(project), "check-stage5"]) == 1


def test_official_run_is_single_and_aggregate_only(project, monkeypatch):
    ref = approved(project)
    assert stage5.run_official(project, ref) == 0
    out = project / "outputs/etapa_05"
    record = json.loads((out / "official_run.json").read_text())
    expected = json.loads((project / "outputs/etapa_02"
                           / "partition_config.json").read_text())
    n_test = next(r["ids"] for r in expected["sizes"]
                  if r["partition"] == "test")
    assert record["status"] == "completed"
    assert record["test_rows"] == n_test == json.loads(
        (out / "test_population.json").read_text())["rows"]
    assert record["reconciliation_all_ok"]
    rec = pd.read_csv(out / "test_reconciliation.csv")
    assert rec["ok"].all()
    assert set(rec["candidate"]) == {"poisson", "global"}
    for name in ("test_metrics.csv", "test_intervals.csv", "panel.csv",
                 "test_calibration.csv", "conclusion.json",
                 "validation_vs_test_bands.csv"):
        assert (out / name).exists()
    assert max(len(pd.read_csv(p)) for p in out.glob("*.csv")) < 1000
    assert stage5.verify_official(project) == 0
    check = artifacts.verify_stage5(project)
    assert check["all_ok"], check["checks"]
    out5 = artifacts.load_stage5(project)
    assert set(artifacts.metrics_view(out5, "pt_br")["candidato"]) == {
        "poisson", "global"}
    view = artifacts.panel_view(out5, "exposure_band", "poisson", "en_us")
    assert {"records", "policy-years", "low support"} <= set(view.columns)
    assert artifacts.run_summary(out5)["status"] == "completed"
    assert artifacts.gate_checks(out5).all()
    for lang in ("pt_br", "en_us"):
        for fig in (plots.plot_panel(out5["panel"], lang, "region"),
                    plots.plot_validation_vs_test(
                        out5["validation_vs_test_bands"], lang),
                    plots.plot_paired_difference(
                        out5["test_intervals"], lang)):
            plots.plt.close(fig)

    def forbidden(*args, **kwargs):
        raise AssertionError("teste relido")

    monkeypatch.setattr(stage5, "_load_test_rows", forbidden)
    with pytest.raises(stage5.AlreadyEvaluatedError):
        stage5.run_official(project, ref)
    monkeypatch.setattr(stage5, "APPROVED", ref)
    assert main(["--root", str(project), "stage5"]) == 4
    # adulterar um agregado e detectado offline
    target = out / "panel.csv"
    target.write_text(target.read_text() + "x\n")
    assert stage5.verify_official(project) == 1


def test_failure_after_test_rows_preserves_state(project, monkeypatch):
    ref = approved(project)

    def boom(*args, **kwargs):
        raise ValueError("falha simulada")

    monkeypatch.setattr(stage5, "evaluate", boom)
    with pytest.raises(ValueError):
        stage5.run_official(project, ref)
    out = project / "outputs/etapa_05"
    record = json.loads((out / "official_run.json").read_text())
    assert record["status"] == "failed" and record["failed_at"] == "evaluate"
    assert record["test_rows_read"] is True
    assert (out / stage5.LOCK_NAME).exists()
    monkeypatch.undo()
    with pytest.raises(stage5.AlreadyEvaluatedError):
        stage5.run_official(project, ref)


def test_evaluate_uses_frozen_state_and_matches_metrics(project):
    frozen = json.loads(
        (project / "outputs/etapa_04/frozen_model.json").read_text())
    from freq_sinistros.data import load_partitions

    val = load_partitions(project, keep=("validation",))[0]["validation"]
    result = stage5.evaluate(val, frozen)
    ev = result["ev"]
    assert np.isclose(
        result["metrics"].set_index("candidate").loc[
            "poisson", "mean_poisson_deviance"],
        ev.dev["poisson"].mean())
    assert result["reconciliation"]["ok"].all()
    assert set(result["conclusion"]["primary"]) >= {"verdict", "ci_low"}


def intervals(est, lo, hi):
    return pd.DataFrame({"indicator": ["d_dev|poisson-global"],
                         "estimate": [est], "ci_low": [lo],
                         "ci_high": [hi]})


def test_conclusion_classifies_gain_tie_and_reference_better():
    met = pd.DataFrame({
        "candidate": ["poisson", "global"],
        "mean_poisson_deviance": [0.3, 0.31], "abs_rate_error": [0.1, 0.2]})
    for est, lo, hi, verdict in ((-0.01, -0.02, -0.005, "ganho_sustentado"),
                                 (-0.01, -0.02, 0.003, "inconclusivo"),
                                 (0.01, 0.002, 0.02, "referencia_melhor")):
        c = stage5.conclusion(met, intervals(est, lo, hi),
                              intervals(est, lo, hi))
        assert c["primary"]["verdict"] == verdict


def test_validation_vs_test_compares_signs_and_ci():
    cal = pd.DataFrame({
        "candidate": ["poisson"] * 2, "family": ["exposure_band"] * 2,
        "segment": ["(0, 0.02]", "= 1"], "rows": [10, 100],
        "observed_rate_per_100": [200.0, 7.0],
        "predicted_rate_per_100": [15.0, 8.0],
        "rate_difference_pred_minus_obs": [-185.0, 1.0],
        "ci_low": [-250.0, 0.5], "ci_high": [-120.0, 1.5],
        "low_support": [False, False]})
    val = pd.DataFrame({"band": ["(0, 0.02]", "= 1"],
                        "diff_poisson": [-224.0, -0.3]})
    out = stage5.validation_vs_test(cal, val)
    assert out["same_sign"].tolist() == [True, False]
    assert out["validation_diff_inside_test_ci"].tolist() == [True, False]


def test_panel_table_has_roteiro_columns():
    cal = pd.DataFrame({
        "candidate": ["poisson"], "family": ["region"], "segment": ["R11"],
        "rows": [1], "exposure": [1.0], "observed_claims": [0.0],
        "predicted_claims": [0.1], "observed_rate_per_100": [0.0],
        "predicted_rate_per_100": [10.0],
        "rate_difference_pred_minus_obs": [10.0], "ci_low": [np.nan],
        "ci_high": [np.nan], "low_support": [True]})
    panel = stage5.panel_table(cal)
    assert {"records", "policy_years", "observed_claims", "predicted_claims",
            "pred_minus_obs", "low_support"} <= set(panel.columns)


def test_original_project_dir_untouched_by_copy_tests(base):
    assert not (Path(base) / "outputs/etapa_05" / stage5.LOCK_NAME).exists()


def _snapshot(root):
    return {p.relative_to(root).as_posix():
            (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_check_stage5_refused_after_lock(tmp_path, capsys):
    out = tmp_path / "outputs/etapa_05"
    out.mkdir(parents=True)
    (out / stage5.LOCK_NAME).write_text("lock", encoding="utf-8")
    before = _snapshot(tmp_path)
    assert main(["--root", str(tmp_path), "check-stage5"]) == 5
    assert "verify-stage5" in capsys.readouterr().out
    assert _snapshot(tmp_path) == before


def test_verify_stage5_cli_combines_checks_and_never_raises(
        project, capsys):
    assert stage5.run_official(project, approved(project)) == 0
    out = project / "outputs/etapa_05"
    argv = ["--root", str(project), "verify-stage5"]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "single_official_run_lock" in text and "all_files_listed" in text

    lock = out / stage5.LOCK_NAME
    lock_bytes = lock.read_bytes()
    lock.unlink()  # trava ausente: so a conferencia completa detecta
    assert main(argv) == 1
    lock.write_bytes(lock_bytes)
    assert main(argv) == 0

    for name in ("panel.csv", "official_run.json", "SHA256SUMS_etapa05.txt"):
        data = (out / name).read_bytes()
        (out / name).unlink()
        assert main(argv) == 1, name
        assert "ERRO" in capsys.readouterr().out
        (out / name).write_bytes(data)
    assert main(argv) == 0
