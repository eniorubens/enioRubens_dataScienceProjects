"""Testes das funcoes de leitura e visualizacao de ``artifacts`` ainda sem
cobertura: dicionario de variaveis, raiz do projeto e artefatos da etapa 4.

Usa um projeto sintetico gerado em ``tmp_path``; nao toca em ``outputs/``.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from freq_sinistros import artifacts, design, stage3, stage4
from tests.test_stage2 import _toy_project


@pytest.fixture(scope="module")
def stage4_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts_s4")
    _toy_project(root, n=9000)
    assert design.run_partition(root) == 0
    assert stage3.run_eda(root) == 0
    assert stage3.run_compare(root) == 0
    assert stage4.run_stage4(root) == 0
    assert stage4.verify_freeze(root) == 0
    return root


# ------------------------------------------------- dicionario e raiz

def test_variable_dictionary_table_both_languages():
    columns = ["VehPower", "ClaimNb", "Exposure", "ColunaInexistente"]
    pt = artifacts.variable_dictionary_table(columns, "pt_br")
    en = artifacts.variable_dictionary_table(pd.Index(columns), "en_us")
    assert list(pt.columns)[0] == "variável"
    assert list(en.columns)[0] == "variable"
    assert len(pt) == len(en) == len(columns)
    assert list(pt.iloc[:, 0]) == columns
    # nome abreviado ganha o significado: "papel: significado"
    assert ": " in pt.iloc[0, 1] and "potência do veículo" in pt.iloc[0, 1]
    assert "vehicle power" in en.iloc[0, 1]
    # coluna sem entrada no dicionario cai em n/d, sem duplicar o texto
    assert list(pt.iloc[3, 1:]) == ["n/d", "n/d", "n/d"]


def test_variable_dictionary_table_rejects_unknown_language():
    with pytest.raises(ValueError, match="idioma"):
        artifacts.variable_dictionary_table(["ClaimNb"], "fr_fr")


def test_project_root_finds_ancestor_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert artifacts.project_root(nested) == tmp_path.resolve()


def test_project_root_raises_without_pyproject(monkeypatch, tmp_path):
    # o diretorio temporario fica dentro do projeto; simula a ausencia
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError, match="pyproject"):
        artifacts.project_root(tmp_path)


# ------------------------------------------------------- etapa 4

def test_load_stage4_reads_tables_and_json(stage4_project):
    s4 = artifacts.load_stage4(stage4_project)
    assert set(artifacts.STAGE4_TABLES) <= set(s4)
    assert set(artifacts.STAGE4_JSON) <= set(s4)
    assert isinstance(s4["model_decision"], pd.DataFrame)
    assert s4["frozen_model"]["production_eligible"] is False


def test_verify_stage4_passes_on_untouched_project(stage4_project):
    result = artifacts.verify_stage4(stage4_project)
    assert result["all_ok"], result["checks"]
    assert {"arff", "verify_freeze_ok", "test_not_used",
            "reproduction_of_stage3_ok", "partition_config",
            "training_sample_matches_stage3"} <= set(result["checks"])


def test_verify_stage4_detects_changed_source(stage4_project, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(stage4_project, copy)
    path = copy / "outputs/etapa_03/protocol_amendment.json"
    path.write_text(path.read_text(encoding="utf-8") + " ",
                    encoding="utf-8")
    result = artifacts.verify_stage4(copy)
    assert not result["all_ok"]
    assert result["checks"]["stage3_amendment"] is False
    assert result["checks"]["partition_config"] is True


def test_stage4_views(stage4_project):
    s4 = artifacts.load_stage4(stage4_project)
    ok = artifacts.zero_reconciliation_ok(s4)
    assert ok.all() and set(ok.index) >= {"poisson"}
    gate = artifacts.zero_gate_table(s4)
    assert "bands_with_support_ci_below_zero" in gate.columns
    assert all(isinstance(v, str)
               for v in gate["bands_with_support_ci_below_zero"])
    second = artifacts.second_process_table(s4)
    assert list(second.index) == [
        "criterion", "hypothesis_documented", "contract_evidence",
        "decision"]
    assert second["decision"] == s4["zero_gate"]["decision"]


# ------------------------------------------- tabelas pequenas a mao

def test_hand_made_stage4_views_without_pipeline():
    s4 = {"zero_reconciliation": pd.DataFrame({
              "candidate": ["a", "a", "b"], "ok": [True, True, False]}),
          "zero_gate": {
              "by_candidate": {"a": {"bands_with_support_ci_below_zero":
                                     ["b1", "b2"]},
                               "b": {"bands_with_support_ci_below_zero":
                                     []}},
              "criterion": "c", "decision": "d",
              "plausible_second_process_hypothesis": {
                  "documented": False, "contract_evidence": "e"}}}
    ok = artifacts.zero_reconciliation_ok(s4)
    assert bool(ok["a"]) and not bool(ok["b"])
    gate = artifacts.zero_gate_table(s4)
    assert gate.loc["a", "bands_with_support_ci_below_zero"] == "b1, b2"
    assert gate.loc["b", "bands_with_support_ci_below_zero"] == ""
    second = artifacts.second_process_table(s4)
    assert second["criterion"] == "c" and not second["hypothesis_documented"]
