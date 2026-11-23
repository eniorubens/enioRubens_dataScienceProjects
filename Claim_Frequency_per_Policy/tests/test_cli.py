"""Testes da CLI: despacho de subcomandos e codigos de saida.

As funcoes que fazem o trabalho pesado (e que gravam em ``outputs``) sao
substituidas por dobles; nada aqui usa rede, dados reais ou o workspace.
"""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from freq_sinistros import acquire as acquire_module
from freq_sinistros import artifacts, cli, design, stage3, stage4, stage5
from freq_sinistros import audit as audit_module

SRC = Path(__file__).resolve().parents[1] / "src"


def _recorder(code, calls):
    """Doble que registra a raiz recebida e devolve ``code``."""
    def fake(root, *args, **kwargs):
        calls.append(Path(root))
        return code

    return fake


# ---------------------------------------------------------------- parser

def test_parser_defaults_and_root():
    parser = cli.build_parser()
    args = parser.parse_args(["audit"])
    assert args.command == "audit"
    assert args.root == Path.cwd()
    args = parser.parse_args(["--root", "x", "verify-stage5"])
    assert args.root == Path("x")


@pytest.mark.parametrize("argv", [[], ["inexistente"]])
def test_parser_rejects_missing_or_unknown_command(argv, capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    assert info.value.code == 2
    assert "usage" in capsys.readouterr().err


# ------------------------------------------- despacho e codigo de saida

@pytest.mark.parametrize("command, module, name", [
    ("eda", stage3, "run_eda"),
    ("compare", stage3, "run_compare"),
    ("stage4", stage4, "run_stage4"),
    ("verify-freeze", stage4, "verify_freeze"),
    ("partition", design, "run_partition"),
    ("verify-partition", design, "verify_partition"),
    ("audit", audit_module, "run_audit"),
])
@pytest.mark.parametrize("code", [0, 1, 2])
def test_dispatch_returns_the_function_exit_code(
        command, module, name, code, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module, name, _recorder(code, calls))
    assert cli.main(["--root", str(tmp_path), command]) == code
    assert calls == [tmp_path]


def test_check_stage5_runs_gate_when_not_locked(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stage5, "check_only", _recorder(1, calls))
    assert cli.main(["--root", str(tmp_path), "check-stage5"]) == 1
    assert calls == [tmp_path]


def test_check_stage5_refuses_after_lock(tmp_path, monkeypatch, capsys):
    lock = tmp_path / stage5.OUT5 / stage5.LOCK_NAME
    lock.parent.mkdir(parents=True)
    lock.write_text("x", encoding="utf-8")
    calls = []
    monkeypatch.setattr(stage5, "check_only", _recorder(0, calls))
    assert cli.main(["--root", str(tmp_path), "check-stage5"]) == 5
    assert calls == []
    assert "verify-stage5" in capsys.readouterr().out


def test_verify_freeze_refuses_after_lock(tmp_path, monkeypatch, capsys):
    lock = tmp_path / stage5.OUT5 / stage5.LOCK_NAME
    lock.parent.mkdir(parents=True)
    lock.write_text("x", encoding="utf-8")
    before = sorted(p.relative_to(tmp_path).as_posix()
                    for p in tmp_path.rglob("*"))
    calls = []
    monkeypatch.setattr(stage4, "verify_freeze", _recorder(0, calls))
    assert cli.main(["--root", str(tmp_path), "verify-freeze"]) == 5
    assert calls == []
    assert lock.read_text(encoding="utf-8") == "x"
    assert sorted(p.relative_to(tmp_path).as_posix()
                  for p in tmp_path.rglob("*")) == before
    assert "verify-stage5" in capsys.readouterr().out


def test_verify_freeze_runs_when_not_locked(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stage4, "verify_freeze", _recorder(1, calls))
    assert cli.main(["--root", str(tmp_path), "verify-freeze"]) == 1
    assert calls == [tmp_path]


def test_stage5_success_and_refusals(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(stage5, "run_official", _recorder(0, calls))
    assert cli.main(["--root", str(tmp_path), "stage5"]) == 0
    assert calls == [tmp_path]
    for error in (stage5.IntegrityError("portao reprovado"),
                  stage5.AlreadyEvaluatedError("ja avaliado")):
        def refuse(root, error=error):
            raise error

        monkeypatch.setattr(stage5, "run_official", refuse)
        assert cli.main(["--root", str(tmp_path), "stage5"]) == 4
        assert str(error) in capsys.readouterr().out


# ----------------------------------------------------------- aquisicao

def test_acquire_success_prints_manifest(tmp_path, monkeypatch, capsys):
    manifest = {"artifact": {"local_path": "data/raw/a.arff",
                             "size_bytes": 12, "sha256": "ab" * 32},
                "checksum_verification": {"match": True}}
    seen = []

    def fake(root, outputs=None):
        seen.append(Path(root))
        return manifest

    monkeypatch.setattr(acquire_module, "acquire", fake)
    assert cli.main(["--root", str(tmp_path), "acquire"]) == 0
    assert seen == [tmp_path]
    out = capsys.readouterr().out
    assert "data/raw/a.arff" in out and "bytes: 12" in out
    assert "ab" * 32 in out and "md5 confere: True" in out


@pytest.mark.parametrize("error", [
    acquire_module.ChecksumMismatchError("md5 divergente"),
    FileExistsError("bruto ja existe")])
def test_acquire_refusals_exit_3(error, tmp_path, monkeypatch, capsys):
    def refuse(root, outputs=None):
        raise error

    monkeypatch.setattr(acquire_module, "acquire", refuse)
    assert cli.main(["--root", str(tmp_path), "acquire"]) == 3
    assert str(error) in capsys.readouterr().out


# ----------------------------------------------------- verify-stage5

def test_verify_stage5_missing_files_exit_1_without_traceback(
        tmp_path, capsys):
    assert cli.main(["--root", str(tmp_path), "verify-stage5"]) == 1
    assert "verificacao impossivel" in capsys.readouterr().out


@pytest.mark.parametrize("official, all_ok, expected", [
    (0, True, 0), (0, False, 1), (1, True, 1), (1, False, 1)])
def test_verify_stage5_combines_both_checks(
        official, all_ok, expected, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stage5, "verify_official", lambda root: official)
    monkeypatch.setattr(
        artifacts, "verify_stage5",
        lambda root: {"all_ok": all_ok, "checks": {"x": all_ok}})
    assert cli.main(["--root", str(tmp_path), "verify-stage5"]) == expected
    assert "conferencia completa" in capsys.readouterr().out


@pytest.mark.parametrize("error", [
    KeyError("chave"), ValueError("valor"),
    FileNotFoundError("arquivo")])
def test_verify_stage5_full_maps_errors_to_1(
        error, tmp_path, monkeypatch, capsys):
    def fail(root):
        raise error

    monkeypatch.setattr(stage5, "verify_official", fail)
    assert cli.verify_stage5_full(tmp_path) == 1
    assert type(error).__name__ in capsys.readouterr().out


# ---------------------------------------------------------- __main__

def test_module_entry_point_runs_main_in_process(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["freq_sinistros", "--help"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("freq_sinistros.__main__", run_name="__main__")
    assert info.value.code == 0
    assert "verify-stage5" in capsys.readouterr().out


def test_module_entry_point_propagates_exit_code(
        tmp_path, monkeypatch):
    monkeypatch.setattr(audit_module, "run_audit", lambda root: 2)
    monkeypatch.setattr(
        sys, "argv", ["freq_sinistros", "--root", str(tmp_path), "audit"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("freq_sinistros.__main__", run_name="__main__")
    assert info.value.code == 2


def test_python_dash_m_help_subprocess(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONNOUSERSITE": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "freq_sinistros", "--help"], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    assert "check-stage5" in result.stdout
