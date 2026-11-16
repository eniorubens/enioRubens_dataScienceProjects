"""Tests for scripts/export_report.py with synthetic notebooks (the real report is never exported)."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_report.py"
_spec = importlib.util.spec_from_file_location("export_report", SCRIPT)
export_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_report)

NEEDS_EXPORT_STACK = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("nbconvert", "nbformat", "plotly")),
    reason="nbconvert, nbformat or plotly is not installed",
)


@pytest.fixture
def tmp_path():
    """Local override: the default pytest temp root is not writable in this Windows setup."""
    with tempfile.TemporaryDirectory(prefix="export_report_") as directory:
        yield Path(directory)


def plotly_output() -> dict:
    spec = {
        "data": [{"type": "bar", "x": ["a", "b"], "y": [1, 2]}],
        "layout": {"title": {"text": "synthetic chart"}},
    }
    return {"output_type": "display_data", "metadata": {}, "data": {export_report.PLOTLY_MIME: spec}}


def notebook(executed: bool, with_figure: bool = False) -> dict:
    outputs = [{"output_type": "stream", "name": "stdout", "text": ["hello\n"]}]
    if with_figure:
        outputs.append(plotly_output())
    code = {
        "cell_type": "code",
        "id": "aaaaaaaa",
        "metadata": {},
        "source": ["print('hello')"],
        "execution_count": 1 if executed else None,
        "outputs": outputs if executed else [],
    }
    text = {"cell_type": "markdown", "id": "bbbbbbbb", "metadata": {}, "source": ["# Title"]}
    return {"cells": [text, code], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def write_notebooks(root: Path, executed: bool = True, with_figure: bool = False) -> None:
    for relative, _ in export_report.TARGETS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook(executed, with_figure)), encoding="utf-8")


def test_execution_problems_flags_unexecuted_and_missing_notebooks(tmp_path):
    path = tmp_path / "n.ipynb"
    path.write_text(json.dumps(notebook(executed=False)), encoding="utf-8")
    problems = export_report.execution_problems(path)
    assert any("not executed" in problem for problem in problems)
    assert any("no outputs" in problem for problem in problems)

    path.write_text(json.dumps(notebook(executed=True)), encoding="utf-8")
    assert export_report.execution_problems(path) == []
    assert "not found" in export_report.execution_problems(tmp_path / "missing.ipynb")[0]


def test_export_refuses_notebooks_without_outputs_and_converts_nothing(tmp_path, capsys):
    write_notebooks(tmp_path, executed=False)
    calls = []
    status = export_report.export_report(tmp_path, converter=lambda *a: calls.append(a))

    assert status == 1
    assert calls == []
    assert not (tmp_path / "reports").exists()
    error = capsys.readouterr().err
    assert "cannot be exported" in error and "no outputs" in error


def test_offline_export_removes_external_ssl_keylog(monkeypatch):
    monkeypatch.setenv("SSLKEYLOGFILE", r"\\.\aswMonFltProxy\invalid")

    export_report._disable_external_ssl_keylog()

    assert "SSLKEYLOGFILE" not in os.environ


def test_export_converts_both_languages_and_creates_reports_dir(tmp_path):
    write_notebooks(tmp_path)
    targets = []

    def fake_converter(notebook_path, output_path):
        targets.append((notebook_path.name, output_path.name))
        output_path.write_text("<html></html>", encoding="utf-8")

    assert export_report.export_report(tmp_path, converter=fake_converter) == 0
    assert (tmp_path / "reports").is_dir()
    assert targets == [
        ("11_relatorio_final.ipynb", "relatorio_final_pt.html"),
        ("11_final_report.ipynb", "final_report_en.html"),
    ]


def test_export_reports_conversion_failure(tmp_path, capsys):
    write_notebooks(tmp_path)

    def failing_converter(notebook_path, output_path):
        raise OSError("[WinError 623] simulated")

    assert export_report.export_report(tmp_path, converter=failing_converter) == 2
    assert "HTML conversion failed" in capsys.readouterr().err


def test_main_explains_how_to_activate_the_environment_when_nbconvert_is_missing(tmp_path, monkeypatch, capsys):
    write_notebooks(tmp_path)
    monkeypatch.setattr(export_report, "missing_modules", lambda: ["nbconvert"])
    monkeypatch.setattr(export_report, "export_report", lambda *a, **k: pytest.fail("must not export"))

    assert export_report.main(["--root", str(tmp_path)]) == 2
    error = capsys.readouterr().err
    assert error.count("conda activate ecom-delivery-sla") == 2  # Portuguese and English messages
    assert not (tmp_path / "reports").exists()


def test_missing_modules_reports_what_cannot_be_imported(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "nbconvert" else object())
    assert export_report.missing_modules() == ["nbconvert"]


def test_export_does_not_use_a_subprocess():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "-m\", \"jupyter" not in source


@NEEDS_EXPORT_STACK
def test_real_conversion_hides_code_and_keeps_text(tmp_path):
    write_notebooks(tmp_path)
    assert export_report.export_report(tmp_path) == 0
    html = (tmp_path / "reports" / "final_report_en.html").read_text(encoding="utf-8")
    assert "hello" in html
    assert "print(&#39;hello&#39;)" not in html and "print('hello')" not in html
    assert "Plotly.newPlot" not in html  # no figure, no plotly.js


@NEEDS_EXPORT_STACK
def test_real_conversion_renders_plotly_figures_offline(tmp_path):
    write_notebooks(tmp_path, with_figure=True)
    assert export_report.export_report(tmp_path) == 0
    html = (tmp_path / "reports" / "relatorio_final_pt.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html and "report-figure-1" in html
    assert "synthetic chart" in html
    assert "PlotlyGeoAssets" in html
    assert "cdn.plot.ly" not in html.split("</head>")[0].split("<head>")[1][:200]
    head = html.split("</head>")[0]
    assert head.index("window.PlotlyGeoAssets") > 0
    if "require" in head:  # plotly.js must come before require.js to define the global Plotly
        assert head.index("plotly.js v") < head.index("require")
    assert "print('hello')" not in html
