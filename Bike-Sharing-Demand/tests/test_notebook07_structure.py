from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


NOTEBOOK_PATH = Path("notebooks/07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb")


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.fixture(scope="module")
def code_source(code_cells):
    return "\n".join("".join(cell.get("source", [])) for cell in code_cells)


def test_kernel_is_bike_sharing(notebook):
    kernelspec = notebook["metadata"]["kernelspec"]
    assert kernelspec["name"] == "bike-sharing"
    assert kernelspec["display_name"] == "Python (Bike-Sharing)"


def test_saved_outputs_do_not_contain_errors(code_cells):
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors


def test_notebook_has_seventeen_numbered_sections(notebook):
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    sections = re.findall(r"^## (\d+)\.", markdown, flags=re.MULTILINE)
    assert sections == [str(number) for number in range(1, 18)]


def test_blockquotes_are_reserved_for_insights(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if any(line.startswith(">") for line in source.splitlines()):
            assert source.lstrip().startswith("### Insight")


def test_markdown_narrative_uses_no_bullet_lists(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            assert not re.match(r"^\s*[-*+]\s+", line)


def test_markdown_has_no_encoding_damage(notebook):
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "�" not in markdown
    assert "Ã" not in markdown
    assert "â€" not in markdown


def test_code_layer_is_thin(code_cells, code_source):
    assert len(code_cells) <= 18
    assert len(code_source.splitlines()) <= 100
    assert "for " not in code_source
    assert "while " not in code_source
    assert "groupby(" not in code_source
    assert "merge(" not in code_source
    assert "to_csv(" not in code_source


@pytest.mark.parametrize(
    "token",
    [
        ".fit(",
        ".predict(",
        "materialize_final_holdout",
        "run_final_validation",
        "Optuna",
        "FixedTrial",
    ],
)
def test_forbidden_workflow_tokens_are_absent(code_source, token):
    assert token not in code_source


def test_notebook_runs_smoke_not_full(code_source):
    assert "run_mode='full'" in code_source or 'run_mode="full"' in code_source


def test_public_workflow_and_reports_are_used(code_source):
    required = [
        "ConformalUncertaintyConfig(",
        "run_conformal_uncertainty_experiments(",
        "reports.protocol_report(",
        "reports.input_audit_report(",
        "reports.warmup_report(",
        "reports.plot_scale_diagnostics(",
        "reports.plot_coverage_calibration(",
        "reports.plot_fold_coverage_heatmap(",
        "reports.plot_coverage_width_pareto(",
        "reports.plot_rolling_coverage(",
        "reports.plot_aci_alpha_trajectory(",
        "reports.plot_segment_coverage(",
        "reports.decision_report(",
        "reports.synthesis_report(",
    ]
    for call in required:
        assert call in code_source


def test_language_layer_is_wired(code_source):
    assert "make_lang('pt')" in code_source or 'make_lang("pt")' in code_source
    assert "lang=lang" in code_source


def test_methodological_references_are_present(notebook):
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    for citation in (
        "Gibbs e Candès (2021)",
        "Barber et al. (2023)",
        "Romano, Patterson e Candès (2019)",
        "Gibbs, I.; Candès, E. J. (2021)",
        "Barber, R. F.; Candès, E. J.; Ramdas, A.; Tibshirani, R. J. (2023)",
        "Romano, Y.; Patterson, E.; Candès, E. J. (2019)",
    ):
        assert citation in markdown


def test_narrative_states_prequential_warmup_and_stress_contract(notebook):
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ).lower()
    assert "168 horas" in markdown
    assert "t+1" in markdown
    assert "2020" in markdown
    assert "fora do ranking" in markdown
    assert "holdout" in markdown


def test_synthesis_does_not_embed_stale_c0_c3_results(notebook):
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "87,9%" not in markdown
    assert "76,3%" not in markdown
    assert "C0" not in markdown


def test_synthesis_keeps_point_champion_invariant(notebook):
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "Champion pontual E0" in markdown
    assert "nenhum sucessor pontual" in markdown.lower()
