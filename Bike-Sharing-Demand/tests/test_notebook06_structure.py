"""Structural guarantees for notebook 06 residual and uncertainty experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb"
)

FORBIDDEN_CODE_TOKENS = {
    "mlflow": "chamada MLflow direta",
    "optuna": "manipulacao direta de Optuna",
    "FixedTrial": "replay manual do manifesto no notebook",
    ".fit(": "ajuste manual no notebook",
    ".predict(": "predicao manual no notebook",
    "groupby": "agregacao manual no notebook",
    "mean_absolute_error": "metrica manual no notebook",
    "r2_score": "metrica manual no notebook",
    "pd.DataFrame": "montagem manual de tabela no notebook",
    "holdout_predictions": "acesso a predicoes do holdout",
    "materialize_final_holdout": "materializacao do holdout final",
    "run_final_validation": "reexecucao da validacao final",
    "FinalValidationResults": "uso de resultados confirmatorios para modelagem",
}


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK_PATH.exists(), f"notebook nao encontrado em {NOTEBOOK_PATH}"
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook) -> list:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.fixture(scope="module")
def code_source(code_cells) -> str:
    return "\n".join("".join(cell["source"]) for cell in code_cells)


def test_kernel_is_bike_sharing(notebook):
    kernelspec = notebook["metadata"]["kernelspec"]
    assert kernelspec["name"] == "bike-sharing"
    assert kernelspec["display_name"] == "Python (Bike-Sharing)"
    assert kernelspec["language"] == "python"


def test_saved_outputs_do_not_contain_errors(code_cells):
    for index, cell in enumerate(code_cells):
        errors = [
            output for output in cell.get("outputs", []) if output.get("output_type") == "error"
        ]
        assert errors == [], f"celula de codigo {index} contem erro salvo"


def test_notebook_has_fourteen_numbered_sections(notebook):
    headings = [
        "".join(cell["source"]).splitlines()[0]
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("## ")
    ]
    numbers = [int(re.match(r"## (\d+)\.", heading).group(1)) for heading in headings]
    assert numbers == list(range(1, 15))


def test_blockquotes_are_reserved_for_insights(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith(">"):
            assert "### Insight" in source


def test_markdown_narrative_uses_no_bullet_lists(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell["source"]).splitlines():
            assert not line.lstrip().startswith(("- ", "* ", "+ "))


def test_markdown_has_no_encoding_replacement_marks(notebook):
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "\ufffd" not in markdown
    assert "?" not in markdown


def test_code_layer_is_thin(code_cells, code_source):
    lines = [
        line for cell in code_cells for line in "".join(cell["source"]).splitlines() if line.strip()
    ]
    assert len(lines) <= 90
    assert not re.search(r"^\s*for\s+\w+", code_source, flags=re.MULTILINE)
    assert not re.search(r"^\s*while\s+", code_source, flags=re.MULTILINE)
    assert not re.search(r"^\s*def\s+", code_source, flags=re.MULTILINE)
    assert not re.search(r"^\s*class\s+", code_source, flags=re.MULTILINE)


@pytest.mark.parametrize("token,reason", sorted(FORBIDDEN_CODE_TOKENS.items()))
def test_forbidden_workflow_tokens_are_absent(code_source, token, reason):
    assert token not in code_source, f"notebook 06 contem {reason}: {token}"


def test_public_workflow_and_report_functions_are_used(code_source):
    for symbol in (
        "UncertaintyExperimentConfig(",
        "prepare_uncertainty_development(",
        "run_uncertainty_experiments(",
        "reports.protocol_report(",
        "reports.development_report(",
        "reports.fold_audit_report(",
        "reports.experiment_spec_report(",
        "reports.point_metrics_report(",
        "reports.plot_point_metrics_by_fold(",
        "reports.probabilistic_metrics_report(",
        "reports.probabilistic_fold_metrics_report(",
        "reports.plot_probabilistic_metrics_by_fold(",
        "reports.residual_dependence_report(",
        "reports.scale_diagnostics_report(",
        "reports.plot_residual_diagnostics_heatmap(",
        "reports.plot_scale_diagnostics(",
        "reports.plot_representative_interval_windows(",
        "reports.plot_segment_coverage(",
        "reports.ablation_report(",
        "reports.successor_message(",
    ):
        assert symbol in code_source


def test_language_layer_is_wired(code_source):
    assert "make_lang('pt')" in code_source
    assert code_source.count("lang=lang") >= 15


def test_methodological_references_are_present(notebook):
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    for token in (
        "Prokhorenkova et al. (2018)",
        "Malinin, Prokhorenkova e Ustimenko (2021)",
        "Duan et al. (2020)",
        "Engle (1982)",
        "Gneiting, Balabdaoui e Raftery (2007)",
        "Gneiting e Raftery (2007)",
        "Bergmeir, Hyndman e Koo (2018)",
        "Prokhorenkova, L.; Gusev, G.; Vorobev, A.; Dorogush, A. V.; Gulin, A.",
        "### Referência metodológica",
    ):
        assert token in markdown


def test_synthesis_declares_no_point_successor(notebook):
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "Nenhum sucessor pontual" in markdown
