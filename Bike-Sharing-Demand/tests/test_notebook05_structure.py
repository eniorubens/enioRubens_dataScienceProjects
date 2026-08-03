"""Structural guarantees for notebooks/05_Seoul_Bike_Final_Validation.ipynb.

Notebook 05 must be a presentation layer: it declares a configuration, calls a
few public functions from :mod:`src.final_validation` and displays what the
reports layer returns. These tests assert the absence of workflow logic (loops,
functions, direct MLflow/SHAP/pickle/JSON handling, manual plotting), the
presence of the thirteen numbered sections and the methodological SHAP
reference, the correct kernel, and that the delivered notebook carries no
outputs. The holdout is never opened here — the notebook is not executed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOK_PATH = (
    Path(__file__).resolve().parent.parent / "notebooks" / "05_Seoul_Bike_Final_Validation.ipynb"
)

# Everything that belongs in src/, not in a notebook cell.
FORBIDDEN_SUBSTRINGS = {
    "mlflow": "chamadas MLflow diretas",
    "optuna": "manipulação direta de estudos Optuna",
    "TreeExplainer": "chamada direta ao SHAP",
    "shap.": "chamada direta ao SHAP",
    "pickle": "leitura manual de pickle",
    "json.load": "leitura manual de JSON",
    "read_data": "leitura manual do dataset bruto",
    "split_dev_holdout": "materialização manual do holdout",
    "materialize_final_holdout": "materialização manual do holdout",
    "load_manifest": "leitura manual do manifesto",
    "seal_holdout": "recorte manual do holdout",
    "estimator_html_repr": "renderização manual do diagrama do pipeline",
    ".predict(": "predição manual dos candidatos",
    ".fit(": "ajuste de modelo no notebook",
}


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK_PATH.exists(), f"notebook não encontrado em {NOTEBOOK_PATH}"
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook) -> list:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.fixture(scope="module")
def code_source(code_cells) -> str:
    return "\n".join("".join(cell["source"]) for cell in code_cells)


class TestNoWorkflowLogic:
    def test_has_no_manual_loop(self, code_source):
        assert not re.search(r"^\s*for\s+\w+", code_source, flags=re.MULTILINE)
        assert not re.search(r"^\s*while\s+", code_source, flags=re.MULTILINE)

    def test_defines_no_functions_or_classes(self, code_source):
        assert not re.search(r"^\s*def\s+", code_source, flags=re.MULTILINE)
        assert not re.search(r"^\s*class\s+", code_source, flags=re.MULTILINE)

    def test_builds_no_dataframes_or_manual_metrics(self, code_source):
        assert "pd.DataFrame" not in code_source
        assert "sort_values" not in code_source
        assert "holdout_mae" not in code_source

    def test_calls_no_private_members(self, code_source):
        assert re.findall(r"\.\_[A-Za-z]\w*", code_source) == []

    @pytest.mark.parametrize("token,reason", sorted(FORBIDDEN_SUBSTRINGS.items()))
    def test_forbidden_token_is_absent(self, code_source, token, reason):
        assert token not in code_source, f"notebook 05 ainda contém {reason} ('{token}')"

    def test_does_no_manual_plotting(self, code_source):
        assert "plt.subplots" not in code_source
        assert "ax." not in code_source
        assert "plt.show" not in code_source
        assert "set_graph_parameters()" in code_source


class TestOrchestrationSurface:
    def test_python_line_count_stays_declarative(self, code_cells):
        lines = [
            line
            for cell in code_cells
            for line in "".join(cell["source"]).splitlines()
            if line.strip()
        ]
        assert len(lines) <= 90, f"{len(lines)} linhas de código — a camada deixou de ser fina"

    def test_declares_the_configuration(self, code_source):
        assert "FinalValidationConfig(" in code_source

    def test_calls_the_public_workflow_functions(self, code_source):
        for symbol in (
            "prepare_final_validation(",
            "run_final_validation(",
            "run_shap_validation(",
        ):
            assert symbol in code_source

    def test_uses_the_reports_layer_for_every_display(self, code_source):
        for report in (
            "protocol_report",
            "provenance_report",
            "candidates_report",
            "holdout_seal_report",
            "comparison_report",
            "metrics_report",
            "plot_comparison",
            "confirmation_report",
            "confirmation_message",
            "residual_diagnostics_report",
            "heteroscedasticity_report",
            "plot_temporal_residuals",
            "plot_residual_structure",
            "residual_triage_report",
            "residual_profile_report",
            "plot_residual_triage",
            "residual_transformation_report",
            "plot_residual_transformation_acf",
            "residual_handoff_message",
            "condition_metrics_report",
            "plot_condition_metrics",
            "shap_methodology_report",
            "shap_additivity_report",
            "shap_grouped_report",
            "plot_shap_summary",
            "plot_shap_local",
            "artifacts_report",
            "synthesis_report",
            "handoff_message",
        ):
            assert f"reports.{report}(" in code_source

    def test_language_layer_is_wired(self, code_source):
        assert "make_lang('pt')" in code_source
        assert code_source.count("lang=lang") >= 15


class TestNarrative:
    def test_has_the_thirteen_numbered_sections(self, notebook):
        headings = [
            "".join(cell["source"]).splitlines()[0]
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("## ")
        ]
        numbers = [
            int(re.match(r"## (\d+)\.", h).group(1)) for h in headings if re.match(r"## \d+\.", h)
        ]
        assert numbers == list(range(1, 14))

    def test_blockquotes_are_reserved_for_insights(self, notebook):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            source = "".join(cell["source"])
            if source.lstrip().startswith(">"):
                assert "### Insight" in source

    def test_narrative_uses_no_bullet_lists(self, notebook):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            for line in "".join(cell["source"]).splitlines():
                assert not line.lstrip().startswith(("- ", "* ", "+ "))

    def test_documents_the_bias_and_residual_conventions(self, notebook):
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        assert "bias" in markdown.lower()
        assert "resíduo" in markdown.lower()
        assert "subestima" in markdown.lower()
        assert "superestima" in markdown.lower()

    def test_documents_formal_heteroscedasticity_diagnostics(self, notebook):
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        assert "Breusch-Pagan" in markdown
        assert "White" in markdown
        assert "Goldfeld-Quandt" in markdown
        assert "Engle ARCH" in markdown
        assert "Holm" in markdown
        assert "nunca uma prova de homocedasticidade" in markdown
        assert "autocorrela" in markdown.lower()

    def test_documents_post_holdout_residual_triage_limitations(self, notebook):
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        assert "### 7.1 Triagem da média residual e da escala do erro" in markdown
        assert "### 7.2 Persistência diária e semanal após padronização" in markdown
        assert "melhor previsão pontual" in markdown
        assert "intervalos" in markdown.lower() and "IID" in markdown
        assert "ARCH/GARCH" in markdown
        assert "sem leakage" in markdown
        assert "nova janela independente" in markdown

    def test_states_the_preregistered_rule_and_no_reopen(self, notebook):
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        assert "pré-registrad" in markdown
        assert "não é reaberta" in markdown or "não reabre" in markdown

    def test_has_the_methodological_shap_reference(self, notebook):
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        assert "### Referência metodológica" in markdown
        assert "Lundberg" in markdown
        assert "2017" in markdown and "2020" in markdown
        assert "Breusch" in markdown
        assert "White" in markdown
        assert "Goldfeld" in markdown
        assert "Engle" in markdown


class TestKernelAndExecutionState:
    def test_the_kernel_is_resolved_by_name(self, notebook):
        kernelspec = notebook["metadata"]["kernelspec"]
        assert kernelspec["name"] == "bike-sharing"
        assert kernelspec["display_name"] == "Python (Bike-Sharing)"
        assert kernelspec["language"] == "python"

    def test_the_published_notebook_is_executed_in_order(self, code_cells):
        counts = [cell.get("execution_count") for cell in code_cells]
        assert counts == list(range(1, len(code_cells) + 1))

    def test_no_cell_ended_in_an_error(self, code_cells):
        for index, cell in enumerate(code_cells):
            errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
            assert not errors, f"célula de código {index} terminou com erro: {errors}"
