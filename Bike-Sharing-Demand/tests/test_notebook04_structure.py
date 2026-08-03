"""Structural guarantees for notebooks/04_Seoul_Bike_Model_Selection.ipynb.

Notebook 04 is required to be a presentation layer: it declares a
configuration, calls public workflow functions and displays what they return.
These tests assert the absence of workflow logic rather than counting cells —
a notebook can be short and still carry an estimator loop, and it is the loop
that makes the analysis untestable, not the length.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOK_PATH = (
    Path(__file__).resolve().parent.parent / "notebooks" / "04_Seoul_Bike_Model_Selection.ipynb"
)

# Everything that belongs in src/, not in a cell.
FORBIDDEN_SUBSTRINGS = {
    "mlflow": "chamadas MLflow detalhadas",
    "log_temporal_model_run": "logging manual de runs",
    "freeze_candidates": "persistência manual de candidatos",
    "select_champion_and_challengers": "seleção manual de champion",
    "TemporalRegressionOptimizer": "instanciação manual do otimizador",
    "optuna": "manipulação direta de estudos Optuna",
    "FixedTrial": "construção manual de trials",
    "study.best_": "leitura manual do resultado de um estudo",
    "_build_pipeline_for_trial": "método privado do otimizador",
    "estimator_html_repr": "renderização manual do diagrama do pipeline",
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
    def test_has_no_manual_estimator_loop(self, code_source):
        assert not re.search(r"^\s*for\s+\w+", code_source, flags=re.MULTILINE)
        assert not re.search(r"^\s*while\s+", code_source, flags=re.MULTILINE)

    def test_defines_no_functions_or_classes(self, code_source):
        assert not re.search(r"^\s*def\s+", code_source, flags=re.MULTILINE)
        assert not re.search(r"^\s*class\s+", code_source, flags=re.MULTILINE)

    def test_builds_no_metric_dictionaries_or_comparison_tables(self, code_source):
        assert "cv_mae_mean" not in code_source
        assert "pd.DataFrame" not in code_source
        assert "sort_values" not in code_source

    def test_calls_no_private_members(self, code_source):
        private_calls = re.findall(r"\.\_[A-Za-z]\w*", code_source)
        assert private_calls == []

    @pytest.mark.parametrize("token,reason", sorted(FORBIDDEN_SUBSTRINGS.items()))
    def test_forbidden_token_is_absent(self, code_source, token, reason):
        assert token not in code_source, f"notebook 04 ainda contém {reason} ('{token}')"

    def test_does_no_plotting_beyond_shared_defaults(self, code_source):
        """Only the project-wide rcParams tweak is allowed; every chart comes
        from src.model_selection_reports."""
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

    def test_declares_the_run_configuration(self, code_source):
        assert "ModelSelectionConfig(" in code_source
        assert "run_mode='full'" in code_source or 'run_mode="full"' in code_source

    def test_calls_the_public_workflow_functions(self, code_source):
        for symbol in (
            "prepare_development_data(",
            "sample_dynamic_pipeline(",
            "run_model_selection(",
        ):
            assert symbol in code_source

    def test_uses_the_reports_layer_for_every_display(self, code_source):
        for report in (
            "run_configuration_report",
            "holdout_seal_report",
            "fold_audit_report",
            "plot_fold_audit",
            "feature_space_report",
            "search_space_report",
            "pipeline_spec_report",
            "dynamic_pipeline_diagram",
            "comparison_report",
            "plot_comparison",
            "fold_metrics_report",
            "condition_metrics_report",
            "selection_report",
            "handoff_report",
        ):
            assert f"reports.{report}(" in code_source

    def test_language_layer_is_wired(self, code_source):
        assert "make_lang('pt')" in code_source
        assert code_source.count("lang=lang") >= 10


class TestNarrative:
    def test_has_the_fifteen_numbered_sections(self, notebook):
        headings = [
            "".join(cell["source"]).splitlines()[0]
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("## ")
        ]
        numbers = [
            int(re.match(r"## (\d+)\.", h).group(1)) for h in headings if re.match(r"## \d+\.", h)
        ]
        assert numbers == list(range(1, 16))

    def test_blockquotes_are_reserved_for_insights(self, notebook):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            source = "".join(cell["source"])
            if source.lstrip().startswith(">"):
                assert "### Insight" in source

    def test_narrative_explains_the_dynamic_preprocessing(self, notebook):
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        assert "não é fixado antecipadamente" in markdown
        assert "modeler_name" in markdown
        assert "família do estimator" in markdown
        assert "smoke" in markdown

    def test_narrative_uses_no_bullet_lists(self, notebook):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            for line in "".join(cell["source"]).splitlines():
                assert not line.lstrip().startswith(("- ", "* ", "+ "))


class TestKernelAndEnvironmentAudit:
    """The notebook must name the project's own kernel, not the generic one.

    ``display_name`` is cosmetic — Jupyter and nbconvert resolve the kernel by
    ``name``, so a notebook labelled "Bike-Sharing" while still declaring
    ``name: python3`` executes under whatever ``python3`` happens to point at.
    That is precisely how the outputs were produced by the wrong environment.
    """

    def test_the_kernel_is_resolved_by_name_not_only_by_label(self, notebook):
        kernelspec = notebook["metadata"]["kernelspec"]
        assert kernelspec["name"] == "bike-sharing"
        assert kernelspec["display_name"] == "Python (Bike-Sharing)"
        assert kernelspec["language"] == "python"

    def test_an_environment_report_is_displayed_for_auditing(self, code_source):
        assert "reports.environment_report(" in code_source

    def test_the_saved_outputs_name_the_project_interpreter(self, code_cells):
        outputs = [cell.get("outputs", []) for cell in code_cells]
        if all(not cell_outputs for cell_outputs in outputs):
            pytest.skip("notebook ainda não executado nesta cópia")
        rendered = json.dumps(outputs, ensure_ascii=False)
        assert "Bike-Sharing" in rendered
        assert "Churn-ML" not in rendered

    def test_no_output_advertises_a_foreign_distribution(self, code_cells):
        rendered = json.dumps([cell.get("outputs", []) for cell in code_cells], ensure_ascii=False)
        assert "customer-segmentation-nba" not in rendered


class TestExecutionState:
    def test_execution_counts_are_sequential(self, code_cells):
        counts = [cell.get("execution_count") for cell in code_cells]
        if all(count is None for count in counts):
            pytest.skip("notebook ainda não executado nesta cópia")
        assert counts == list(range(1, len(code_cells) + 1))

    def test_no_cell_ended_in_an_error(self, code_cells):
        for index, cell in enumerate(code_cells):
            outputs = cell.get("outputs", [])
            errors = [o for o in outputs if o.get("output_type") == "error"]
            assert not errors, f"célula de código {index} terminou com erro: {errors}"

    def test_no_holdout_metric_appears_in_any_output(self, code_cells):
        rendered = json.dumps([cell.get("outputs", []) for cell in code_cells], ensure_ascii=False)
        assert "holdout_mae" not in rendered
        assert "holdout_rmse" not in rendered
        assert "holdout_r2" not in rendered
