import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_Seoul_Bike_Operational_Forecast_Demo.ipynb"


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source(cell):
    return "".join(cell.get("source", []))


def test_notebook_exists_and_uses_project_kernel():
    notebook = _notebook()
    kernelspec = notebook["metadata"]["kernelspec"]
    assert kernelspec["name"] == "bike-sharing"
    assert kernelspec["display_name"] == "Python (Bike-Sharing)"


def test_notebook_has_expected_pt_br_sections():
    markdown = "\n".join(
        _source(cell) for cell in _notebook()["cells"] if cell["cell_type"] == "markdown"
    )
    expected = [
        "# Demonstração operacional de uma previsão horária",
        "## 1. Objetivo e fronteiras da demonstração",
        "## 2. Contrato do replay operacional",
        "## 3. Seleção temporal reproduzível",
        "## 4. Perfil conhecido no instante da previsão",
        "## 5. Previsão pontual e intervalo preditivo",
        "## 6. Decisão antes da revelação da demanda",
        "## 7. Revelação e auditoria",
        "## 8. Leitura visual da decisão",
        "### Insights",
        "## 9. Síntese",
        "## 10. Proveniência e reprodutibilidade",
        "### Referência metodológica",
    ]
    for heading in expected:
        assert heading in markdown


def test_notebook_is_thin_and_calls_only_modular_workflow():
    code = "\n".join(_source(cell) for cell in _notebook()["cells"] if cell["cell_type"] == "code")
    non_empty_lines = [line for line in code.splitlines() if line.strip()]
    assert len(non_empty_lines) <= 55
    assert "OperationalDemoConfig" in code
    assert "build_operational_replay" in code
    assert "operational_demo_reports" in code
    for forbidden in (
        "pd.read_csv",
        "pickle",
        "joblib",
        ".fit(",
        ".predict(",
        "holdout_predictions",
        "y_true",
    ):
        assert forbidden not in code


def test_configuration_declares_reproducible_operational_contract():
    code = "\n".join(_source(cell) for cell in _notebook()["cells"] if cell["cell_type"] == "code")
    assert "candidate_id='U4b_g0p01'" in code
    assert "coverage=0.90" in code
    assert "random_state=2026" in code
    assert "planned_capacity=4000.0" in code
    assert "available_bikes" not in code


def test_information_order_hides_actual_until_audit():
    code = "\n".join(_source(cell) for cell in _notebook()["cells"] if cell["cell_type"] == "code")
    profile_position = code.index("profile_report")
    forecast_position = code.index("forecast_report")
    decision_position = code.index("decision_report")
    audit_position = code.index("audit_report")
    plot_position = code.index("plot_operational_forecast")
    assert profile_position < forecast_position < decision_position < audit_position
    assert audit_position < plot_position


def test_only_insight_cell_uses_blockquotes():
    for cell in _notebook()["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = _source(cell)
        if any(line.startswith(">") for line in source.splitlines()):
            assert source.startswith("### Insights")


def test_narrative_states_scope_and_no_holdout_access():
    markdown = "\n".join(
        _source(cell) for cell in _notebook()["cells"] if cell["cell_type"] == "markdown"
    ).lower()
    assert "holdout" in markdown
    assert "não utilizado" not in markdown or "não" in markdown
    assert "capacidade operacional" in markdown
    assert "não contém estoques" in markdown
    assert "não é afirmado que exista exatamente 90% de probabilidade" in markdown
    assert "168 horas" in markdown


def test_methodological_citations_appear_in_text_and_references():
    markdown = "\n".join(
        _source(cell) for cell in _notebook()["cells"] if cell["cell_type"] == "markdown"
    )
    assert "Gibbs e Candès (2021)" in markdown
    assert "Gneiting e Raftery (2007)" in markdown
    assert "Gibbs, I.; Candès, E. J. (2021)" in markdown
    assert "Gneiting, T.; Raftery, A. E. (2007)" in markdown


def test_notebook_has_no_mojibake():
    raw = NOTEBOOK_PATH.read_text(encoding="utf-8")
    for token in ("�", "Ã§", "Ã£", "Ã©", "Ã¡", "Ã³", "Ãª", "Â", "â€"):
        assert token not in raw


def test_executed_notebook_is_sequential_and_error_free():
    code_cells = [cell for cell in _notebook()["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code_cells]
    assert counts == list(range(1, len(code_cells) + 1))
    for cell in code_cells:
        assert all(output.get("output_type") != "error" for output in cell["outputs"])
