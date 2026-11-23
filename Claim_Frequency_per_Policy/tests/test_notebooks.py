"""Estrutura e paridade dos notebooks versionados (sem executa-los)."""

from pathlib import Path

import pytest

from freq_sinistros import notebook_checks

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["01_eda_treino.ipynb", "02_comparacao_modelos.ipynb",
            "03_diagnostico_zeros.ipynb", "04_avaliacao_final.ipynb"]


def test_repository_notebooks_have_parity():
    folder = ROOT / "notebooks" / "pt_br"
    if not folder.exists():
        pytest.skip("notebooks ainda nao gerados")
    result = notebook_checks.validate_directory(ROOT)
    assert sorted(result) == EXPECTED
    assert all(not problems for problems in result.values()), result


@pytest.mark.parametrize("name", EXPECTED)
@pytest.mark.parametrize(
    ("language", "question"),
    [("pt_br", "**Pergunta.**"), ("en_us", "**Question.**")],
)
def test_notebooks_follow_question_execution_insights(
    name, language, question
):
    path = ROOT / "notebooks" / language / name
    if not path.exists():
        pytest.skip("notebooks ainda nao gerados")
    nb = notebook_checks.load_notebook(path)
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds[0] == "markdown" and kinds[-1] == "markdown"
    text = "\n".join(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "markdown")
    assert text.count(question) >= 5
    assert text.splitlines().count("### Insights") >= 5
    assert "**Interpretação.**" not in text
    assert "**Interpretation.**" not in text
