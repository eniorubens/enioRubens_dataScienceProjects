"""Tests that the notebooks use the package and keep the published outputs."""
import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(
    (Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")
)
# Chart and narrative helpers that still live in the notebooks. Everything
# that computes a result has to come from the package.
PRESENTATION_HELPERS = {
    "set_corporate_theme",
    "add_titles",
    "plot_z_test_distribution",
    "get_transformed_feature_names",
    "plot_retention_diagnostics",
    "plot_model_signal_summary",
    "plot_retention_experiment_comparison",
    "_safe_float",
    "_normalize_lang_flag",
    "_norm_text",
    "_metric_column_name",
    "get_metric_value",
}
ALGORITHM_CALLS = [
    "KMeans(",
    "LogisticRegression(",
    "train_test_split(",
    "proportions_ztest(",
    "silhouette_score(",
    "roc_auc_score(",
    "zt_ind_solve_power(",
]
PUBLISHED_FIGURES = ["6,849", "460", "3,916", "80.0%", "0.0779"]
LANGUAGE_FIX = {
    "EN": "Fix Language -> then Loyalty Campaign",
    "PT": "Corrigir Idioma -> depois Campanha de Fideliza",
}


def _cells(notebook: Path) -> list[dict]:
    """Read the cells of a notebook.

    Parameters
    ----------
    notebook : Path
        Path of an ``.ipynb`` file.

    Returns
    -------
    list of dict
        The notebook cells as stored in the JSON file.
    """
    return json.loads(notebook.read_text(encoding="utf-8"))["cells"]


def _code(notebook: Path) -> str:
    """Join the source of every code cell of a notebook.

    Parameters
    ----------
    notebook : Path
        Path of an ``.ipynb`` file.

    Returns
    -------
    str
        All code cells separated by blank lines, without IPython magics.
    """
    sources = [
        "".join(cell["source"])
        for cell in _cells(notebook)
        if cell["cell_type"] == "code"
    ]
    lines = "\n\n".join(sources).splitlines()
    return "\n".join(
        line for line in lines if not line.lstrip().startswith("%")
    )


def _output_text(notebook: Path) -> str:
    """Join the saved text outputs of every code cell of a notebook.

    Parameters
    ----------
    notebook : Path
        Path of an ``.ipynb`` file.

    Returns
    -------
    str
        Stream text, markdown and ``text/plain`` results of all cells.
    """
    parts = []
    for cell in _cells(notebook):
        for output in cell.get("outputs", []):
            if output["output_type"] == "stream":
                parts.append("".join(output["text"]))
                continue
            data = output.get("data", {})
            for mime in ("text/plain", "text/markdown", "text/html"):
                parts.append("".join(data.get(mime, [])))
    return "\n".join(parts)


def _language(notebook: Path) -> str:
    """Return ``"EN"`` or ``"PT"`` from the notebook file name.

    Parameters
    ----------
    notebook : Path
        Path of an ``.ipynb`` file.

    Returns
    -------
    str
        The two-letter language suffix of the file name.
    """
    return notebook.stem[-2:]


def test_both_language_notebooks_are_present():
    """The English and Portuguese notebooks exist."""
    assert [_language(path) for path in NOTEBOOKS] == ["EN", "PT"]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_defines_only_presentation_helpers(notebook):
    """No class and no result-computing function is defined inline."""
    tree = ast.parse(_code(notebook))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }

    assert defined <= PRESENTATION_HELPERS, defined - PRESENTATION_HELPERS
    assert not [n for n in tree.body if isinstance(n, ast.ClassDef)]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_imports_the_package_and_selects_its_language(notebook):
    """The package is imported and the labels follow the file language."""
    code = _code(notebook)

    assert "from marketing_campaign" in code
    assert f'LANG = "{_language(notebook).lower()}"' in code


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_does_not_call_the_algorithms_directly(notebook):
    """Clustering, modelling and testing run inside the package."""
    code = _code(notebook)

    for call in ALGORITHM_CALLS:
        assert call not in code, call


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_outputs_hold_the_published_figures(notebook):
    """The saved outputs still show the numbers of the report."""
    text = _output_text(notebook)

    for figure in PUBLISHED_FIGURES:
        assert figure in text, figure
    assert not any(
        output["output_type"] == "error"
        for cell in _cells(notebook)
        for output in cell.get("outputs", [])
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_outputs_include_the_language_fix_action(notebook):
    """The recommendations show users routed to the language fix."""
    assert LANGUAGE_FIX[_language(notebook)] in _output_text(notebook)
