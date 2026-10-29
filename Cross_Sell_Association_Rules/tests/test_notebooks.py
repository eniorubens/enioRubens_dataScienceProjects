"""Tests that the notebooks use the package and keep the published outputs."""
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted(
    (Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")
)
PUBLISHED_FIGURES = [
    "38765",
    "14963",
    "Lift: 0.77",
    "Conviction: 0.97",
    "Zhang : -0.2625",
    "216 750",
    "210 648",
]


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
        All code cells separated by blank lines.
    """
    return "\n\n".join(
        "".join(cell["source"])
        for cell in _cells(notebook)
        if cell["cell_type"] == "code"
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
        Stream text and ``text/plain`` results of all code cells.
    """
    parts = []
    for cell in _cells(notebook):
        for output in cell.get("outputs", []):
            if output["output_type"] == "stream":
                parts.append("".join(output["text"]))
            else:
                parts.append("".join(output.get("data", {}).get(
                    "text/plain", [])))
    return "\n".join(parts)


def test_both_language_notebooks_are_present():
    """The English and Portuguese notebooks exist."""
    assert [path.stem[-2:] for path in NOTEBOOKS] == ["EN", "PT"]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_calls_the_package_instead_of_defining_logic(notebook):
    """No inline function or class, and no direct mlxtend mining calls."""
    code = _code(notebook)

    assert "from cross_sell_rules import" in code
    assert not re.findall(r"^(?:def|class) \w+", code, flags=re.M)
    assert "apriori(" not in code
    assert "association_rules(" not in code


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
