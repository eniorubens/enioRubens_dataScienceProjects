"""Notebook contract: synthetic notebooks for the rules, then the repository's own notebooks."""

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from retail_multilevel.nbcontract import check_notebooks
from retail_multilevel.paths import ROOT


def _write(path, lang, code_extra="", md="texto"):
    nb = new_notebook(
        cells=[
            new_markdown_cell(md),
            new_code_cell(f'LANG = "{lang}"'),
            new_code_cell("from retail_multilevel.runs import load_run\nrun = load_run('ols')" + code_extra),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)


@pytest.fixture
def nb_root(tmp_path):
    root = tmp_path / "notebooks"
    _write(root / "pt_br" / "01_eda.ipynb", "pt")
    _write(root / "en_us" / "01_eda.ipynb", "en", md="text")
    return root


def test_valid_pair_passes(nb_root):
    assert check_notebooks(nb_root) == []


def test_missing_pair_is_reported(nb_root):
    _write(nb_root / "pt_br" / "02_modelo_nulo.ipynb", "pt")
    assert any("without EN pair" in p for p in check_notebooks(nb_root))


def test_code_divergence_is_reported(nb_root):
    _write(nb_root / "en_us" / "01_eda.ipynb", "en", code_extra="\nprint(1)")
    assert any("code cells differ" in p for p in check_notebooks(nb_root))


def test_wrong_lang_is_reported(nb_root):
    _write(nb_root / "en_us" / "01_eda.ipynb", "pt")
    assert any("expected exactly one LANG = 'en'" in p for p in check_notebooks(nb_root))


def test_fitting_in_notebook_is_reported(nb_root):
    extra = "\nrun = get_or_fit('ols', {}, fit)"
    _write(nb_root / "pt_br" / "01_eda.ipynb", "pt", code_extra=extra)
    _write(nb_root / "en_us" / "01_eda.ipynb", "en", code_extra=extra)
    assert any("must not fit models" in p for p in check_notebooks(nb_root))


def test_repository_notebooks_follow_the_contract():
    assert check_notebooks(ROOT / "notebooks") == []
