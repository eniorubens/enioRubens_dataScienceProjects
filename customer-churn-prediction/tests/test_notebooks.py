from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks"
NOTEBOOKS = [
    NOTEBOOK_DIR / "Customer-Churn-Prediction-EN.ipynb",
    NOTEBOOK_DIR / "Customer-Churn-Prediction-PT.ipynb",
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_cells(nb: dict) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_bilingual_notebooks_have_equal_executable_cell_counts():
    cells = [_code_cells(_load(path)) for path in NOTEBOOKS]
    assert [len(items) for items in cells] == [58, 58]
    assert all(sum("recall_macro = 0.7640" in source for source in items) == 1 for items in cells)


def test_notebooks_use_project_local_language_and_backend_boundaries():
    forbidden = (
        "from multilang",
        "LangMap",
        "ds_toolkit",
        "sys.path.insert",
        "from opt_binary_clf_pipe",
    )
    for path in NOTEBOOKS:
        notebook_text = path.read_text(encoding="utf-8")
        assert "from churn_project.i18n import set_language, t" in notebook_text
        assert "from churn_project.modeling_backend import (" in notebook_text
        assert not any(token in notebook_text for token in forbidden)


def test_duplicate_annual_revenue_cell_is_archived_not_executable():
    en = _load(NOTEBOOKS[0])
    archived = [
        cell
        for cell in en["cells"]
        if "archived-duplicate" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(archived) == 1
    assert archived[0]["cell_type"] == "markdown"
    assert archived[0].get("execution_count") is None


def test_error_analysis_has_shared_contract_distribution_logic():
    for path in NOTEBOOKS:
        sources = _code_cells(_load(path))
        error_analysis = next(source for source in sources if "error_df = test_features.copy()" in source)
        assert "error_labels =" in error_analysis
        assert "contract_cmp = pd.DataFrame" in error_analysis
        assert "t('Contract Distribution in FN vs FP')" in error_analysis
