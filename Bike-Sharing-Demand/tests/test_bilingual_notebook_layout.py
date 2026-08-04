"""Structural contract for the bilingual notebook editions."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"
PT_BR_DIR = NOTEBOOKS_ROOT / "pt-BR"
EN_US_DIR = NOTEBOOKS_ROOT / "en-US"

NOTEBOOK_NAMES = (
    "01_Seoul_Bike_2015-2024_EDA.ipynb",
    "02_Seoul_Bike_Multivariate_Analysis.ipynb",
    "03_Feature_Engineering_EDA.ipynb",
    "04_Seoul_Bike_Model_Selection.ipynb",
    "05_Seoul_Bike_Final_Validation.ipynb",
    "06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb",
    "07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb",
    "08_Seoul_Bike_Operational_Forecast_Demo.ipynb",
)
TRANSLATED_NOTEBOOK_NAMES = (
    "01_Seoul_Bike_2015-2024_EDA.ipynb",
    "02_Seoul_Bike_Multivariate_Analysis.ipynb",
    "03_Feature_Engineering_EDA.ipynb",
    "04_Seoul_Bike_Model_Selection.ipynb",
    "05_Seoul_Bike_Final_Validation.ipynb",
    "06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb",
    "07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb",
    "08_Seoul_Bike_Operational_Forecast_Demo.ipynb",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_canonical_edition_contains_the_eight_notebooks():
    assert tuple(path.name for path in sorted(PT_BR_DIR.glob("*.ipynb"))) == NOTEBOOK_NAMES
    assert not tuple(NOTEBOOKS_ROOT.glob("*.ipynb"))


def test_english_directory_is_reserved_for_the_mirrored_edition():
    assert EN_US_DIR.is_dir()


def test_canonical_notebooks_use_robust_project_root_discovery():
    for name in NOTEBOOK_NAMES:
        code = _code(_load(PT_BR_DIR / name))
        assert "filter(" in code
        assert "lambda candidate:" in code
        assert "os.path.abspath('..')" not in code
        assert "Path('../dataset" not in code
        assert "make_lang('pt')" in code


def test_canonical_notebooks_keep_the_project_kernel():
    for name in NOTEBOOK_NAMES:
        notebook = _load(PT_BR_DIR / name)
        kernelspec = notebook["metadata"]["kernelspec"]
        assert kernelspec["name"] == "bike-sharing"
        assert kernelspec["display_name"] == "Python (Bike-Sharing)"


def test_translated_pilot_preserves_cell_and_code_identity():
    for name in TRANSLATED_NOTEBOOK_NAMES:
        source = _load(PT_BR_DIR / name)
        target = _load(EN_US_DIR / name)
        assert len(source["cells"]) == len(target["cells"])
        assert [cell.get("id") for cell in source["cells"]] == [
            cell.get("id") for cell in target["cells"]
        ]
        assert [cell["cell_type"] for cell in source["cells"]] == [
            cell["cell_type"] for cell in target["cells"]
        ]

        source_code = _code(source)
        target_code = _code(target).replace("make_lang('en')", "make_lang('pt')")
        assert target_code == source_code
        assert target["metadata"]["project_language"] == "en-US"


def test_translated_notebooks_have_english_markdown_and_coherent_execution_state():
    expected_titles = {
        "01_Seoul_Bike_2015-2024_EDA.ipynb": (
            "# Seoul Bike Sharing, 2015–2024 — Exploratory Data Analysis"
        ),
        "02_Seoul_Bike_Multivariate_Analysis.ipynb": (
            "# Seoul Bike Sharing, 2015–2024 — Multivariate Analysis"
        ),
        "03_Feature_Engineering_EDA.ipynb": ("# Feature Engineering and EDA of Derived Features"),
        "04_Seoul_Bike_Model_Selection.ipynb": (
            "# Model Selection — Seoul Bike Sharing 2015–2024 (Notebook 04)"
        ),
        "05_Seoul_Bike_Final_Validation.ipynb": (
            "# Final Validation \u2014 Seoul Bike Sharing 2015\u20132024"
        ),
        "06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb": (
            "# Residual Structure and Uncertainty Experiments"
        ),
        "07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb": (
            "# Temporal Conformal Calibration and Predictive Intervals"
        ),
        "08_Seoul_Bike_Operational_Forecast_Demo.ipynb": (
            "# Operational Demonstration of an Hourly Forecast"
        ),
    }
    for name in TRANSLATED_NOTEBOOK_NAMES:
        notebook = _load(EN_US_DIR / name)
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert expected_titles[name] in markdown
        assert "# Compartilhamento de bicicletas" not in markdown
        assert not any(token in markdown for token in ("Ã", "â€", "�"))

        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        counts = [cell.get("execution_count") for cell in code_cells]
        assert all(count is None for count in counts) or all(
            isinstance(count, int) for count in counts
        )
        assert not any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )
