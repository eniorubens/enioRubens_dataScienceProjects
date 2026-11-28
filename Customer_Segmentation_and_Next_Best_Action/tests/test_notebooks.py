"""Structural contracts for the bilingual notebooks."""

from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def _code_sources(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def _semantic_lines(source: str) -> list[str]:
    """Ignore cell-local formatting while preserving executable line order."""
    return [line.strip() for line in source.splitlines() if line.strip()]


def _canonicalize_language(source: str) -> str:
    return source.replace('set_language("pt")', 'set_language("en")')


def test_bilingual_notebooks_share_code_cells_and_logic() -> None:
    en = _code_sources(
        NOTEBOOK_DIR / "Customer_Segmentation_and_Next_Best_Action-EN.ipynb"
    )
    pt = _code_sources(
        NOTEBOOK_DIR / "Customer_Segmentation_and_Next_Best_Action-PT.ipynb"
    )

    assert len(en) == len(pt) == 32
    for en_cell, pt_cell in zip(en, pt):
        assert _semantic_lines(en_cell) == _semantic_lines(
            _canonicalize_language(pt_cell)
        )


def test_notebooks_delegate_repurchase_features_and_have_no_inline_functions() -> None:
    sources = _code_sources(
        NOTEBOOK_DIR / "Customer_Segmentation_and_Next_Best_Action-EN.ipynb"
    ) + _code_sources(
        NOTEBOOK_DIR / "Customer_Segmentation_and_Next_Best_Action-PT.ipynb"
    )
    joined = "\n".join(sources)

    assert "build_repurchase_dataset" in joined
    assert "mode_or_unknown" not in joined
    assert not re.search(r"(?m)^\s*def\s+", joined)
