"""Structural contracts for the bilingual CLTV notebooks."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MULTILANG_ROOT = ROOT.parent / "ds_toolkit" / "multilang"
sys.path.insert(0, str(MULTILANG_ROOT))

from src.i18n import make_lang  # noqa: E402


NOTEBOOKS = (
    ROOT / "notebooks" / "Customer_Lifetime_Value_PT.ipynb",
    ROOT / "notebooks" / "Customer_Lifetime_Value_EN.ipynb",
)
LANG_SELECTOR = re.compile(r'_t = make_lang\("(?:pt|en)"\)')
REQUIRED_LABELS = (
    "Cleaning Summary",
    "Customers in RFM",
    "Repeat customers (frequency > 0)",
    "Single-purchase customers",
    "Observation period end",
    "Modeling base: {n:,} customers (frequency > 0 and monetary_value > 0)",
    "Excluded (single purchase / zero monetary): {n:,}",
    "Assumption holds (|r| < 0.3): {holds}",
    "── Holdout Validation Results ──────────────────────────",
    "Calibration Plot — Holdout Predicted vs Actual\nBy purchase frequency decile",
    "Predicted purchases (decile mean)",
    "Actual purchases (decile mean)",
    "── BG/NBD Parameters (MAP) ────────────────────────────",
    "── Top 10 Customers by Predicted Purchases ({horizon}d) ────",
    "── Gamma-Gamma Parameters (MAP) ───────────────────────",
    "── Top 10 Customers by Expected Spend per Transaction ──",
    "── Top 10 Customers by {horizon}-day CLTV ────────────────────",
    "180-Day CLTV Distribution",
    "Log scale (log1p) reduces the impact of the long right tail.",
    "log1p(CLTV 180d)",
    "Customers",
    "P(alive) threshold: {threshold:.4f}",
    "── Decision Matrix ─────────────────────────────────────",
    "── Segment Summary ─────────────────────────────────────",
    "Log scale reduces the impact of the long right tail.",
    "log1p(CLTV)",
    "Total expected {horizon}d revenue:   GBP {total:>14,.0f}",
    "Top 10% customers share:        {share:.1f}%",
    "Top 20% customers share:        {share:.1f}%",
    "Segment",
    "Revenue",
    "Share",
    "{action:<26}  {customers:>9,}  GBP {revenue:>10,.0f}  {share:>6.1f}%",
    "Model saved to {model_dir}",
)


def _code_cells(path: Path) -> list[dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _normalise_language(source: str) -> str:
    return LANG_SELECTOR.sub('_t = make_lang("<LANG>")', source)


class NotebookContractTests(unittest.TestCase):
    """Keep the two notebooks on one executable implementation."""

    def test_code_cells_are_identical_after_language_selection(self):
        pt_cells = _code_cells(NOTEBOOKS[0])
        en_cells = _code_cells(NOTEBOOKS[1])
        self.assertEqual(len(pt_cells), 15)
        self.assertEqual(len(en_cells), 15)

        for position, (pt_cell, en_cell) in enumerate(zip(pt_cells, en_cells)):
            self.assertEqual(
                _normalise_language(_source(pt_cell)),
                _normalise_language(_source(en_cell)),
                position,
            )

    def test_notebooks_keep_modeling_logic_out_of_cells(self):
        for notebook in NOTEBOOKS:
            source = "\n".join(_source(cell) for cell in _code_cells(notebook))
            self.assertNotRegex(source, r"(?m)^\s*(def|class)\s+")

    def test_local_catalog_covers_all_shared_labels(self):
        language_map = make_lang("pt")
        translated = language_map({key: key for key in REQUIRED_LABELS})
        self.assertEqual(set(translated), set(REQUIRED_LABELS))
        self.assertEqual(make_lang("en")({"value": "Customers"})["value"], "Customers")


if __name__ == "__main__":
    unittest.main()
