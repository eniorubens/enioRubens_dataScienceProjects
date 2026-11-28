"""Contracts for the executable parity of the bilingual notebooks."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from i18n import SUPPORTED_LANGUAGES, get_labels  # noqa: E402


NOTEBOOK_PAIRS = (
    ("pt-BR/01_contrato_e_eda.ipynb", "en-US/01_data_contract_and_eda.ipynb"),
    ("pt-BR/02_eda_exploratoria.ipynb", "en-US/02_exploratory_eda.ipynb"),
    ("pt-BR/03_baseline_comparacao.ipynb", "en-US/03_baseline_comparison.ipynb"),
    ("pt-BR/04_calibracao_auditoria.ipynb", "en-US/04_calibration_audit.ipynb"),
    ("pt-BR/05_selecao_modelo_threshold.ipynb", "en-US/05_model_threshold_selection.ipynb"),
    ("pt-BR/05a_ajuste_hiperparametros.ipynb", "en-US/05a_hyperparameter_tuning.ipynb"),
    ("pt-BR/06_avaliacao_final_test.ipynb", "en-US/06_final_test_evaluation.ipynb"),
    ("pt-BR/07_aplicacao_pratica_modelo.ipynb", "en-US/07_practical_model_application.ipynb"),
)
LANGUAGE_ASSIGNMENT = re.compile(r'LANG = "(?:en|pt)"')


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code_cells(path: Path) -> list[dict]:
    return [cell for cell in _cells(path) if cell["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _normalise_language(source: str) -> str:
    return LANGUAGE_ASSIGNMENT.sub('LANG = "<LANG>"', source)


class NotebookParityTests(unittest.TestCase):
    """Keep PT-BR and EN-US executable cells on one shared implementation."""

    def test_notebook_pairs_have_equal_code_after_language_selection(self):
        total_code_cells = 0
        for pt_name, en_name in NOTEBOOK_PAIRS:
            pt_cells = _code_cells(ROOT / pt_name)
            en_cells = _code_cells(ROOT / en_name)
            self.assertEqual(len(pt_cells), len(en_cells), (pt_name, en_name))
            total_code_cells += len(pt_cells)

            for position, (pt_cell, en_cell) in enumerate(zip(pt_cells, en_cells)):
                self.assertEqual(
                    _normalise_language(_source(pt_cell)),
                    _normalise_language(_source(en_cell)),
                    (pt_name, en_name, position),
                )

        self.assertEqual(total_code_cells, 35)

    def test_language_selectors_are_declared_only_where_catalog_labels_are_used(self):
        for pt_name, en_name in NOTEBOOK_PAIRS:
            for language, notebook_name in (("pt", pt_name), ("en", en_name)):
                source = "\n".join(_source(cell) for cell in _code_cells(ROOT / notebook_name))
                assignments = LANGUAGE_ASSIGNMENT.findall(source)
                if "get_labels" in source:
                    self.assertEqual(assignments, [f'LANG = "{language}"'])
                else:
                    self.assertEqual(assignments, [])

    def test_catalog_has_the_shared_notebook_label_contract(self):
        expected_keys = {
            "table_value",
            "threshold_precision",
            "threshold_recall",
            "threshold_f1",
            "threshold_selection_rate",
            "group_selection_rate",
            "group_tpr",
            "group_fpr",
            "group_fnr",
            "group_ppv",
        }

        self.assertEqual(SUPPORTED_LANGUAGES, ("en", "pt"))
        for language in SUPPORTED_LANGUAGES:
            labels = get_labels(language)
            self.assertEqual(set(labels), expected_keys)
            self.assertTrue(all(isinstance(value, str) and value for value in labels.values()))


if __name__ == "__main__":
    unittest.main()
