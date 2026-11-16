import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def normalize_language(source: str) -> str:
    return source.replace('LANG = "pt"', 'LANG = "xx"').replace('LANG = "en"', 'LANG = "xx"')


def test_bilingual_notebooks_share_executable_code():
    pairs = [
        (NOTEBOOK_DIR / "pt_br" / "01_apresentacao_e_pre_analise_olist.ipynb", NOTEBOOK_DIR / "en_us" / "01_apresentacao_e_pre_analise_olist.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "02_eda_atraso_sla_olist.ipynb", NOTEBOOK_DIR / "en_us" / "02_eda_atraso_sla_olist.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "03_eda_geografia_e_composicao_pedido.ipynb", NOTEBOOK_DIR / "en_us" / "03_eda_geography_and_order_composition.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "04_tabela_analitica_candidata.ipynb", NOTEBOOK_DIR / "en_us" / "04_candidate_analytical_table.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "05_contrato_temporal_e_split.ipynb", NOTEBOOK_DIR / "en_us" / "05_temporal_contract_and_split.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "06_modeling_baseline_ols_sklearn.ipynb", NOTEBOOK_DIR / "en_us" / "06_modeling_baseline_ols_sklearn.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "07_tournament_estimators.ipynb", NOTEBOOK_DIR / "en_us" / "07_estimator_tournament.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "08_otimizacao_optuna.ipynb", NOTEBOOK_DIR / "en_us" / "08_optuna_optimization.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "09_avaliacao_final.ipynb", NOTEBOOK_DIR / "en_us" / "09_final_evaluation.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "10_pos_hoc_f2.ipynb", NOTEBOOK_DIR / "en_us" / "10_post_hoc_f2.ipynb"),
        (NOTEBOOK_DIR / "pt_br" / "11_relatorio_final.ipynb", NOTEBOOK_DIR / "en_us" / "11_final_report.ipynb"),
    ]

    for portuguese, english in pairs:
        pt_cells = code_cells(portuguese)
        en_cells = code_cells(english)
        assert len(pt_cells) == len(en_cells)
        assert [normalize_language(cell) for cell in pt_cells] == [normalize_language(cell) for cell in en_cells]


def test_notebooks_delegate_logic_to_src():
    notebook_paths = sorted(NOTEBOOK_DIR.rglob("*.ipynb"))
    assert notebook_paths

    for path in notebook_paths:
        source = "\n".join(code_cells(path))
        assert "from ecom_delivery_sla import" in source
        assert "pd.read_csv" not in source
        assert "order_delivered_customer_date" not in source
        assert "delivery_delay_days =" not in source


def test_analytical_notebooks_follow_markdown_python_insight_sequence():
    """Require narrative context before and after each executable block."""
    for path in sorted(NOTEBOOK_DIR.rglob("*.ipynb")):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            assert index > 0 and cells[index - 1]["cell_type"] == "markdown"
            assert index + 1 < len(cells) and cells[index + 1]["cell_type"] == "markdown"

        markdown = "\n".join(
            "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
        ).lower()
        assert "insight" in markdown


def output_problems(notebook: dict) -> list[str]:
    """Return why a notebook is not a clean, executed top-to-bottom run."""
    problems = []
    expected = 1
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        if cell.get("execution_count") != expected:
            problems.append(
                f"cell {index}: execution_count={cell.get('execution_count')!r}, "
                f"expected {expected}"
            )
        if not cell.get("outputs"):
            problems.append(f"cell {index}: no outputs")
        expected += 1
    return problems


def test_output_problems_detects_missing_counts_and_outputs():
    clean = {
        "cells": [
            {"cell_type": "markdown", "source": []},
            {"cell_type": "code", "execution_count": 1, "outputs": [{}], "source": []},
            {"cell_type": "code", "execution_count": 2, "outputs": [{}], "source": []},
        ]
    }
    stale = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "outputs": [], "source": []},
            {"cell_type": "code", "execution_count": 3, "outputs": [{}], "source": []},
        ]
    }

    assert output_problems(clean) == []
    assert len(output_problems(stale)) == 3


@pytest.mark.skipif(
    os.environ.get("REQUIRE_NOTEBOOK_OUTPUTS") != "1"
    or os.environ.get("SKIP_NOTEBOOK_OUTPUTS") == "1",
    reason=(
        "Skipped by default. Notebooks are edited without executing them; after "
        "re-running 01-11 (PT and EN) in VS Code, set REQUIRE_NOTEBOOK_OUTPUTS=1 "
        "to require sequential execution_count and at least one output per code cell. "
        "SKIP_NOTEBOOK_OUTPUTS=1 forces a skip."
    ),
)
def test_executed_notebooks_01_to_11_have_sequential_counts_and_outputs():
    paths = sorted(
        path
        for path in NOTEBOOK_DIR.rglob("*.ipynb")
        if path.name[:2] in {f"{number:02d}" for number in range(1, 12)}
    )
    assert len(paths) == 22  # eleven notebooks in each language

    failures = {}
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        problems = output_problems(notebook)
        if problems:
            failures[str(path.relative_to(ROOT))] = problems

    assert not failures, failures



_ANY_BULLET_LINE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s", flags=re.MULTILINE)
# Editorial exception (REGISTRO-DE-DECISOES.md, question 22): the variables of a
# formula are defined after "Onde:"/"Where:" as "- $symbol$: definition" bullets.
FORMULA_VARIABLE_LINE = re.compile(r"^- \$[^$\n]+\$(?:, \$[^$\n]+\$)*: \S")


class _ProseBulletPattern:
    """Find bullet lines, ignoring formula-variable definitions."""

    def search(self, text: str):
        for match in _ANY_BULLET_LINE.finditer(text):
            line = text[match.start():].lstrip("\n").split("\n", 1)[0]
            if not FORMULA_VARIABLE_LINE.match(line):
                return match
        return None


BULLET_LINE = _ProseBulletPattern()


def test_bullet_rule_allows_only_formula_variable_definitions():
    """Formula variables may be bulleted; any other bullet or numbered line is rejected."""
    assert BULLET_LINE.search("Onde:\n\n- $s$: número de violações;\n- $\\varphi_1$, $\\varphi_2$: latitudes.") is None
    assert BULLET_LINE.search("Texto.\n- item comum de lista") is not None
    assert BULLET_LINE.search("1. primeiro passo") is not None
    assert BULLET_LINE.search("- $s$ sem dois-pontos") is not None


def test_notebook_09_is_prose_without_bullets_and_single_call():
    """The final evaluation has no bullet lists and a single test-opening call.

    Notebook 09 was executed by the owner on 2026-11-14 (F1), so it now carries
    outputs; the unexecuted-state check applies to notebook 10 only.
    """
    for path in (
        NOTEBOOK_DIR / "pt_br" / "09_avaliacao_final.ipynb",
        NOTEBOOK_DIR / "en_us" / "09_final_evaluation.ipynb",
    ):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            assert cell.get("id")
            if cell["cell_type"] != "code":
                assert not BULLET_LINE.search("".join(cell["source"])), cell["id"]
        assert "\n".join(code_cells(path)).count("run_final_evaluation(") == 1


def test_notebook_10_is_prose_without_bullets_and_single_call():
    """The F2 post-hoc evaluation has no bullet lists and a single test-opening call.

    Notebook 10 was executed by the owner on 2026-11-14 (F2), so it now carries
    outputs and the unexecuted-state check no longer applies.
    """
    for path in (
        NOTEBOOK_DIR / "pt_br" / "10_pos_hoc_f2.ipynb",
        NOTEBOOK_DIR / "en_us" / "10_post_hoc_f2.ipynb",
    ):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        reference = json.loads((NOTEBOOK_DIR / "pt_br" / "09_avaliacao_final.ipynb").read_text(encoding="utf-8"))
        assert notebook["metadata"]["kernelspec"] == reference["metadata"]["kernelspec"]
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(ids) and len(ids) == len(set(ids))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                assert not BULLET_LINE.search("".join(cell["source"])), cell["id"]
        source = "\n".join(code_cells(path))
        assert source.count("run_posthoc_evaluation(") == 1
        assert source.count("allow_test=True") == 1
        assert 'version=POSTHOC_VERSION' in source and 'POSTHOC_VERSION = "F2"' in source
        assert "run_final_evaluation" not in source


REPORT_NOTEBOOKS = (
    NOTEBOOK_DIR / "pt_br" / "11_relatorio_final.ipynb",
    NOTEBOOK_DIR / "en_us" / "11_final_report.ipynb",
)
REPORT_FORBIDDEN_CALLS = (
    "run_final_evaluation(",
    "run_posthoc_evaluation(",
    "cached_run(",
    ".fit(",
    "run_optuna_study(",
    "run_classification_tournament",
    "run_regression_tournament",
    "run_distance_ablation(",
)


def test_notebook_11_is_prose_and_reads_only_stored_results():
    """The final report has no bullets, unique ids and no call that trains or opens the test.

    Notebook 11 was executed by the owner on 2026-11-16, so outputs are allowed.
    """
    reference = json.loads((NOTEBOOK_DIR / "pt_br" / "10_pos_hoc_f2.ipynb").read_text(encoding="utf-8"))
    for path in REPORT_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"] == reference["metadata"]["kernelspec"]
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(re.fullmatch(r"[0-9a-f]{8}", str(identifier)) for identifier in ids), path.name
        assert len(ids) == len(set(ids)), path.name
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                assert not BULLET_LINE.search("".join(cell["source"])), (path.name, cell["id"])
        source = "\n".join(code_cells(path))
        for call in REPORT_FORBIDDEN_CALLS:
            assert call not in source, (path.name, call)
        assert "allow_test" not in source
        assert "load_run_unchecked(" in source


def test_notebook_11_states_the_governance_in_the_opening():
    for path in REPORT_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        opening = "".join(notebook["cells"][0]["source"])
        for fragment in ("F1", "F2", "2018", "production_eligible = false"):
            assert fragment in opening, (path.name, fragment)


_CITATION_AUTHOR = r"[A-ZÀ-Ý][^();,\d]*?"
_CITATION_ITEM = rf"{_CITATION_AUTHOR}, \d{{4}}"
CITATION = re.compile(rf"\(({_CITATION_ITEM}(?:; {_CITATION_ITEM})*)\)")
REFERENCES_HEADING = re.compile(r"^## (?:Referências|References)\s*\n")


def citation_keys(text: str) -> set[tuple[str, int]]:
    """Return (first-author surname, year) for every author-date citation."""
    keys = set()
    for group in CITATION.findall(text):
        for item in group.split("; "):
            names, year = item.rsplit(", ", 1)
            first = re.split(r" (?:e|and) ", names)[0].replace(" et al.", "").strip()
            keys.add((first, int(year)))
    return keys


def reference_entry_keys(text: str) -> set[tuple[str, int]]:
    """Return (first-author surname, year) for each paragraph of a references cell."""
    keys = set()
    for entry in REFERENCES_HEADING.sub("", text).strip().split("\n\n"):
        surname = re.match(r"[^,(]+", entry.strip())
        year = re.search(r"\((\d{4})\)", entry)
        assert surname and year, entry
        keys.add((surname.group().strip(), int(year.group(1))))
    return keys


def test_citation_regex_ignores_non_citations():
    for text in ("(P-02)", "(T-05)", "(2018-07-01)", "(IC 95%)", "(`code`)", "(Jul–Aug 2018)"):
        assert not citation_keys(text), text
    assert citation_keys("(Sinnott, 1984)") == {("Sinnott", 1984)}
    assert citation_keys("(Breusch e Pagan, 1979)") == {("Breusch", 1979)}
    assert citation_keys("(Breusch and Pagan, 1979)") == {("Breusch", 1979)}
    assert citation_keys("(Akiba et al., 2019)") == {("Akiba", 2019)}
    assert citation_keys("(Tashman, 2000; Bergmeir e Benítez, 2012)") == {
        ("Tashman", 2000),
        ("Bergmeir", 2012),
    }


def test_notebook_citations_match_references_and_languages():
    """Every in-text citation has a reference entry, and vice versa, in PT and EN alike."""
    cited_by_notebook: dict[str, dict[str, set[tuple[str, int]]]] = {}
    for path in sorted(NOTEBOOK_DIR.rglob("*.ipynb")):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        markdown = [(index, "".join(cell["source"])) for index, cell in enumerate(cells) if cell["cell_type"] == "markdown"]
        last = cells[-1]
        last_text = "".join(last["source"])
        has_references = last["cell_type"] == "markdown" and bool(REFERENCES_HEADING.match(last_text))
        body = "\n".join(text for index, text in markdown if not (has_references and index == len(cells) - 1))
        cited = citation_keys(body)
        # No references cell may sit anywhere but at the end.
        assert sum(1 for _, text in markdown if REFERENCES_HEADING.match(text)) == int(has_references), path.name
        if cited:
            assert has_references, f"{path.name}: citations without a final references cell"
            assert reference_entry_keys(last_text) == cited, path.name
        else:
            assert not has_references, f"{path.name}: references cell without citations"
        cited_by_notebook.setdefault(path.name[:2], {})[path.parent.name] = cited

    assert len(cited_by_notebook) == 11
    for number, by_language in cited_by_notebook.items():
        assert set(by_language) == {"pt_br", "en_us"}, number
        assert by_language["pt_br"] == by_language["en_us"], number
