"""Contract between the pt_br and en_us notebooks (skipped while no notebooks exist)."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PT = sorted((ROOT / "notebooks" / "pt_br").glob("*.ipynb"))
EN = sorted((ROOT / "notebooks" / "en_us").glob("*.ipynb"))
ALL = PT + EN
FORBIDDEN = ("pd.read_csv", "open(", "groupby", ".plot(", "scatter(", "subplots(", "np.", "pd.", "hashlib")

pytestmark = pytest.mark.skipif(not (PT or EN), reason="no notebooks yet")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def lang_of(path):
    return "pt" if path.parent.name == "pt_br" else "en"


def source(cell):
    return "".join(cell["source"])


def code_sources(path, lang):
    out = []
    for cell in load(path):
        if cell["cell_type"] == "code":
            out.append(source(cell).replace(f'LANG = "{lang}"', 'LANG = "<lang>"'))
    return out


def test_same_count():
    assert len(PT) == len(EN)


def test_numbering_pairs():
    for pt, en in zip(PT, EN, strict=True):
        assert pt.name[:2] == en.name[:2], (pt.name, en.name)


def test_code_cells_identical():
    for pt, en in zip(PT, EN, strict=True):
        assert code_sources(pt, "pt") == code_sources(en, "en"), (pt.name, en.name)


def test_markdown_cells_pair_up():
    for pt, en in zip(PT, EN, strict=True):
        kinds_pt = [c["cell_type"] for c in load(pt)]
        kinds_en = [c["cell_type"] for c in load(en)]
        assert kinds_pt == kinds_en, (pt.name, en.name)


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_language_parameter(path):
    first = next(source(c) for c in load(path) if c["cell_type"] == "code")
    assert first.startswith(f'LANG = "{lang_of(path)}"')
    assert sum(source(c).count("LANG = ") for c in load(path) if c["cell_type"] == "code") == 1


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_imports_and_no_logic(path):
    code = "\n".join(source(c) for c in load(path) if c["cell_type"] == "code")
    assert re.search(r"^(from delivery_routing_vrptw import|import delivery_routing_vrptw)", code, re.M)
    for token in FORBIDDEN:
        assert token not in code, f"{path.name}: forbidden token {token!r} in code cells"


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_alternation_and_insight(path):
    cells = load(path)
    assert cells[0]["cell_type"] == "markdown"
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            assert 0 < i < len(cells) - 1, f"{path.name}: code cell {i} lacks surrounding markdown"
            assert cells[i - 1]["cell_type"] == "markdown"
            assert cells[i + 1]["cell_type"] == "markdown"
            assert "Insight" in source(cells[i + 1]), f"{path.name}: cell {i + 1} is not an Insight"
    assert any("Insight" in source(c) for c in cells if c["cell_type"] == "markdown")


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_executed_without_errors(path):
    code = [c for c in load(path) if c["cell_type"] == "code"]
    assert [c["execution_count"] for c in code] == list(range(1, len(code) + 1))
    for cell in code:
        assert cell["outputs"], f"{path.name}: code cell without output"
        assert all(o["output_type"] != "error" for o in cell["outputs"])


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_eda_notebook_uses_inputs_only(path):
    if not path.name.startswith("02_"):
        pytest.skip("only the EDA notebook is restricted to inputs")
    code = "\n".join(source(c) for c in load(path) if c["cell_type"] == "code").lower()
    for token in ("solve", "ortools", "verify", "baselines", "cvrplib", "convention"):
        assert token not in code, f"{path.name}: {token!r} must not appear in the EDA code cells"


def _eval_names():
    rows = (ROOT / "data" / "reference" / "split.csv").read_text(encoding="utf-8").splitlines()[1:]
    return [r.split(",")[0] for r in rows if r.split(",")[2] == "eval"]


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_result_notebooks_use_dev_only_and_no_solver(path):
    if not path.name.startswith(("03_", "04_")):
        pytest.skip("only notebooks 03 and 04 are restricted to development results")
    code = "\n".join(source(c) for c in load(path) if c["cell_type"] == "code")
    names = _eval_names()
    assert len(names) == 42
    hit = re.findall(r"\b(" + "|".join(names) + r")\b", code, flags=re.I)
    assert not hit, f"{path.name}: evaluation instances referenced in code: {hit}"
    for token in ("solve", "minimize_fleet", "run_gate", "run_dev_grid"):
        assert token not in code, f"{path.name}: {token!r} must not be called in notebooks 03/04"


SOLVER_TOKENS = ("solve", "minimize_fleet", "run_gate", "run_dev_grid", "ortools", "run_evaluation", "run_window", "run_early")
SINTEF_TOKENS = (
    "sintef_bks", "cvrplib", "load_dev_instances", "read_references", "ref_vehicles", "ref_distance",
    "delta_vehicles", "gap_distance", "load_evaluation", "load_eval_instances", "load_time_sensitivity",
    "load_dev_grid", "load_gate", "load_selected_config", "load_ranking", "final_eval", "beats_reference",
)


def _code(path):
    return "\n".join(source(c) for c in load(path) if c["cell_type"] == "code")


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebooks_05_06_never_call_solvers_or_name_eval_instances(path):
    if not path.name.startswith(("05_", "06_")):
        pytest.skip("only notebooks 05 and 06")
    code = _code(path)
    for token in SOLVER_TOKENS:
        assert token not in code, f"{path.name}: {token!r} must not appear in notebooks 05/06"
    hit = re.findall(r"\b(" + "|".join(_eval_names()) + r")\b", code, flags=re.I)
    assert not hit, f"{path.name}: evaluation instance names hard-coded in code: {hit}"


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_05_reads_eval_data_only_through_the_loaders(path):
    if not path.name.startswith("05_"):
        pytest.skip("only notebook 05")
    code = _code(path)
    assert "results.load_evaluation(" in code
    for token in ("outputs/", "evaluation/results", "split.csv", "read_archive", "zipfile"):
        assert token not in code, f"{path.name}: eval data must come only from load_evaluation ({token!r})"
    # the only other eval-set loader allowed is the instance loader used for the route maps
    assert code.count("load_eval_instances(") <= 1


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_06_has_no_sintef_references(path):
    if not path.name.startswith("06_"):
        pytest.skip("only notebook 06")
    code = _code(path).lower()
    for token in SINTEF_TOKENS:
        assert token.lower() not in code, f"{path.name}: layer B must not use {token!r}"
    assert "load_dev_inputs" in code and "load_window_scenarios" in code and "load_early_delivery" in code


# --------------------------------------------------------------------------- v2 / phase C / phase D sections
def _headings(path):
    """Numbered/section headings; the closing References cell is not a content section."""
    heads = [source(c).splitlines()[0] for c in load(path) if c["cell_type"] == "markdown" and source(c).startswith("## ")]
    return [h for h in heads if h not in ("## Referências", "## References")]


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_04_v2_section_reads_dev_results_only(path):
    if not path.name.startswith("04_"):
        pytest.skip("only notebook 04")
    code = _code(path)
    assert "v2_view" in code and "results.load_v2_block(" in code and "results.load_v2_decision(" in code
    assert any("v2" in h for h in _headings(path)), "notebook 04 needs the v2 diagnostics section"
    # the synthesis stays the last section, after the v2 section
    heads = _headings(path)
    assert heads.index(next(h for h in heads if "v2" in h)) < len(heads) - 1


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_05_post_hoc_section(path):
    if not path.name.startswith("05_"):
        pytest.skip("only notebook 05")
    code = _code(path)
    assert code.count("results.load_evaluation(") == 2  # v1 and the post-hoc v2 results
    assert "results.load_evaluation_comparison(" in code and "evaluation_v2" in code
    heads = _headings(path)
    assert any("pós-hoc" in h or "Post-hoc" in h for h in heads)
    text = "\n".join(source(c) for c in load(path) if c["cell_type"] == "markdown")
    assert "post-hoc" in text.lower() or "pós-hoc" in text.lower()
    assert "v1" in text and "v2" in text


@pytest.mark.parametrize("path", ALL, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_06_warm_start_section(path):
    if not path.name.startswith("06_"):
        pytest.skip("only notebook 06")
    code = _code(path)
    assert "v2_B5_windows" in code and "v2_B5_early" in code and "v2_view." in code
    assert any("B5" in h for h in _headings(path))
    text = "\n".join(source(c) for c in load(path) if c["cell_type"] == "markdown")
    assert "experimento B5 verificará" not in text and "experiment B5 will check" not in text


NB07 = [p for p in ALL if p.name.startswith("07_")]


def test_notebook_07_exists_in_both_languages():
    assert [p.parent.name for p in NB07] == ["pt_br", "en_us"]
    assert NB07[0].name == "07_robustez_e_cenarios_operacionais.ipynb"
    assert NB07[1].name == "07_robustness_and_operational_scenarios.ipynb"


@pytest.mark.parametrize("path", NB07, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_07_structure_and_layer_b_rules(path):
    code = _code(path)
    # no solver calls and no evaluation instance names anywhere in the code
    for token in SOLVER_TOKENS:
        assert token not in code, f"{path.name}: {token!r} must not appear in notebook 07"
    hit = re.findall(r"\b(" + "|".join(_eval_names()) + r")\b", code, flags=re.I)
    assert not hit, f"{path.name}: evaluation instance names hard-coded in code: {hit}"
    # layer B: no SINTEF reference in C1 (evaluation plans, descriptive) nor in C2 / C3 (development instances)
    for token in SINTEF_TOKENS:
        assert token.lower() not in code.lower(), f"{path.name}: layer B must not use {token!r}"
    # each set is read only through its guarded loader: C1 evaluation plans, C2 and C3 development cells
    assert code.count("results.load_robustness(") == 1
    assert code.count("results.load_phase_c_cells(") == 5
    for token in ("outputs/", "pd.read_csv", "split.csv", "read_archive", "zipfile", "load_dev_inputs"):
        assert token not in code, f"{path.name}: {token!r} not allowed"
    heads = _headings(path)
    assert len(heads) == 7
    for marker in ("C1", "C2.1", "C2.2", "C2.3", "C3"):
        assert any(marker in h for h in heads), f"{path.name}: no section for {marker}"
    # figures use the retina setup and the package plots
    assert "setup_notebook_figures()" in code and code.count("viz_robustness.plot_") == 3


@pytest.mark.parametrize("path", NB07, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_07_states_low_information_of_soft_windows(path):
    text = "\n".join(source(c) for c in load(path) if c["cell_type"] == "markdown")
    assert "3 de 14" in text or "3 of 14" in text
    assert "7" in text and ("falha na construção inicial" in text or "initial construction" in text)
    assert "capacidade" in text.lower() or "capacity" in text.lower()
    # layer B: SINTEF is only mentioned to say that no comparison is made
    for line in text.splitlines():
        if "SINTEF" in line:
            assert any(w in line.lower() for w in ("sem comparação", "no comparison", "nenhuma comparação",
                                                    "sem a referência", "comparison with sintef", "comparação com o sintef",
                                                    "answers", "responde")), line
