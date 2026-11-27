"""Structural contracts for the bilingual notebook pairs."""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PAIRS = (
    ("01_Framing_EDA_EN.ipynb", "01_Framing_EDA_PT.ipynb"),
    ("02_Baseline_Propensity_EN.ipynb", "02_Baseline_Propensity_PT.ipynb"),
    ("03_Meta_Learners_EN.ipynb", "03_Meta_Learners_PT.ipynb"),
    ("04_Causal_Forest_Uplift_Trees_EN.ipynb", "04_Causal_Forest_Uplift_Trees_PT.ipynb"),
    ("05_Evaluation_Sealed_Test_EN.ipynb", "05_Evaluation_Sealed_Test_PT.ipynb"),
    ("06_Heterogeneity_Uplift_Funnel_EN.ipynb", "06_Heterogeneity_Uplift_Funnel_PT.ipynb"),
    ("07_Policy_Learning_ROI_EN.ipynb", "07_Policy_Learning_ROI_PT.ipynb"),
    ("08_Robustness_Limitations_EN.ipynb", "08_Robustness_Limitations_PT.ipynb"),
)
LANG_ASSIGNMENT = re.compile(r"lang = make_lang\(['\"](?:en|pt)['\"]\)")
LANG_NORMALISATION = re.compile(r"lang = make_lang\(['\"](?:en|pt)['\"]\)")


def _code_cells(path: Path) -> list[dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def _source(cells: list[dict]) -> str:
    return "".join("".join(cell.get("source", [])) for cell in cells)


def _normalise_language(source: str) -> str:
    return LANG_NORMALISATION.sub("lang = make_lang('<LANG>')", source)


def test_bilingual_notebooks_share_executable_code():
    for en_name, pt_name in NOTEBOOK_PAIRS:
        en_cells = _code_cells(PROJECT_ROOT / "notebooks" / "en-US" / en_name)
        pt_cells = _code_cells(PROJECT_ROOT / "notebooks" / "pt-BR" / pt_name)

        assert len(en_cells) == len(pt_cells), (en_name, pt_name)
        assert _normalise_language(_source(en_cells)) == _normalise_language(_source(pt_cells)), (
            en_name,
            pt_name,
        )


def test_bilingual_notebooks_use_the_shared_catalog_layer():
    for en_name, pt_name in NOTEBOOK_PAIRS:
        for language, notebook_name in (("en-US", en_name), ("pt-BR", pt_name)):
            source = _source(_code_cells(PROJECT_ROOT / "notebooks" / language / notebook_name))
            assert "from src.i18n import make_lang" in source
            assert len(LANG_ASSIGNMENT.findall(source)) == 1, (language, notebook_name)


def test_catalog_contains_the_keys_used_by_the_shared_cells():
    catalog = json.loads(
        (PROJECT_ROOT / "src" / "i18n_catalogs" / "pt_en.json").read_text(encoding="utf-8")
    )
    required_keys = {
        "linhas",
        "tratados",
        "Desvios reais por learner",
        "Interações não nulas do ElasticNet",
        "Fonte: Validação, 12.800 linhas  |  Método: LightGBM sobre o braço tratado",
        "Hash do pré-registro embutido em final_models.joblib não bate com o preregistration.json atual -- os modelos podem ter sido congelados sob uma especificação diferente da vigente.",
        "{path} já existe -- S6 já foi executada. Esta célula não reabre o teste selado nem recalcula métricas silenciosamente. Leia s6_results.json para o resultado já produzido; não há opção force=True para contornar esta proteção.",
        "Essas comparações são pré-especificadas e reportadas por completude (inclusive para encerrar a pergunta de S4 sobre X+Tree/S+LightGBM vs. baseline), mas não substituem retrospectivamente o modelo primário nem elegem um novo 'vencedor' com base no maior Qini absoluto observado aqui.",
    }

    assert required_keys <= catalog.keys()
    assert all(isinstance(catalog[key], str) and catalog[key] for key in required_keys)
