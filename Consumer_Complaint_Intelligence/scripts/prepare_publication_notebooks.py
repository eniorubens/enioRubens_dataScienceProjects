"""Prepare project notebooks for a privacy-safe public review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"


def _source(text: str) -> list[str]:
    """Convert cell text to the line-list representation used by notebooks.

    Args:
        text: Complete cell source.

    Returns:
        Source split into lines while preserving newline characters.
    """

    return text.splitlines(keepends=True)


def _load(relative_path: str) -> dict[str, Any]:
    """Load one notebook relative to the project root.

    Args:
        relative_path: Project-relative notebook path.

    Returns:
        Parsed notebook document.
    """

    path = PROJECT_ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def _write(relative_path: str, notebook: dict[str, Any]) -> None:
    """Write one notebook using stable UTF-8 JSON formatting.

    Args:
        relative_path: Project-relative notebook path.
        notebook: Parsed notebook document.
    """

    path = PROJECT_ROOT / relative_path
    payload = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def _normalize_cell_ids(relative_path: str, notebook: dict[str, Any]) -> None:
    """Add deterministic IDs to cells that predate the notebook ID field.

    Args:
        relative_path: Project-relative notebook path used as ID namespace.
        notebook: Parsed notebook document to update in place.
    """

    for index, cell in enumerate(notebook["cells"]):
        if "id" in cell:
            continue
        token = f"{relative_path}:{index}".encode("utf-8")
        cell["id"] = f"cell-{hashlib.sha256(token).hexdigest()[:12]}"


def _clear_outputs(notebook: dict[str, Any]) -> None:
    """Remove persisted runtime output from every code cell.

    Args:
        notebook: Parsed notebook document to update in place.
    """

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def _update_import_notebook(relative_path: str, language: str) -> None:
    """Point a V2 import notebook at stable public evidence.

    Args:
        relative_path: Project-relative notebook path.
        language: Narrative language, either ``pt-BR`` or ``en-US``.
    """

    notebook = _load(relative_path)
    _clear_outputs(notebook)
    if language == "pt-BR":
        notebook["cells"][0]["source"] = _source(
            """# V2-D1: Importação dos artefatos do Kaggle

Este notebook é um auxiliar de validação, somente texto: lê o resultado
agregado V2-D1 importado do Kaggle e seu manifesto público. Não há gráficos,
dataframes ou modelagem; apenas renderização da evidência já publicada.

As entradas estáveis são:

- `temp/v2/v2_classical_benchmark.json`: resultado completo do benchmark;
- `config/v2_classical_results.json`: manifesto público com as assinaturas da
  execução.

Logs brutos e diretórios transitórios do Kaggle não fazem parte da publicação.
"""
        )
        notebook["cells"][5]["source"] = _source(
            """## V2.1-D2: Importação do desafio do Transformer

Esta seção exibe o resultado do desafio D2 (ADR-012), no qual um
`distilbert-base-uncased` compacto disputou com o vencedor clássico V2.1-D1.
Ela lê `temp/v2/v2_transformer_challenge.json` e o manifesto
`config/v2_transformer_results.json`.

A tabela por semente marca com `*` a semente reportada, aquela que atingiu a
mediana das três. O bloco de decisão mostra cada condição pré-registrada como
PASS/FAIL. `CLASSICAL_WINNER_STANDS` é um resultado válido: nenhuma semente do
Transformer deslocou o vencedor clássico.
"""
        )
    else:
        notebook["cells"][0]["source"] = _source(
            """# V2-D1: Kaggle artifact import

This text-only validation notebook reads the aggregate V2-D1 result imported
from Kaggle and its public manifest. It contains no plots, dataframes, or
modeling; it only renders already-published evidence.

The stable inputs are:

- `temp/v2/v2_classical_benchmark.json`: complete benchmark result;
- `config/v2_classical_results.json`: public execution manifest and hashes.

Raw logs and transient Kaggle directories are not part of the publication.
"""
        )
        notebook["cells"][5]["source"] = _source(
            """## V2.1-D2: Transformer challenge import

This section displays the D2 challenge result (ADR-012), in which a compact
`distilbert-base-uncased` challenged the V2.1-D1 classical winner. It reads
`temp/v2/v2_transformer_challenge.json` and the public manifest at
`config/v2_transformer_results.json`.

The per-seed table marks with `*` the seed that attained the median of the
three. The decision block shows every pre-registered condition as PASS/FAIL.
`CLASSICAL_WINNER_STANDS` is a valid result: no Transformer seed displaced the
classical winner.
"""
        )
    notebook["cells"][2]["source"] = _source(
        """D1_RESULT_PATH = PROJECT_ROOT / 'temp' / 'v2' / 'v2_classical_benchmark.json'
D1_MANIFEST_PATH = PROJECT_ROOT / 'config' / 'v2_classical_results.json'
"""
    )
    notebook["cells"][3]["source"] = _source(
        """print(render_kaggle_import_report(
    D1_RESULT_PATH,
    manifest_path=D1_MANIFEST_PATH,
))
"""
    )
    notebook["cells"][6]["source"] = _source(
        """from consumer_complaint_intelligence.v2_import import (
    render_d2_import_report,
)

D2_RESULT_PATH = PROJECT_ROOT / 'temp' / 'v2' / 'v2_transformer_challenge.json'
D2_MANIFEST_PATH = PROJECT_ROOT / 'config' / 'v2_transformer_results.json'

print(render_d2_import_report(
    D2_RESULT_PATH,
    manifest_path=D2_MANIFEST_PATH,
))
"""
    )
    _normalize_cell_ids(relative_path, notebook)
    _write(relative_path, notebook)


def _update_package_notebook(relative_path: str, language: str) -> None:
    """Point a package notebook at stable public freeze evidence.

    Args:
        relative_path: Project-relative notebook path.
        language: Narrative language, either ``pt-BR`` or ``en-US``.
    """

    notebook = _load(relative_path)
    _clear_outputs(notebook)
    if language == "pt-BR":
        notebook["cells"][0]["source"] = _source(
            """# V2.1-P: Importação do pacote congelado (passo 6)

Este notebook é um auxiliar de validação, somente texto. Ele lê o resultado
agregado do congelamento em `temp/v2/v2_package.json` e seu manifesto público
em `config/v2_results.json`. Não há gráficos, dataframes ou modelagem.

O passo 6, regido por `docs/ADR-013-v2-frozen-package.md`, congela o pacote
hierárquico V2 completo: detector crítico, limiar calibrado, referência ao S7
congelado e regra de combinação. Logs brutos e diretórios transitórios do
Kaggle não fazem parte da publicação.
"""
        )
        identity_text = (
            "A comparação é exata nas verificações agregadas. Ela comprova "
            "reprodução comportamental nas medidas registradas, mas não prova "
            "identidade linha a linha do pool, pois o D1 não persistiu uma "
            "assinatura equivalente."
        )
    else:
        notebook["cells"][0]["source"] = _source(
            """# V2.1-P: Frozen package import (step 6)

This text-only validation notebook reads the aggregate freeze result from
`temp/v2/v2_package.json` and its public manifest at `config/v2_results.json`.
It contains no plots, dataframes, or modeling.

Step 6, governed by `docs/ADR-013-v2-frozen-package.md`, freezes the complete
V2 hierarchical package: critical detector, calibrated threshold, reference to
the frozen S7 package, and combination rule. Raw logs and transient Kaggle
directories are not part of the publication.
"""
        )
        identity_text = (
            "The comparison is exact on the aggregate checks. It demonstrates "
            "behavioral reproduction on the recorded measures, but it does not "
            "prove row-level pool identity because D1 persisted no equivalent "
            "signature."
        )
    cell_one = "".join(notebook["cells"][1]["source"])
    marker = (
        "A comparação é exata e não por tolerância porque a pergunta não é se\n"
        "dois números estão próximos. É se o objeto congelado é o mesmo objeto\n"
        "que foi medido."
        if language == "pt-BR"
        else
        "The comparison is exact rather than tolerant because the question is\n"
        "not whether two numbers are close. It is whether the frozen object is\n"
        "the same object that was measured."
    )
    if marker not in cell_one:
        raise ValueError(f"Expected identity text not found in {relative_path}")
    notebook["cells"][1]["source"] = _source(
        cell_one.replace(marker, identity_text)
    )
    notebook["cells"][3]["source"] = _source(
        """PACKAGE_RESULT_PATH = PROJECT_ROOT / 'temp' / 'v2' / 'v2_package.json'
PACKAGE_MANIFEST_PATH = PROJECT_ROOT / 'config' / 'v2_results.json'
"""
    )
    notebook["cells"][4]["source"] = _source(
        """print(render_package_import_report(
    PACKAGE_RESULT_PATH,
    manifest_path=PACKAGE_MANIFEST_PATH,
))
"""
    )
    _normalize_cell_ids(relative_path, notebook)
    _write(relative_path, notebook)


def prepare_notebooks() -> None:
    """Apply the complete publication preparation policy."""

    output_free = (
        "notebooks/01_CFPB_Data_Inspection_PT.ipynb",
        "notebooks/pt-BR/01_Data_Inspection_PT.ipynb",
        "notebooks/en-US/01_Data_Inspection_EN.ipynb",
    )
    for relative_path in output_free:
        notebook = _load(relative_path)
        _clear_outputs(notebook)
        _normalize_cell_ids(relative_path, notebook)
        _write(relative_path, notebook)

    _update_import_notebook(
        "notebooks/pt-BR/11_V2_Kaggle_Import_PT.ipynb",
        "pt-BR",
    )
    _update_import_notebook(
        "notebooks/en-US/11_V2_Kaggle_Import_EN.ipynb",
        "en-US",
    )
    _update_package_notebook(
        "notebooks/pt-BR/12_V2_Frozen_Package_PT.ipynb",
        "pt-BR",
    )
    _update_package_notebook(
        "notebooks/en-US/12_V2_Frozen_Package_EN.ipynb",
        "en-US",
    )

    for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        notebook = _load(relative_path)
        _normalize_cell_ids(relative_path, notebook)
        _write(relative_path, notebook)


if __name__ == "__main__":
    prepare_notebooks()
