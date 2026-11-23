"""Validacao estrutural e de paridade dos notebooks PT-BR e EN-US.

Nao executa os notebooks. Verifica: mesma sequencia de tipos de celula,
codigo identico exceto a linha ``LANG = ...``, ausencia de logica
analitica duplicada nos notebooks, citacoes ``(Autor, ano)`` iguais nos
dois idiomas e com entrada na secao final de referencias.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LANG_LINE = re.compile(r'^LANG\s*=\s*"(pt_br|en_us)"\s*$', re.M)
CITATION = re.compile(
    r"\(([A-ZÀ-Ý][\w'’\-]+(?: et al\.| (?:&|e|and) [A-ZÀ-Ý][\w'’\-]+)?),"
    r" (\d{4})\)")
FORBIDDEN = ("read_csv", "read_arff", ".groupby(", "sm.GLM", "np.exp(",
             "minimize", "np.linalg")
REFERENCE_HEADINGS = {"pt_br": "Referências", "en_us": "References"}


def load_notebook(path: Path) -> dict:
    """Leia um ``.ipynb`` como dicionario."""
    return json.loads(Path(path).read_text("utf-8"))


def _source(cell: dict) -> str:
    """Texto-fonte de uma celula."""
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def citations(nb: dict) -> set[tuple[str, str]]:
    """Citacoes ``(Autor, ano)`` presentes nas celulas markdown."""
    found = set()
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            found |= set(CITATION.findall(_source(cell)))
    return found


def reference_section(nb: dict) -> str:
    """Texto da ultima celula (secao de referencias)."""
    return _source(nb["cells"][-1])


def validate_structure(nb: dict, language: str) -> list[str]:
    """Problemas estruturais de um notebook (lista vazia = ok)."""
    problems = []
    cells = nb["cells"]
    if not cells or cells[0]["cell_type"] != "markdown":
        problems.append("a primeira celula deve ser markdown")
    last = cells[-1] if cells else None
    heading = REFERENCE_HEADINGS[language]
    if (not last or last["cell_type"] != "markdown"
            or heading not in _source(last).splitlines()[0]):
        problems.append(f"a ultima celula deve ser '{heading}'")
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        code = _source(cell)
        for token in FORBIDDEN:
            if token in code:
                problems.append(f"celula {i}: logica proibida '{token}'")
    langs = [m for c in cells if c["cell_type"] == "code"
             for m in LANG_LINE.findall(_source(c))]
    if langs != [language]:
        problems.append(f"LANG deve aparecer uma vez como {language}")
    refs = reference_section(nb)
    for author, year in citations(nb):
        surname = author.split()[0].rstrip(",")
        if surname not in refs or year not in refs:
            problems.append(f"citacao sem referencia: ({author}, {year})")
    return problems


def validate_pair(pt: dict, en: dict) -> list[str]:
    """Problemas de paridade entre as duas versoes (lista vazia = ok)."""
    problems = validate_structure(pt, "pt_br") + [
        f"[en] {p}" for p in validate_structure(en, "en_us")]
    a, b = pt["cells"], en["cells"]
    if [c["cell_type"] for c in a] != [c["cell_type"] for c in b]:
        problems.append("sequencia de tipos de celula difere")
        return problems
    for i, (ca, cb) in enumerate(zip(a, b, strict=True)):
        if ca["cell_type"] != "code":
            continue
        sa = LANG_LINE.sub("LANG", _source(ca))
        sb = LANG_LINE.sub("LANG", _source(cb))
        if sa != sb:
            problems.append(f"celula de codigo {i} difere entre idiomas")
    if citations(pt) != citations(en):
        problems.append("citacoes diferem entre idiomas")
    return problems


def validate_directory(root: Path) -> dict[str, list[str]]:
    """Valide todos os pares ``notebooks/pt_br`` x ``notebooks/en_us``.

    Args:
        root: raiz do projeto.

    Returns:
        Problemas por nome de arquivo; inclui nomes sem par.
    """
    pt_dir = Path(root) / "notebooks" / "pt_br"
    en_dir = Path(root) / "notebooks" / "en_us"
    names = sorted({p.name for p in pt_dir.glob("*.ipynb")}
                   | {p.name for p in en_dir.glob("*.ipynb")})
    out = {}
    for name in names:
        if not (pt_dir / name).exists() or not (en_dir / name).exists():
            out[name] = ["sem par no outro idioma"]
            continue
        out[name] = validate_pair(load_notebook(pt_dir / name),
                                  load_notebook(en_dir / name))
    return out


def execution_status(nb: dict) -> dict:
    """Resumo da execucao persistida: celulas de codigo e erros."""
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    executed = [c for c in code if c.get("execution_count") is not None]
    errors = [o for c in code for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    return {"code_cells": len(code), "executed": len(executed),
            "errors": len(errors)}
