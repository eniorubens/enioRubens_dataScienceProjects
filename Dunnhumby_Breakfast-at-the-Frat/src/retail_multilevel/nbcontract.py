"""Structural contract of the bilingual notebooks, checked in CI without executing them.

- Every ``notebooks/pt_br/NN_*.ipynb`` has an ``notebooks/en_us/NN_*.ipynb`` (same number prefix).
- Both run identical code: code cells match exactly once the ``LANG = "..."`` line is masked.
- Each notebook sets ``LANG`` to its own language, and both have the same number of Markdown cells.
- Notebooks never fit models: training runs in scripts/, notebooks only read stored runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

LANG_LINE = re.compile(r"""^LANG\s*=\s*["'](?P<lang>en|pt)["']\s*$""", re.MULTILINE)
FORBIDDEN = ("get_or_fit(", 'mode="train"', "mode='train'")
DIR_LANG = {"pt_br": "pt", "en_us": "en"}


def _prefix(path: Path) -> str:
    return path.name.split("_", 1)[0]


def pair_notebooks(nb_root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Pair PT and EN notebooks by number prefix; return ``(pairs, problems)``."""
    pt = {_prefix(p): p for p in sorted((nb_root / "pt_br").glob("*.ipynb"))}
    en = {_prefix(p): p for p in sorted((nb_root / "en_us").glob("*.ipynb"))}
    problems = [f"PT notebook without EN pair: {pt[k].name}" for k in sorted(pt.keys() - en.keys())]
    problems += [f"EN notebook without PT pair: {en[k].name}" for k in sorted(en.keys() - pt.keys())]
    return [(pt[k], en[k]) for k in sorted(pt.keys() & en.keys())], problems


def _cells(path: Path, kind: str) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    return [c.source for c in nb.cells if c.cell_type == kind]


def _mask_lang(src: str) -> str:
    return LANG_LINE.sub('LANG = "<lang>"', src)


def check_pair(pt_path: Path, en_path: Path) -> list[str]:
    """Return the contract violations of one PT/EN notebook pair (empty list if none)."""
    problems: list[str] = []
    pt_code, en_code = _cells(pt_path, "code"), _cells(en_path, "code")
    for path, code in ((pt_path, pt_code), (en_path, en_code)):
        langs = [m.group("lang") for src in code for m in LANG_LINE.finditer(src)]
        expected = DIR_LANG.get(path.parent.name)
        if langs != [expected]:
            problems.append(f"{path.name}: expected exactly one LANG = {expected!r}, found {langs}")
        for src in code:
            for bad in FORBIDDEN:
                if bad in src:
                    problems.append(f"{path.name}: notebooks must not fit models ({bad!r} found)")
    if [_mask_lang(s) for s in pt_code] != [_mask_lang(s) for s in en_code]:
        problems.append(f"{pt_path.name} / {en_path.name}: code cells differ beyond the LANG line")
    n_md_pt, n_md_en = len(_cells(pt_path, "markdown")), len(_cells(en_path, "markdown"))
    if n_md_pt != n_md_en:
        problems.append(f"{pt_path.name} / {en_path.name}: {n_md_pt} vs {n_md_en} Markdown cells")
    return problems


def check_notebooks(nb_root: Path) -> list[str]:
    """Return every contract violation under ``nb_root`` (empty list if the contract holds)."""
    pairs, problems = pair_notebooks(nb_root)
    for pt_path, en_path in pairs:
        problems += check_pair(pt_path, en_path)
    return problems
