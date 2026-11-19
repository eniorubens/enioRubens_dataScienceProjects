"""Author-year citations in the notebooks agree with the curated list in docs/referencias.md."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF_FILE = ROOT / "docs" / "referencias.md"
PT = sorted((ROOT / "notebooks" / "pt_br").glob("*.ipynb"))
EN = sorted((ROOT / "notebooks" / "en_us").glob("*.ipynb"))
YEAR = r"(?:\d{4}[ab]?|s\.d\.)"
# "Name, 1999" | "A & B, 1999" | "A et al., 1999"
CITATION = re.compile(rf"[A-ZÀ-Ý][\wÀ-ÿ\-]+(?: & [A-ZÀ-Ý][\wÀ-ÿ\-]+| et al\.)?, {YEAR}")
HEADINGS = {"pt": "## Referências", "en": "## References"}

pytestmark = pytest.mark.skipif(not (PT and EN and REF_FILE.exists()), reason="no notebooks or reference list")


def _reference_data():
    text = REF_FILE.read_text(encoding="utf-8")
    table = text.split("## Mapa")[1].split("## Entradas")[0]
    keys = []
    for key in re.findall(rf"\(([^()]+?, {YEAR})\)", table):
        if key not in keys:
            keys.append(key)
    entries = [line[2:].strip() for line in text.split("## Entradas")[1].splitlines() if line.startswith("- ")]
    return keys, entries


KEYS, ENTRIES = _reference_data()


def _entry_for(key):
    surname = key.split(",")[0].split(" ")[0]
    year = key.split(", ")[-1]
    found = [e for e in ENTRIES if re.match(re.escape(surname) + r"[.,]", e) and f"({year})" in e]
    assert len(found) == 1, (key, found)
    return found[0]


def _cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _src(cell):
    return "".join(cell["source"])


def _split(path):
    """Return (text of all markdown cells but the last, last cell)."""
    cells = _cells(path)
    body = "\n".join(_src(c) for c in cells[:-1] if c["cell_type"] == "markdown")
    return body, cells[-1]


def _lang(path):
    return "pt" if path.parent.name == "pt_br" else "en"


def _cited(path):
    body, _ = _split(path)
    return set(CITATION.findall(body))


def test_reference_list_is_consistent():
    assert len(KEYS) == len(ENTRIES)
    assert [_entry_for(k) for k in KEYS]  # each mapped citation has exactly one entry
    assert len({_entry_for(k) for k in KEYS}) == len(KEYS)


@pytest.mark.parametrize("path", PT + EN, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_citations_exist_in_reference_list(path):
    unknown = _cited(path) - set(KEYS)
    assert not unknown, f"{path.name}: not in docs/referencias.md: {sorted(unknown)}"


@pytest.mark.parametrize("path", PT + EN, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_references_cell_is_last_and_matches_citations(path):
    body, last = _split(path)
    assert last["cell_type"] == "markdown"
    text = _src(last)
    assert text.startswith(HEADINGS[_lang(path)] + "\n"), path.name
    assert sum(_src(c).lstrip().startswith(HEADINGS[_lang(path)]) for c in _cells(path)) == 1
    listed = [line[2:].strip() for line in text.splitlines() if line.startswith("- ")]
    cited = _cited(path)
    assert cited, f"{path.name}: no citations"
    expected = sorted((_entry_for(k) for k in cited), key=ENTRIES.index)
    assert listed == expected, f"{path.name}: references cell differs from the works cited in the text"
    assert all(e in ENTRIES for e in listed)


def test_pt_and_en_cite_the_same_works():
    assert len(PT) == len(EN)
    for pt, en in zip(PT, EN, strict=True):
        assert _cited(pt) == _cited(en), (pt.name, en.name)
