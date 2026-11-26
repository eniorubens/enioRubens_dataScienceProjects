"""The references module mirrors docs/REFERENCIAS.md and renders a list-free references section."""

import re

import pytest

from retail_multilevel.paths import ROOT
from retail_multilevel.references import REFERENCES, cite, render_references

DOC = ROOT / "docs" / "REFERENCIAS.md"
LIST_MARKER = re.compile(r"^\s*([-*+]|\d+\.)\s", re.MULTILINE)


def doc_table() -> dict[str, tuple[str, str]]:
    """Parse the Markdown table of the vetted list: key -> (entry, verification cell).

    ``docs/`` is internal and is not published with the portfolio copy, so the checks
    against the document skip when it is absent; the module itself is still tested.
    """
    if not DOC.exists():
        pytest.skip("docs/REFERENCIAS.md is internal and not published")
    rows = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\w+\d{4}) \| (.*?) \| (.*?) \| (.*?) \|\s*$", line)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(4))
    return rows


def test_every_key_of_the_doc_is_present_and_nothing_else():
    assert set(doc_table()) == set(REFERENCES)
    assert len(REFERENCES) == 28


def test_entries_match_the_doc_exactly():
    for key, (entry, _) in doc_table().items():
        assert REFERENCES[key]["entry"] == entry, key


def test_status_follows_the_verification_column():
    expected = {"DOI": "doi", "pág": "page", "par": "partial"}
    for key, (_, verification) in doc_table().items():
        assert REFERENCES[key]["status"] == expected[verification[:3]], key


def test_inline_cites_are_unique_and_use_ampersand():
    cites = [cite(k, "pt") for k in REFERENCES]
    assert len(set(cites)) == len(cites)
    for key in REFERENCES:
        assert cite(key, "pt") == cite(key, "en")
        assert " and " not in cite(key, "en") and " e " not in cite(key, "pt")
        assert re.search(r", \d{4}$", cite(key, "pt")), key


@pytest.mark.parametrize("lang, heading", [("pt", "## Referências"), ("en", "## References")])
def test_render_has_heading_paragraphs_and_no_list_markers(lang, heading):
    text = render_references(list(REFERENCES), lang)
    assert text.startswith(heading + "\n\n")
    assert not LIST_MARKER.search(text)
    paragraphs = text.split("\n\n")
    assert len(paragraphs) == len(REFERENCES) + 1


def test_render_is_sorted_by_first_author_and_deduplicates():
    text = render_references(["wooldridge2010", "akiba2019", "gelman2007", "akiba2019"], "en")
    body = text.split("\n\n")[1:]
    assert [p.split(",")[0] for p in body] == ["Akiba", "Gelman", "Wooldridge"]


def test_partial_entries_carry_a_note_and_others_do_not():
    pt = render_references(["raudenbush2002", "holm1979", "akiba2019"], "pt").split("\n\n")[1:]
    en = render_references(["raudenbush2002", "holm1979", "akiba2019"], "en").split("\n\n")[1:]
    assert "verificação parcial" not in pt[0] and "partial verification" not in en[0]
    assert pt[1].endswith("(verificação parcial: fonte não acessível para conferência automática)")
    assert en[1].endswith("(partial verification: source not accessible for automated checking)")
    assert pt[2].endswith("(verificação parcial: ano da edição não conferido em fonte primária)")
    assert en[2].endswith("(partial verification: edition year not checked against a primary source)")


def test_unknown_language_is_rejected():
    with pytest.raises(ValueError):
        render_references(["akiba2019"], "fr")


@pytest.fixture(scope="module")
def builder():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("build_notebooks_refs", ROOT / "scripts" / "build_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_each_notebook_ends_with_references_matching_its_inline_citations(builder, lang):
    for spec in builder.NOTEBOOKS:
        mds = [getattr(c, lang) for c in spec.cells if c.kind == "md"]
        body, last = mds[:-1], mds[-1]
        assert last.startswith("## Referências" if lang == "pt" else "## References")
        cited = builder.check_citations(body, lang)
        listed = {k for k, ref in REFERENCES.items() if ref["entry"] in last}
        assert cited == listed and cited, spec.number
        assert not LIST_MARKER.search(last)


def test_notebook_prose_has_no_list_markers_outside_tables_and_no_course_words(builder):
    banned = re.compile(r"curso|course|MBA|\baula|lesson", re.IGNORECASE)
    for spec in builder.NOTEBOOKS:
        for cell in spec.cells:
            if cell.kind != "md":
                continue
            for text in (cell.pt, cell.en):
                prose = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("|"))
                assert not LIST_MARKER.search(prose), spec.number
                assert not banned.search(text), spec.number
