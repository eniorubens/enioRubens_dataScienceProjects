"""Tests for scripts/build_site.py (number formatting, manifest hashing, page structure, PT/EN parity)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_site.py"
_spec = importlib.util.spec_from_file_location("build_site", SCRIPT)
build_site = importlib.util.module_from_spec(_spec)
sys.modules["build_site"] = build_site  # needed by @dataclass under postponed annotations
_spec.loader.exec_module(build_site)


@pytest.fixture
def tmp_path():
    """Local override: the default pytest temp root is not writable in this Windows setup."""
    with tempfile.TemporaryDirectory(prefix="build_site_") as directory:
        yield Path(directory)


def without_scripts(document: str) -> str:
    return re.sub(r"<script.*?</script>", "", document, flags=re.S)


# ---------------------------------------------------------------- formatting


def test_number_format_uses_decimal_comma_and_thousand_point_in_portuguese():
    assert build_site.fmt_number(96470, 0, "pt") == "96.470"
    assert build_site.fmt_number(0.17499, 3, "pt") == "0,175"
    assert build_site.fmt_number(1234567.891, 2, "pt") == "1.234.567,89"


def test_number_format_uses_decimal_point_and_thousand_comma_in_english():
    assert build_site.fmt_number(96470, 0, "en") == "96,470"
    assert build_site.fmt_number(0.17499, 3, "en") == "0.175"
    assert build_site.fmt_number(1234567.891, 2, "en") == "1,234,567.89"


def test_signed_numbers_use_the_true_minus_and_never_show_negative_zero():
    assert build_site.fmt_number(-0.002267, 3, "pt", signed=True) == "−0,002"
    assert build_site.fmt_number(0.0639, 3, "en", signed=True) == "+0.064"
    assert build_site.fmt_number(-0.0001, 3, "pt") == "0,000"
    assert build_site.fmt_number(float("nan")) == build_site.DASH
    assert build_site.fmt_number(None) == build_site.DASH


def test_percentage_range_month_and_date_formats():
    assert build_site.fmt_pct(0.139115, 1, "pt", signed=True) == "+13,9%"
    assert build_site.fmt_pct(0.139115, 1, "en") == "13.9%"
    assert build_site.fmt_range(-0.0325, 0.064, 3, "pt", True) == "−0,033 a +0,064"
    assert build_site.fmt_range(0.056, 0.134, 3, "en") == "0.056 to 0.134"
    assert build_site.fmt_month("2017-10", "pt") == "outubro de 2017"
    assert build_site.fmt_month("2017-10", "en") == "October 2017"
    assert build_site.fmt_date("2018-06-01", "pt") == "1º de junho de 2018"
    assert build_site.fmt_date("2018-08-14", "en") == "August 14, 2018"


# ---------------------------------------------------------------- manifest hashing


def test_hash_sources_hashes_files_and_directories_in_sorted_order(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.csv").write_bytes(b"x")
    (tmp_path / "a" / "y.csv").write_bytes(b"y")
    (tmp_path / "b.txt").write_bytes(b"")
    hashes = build_site.hash_sources(tmp_path, ["b.txt", "a"])
    assert list(hashes) == ["a/x.csv", "a/y.csv", "b.txt"]
    assert hashes["a/x.csv"] == "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"
    assert hashes["b.txt"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    with pytest.raises(FileNotFoundError):
        build_site.hash_sources(tmp_path, ["missing.csv"])


def test_references_are_selected_by_citation_and_rendered_as_html():
    entries = [
        "Olist (2018). *Brazilian E-Commerce Public Dataset by Olist*. Kaggle. https://example.org/x",
        "Wilson, E. B. (1927). Probable inference & more.",
    ]
    assert build_site.select_references(entries, ["wilson"]) == [entries[1]]
    rendered = build_site.reference_html(entries[0])
    assert "<em>Brazilian E-Commerce Public Dataset by Olist</em>" in rendered
    assert '<a href="https://example.org/x">' in rendered
    assert "&amp;" in build_site.reference_html(entries[1])
    with pytest.raises(build_site.MissingInputs):
        build_site.select_references(entries, ["nosek"])


def test_facts_record_only_the_citations_a_template_uses():
    facts = build_site.Facts({"n": "1"}, "pt")
    assert build_site.render("{n} {c_wilson}", facts) == "1 (Wilson, 1927)"
    assert facts.cited == {"wilson"}
    assert build_site.render("{c_klein}", build_site.Facts({}, "en")) == "(Klein and Moeschberger, 2003)"
    with pytest.raises(KeyError):
        build_site.render("{missing}", facts)


def test_narrative_check_rejects_results_that_contradict_the_text():
    from_pd = pytest.importorskip("pandas")
    ap = from_pd.Series({"estimate": 0.1})
    raw = {
        "candidates": ["XGBoost"], "reg_candidates": [], "outcome_en": "inconclusive", "delta_low": -0.03,
        "delta_high": 0.06, "ap_f1_rule": ap, "ap_f1_cat": from_pd.Series({"estimate": 0.08}),
        "ap_f1_log": from_pd.Series({"estimate": 0.08}), "ap2": {"A": 0.06}, "ap2_log": from_pd.Series({"estimate": 0.09}),
        "vs_log_ci_excludes_zero": True, "rj_km": 1.0, "mg_km": 2.0, "rj_rate": 0.2, "mg_rate": 0.1,
        "prod_eligible": False, "ablation_adopted": False,
        "label_jul": {"prevalence_unknown": 0.1, "prevalence_known": 0.01},
    }
    with pytest.raises(build_site.NarrativeMismatch, match="only classification candidate"):
        build_site.check_narrative(raw)
    raw["candidates"] = ["CatBoost"]
    build_site.check_narrative(raw)


# ---------------------------------------------------------------- templates (no stored data needed)


def test_templates_have_the_same_keys_and_no_bullet_markup_in_both_languages():
    assert set(build_site.TEXT["pt"]) == set(build_site.TEXT["en"])
    assert set(build_site.COLS["pt"]) == set(build_site.COLS["en"])
    assert set(build_site.TEXT["pt"]["nav"]) == set(build_site.TEXT["en"]["nav"]) == set(build_site.SECTION_IDS)
    for lang in ("pt", "en"):
        joined = json.dumps(build_site.TEXT[lang])
        assert not re.search(r"<(ul|ol|li)\b", joined)


def test_css_keeps_the_reference_look_tokens():
    css = build_site.CSS
    for token in (".section-no", ".insight", ".stats", ".downloads", ".table-scroll", "@media print", "--blue:#2563eb"):
        assert token in css


def test_main_exits_with_one_when_stored_results_are_missing(tmp_path):
    assert build_site.inputs_available(tmp_path) is False
    assert build_site.main(["--root", str(tmp_path), "--output", str(tmp_path / "out")]) == 1


# ---------------------------------------------------------------- full build (skipped without stored results)

pytestmark_full = pytest.mark.skipif(
    not build_site.inputs_available(ROOT) or importlib.util.find_spec("plotly") is None,
    reason="outputs/runs or data/raw/olist/csv is missing",
)


@pytest.fixture(scope="module")
def built_site():
    with tempfile.TemporaryDirectory(prefix="build_site_full_") as directory:
        ledger = (ROOT / "outputs/runs/final_evaluation/LEDGER.jsonl").read_bytes()
        output = build_site.build_site(ROOT, Path(directory))
        assert (ROOT / "outputs/runs/final_evaluation/LEDGER.jsonl").read_bytes() == ledger
        yield output


@pytestmark_full
def test_full_build_writes_pages_assets_and_manifest(built_site):
    for name in ("index.html", "en.html", "assets/report.css", "assets/plotly.min.js", "assets/build-manifest.json",
                 "assets/README.md", "assets/chart-route-edges.csv", "assets/chart-f1-metrics.csv"):
        assert (built_site / name).is_file(), name
    manifest = json.loads((built_site / "assets/build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_performed"] is False
    assert manifest["test_reopened"] is False
    assert manifest["production_eligible"] is False
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in manifest["sources"].values())
    assert "outputs/runs/final_evaluation/LEDGER.jsonl" in manifest["sources"]
    assert "data/raw/olist/csv/olist_orders_dataset.csv" in manifest["sources"]
    assert "assets/plotly.min.js" in manifest["generated"]
    assert manifest["dataset_provenance"]["repository"].startswith("https://www.kaggle.com/")


@pytestmark_full
def test_pages_have_the_same_sections_navigation_and_charts_in_both_languages(built_site):
    pages = {lang: (built_site / name).read_text(encoding="utf-8") for lang, name in (("pt", "index.html"), ("en", "en.html"))}
    assert 'lang="pt-BR"' in pages["pt"] and 'lang="en"' in pages["en"]
    ids = {}
    for lang, page in pages.items():
        body = without_scripts(page)
        ids[lang] = re.findall(r'<section id="([^"]+)"', body)
        nav = re.search(r"<nav[^>]*>(.*?)</nav>", body, flags=re.S).group(1)
        assert re.findall(r'href="#([^"]+)"', nav) == build_site.SECTION_IDS
        assert 'class="skip"' in body and "<noscript>" in body
        assert 'hreflang="pt-BR" href="index.html"' in body and 'hreflang="en" href="en.html"' in body
        assert 'href="#top"' in body and 'id="top"' in body
        assert body.count('class="section-no"') == 6 and body.count("<details>") >= 7
        assert 'class="insight"' in body and "<h1>" in body and 'class="stats"' in body
        assert not re.search(r"<(ul|ol|li)\b", body), "bullet lists are not allowed"
        assert 'src="assets/plotly.min.js"' in page
        assert not re.search(r'<script[^>]+src="https?://', page), "no CDN scripts"
        assert len(re.findall(r'class="plotly-graph-div"', page)) == 8
    assert ids["pt"] == ids["en"] == build_site.SECTION_IDS
    chart_ids = {lang: [i.replace(f"-{lang}-", "-") for i in re.findall(r'id="(chart-[^"]+)"', page)]
                 for lang, page in pages.items()}
    assert chart_ids["pt"] == chart_ids["en"]


@pytest.mark.skipif(not (ROOT / "docs/REGISTRO-DE-DECISOES.md").is_file(), reason="register not available")
@pytestmark_full
def test_page_numbers_match_the_decision_register(built_site):
    register = (ROOT / "docs/REGISTRO-DE-DECISOES.md").read_text(encoding="utf-8")
    pt = without_scripts((built_site / "index.html").read_text(encoding="utf-8"))
    en = without_scripts((built_site / "en.html").read_text(encoding="utf-8"))
    for pt_value, en_value in (("77.867", "77,867"), ("12.507", "12,507"), ("+13,9%", "+13.9%"), ("−0,002", "−0.002"),
                               ("0,175", "0.175"), ("0,154", "0.154"), ("0,179", "0.179"), ("6.351", "6,351"),
                               ("0,094", "0.094")):
        assert pt_value in pt and en_value in en
        assert pt_value in register, f"{pt_value} is not in the register"
    assert "96.470" in pt and "96,470" in en  # supervised orders (not stated in the register)
    assert "Inconclusivo · ΔAP −0,002" in pt and "Inconclusive · ΔAP −0.002" in en
    assert "+13,9% · 8/9" in pt and "+13.9% · 8/9" in en


@pytestmark_full
def test_references_are_only_the_cited_ones_and_copied_verbatim(built_site):
    page = (built_site / "index.html").read_text(encoding="utf-8")
    notebook = json.loads((ROOT / build_site.NOTEBOOKS["pt"]).read_text(encoding="utf-8"))
    references = "".join("".join(c["source"]) for c in notebook["cells"] if "".join(c["source"]).startswith("## Refer"))
    refs_html = page.split('<div class="refs">')[1]
    assert "Olist (2018)" in refs_html and "Akiba" not in refs_html
    assert "Wilson, E. B. (1927). Probable inference, the law of succession" in references
    assert "Wilson, E. B. (1927). Probable inference, the law of succession" in refs_html
