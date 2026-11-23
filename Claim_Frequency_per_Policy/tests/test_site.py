"""O relatorio executivo (site/) traz os numeros oficiais e links validos."""

import csv
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
STAGE5 = ROOT / "outputs" / "etapa_05"
PAGES = {"pt": SITE / "index.html", "en": SITE / "en.html"}
MINUS = "−"

pytestmark = pytest.mark.skipif(
    not (SITE / "index.html").exists() or not STAGE5.is_dir(),
    reason="site/ ou outputs/etapa_05 ausente",
)


def fmt(value, decimals, lang, signed=False):
    """Formata como nos textos: PT '1.234,56' e EN '1,234.56'; sinal U+2212."""
    text = f"{abs(value):,.{decimals}f}"
    if lang == "pt":
        text = text.replace(",", "#").replace(".", ",").replace("#", ".")
    if value < 0:
        return MINUS + text
    return ("+" + text) if signed else text


def read_csv(name):
    with (STAGE5 / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def official_values():
    """Numeros oficiais lidos dos agregados gravados (nada fixo no codigo)."""
    metrics = {row["candidate"]: row for row in read_csv("test_metrics.csv")
               if row["subset"] == "all_rows"}
    conclusion = json.loads(
        (STAGE5 / "conclusion.json").read_text(encoding="utf-8"))
    primary = conclusion["primary"]
    band = next(row for row in read_csv("panel.csv")
                if row["candidate"] == "poisson"
                and row["family"] == "exposure_band"
                and row["segment"] == "(0, 0.02]")
    return {
        "records": (int(metrics["poisson"]["rows"]), 0),
        "dev_poisson": (float(metrics["poisson"]["mean_poisson_deviance"]), 5),
        "dev_global": (float(metrics["global"]["mean_poisson_deviance"]), 5),
        "diff": (primary["d_dev_poisson_minus_global"], 5),
        "ci_low": (primary["ci_low"], 5),
        "ci_high": (primary["ci_high"], 5),
        "rate_obs": (float(metrics["poisson"]["observed_rate_per_100"]), 3),
        "rate_poisson": (
            float(metrics["poisson"]["predicted_rate_per_100"]), 3),
        "rate_global": (
            float(metrics["global"]["predicted_rate_per_100"]), 3),
        "short_obs": (float(band["observed_rate_per_100"]), 2),
        "short_pred": (float(band["predicted_rate_per_100"]), 2),
    }


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_site_carries_the_official_numbers(lang):
    text = PAGES[lang].read_text(encoding="utf-8")
    for key, (value, decimals) in official_values().items():
        assert fmt(value, decimals, lang) in text, (key, lang)


def test_official_values_match_the_reported_ones():
    # Valores publicados nos READMEs e relatorios (conferencia independente).
    values = official_values()
    expected = {"records": 101859, "dev_poisson": 0.31902,
                "dev_global": 0.33079, "diff": -0.01177,
                "ci_low": -0.01317, "ci_high": -0.01018,
                "rate_obs": 10.058, "rate_poisson": 10.108,
                "rate_global": 10.084, "short_obs": 255.51,
                "short_pred": 14.04}
    for key, want in expected.items():
        value, decimals = values[key]
        assert round(value, decimals) == want, key


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.urls.append(value)


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_site_local_links_exist(lang):
    parser = _Links()
    parser.feed(PAGES[lang].read_text(encoding="utf-8"))
    local = [u for u in parser.urls
             if not urlparse(u).scheme and not u.startswith(("#", "//"))]
    assert local
    for url in local:
        target = (PAGES[lang].parent / unquote(urlparse(url).path)).resolve()
        assert target.exists(), url
