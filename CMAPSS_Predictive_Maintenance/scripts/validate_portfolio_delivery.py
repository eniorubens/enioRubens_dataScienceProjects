"""Verifica arquivos, links locais, idiomas e fontes do relatório bilíngue."""
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.remote_assets = []
        self.ids = []
        self.lang = None
        self.charts = 0
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "html":
            self.lang = attrs.get("lang")
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "div" and "plotly-graph-div" in attrs.get("class", ""):
            self.charts += 1
        if tag == "table":
            self.tables += 1
        if tag in ["a", "link", "script"] and ("href" in attrs or "src" in attrs):
            link = attrs.get("href", attrs.get("src"))
            self.links.append(link)
            if tag in ["link", "script"] and "://" in link:
                self.remote_assets.append(link)


def markdown_links(path):
    text = path.read_text(encoding="utf-8")
    # Inline code blocks are not inspected as links.
    for link in re.findall(r"\]\(([^)]+)\)", text):
        if "://" not in link and not link.startswith("#"):
            assert (path.parent / link.split("#")[0]).exists(), (path, link)


def main():
    for filename, language in [("index.html", "pt-BR"), ("en.html", "en-US")]:
        path = ROOT / "site" / filename
        page = Page()
        text = path.read_text(encoding="utf-8")
        page.feed(text)
        assert page.lang == language
        assert page.charts == 5 and page.tables == 5
        assert not page.remote_assets, page.remote_assets
        assert "National Aeronautics and Space Administration" in text and "Saxena" in text
        assert all(v == 1 for v in Counter(page.ids).values()), "Duplicate HTML identifiers"
        for link in page.links:
            if "://" in link:
                assert link in ["https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/", "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"], link
                continue
            target, _, anchor = link.partition("#")
            assert not target or (path.parent / target).exists(), link
            if target:
                assert (path.parent / target).resolve().is_relative_to(ROOT / "site"), link
            if not target and anchor:
                assert anchor in page.ids, link
        assert "#2CA02C" in text and "#D62728" in text and "#FF7F0E" in text
        assert "91.3" in text or "91,3" in text
    for name in ["README.md", "README.en.md"]:
        markdown_links(ROOT / name)
    pt = ROOT / "notebooks/pt-BR/27_conclusao_entrega_fd001.ipynb"
    en = ROOT / "notebooks/en-US/27_conclusion_delivery_fd001.ipynb"
    pt_nb, en_nb = [json.loads(p.read_text(encoding="utf-8")) for p in [pt, en]]
    assert len(pt_nb["cells"]) == len(en_nb["cells"]) == 6
    for source_path, notebook in [(pt, pt_nb), (en, en_nb)]:
        assert notebook["nbformat"] == 4 and all(c["cell_type"] == "markdown" for c in notebook["cells"])
        for cell in notebook["cells"]:
            source = cell["source"]
            source = source if isinstance(source, str) else "".join(source)
            for link in re.findall(r"\]\(([^)]+)\)", source):
                if "://" not in link:
                    assert (source_path.parent / link).exists(), link
    en_source = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in en_nb["cells"])
    for number in ["16.23", "23.80", "+10.08", "85.9%", "91.3", "46%", "76.4%"]:
        assert number in en_source, number
    manifest = json.loads((ROOT / "site/assets/build-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sources"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((ROOT / "data/raw/dataset/readme.txt").read_bytes()).hexdigest() == manifest["dataset_provenance"]["local_readme_sha256"]
    metadata = json.loads((ROOT / "outputs/models/extra_trees_fd001_v1/metadata.json").read_text(encoding="utf-8"))
    model = ROOT / "outputs/models/extra_trees_fd001_v1/model.joblib"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == metadata["sha256_model"]
    assert len(metadata["features"]) == 205 and len(metadata["selected_features"]) == 93
    workflow = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow and "path: site" in workflow
    assert "  push:" not in workflow and "path: docs" not in workflow
    assert "/docs/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert not (ROOT / "docs/index.html").exists()
    print("Validados: duas páginas, dez gráficos e tabelas, links, tradução dos números, fontes e artefato original.")
    print("Publicação restrita a site/, execução manual e exclusão de docs/ verificados.")
    print("Aparência visual não verificada por este comando.")


if __name__ == "__main__":
    main()
