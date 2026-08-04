"""Tests for the provider-neutral notebook translation workflow."""

from __future__ import annotations

import json

import nbformat
import pytest

from scripts.notebook_translation import (
    apply_review_corrections,
    audit_notebook_pair,
    clone_notebook,
    export_markdown_bundle,
    import_markdown_bundle,
)


def _canonical_notebook():
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# Título\n\nTexto em português."),
        nbformat.v4.new_code_cell("from src.i18n import make_lang\nlang = make_lang('pt')"),
    ]
    return notebook


def test_clone_changes_only_language_and_clears_outputs(tmp_path):
    source = tmp_path / "pt-BR" / "sample.ipynb"
    target = tmp_path / "en-US" / "sample.ipynb"
    source.parent.mkdir()
    notebook = _canonical_notebook()
    notebook.cells[1].outputs = [nbformat.v4.new_output("stream", text="Português")]
    notebook.cells[1].execution_count = 1
    nbformat.write(notebook, source)

    clone_notebook(source, target)
    translated = nbformat.read(target, as_version=4)

    assert "make_lang('en')" in translated.cells[1].source
    assert translated.cells[1].outputs == []
    assert translated.cells[1].execution_count is None
    assert translated.metadata["project_language"] == "en-US"
    assert audit_notebook_pair(source, target) == []


def test_export_and_import_preserve_cell_identity(tmp_path):
    source_dir = tmp_path / "pt-BR"
    target_dir = tmp_path / "en-US"
    source_dir.mkdir()
    source = source_dir / "sample.ipynb"
    target = target_dir / "sample.ipynb"
    nbformat.write(_canonical_notebook(), source)
    clone_notebook(source, target)

    bundle_path = tmp_path / "bundle.json"
    bundle = export_markdown_bundle((source,), bundle_path)
    bundle["entries"][0]["translation"] = "# Title\n\nText in English."
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")

    assert import_markdown_bundle(bundle_path, source_dir, target_dir) == 1
    translated = nbformat.read(target, as_version=4)
    assert translated.cells[0].source == "# Title\n\nText in English."
    assert audit_notebook_pair(source, target) == []


def test_import_stops_when_canonical_markdown_changed(tmp_path):
    source_dir = tmp_path / "pt-BR"
    target_dir = tmp_path / "en-US"
    source_dir.mkdir()
    source = source_dir / "sample.ipynb"
    target = target_dir / "sample.ipynb"
    notebook = _canonical_notebook()
    nbformat.write(notebook, source)
    clone_notebook(source, target)

    bundle_path = tmp_path / "bundle.json"
    bundle = export_markdown_bundle((source,), bundle_path)
    bundle["entries"][0]["translation"] = "# Title"
    notebook.cells[0].source = "# Texto alterado"
    nbformat.write(notebook, source)
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Canonical Markdown changed"):
        import_markdown_bundle(bundle_path, source_dir, target_dir)


def test_audit_detects_executable_drift(tmp_path):
    source = tmp_path / "source.ipynb"
    target = tmp_path / "target.ipynb"
    notebook = _canonical_notebook()
    nbformat.write(notebook, source)
    clone_notebook(source, target)

    translated = nbformat.read(target, as_version=4)
    translated.cells[1].source += "\nvalue = 42"
    nbformat.write(translated, target)

    assert "executable code differs beyond make_lang target" in audit_notebook_pair(source, target)


def test_apply_review_corrections_requires_explicit_approval():
    bundle = {
        "entries": [
            {"cell_index": 0, "translation": "Machine draft"},
            {"cell_index": 2, "translation": "Second draft"},
        ]
    }

    reviewed = apply_review_corrections(
        bundle,
        {"approve_all": True, "corrections": {"2": "Reviewed translation"}},
    )

    assert reviewed["entries"][0]["translation_status"] == "reviewed"
    assert reviewed["entries"][1]["translation"] == "Reviewed translation"
    assert reviewed["human_review"]["correction_count"] == 1


def test_apply_review_corrections_rejects_incomplete_bundle():
    bundle = {"entries": [{"cell_index": 0, "translation": None}]}

    with pytest.raises(ValueError, match="missing translations"):
        apply_review_corrections(bundle, {"approve_all": True, "corrections": {}})
