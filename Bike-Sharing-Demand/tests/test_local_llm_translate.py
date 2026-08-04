"""Tests for the optional local-LLM Markdown draft generator."""

from __future__ import annotations

from scripts.local_llm_translate import (
    chunk_markdown,
    clean_translation,
    translate_bundle,
    translate_markdown,
)


def test_chunk_markdown_preserves_complete_source():
    source = "# Título\n\nPrimeiro parágrafo.\n\nSegundo parágrafo longo.\n"

    chunks = chunk_markdown(source, max_chars=500)

    assert "".join(chunks) == source


def test_clean_translation_removes_only_outer_markdown_fence():
    assert clean_translation("```markdown\n# Title\n```") == "# Title"
    assert clean_translation("Text with `code`.") == "Text with `code`."


def test_translate_markdown_joins_translated_chunks():
    source = "A" * 400 + "\n\n" + "B" * 400

    translated = translate_markdown(
        source,
        complete=lambda chunk: f"EN:{chunk}",
        max_chars=500,
    )

    assert translated.startswith("EN:")
    assert translated.count("EN:") == 2
    assert translated.endswith("\n")


def test_translate_bundle_resumes_existing_drafts_and_checkpoints():
    bundle = {
        "entries": [
            {"notebook": "sample.ipynb", "cell_index": 0, "source": "Um"},
            {
                "notebook": "sample.ipynb",
                "cell_index": 1,
                "source": "Dois",
                "translation": "Two",
            },
        ]
    }
    checkpoints = []

    result = translate_bundle(
        bundle,
        complete=lambda text: {"Um": "One"}[text],
        checkpoint=lambda value: checkpoints.append(value.copy()),
    )

    assert result["entries"][0]["translation"] == "One\n"
    assert result["entries"][0]["translation_status"] == "machine_draft"
    assert result["entries"][1]["translation"] == "Two"
    assert len(checkpoints) == 1
