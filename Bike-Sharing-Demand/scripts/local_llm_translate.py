"""Generate reviewable notebook-translation drafts with a local GGUF model.

This module is intentionally separate from ``notebook_translation``.  The
canonical import/export/audit workflow has no LLM dependency; this optional
helper only fills the ``translation`` fields of an exported JSON bundle.
Human review remains mandatory before the bundle is imported.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable


SYSTEM_PROMPT = """You are a meticulous technical translator for a data-science portfolio.
Translate Brazilian Portuguese into natural, professional US English.

Rules:
- Return only the translation, without commentary or Markdown fences.
- Preserve Markdown structure, headings, blockquotes, blank lines and emphasis.
- Preserve inline code, fenced code, LaTeX, URLs, citations and bibliographic data.
- Preserve every number, threshold, equation, variable, function and file name.
- Do not simplify, summarize, expand or reinterpret the analysis.
- Keep paper titles that are already written in English unchanged.
- Use consistent data-science and statistical terminology.
"""


def chunk_markdown(text: str, max_chars: int = 4_200) -> list[str]:
    """Split Markdown at paragraph boundaries without losing source text."""
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500")
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines(keepends=True)
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if not line.strip():
            blocks.append("".join(current))
            current = []
    if current:
        blocks.append("".join(current))

    chunks: list[str] = []
    pending = ""
    for block in blocks:
        if len(block) > max_chars:
            if pending:
                chunks.append(pending)
                pending = ""
            block_lines = block.splitlines(keepends=True)
            line_chunk = ""
            for line in block_lines:
                if line_chunk and len(line_chunk) + len(line) > max_chars:
                    chunks.append(line_chunk)
                    line_chunk = ""
                line_chunk += line
            if line_chunk:
                chunks.append(line_chunk)
            continue
        if pending and len(pending) + len(block) > max_chars:
            chunks.append(pending)
            pending = ""
        pending += block
    if pending:
        chunks.append(pending)
    return chunks


def clean_translation(text: str) -> str:
    """Remove an accidental outer Markdown fence while preserving content."""
    cleaned = text.strip()
    if cleaned.startswith("```markdown") and cleaned.endswith("```"):
        cleaned = cleaned[len("```markdown") : -3].strip()
    elif cleaned.startswith("```md") and cleaned.endswith("```"):
        cleaned = cleaned[len("```md") : -3].strip()
    return cleaned


def translate_markdown(
    text: str,
    complete: Callable[[str], str],
    max_chars: int = 4_200,
) -> str:
    """Translate one Markdown cell, chunking only at safe boundaries."""
    translations: list[str] = []
    for chunk in chunk_markdown(text, max_chars=max_chars):
        translated = clean_translation(complete(chunk))
        if not translated:
            raise RuntimeError("The local model returned an empty translation.")
        translations.append(translated)
    return "\n\n".join(translations).strip() + "\n"


def translate_bundle(
    bundle: dict[str, Any],
    complete: Callable[[str], str],
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    max_chars: int = 4_200,
) -> dict[str, Any]:
    """Fill missing bundle translations and checkpoint after every cell."""
    entries = bundle.get("entries", [])
    for position, entry in enumerate(entries, start=1):
        if isinstance(entry.get("translation"), str) and entry["translation"].strip():
            continue
        entry["translation"] = translate_markdown(
            entry["source"], complete=complete, max_chars=max_chars
        )
        entry["translation_status"] = "machine_draft"
        if checkpoint is not None:
            checkpoint(bundle)
        print(
            f"Translated {position}/{len(entries)}: " f"{entry['notebook']}:{entry['cell_index']}",
            flush=True,
        )
    return bundle


def _load_preset(preset_path: Path):
    spec = importlib.util.spec_from_file_location("local_llm_preset", preset_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load preset: {preset_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_llm


def _build_completion(llm: Any, max_tokens: int) -> Callable[[str], str]:
    def complete(markdown: str) -> str:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": markdown},
            ],
            temperature=0.0,
            seed=42,
            max_tokens=max_tokens,
        )
        return str(response["choices"][0]["message"]["content"])

    return complete


def _write_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--n-ctx", type=int, default=4_096)
    parser.add_argument("--max-tokens", type=int, default=1_800)
    parser.add_argument("--chunk-chars", type=int, default=4_200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_path = args.output if args.output.exists() else args.bundle
    bundle = json.loads(source_path.read_text(encoding="utf-8"))

    get_llm = _load_preset(args.preset)
    llm = get_llm(
        str(args.model),
        n_ctx=args.n_ctx,
        offload_kqv=False,
        seed=42,
        verbose=False,
    )
    bundle["draft_provider"] = {
        "name": "local-llama-cpp",
        "model": args.model.name,
        "temperature": 0.0,
        "human_review_required": True,
    }

    def checkpoint(updated: dict[str, Any]) -> None:
        _write_bundle(updated, args.output)

    translate_bundle(
        bundle,
        complete=_build_completion(llm, max_tokens=args.max_tokens),
        checkpoint=checkpoint,
        max_chars=args.chunk_chars,
    )
    _write_bundle(bundle, args.output)
    print(f"Draft saved to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
