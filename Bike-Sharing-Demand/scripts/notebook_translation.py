"""Safe, provider-neutral workflow for bilingual Jupyter notebooks.

The canonical PT-BR notebook remains the analytical source of truth. This
utility clones it into the EN-US edition, exports only Markdown cells for
translation, imports reviewed translations, and verifies that executable code
has not diverged beyond the explicit ``make_lang`` target.

No translation provider is imported here. An AI assistant or a future external
service may fill the exported bundle without becoming a runtime dependency of
the analytical project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"
SOURCE_DIR = NOTEBOOKS_ROOT / "pt-BR"
TARGET_DIR = NOTEBOOKS_ROOT / "en-US"
BUNDLE_SCHEMA_VERSION = 1


def _source(cell: Any) -> str:
    value = cell.get("source", "")
    if isinstance(value, list):
        return "".join(value)
    return str(value)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_notebook(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(raw.get("cells", [])):
        if cell.get("id"):
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        identity = f"{index}\0{cell.get('cell_type', '')}\0{source}"
        cell["id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
    return nbformat.from_dict(raw)


def _write_notebook(notebook: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)


def _markdown_entries(notebook: Any, notebook_name: str) -> Iterable[dict[str, Any]]:
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        source = _source(cell)
        yield {
            "notebook": notebook_name,
            "cell_index": index,
            "cell_id": cell.get("id"),
            "source_sha256": _sha256(source),
            "source": source,
            "translation": None,
        }


def clone_notebook(source_path: Path, target_path: Path) -> None:
    """Clone one canonical notebook as an unexecuted EN-US draft."""
    if target_path.exists():
        raise FileExistsError(target_path)

    notebook = _read_notebook(source_path)
    replacements = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = _source(cell)
        updated = source.replace("make_lang('pt')", "make_lang('en')")
        replacements += int(updated != source)
        cell.source = updated
        cell.outputs = []
        cell.execution_count = None

    if replacements != 1:
        raise ValueError(
            f"Expected one make_lang('pt') call in {source_path.name}; " f"found {replacements}."
        )

    notebook.metadata["project_language"] = "en-US"
    _write_notebook(notebook, target_path)


def export_markdown_bundle(notebook_paths: Iterable[Path], output_path: Path) -> dict:
    """Export translatable Markdown while retaining source identity hashes."""
    paths = tuple(sorted(notebook_paths))
    if not paths:
        raise ValueError("No canonical notebooks were selected for export.")

    entries: list[dict[str, Any]] = []
    for path in paths:
        notebook = _read_notebook(path)
        entries.extend(_markdown_entries(notebook, path.name))

    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_language": "pt-BR",
        "target_language": "en-US",
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return bundle


def import_markdown_bundle(bundle_path: Path, source_dir: Path, target_dir: Path) -> int:
    """Import reviewed translations after verifying the canonical source hash."""
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Unsupported translation bundle schema.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in bundle.get("entries", []):
        grouped.setdefault(entry["notebook"], []).append(entry)

    imported = 0
    for notebook_name, entries in grouped.items():
        source_notebook = _read_notebook(source_dir / notebook_name)
        target_path = target_dir / notebook_name
        target_notebook = _read_notebook(target_path)

        for entry in entries:
            index = int(entry["cell_index"])
            source_cell = source_notebook.cells[index]
            target_cell = target_notebook.cells[index]
            source_text = _source(source_cell)
            translation = entry.get("translation")

            if source_cell.cell_type != "markdown" or target_cell.cell_type != "markdown":
                raise ValueError(f"Cell {index} in {notebook_name} is no longer Markdown.")
            if entry.get("cell_id") != source_cell.get("id"):
                raise ValueError(f"Cell ID drift at {notebook_name}:{index}.")
            if entry.get("source_sha256") != _sha256(source_text):
                raise ValueError(f"Canonical Markdown changed at {notebook_name}:{index}.")
            if not isinstance(translation, str) or not translation.strip():
                raise ValueError(f"Missing reviewed translation at {notebook_name}:{index}.")

            target_cell.id = source_cell.id
            target_cell.source = translation
            imported += 1

        _write_notebook(target_notebook, target_path)
    return imported


def apply_review_corrections(bundle: dict[str, Any], corrections: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit human corrections and mark a reviewed draft bundle."""
    by_index = {int(entry["cell_index"]): entry for entry in bundle.get("entries", [])}
    requested = corrections.get("corrections", {})
    for raw_index, translation in requested.items():
        index = int(raw_index)
        if index not in by_index:
            raise ValueError(f"Unknown Markdown cell index in corrections: {index}")
        if not isinstance(translation, str) or not translation.strip():
            raise ValueError(f"Empty reviewed translation for cell {index}")
        by_index[index]["translation"] = translation

    if corrections.get("approve_all") is not True:
        raise ValueError("Review manifest must explicitly set approve_all=true")
    missing = [
        entry["cell_index"]
        for entry in bundle.get("entries", [])
        if not isinstance(entry.get("translation"), str) or not entry["translation"].strip()
    ]
    if missing:
        raise ValueError(f"Cannot approve bundle with missing translations: {missing}")
    for entry in bundle.get("entries", []):
        entry["translation_status"] = "reviewed"
    bundle["human_review"] = {
        "status": "reviewed",
        "correction_count": len(requested),
    }
    return bundle


def _normalized_code(notebook: Any) -> list[str]:
    normalized: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = _source(cell).replace("make_lang('en')", "make_lang('pt')")
        normalized.append(source)
    return normalized


def audit_notebook_pair(source_path: Path, target_path: Path) -> list[str]:
    """Return structural parity errors for a PT-BR/EN-US notebook pair."""
    source_notebook = _read_notebook(source_path)
    target_notebook = _read_notebook(target_path)
    errors: list[str] = []

    if len(source_notebook.cells) != len(target_notebook.cells):
        errors.append("cell count differs")
        return errors

    for index, (source_cell, target_cell) in enumerate(
        zip(source_notebook.cells, target_notebook.cells)
    ):
        if source_cell.cell_type != target_cell.cell_type:
            errors.append(f"cell type differs at index {index}")
        if source_cell.get("id") != target_cell.get("id"):
            errors.append(f"cell ID differs at index {index}")

    if _normalized_code(source_notebook) != _normalized_code(target_notebook):
        errors.append("executable code differs beyond make_lang target")
    return errors


def _selected_paths(name: str | None) -> tuple[Path, ...]:
    if name:
        path = SOURCE_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        return (path,)
    return tuple(sorted(SOURCE_DIR.glob("*.ipynb")))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    clone = subparsers.add_parser("clone", help="Create unexecuted EN-US draft notebooks.")
    clone.add_argument("--name", help="Clone one notebook; defaults to all canonical notebooks.")

    export = subparsers.add_parser("export", help="Export Markdown cells for translation.")
    export.add_argument("--name", help="Export one notebook; defaults to all canonical notebooks.")
    export.add_argument("--output", type=Path, required=True)

    import_parser = subparsers.add_parser("import", help="Import reviewed translations.")
    import_parser.add_argument("--bundle", type=Path, required=True)

    review = subparsers.add_parser("review", help="Apply a human-review manifest.")
    review.add_argument("--bundle", type=Path, required=True)
    review.add_argument("--corrections", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="Audit translated notebook parity.")
    audit.add_argument("--name", help="Audit one notebook; defaults to all EN-US notebooks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "clone":
        paths = _selected_paths(args.name)
        for source_path in paths:
            clone_notebook(source_path, TARGET_DIR / source_path.name)
        print(f"Created {len(paths)} EN-US draft notebook(s).")
        return 0

    if args.command == "export":
        bundle = export_markdown_bundle(_selected_paths(args.name), args.output)
        print(f"Exported {len(bundle['entries'])} Markdown cell(s) to {args.output}.")
        return 0

    if args.command == "import":
        imported = import_markdown_bundle(args.bundle, SOURCE_DIR, TARGET_DIR)
        print(f"Imported {imported} reviewed Markdown translation(s).")
        return 0

    if args.command == "review":
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
        corrections = json.loads(args.corrections.read_text(encoding="utf-8"))
        corrections.setdefault("corrections", {})
        for index, relative_path in corrections.get("correction_files", {}).items():
            correction_path = args.corrections.parent / relative_path
            corrections["corrections"][index] = correction_path.read_text(encoding="utf-8")
        reviewed = apply_review_corrections(bundle, corrections)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"Reviewed {len(reviewed['entries'])} Markdown translation(s); "
            f"applied {reviewed['human_review']['correction_count']} correction(s)."
        )
        return 0

    if args.command == "audit":
        if args.name:
            target_paths = (TARGET_DIR / args.name,)
        else:
            target_paths = tuple(sorted(TARGET_DIR.glob("*.ipynb")))
        if not target_paths:
            raise ValueError("No EN-US notebooks are available for audit.")

        failures = 0
        for target_path in target_paths:
            errors = audit_notebook_pair(SOURCE_DIR / target_path.name, target_path)
            if errors:
                failures += 1
                print(f"FAIL {target_path.name}: {'; '.join(errors)}")
            else:
                print(f"OK   {target_path.name}")
        return int(failures > 0)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
