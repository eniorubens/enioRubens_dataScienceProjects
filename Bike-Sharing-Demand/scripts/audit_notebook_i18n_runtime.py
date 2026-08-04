"""Collect offline-catalog misses while executing a notebook prefix in memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--stop-before", default="")
    parser.add_argument("--kernel", default="bike-sharing")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    notebook = nbformat.read(args.notebook, as_version=4)
    first_code = next(cell for cell in notebook.cells if cell.cell_type == "code")
    first_code.source += """

_runtime_i18n_missing = set()

def _capture_i18n_missing(texts):
    _runtime_i18n_missing.update(texts)
    return texts

lang._translate_batch = _capture_i18n_missing
lang._save_cache = lambda: None
"""

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(args.workdir)}},
    )
    client.reset_execution_trackers()
    with client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            if args.stop_before and args.stop_before in cell.source:
                break
            if cell.cell_type == "code":
                client.execute_cell(cell, index)

        report = nbformat.v4.new_code_cell(
            "import json\nprint(json.dumps(sorted(_runtime_i18n_missing), " "ensure_ascii=False))"
        )
        notebook.cells.append(report)
        client.execute_cell(report, len(notebook.cells) - 1)

    print(report.outputs[0]["text"].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
