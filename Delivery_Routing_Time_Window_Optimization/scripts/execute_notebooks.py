"""Execute notebooks in place with nbclient and persist their outputs.

Usage: ``python scripts/execute_notebooks.py notebooks/pt_br/01_*.ipynb ...``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before the kernel starts: avoids an MKL/OpenMP crash in matplotlib on this machine.
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def execute(path: Path, timeout: int = 600) -> None:
    """Execute one notebook from the project root and overwrite it with the outputs."""
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(notebook, path)


def main() -> None:
    """Execute every notebook given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    for path in args.notebooks:
        print(f"executing {path} ...", flush=True)
        execute(path.resolve(), args.timeout)
        print(f"done {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
