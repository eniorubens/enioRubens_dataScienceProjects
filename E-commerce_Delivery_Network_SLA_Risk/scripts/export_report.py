"""Export the executed final report notebooks (11, PT and EN) to self-contained HTML without code.

The notebooks must already be executed (every code cell with an ``execution_count`` and at
least one output): this script never executes them. It writes
``reports/relatorio_final_pt.html`` and ``reports/final_report_en.html``.

The conversion runs in-process through the ``nbconvert`` Python API (``HTMLExporter`` with
the input cells and prompts excluded), so it does not depend on the ``jupyter`` launcher,
which can fail on Windows (for example with ``WinError 623``). Plotly figures are stored by
VS Code only as ``application/vnd.plotly.v1+json``, a format nbconvert cannot draw; each one
is therefore re-rendered as an HTML div, and plotly.js plus the vendored map outline are
embedded in the page, so the report works offline and when opened from disk.

Exit codes: 0 success, 1 a notebook is missing or has code cells without outputs, 2 the
conversion failed or ``nbconvert``/``plotly`` is missing in the running interpreter (usually
the project environment is not active: ``conda activate ecom-delivery-sla``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    (Path("notebooks/pt_br/11_relatorio_final.ipynb"), Path("reports/relatorio_final_pt.html")),
    (Path("notebooks/en_us/11_final_report.ipynb"), Path("reports/final_report_en.html")),
)
PLOTLY_MIME = "application/vnd.plotly.v1+json"
TOPOJSON_SOURCE = Path(__file__).resolve().parent / "vendor"
TOPOJSON_FILES = ("south-america_110m.json",)

REQUIRED_MODULES = ("nbconvert", "nbformat", "plotly")
ENVIRONMENT_MESSAGE = (
    "ERROR: nbconvert/nbformat/plotly nao esta instalado neste interpretador ({python}). "
    "Ative o ambiente do projeto (`conda activate ecom-delivery-sla`) e execute novamente.\n"
    "ERROR: nbconvert/nbformat/plotly is not installed in this interpreter ({python}). "
    "Activate the project environment (`conda activate ecom-delivery-sla`) and run again."
)


def missing_modules() -> list[str]:
    """Return the modules required by the export that the running interpreter cannot import."""
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def _disable_external_ssl_keylog() -> None:
    """Disable an external SSL key-log endpoint during the offline export."""
    os.environ.pop("SSLKEYLOGFILE", None)


def execution_problems(notebook_path: Path) -> list[str]:
    """Return why a notebook is not an executed one (empty list when it is).

    Parameters
    ----------
    notebook_path : pathlib.Path
        Notebook file to inspect.

    Returns
    -------
    list of str
        One message per code cell without ``execution_count`` or without outputs;
        a single message when the file is missing or has no code cells.
    """
    if not notebook_path.exists():
        return [f"notebook not found: {notebook_path}"]
    cells = json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
    code = [(index, cell) for index, cell in enumerate(cells) if cell["cell_type"] == "code"]
    if not code:
        return ["notebook has no code cells"]
    problems = []
    for index, cell in code:
        if cell.get("execution_count") is None:
            problems.append(f"cell {index}: not executed (execution_count is null)")
        if not cell.get("outputs"):
            problems.append(f"cell {index}: no outputs")
    return problems


def render_plotly_outputs(notebook) -> int:
    """Replace every Plotly JSON output with an HTML div drawn by plotly.js.

    Parameters
    ----------
    notebook : nbformat.NotebookNode
        Notebook modified in place (only the in-memory copy, never the file).

    Returns
    -------
    int
        Number of figures converted.
    """
    import plotly.io as pio  # noqa: PLC0415

    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            spec = data.get(PLOTLY_MIME)
            if spec is None:
                continue
            count += 1
            figure = pio.from_json(json.dumps(spec), skip_invalid=True)
            data["text/html"] = figure.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=f"report-figure-{count}",
                config={"responsive": True, "displaylogo": False},
            )
            del data[PLOTLY_MIME]
    return count


def head_scripts() -> str:
    """Return the inline plotly.js and map-outline scripts for the page head.

    plotly.js is placed before the template's require.js so that it defines the
    global ``Plotly`` instead of registering an anonymous AMD module. The map outline
    is preloaded into ``window.PlotlyGeoAssets``, which plotly.js checks before
    fetching it from the network.
    """
    from plotly.offline import get_plotlyjs  # noqa: PLC0415

    geo = ["window.PlotlyGeoAssets = window.PlotlyGeoAssets || {topojson: {}};",
           "window.PlotlyGeoAssets.topojson = window.PlotlyGeoAssets.topojson || {};"]
    for name in TOPOJSON_FILES:
        payload = json.loads((TOPOJSON_SOURCE / name).read_text(encoding="utf-8"))
        geo.append(f"window.PlotlyGeoAssets.topojson[{json.dumps(name.removesuffix('.json'))}] = "
                   f"{json.dumps(payload, separators=(',', ':'))};")
    return ("<script type=\"text/javascript\">" + get_plotlyjs() + "</script>\n"
            + "<script type=\"text/javascript\">" + "\n".join(geo) + "</script>\n")


def convert_notebook(notebook_path: Path, output_path: Path) -> None:
    """Convert one executed notebook into a self-contained HTML file without code.

    Parameters
    ----------
    notebook_path : pathlib.Path
        Executed notebook (read only).
    output_path : pathlib.Path
        HTML file to write.
    """
    _disable_external_ssl_keylog()
    import nbformat  # noqa: PLC0415
    from nbconvert import HTMLExporter  # noqa: PLC0415

    notebook = nbformat.read(notebook_path, as_version=4)
    figures = render_plotly_outputs(notebook)
    exporter = HTMLExporter(exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = exporter.from_notebook_node(notebook)
    if figures:
        if "<head>" not in body:
            raise ValueError("exported HTML has no <head> to receive plotly.js")
        body = body.replace("<head>", "<head>\n" + head_scripts(), 1)
    output_path.write_text(body, encoding="utf-8")


def export_report(
    root: Path = ROOT,
    converter: Callable[[Path, Path], None] = convert_notebook,
) -> int:
    """Export both report notebooks and return the process exit code.

    Parameters
    ----------
    root : pathlib.Path, default project root
        Directory holding ``notebooks/`` and receiving ``reports/``.
    converter : callable, default ``convert_notebook``
        Function ``(notebook_path, output_path)`` that writes the HTML; replaceable in tests.

    Returns
    -------
    int
        0 on success, 1 when a notebook is missing or not executed (nothing is
        exported in that case), 2 when the conversion fails.
    """
    failures = {}
    for notebook, _ in TARGETS:
        problems = execution_problems(root / notebook)
        if problems:
            failures[notebook] = problems
    if failures:
        for notebook, problems in failures.items():
            print(f"ERROR: {notebook} cannot be exported", file=sys.stderr)
            for problem in problems[:10]:
                print(f"  {problem}", file=sys.stderr)
            if len(problems) > 10:
                print(f"  ... and {len(problems) - 10} more", file=sys.stderr)
        print("Execute the notebooks (all cells, top to bottom) and run this script again.", file=sys.stderr)
        return 1

    for notebook, output in TARGETS:
        output_path = root / output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            converter(root / notebook, output_path)
        except Exception as error:  # noqa: BLE001 - report any conversion failure as exit code 2
            print(f"ERROR: HTML conversion failed for {notebook}: {error!r}", file=sys.stderr)
            return 2
        print(f"wrote {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and export the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="project root (default: this repository)")
    arguments = parser.parse_args(argv)
    if missing_modules():
        print(ENVIRONMENT_MESSAGE.format(python=sys.executable), file=sys.stderr)
        return 2
    return export_report(arguments.root)


if __name__ == "__main__":
    sys.exit(main())
