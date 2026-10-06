"""Executa a análise de incerteza e salva o notebook sem iniciar kernel Jupyter."""
from pathlib import Path
import contextlib
import io
import json
import os
import sys

import nbformat
import pandas as pd
import plotly.basedatatypes
import plotly.io as pio
import IPython.display

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "pt-BR" / "26_incerteza_extra_trees_final_fd001.ipynb"


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    outputs = []
    original_display = IPython.display.display
    original_show = plotly.basedatatypes.BaseFigure.show
    original_cwd = Path.cwd()

    def display(value):
        data = {"text/plain": str(value)}
        if isinstance(value, pd.DataFrame):
            data["text/html"] = value.to_html()
        outputs.append(nbformat.v4.new_output("display_data", data=data))

    def show(figure, *args, **kwargs):
        payload = json.loads(pio.to_json(figure))
        outputs.append(nbformat.v4.new_output(
            "display_data", data={"application/vnd.plotly.v1+json": payload}))

    scope = {"__name__": "__main__"}
    try:
        os.chdir(ROOT)
        IPython.display.display = display
        plotly.basedatatypes.BaseFigure.show = show
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            outputs.clear()
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                exec(compile(cell.source, str(NOTEBOOK), "exec"), scope)
            if captured.getvalue():
                outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.getvalue()))
            cell.outputs = list(outputs)
            cell.execution_count = count
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)
        print(scope["summary"].to_string(index=False))
        print(scope["bands"].to_string(index=False))
        print("Notebook executado diretamente, com tabelas e gráficos salvos.")
    finally:
        IPython.display.display = original_display
        plotly.basedatatypes.BaseFigure.show = original_show
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()

