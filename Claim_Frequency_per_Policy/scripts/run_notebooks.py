"""Execute os notebooks em kernel limpo e persista as saidas.

Uso, na raiz do projeto:

* ``python scripts/run_notebooks.py``: executa os oito notebooks (exige o
  ARFF e as atribuicoes locais: o 01 le os dados e o 02 e o 03 conferem o
  hash do ARFF). Grava ``outputs/notebook_execution_etapa03.json``,
  ``outputs/notebook_execution_etapa04.json`` e o da etapa 5, todos fora
  das pastas ``outputs/etapa_*`` seladas por checksum.
* ``python scripts/run_notebooks.py --final``: executa so os dois notebooks
  finais (``04_avaliacao_final``), que leem apenas os agregados gravados e
  nao precisam do ARFF nem das atribuicoes. Grava so
  ``outputs/notebook_execution_etapa05.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["01_eda_treino.ipynb", "02_comparacao_modelos.ipynb",
         "03_diagnostico_zeros.ipynb", "04_avaliacao_final.ipynb"]
LANGS = ["pt_br", "en_us"]
TIMEOUT_S = 1800


def run(path: Path) -> dict:
    """Execute um notebook e grave-o com as saidas."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=TIMEOUT_S, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    t0 = time.perf_counter()
    status, error = "success", None
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001 - registrar qualquer falha
        status, error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
    elapsed = time.perf_counter() - t0
    nbformat.write(nb, path)
    code = [c for c in nb.cells if c.cell_type == "code"]
    return {"notebook": str(path.relative_to(ROOT)).replace("\\", "/"),
            "status": status, "error": error,
            "wall_s": round(elapsed, 1), "code_cells": len(code),
            "executed_cells": sum(c.execution_count is not None
                                  for c in code)}


def main(argv: list[str] | None = None) -> int:
    """Execute os notebooks (todos, ou so os finais com ``--final``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true",
                        help="executa so 04_avaliacao_final (sem ARFF)")
    args = parser.parse_args(argv)
    names = ["04_avaliacao_final.ipynb"] if args.final else NAMES
    results = [run(ROOT / "notebooks" / lang / name)
               for name in names for lang in LANGS]
    if not args.final:
        # fora das pastas seladas por SHA256SUMS (etapas 3 e 4)
        out = ROOT / "outputs" / "notebook_execution_etapa03.json"
        out.write_text(json.dumps({
            "command": "python scripts/run_notebooks.py",
            "python": sys.version, "results": results}, indent=2,
            ensure_ascii=False), encoding="utf-8")
        stage4 = [r for r in results if "03_diagnostico" in r["notebook"]]
        (ROOT / "outputs" / "notebook_execution_etapa04.json"
         ).write_text(json.dumps({
             "command": "python scripts/run_notebooks.py",
             "python": sys.version, "results": stage4}, indent=2,
             ensure_ascii=False), encoding="utf-8")
    final = [r for r in results if "04_avaliacao" in r["notebook"]]
    # fora de outputs/etapa_05, cujos hashes ficam congelados na execucao
    # oficial
    (ROOT / "outputs" / "notebook_execution_etapa05.json").write_text(
        json.dumps({"command": "python scripts/run_notebooks.py"
                    + (" --final" if args.final else ""),
                    "python": sys.version, "results": final}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    for r in results:
        print(r)
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Encerra sem aguardar threads residuais do kernel (ZMQ), que em alguns
    # ambientes alteravam o codigo de saida depois do resumo.
    os._exit(code)
