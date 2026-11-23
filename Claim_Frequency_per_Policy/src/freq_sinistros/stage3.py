"""Comandos da etapa 3: EDA de treino e comparacao em validacao."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freq_sinistros import __version__
from freq_sinistros.audit import _jsonable, _peak_mb
from freq_sinistros.data import load_partitions
from freq_sinistros.eda import compute_eda

OUTPUTS_DIR = Path("outputs") / "etapa_03"


def write_json(path: Path, payload: Any) -> None:
    """Grave JSON deterministico (chaves ordenadas, UTF-8)."""
    path.write_text(json.dumps(_jsonable(payload), indent=2,
                               ensure_ascii=False, sort_keys=True),
                    encoding="utf-8")


def write_table(path: Path, table: pd.DataFrame) -> None:
    """Grave CSV deterministico (UTF-8, LF, 12 digitos significativos)."""
    table.to_csv(path, index=False, lineterminator="\n",
                 float_format="%.12g")


def run_eda(root: Path) -> int:
    """Calcule a EDA usando somente o treino e grave em ``outputs``.

    Efeitos colaterais: le o ARFF local (sem rede) e grava
    ``outputs/etapa_03/eda_*``.

    Args:
        root: raiz do projeto.

    Returns:
        0 em sucesso.
    """
    root = Path(root)
    out = root / OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    parts, provenance = load_partitions(root, keep=("train",))
    train = parts["train"]
    tables, summary = compute_eda(train)
    for name, table in tables.items():
        write_table(out / f"{name}.csv", table)
    summary["provenance"] = {k: v for k, v in provenance.items()
                             if k != "sizes"}
    summary["partition_used"] = "train"
    summary["package_version"] = __version__
    write_json(out / "eda_summary.json", summary)
    write_json(out / "eda_cost.json", {
        "wall_s": round(time.perf_counter() - t0, 3),
        "peak_rss_mib": round(_peak_mb(), 1),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__})
    print("eda ok:", len(tables), "tabelas; treino", len(train))
    return 0


def run_compare(root: Path) -> int:
    """Execute a comparacao completa em validacao (ver ``compare``)."""
    from freq_sinistros.compare import run_comparison

    return run_comparison(Path(root))
