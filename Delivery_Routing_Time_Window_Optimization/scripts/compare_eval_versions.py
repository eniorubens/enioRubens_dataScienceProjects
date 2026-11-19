"""Compare the v1 and v2 final evaluations (``reports/protocolo-fase-d.md``), descriptively.

Reads the frozen v1 ``results.csv`` and the v2 one, and writes ``comparison_by_instance.csv`` and
``comparison_aggregate.csv`` (median replicate of each version) into the v2 output folder. Nothing is solved.

Usage: ``python scripts/compare_eval_versions.py [--v1 CSV] [--v2 CSV] [--out DIR]``
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pandas as pd  # noqa: E402

from delivery_routing_vrptw import benchmark, paths, results  # noqa: E402
from delivery_routing_vrptw.evaluation import compare_versions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    """Write the per-instance and aggregate comparison tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", default="outputs/frozen/evaluation/results.csv")
    parser.add_argument("--v2", default="outputs/evaluation_v2/results.csv")
    parser.add_argument("--out", default="outputs/evaluation_v2")
    args = parser.parse_args()

    split = benchmark.read_split(paths.data_paths(ROOT)["split"])
    v1 = results.load_evaluation(_abs(args.v1), split)
    v2 = results.load_evaluation(_abs(args.v2), split)
    by_instance, aggregate = compare_versions(v1, v2)
    out = _abs(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_instance.to_csv(out / "comparison_by_instance.csv", index=False)
    aggregate.to_csv(out / "comparison_aggregate.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.2f}".format):
        print(aggregate.to_string(index=False))
    print(f"{len(by_instance)} instances compared -> {out}")


if __name__ == "__main__":
    main()
