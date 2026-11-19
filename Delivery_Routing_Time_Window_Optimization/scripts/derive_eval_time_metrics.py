"""Derive time metrics (duration, waiting, makespan) of the final evaluation from the frozen routes.

Reads only frozen files (``outputs/frozen/evaluation/results.csv`` and the SINTEF archive) and writes new
derived files next to them; nothing is solved and no frozen file is edited:

- ``outputs/frozen/evaluation/time_metrics.csv``: one row per cell (instance, method, replicate);
- ``outputs/frozen/evaluation/time_metrics_aggregate.csv``: median replicate per instance, then by scope
  (overall, family) and method.

Usage: ``python scripts/derive_eval_time_metrics.py [--results CSV] [--out DIR]``. The defaults reproduce the
v1 files; for v2 pass ``--results outputs/evaluation_v2/results.csv --out outputs/evaluation_v2``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from delivery_routing_vrptw import benchmark, final_eval, paths, results  # noqa: E402
from delivery_routing_vrptw.io import read_archive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Compute and write the derived time-metric files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None, help="results.csv (default: frozen v1 evaluation)")
    parser.add_argument("--out", default=None, help="output directory (default: the results directory)")
    args = parser.parse_args()
    P = paths.data_paths(ROOT)
    R = paths.result_paths(ROOT)
    split = benchmark.read_split(P["split"])
    src = Path(args.results) if args.results else R["evaluation"]
    if not src.is_absolute():
        src = ROOT / src
    rows = results.load_evaluation(src, split)
    instances = read_archive(P["sintef_zip"])
    tm = final_eval.time_metrics_table(rows, instances)
    out = Path(args.out) if args.out else src.parent
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    tm.to_csv(out / "time_metrics.csv", index=False)
    agg = final_eval.time_aggregate(final_eval.time_representatives(tm))
    agg.to_csv(out / "time_metrics_aggregate.csv", index=False)
    print(f"{len(tm)} cells; max |distance - stored distance| = {tm['distance_diff'].max():.2e}")
    print(f"wrote {out / 'time_metrics.csv'} and {out / 'time_metrics_aggregate.csv'}")


if __name__ == "__main__":
    main()
