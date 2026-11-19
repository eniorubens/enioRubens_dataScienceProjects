"""Derive the layer-B summaries with a coherent representative replicate (lexicographic selection).

Reads only frozen files and writes a NEW file next to them; the old ``summary.csv`` files are kept for
traceability:

- ``outputs/frozen/early_delivery/summary_lex.csv``: per (instance, tolerance), the real plan at position
  ``ceil(R / 2)`` of the order (infeasible, vehicles, distance), with min/max over replicates, instead of the
  column-wise median of ``summary.csv``;
- the window-widening run has one replicate per cell, so its summary does not change: this script only checks
  that ``outputs/frozen/window_scenarios/summary.csv`` still equals the recomputed one.

Usage: ``python scripts/derive_layerb_summaries.py``
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pandas as pd  # noqa: E402

from delivery_routing_vrptw import benchmark, paths, results, route_compare, scenarios  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Write ``summary_lex.csv`` and check the window summary."""
    P = paths.data_paths(ROOT)
    R = paths.result_paths(ROOT)
    split = benchmark.read_split(P["split"])
    early = results.load_early_delivery(R["early_delivery"], split)
    lex = route_compare.early_delivery_summary(early)
    out = R["early_delivery"].parent / "summary_lex.csv"
    lex.to_csv(out, index=False)
    print(f"wrote {out} ({len(lex)} rows)")

    window = results.load_window_scenarios(R["window_scenarios"], split)
    new = scenarios.window_scenario_summary(window)
    old = pd.read_csv(R["window_scenarios"].parent / "summary.csv")
    try:
        pd.testing.assert_frame_equal(new, old, check_dtype=False, atol=1e-9, rtol=0)
        same = True
    except AssertionError:
        same = False
    print(f"window summary unchanged: {same}")


if __name__ == "__main__":
    main()
