"""Derive the 2026-11-19 phase-C corrections from the frozen results (no solver, no frozen file edited).

* C3: ``outputs/frozen/phase_c/C3/<block>/aggregate_representative.csv``, the aggregate over the representative
  replicate of each instance (protocol rule) with the noise floor as a distribution, beside the pooled
  ``aggregate.csv`` of the run (kept for traceability).
* C2.2: ``outputs/frozen/phase_c/C2/fleet_priority/dropped_spread.csv``, the spread of the customers dropped
  across replicates of the same cell.

Run: ``python scripts/derive_phase_c_corrections.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

ROOT = Path(__file__).resolve().parents[1]
FROZEN_C = ROOT / "outputs" / "frozen" / "phase_c"


def derive_c3(block_dir: Path) -> Path:
    """Write ``aggregate_representative.csv`` next to the ``results.csv`` of a C3 block and return its path."""
    from delivery_routing_vrptw import phase_c

    rows = phase_c.read_results(block_dir / "results.csv")
    table = phase_c.reopt_aggregate_representative(rows, phase_c.noise_table(rows))
    path = block_dir / "aggregate_representative.csv"
    table.to_csv(path, index=False)
    return path


def derive_c2_spread(block_dir: Path) -> Path:
    """Write ``dropped_spread.csv`` next to the ``results.csv`` of the C2.2 block and return its path."""
    from delivery_routing_vrptw import phase_c

    rows = phase_c.read_results(block_dir / "results.csv")
    path = block_dir / "dropped_spread.csv"
    phase_c.dropped_spread(rows).to_csv(path, index=False)
    return path


def main() -> None:
    """Derive the files for every block present under the frozen phase-C folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=FROZEN_C, help="phase-C folder (default: outputs/frozen/phase_c)")
    args = parser.parse_args()
    for block in ("removal", "insertion"):
        folder = args.frozen / "C3" / block
        if (folder / "results.csv").exists():
            print("wrote", derive_c3(folder))
    folder = args.frozen / "C2" / "fleet_priority"
    if (folder / "results.csv").exists():
        print("wrote", derive_c2_spread(folder))


if __name__ == "__main__":
    main()
