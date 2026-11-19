"""Protocol v2: apply the pre-registered comparison rule to a block and write ``decision.json``.

Per instance the representative replicate is position 2 of 3 of the lexicographic order (infeasible,
vehicles, distance); levels are ranked by the mean lexicographic rank over the instances, ties broken by the
mean difference in vehicles to SINTEF and then by the median distance gap among fleet-equal instances. When
all of those tie, the protocol's preference applies: B1 -> scale 10000, B2 -> k = 10; B3 requires a strictly
better mean rank for ``reduce_fleet_v2`` to replace ``fixed_cost``.

Examples
--------
python scripts/select_v2.py --block B1 --inputs outputs/v2/B1/results.csv
python scripts/select_v2.py --block B2 --scale 10000 --inputs outputs/v2/B1/results.csv outputs/v2/B2/results.csv
python scripts/select_v2.py --block B3 --scale 10000 --k 10 --inputs outputs/v2/B1/results.csv \
    outputs/v2/B2/results.csv outputs/v2/B3/results.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Rank the levels of a block, print the table and write ``decision.json``."""
    from delivery_routing_vrptw import v2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", required=True, choices=["B1", "B2", "B3"])
    parser.add_argument("--inputs", nargs="+", required=True, help="result CSVs (B1/B2/B3 outputs as needed)")
    parser.add_argument("--scale", type=int, default=None, help="B2, B3: scale chosen in B1")
    parser.add_argument("--k", type=float, default=None, help="B3: k chosen in B2")
    parser.add_argument("--out", default=None, help="output folder (default: outputs/v2/<block>)")
    args = parser.parse_args()

    frames = [pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in args.inputs]
    rows = pd.concat(frames, ignore_index=True)
    try:
        decision, ranking = v2.decision_record(args.block, rows, args.scale, args.k)
    except ValueError as exc:
        raise SystemExit(f"cannot decide: {exc}") from exc
    decision["inputs"] = [str(p) for p in args.inputs]
    out = Path(args.out) if args.out else Path("outputs") / "v2" / args.block
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out / "ranking.csv", index=False)
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(ranking.round(4).to_string(index=False))
    print(f"winner: {decision['level_column']} = {decision['winner']}  (decided by {decision['decided_by']})")
    if decision.get("r110") is not None:
        print("R110 vehicles by level:", json.dumps(decision["r110"]))
    print(f"wrote {out / 'decision.json'}")


if __name__ == "__main__":
    main()
