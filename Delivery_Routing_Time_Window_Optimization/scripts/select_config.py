"""Apply the pre-registered selection rule to the tournament rows and freeze the winner."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pandas as pd  # noqa: E402

from delivery_routing_vrptw import rank_configurations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "reports" / "protocolo-grade-dev.md"
KEYS = ["fleet_strategy", "first_solution", "metaheuristic"]


def main() -> None:
    """Rank the 18 configurations and write ranking.csv and the frozen config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/dev_grid/results.csv")
    parser.add_argument("--frozen-out", default="outputs/frozen/selected_config.json")
    parser.add_argument("--force", action="store_true", help="overwrite an existing frozen file")
    args = parser.parse_args()

    frozen = Path(args.frozen_out)
    if frozen.exists() and not args.force:
        raise SystemExit(f"refused: {frozen} already exists (use --force to overwrite)")
    results = Path(args.results)
    df = pd.read_csv(results, keep_default_na=False, na_values=[""])
    df = df[df["tournament"].astype(str) == "True"]
    if df.empty:
        raise SystemExit("no tournament rows in results")
    ranking = rank_configurations(df, KEYS)
    ranking_path = results.parent / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    best = ranking.iloc[0]
    stat_cols = ("mean_rank", "mean_delta_vehicles", "n_fleet_equal_ref", "median_gap_equal_fleet",
                 "mean_runtime", "n_instances")
    stats = {k: (None if pd.isna(best[k]) else float(best[k])) for k in stat_cols}
    payload = {
        "config": {k: str(best[k]) for k in KEYS},
        "selection_stats": stats,
        "n_configurations": int(len(ranking)),
        "instances": sorted(df["instance"].unique().tolist()),
        "results_file": str(results),
        "protocol": str(PROTOCOL),
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(ranking.round(3).to_string(index=False))
    print(f"wrote {ranking_path} and {frozen}")


if __name__ == "__main__":
    main()
