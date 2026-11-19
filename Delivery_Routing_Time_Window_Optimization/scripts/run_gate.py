"""Gate run: one dev instance per family, three methods, verified and compared with SINTEF BKS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from delivery_routing_vrptw import (
    compare,
    exact,
    greedy_deadline,
    nearest_neighbor_tw,
    read_archive,
    read_references,
    read_split,
    report_to_dict,
    solve,
    verify,
)
from delivery_routing_vrptw.config import add_config_args, resolve_config, solver_config

ROOT = Path(__file__).resolve().parents[1]
FAMILY_ORDER = ["C1", "C2", "R1", "R2", "RC1", "RC2"]


def main() -> None:
    """Run the gate on the first dev instance of each family and write the results CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = resolve_config(args)
    print("effective config:", json.dumps(cfg.solver.__dict__), flush=True)

    instances = read_archive(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")
    refs = read_references(ROOT / "data" / "reference" / "sintef_bks_100.csv")
    split = read_split(ROOT / "data" / "reference" / "split.csv")
    split_df = pd.DataFrame({"instance": list(split), "set": list(split.values())})
    dev = sorted(split_df.loc[split_df["set"] == "dev", "instance"])
    chosen = []
    for family in FAMILY_ORDER:
        chosen.append(next(n for n in dev if instances[n].family == family))

    config = solver_config(cfg)
    rows = []
    for name in chosen:
        inst = instances[name]
        matrix = exact(inst)
        methods = {
            "greedy_deadline": lambda i=inst: greedy_deadline(i),
            "nearest_neighbor_tw": lambda i=inst: nearest_neighbor_tw(i),
            "ortools": lambda i=inst: solve(i, config),
        }
        for method, run in methods.items():
            sol = run()
            report = verify(inst, sol.routes, matrix)
            row = {"instance": name, "family": inst.family, "method": method}
            row.update(report_to_dict(report))
            comp = compare(report, refs[name])
            row.update({k: v for k, v in comp.items() if k not in row})
            row["vehicles"] = report.vehicles
            row["distance"] = report.distance
            row["runtime_s"] = sol.runtime_s
            row["status"] = sol.meta.get("status", "")
            rows.append(row)
            print(f"done {name} {method}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "gate" / "gate_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (out.parent / "gate_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    cols = ["instance", "method", "feasible", "vehicles", "distance", "ref_vehicles", "ref_distance",
            "delta_vehicles", "gap_distance_pct", "runtime_s"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].round({"distance": 2, "gap_distance_pct": 2, "runtime_s": 2}).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
