"""Development grid: 18 OR-Tools configurations x dev instances, plus reference cells (see protocol)."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "reports" / "protocolo-grade-dev.md"
FLEETS = ("fixed_cost", "reduce_fleet")
FIRSTS = ("PARALLEL_CHEAPEST_INSERTION", "PATH_CHEAPEST_ARC", "SAVINGS")
METAS = ("GUIDED_LOCAL_SEARCH", "TABU_SEARCH", "SIMULATED_ANNEALING")
BASELINES = ("greedy_deadline", "nearest_neighbor_tw")
KEY_COLS = ["instance", "fleet_strategy", "first_solution", "metaheuristic", "method"]


def build_cells(names: list[str], include_reference: bool) -> list[dict]:
    """Return the cells to run for the given dev instances."""
    cells = []
    for name in names:
        for fleet, first, meta in itertools.product(FLEETS, FIRSTS, METAS):
            cells.append({
                "instance": name, "fleet_strategy": fleet, "first_solution": first,
                "metaheuristic": meta, "tournament": True,
                "method": "ortools" if fleet == "fixed_cost" else "ortools_minimize_fleet",
            })
        if include_reference:
            for first in FIRSTS:
                cells.append({
                    "instance": name, "fleet_strategy": "fixed_cost", "first_solution": first,
                    "metaheuristic": "NONE", "tournament": False, "method": "ortools",
                })
            for base in BASELINES:
                cells.append({
                    "instance": name, "fleet_strategy": "none", "first_solution": "none",
                    "metaheuristic": "none", "tournament": False, "method": base,
                })
    return cells


def run_cell(cell: dict, budget: float, archive: str, refs_path: str) -> dict:
    """Run one cell in a worker process and return one result row."""
    from delivery_routing_vrptw import (
        SolverConfig,
        compare,
        exact,
        greedy_deadline,
        minimize_fleet,
        nearest_neighbor_tw,
        read_archive,
        read_references,
        solve,
        verify,
    )

    name = cell["instance"]
    inst = read_archive(archive)[name]
    ref = read_references(refs_path)[name]
    method = cell["method"]
    if method == "greedy_deadline":
        sol = greedy_deadline(inst)
    elif method == "nearest_neighbor_tw":
        sol = nearest_neighbor_tw(inst)
    else:
        cfg = SolverConfig(
            first_solution=cell["first_solution"], metaheuristic=cell["metaheuristic"], time_limit_s=budget
        )
        if cell["fleet_strategy"] == "reduce_fleet":
            sol = minimize_fleet(inst, cfg, step_time_s=5, total_time_s=budget, first_step_time_s=budget / 3)
        else:
            sol = solve(inst, cfg)
    report = verify(inst, sol.routes, exact(inst))
    comp = compare(report, ref)
    row = {**cell, "family": inst.family}
    row["feasible"] = report.feasible
    row["vehicles"] = report.vehicles
    row["distance"] = report.distance
    row["distance_2dp"] = round(report.distance, 2)
    row["duration"] = report.duration
    row["waiting"] = report.waiting
    row["runtime_s"] = sol.runtime_s
    row["status"] = sol.meta.get("status", "")
    row["attempts"] = json.dumps(sol.meta["attempts"]) if "attempts" in sol.meta else ""
    for k in ("ref_vehicles", "ref_distance", "delta_vehicles", "gap_distance_pct",
              "distance_comparable", "beats_reference"):
        row[k] = comp[k]
    row["routes"] = json.dumps(sol.routes)
    return row


def main() -> None:
    """Run the grid with resume and incremental CSV output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="outputs/dev_grid")
    parser.add_argument("--instances", default=None, help="comma list, must be dev instances")
    parser.add_argument("--include-reference", action=argparse.BooleanOptionalAction, default=True)
    from delivery_routing_vrptw import SolverConfig, read_split, runs

    runs.add_resume_args(parser)
    args = parser.parse_args()

    split = read_split(ROOT / "data" / "reference" / "split.csv")
    dev = sorted(n for n, s in split.items() if s == "dev")
    if args.instances:
        names = [n.strip().upper() for n in args.instances.split(",") if n.strip()]
        bad = [n for n in names if split.get(n) != "dev"]
        if bad:
            raise SystemExit(f"refused: not dev instances (set != 'dev'): {bad}")
    else:
        names = dev
    if not names:
        raise SystemExit("refused: no instances requested")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    results = out / "results.csv"
    archive = str(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")
    refs_path = str(ROOT / "data" / "reference" / "sintef_bks_100.csv")

    import numpy
    import ortools

    meta = {
        "timestamp": datetime.now().astimezone().isoformat(), "budget_s": args.budget,
        "workers": args.workers, "instances": names, "include_reference": args.include_reference,
        "ortools": ortools.__version__, "python": platform.python_version(), "numpy": numpy.__version__,
        "cpu_count": os.cpu_count(), "protocol": str(PROTOCOL), "argv": sys.argv,
    }
    # strategies vary per cell (they are cell keys); the budget and the fixed-cost rule are shared
    fields = {"runner": "run_dev_grid", "budget_s": args.budget, "workers": args.workers, **runs.fixed_cost_fields(SolverConfig())}
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, fields, meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    cells = build_cells(names, args.include_reference)
    done_keys: set[tuple] = set()
    if results.exists() and results.stat().st_size > 0:
        prev = pd.read_csv(results, usecols=KEY_COLS, keep_default_na=False, dtype=str)
        done_keys = set(map(tuple, prev[KEY_COLS].values))
    todo = [c for c in cells if tuple(str(c[k]) for k in KEY_COLS) not in done_keys]
    total = len(todo)
    print(f"cells: {len(cells)} total, {len(cells) - total} already done, {total} to run", flush=True)
    if not todo:
        return

    k = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, c, args.budget, archive, refs_path): c for c in todo}
        for fut in as_completed(futures):
            row = fut.result()
            runs.append_rows(results, pd.DataFrame([row]), fingerprint)
            k += 1
            cfg = f"{row['fleet_strategy']}/{row['first_solution']}/{row['metaheuristic']}/{row['method']}"
            print(
                f"done {k}/{total} {row['instance']} {cfg} {row['feasible']} {row['vehicles']} "
                f"{row['distance_2dp']} {row['runtime_s']:.1f}s",
                flush=True,
            )
    print(f"finished in {time.perf_counter() - t0:.0f}s -> {results}", flush=True)


if __name__ == "__main__":
    main()
