"""Layer A: time sensitivity. Does a larger OR-Tools search budget reduce the fleet? (vs SINTEF BKS)."""

from __future__ import annotations

import argparse
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
KEY_COLS = ["instance", "budget_s"]


def run_cell(instance: str, budget: float, cfg_dict: dict, archive: str, refs_path: str) -> dict:
    """Solve one (instance, budget) cell with the frozen config and return one result row."""
    from delivery_routing_vrptw import compare, exact, read_archive, read_references, verify
    from delivery_routing_vrptw.config import ExperimentConfig, SolverSection, solve_configured

    cfg = ExperimentConfig(solver=SolverSection(**cfg_dict["solver"]))
    inst = read_archive(archive)[instance]
    ref = read_references(refs_path)[instance]
    sol = solve_configured(inst, cfg, time_limit_s=budget)
    report = verify(inst, sol.routes, exact(inst))
    comp = compare(report, ref)
    row = {"instance": instance, "family": inst.family, "budget_s": budget}
    row["feasible"] = report.feasible
    row["vehicles"] = report.vehicles
    row["distance"] = report.distance
    row["runtime_s"] = sol.runtime_s
    row["status"] = sol.meta.get("status", "")
    for k in ("ref_vehicles", "ref_distance", "delta_vehicles", "gap_distance_pct", "distance_comparable"):
        row[k] = comp[k]
    row["routes"] = json.dumps(sol.routes)
    return row


def main() -> None:
    """Run the time-sensitivity grid with resume and incremental CSV output."""
    from delivery_routing_vrptw import read_split, runs
    from delivery_routing_vrptw.config import add_config_args, resolve_config, select_instances, solver_config

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser, budgets=True)
    parser.add_argument("--instances", default=None, help="comma list of instance names within the set")
    parser.add_argument("--out", default="outputs/time_sensitivity")
    parser.add_argument("--allow-eval", action="store_true", help="permit instance_set eval/all")
    runs.add_resume_args(parser)
    args = parser.parse_args()

    cfg = resolve_config(args)
    ts = cfg.time_sensitivity
    print("effective config:", json.dumps(cfg.to_dict()), flush=True)
    split = read_split(ROOT / "data" / "reference" / "split.csv")
    if ts.instance_set != "dev":
        print("WARNING: the evaluation set must only be opened after the protocol says so.", flush=True)
    try:
        names = select_instances(
            split, ts.instance_set, args.instances.split(",") if args.instances else None, args.allow_eval
        )
    except (PermissionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    results = out / "results.csv"
    archive = str(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")
    refs_path = str(ROOT / "data" / "reference" / "sintef_bks_100.csv")

    import numpy
    import ortools

    meta = {
        "timestamp": datetime.now().astimezone().isoformat(), "config": cfg.to_dict(),
        "config_file": args.config or "config/experiment.toml", "instances": names,
        "ortools": ortools.__version__, "python": platform.python_version(), "numpy": numpy.__version__,
        "cpu_count": os.cpu_count(), "argv": sys.argv,
    }
    fields = {"runner": "run_time_sensitivity", "workers": cfg.solver.workers, "fleet_strategy": cfg.solver.fleet_strategy,
              **runs.solver_fields(solver_config(cfg), with_time_limit=False)}  # the budget is a cell key
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, fields, meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    cells = [(n, float(b)) for n in names for b in ts.budgets_s]
    done: set[tuple[str, float]] = set()
    if results.exists() and results.stat().st_size > 0:
        prev = pd.read_csv(results, usecols=KEY_COLS)
        done = {(str(i), float(b)) for i, b in prev[KEY_COLS].values}
    todo = [c for c in cells if c not in done]
    total = len(todo)
    print(f"cells: {len(cells)} total, {len(cells) - total} already done, {total} to run", flush=True)
    if not todo:
        return

    k = 0
    t0 = time.perf_counter()
    cfg_dict = cfg.to_dict()
    with ProcessPoolExecutor(max_workers=cfg.solver.workers) as pool:
        futures = {pool.submit(run_cell, n, b, cfg_dict, archive, refs_path): (n, b) for n, b in todo}
        for fut in as_completed(futures):
            row = fut.result()
            runs.append_rows(results, pd.DataFrame([row]), fingerprint)
            k += 1
            print(
                f"done {k}/{total} {row['instance']} budget={row['budget_s']:g}s feasible={row['feasible']} "
                f"vehicles={row['vehicles']} (ref {row['ref_vehicles']}) distance={row['distance']:.2f} "
                f"runtime={row['runtime_s']:.1f}s",
                flush=True,
            )
    print(f"finished in {time.perf_counter() - t0:.0f}s -> {results}", flush=True)


if __name__ == "__main__":
    main()
