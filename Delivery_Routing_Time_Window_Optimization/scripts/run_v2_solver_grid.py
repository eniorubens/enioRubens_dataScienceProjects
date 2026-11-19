"""Protocol v2, blocks B1-B4: solver diagnostics on the development instances (reports/protocolo-v2.md).

B1: scale in {100, 10000} at k = 10 (fixed_cost). B2: k in {1, 100} at --scale (k = 10 comes from B1).
B3: reduce_fleet v2 at --scale --k. B4: --scale --k --fleet-strategy with the B4 worker count.
Only development instances can be run: there is no override flag for the evaluation set.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_cell(cell: dict, v2_dict: dict, solver_dict: dict, time_limit: float, workers: int,
             archive: str, refs_path: str) -> dict:
    """Solve one cell, verify it against the original instance and return one result row."""
    from delivery_routing_vrptw import (
        SolverConfig,
        compare,
        exact,
        fixed_cost_for,
        minimize_fleet_v2,
        read_archive,
        read_references,
        solve,
        verify,
    )

    name = cell["instance"]
    inst = read_archive(archive, names=[name])[name]
    ref = read_references(refs_path)[name]
    cfg = SolverConfig(
        first_solution=solver_dict["first_solution"], metaheuristic=solver_dict["metaheuristic"],
        time_limit_s=time_limit, scale=int(cell["scale"]), fixed_cost_k=float(cell["k"]),
    )
    if cell["fleet_strategy"] == "reduce_fleet_v2":
        sol = minimize_fleet_v2(
            inst, cfg, total_time_s=time_limit, first_fraction=v2_dict["first_fraction"],
            step_time_s=v2_dict["step_time_s"], drop_penalty_factor=v2_dict["drop_penalty_factor"],
        )
    else:
        sol = solve(inst, cfg)
    report = verify(inst, sol.routes, exact(inst))
    comp = compare(report, ref)
    row = {**cell, "family": inst.family, "time_limit_s": time_limit, "workers": workers,
           "vehicle_fixed_cost": fixed_cost_for(inst, cfg)}
    row["feasible"] = report.feasible
    row["vehicles"] = report.vehicles
    row["distance"] = report.distance
    row["duration"] = report.duration
    row["waiting"] = report.waiting
    row["runtime_s"] = sol.runtime_s
    row["status"] = sol.meta.get("status", "")
    row["warm_start"] = sol.meta.get("warm_start", "")
    row["attempts"] = json.dumps(sol.meta["attempts"]) if "attempts" in sol.meta else ""
    for k in ("ref_vehicles", "ref_distance", "delta_vehicles", "gap_distance_pct",
              "distance_comparable", "beats_reference"):
        row[k] = comp[k]
    row["routes"] = json.dumps(sol.routes)
    return row


def main() -> None:
    """Run one block of the v2 solver grid with resume and incremental CSV output."""
    from delivery_routing_vrptw import read_split, runs, v2
    from delivery_routing_vrptw.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, choices=v2.BLOCKS)
    parser.add_argument("--out", default=None, help="default: outputs/v2/<block>")
    parser.add_argument("--config", default=None, help="TOML file (default: config/experiment.toml)")
    parser.add_argument("--scale", type=int, default=None, help="B2-B4: scale chosen in B1")
    parser.add_argument("--k", type=float, default=None, help="B3-B4: fixed-cost multiplier chosen in B2")
    parser.add_argument("--fleet-strategy", choices=v2.FLEET_STRATEGIES_V2, default=None, help="B4 only")
    parser.add_argument("--instances", default=None, help="comma list of development instances")
    parser.add_argument("--set", dest="instance_set", default="dev", choices=["dev", "eval", "all"],
                        help="only 'dev' is accepted (kept so that a request for eval is refused explicitly)")
    parser.add_argument("--time-limit", type=float, default=None, help="override [v2].time_limit_s (smoke tests)")
    parser.add_argument("--replicates", type=int, default=None, help="override [v2].replicates")
    parser.add_argument("--workers", type=int, default=None, help="override the block's worker count")
    runs.add_resume_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cv2 = cfg.v2
    time_limit = args.time_limit if args.time_limit is not None else cv2.time_limit_s
    workers = args.workers if args.workers is not None else v2.block_workers(args.block, cv2)
    reps = args.replicates if args.replicates is not None else cv2.replicates
    split = read_split(ROOT / "data" / "reference" / "split.csv")
    try:
        names = v2.select_dev_instances(
            split, args.instances.split(",") if args.instances else None, args.instance_set
        )
        cells = v2.block_cells(args.block, names, cv2, args.scale, args.k, args.fleet_strategy, reps)
    except (PermissionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    cfg = replace(cfg, v2=replace(cv2, time_limit_s=time_limit, replicates=reps))
    print("effective config:", json.dumps(cfg.to_dict()), flush=True)

    out = Path(args.out) if args.out else Path("outputs") / "v2" / args.block
    if not out.is_absolute():
        out = ROOT / out
    results = out / "results.csv"
    archive = str(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")
    refs_path = str(ROOT / "data" / "reference" / "sintef_bks_100.csv")
    meta = v2.run_metadata(
        cfg.to_dict(), args.config or "config/experiment.toml", str(ROOT / cv2.protocol), names,
        block=args.block, scale=args.scale, k=args.k, fleet_strategy=args.fleet_strategy,
        time_limit_s=time_limit, workers=workers, replicates=reps,
    )
    # scale, k and the fleet strategy are cell keys; the rest of the effective v2 configuration is shared
    fields = {
        "runner": "run_v2_solver_grid", "workers": workers, "block": args.block, "first_solution": cfg.solver.first_solution,
        "metaheuristic": cfg.solver.metaheuristic, "time_limit_s": time_limit, "replicates": reps,
        "first_fraction": cv2.first_fraction, "step_time_s": cv2.step_time_s,
        "drop_penalty_factor": cv2.drop_penalty_factor,
    }
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, fields, meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    done: set[tuple] = set()
    if results.exists() and results.stat().st_size > 0:
        prev = pd.read_csv(results, usecols=v2.CELL_KEY, keep_default_na=False)
        done = {v2.cell_key(r) for r in prev.to_dict(orient="records")}
    todo = [c for c in cells if v2.cell_key(c) not in done]
    total = len(todo)
    print(f"cells: {len(cells)} total, {len(cells) - total} already done, {total} to run, workers={workers}",
          flush=True)
    if not todo:
        return

    t0 = time.perf_counter()
    k = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_cell, c, cv2.__dict__, cfg.solver.__dict__, time_limit, workers, archive, refs_path): c
            for c in todo
        }
        for fut in as_completed(futures):
            row = fut.result()
            runs.append_rows(results, pd.DataFrame([row]), fingerprint)
            k += 1
            print(
                f"done {k}/{total} {row['instance']} scale={row['scale']} k={v2.k_label(row['k'])} "
                f"{row['fleet_strategy']} rep={row['replicate']} feasible={row['feasible']} "
                f"vehicles={row['vehicles']} (ref {row['ref_vehicles']}) distance={row['distance']:.2f} "
                f"runtime={row['runtime_s']:.1f}s warm_start={row['warm_start']}",
                flush=True,
            )
    print(f"finished in {time.perf_counter() - t0:.0f}s -> {results}", flush=True)


if __name__ == "__main__":
    main()
