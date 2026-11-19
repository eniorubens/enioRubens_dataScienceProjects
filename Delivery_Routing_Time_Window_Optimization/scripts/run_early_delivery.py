"""Layer B: early-delivery scenario (business layer; no SINTEF comparison).

Customers accept deliveries before their window opens. The scenario only transforms the input
(``allow_early``); the solver and the verifier keep the "wait until ready" rule. Each cell is solved
``replicates`` times to separate the real effect from run-to-run (wall-clock) noise.
"""

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
KEY_COLS = ["instance", "tolerance_label", "replicate"]


def run_cell(instance: str, tolerance: float, replicate: int, time_limit: float, cfg_dict: dict, archive: str) -> dict:
    """Transform the instance, solve with the frozen config, verify and measure on the transformed one."""
    import numpy as np

    from delivery_routing_vrptw import (
        allow_early,
        exact,
        read_archive,
        schedule_metrics,
        tolerance_label,
        verify,
    )
    from delivery_routing_vrptw.config import ExperimentConfig, SolverSection, solve_configured

    cfg = ExperimentConfig(solver=SolverSection(**cfg_dict["solver"]))
    plain = read_archive(archive)[instance]
    inst = allow_early(plain, tolerance)
    if tolerance == 0:
        same = np.array_equal(inst.ready, plain.ready) and np.array_equal(inst.due, plain.due)
        assert inst.name == plain.name and same, "tolerance 0 must be the plain benchmark instance"
    sol = solve_configured(inst, cfg, time_limit_s=time_limit)
    matrix = exact(inst)
    report = verify(inst, sol.routes, matrix)
    row = {
        "instance": instance,
        "family": plain.family,
        "tolerance_label": tolerance_label(tolerance),
        "tolerance": tolerance,
        "replicate": replicate,
        "scenario_name": inst.name,
        "time_limit_s": time_limit,
        "feasible": report.feasible,
    }
    row.update(schedule_metrics(inst, sol.routes, matrix))
    row["runtime_s"] = sol.runtime_s
    row["status"] = sol.meta.get("status", "")
    row["routes"] = json.dumps(sol.routes)
    return row


def main() -> None:
    """Run the early-delivery scenarios with resume and incremental CSV output."""
    from delivery_routing_vrptw import (
        early_delivery_summary,
        read_split,
        reallocation_summary,
        reallocation_table,
        runs,
        tolerance_label,
    )
    from delivery_routing_vrptw.config import add_config_args, resolve_config, select_instances, solver_config

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser, early=True)
    parser.add_argument("--instances", default=None, help="comma list of instance names within the set")
    parser.add_argument("--out", default="outputs/early_delivery")
    parser.add_argument("--allow-eval", action="store_true", help="permit instance_set eval/all")
    runs.add_resume_args(parser)
    args = parser.parse_args()

    cfg = resolve_config(args)
    ed = cfg.early_delivery
    print("effective config:", json.dumps(cfg.to_dict()), flush=True)
    split = read_split(ROOT / "data" / "reference" / "split.csv")
    if ed.instance_set != "dev":
        print("WARNING: the evaluation set must only be opened after the protocol says so.", flush=True)
    try:
        names = select_instances(
            split, ed.instance_set, args.instances.split(",") if args.instances else None, args.allow_eval
        )
    except (PermissionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    results = out / "results.csv"
    archive = str(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")

    import numpy
    import ortools

    meta = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "config": cfg.to_dict(),
        "config_file": args.config or "config/experiment.toml",
        "instances": names,
        "levels": [tolerance_label(t) for t in ed.levels()],
        "ortools": ortools.__version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
    }
    fields = {"runner": "run_early_delivery", "workers": cfg.solver.workers, "fleet_strategy": cfg.solver.fleet_strategy,
              "replicates": ed.replicates, **runs.solver_fields(solver_config(cfg, ed.time_limit_s))}
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, fields, meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    cells = [(n, t, r) for n in names for t in ed.levels() for r in range(ed.replicates)]
    done: set[tuple[str, str, int]] = set()
    if results.exists() and results.stat().st_size > 0:
        prev = pd.read_csv(results, usecols=KEY_COLS, dtype={"tolerance_label": str})
        done = {(str(i), str(lab), int(r)) for i, lab, r in prev[KEY_COLS].values}
    todo = [c for c in cells if (c[0], tolerance_label(c[1]), c[2]) not in done]
    total = len(todo)
    print(f"cells: {len(cells)} total, {len(cells) - total} already done, {total} to run", flush=True)

    k = 0
    t0 = time.perf_counter()
    cfg_dict = cfg.to_dict()
    if todo:
        with ProcessPoolExecutor(max_workers=cfg.solver.workers) as pool:
            futures = {
                pool.submit(run_cell, n, t, r, ed.time_limit_s, cfg_dict, archive): (n, t, r) for n, t, r in todo
            }
            for fut in as_completed(futures):
                row = fut.result()
                runs.append_rows(results, pd.DataFrame([row]), fingerprint)
                k += 1
                print(
                    f"done {k}/{total} {row['scenario_name']} rep={row['replicate']} feasible={row['feasible']} "
                    f"vehicles={row['vehicles']} distance={row['distance']:.2f} "
                    f"duration={row['total_duration']:.1f} runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
        print(f"finished in {time.perf_counter() - t0:.0f}s -> {results}", flush=True)

    rows = pd.read_csv(results, dtype={"tolerance_label": str})
    summary = early_delivery_summary(rows)
    summary.to_csv(out / "summary.csv", index=False)
    table = reallocation_table(rows)
    table.to_csv(out / "reallocation.csv", index=False)
    agg = reallocation_summary(table)
    agg.to_csv(out / "reallocation_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary.round(2).to_string(index=False))
        print(agg.round(3).to_string(index=False))
    print(f"wrote {out / 'summary.csv'}, {out / 'reallocation.csv'}, {out / 'reallocation_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
