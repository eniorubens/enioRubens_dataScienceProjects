"""Layer B: customer time-window widening scenario (business layer; no SINTEF comparison)."""

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
KEY_COLS = ["instance", "factor", "mode"]


def run_cell(instance: str, factor: float, mode: str, time_limit: float, cfg_dict: dict, archive: str) -> dict:
    """Widen the windows, solve with the frozen config and verify against the widened instance."""
    import numpy as np

    from delivery_routing_vrptw import exact, read_archive, verify, widen_windows
    from delivery_routing_vrptw.config import ExperimentConfig, SolverSection, solve_configured

    cfg = ExperimentConfig(solver=SolverSection(**cfg_dict["solver"]))
    plain = read_archive(archive)[instance]
    inst = widen_windows(plain, factor, mode)
    if factor == 1.0:
        same = (inst.name == plain.name and np.array_equal(inst.ready, plain.ready)
                and np.array_equal(inst.due, plain.due))
        assert same, "factor 1.0 must be the plain benchmark instance"
    sol = solve_configured(inst, cfg, time_limit_s=time_limit)
    report = verify(inst, sol.routes, exact(inst))
    row = {"instance": instance, "family": plain.family, "factor": factor, "mode": mode,
           "scenario_name": inst.name, "time_limit_s": time_limit}
    row["feasible"] = report.feasible
    row["vehicles"] = report.vehicles
    row["distance"] = report.distance
    row["duration"] = report.duration
    row["waiting"] = report.waiting
    row["runtime_s"] = sol.runtime_s
    row["status"] = sol.meta.get("status", "")
    row["routes"] = json.dumps(sol.routes)
    return row


def main() -> None:
    """Run the widening scenarios with resume and incremental CSV output."""
    from delivery_routing_vrptw import read_split, runs, window_scenario_summary
    from delivery_routing_vrptw.config import add_config_args, resolve_config, select_instances, solver_config

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser, factors=True)
    parser.add_argument("--instances", default=None, help="comma list of instance names within the set")
    parser.add_argument("--out", default="outputs/window_scenarios")
    parser.add_argument("--allow-eval", action="store_true", help="permit instance_set eval/all")
    runs.add_resume_args(parser)
    args = parser.parse_args()

    cfg = resolve_config(args)
    ww = cfg.window_widening
    print("effective config:", json.dumps(cfg.to_dict()), flush=True)
    split = read_split(ROOT / "data" / "reference" / "split.csv")
    if ww.instance_set != "dev":
        print("WARNING: the evaluation set must only be opened after the protocol says so.", flush=True)
    try:
        names = select_instances(
            split, ww.instance_set, args.instances.split(",") if args.instances else None, args.allow_eval
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
        "timestamp": datetime.now().astimezone().isoformat(), "config": cfg.to_dict(),
        "config_file": args.config or "config/experiment.toml", "instances": names,
        "ortools": ortools.__version__, "python": platform.python_version(), "numpy": numpy.__version__,
        "cpu_count": os.cpu_count(), "argv": sys.argv,
    }
    fields = {"runner": "run_window_scenarios", "workers": cfg.solver.workers, "fleet_strategy": cfg.solver.fleet_strategy, "mode": ww.mode,
              **runs.solver_fields(solver_config(cfg, ww.time_limit_s))}  # the factor is a cell key
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, fields, meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    cells = [(n, float(f), ww.mode) for n in names for f in ww.factors]
    done: set[tuple[str, float, str]] = set()
    if results.exists() and results.stat().st_size > 0:
        prev = pd.read_csv(results, usecols=KEY_COLS)
        done = {(str(i), float(f), str(m)) for i, f, m in prev[KEY_COLS].values}
    todo = [c for c in cells if c not in done]
    total = len(todo)
    print(f"cells: {len(cells)} total, {len(cells) - total} already done, {total} to run", flush=True)

    k = 0
    t0 = time.perf_counter()
    cfg_dict = cfg.to_dict()
    if todo:
        with ProcessPoolExecutor(max_workers=cfg.solver.workers) as pool:
            futures = {
                pool.submit(run_cell, n, f, m, ww.time_limit_s, cfg_dict, archive): (n, f, m)
                for n, f, m in todo
            }
            for fut in as_completed(futures):
                row = fut.result()
                runs.append_rows(results, pd.DataFrame([row]), fingerprint)
                k += 1
                print(
                    f"done {k}/{total} {row['scenario_name']} {row['mode']} feasible={row['feasible']} "
                    f"vehicles={row['vehicles']} distance={row['distance']:.2f} runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
        print(f"finished in {time.perf_counter() - t0:.0f}s -> {results}", flush=True)

    summary = window_scenario_summary(pd.read_csv(results))
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round({"distance": 2, "distance_change_pct": 2}).to_string(index=False))
    print(f"wrote {out / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
