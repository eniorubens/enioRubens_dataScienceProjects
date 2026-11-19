"""Protocol v2, block B5: layer-B scenarios with warm start between levels (reports/protocolo-v2.md).

Same levels, replicates, time budget (30 s) and solver configuration as v1, but each level starts from the
solution of the previous (less permissive) level, which is feasible for it by construction. Every replicate
runs its own independent ladder. Writes the lexicographic summaries and a comparison with the frozen v1 run.
Only development instances can be run.
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


def run_ladder(scenario: str, instance: str, replicate: int, levels: list[float], mode: str,
               time_limit: float, cfg_dict: dict, archive: str) -> list[dict]:
    """Solve one instance's ladder of levels (cold, then warm) and return one verified row per level."""
    import numpy as np

    from delivery_routing_vrptw import (
        allow_early,
        exact,
        read_archive,
        schedule_metrics,
        tolerance_label,
        verify,
        widen_windows,
    )
    from delivery_routing_vrptw.config import ExperimentConfig, SolverSection, solver_config
    from delivery_routing_vrptw.scenarios import solve_ladder

    cfg = ExperimentConfig(solver=SolverSection(**cfg_dict["solver"]))
    plain = read_archive(archive, names=[instance])[instance]
    if scenario == "windows":
        insts = [widen_windows(plain, f, mode) for f in levels]
    else:
        insts = [allow_early(plain, t) for t in levels]
    base = insts[0]
    assert (base.name == plain.name and np.array_equal(base.ready, plain.ready)
            and np.array_equal(base.due, plain.due)), "level 0 must be the plain benchmark instance"
    sols = solve_ladder(insts, solver_config(cfg, time_limit))
    rows = []
    for i, (level, inst, sol) in enumerate(zip(levels, insts, sols, strict=True)):
        matrix = exact(inst)
        report = verify(inst, sol.routes, matrix)
        row = {"instance": instance, "family": plain.family}
        if scenario == "windows":
            row.update(factor=level, mode=mode)
        else:
            row.update(tolerance_label=tolerance_label(level), tolerance=level)
        row.update(replicate=replicate, level_index=i, scenario_name=inst.name, time_limit_s=time_limit,
                   feasible=report.feasible)
        if scenario == "windows":
            row.update(vehicles=report.vehicles, distance=report.distance, duration=report.duration,
                       waiting=report.waiting)
        else:
            row.update(schedule_metrics(inst, sol.routes, matrix))
        row["runtime_s"] = sol.runtime_s
        row["status"] = sol.meta.get("status", "")
        row["warm_start"] = sol.meta.get("warm_start", "")
        row["routes"] = json.dumps(sol.routes)
        rows.append(row)
    return rows


def main() -> None:
    """Run the warm-started ladders with resume, then write summaries and the v1 comparison."""
    from delivery_routing_vrptw import paths, read_split, runs, tolerance_label, v2
    from delivery_routing_vrptw.config import add_config_args, resolve_config, solver_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=["windows", "early"])
    add_config_args(parser, factors=True, early=True)
    parser.add_argument("--instances", default=None, help="comma list of development instances")
    parser.add_argument("--out", default=None, help="default: outputs/v2/B5_<scenario>")
    runs.add_resume_args(parser)
    args = parser.parse_args()

    cfg = resolve_config(args)
    cv2 = cfg.v2
    scenario = args.scenario
    time_limit = args.time_limit if args.time_limit is not None else cv2.b5_time_limit_s
    workers = args.workers if args.workers is not None else cv2.workers
    split = read_split(ROOT / "data" / "reference" / "split.csv")
    try:
        names = v2.select_dev_instances(
            split, args.instances.split(",") if args.instances else None, args.instance_set or "dev"
        )
    except (PermissionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    cfg = replace(cfg, solver=replace(cfg.solver, time_limit_s=time_limit, workers=workers))
    print("effective config:", json.dumps(cfg.to_dict()), flush=True)

    if scenario == "windows":
        levels, mode, reps = sorted(cfg.window_widening.factors), cfg.window_widening.mode, 1
        level_names = [f"{f:g}" for f in levels]
    else:
        levels, mode, reps = cfg.early_delivery.levels(), "", cfg.early_delivery.replicates
        level_names = [tolerance_label(t) for t in levels]

    out = Path(args.out) if args.out else Path("outputs") / "v2" / f"B5_{scenario}"
    if not out.is_absolute():
        out = ROOT / out
    results = out / "results.csv"
    archive = str(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")
    meta = v2.run_metadata(
        cfg.to_dict(), args.config or "config/experiment.toml", str(ROOT / cv2.protocol), names,
        block=f"B5_{scenario}", scenario=scenario, levels=level_names, replicates=reps,
        time_limit_s=time_limit, workers=workers, warm_start="previous level",
    )
    # the ladder (levels in order, warm start between them) determines every row
    fields = {
        "runner": "run_v2_layerb_warm", "workers": workers, "scenario": scenario, "levels": level_names, "mode": mode,
        "replicates": reps, "fleet_strategy": cfg.solver.fleet_strategy, "warm_start": "previous level",
        **runs.solver_fields(solver_config(cfg, time_limit)),
    }
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, fields, meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    level_col = "factor" if scenario == "windows" else "tolerance_label"
    tasks = [(n, r) for n in names for r in range(reps)]
    done: set[tuple[str, int]] = set()
    if results.exists() and results.stat().st_size > 0:
        prev = pd.read_csv(results, dtype={"tolerance_label": str})
        prev["_lvl"] = prev[level_col].astype(str)
        complete = prev.groupby(["instance", "replicate"])["_lvl"].nunique()
        done = {(str(i), int(r)) for (i, r), c in complete.items() if c >= len(levels)}
        keep = prev[[(str(i), int(r)) in done for i, r in zip(prev["instance"], prev["replicate"], strict=True)]]
        if len(keep) != len(prev):  # drop the rows of an interrupted ladder, then re-run it whole
            keep.drop(columns="_lvl").to_csv(results, index=False)
    todo = [t for t in tasks if t not in done]
    total = len(todo)
    print(f"ladders: {len(tasks)} total, {len(tasks) - total} already done, {total} to run "
          f"({len(levels)} levels each), workers={workers}", flush=True)

    t0 = time.perf_counter()
    cfg_dict = cfg.to_dict()
    if todo:
        k = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_ladder, scenario, n, r, levels, mode, time_limit, cfg_dict, archive): (n, r)
                for n, r in todo
            }
            for fut in as_completed(futures):
                rows = fut.result()
                runs.append_rows(results, pd.DataFrame(rows), fingerprint)
                k += 1
                trail = " ".join(f"{r['vehicles']}/{r['warm_start']}" for r in rows)
                print(f"done {k}/{total} {rows[0]['instance']} rep={rows[0]['replicate']} "
                      f"vehicles/warm_start per level: {trail}", flush=True)
        print(f"finished in {time.perf_counter() - t0:.0f}s -> {results}", flush=True)

    rows = pd.read_csv(results, dtype={"tolerance_label": str})
    parts = v2.layerb_summary(scenario, rows)
    for name, table in parts.items():
        table.to_csv(out / f"{name}.csv", index=False)
    rp = paths.result_paths(ROOT)
    v1_path = rp["window_scenarios"] if scenario == "windows" else rp["early_delivery"]
    v1_rows = pd.read_csv(v1_path, dtype={"tolerance_label": str})
    comp = v2.compare_layerb(scenario, v1_rows, rows)
    comp["levels"].to_csv(out / "comparison_levels.csv", index=False)
    comp["nonmonotonic"].to_csv(out / "comparison_nonmonotonic.csv", index=False)
    warm = rows["warm_start"].value_counts().to_dict()
    (out / "comparison.json").write_text(json.dumps({
        "frozen_v1": str(v1_path), "instances": names, "warm_start_counts": warm,
        "nonmonotonic": comp["nonmonotonic"].to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(parts["by_level"].to_string(index=False))
        print(comp["levels"].round(3).to_string(index=False))
        print(comp["nonmonotonic"].to_string(index=False))
    print(f"warm_start counts: {warm}")
    print(f"wrote summaries and comparison to {out}", flush=True)


if __name__ == "__main__":
    main()
