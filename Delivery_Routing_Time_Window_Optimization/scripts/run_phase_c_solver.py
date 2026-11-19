"""Phase C2: operational scenarios with the v2 solver on the development instances (reports/protocolo-fase-c.md).

Blocks (run them in this order, one at a time):

* ``capacity``: Q x {1.0, 0.9, 0.8}, rigid model. The factor-1.0 cells are the **reference plans**
  (v2 configuration on the unmodified instance, same budget and replicates as the scenarios), computed here
  and reused by the other two blocks.
* ``fleet_priority``: K = reference vehicles - {1, 2}, every customer droppable at a penalty; 20% priority
  customers at 10x the common penalty against the same K with a uniform penalty.
* ``soft_windows``: K = reference vehicles - 1, soft due dates with a cost per unit of lateness of 1x and 10x
  the cost per unit of distance; the depot return stays hard.

Only development instances can be run: ``--set eval`` and ``--set all`` are refused. Re-running resumes:
cells already in ``results.csv`` are skipped.
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


def run_cell(cell: dict, cfg, time_limit: float, archive: str) -> dict:
    """Solve one cell against the original instance and return one verified result row."""
    from delivery_routing_vrptw import phase_c, read_archive

    inst = read_archive(archive, names=[cell["instance"]])[cell["instance"]]
    config = phase_c.phase_c_solver_config(cfg, time_limit)
    return phase_c.run_solver_cell(inst, cell, config, cfg.phase_c)


def write_summaries(block: str, rows: pd.DataFrame, ref: pd.DataFrame, out: Path) -> dict[str, pd.DataFrame]:
    """Write ``summary.csv`` and ``aggregate.csv`` of a block and return them."""
    from delivery_routing_vrptw import phase_c

    if block == phase_c.CAPACITY:
        summary = phase_c.capacity_summary(rows, ref)
        aggregate = phase_c.capacity_aggregate(summary)
    elif block == phase_c.FLEET_PRIORITY:
        summary = phase_c.fleet_priority_summary(rows, ref)
        aggregate = phase_c.fleet_priority_aggregate(summary)
    else:
        summary = phase_c.soft_summary(rows, ref)
        aggregate = phase_c.soft_aggregate(summary)
    summary.to_csv(out / "summary.csv", index=False)
    aggregate.to_csv(out / "aggregate.csv", index=False)
    tables = {"summary": summary, "aggregate": aggregate}
    if block != phase_c.FLEET_PRIORITY:
        tables["infeasible"] = phase_c.infeasible_breakdown(summary)
        tables["infeasible"].to_csv(out / "infeasible.csv", index=False)
    return tables


def phase_c_fields(cfg, block: str, time_limit: float, reference: Path | None) -> dict:
    """Return the effective C2 configuration that determines the results (the fingerprint input).

    Includes the process count (wall-clock limits make results depend on CPU contention); excludes paths; the reference plans enter through the hash of their file.
    """
    from delivery_routing_vrptw import phase_c, runs

    pc = {k: v for k, v in cfg.phase_c.__dict__.items() if k not in ("workers", "protocol")}
    return {
        "runner": "run_phase_c_solver", "workers": cfg.phase_c.workers, "block": block, "phase_c": pc,
        "fleet_strategy": cfg.solver.fleet_strategy,
        **runs.solver_fields(phase_c.phase_c_solver_config(cfg, time_limit)),
        "reference_sha256": runs.file_fingerprint(reference) if reference is not None else None,
    }


def main() -> None:
    """Run one C2 block with resume and incremental CSV output, then write its summaries."""
    from delivery_routing_vrptw import phase_c, read_split, runs, v2
    from delivery_routing_vrptw.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, choices=phase_c.SOLVER_BLOCKS)
    parser.add_argument("--out", default=None, help="default: outputs/phase_c/C2/<block>")
    parser.add_argument("--config", default=None, help="TOML file (default: config/experiment.toml)")
    parser.add_argument("--reference", default=None,
                        help="capacity results.csv holding the reference plans (default: outputs/phase_c/C2/"
                             "capacity/results.csv); only used by fleet_priority and soft_windows")
    parser.add_argument("--instances", default=None, help="comma list of development instances")
    parser.add_argument("--set", dest="instance_set", default="dev", choices=["dev", "eval", "all"],
                        help="only 'dev' is accepted (kept so that a request for eval is refused explicitly)")
    parser.add_argument("--time-limit", type=float, default=None, help="override [phase_c].time_limit_s (smoke tests)")
    parser.add_argument("--replicates", type=int, default=None, help="override [phase_c].replicates")
    parser.add_argument("--workers", type=int, default=None, help="override [phase_c].workers")
    runs.add_resume_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pc = cfg.phase_c
    time_limit = args.time_limit if args.time_limit is not None else pc.time_limit_s
    reps = args.replicates if args.replicates is not None else pc.replicates
    workers = args.workers if args.workers is not None else pc.workers
    split = read_split(ROOT / "data" / "reference" / "split.csv")
    try:
        names = v2.select_dev_instances(
            split, args.instances.split(",") if args.instances else None, args.instance_set
        )
    except (PermissionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    cfg = replace(cfg, phase_c=replace(pc, time_limit_s=time_limit, replicates=reps, workers=workers))
    print("effective config:", json.dumps(cfg.to_dict()), flush=True)

    out = Path(args.out) if args.out else Path("outputs") / "phase_c" / "C2" / args.block
    if not out.is_absolute():
        out = ROOT / out
    results = out / "results.csv"

    ref = None
    ref_path = None
    ref_vehicles = None
    ref_source = "computed in this block (capacity factor 1.0)"
    if args.block != phase_c.CAPACITY:
        ref_path = Path(args.reference) if args.reference else ROOT / "outputs" / "phase_c" / "C2" / "capacity" / "results.csv"
        if not ref_path.exists():
            raise SystemExit(f"reference plans not found: {ref_path}. Run --block capacity first.")
        ref = phase_c.reference_table(phase_c.read_results(ref_path))
        ref_vehicles = dict(zip(ref["instance"], ref["ref_vehicles"], strict=True))
        ref_source = str(ref_path)
        print(f"reference plans: {len(ref)} instances from {ref_path}", flush=True)
    try:
        cells = phase_c.solver_cells(args.block, names, cfg.phase_c, reps, ref_vehicles)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    archive = str(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")
    meta = v2.run_metadata(
        cfg.to_dict(), args.config or "config/experiment.toml", str(ROOT / pc.protocol), names, block=args.block,
        time_limit_s=time_limit, workers=workers, replicates=reps, reference_source=ref_source,
        representative_order=["not feasible", *phase_c.ordering(args.block), "replicate"],
        solver="v2: fixed_cost + first_solution + metaheuristic of [solver], scale/k of [phase_c]",
    )
    fields = phase_c_fields(cfg, args.block, time_limit, ref_path if ref is not None else None)
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, fields, meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    done: set[tuple] = set()
    if results.exists() and results.stat().st_size > 0:
        prev = phase_c.read_results(results)
        done = {phase_c.cell_key(r) for r in prev.to_dict(orient="records")}
    todo = [c for c in cells if phase_c.cell_key(c) not in done]
    print(f"cells: {len(cells)} total, {len(cells) - len(todo)} already done, {len(todo)} to run, workers={workers}",
          flush=True)
    t0 = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_cell, c, cfg, time_limit, archive): c for c in todo}
            for k, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                runs.append_rows(results, phase_c.as_frame(row), fingerprint)
                extra = {
                    "capacity": f"vehicles={row['vehicles']} distance={row['distance']:.1f}",
                    "fleet_priority": f"dropped={row.get('n_dropped')} (priority {row.get('n_dropped_priority')})",
                    "soft_windows": f"late={row.get('n_late')} total_lateness={row.get('total_lateness')}",
                }[args.block]
                print(f"done {k}/{len(todo)} {row['instance']} level={row['level']} variant={row['variant']} "
                      f"rep={row['replicate']} feasible={row['feasible']} {extra} status={row['status']}",
                      flush=True)
        print(f"finished in {time.perf_counter() - t0:.0f}s -> {results}", flush=True)

    rows = phase_c.read_results(results)
    if ref is None:
        ref = phase_c.reference_table(rows)
    tables = write_summaries(args.block, rows, ref, out)
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(tables["aggregate"].round(3).to_string(index=False))
        if "infeasible" in tables:
            print(tables["infeasible"].to_string(index=False))
    print(f"wrote summaries to {out}", flush=True)


if __name__ == "__main__":
    main()
