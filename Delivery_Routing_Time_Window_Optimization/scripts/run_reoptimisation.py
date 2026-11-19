"""Phase C3: stability of reoptimisation after removing or inserting 10% of the customers (development only).

* ``removal``: 10% of the customers (seeded draw, recorded) leave. The instance without them is reoptimised
  **warm** (start = the reference plan without them) and **cold**. The reference plan comes from the C2
  ``capacity`` block (factor 1.0).
* ``insertion``: the plan of the 90% (cold run, ``plan90``) is the original plan; the full instance is then
  reoptimised **warm** (start = plan90 completed by cheapest insertion of the 10%; no dropping is ever
  allowed) and **cold**.

Reallocation (customers that change vehicle, pairs kept, arcs kept) is measured on the common customers in
original ids between the original and the reoptimised plan; the noise floor is the same measure between two
cold replicates. Only development instances can be run (``--set eval``/``all`` are refused). Re-running
resumes: cells already in ``results.csv`` are skipped.

Outputs: ``aggregate_representative.csv`` is the protocol aggregate (``phase_c.reopt_aggregate_representative``);
``aggregate.csv`` is the table pooled over all replicates, kept only for traceability with the 2026-11-18 run.
For the frozen folders the protocol aggregate is derived with ``scripts/derive_phase_c_corrections.py`` (no solver).
The fingerprint includes the process count (wall-clock limits make results depend on CPU contention).
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


def run_cell(cell: dict, cfg, time_limit: float, archive: str, original_routes: list[list[int]]) -> dict:
    """Solve one C3 cell against the scenario instance and return one verified result row."""
    from delivery_routing_vrptw import phase_c, read_archive

    inst = read_archive(archive, names=[cell["instance"]])[cell["instance"]]
    config = phase_c.phase_c_solver_config(cfg, time_limit)
    return phase_c.solve_reopt_cell(
        inst, cell["block"], cell["variant"], int(cell["replicate"]), config, cfg.phase_c, original_routes
    )


def reopt_fields(cfg, block: str, time_limit: float, reference: Path | None) -> dict:
    """Return the effective C3 configuration that determines the results (the fingerprint input).

    Includes the process count (wall-clock limits make results depend on CPU contention); excludes paths; the removal reference plans enter through the hash of their file.
    """
    from delivery_routing_vrptw import phase_c, runs

    pc = {k: v for k, v in cfg.phase_c.__dict__.items() if k not in ("workers", "protocol")}
    return {
        "runner": "run_reoptimisation", "workers": cfg.phase_c.workers, "block": block, "phase_c": pc,
        "fleet_strategy": cfg.solver.fleet_strategy,
        **runs.solver_fields(phase_c.phase_c_solver_config(cfg, time_limit)),
        "reference_sha256": runs.file_fingerprint(reference) if reference is not None else None,
    }


def main() -> None:
    """Run one C3 block with resume and incremental CSV output, then write its summaries."""
    from delivery_routing_vrptw import phase_c, read_split, runs, v2
    from delivery_routing_vrptw.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, choices=phase_c.REOPT_BLOCKS)
    parser.add_argument("--out", default=None, help="default: outputs/phase_c/C3/<block>")
    parser.add_argument("--config", default=None, help="TOML file (default: config/experiment.toml)")
    parser.add_argument("--reference", default=None,
                        help="C2 capacity results.csv with the reference plans (removal only; default: "
                             "outputs/phase_c/C2/capacity/results.csv)")
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

    out = Path(args.out) if args.out else Path("outputs") / "phase_c" / "C3" / args.block
    if not out.is_absolute():
        out = ROOT / out
    results = out / "results.csv"
    archive = str(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")

    originals: dict[str, list[list[int]]] = {}
    ref_path = None
    ref_source = "plan90 solved in this block (insertion)"
    if args.block == "removal":
        ref_path = Path(args.reference) if args.reference else ROOT / "outputs" / "phase_c" / "C2" / "capacity" / "results.csv"
        if not ref_path.exists():
            raise SystemExit(f"reference plans not found: {ref_path}. Run run_phase_c_solver.py --block capacity first.")
        ref = phase_c.reference_table(phase_c.read_results(ref_path))
        originals = dict(zip(ref["instance"], ref["ref_routes"], strict=True))
        missing = [n for n in names if n not in originals]
        if missing:
            raise SystemExit(f"no feasible reference plan for: {missing}")
        ref_source = str(ref_path)
    meta = v2.run_metadata(
        cfg.to_dict(), args.config or "config/experiment.toml", str(ROOT / pc.protocol), names, block=args.block,
        time_limit_s=time_limit, workers=workers, replicates=reps, reference_source=ref_source,
        reopt_share=pc.reopt_share, seed=pc.seed,
    )
    fields = reopt_fields(cfg, args.block, time_limit, ref_path)
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, fields, meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    def execute(cells: list[dict], label: str) -> None:
        done: set[tuple] = set()
        if results.exists() and results.stat().st_size > 0:
            done = {phase_c.cell_key(r) for r in phase_c.read_results(results).to_dict(orient="records")}
        todo = [c for c in cells if phase_c.cell_key(c) not in done]
        print(f"{label}: {len(cells)} cells, {len(cells) - len(todo)} already done, {len(todo)} to run, "
              f"workers={workers}", flush=True)
        t0 = time.perf_counter()
        if not todo:
            return
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_cell, c, cfg, time_limit, archive, originals.get(c["instance"], [])): c for c in todo
            }
            for k, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                runs.append_rows(results, phase_c.as_frame(row), fingerprint)
                print(f"done {k}/{len(todo)} {row['instance']} {row['variant']} rep={row['replicate']} "
                      f"feasible={row['feasible']} vehicles={row['vehicles']} distance={row['distance']:.1f} "
                      f"reassigned={row['realloc_customers_reassigned_share']} warm_start={row['warm_start']}"
                      f"/{row['warm_start_input']}", flush=True)
        print(f"{label} finished in {time.perf_counter() - t0:.0f}s", flush=True)

    if args.block == "insertion":
        execute(phase_c.reopt_cells("insertion", names, reps, "plan"), "plan90")
        plans = phase_c.read_results(results)
        plans = plans[plans["variant"] == "plan90"]
        bad = [n for n in names if not plans[(plans["instance"] == n) & plans["feasible"]].shape[0]]
        if bad:
            raise SystemExit(f"plan90 infeasible or missing for: {bad}")
        originals = {r.instance: json.loads(r.routes) for r in plans[plans["feasible"]].itertuples()}
    execute(phase_c.reopt_cells(args.block, names, reps, "reopt"), "reoptimisation")

    rows = phase_c.read_results(results)
    noise = phase_c.noise_table(rows)
    aggregate = phase_c.reopt_aggregate(rows, noise)  # pooled over all replicates (traceability only)
    representative = phase_c.reopt_aggregate_representative(rows, noise)  # protocol aggregate
    phase_c.reopt_summary(rows).to_csv(out / "summary.csv", index=False)
    noise.to_csv(out / "noise.csv", index=False)
    aggregate.to_csv(out / "aggregate.csv", index=False)
    representative.to_csv(out / "aggregate_representative.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print("== aggregate_representative.csv (protocol aggregate) ==")
        print(representative.round(4).to_string(index=False))
        print("(aggregate.csv: pooled over all replicates, kept for traceability with the 2026-11-18 run)")
    print(f"wrote summaries to {out}", flush=True)


if __name__ == "__main__":
    main()
