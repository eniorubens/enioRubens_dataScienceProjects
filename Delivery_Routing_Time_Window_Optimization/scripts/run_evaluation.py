"""Final evaluation (Layer A), pre-registered in reports/protocolo-avaliacao.md.

Frozen configuration, [evaluation] budget and replicates, OR-Tools plus deterministic baselines,
parallel by cell with incremental CSV output and resume. Touching the evaluation instances requires
``--confirm-eval``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "reports" / "protocolo-avaliacao.md"
FROZEN = ROOT / "outputs" / "frozen" / "selected_config.json"
PROTOCOL_D = ROOT / "reports" / "protocolo-fase-d.md"
FROZEN_V1_RESULTS = ROOT / "outputs" / "frozen" / "evaluation" / "results.csv"
V2_LABEL = "v2, pós-hoc"
V2_SCALE = 10000  # protocolo-v2 / protocolo-fase-d: escala v2 (com custo fixo relativo k = 10; ver Correção de 2026-11-19)
KEY_COLS = ["instance", "method", "replicate"]
BASELINES = ("greedy_deadline", "nearest_neighbor_tw")
RESULT_COLS = [
    "instance", "family", "method", "replicate", "feasible", "vehicles", "distance", "runtime_s", "status",
    "ref_vehicles", "ref_distance", "delta_vehicles", "gap_distance_pct", "distance_comparable",
    "beats_reference", "routes",
]


def run_cell(
    instance: str, method: str, replicate: int, cfg_dict: dict, archive: str, refs_path: str, version: str = "v1"
) -> dict:
    """Solve one (instance, method, replicate) cell and return one verified result row.

    ``version="v2"`` solves with the v2 solver (scale ``V2_SCALE``, relative fixed cost ``k``, ``fixed_cost``
    strategy); only OR-Tools cells exist in v2.
    """
    from delivery_routing_vrptw import (
        compare,
        exact,
        greedy_deadline,
        nearest_neighbor_tw,
        read_archive,
        read_references,
        verify,
    )
    from delivery_routing_vrptw.config import ExperimentConfig, SolverSection, solve_configured

    inst = read_archive(archive)[instance]
    ref = read_references(refs_path)[instance]
    if method == "ortools" and version == "v2":
        from delivery_routing_vrptw import SolverConfig, solve

        s = cfg_dict["solver"]
        sol = solve(inst, SolverConfig(
            first_solution=s["first_solution"], metaheuristic=s["metaheuristic"], time_limit_s=s["time_limit_s"],
            scale=V2_SCALE, fixed_cost_k=float(cfg_dict["v2"]["k_default"]),
        ))
    elif method == "ortools":
        cfg = ExperimentConfig(solver=SolverSection(**cfg_dict["solver"]))
        sol = solve_configured(inst, cfg)
    else:
        sol = {"greedy_deadline": greedy_deadline, "nearest_neighbor_tw": nearest_neighbor_tw}[method](inst)
    report = verify(inst, sol.routes, exact(inst))
    comp = compare(report, ref)
    row = {"instance": instance, "family": inst.family, "method": method, "replicate": replicate}
    row["feasible"] = report.feasible
    row["vehicles"] = report.vehicles
    row["distance"] = report.distance
    row["runtime_s"] = sol.runtime_s
    row["status"] = sol.meta.get("status", "")
    for k in ("ref_vehicles", "ref_distance", "delta_vehicles", "gap_distance_pct", "distance_comparable",
              "beats_reference"):
        row[k] = comp[k]
    row["routes"] = json.dumps(sol.routes)
    return row


def write_tables(results: Path, out: Path, version: str = "v1") -> None:
    """Write ``instance_table.csv`` and ``aggregate_table.csv`` from the results file.

    In v2 the baselines are not re-run: their rows come from the frozen v1 results, restricted to the
    instances present in the v2 results.
    """
    from delivery_routing_vrptw.evaluation import aggregate_table, instance_table

    rows = pd.read_csv(results)
    if version == "v2":
        v1 = pd.read_csv(FROZEN_V1_RESULTS)
        base = v1[v1["method"].isin(BASELINES) & v1["instance"].isin(rows["instance"].unique())]
        rows = pd.concat([rows, base], ignore_index=True)
    inst = instance_table(rows)
    agg = aggregate_table(rows)
    inst.to_csv(out / "instance_table.csv", index=False)
    agg.to_csv(out / "aggregate_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.2f}".format):
        print("\n== instance table (median replicate; best_of_R and variability to the right) ==", flush=True)
        print(inst.to_string(index=False), flush=True)
        print("\n== aggregate table ==", flush=True)
        print(agg.to_string(index=False), flush=True)


def evaluation_fields(cfg, version: str) -> dict:
    """Return the effective configuration that determines the evaluation results (the fingerprint input).

    Includes the process count (wall-clock limits make results depend on CPU contention); excludes
    timestamps and paths. v1 uses the absolute fixed cost of ``SolverConfig``; v2 the
    scale ``V2_SCALE`` with the relative rule ``k``.
    """
    from delivery_routing_vrptw import runs
    from delivery_routing_vrptw.config import solver_config

    sc = solver_config(cfg)
    if version == "v2":
        sc = replace(sc, scale=V2_SCALE, fixed_cost_k=float(cfg.v2.k_default))
    return {
        "runner": "run_evaluation", "version": version, "workers": cfg.solver.workers, **runs.solver_fields(sc),
        "fleet_strategy": cfg.solver.fleet_strategy, "replicates": cfg.evaluation.replicates,
        "include_baselines": bool(cfg.evaluation.include_baselines and version != "v2"),
    }


def plan_cells(names: list[str], replicates: int, include_baselines: bool, version: str, results: Path):
    """Return ``(cells, todo)``: all cells of the run and those not yet in ``results`` (resume by key)."""
    cells = [(n, "ortools", r) for n in names for r in range(1, replicates + 1)]
    if include_baselines and version != "v2":
        cells += [(n, m, 1) for n in names for m in BASELINES]
    done: set[tuple[str, str, int]] = set()
    if results.exists() and results.stat().st_size > 0:
        prev = pd.read_csv(results, usecols=KEY_COLS)
        done = {(str(i), str(m), int(r)) for i, m, r in prev[KEY_COLS].values}
    return cells, [c for c in cells if c not in done]


def main() -> None:
    """Run the pre-registered evaluation with resume and incremental CSV output."""
    from delivery_routing_vrptw import read_split, runs
    from delivery_routing_vrptw.config import add_config_args, load_config, resolve_config, select_instances
    from delivery_routing_vrptw.evaluation import v2_eval_guard

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--replicates", type=int, default=None, help="override evaluation.replicates")
    parser.add_argument("--instances", default=None, help="comma list of instance names within the set")
    parser.add_argument("--out", default=None, help="default: outputs/evaluation (v1), outputs/evaluation_v2 (v2)")
    parser.add_argument("--confirm-eval", action="store_true", help="required to touch the evaluation instances")
    parser.add_argument("--version", choices=("v1", "v2"), default="v1", help="v2 = post-hoc re-evaluation")
    parser.add_argument(
        "--post-hoc", action="store_true",
        help=f"v2 on the evaluation instances also needs this flag (see {PROTOCOL_D.relative_to(ROOT).as_posix()})",
    )
    runs.add_resume_args(parser)
    parser.add_argument("--summarize-only", action="store_true", help="only rebuild the tables from results.csv")
    args = parser.parse_args()
    v2 = args.version == "v2"

    out = Path(args.out or ("outputs/evaluation_v2" if v2 else "outputs/evaluation"))
    if not out.is_absolute():
        out = ROOT / out
    results = out / "results.csv"
    if args.summarize_only:
        if not results.exists():
            raise SystemExit(f"nothing to summarize: {results} not found")
        write_tables(results, out, args.version)
        return

    file_cfg = load_config(args.config)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["config"]
    mine = {k: getattr(file_cfg.solver, k) for k in frozen}
    if mine != frozen:
        raise SystemExit(f"refused: [solver] differs from the frozen config {FROZEN}: {mine} != {frozen}")

    cfg = resolve_config(args)
    ev = cfg.evaluation
    cfg = replace(cfg, solver=replace(cfg.solver, time_limit_s=ev.time_limit_s))
    if v2:
        try:
            v2_eval_guard(ev.instance_set, args.confirm_eval, args.post_hoc, PROTOCOL_D.relative_to(ROOT).as_posix())
        except PermissionError as exc:
            raise SystemExit(str(exc)) from exc
        if cfg.solver.fleet_strategy != "fixed_cost" or V2_SCALE not in cfg.v2.scales or cfg.v2.k_default != 10:
            raise SystemExit("refused: [solver]/[v2] do not match the v2 configuration of reports/protocolo-fase-d.md")
        if ev.instance_set != "dev" and (ev.time_limit_s != 120 or ev.replicates != 5):
            raise SystemExit("refused: the v2 evaluation of non-dev instances is fixed at 120 s x 5 replicates")
    print("effective config:", json.dumps(cfg.to_dict()), flush=True)

    split = read_split(ROOT / "data" / "reference" / "split.csv")
    if ev.instance_set != "dev" and not args.confirm_eval:
        raise SystemExit(
            f"refused: instance_set={ev.instance_set!r} touches the evaluation instances. They are opened only "
            f"for the pre-registered run described in {PROTOCOL}; pass --confirm-eval to proceed."
        )
    try:
        names = select_instances(
            split, ev.instance_set, args.instances.split(",") if args.instances else None, args.confirm_eval
        )
    except (PermissionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    archive = str(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")
    refs_path = str(ROOT / "data" / "reference" / "sintef_bks_100.csv")

    import numpy
    import ortools

    meta = {
        "timestamp": datetime.now().astimezone().isoformat(), "config": cfg.to_dict(),
        "config_file": args.config or "config/experiment.toml", "protocol": str(PROTOCOL),
        "frozen_config": str(FROZEN), "instances": names,
        "ortools": ortools.__version__, "python": platform.python_version(), "numpy": numpy.__version__,
        "pandas": pd.__version__, "cpu_count": os.cpu_count(), "argv": sys.argv,
    }
    if v2:
        meta.update({
            "version": "v2", "label": V2_LABEL, "post_hoc": True, "protocol": str(PROTOCOL_D),
            "protocol_v1": str(PROTOCOL), "protocol_v2": str(ROOT / cfg.v2.protocol),
            "scale": V2_SCALE, "fixed_cost_k": cfg.v2.k_default, "baselines": f"reused from {FROZEN_V1_RESULTS}",
        })
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, evaluation_fields(cfg, args.version), meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    cells, todo = plan_cells(names, ev.replicates, ev.include_baselines, args.version, results)
    total = len(todo)
    print(f"cells: {len(cells)} total, {len(cells) - total} already done, {total} to run", flush=True)

    if todo:
        k = 0
        t0 = time.perf_counter()
        cfg_dict = cfg.to_dict()
        with ProcessPoolExecutor(max_workers=cfg.solver.workers) as pool:
            futures = {pool.submit(run_cell, n, m, r, cfg_dict, archive, refs_path, args.version): (n, m, r) for n, m, r in todo}
            for fut in as_completed(futures):
                row = fut.result()
                runs.append_rows(results, pd.DataFrame([row], columns=RESULT_COLS), fingerprint)
                k += 1
                print(
                    f"done {k}/{total} {row['instance']} {row['method']} rep={row['replicate']} "
                    f"feasible={row['feasible']} vehicles={row['vehicles']} (ref {row['ref_vehicles']}) "
                    f"distance={row['distance']:.2f} runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
        print(f"finished in {time.perf_counter() - t0:.0f}s -> {results}", flush=True)
    write_tables(results, out, args.version)


if __name__ == "__main__":
    main()
