"""Phase C1: robustness of the frozen evaluation plans under random delays (reports/protocolo-fase-c.md).

No solver runs. The plans are the median OR-Tools replicate of every evaluation instance and the two
baselines (``outputs/frozen/evaluation/results.csv``), executed as planned (no re-planning) under travel
and service shocks with Monte Carlo draws. Reading the final plans is descriptive use only; nothing is tuned.

Writes, under ``outputs/phase_c/C1``: ``results.csv`` (instance x method x scenario),
``slack_by_customer.csv``, ``summary_by_family_method.csv``, ``slack_lateness.csv`` (late rate by nominal
slack bin), ``slack_correlation.csv`` and ``run_metadata.json``. Re-running resumes: instances already in
``results.csv`` are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ortools", "greedy_deadline", "nearest_neighbor_tw")
SLACK_EDGES = [-np.inf, 1e-9, 5, 10, 20, 50, 100, np.inf]
SLACK_LABELS = ["0", "(0, 5]", "(5, 10]", "(10, 20]", "(20, 50]", "(50, 100]", ">100"]
METRICS = [
    "late_customer_share", "routes_with_late_share", "draws_with_late_share", "mean_lateness_late",
    "p95_lateness_late", "mean_lateness_all", "late_return_share", "nominal_slack_mean", "nominal_slack_min",
]


def pick_plans(results: pd.DataFrame, instance: str) -> dict[str, list[list[int]] | None]:
    """Return method -> routes of the plan to stress: median OR-Tools replicate and the baselines.

    A plan that is infeasible at nominal conditions is returned as ``None`` (it is not stressed).
    """
    from delivery_routing_vrptw.evaluation import median_position, sort_replicates

    sub = results[results["instance"] == instance]
    plans: dict[str, list[list[int]] | None] = {}
    for method in METHODS:
        g = sub[sub["method"] == method]
        if g.empty:
            continue
        row = sort_replicates(g).iloc[median_position(len(g))] if method == "ortools" else g.iloc[0]
        plans[method] = list(row["routes"]) if bool(row["feasible"]) else None
    return plans


def run_instance(instance: str, plans: dict, scenarios: list[dict], draws: int, seed: int, archive: str) -> tuple:
    """Stress every plan of one instance under every scenario; return result rows and per-customer rows."""
    from delivery_routing_vrptw import exact, read_archive
    from delivery_routing_vrptw.robustness import ShockScenario, monte_carlo

    inst = read_archive(archive, names=[instance])[instance]
    matrix = exact(inst)
    rows, slack_rows = [], []
    for method, routes in plans.items():
        if routes is None:
            rows.append({"instance": instance, "family": inst.family, "method": method, "scenario": "nominal",
                         "plan_feasible": False})
            continue
        stream = [seed, zlib.crc32(instance.encode()), zlib.crc32(method.encode())]
        for sc in scenarios:
            res = monte_carlo(inst, routes, ShockScenario(**sc), draws, stream, matrix)
            rows.append({"instance": instance, "family": inst.family, "method": method, "scenario": sc["name"],
                         "plan_feasible": True, **res.metrics})
            slack_rows.extend(
                {"instance": instance, "method": method, "scenario": sc["name"], "customer": int(c),
                 "slack": float(s), "late_rate": float(r)}
                for c, s, r in zip(res.customers, res.slack, res.late_rate, strict=True)
            )
    return rows, slack_rows


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Average the plan metrics over instances, by scope (overall or family), method and scenario."""
    ok = results[results["plan_feasible"]]
    out = []
    for scope, g in [("overall", ok), *[(f, d) for f, d in ok.groupby("family", sort=True)]]:
        for (method, scenario), part in g.groupby(["method", "scenario"], sort=False):
            out.append({"scope": scope, "method": method, "scenario": scenario, "n_plans": len(part),
                        **{m: part[m].mean() for m in METRICS}})
    return pd.DataFrame(out)


def slack_tables(slack: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the late rate by nominal-slack bin and the Spearman correlation between slack and late rate."""
    s = slack.copy()
    s["slack_bin"] = pd.cut(s["slack"], bins=SLACK_EDGES, labels=SLACK_LABELS, right=True)
    binned = (
        s.groupby(["scenario", "method", "slack_bin"], observed=True)
        .agg(n_customers=("customer", "size"), late_rate=("late_rate", "mean"))
        .reset_index()
    )
    corr = (
        s.groupby(["scenario", "method"])
        .apply(lambda g: g["slack"].rank().corr(g["late_rate"].rank()), include_groups=False)
        .rename("spearman_slack_late_rate")
        .reset_index()
    )
    return binned, corr


def main() -> None:
    """Run C1 with per-instance resume, then write the summaries."""
    from delivery_routing_vrptw import paths, read_split, runs, v2
    from delivery_routing_vrptw.config import load_config
    from delivery_routing_vrptw.results import assert_eval_only, load_evaluation
    from delivery_routing_vrptw.robustness import protocol_scenarios

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="TOML file (default: config/experiment.toml)")
    parser.add_argument("--results", default=None,
                        help="plans file (default: the frozen evaluation results.csv; a custom file is for smoke "
                             "tests and skips the evaluation-only check)")
    parser.add_argument("--out", default=None, help="default: outputs/phase_c/C1")
    parser.add_argument("--instances", default=None, help="comma list of instances")
    parser.add_argument("--draws", type=int, default=None, help="override [phase_c].c1_draws (smoke tests)")
    parser.add_argument("--workers", type=int, default=None, help="override [phase_c].workers")
    runs.add_resume_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pc = cfg.phase_c
    draws = args.draws if args.draws is not None else pc.c1_draws
    workers = args.workers if args.workers is not None else pc.workers
    split = read_split(ROOT / "data" / "reference" / "split.csv")
    src = Path(args.results) if args.results else paths.result_paths(ROOT)["evaluation"]
    if args.results:
        results = pd.read_csv(src, keep_default_na=False, na_values=[""])
        results["routes"] = results["routes"].map(json.loads)
    else:
        results = load_evaluation(src, split)
        assert_eval_only(results["instance"].unique(), split)
    names = sorted(results["instance"].unique())
    if args.instances:
        wanted = [n.strip().upper() for n in args.instances.split(",") if n.strip()]
        missing = [n for n in wanted if n not in names]
        if missing:
            raise SystemExit(f"instances not in the plans file: {missing}")
        names = sorted(wanted)
    scenarios = [
        {"name": s.name, "travel_a": s.travel_a, "service_b": s.service_b} for s in protocol_scenarios()
    ]
    out = Path(args.out) if args.out else ROOT / "outputs" / "phase_c" / "C1"
    if not out.is_absolute():
        out = ROOT / out
    res_path, slack_path = out / "results.csv", out / "slack_by_customer.csv"
    archive = str(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")
    meta = v2.run_metadata(
        cfg.to_dict(), args.config or "config/experiment.toml", str(ROOT / pc.protocol), names, block="C1",
        draws=draws, seed=pc.seed, plans_file=str(src), scenarios=scenarios, workers=workers,
    )
    fields = {"runner": "run_robustness", "workers": workers, "draws": draws, "seed": pc.seed, "scenarios": scenarios,
              "plans_sha256": runs.file_fingerprint(src)}
    try:  # resume guard first; run_metadata.json is written only after it passes
        fingerprint = runs.start_run(out, fields, meta, args.allow_legacy_resume)
    except runs.ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    done: set[str] = set()
    if res_path.exists() and res_path.stat().st_size > 0:
        done = set(pd.read_csv(res_path, usecols=["instance"])["instance"])
    todo = [n for n in names if n not in done]
    print(f"instances: {len(names)} total, {len(names) - len(todo)} already done, {len(todo)} to run, "
          f"draws={draws}, workers={workers}", flush=True)
    t0 = time.perf_counter()
    if todo:
        header_slack = not slack_path.exists() or slack_path.stat().st_size == 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_instance, n, pick_plans(results, n), scenarios, draws, pc.seed, archive): n
                for n in todo
            }
            for k, fut in enumerate(as_completed(futures), 1):
                rows, slack_rows = fut.result()
                runs.append_rows(res_path, pd.DataFrame(rows), fingerprint)
                pd.DataFrame(slack_rows).to_csv(slack_path, mode="a", header=header_slack, index=False)
                header_slack = False
                print(f"done {k}/{len(todo)} {futures[fut]}", flush=True)
    print(f"stress runs finished in {time.perf_counter() - t0:.0f}s", flush=True)

    all_rows = pd.read_csv(res_path)
    all_rows["plan_feasible"] = all_rows["plan_feasible"].astype(str).str.lower().eq("true")
    summary = summarise(all_rows)
    summary.to_csv(out / "summary_by_family_method.csv", index=False)
    binned, corr = slack_tables(pd.read_csv(slack_path))
    binned.to_csv(out / "slack_lateness.csv", index=False)
    corr.to_csv(out / "slack_correlation.csv", index=False)
    skipped = all_rows[~all_rows["plan_feasible"]]
    if len(skipped):
        print(f"plans not stressed (infeasible at nominal): {skipped[['instance', 'method']].values.tolist()}")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary[summary["scope"] == "overall"].round(4).to_string(index=False))
    print(f"wrote results and summaries to {out}", flush=True)


if __name__ == "__main__":
    main()
