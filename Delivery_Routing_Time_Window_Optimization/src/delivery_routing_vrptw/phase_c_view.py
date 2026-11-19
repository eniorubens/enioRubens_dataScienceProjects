"""Display tables of phase C (robustness and operational scenarios; layer B, no SINTEF reference anywhere).

The numbers come from :mod:`delivery_routing_vrptw.phase_c` (summaries of the solver blocks and of the
reoptimisation) and from the frozen C1 summaries. Every scenario is compared with the reference plan of the same
instance (the v2 solver on the unmodified instance, 30 s, three replicates), never with a benchmark value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import phase_c
from .config import ExperimentConfig
from .io import FAMILIES
from .labels import get_labels

METHODS = ("ortools", "nearest_neighbor_tw", "greedy_deadline")
SCENARIO_ORDER = ("travel_0.05", "travel_0.10", "travel_0.20", "service", "travel_0.10+service")
SLACK_ORDER = ("0", "(0, 5]", "(5, 10]", "(10, 20]", "(20, 50]", "(50, 100]", ">100")
LEVEL_PRIORITY = ("K-1", "K-2")


def _scenario_text(name: str, lang: str) -> str:
    return get_labels(lang)["phase_c"][f"scenario_{name}"]


def _pct(values: pd.Series, digits: int = 1) -> pd.Series:
    return (100.0 * values).round(digits)


# --------------------------------------------------------------------------- configuration and coverage
def config_table(cfg: ExperimentConfig, lang: str = "en") -> pd.DataFrame:
    """Tabulate the effective phase C configuration read from ``config/experiment.toml``.

    Parameters
    ----------
    cfg : ExperimentConfig
        Output of :func:`delivery_routing_vrptw.config.load_config`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric`` and ``value``: solver configuration, budget, replicates, seed and the parameters of
        C1, C2 and C3.
    """
    names = get_labels(lang)["phase_c_cfg"]
    pc, solver = cfg.phase_c, cfg.solver
    items = {
        "solver": f"{solver.fleet_strategy} + {solver.first_solution} + {solver.metaheuristic}",
        "scale": f"{pc.scale}, k = {pc.fixed_cost_k:g}",
        "budget": f"{pc.time_limit_s:g} · {pc.replicates} · {pc.workers}",
        "seed": f"{pc.seed}",
        "c1": f"{pc.c1_draws} · " + ", ".join(f"{100 * a:g}%" for a in pc.c1_travel_a)
        + f" · {100 * pc.c1_service_b:g}% · {100 * pc.c1_combined_travel_a:g}%",
        "capacity": ", ".join(f"{100 * f:g}%" for f in pc.capacity_factors),
        "fleet": ", ".join(f"K − {r}" for r in pc.fleet_reductions)
        + f" · {100 * pc.priority_share:g}% · {pc.priority_penalty_multiplier}×",
        "soft": f"K − {pc.soft_fleet_reduction} · " + ", ".join(f"{m}×" for m in pc.soft_cost_multipliers),
        "reopt": f"{100 * pc.reopt_share:g}%",
    }
    return pd.DataFrame({"metric": [names[k] for k in items], "value": list(items.values())})


def block_overview(blocks: dict[str, pd.DataFrame], lang: str = "en") -> pd.DataFrame:
    """Audit the frozen solver blocks (C2 and C3): cells, instances, feasible cells and mean runtime.

    Parameters
    ----------
    blocks : dict
        Block name (``capacity``, ``fleet_priority``, ``soft_windows``, ``removal``, ``insertion``) -> cells from
        :func:`delivery_routing_vrptw.results.load_phase_c_cells`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per block: ``block`` (text), ``n_cells``, ``n_instances``, ``n_feasible`` and ``mean_runtime``
        (seconds).
    """
    text = get_labels(lang)["phase_c"]
    rows = []
    for name, df in blocks.items():
        rows.append({
            "block": text[f"block_{name}"], "n_cells": len(df), "n_instances": int(df["instance"].nunique()),
            "n_feasible": int(df["feasible"].sum()),
            "mean_runtime": round(float(pd.to_numeric(df["runtime_s"], errors="coerce").mean()), 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- C1
def robustness_integrity(cells: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Audit the C1 run: plans, scenarios and draws per method.

    Parameters
    ----------
    cells : pandas.DataFrame
        ``cells`` of :func:`delivery_routing_vrptw.results.load_robustness`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per method: ``method`` (text), ``n_instances``, ``n_plans_stressed`` (plans feasible at the
        nominal level), ``n_scenarios`` and ``n_draws`` (per plan and scenario).
    """
    names = get_labels(lang)["methods"]
    rows = []
    for method in METHODS:
        part = cells[cells["method"] == method]
        ok = part[part["plan_feasible"]]
        rows.append({
            "method": names[method], "n_instances": int(part["instance"].nunique()),
            "n_plans_stressed": int(ok["instance"].nunique()), "n_scenarios": int(ok["scenario"].nunique()),
            "n_draws": int(ok["n_draws"].max()) if len(ok) else 0,
        })
    return pd.DataFrame(rows)


def robustness_table(summary: pd.DataFrame, scope: str = "overall", lang: str = "en") -> pd.DataFrame:
    """Tabulate the C1 stress results by scenario and method (means over plans, in percent).

    Parameters
    ----------
    summary : pandas.DataFrame
        ``summary`` of :func:`delivery_routing_vrptw.results.load_robustness`.
    scope : str
        ``"overall"`` or a family.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per (scenario, method): ``scenario`` (text), ``method`` (text), ``n_plans``,
        ``late_customer_pct``, ``late_return_pct``, ``routes_with_late_pct``, ``mean_lateness_late`` and
        ``p95_lateness_late`` (time units, among the late customers).
    """
    names = get_labels(lang)["methods"]
    part = summary[summary["scope"] == scope]
    rows = []
    for scenario in SCENARIO_ORDER:
        for method in METHODS:
            r = part[(part["scenario"] == scenario) & (part["method"] == method)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "scenario": _scenario_text(scenario, lang), "method": names[method], "n_plans": int(r["n_plans"]),
                "late_customer_pct": round(100.0 * r["late_customer_share"], 1),
                "late_return_pct": round(100.0 * r["late_return_share"], 1),
                "routes_with_late_pct": round(100.0 * r["routes_with_late_share"], 1),
                "mean_lateness_late": round(float(r["mean_lateness_late"]), 1),
                "p95_lateness_late": round(float(r["p95_lateness_late"]), 1),
            })
    return pd.DataFrame(rows)


def slack_correlation_table(correlation: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Tabulate the Spearman correlation between nominal slack and late rate, by scenario and method.

    Parameters
    ----------
    correlation : pandas.DataFrame
        ``slack_correlation`` of :func:`delivery_routing_vrptw.results.load_robustness`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per scenario, one column per method (text), values rounded to two decimals.
    """
    names = get_labels(lang)["methods"]
    wide = correlation.pivot(index="scenario", columns="method", values="spearman_slack_late_rate")
    out = wide.reindex(index=list(SCENARIO_ORDER), columns=list(METHODS)).round(2)
    out.columns = [names[m] for m in out.columns]
    out.insert(0, "scenario", [_scenario_text(s, lang) for s in out.index])
    return out.reset_index(drop=True)


def slack_bins_table(slack_lateness: pd.DataFrame, scenario: str = "travel_0.10", lang: str = "en") -> pd.DataFrame:
    """Tabulate the late rate (percent of draws) by nominal-slack bin and method for one scenario.

    Parameters
    ----------
    slack_lateness : pandas.DataFrame
        ``slack_lateness`` of :func:`delivery_routing_vrptw.results.load_robustness`.
    scenario : str
        Scenario name.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per slack bin (``slack_bin``), one column per method with the late rate and one with the
        number of customers of the OR-Tools plans in the bin.
    """
    names = get_labels(lang)["methods"]
    part = slack_lateness[slack_lateness["scenario"] == scenario]
    rate = part.pivot(index="slack_bin", columns="method", values="late_rate").reindex(list(SLACK_ORDER))
    count = part[part["method"] == METHODS[0]].set_index("slack_bin")["n_customers"].reindex(list(SLACK_ORDER))
    out = pd.DataFrame({"slack_bin": list(SLACK_ORDER), "n_customers_first": count.to_numpy()})
    for method in METHODS:
        out[names[method]] = _pct(rate[method]).to_numpy()
    return out.rename(columns={"n_customers_first": "n_customers"})


def robustness_by_family(summary: pd.DataFrame, scenario: str = "travel_0.10", lang: str = "en") -> pd.DataFrame:
    """Tabulate the late-customer share by family and method for one scenario, with the mean nominal slack.

    Parameters
    ----------
    summary : pandas.DataFrame
        ``summary`` of :func:`delivery_routing_vrptw.results.load_robustness`.
    scenario : str
        Scenario name.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per family: ``family``, ``n_plans``, ``nominal_slack_mean`` (OR-Tools plans, time units) and,
        per method, the percentage of customers served late.
    """
    names = get_labels(lang)["methods"]
    part = summary[(summary["scenario"] == scenario) & (summary["scope"] != "overall")]
    rows = []
    for family in FAMILIES:
        base = part[(part["scope"] == family) & (part["method"] == METHODS[0])]
        if base.empty:
            continue
        row = {"family": family, "n_plans": int(base.iloc[0]["n_plans"]),
               "nominal_slack_mean": round(float(base.iloc[0]["nominal_slack_mean"]), 1)}
        for method in METHODS:
            r = part[(part["scope"] == family) & (part["method"] == method)].iloc[0]
            row[names[method]] = round(100.0 * float(r["late_customer_share"]), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- C2
def reference_plans(capacity_cells: pd.DataFrame) -> pd.DataFrame:
    """Return the reference plan (capacity factor 1, representative replicate) of every instance.

    Parameters
    ----------
    capacity_cells : pandas.DataFrame
        Cells of the C2.1 block (:func:`delivery_routing_vrptw.results.load_phase_c_cells`).

    Returns
    -------
    pandas.DataFrame
        See :func:`delivery_routing_vrptw.phase_c.reference_table`.
    """
    return phase_c.reference_table(capacity_cells)


def _fleet_text(cls, lang: str):
    if not isinstance(cls, str):
        return cls
    return get_labels(lang)["values"][f"fleet_{cls}"]


def capacity_by_level(cells: pd.DataFrame, ref: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Aggregate C2.1 by capacity factor and fleet class (distance never pooled across classes).

    Parameters
    ----------
    cells : pandas.DataFrame
        C2.1 cells.
    ref : pandas.DataFrame
        Output of :func:`reference_plans`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``level`` (capacity as % of Q), ``fleet_class`` (text; "all" for the factor-level
        row), ``n_instances``, ``n_infeasible``, ``extra_vehicles_mean``, ``distance_change_mean`` and
        ``distance_change_median``.
    """
    agg = phase_c.capacity_aggregate(phase_c.capacity_summary(cells, ref))
    agg = agg.assign(
        level=[f"{100 * float(v):g}%" for v in agg["level"]],
        fleet_class=[get_labels(lang)["phase_c"]["all"] if c == "all" else _fleet_text(c, lang) for c in agg["fleet_class"]],
    )
    agg = agg[agg["level"] != "100%"].reset_index(drop=True)
    for col in ("extra_vehicles_mean", "distance_change_mean", "distance_change_median"):
        agg[col] = agg[col].round(2)
    return agg


def capacity_instances(cells: pd.DataFrame, ref: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """List the instances whose fleet grows when the capacity is reduced.

    Parameters
    ----------
    cells : pandas.DataFrame
        C2.1 cells.
    ref : pandas.DataFrame
        Output of :func:`reference_plans`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, capacity factor) with more vehicles than the reference plan: ``instance``,
        ``level``, ``vehicles`` (reference), ``vehicles_scenario``, ``extra_vehicles`` and
        ``distance_change_pct``.
    """
    s = phase_c.capacity_summary(cells, ref)
    s = s[(s["extra_vehicles"] > 0)].sort_values(["level", "instance"], ascending=[False, True])
    return pd.DataFrame({
        "instance": s["instance"].to_numpy(), "level": [f"{100 * float(v):g}%" for v in s["level"]],
        "vehicles_ref": s["ref_vehicles"].astype(int).to_numpy(), "vehicles_scenario": s["vehicles"].astype(int).to_numpy(),
        "extra_vehicles": s["extra_vehicles"].astype(int).to_numpy(),
        "distance_change_pct": s["distance_change_pct"].round(1).to_numpy(),
    })


def fleet_priority_by_level(cells: pd.DataFrame, ref: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Aggregate C2.2 by fleet limit and variant (uniform penalty or priority), plus paired differences.

    Parameters
    ----------
    cells : pandas.DataFrame
        C2.2 cells.
    ref : pandas.DataFrame
        Output of :func:`reference_plans`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``level`` (K minus 1 or 2), ``variant`` (text), ``n_instances``, ``dropped_total``,
        ``dropped_priority_total``, ``priority_dropped_pct``, ``common_dropped_pct``, ``dropped_demand_mean``
        and ``distance_change_mean`` (the distance covers fewer customers and is not comparable one to one);
        the rows ``priority - uniform`` hold mean paired differences per instance.
    """
    text = get_labels(lang)["phase_c"]
    agg = phase_c.fleet_priority_aggregate(phase_c.fleet_priority_summary(cells, ref))
    agg["variant"] = [text.get(f"variant_{v}", v) for v in agg["variant"]]
    agg["priority_dropped_pct"] = (100 * agg["priority_dropped_share"]).round(1)
    agg["common_dropped_pct"] = (100 * agg["common_dropped_share"]).round(1)
    for col in ("dropped_total", "dropped_priority_total", "dropped_demand_mean", "distance_change_mean"):
        agg[col] = agg[col].round(1)
    agg["level"] = [str(v).replace("K-", "K − ") for v in agg["level"]]
    cols = ["level", "variant", "n_instances", "dropped_total", "dropped_priority_total", "priority_dropped_pct",
            "common_dropped_pct", "dropped_demand_mean", "distance_change_mean"]
    return agg[cols]


def fleet_priority_instances(cells: pd.DataFrame, ref: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """List, per instance and fleet limit, the customers dropped with and without priority (representative plans).

    Parameters
    ----------
    cells : pandas.DataFrame
        C2.2 cells.
    ref : pandas.DataFrame
        Output of :func:`reference_plans`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, K level): ``instance``, ``level``, ``K``, ``n_priority``, and the numbers of
        customers dropped (``n_dropped_uniform`` / ``n_dropped_priority_variant``) and of priority customers
        dropped (``n_prio_dropped_uniform`` / ``n_prio_dropped_priority_variant``).
    """
    s = phase_c.fleet_priority_summary(cells, ref)
    s = s[s["feasible"]]
    a = s[s["variant"] == "uniform"].set_index(["instance", "level"])
    b = s[s["variant"] == "priority"].set_index(["instance", "level"]).loc[a.index]
    out = pd.DataFrame({
        "K": a["K"].astype(int).to_numpy(), "n_priority": a["n_priority"].astype(int).to_numpy(),
        "n_dropped_uniform": a["n_dropped"].astype(int).to_numpy(),
        "n_dropped_priority_variant": b["n_dropped"].astype(int).to_numpy(),
        "n_prio_dropped_uniform": a["n_dropped_priority"].astype(int).to_numpy(),
        "n_prio_dropped_priority_variant": b["n_dropped_priority"].astype(int).to_numpy(),
    }, index=a.index).reset_index()
    out["level"] = [str(v).replace("K-", "K − ") for v in out["level"]]
    return out[["instance", "level", "K", "n_priority", "n_dropped_uniform", "n_dropped_priority_variant",
                "n_prio_dropped_uniform", "n_prio_dropped_priority_variant"]]


def soft_breakdown(cells: pd.DataFrame, ref: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Count the C2.3 cells by outcome: solved, infeasible by capacity, or failed in the initial construction.

    Parameters
    ----------
    cells : pandas.DataFrame
        C2.3 cells.
    ref : pandas.DataFrame
        Output of :func:`reference_plans`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per cost level: ``level`` (text), ``n_cells``, ``n_feasible``, ``n_capacity_infeasible`` and
        ``n_search_failed`` (see :func:`delivery_routing_vrptw.phase_c.infeasible_breakdown`).
    """
    out = phase_c.infeasible_breakdown(phase_c.soft_summary(cells, ref))
    out["level"] = [get_labels(lang)["phase_c"][f"soft_{v}"] for v in out["level"]]
    return out


def soft_status_by_instance(cells: pd.DataFrame, ref: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Classify every instance of C2.3 by outcome, with the longest runtime of its failed attempts.

    Parameters
    ----------
    cells : pandas.DataFrame
        C2.3 cells (all cost levels and replicates).
    ref : pandas.DataFrame
        Output of :func:`reference_plans`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per instance: ``instance``, ``family``, ``vehicles_ref``, ``K`` (limit used), ``outcome``
        (solved / capacity-infeasible / construction failure, text) and ``runtime_max`` (longest runtime of any
        cell of the instance, in seconds).
    """
    text = get_labels(lang)["phase_c"]
    summary = phase_c.soft_summary(cells, ref)
    rows = []
    for name, g in summary.groupby("instance", sort=True):
        feasible = bool(g["feasible"].any())
        capacity = bool(g["capacity_infeasible"].astype(str).str.lower().eq("true").any())
        outcome = "solved" if feasible else ("capacity" if capacity else "construction")
        runtime = pd.to_numeric(cells.loc[cells["instance"] == name, "runtime_s"], errors="coerce").max()
        rows.append({
            "instance": name, "family": g["family"].iloc[0], "vehicles_ref": int(g["ref_vehicles"].iloc[0]),
            "K": int(g["K"].iloc[0]), "outcome": text[f"outcome_{outcome}"], "runtime_max": round(float(runtime), 1),
        })
    return pd.DataFrame(rows)


def soft_by_level(cells: pd.DataFrame, ref: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Aggregate the solved C2.3 instances by lateness-cost level (only those instances; see the breakdown).

    Parameters
    ----------
    cells : pandas.DataFrame
        C2.3 cells.
    ref : pandas.DataFrame
        Output of :func:`reference_plans`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``level`` (text), ``fleet_class``, ``n_instances``, ``n_late_mean``,
        ``total_lateness_mean``, ``max_lateness_mean``, ``max_lateness_max`` and ``distance_change_mean``.
    """
    agg = phase_c.soft_aggregate(phase_c.soft_summary(cells, ref))
    agg["level"] = [get_labels(lang)["phase_c"][f"soft_{v}"] for v in agg["level"]]
    agg["fleet_class"] = [_fleet_text(c, lang) for c in agg["fleet_class"]]
    cols = ["level", "fleet_class", "n_instances", "n_late_mean", "total_lateness_mean", "max_lateness_mean",
            "max_lateness_max", "distance_change_mean"]
    return agg[cols].round(1)


# --------------------------------------------------------------------------- C3
def reopt_table(removal: pd.DataFrame, insertion: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Aggregate the reoptimisation after removing or inserting 10% of the customers (warm, cold, noise).

    Parameters
    ----------
    removal, insertion : pandas.DataFrame
        C3 cells (:func:`delivery_routing_vrptw.results.load_phase_c_cells`).
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per (event, variant): label-key columns ``event`` (text), ``variant`` (warm, cold or noise
        floor), ``n`` (instance-replicate pairs), ``reassigned_pct`` (median share of the common customers that
        change vehicle), ``pairs_kept_pct``, ``arcs_kept_pct``, ``delta_vehicles_mean`` and
        ``delta_distance_pct_median`` (original plan on the same customers as the reference).
    """
    text = get_labels(lang)["phase_c"]
    rows = []
    for key, cells in (("removal", removal), ("insertion", insertion)):
        agg = phase_c.reopt_aggregate(cells, phase_c.noise_table(cells))
        for r in agg.itertuples():
            rows.append({
                "event": text[f"event_{key}"], "variant": text[f"variant_{r.variant}"], "n": int(r.n),
                "reassigned_pct": round(100.0 * float(r.realloc_customers_reassigned_share), 1),
                "pairs_kept_pct": round(100.0 * float(r.realloc_pairs_kept), 1),
                "arcs_kept_pct": round(100.0 * float(r.realloc_arcs_kept), 1),
                "delta_vehicles_mean": round(float(r.delta_vehicles_mean), 2) if r.variant != "noise" else np.nan,
                "delta_distance_pct_median": round(float(r.delta_distance_pct_median), 1) if r.variant != "noise" else np.nan,
            })
    return pd.DataFrame(rows)


def reopt_table_representative(removal: pd.DataFrame, insertion: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Aggregate C3 over the representative replicate of each instance (warm and cold), as the protocol prescribes.

    Parameters
    ----------
    removal, insertion : pandas.DataFrame
        C3 cells (:func:`delivery_routing_vrptw.results.load_phase_c_cells`).
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per (event, variant in warm, cold): ``event``, ``variant``, ``n_instances``, ``reassigned_pct``
        (median over instances of the share of common customers that change vehicle),
        ``pooled_reassigned_pct`` (the same median over **all** replicates pooled, the first aggregation of
        the run, for comparison), ``pairs_kept_pct``, ``arcs_kept_pct``, ``delta_vehicles_mean`` and
        ``delta_distance_pct_median``.
    """
    text = get_labels(lang)["phase_c"]
    rows = []
    for key, cells in (("removal", removal), ("insertion", insertion)):
        noise = phase_c.noise_table(cells)
        rep = phase_c.reopt_aggregate_representative(cells, noise).set_index("variant")
        pooled = phase_c.reopt_aggregate(cells, noise).set_index("variant")
        for variant in ("warm", "cold"):
            r = rep.loc[variant]
            rows.append({
                "event": text[f"event_{key}"], "variant": text[f"variant_{variant}"], "n_instances": int(r["n"]),
                "reassigned_pct": round(100.0 * float(r["realloc_customers_reassigned_share"]), 1),
                "pooled_reassigned_pct": round(
                    100.0 * float(pooled.loc[variant, "realloc_customers_reassigned_share"]), 1),
                "pairs_kept_pct": round(100.0 * float(r["realloc_pairs_kept"]), 1),
                "arcs_kept_pct": round(100.0 * float(r["realloc_arcs_kept"]), 1),
                "delta_vehicles_mean": round(float(r["delta_vehicles_mean"]), 2),
                "delta_distance_pct_median": round(float(r["delta_distance_pct_median"]), 1),
            })
    return pd.DataFrame(rows)


def reopt_noise_table(removal: pd.DataFrame, insertion: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Show the noise floor of C3 as a distribution, not as a single number.

    Parameters
    ----------
    removal, insertion : pandas.DataFrame
        C3 cells.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per event: ``event``, ``n_pairs`` (cold-versus-cold comparisons), ``noise_median_pct``,
        ``noise_n_positive`` (comparisons that reassigned any customer), ``noise_max_pct`` and
        ``noise_instances`` (instance, share of customers reassigned and number of comparisons).
    """
    text = get_labels(lang)["phase_c"]
    rows = []
    for key, cells in (("removal", removal), ("insertion", insertion)):
        d = phase_c.noise_distribution(phase_c.noise_table(cells)).iloc[0]
        rows.append({
            "event": text[f"event_{key}"], "n_pairs": int(d["n_pairs"]),
            "noise_median_pct": round(100.0 * float(d["median_share"]), 1), "noise_n_positive": int(d["n_positive"]),
            "noise_max_pct": round(100.0 * float(d["max_share"]), 1), "noise_instances": d["instances"] or "-",
        })
    return pd.DataFrame(rows)


def dropped_spread_summary(cells: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Count the C2.2 cell groups whose replicates drop a different number of customers, or different ones.

    Parameters
    ----------
    cells : pandas.DataFrame
        C2.2 cells.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``item`` and ``count``: groups (instance x level x variant), groups whose replicates
        drop a different number of customers and groups whose replicates drop different sets of customers.
    """
    text = get_labels(lang)["phase_c"]
    s = phase_c.dropped_spread(cells)
    rows = [("spread_groups", len(s)), ("spread_count", int((s["dropped_range"] > 0).sum())),
            ("spread_sets", int((s["n_distinct_sets"] > 1).sum()))]
    return pd.DataFrame({"item": [text[k] for k, _ in rows], "count": [c for _, c in rows]})


def dropped_spread_table(cells: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """List the C2.2 cell groups whose replicates differ in the customers they drop.

    Parameters
    ----------
    cells : pandas.DataFrame
        C2.2 cells.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per group with a different number or a different set of dropped customers: ``instance``,
        ``level``, ``variant``, ``n_replicates``, ``dropped_values`` (per replicate), ``dropped_range``,
        ``priority_dropped_values`` and ``n_distinct_sets``; sorted by range, then by instance.
    """
    text = get_labels(lang)["phase_c"]
    s = phase_c.dropped_spread(cells)
    s = s[(s["dropped_range"] > 0) | (s["n_distinct_sets"] > 1)].sort_values(
        ["dropped_range", "instance", "level", "variant"], ascending=[False, True, True, True])
    out = s[["instance", "level", "variant", "n_replicates", "dropped_values", "dropped_range",
             "priority_dropped_values", "n_distinct_sets"]].copy()
    out["level"] = [str(v).replace("K-", "K − ") for v in out["level"]]
    out["variant"] = [text[f"variant_{v}"] for v in out["variant"]]
    return out.reset_index(drop=True)
