"""Tables and panels for the final-evaluation notebook (layer A, pre-registered protocol).

Every number comes from the frozen evaluation results (and, for the consistency check, from the frozen
development time-budget study); nothing is hard-coded. The distance gap is only defined where the fleet
equals the SINTEF reference, as in :mod:`delivery_routing_vrptw.evaluation`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .analysis import _dash
from .config import ExperimentConfig
from .distance import exact
from .evaluation import (
    BEST,
    ORTOOLS,
    aggregate_table,
    instance_table,
    median_position,
    sort_replicates,
)
from .io import FAMILIES, Instance
from .labels import get_labels
from .results import get_routes
from .route_compare import schedule_metrics
from .solver import SolverConfig, fixed_cost_for

DISTANCE_CENT = 0.005
"""Largest absolute distance difference (in distance units) still considered "to two decimals"."""


def _scope_label(scope: str, lang: str) -> str:
    return get_labels(lang)["scopes"].get(scope, scope)


def _scoped(table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return ``(scope, rows)`` for the whole table and for each family present."""
    out = [("overall", table)]
    out += [(f, table[table["family"] == f]) for f in FAMILIES if f in set(table["family"])]
    return out


def reproduces_reference(table: pd.DataFrame, tol: float = DISTANCE_CENT) -> pd.Series:
    """Flag instances whose fleet equals the reference and whose distance matches it to two decimals.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table` (``distance_comparable``,
        ``distance`` and ``ref_distance``).
    tol : float
        Largest accepted absolute difference between the solution distance and the reference.

    Returns
    -------
    pandas.Series
        Boolean flag per row.
    """
    comparable = table["distance_comparable"].astype(bool)
    return comparable & ((table["distance"] - table["ref_distance"]).abs() <= tol)


# --------------------------------------------------------------------------- protocol and integrity
def protocol_table(cfg: ExperimentConfig, selected: Mapping, lang: str = "en") -> pd.DataFrame:
    """Tabulate the pre-registered protocol: the frozen configuration and the evaluation settings.

    Parameters
    ----------
    cfg : ExperimentConfig
        Effective configuration read from ``config/experiment.toml``.
    selected : mapping
        Frozen configuration (:func:`delivery_routing_vrptw.results.load_selected_config`).
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Two columns, ``metric`` and ``value``.
    """
    names = get_labels(lang)["protocol"]
    ev = cfg.evaluation
    median_rank = median_position(ev.replicates) + 1
    items = {
        "fleet_strategy": selected["config"]["fleet_strategy"],
        "first_solution": selected["config"]["first_solution"],
        "metaheuristic": selected["config"]["metaheuristic"],
        "tournament_budget": f"{cfg.solver.time_limit_s:g}",
        "budget": f"{ev.time_limit_s:g}",
        "replicates": f"{ev.replicates}",
        "main_result": names["main_value"].format(rank=median_rank, n=ev.replicates),
        "baselines": names["yes"] if ev.include_baselines else names["no"],
        "instance_set": ev.instance_set,
    }
    return pd.DataFrame({"metric": [names[k] for k in items], "value": list(items.values())})


def integrity_table(rows: pd.DataFrame, split: Mapping[str, str], lang: str = "en") -> pd.DataFrame:
    """Audit the execution of the evaluation from the result file itself.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_evaluation`.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Two columns, ``metric`` and ``value`` (formatted text): cells, instances, replicates, feasibility,
        solver status and runtime.
    """
    names = get_labels(lang)["integrity"]
    ort = rows[rows["method"] == ORTOOLS]
    reps = ort.groupby("instance").size()
    n_eval = sum(1 for v in split.values() if v == "eval")
    n_dev_found = sum(1 for n in rows["instance"].unique() if split.get(n) == "dev")
    run = ort["runtime_s"]
    items = {
        "cells": f"{len(rows)}",
        "cells_ortools": f"{len(ort)}",
        "cells_baselines": f"{len(rows) - len(ort)}",
        "n_instances": f"{rows['instance'].nunique()}",
        "n_eval_split": f"{n_eval}",
        "n_dev_found": f"{n_dev_found}",
        "replicates": f"{int(reps.min())}" if reps.min() == reps.max() else f"{int(reps.min())}-{int(reps.max())}",
        "feasible": f"{int(rows['feasible'].astype(bool).sum())} / {len(rows)}",
        "status": f"{int((ort['status'] == 'ROUTING_SUCCESS').sum())} / {len(ort)}",
        "mean_runtime": f"{run.mean():.1f}",
        "runtime_range": f"{run.min():.2f} - {run.max():.2f}",
    }
    return pd.DataFrame({"metric": [names[k] for k in items], "value": list(items.values())})


# --------------------------------------------------------------------------- main result
def main_table(rows: pd.DataFrame, lang: str = "en", best: bool = False) -> pd.DataFrame:
    """Aggregate the OR-Tools result overall and by family, against the SINTEF reference.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_evaluation`.
    lang : str
        Label language.
    best : bool
        If true, describe the best replicate ("best of R") instead of the median replicate (main result).

    Returns
    -------
    pandas.DataFrame
        One row per scope. Columns are label keys (translate with
        :func:`delivery_routing_vrptw.eda.localize_columns`): ``result_kind``, ``scope``, ``n_instances``,
        ``mean_delta_vehicles``, ``n_fleet_equal_ref``, ``median_gap_equal_fleet`` (``"-"`` if none) and
        ``n_beats_reference``.
    """
    agg = aggregate_table(rows)
    agg = agg[agg["method"] == ORTOOLS]
    prefix = BEST if best else ""
    reps = int(rows[rows["method"] == ORTOOLS].groupby("instance").size().max())
    kinds = get_labels(lang)["kinds"]
    kind = kinds["best"].format(n=reps) if best else kinds["median"]
    out = pd.DataFrame({
        "result_kind": kind,
        "scope": [_scope_label(s, lang) for s in agg["scope"]],
        "n_instances": agg["n_instances"].to_numpy(),
        "mean_delta_vehicles": agg[f"{prefix}mean_delta_vehicles"].round(2).to_numpy(),
        "n_fleet_equal_ref": agg[f"{prefix}n_fleet_equal"].astype(int).to_numpy(),
        "median_gap_equal_fleet": _dash(agg[f"{prefix}median_gap_fleet_equal"], 2),
        "n_beats_reference": agg[f"{prefix}n_beats_reference"].astype(int).to_numpy(),
    })
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- distribution of delta vehicles
def delta_distribution(table: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Count instances by Δ vehicles (median replicate), overall and by family.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Columns (label keys) ``scope``, ``n_instances``, ``n_below`` (Δ < 0), ``n_delta0``, ``n_delta1``,
        ``n_delta2`` and ``n_delta3plus``.
    """
    rows = []
    for scope, sub in _scoped(table):
        d = sub["delta_vehicles"]
        rows.append({
            "scope": _scope_label(scope, lang), "n_instances": len(sub), "n_below": int((d < 0).sum()),
            "n_delta0": int((d == 0).sum()), "n_delta1": int((d == 1).sum()), "n_delta2": int((d == 2).sum()),
            "n_delta3plus": int((d >= 3).sum()),
        })
    return pd.DataFrame(rows)


def delta_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Count instances per family (rows) and Δ vehicles class (columns).

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.

    Returns
    -------
    pandas.DataFrame
        Index = families present (benchmark order); columns ``<0``, ``0``, ``1``, ``2`` and ``3+``.
    """
    d = table["delta_vehicles"]
    cls = pd.Series(np.select([d < 0, d == 0, d == 1, d == 2], ["<0", "0", "1", "2"], default="3+"), index=table.index)
    counts = pd.crosstab(table["family"], cls)
    counts = counts.reindex(index=[f for f in FAMILIES if f in counts.index],
                            columns=["<0", "0", "1", "2", "3+"], fill_value=0)
    return counts


def delta_instances(table: pd.DataFrame, minimum: int) -> pd.DataFrame:
    """List the instances whose Δ vehicles is at least ``minimum`` (median replicate).

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.
    minimum : int
        Smallest Δ vehicles to list.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``instance``, ``family``, ``ref_vehicles``, ``vehicles`` and ``delta_vs_ref``.
    """
    sub = table[table["delta_vehicles"] >= minimum]
    cols = ["instance", "family", "ref_vehicles", "vehicles", "delta_vehicles"]
    out = sub[cols].astype({"ref_vehicles": int, "vehicles": int, "delta_vehicles": int})
    return out.rename(columns={"delta_vehicles": "delta_vs_ref"}).reset_index(drop=True)


# --------------------------------------------------------------------------- fleet-equal instances
def fleet_equal_table(table: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """List the instances whose median replicate uses the reference fleet, with the distance gap.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per fleet-equal instance (label-key columns): ``instance``, ``family``, ``ref_vehicles``,
        ``ref_distance``, ``vehicles``, ``distance``, ``gap_distance_pct`` and ``reproduces`` (yes/no).
    """
    yes_no = get_labels(lang)["yesno"]
    sub = table[table["distance_comparable"].astype(bool)].copy()
    sub["reproduces"] = [yes_no["yes" if r else "no"] for r in reproduces_reference(sub)]
    out = sub[["instance", "family", "ref_vehicles", "ref_distance", "vehicles", "distance",
               "gap_distance_pct", "reproduces"]].copy()
    out["ref_vehicles"] = out["ref_vehicles"].astype(int)
    out["vehicles"] = out["vehicles"].astype(int)
    out["distance"] = out["distance"].round(2)
    out["gap_distance_pct"] = out["gap_distance_pct"].round(2) + 0.0
    return out.reset_index(drop=True)


def reproduction_summary(table: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Summarise the fleet-equal instances by scope: how many reproduce the reference distance.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``scope``, ``n_instances``, ``n_fleet_equal_ref``, ``n_reproduced`` (distance
        equal to the reference to two decimals), ``n_above`` (fleet equal, distance above the reference),
        ``median_gap_equal_fleet`` and ``max_gap`` (both ``"-"`` if no instance has the reference fleet).
    """
    rows = []
    for scope, sub in _scoped(table):
        eq = sub[sub["distance_comparable"].astype(bool)]
        rep = reproduces_reference(eq)
        gap = eq["gap_distance_pct"]
        rows.append({
            "scope": _scope_label(scope, lang), "n_instances": len(sub), "n_fleet_equal_ref": len(eq),
            "n_reproduced": int(rep.sum()), "n_above": int((~rep).sum()),
            "median_gap_equal_fleet": gap.median() if len(eq) else np.nan,
            "max_gap": gap.max() if len(eq) else np.nan,
        })
    out = pd.DataFrame(rows)
    for col in ("median_gap_equal_fleet", "max_gap"):
        out[col] = _dash(out[col], 2)
    return out


# --------------------------------------------------------------------------- variability
def variability_table(table: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Summarise the variability across replicates, overall and by family.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``scope``, ``n_instances``, ``n_vehicles_vary`` (instances whose fleet differs
        between replicates), ``max_vehicles_range``, ``n_distance_vary`` (instances whose distance differs),
        ``median_distance_range_pct`` and ``max_distance_range_pct``.
    """
    rows = []
    for scope, sub in _scoped(table):
        rows.append({
            "scope": _scope_label(scope, lang), "n_instances": len(sub),
            "n_vehicles_vary": int((sub["vehicles_range"] > 0).sum()),
            "max_vehicles_range": int(sub["vehicles_range"].max()),
            "n_distance_vary": int((sub["distance_range_pct"] > 0).sum()),
            "median_distance_range_pct": round(float(sub["distance_range_pct"].median()), 2) + 0.0,
            "max_distance_range_pct": round(float(sub["distance_range_pct"].max()), 2) + 0.0,
        })
    return pd.DataFrame(rows)


def varying_instances(table: pd.DataFrame) -> pd.DataFrame:
    """List the instances whose distance differs across replicates.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``instance``, ``family``, ``vehicles``, ``distance_min``, ``distance`` (median
        replicate), ``distance_max`` and ``distance_range_pct``, sorted by decreasing range.
    """
    sub = table[table["distance_range_pct"] > 0].sort_values("distance_range_pct", ascending=False)
    out = sub[["instance", "family", "vehicles", "distance_min", "distance", "distance_max",
               "distance_range_pct"]].copy()
    out["vehicles"] = out["vehicles"].astype(int)
    for col in ("distance_min", "distance", "distance_max", "distance_range_pct"):
        out[col] = out[col].round(2)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- baselines
def method_comparison(rows: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Compare OR-Tools (median replicate) with the baselines over all instances.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_evaluation`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per method (OR-Tools first, then baselines by mean Δ vehicles) with label-key columns
        ``method``, ``n_instances``, ``n_feasible``, ``mean_delta_vehicles``, ``n_fleet_equal_ref``,
        ``median_gap_equal_fleet`` and ``n_beats_reference``.
    """
    names = get_labels(lang)["methods"]
    agg = aggregate_table(rows)
    agg = agg[agg["scope"] == "overall"].copy()
    agg["_o"] = (agg["method"] != ORTOOLS).astype(int)
    agg = agg.sort_values(["_o", "mean_delta_vehicles"])
    return pd.DataFrame({
        "method": [names[m] for m in agg["method"]],
        "n_instances": agg["n_instances"].to_numpy(),
        "n_feasible": agg["n_feasible"].to_numpy(),
        "mean_delta_vehicles": agg["mean_delta_vehicles"].round(2).to_numpy(),
        "n_fleet_equal_ref": agg["n_fleet_equal"].astype(int).to_numpy(),
        "median_gap_equal_fleet": _dash(agg["median_gap_fleet_equal"], 2),
        "n_beats_reference": agg["n_beats_reference"].astype(int).to_numpy(),
    }).reset_index(drop=True)


def method_by_family(rows: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Return the mean Δ vehicles of each method per family (columns) and overall.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_evaluation`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per method (same order as :func:`method_comparison`); first column ``method`` (label key),
        then the overall column and one column per family.
    """
    names = get_labels(lang)["methods"]
    agg = aggregate_table(rows)
    order = method_comparison(rows, lang)["method"].tolist()
    wide = agg.pivot(index="method", columns="scope", values="mean_delta_vehicles")
    wide = wide[["overall", *[f for f in FAMILIES if f in wide.columns]]].round(2)
    wide.columns = [_scope_label(c, lang) for c in wide.columns]
    wide.index = [names[m] for m in wide.index]
    return wide.loc[order].rename_axis("method").reset_index()


# --------------------------------------------------------------------------- development consistency
def dev_vs_eval(time_df: pd.DataFrame, table: pd.DataFrame, budget_s: float, lang: str = "en") -> pd.DataFrame:
    """Place the development result at the evaluation budget next to the final evaluation (descriptive).

    Parameters
    ----------
    time_df : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_time_sensitivity` (one run per instance and budget).
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.
    budget_s : float
        Budget of the development study to compare with (the evaluation budget).
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Two rows (development, evaluation) with label-key columns ``set``, ``n_instances``, ``n_runs``
        (runs per instance), ``mean_delta_vehicles``, ``fleet_equal_share`` (``"k / n"``) and
        ``median_gap_equal_fleet``.
    """
    sets = get_labels(lang)["sets"]
    dev = time_df[time_df["budget_s"] == budget_s]
    dev_eq = dev[dev["distance_comparable"]]
    ev_eq = table[table["distance_comparable"].astype(bool)]
    rows = [
        {"set": sets["dev"], "n_instances": dev["instance"].nunique(), "n_runs": 1,
         "mean_delta_vehicles": round(float(dev["delta_vehicles"].mean()), 2),
         "fleet_equal_share": f"{len(dev_eq)} / {len(dev)}",
         "median_gap_equal_fleet": dev_eq["gap_distance_pct"].median() if len(dev_eq) else np.nan},
        {"set": sets["eval"], "n_instances": len(table), "n_runs": int(table["n_replicates"].max()),
         "mean_delta_vehicles": round(float(table["delta_vehicles"].mean()), 2),
         "fleet_equal_share": f"{len(ev_eq)} / {len(table)}",
         "median_gap_equal_fleet": ev_eq["gap_distance_pct"].median() if len(ev_eq) else np.nan},
    ]
    out = pd.DataFrame(rows)
    out["median_gap_equal_fleet"] = _dash(out["median_gap_equal_fleet"], 2)
    return out


def dev_vs_eval_by_family(
    time_df: pd.DataFrame, table: pd.DataFrame, budget_s: float, lang: str = "en"
) -> pd.DataFrame:
    """Compare development (at ``budget_s``) and evaluation by family: mean Δ vehicles and fleet-equal counts.

    Parameters
    ----------
    time_df : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_time_sensitivity`.
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.
    budget_s : float
        Development budget to compare with.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per family, with flat column names already translated (``dev`` / ``eval`` prefixes).
    """
    sets = get_labels(lang)["sets"]
    cols = get_labels(lang)["columns"]
    dev = time_df[time_df["budget_s"] == budget_s]
    rows = []
    for fam in FAMILIES:
        d, e = dev[dev["family"] == fam], table[table["family"] == fam]
        if d.empty and e.empty:
            continue
        rows.append({
            "family": fam,
            f"{sets['dev']} · {cols['n_instances']}": len(d),
            f"{sets['dev']} · {cols['mean_delta_vehicles']}": round(float(d["delta_vehicles"].mean()), 2)
            if len(d) else np.nan,
            f"{sets['dev']} · {cols['n_fleet_equal_ref']}": int(d["distance_comparable"].sum()),
            f"{sets['eval']} · {cols['n_instances']}": len(e),
            f"{sets['eval']} · {cols['mean_delta_vehicles']}": round(float(e["delta_vehicles"].mean()), 2)
            if len(e) else np.nan,
            f"{sets['eval']} · {cols['n_fleet_equal_ref']}": int(e["distance_comparable"].astype(bool).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- route maps
def pick_map_instances(table: pd.DataFrame) -> list[str]:
    """Choose two instances to draw: a C instance that reproduces the reference and the first RC1 with the largest Δ.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.

    Returns
    -------
    list of str
        ``[first C instance (by name) reproducing the reference distance to two decimals,
        first RC1 instance (by name) with the largest Δ vehicles of its family]``.

    Raises
    ------
    ValueError
        If no C instance reproduces the reference or the table has no RC1 instance.
    """
    t = table.sort_values("instance")
    c = t[t["family"].isin(["C1", "C2"]) & reproduces_reference(t)]
    rc = t[t["family"] == "RC1"]
    if c.empty or rc.empty:
        raise ValueError("need a reproducing C instance and an RC1 instance")
    top = rc[rc["delta_vehicles"] == rc["delta_vehicles"].max()]
    return [str(c.iloc[0]["instance"]), str(top.iloc[0]["instance"])]


def median_panels(
    names: list[str], instances: Mapping[str, Instance], rows: pd.DataFrame, table: pd.DataFrame,
    lang: str = "en",
) -> list[tuple[Instance, str, list[list[int]]]]:
    """Build route-map panels of the median replicate of OR-Tools for several instances.

    Parameters
    ----------
    names : list of str
        Instance names.
    instances : mapping
        Instances keyed by name.
    rows : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_evaluation`.
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table` (gives the median replicate).
    lang : str
        Label language.

    Returns
    -------
    list of tuple
        ``(instance, label, routes)`` ready for :func:`delivery_routing_vrptw.viz_results.plot_route_panels`;
        the label names the replicate and the SINTEF reference.
    """
    template = get_labels(lang)["panels"]["median"]
    out = []
    info = table.set_index("instance")
    for name in names:
        rep = int(info.loc[name, "median_replicate"])
        routes = get_routes(rows, name, method=ORTOOLS, replicate=rep)
        label = template.format(rep=rep, ref=int(info.loc[name, "ref_vehicles"]),
                                dist=float(info.loc[name, "ref_distance"]))
        out.append((instances[name], label, routes))
    return out


# --------------------------------------------------------------------------- time metrics (audit finding 3)
TIME_METRICS = ("total_duration", "total_waiting", "total_duration_min", "total_waiting_min", "makespan")
"""Time metrics derived from the frozen routes: depart at 0, optimal departure and last return."""


def time_metrics_table(rows: pd.DataFrame, instances: Mapping[str, Instance]) -> pd.DataFrame:
    """Compute the time metrics of every frozen plan of the final evaluation, without solving anything.

    Each plan is scheduled on the original instance with
    :func:`delivery_routing_vrptw.route_compare.schedule_metrics` (earliest start, wait until ``ready``).

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_evaluation` (``routes`` parsed into lists).
    instances : mapping
        The evaluation instances keyed by name.

    Returns
    -------
    pandas.DataFrame
        One row per input row, keyed by ``instance``, ``family``, ``method``, ``replicate``, with ``feasible``,
        ``vehicles``, ``distance`` (recomputed), ``distance_diff`` (absolute difference to the stored
        distance), ``total_duration``, ``total_waiting`` (every route departs at 0),
        ``total_duration_min``, ``total_waiting_min`` (optimal departure) and ``makespan``.
    """
    matrices: dict[str, np.ndarray] = {}
    out = []
    for row in rows.itertuples(index=False):
        name = row.instance
        if name not in matrices:
            matrices[name] = exact(instances[name])
        m = schedule_metrics(instances[name], row.routes, matrices[name])
        out.append({
            "instance": name, "family": row.family, "method": row.method, "replicate": int(row.replicate),
            "feasible": bool(row.feasible), "vehicles": m["vehicles"], "distance": m["distance"],
            "distance_diff": abs(m["distance"] - float(row.distance)),
            **{k: m[k] for k in TIME_METRICS},
        })
    return pd.DataFrame(out)


def time_representatives(tm: pd.DataFrame) -> pd.DataFrame:
    """Pick one plan per (instance, method): the median replicate for OR-Tools, the only run for baselines.

    Replicates are ordered by (infeasible, vehicles, distance) and the one at position ``ceil(R / 2)`` is taken,
    the same rule as :func:`delivery_routing_vrptw.evaluation.instance_table`.

    Parameters
    ----------
    tm : pandas.DataFrame
        Output of :func:`time_metrics_table` (or the derived ``time_metrics.csv``).

    Returns
    -------
    pandas.DataFrame
        One row per (instance, method) with all columns of the input.
    """
    d = tm.assign(feasible=tm["feasible"].astype(bool))
    picks = [
        sort_replicates(g).iloc[median_position(len(g))]
        for _, g in d.groupby(["instance", "method"], sort=True)
    ]
    return pd.DataFrame(picks).reset_index(drop=True)


def _ratio(rep: pd.DataFrame, column: str) -> pd.Series:
    """Per-instance ratio (%) of a method to OR-Tools on one metric, indexed like ``rep``."""
    base = rep[rep["method"] == ORTOOLS].set_index("instance")[column].replace(0, np.nan)
    return 100.0 * rep[column] / rep["instance"].map(base)


def time_aggregate(rep: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the representative plans by scope (overall and family) and method.

    Parameters
    ----------
    rep : pandas.DataFrame
        Output of :func:`time_representatives`.

    Returns
    -------
    pandas.DataFrame
        One row per (scope, method). Medians over instances: ``dist_median``, ``dur0_median``,
        ``wait0_median``, ``durmin_median``, ``waitmin_median``, ``makespan_median``; shares of waiting in
        the duration (``wait_share_pct`` departing at 0, ``wait_share_min_pct`` with optimal departure,
        medians of per-instance shares); and medians of the per-instance ratio to the OR-Tools plan of the
        same instance (``ratio_distance_pct`` and ``ratio_duration_min_pct``; 100 for OR-Tools itself).
    """
    r = rep.assign(
        ratio_distance_pct=_ratio(rep, "distance"), ratio_duration_min_pct=_ratio(rep, "total_duration_min"),
        wait_share_pct=100.0 * rep["total_waiting"] / rep["total_duration"],
        wait_share_min_pct=100.0 * rep["total_waiting_min"] / rep["total_duration_min"],
    )
    out = []
    for scope, part in _scoped(r):
        for method in [ORTOOLS, *sorted(m for m in r["method"].unique() if m != ORTOOLS)]:
            g = part[part["method"] == method]
            if g.empty:
                continue
            out.append({
                "scope": scope, "method": method, "n_instances": len(g),
                "dist_median": g["distance"].median(), "dur0_median": g["total_duration"].median(),
                "wait0_median": g["total_waiting"].median(), "durmin_median": g["total_duration_min"].median(),
                "waitmin_median": g["total_waiting_min"].median(), "makespan_median": g["makespan"].median(),
                "wait_share_pct": g["wait_share_pct"].median(),
                "wait_share_min_pct": g["wait_share_min_pct"].median(),
                "ratio_distance_pct": g["ratio_distance_pct"].median(),
                "ratio_duration_min_pct": g["ratio_duration_min_pct"].median(),
            })
    return pd.DataFrame(out)


_TIME_COLS = ["dist_median", "dur0_median", "wait0_median", "durmin_median", "waitmin_median",
              "makespan_median", "wait_share_pct", "wait_share_min_pct"]


def time_method_table(agg: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Show the overall time metrics per method (OR-Tools first, then baselines).

    Parameters
    ----------
    agg : pandas.DataFrame
        Output of :func:`time_aggregate`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``method``, ``n_instances``, ``dist_median``, ``dur0_median``, ``wait0_median``,
        ``durmin_median``, ``waitmin_median``, ``makespan_median``, ``wait_share_pct`` and
        ``wait_share_min_pct``.
    """
    names = get_labels(lang)["methods"]
    t = agg[agg["scope"] == "overall"]
    out = pd.DataFrame({"method": [names[m] for m in t["method"]], "n_instances": t["n_instances"].to_numpy()})
    for c in _TIME_COLS:
        out[c] = t[c].round(1).to_numpy()
    return out


def time_family_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Show the OR-Tools time metrics per family (median replicate).

    Parameters
    ----------
    agg : pandas.DataFrame
        Output of :func:`time_aggregate`.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``family``, ``n_instances`` and the metric columns of :func:`time_method_table`;
        one row per family.
    """
    t = agg[(agg["method"] == ORTOOLS) & (agg["scope"] != "overall")]
    out = t[["scope", "n_instances", *_TIME_COLS]].rename(columns={"scope": "family"}).reset_index(drop=True)
    return out.round({c: 1 for c in _TIME_COLS})


def time_ratio_table(agg: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Show, per scope and baseline, the per-instance median ratio to OR-Tools (OR-Tools = 100%).

    Parameters
    ----------
    agg : pandas.DataFrame
        Output of :func:`time_aggregate`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``family`` (``"overall"`` translated), ``method``, ``n_instances``,
        ``ratio_distance_pct`` and ``ratio_duration_min_pct`` for the baselines.
    """
    names = get_labels(lang)["methods"]
    t = agg[agg["method"] != ORTOOLS]
    out = pd.DataFrame({
        "family": [_scope_label(s, lang) for s in t["scope"]],
        "method": [names[m] for m in t["method"]],
        "n_instances": t["n_instances"].to_numpy(),
    })
    for c in ("ratio_distance_pct", "ratio_duration_min_pct"):
        out[c] = t[c].round(1).to_numpy()
    return out


__all__ = [
    "delta_counts", "delta_distribution", "delta_instances", "dev_vs_eval", "dev_vs_eval_by_family",
    "fleet_equal_table", "instance_table", "integrity_table", "main_table", "median_panels",
    "method_by_family", "method_comparison", "pick_map_instances", "protocol_table",
    "reproduces_reference", "reproduction_summary", "time_aggregate", "time_method_table",
    "time_metrics_table", "time_family_table", "time_ratio_table", "time_representatives",
    "variability_table", "varying_instances",
    "fixed_cost_ratio_table", "fixed_cost_ratio_summary",
]


# --------------------------------------------------------------------------- v1 against v2 (post-hoc)
def version_aggregate_table(aggregate: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Format the v1-against-v2 aggregate (median replicate of each version), overall and by family.

    Parameters
    ----------
    aggregate : pandas.DataFrame
        Second output of :func:`delivery_routing_vrptw.evaluation.compare_versions`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per scope; label-key columns ``scope``, ``n_instances``, ``v1_mean_delta_vehicles``,
        ``v2_mean_delta_vehicles``, ``v1_n_fleet_equal``, ``v2_n_fleet_equal``, ``v1_median_gap_fleet_equal``,
        ``v2_median_gap_fleet_equal`` (``"-"`` if no instance has the reference fleet) and ``n_within_noise``
        (instances with equal fleets in both versions and a distance difference below the noise threshold).
    """
    out = pd.DataFrame({
        "scope": [_scope_label(s, lang) for s in aggregate["scope"]],
        "n_instances": aggregate["n_instances"].to_numpy(),
        "v1_mean_delta_vehicles": aggregate["v1_mean_delta_vehicles"].round(2).to_numpy(),
        "v2_mean_delta_vehicles": aggregate["v2_mean_delta_vehicles"].round(2).to_numpy(),
        "v1_n_fleet_equal": aggregate["v1_n_fleet_equal"].astype(int).to_numpy(),
        "v2_n_fleet_equal": aggregate["v2_n_fleet_equal"].astype(int).to_numpy(),
        "v1_median_gap_fleet_equal": _dash(aggregate["v1_median_gap_fleet_equal"], 2),
        "v2_median_gap_fleet_equal": _dash(aggregate["v2_median_gap_fleet_equal"], 2),
        "n_within_noise": aggregate["n_within_noise"].astype(int).to_numpy(),
    })
    return out.reset_index(drop=True)


def version_distance_summary(by_instance: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Summarise how the fleet and the distance change between v1 and v2, instance by instance.

    The distance difference is defined only where both versions use the same fleet; a difference below the
    noise threshold (``evaluation.NOISE_PCT`` percent) is read as within the variation between replicates and
    runs, as pre-registered in ``reports/protocolo-fase-d.md``.

    Parameters
    ----------
    by_instance : pandas.DataFrame
        First output of :func:`delivery_routing_vrptw.evaluation.compare_versions`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Two label-key columns ``item`` and ``count`` (plus ``value`` for ranges), one row per statistic: instances
        compared, fleet changes (fewer or more vehicles in v2), instances with equal fleets in both versions,
        distance within noise, better (v2 shorter beyond the threshold), worse and the extreme differences.
    """
    from .evaluation import NOISE_PCT

    text = get_labels(lang)["versions"]
    d = by_instance
    eq = d["fleet_equal_versions"].astype(bool)
    diff = d["distance_diff_pct"]
    better = eq & (diff <= -NOISE_PCT)
    worse = eq & (diff >= NOISE_PCT)
    fewer = d["delta_vehicles_v2_minus_v1"] < 0
    more = d["delta_vehicles_v2_minus_v1"] > 0
    rows = [
        ("n_instances", len(d), ""),
        ("n_fleet_fewer", int(fewer.sum()), ", ".join(d.loc[fewer, "instance"])),
        ("n_fleet_more", int(more.sum()), ", ".join(d.loc[more, "instance"])),
        ("n_fleet_same", int(eq.sum()), ""),
        ("n_within_noise", int((eq & (diff.abs() < NOISE_PCT)).sum()), f"|Δ| < {NOISE_PCT:g}%"),
        ("n_better", int(better.sum()), f"Δ ≤ −{NOISE_PCT:g}%"),
        ("n_worse", int(worse.sum()), f"Δ ≥ +{NOISE_PCT:g}%"),
        ("diff_min", np.nan, f"{float(diff[eq].min()):+.1f}%"),
        ("diff_max", np.nan, f"{float(diff[eq].max()):+.1f}%"),
    ]
    return pd.DataFrame({
        "item": [text[k] for k, _, _ in rows], "count": [("-" if pd.isna(c) else int(c)) for _, c, _ in rows],
        "detail": [d for _, _, d in rows],
    })


def version_changes(by_instance: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """List the instances whose fleet changed or whose distance moved beyond the noise threshold.

    Parameters
    ----------
    by_instance : pandas.DataFrame
        First output of :func:`delivery_routing_vrptw.evaluation.compare_versions`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per changed instance, sorted by the distance difference (fleet changes first): ``instance``,
        ``family``, ``ref_vehicles``, ``v1_vehicles``, ``v2_vehicles``, ``v1_distance``, ``v2_distance``,
        ``version_distance_diff_pct`` (``"-"`` when the fleets differ) and ``v2_gap_distance_pct`` (gap of the v2
        plan to the SINTEF distance, defined only where the v2 fleet equals the reference, else ``"-"``).
    """
    d = by_instance
    changed = (d["delta_vehicles_v2_minus_v1"] != 0) | ~d["within_noise"].astype(bool)
    sub = d[changed].copy()
    sub["_fleet"] = (sub["delta_vehicles_v2_minus_v1"] == 0).astype(int)
    sub = sub.sort_values(["_fleet", "distance_diff_pct"]).reset_index(drop=True)
    out = pd.DataFrame({
        "instance": sub["instance"], "family": sub["family"], "ref_vehicles": sub["ref_vehicles"].astype(int),
        "v1_vehicles": sub["v1_vehicles"].astype(int), "v2_vehicles": sub["v2_vehicles"].astype(int),
        "v1_distance": sub["v1_distance"].round(1), "v2_distance": sub["v2_distance"].round(1),
        "version_distance_diff_pct": _dash(sub["distance_diff_pct"], 2),
        "v2_gap_distance_pct": _dash(sub["v2_gap_distance_pct"].where(sub["v2_distance_comparable"].astype(bool)), 2),
    })
    return out


# --------------------------------------------------------------------------- fixed cost of v1 against v2
def fixed_cost_ratio_table(
    instances: Mapping[str, Instance], scale_v2: int = 10000, k_v2: float = 10.0
) -> pd.DataFrame:
    """Compare the fixed cost per vehicle of v1 and v2 in distance units, instance by instance.

    v1 uses the absolute ``SolverConfig.vehicle_fixed_cost`` at the default scale (10^7 at scale 100, that is
    10^5 distance units); v2 uses ``round(k * scale * D_ub)`` with ``D_ub = sum_i 2 d(0, i)``, that is
    ``k * D_ub`` distance units. The two are therefore not equivalent: the ratio depends on the instance.

    Parameters
    ----------
    instances : mapping of str to Instance
        Instances to compare.
    scale_v2 : int, default 10000
        Integer scale of v2.
    k_v2 : float, default 10.0
        Relative fixed-cost multiplier of v2.

    Returns
    -------
    pandas.DataFrame
        One row per instance (sorted by name): ``instance``, ``family``, ``fixed_cost_v1`` and
        ``fixed_cost_v2`` (distance units: solver cost divided by the scale of its version) and
        ``ratio_pct`` (v2 as a percentage of v1).
    """
    v1 = SolverConfig()
    v2 = SolverConfig(scale=scale_v2, fixed_cost_k=k_v2)
    rows = []
    for name in sorted(instances):
        inst = instances[name]
        fc1 = fixed_cost_for(inst, v1) / v1.scale
        fc2 = fixed_cost_for(inst, v2) / v2.scale
        rows.append({"instance": name, "family": inst.family, "fixed_cost_v1": fc1, "fixed_cost_v2": fc2,
                     "ratio_pct": 100.0 * fc2 / fc1})
    return pd.DataFrame(rows)


def fixed_cost_ratio_summary(table: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Summarise the v2 / v1 fixed-cost ratio overall and by family (minimum, median, maximum).

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`fixed_cost_ratio_table`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per scope with label-key columns ``scope``, ``n_instances``, ``fixed_cost_v1``,
        ``ratio_min_pct``, ``ratio_median_pct``, ``ratio_max_pct`` and the instances of the extremes.
    """
    out = []
    for scope, sub in _scoped(table):
        ratio = sub["ratio_pct"]
        out.append({
            "scope": _scope_label(scope, lang), "n_instances": len(sub),
            "fixed_cost_v1": float(sub["fixed_cost_v1"].iloc[0]), "ratio_min_pct": round(float(ratio.min()), 1),
            "ratio_median_pct": round(float(ratio.median()), 1), "ratio_max_pct": round(float(ratio.max()), 1),
            "ratio_min_instance": sub.loc[ratio.idxmin(), "instance"],
            "ratio_max_instance": sub.loc[ratio.idxmax(), "instance"],
        })
    return pd.DataFrame(out)


def evaluation_fixed_cost_table(
    archive, split: Mapping[str, str], names, scale_v2: int = 10000, k_v2: float = 10.0
) -> pd.DataFrame:
    """Load the evaluation instances (evaluation-only loader) and return :func:`fixed_cost_ratio_table`.

    Parameters
    ----------
    archive : str or pathlib.Path
        SINTEF ``solomon-100.zip``.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``; any non-evaluation instance in ``names`` is refused.
    names : iterable of str
        Evaluation instances to compare.
    scale_v2, k_v2 : int, float
        v2 integer scale and relative fixed-cost multiplier.

    Returns
    -------
    pandas.DataFrame
        See :func:`fixed_cost_ratio_table`.
    """
    from .results import load_eval_instances

    return fixed_cost_ratio_table(load_eval_instances(archive, split, names), scale_v2, k_v2)
