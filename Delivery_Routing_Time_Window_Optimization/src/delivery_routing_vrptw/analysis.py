"""Tables for the verifier/baselines and development-tournament notebooks.

Functions return data frames whose column names are either keys of the ``columns`` label group (translate
them with :func:`delivery_routing_vrptw.eda.localize_columns`) or already translated flat names.
All numbers come from the verifier or from the frozen result files; nothing is hard-coded.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .baselines import greedy_deadline, nearest_neighbor_tw
from .benchmark import compare
from .distance import exact
from .io import FAMILIES, Instance, Reference, family_of
from .labels import get_labels
from .results import get_routes
from .verify import verify

BASELINES = {"greedy_deadline": greedy_deadline, "nearest_neighbor_tw": nearest_neighbor_tw}
CONSTRUCTIONS = ("greedy_deadline", "nearest_neighbor_tw", "PARALLEL_CHEAPEST_INSERTION",
                 "PATH_CHEAPEST_ARC", "SAVINGS")
CONFIG_KEYS = ("fleet_strategy", "first_solution", "metaheuristic")
EARLY_STOP_S = 29.0


def _dash(values: pd.Series, digits: int = 2) -> np.ndarray:
    """Round numbers and replace missing values by ``"-"`` (also normalises ``-0.0``)."""
    return np.array([("-" if pd.isna(v) else round(float(v), digits) + 0.0) for v in values], dtype=object)


# --------------------------------------------------------------------------- baselines (live run)
def run_baselines(
    instances: Mapping[str, Instance], references: Mapping[str, Reference], names: list[str]
) -> pd.DataFrame:
    """Run the two constructive baselines, verify them and compare with SINTEF.

    Parameters
    ----------
    instances : mapping
        Instances keyed by name.
    references : mapping
        SINTEF references keyed by name.
    names : list of str
        Instances to run.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, method) with ``instance``, ``family``, ``method``, ``feasible``,
        ``vehicles``, ``distance``, ``runtime_s``, ``ref_vehicles``, ``ref_distance``, ``delta_vehicles``,
        ``gap_distance_pct``, ``distance_comparable`` and the solution ``routes``. Metrics come from the
        verifier, not from the heuristics.
    """
    rows = []
    for name in names:
        inst = instances[name]
        matrix = exact(inst)
        for method, fn in BASELINES.items():
            sol = fn(inst)
            report = verify(inst, sol.routes, matrix)
            cmp = compare(report, references[name])
            rows.append({
                "instance": name, "family": family_of(name), "method": method,
                "feasible": report.feasible, "vehicles": report.vehicles, "distance": report.distance,
                "runtime_s": sol.runtime_s, "ref_vehicles": cmp["ref_vehicles"],
                "ref_distance": cmp["ref_distance"], "delta_vehicles": cmp["delta_vehicles"],
                "gap_distance_pct": cmp["gap_distance_pct"],
                "distance_comparable": cmp["distance_comparable"], "routes": sol.routes,
            })
    return pd.DataFrame(rows)


def baseline_table(df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Wide per-instance table: SINTEF vehicles, then vehicles/distance/runtime/Δ vehicles of each baseline.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`run_baselines`.
    lang : str
        Language of the flat column names.

    Returns
    -------
    pandas.DataFrame
        One row per instance; ``instance``, ``family`` and ``ref_vehicles`` are label keys, the other
        columns are already translated.
    """
    lab = get_labels(lang)
    first = df[df["method"] == df["method"].iloc[0]].set_index("instance")
    out = pd.DataFrame({"instance": first.index, "family": first["family"].to_numpy(),
                        "ref_vehicles": first["ref_vehicles"].to_numpy()})
    for method in [m for m in BASELINES if m in set(df["method"])]:
        sub = df[df["method"] == method].set_index("instance").loc[first.index]
        tag = lab["methods"][method]
        out[f"{tag} · {lab['metrics']['vehicles']}"] = sub["vehicles"].to_numpy()
        out[f"{tag} · {lab['metrics']['distance']}"] = sub["distance"].round(1).to_numpy()
        out[f"{tag} · {lab['metrics']['runtime']}"] = sub["runtime_s"].round(4).to_numpy()
        out[f"{tag} · {lab['metrics']['delta_vehicles']}"] = sub["delta_vehicles"].to_numpy()
    return out.reset_index(drop=True)


def baseline_family_summary(df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Per family (plus a total row): instances, mean Δ vehicles and feasible count of each baseline.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`run_baselines`.
    lang : str
        Language of the flat column names and of the total row.

    Returns
    -------
    pandas.DataFrame
        One row per family present, then a total row.
    """
    lab = get_labels(lang)
    cols = lab["columns"]
    rows = []
    groups = [(fam, df[df["family"] == fam]) for fam in FAMILIES if fam in set(df["family"])]
    groups.append((cols["total"], df))
    for fam, sub in groups:
        row: dict[str, object] = {"family": fam, "n_instances": sub["instance"].nunique()}
        for method in [m for m in BASELINES if m in set(df["method"])]:
            part = sub[sub["method"] == method]
            row[f"{lab['methods'][method]} · {cols['mean_delta_vehicles']}"] = round(
                float(part["delta_vehicles"].mean()), 2)
            row[f"{lab['methods'][method]} · {cols['n_feasible']}"] = int(part["feasible"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def baselines_match_grid(live: pd.DataFrame, grid: pd.DataFrame, tol: float = 1e-6) -> tuple[int, int]:
    """Count live baseline rows that reproduce the frozen grid rows (same vehicles and distance).

    Parameters
    ----------
    live : pandas.DataFrame
        Output of :func:`run_baselines`.
    grid : pandas.DataFrame
        Frozen development grid.
    tol : float
        Absolute tolerance on the distance.

    Returns
    -------
    tuple of int
        ``(n_matching, n_compared)``.
    """
    ref = grid[grid["method"].isin(BASELINES)].set_index(["instance", "method"])
    matched = 0
    for row in live.itertuples():
        other = ref.loc[(row.instance, row.method)]
        if int(other["vehicles"]) == row.vehicles and abs(float(other["distance"]) - row.distance) <= tol:
            matched += 1
    return matched, len(live)


# --------------------------------------------------------------------------- construction vs grid
def construction_rows(grid: pd.DataFrame) -> pd.DataFrame:
    """Return the reference cells without local search, with a ``construction`` column naming the method."""
    ref = grid[~grid["tournament"]].copy()
    ref["construction"] = np.where(ref["method"] == "ortools", ref["first_solution"], ref["method"])
    return ref


def construction_comparison(grid: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Compare the five constructions without local search against SINTEF.

    The distance excess is descriptive: it is computed over all instances, although a distance gap is only
    comparable (``n_fleet_equal_ref``) where the fleet equals the reference.

    Parameters
    ----------
    grid : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_dev_grid`.
    lang : str
        Language of the method names.

    Returns
    -------
    pandas.DataFrame
        One row per construction with ``method`` (translated), ``construction`` (key), ``n_instances``,
        ``n_feasible``, ``mean_delta_vehicles``, ``median_gap_all``, ``n_fleet_equal_ref`` and
        ``median_gap_equal_fleet``.
    """
    names = get_labels(lang)["constructions"]
    ref = construction_rows(grid)
    rows = []
    for key in CONSTRUCTIONS:
        sub = ref[ref["construction"] == key]
        if sub.empty:
            continue
        equal = sub[sub["distance_comparable"]]
        rows.append({
            "method": names[key], "construction": key, "n_instances": int(sub["instance"].nunique()),
            "n_feasible": int(sub["feasible"].sum()),
            "mean_delta_vehicles": float(sub["delta_vehicles"].mean()),
            "median_gap_all": float(sub["gap_distance_pct"].median()),
            "n_fleet_equal_ref": int(len(equal)),
            "median_gap_equal_fleet": float(equal["gap_distance_pct"].median()) if len(equal) else np.nan,
        })
    return pd.DataFrame(rows)


def construction_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Display version of :func:`construction_comparison` (rounded, ``"-"`` where undefined)."""
    out = comparison.drop(columns="construction").copy()
    out["mean_delta_vehicles"] = out["mean_delta_vehicles"].round(2)
    out["median_gap_all"] = _dash(out["median_gap_all"], 1)
    out["median_gap_equal_fleet"] = _dash(out["median_gap_equal_fleet"], 1)
    return out


# --------------------------------------------------------------------------- tournament
def tournament_rows(grid: pd.DataFrame) -> pd.DataFrame:
    """Return the 18-configuration tournament cells of the grid."""
    return grid[grid["tournament"]].copy()


def grid_coverage(grid: pd.DataFrame, split: Mapping[str, str], lang: str = "en") -> pd.DataFrame:
    """Compute coverage and feasibility of the grid.

    Parameters
    ----------
    grid : pandas.DataFrame
        Frozen development grid.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.
    lang : str
        Language of the metric names.

    Returns
    -------
    pandas.DataFrame
        Two columns, ``metric`` and ``value``.
    """
    names = get_labels(lang)["coverage"]
    tour = tournament_rows(grid)
    ref = construction_rows(grid)
    partial = grid["status"].fillna("").str.contains("PARTIAL")
    partial_construction = partial & (grid["metaheuristic"] == "NONE")
    items = {
        "cells_total": len(grid),
        "cells_tournament": len(tour),
        "cells_construction": int((ref["method"] == "ortools").sum()),
        "cells_baselines": int((ref["method"] != "ortools").sum()),
        "n_configs": int(tour.groupby(list(CONFIG_KEYS)).ngroups),
        "n_instances": int(grid["instance"].nunique()),
        "n_not_dev": int(sum(split.get(n) != "dev" for n in grid["instance"].unique())),
        "n_feasible": int(grid["feasible"].sum()),
        "n_partial": int(partial.sum()),
        "n_partial_construction": int(partial_construction.sum()),
    }
    return pd.DataFrame({"metric": [names[k] for k in items], "value": list(items.values())})


def ranking_table(ranking: pd.DataFrame) -> pd.DataFrame:
    """Return the ranking with a 1-based position column and rounded statistics.

    Parameters
    ----------
    ranking : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_ranking` (already sorted best first).

    Returns
    -------
    pandas.DataFrame
        Columns are keys of the ``columns`` label group.
    """
    out = ranking.copy()
    out.insert(0, "position", np.arange(1, len(out) + 1))
    for col, digits in (("mean_rank", 3), ("mean_delta_vehicles", 3), ("median_gap_equal_fleet", 3),
                        ("mean_runtime", 1)):
        out[col] = out[col].round(digits)
    return out[["position", *CONFIG_KEYS, "mean_rank", "mean_delta_vehicles", "n_fleet_equal_ref",
                "median_gap_equal_fleet", "mean_runtime"]]


def frozen_rows(grid: pd.DataFrame, config: Mapping[str, str]) -> pd.DataFrame:
    """Return the tournament cells of the frozen configuration, one per instance, sorted by instance."""
    mask = grid["tournament"].copy()
    for key in CONFIG_KEYS:
        mask &= grid[key] == config[key]
    return grid[mask].sort_values("instance").reset_index(drop=True)


def best_of_tournament(grid: pd.DataFrame) -> pd.DataFrame:
    """Return, per instance, the tournament cell with the fewest vehicles and then the shortest distance."""
    tour = tournament_rows(grid)
    tour = tour[tour["feasible"]].sort_values(["instance", "vehicles", "distance"], kind="mergesort")
    return tour.groupby("instance", sort=True).head(1).reset_index(drop=True)


def frozen_vs_reference(grid: pd.DataFrame, config: Mapping[str, str]) -> pd.DataFrame:
    """Compare the frozen configuration with SINTEF per instance, with the best of the 18 as an oracle.

    The distance gap is shown only where the fleet equals the reference (otherwise ``"-"``). The ``best_*``
    columns pick, per instance and after the fact, the best of the 18 configurations: an optimistic oracle,
    not a selectable configuration.

    Parameters
    ----------
    grid : pandas.DataFrame
        Frozen development grid.
    config : mapping
        Frozen configuration (keys ``fleet_strategy``, ``first_solution``, ``metaheuristic``).

    Returns
    -------
    pandas.DataFrame
        One row per instance; columns are keys of the ``columns`` label group.
    """
    frozen = frozen_rows(grid, config).set_index("instance")
    best = best_of_tournament(grid).set_index("instance").loc[frozen.index]
    gap = _dash(frozen["gap_distance_pct"].where(frozen["distance_comparable"]))
    return pd.DataFrame({
        "instance": frozen.index, "family": frozen["family"].to_numpy(),
        "ref_vehicles": frozen["ref_vehicles"].to_numpy(),
        "ref_distance": frozen["ref_distance"].to_numpy(),
        "vehicles": frozen["vehicles"].to_numpy(), "distance": frozen["distance"].round(2).to_numpy(),
        "delta_vs_ref": frozen["delta_vehicles"].to_numpy(), "gap_distance_pct": gap,
        "best_vehicles": best["vehicles"].to_numpy(),
        "best_distance": best["distance"].round(2).to_numpy(),
        "best_delta_vehicles": best["delta_vehicles"].to_numpy(),
    }).reset_index(drop=True)


def frozen_family_summary(table: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Summarise the output of :func:`frozen_vs_reference` per family, plus a total row.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`frozen_vs_reference`.
    lang : str
        Language of the total row.

    Returns
    -------
    pandas.DataFrame
        Columns are keys of the ``columns`` label group.
    """
    rows = []
    groups = [(fam, table[table["family"] == fam]) for fam in FAMILIES if fam in set(table["family"])]
    groups.append((get_labels(lang)["columns"]["total"], table))
    for fam, sub in groups:
        equal = sub[sub["delta_vs_ref"] == 0]
        gaps = pd.to_numeric(equal["gap_distance_pct"], errors="coerce")
        rows.append({
            "family": fam, "n_instances": len(sub),
            "mean_delta_vehicles": round(float(sub["delta_vs_ref"].mean()), 2),
            "n_fleet_equal_ref": len(equal),
            "median_gap_equal_fleet": round(float(gaps.median()), 2) if len(equal) else "-",
            "best_mean_delta_vehicles": round(float(sub["best_delta_vehicles"].mean()), 2),
        })
    return pd.DataFrame(rows)


def local_search_effect(grid: pd.DataFrame, config: Mapping[str, str]) -> pd.DataFrame:
    """Compare each construction-only cell with the same first solution plus the frozen local search.

    For every first solution, the cell without metaheuristic is paired, per instance, with the tournament
    cell that has the same first solution and the frozen fleet strategy and metaheuristic.

    Parameters
    ----------
    grid : pandas.DataFrame
        Frozen development grid.
    config : mapping
        Frozen configuration.

    Returns
    -------
    pandas.DataFrame
        One row per first solution with ``first_solution``, ``delta_construction``, ``delta_search``,
        ``vehicles_saved_mean`` and the numbers of instances with fewer, equal and more vehicles.
    """
    base = construction_rows(grid)
    base = base[base["method"] == "ortools"].set_index(["first_solution", "instance"])
    rows = []
    for first in sorted(base.index.get_level_values(0).unique()):
        mask = grid["tournament"].copy()
        mask &= grid["fleet_strategy"] == config["fleet_strategy"]
        mask &= grid["metaheuristic"] == config["metaheuristic"]
        mask &= grid["first_solution"] == first
        search = grid[mask].set_index("instance")
        cons = base.loc[first].loc[search.index]
        saved = cons["vehicles"].to_numpy() - search["vehicles"].to_numpy()
        rows.append({
            "first_solution": first,
            "delta_construction": round(float(cons["delta_vehicles"].mean()), 2),
            "delta_search": round(float(search["delta_vehicles"].mean()), 2),
            "vehicles_saved_mean": round(float(saved.mean()), 2),
            "n_better": int((saved > 0).sum()), "n_equal": int((saved == 0).sum()),
            "n_worse": int((saved < 0).sum()),
        })
    return pd.DataFrame(rows)


def limitations_table(grid: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Audit, from the data, the known limitations of the grid.

    ``reduce_fleet`` cells record their attempts as ``[vehicle_limit, feasible, vehicles, distance]``; the
    first attempt is the initial solve and the second is the first fleet reduction.

    Parameters
    ----------
    grid : pandas.DataFrame
        Frozen development grid.
    lang : str
        Language of the metric names.

    Returns
    -------
    pandas.DataFrame
        Two columns, ``metric`` and ``value`` (formatted text).
    """
    names = get_labels(lang)["limits"]
    tour = tournament_rows(grid)
    red = tour[tour["fleet_strategy"] == "reduce_fleet"]
    failed = sum(
        1 for a in red["attempts"] if isinstance(a, list) and len(a) >= 2 and not bool(a[1][1])
    )
    fixed = tour[tour["fleet_strategy"] == "fixed_cost"]
    tabu = fixed[fixed["metaheuristic"] == "TABU_SEARCH"]
    gls = fixed[fixed["metaheuristic"] == "GUIDED_LOCAL_SEARCH"]
    per_cell = tour.groupby(["instance", *CONFIG_KEYS]).size()
    items = {
        "reduce_cells": f"{len(red)}",
        "reduce_failed": f"{failed} / {len(red)}",
        "reduce_runtime": f"{red['runtime_s'].mean():.1f}",
        "tabu_cells": f"{len(tabu)}",
        "tabu_early": f"{int((tabu['runtime_s'] < EARLY_STOP_S).sum())} / {len(tabu)}",
        "tabu_runtime": f"{tabu['runtime_s'].mean():.1f}",
        "gls_runtime": f"{gls['runtime_s'].mean():.1f}",
        "replicates": f"{int(per_cell.max())}",
    }
    labels = {k: v.format(threshold=EARLY_STOP_S) for k, v in names.items()}
    return pd.DataFrame({"metric": [labels[k] for k in items], "value": list(items.values())})


def reduce_fleet_timing(
    grid: pd.DataFrame, first_step_s: float = 10.0, step_s: float = 5.0, budget_s: float = 30.0
) -> pd.DataFrame:
    """Summarise, per first solution, how long the ``reduce_fleet`` cells really ran.

    The loop solves once with ``first_step_s`` and then, one fleet size lower, with at most ``step_s``;
    it stops at the first failure and discards the remaining budget. A total runtime close to
    ``first_step_s`` therefore means that the second solve failed almost at once (construction failure),
    whereas a total close to ``first_step_s + step_s`` means that the second solve used its whole step.

    Parameters
    ----------
    grid : pandas.DataFrame
        Frozen development grid (``fleet_strategy``, ``first_solution``, ``runtime_s``, ``tournament``).
    first_step_s : float
        Time limit of the first solve.
    step_s : float
        Time limit of each later solve.
    budget_s : float
        Total budget of the cell, used for the share of the budget actually used.

    Returns
    -------
    pandas.DataFrame
        One row per first solution with ``first_solution``, ``n_cells``, ``runtime_min``, ``runtime_median``,
        ``runtime_max``, ``n_fast_fail`` (cells with runtime below ``first_step_s + 1``, that is, a second
        solve that failed in under 1 s, or a first solve that stopped early), ``n_step_used`` (cells with
        runtime of at least ``first_step_s + step_s - 1``, that is, a second solve that used its whole
        step) and ``budget_used_pct`` (median runtime as a share of ``budget_s``).
    """
    red = tournament_rows(grid)
    red = red[red["fleet_strategy"] == "reduce_fleet"]
    rows = []
    for first, g in red.groupby("first_solution", sort=True):
        t = g["runtime_s"].astype(float)
        rows.append({
            "first_solution": first,
            "n_cells": len(g),
            "runtime_min": round(float(t.min()), 2),
            "runtime_median": round(float(t.median()), 2),
            "runtime_max": round(float(t.max()), 2),
            "n_fast_fail": int((t < first_step_s + 1.0).sum()),
            "n_step_used": int((t >= first_step_s + step_s - 1.0).sum()),
            "budget_used_pct": round(100.0 * float(t.median()) / budget_s, 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- time-budget sensitivity
def time_curve_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the time-budget sensitivity by budget.

    The distance gap is only comparable where the fleet equals the reference, so the median gap uses only
    those instances (the set can change from one budget to the next).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_time_sensitivity`.

    Returns
    -------
    pandas.DataFrame
        One row per budget (ascending) with ``budget_s``, ``n_instances``, ``mean_delta_vehicles``,
        ``n_fleet_equal_ref``, ``median_gap_equal_fleet`` (NaN if no instance has the reference fleet)
        and ``mean_runtime``.
    """
    rows = []
    for budget, sub in df.groupby("budget_s", sort=True):
        equal = sub[sub["distance_comparable"]]
        rows.append({
            "budget_s": float(budget), "n_instances": int(sub["instance"].nunique()),
            "mean_delta_vehicles": float(sub["delta_vehicles"].mean()),
            "n_fleet_equal_ref": int(len(equal)),
            "median_gap_equal_fleet": float(equal["gap_distance_pct"].median()) if len(equal) else np.nan,
            "mean_runtime": float(sub["runtime_s"].mean()),
        })
    return pd.DataFrame(rows)


def time_curve_table_display(curve: pd.DataFrame) -> pd.DataFrame:
    """Display version of :func:`time_curve_table` (rounded, ``"-"`` where the gap is undefined)."""
    out = curve.copy()
    out["budget_s"] = out["budget_s"].map(lambda b: f"{b:g}")
    out["mean_delta_vehicles"] = out["mean_delta_vehicles"].round(2)
    out["median_gap_equal_fleet"] = _dash(out["median_gap_equal_fleet"], 2)
    out["mean_runtime"] = out["mean_runtime"].round(1)
    return out


def time_curve_by_instance(df: pd.DataFrame) -> pd.DataFrame:
    """Return the number of vehicles per instance (rows) and budget (columns), with the reference.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_time_sensitivity`.

    Returns
    -------
    pandas.DataFrame
        Columns ``instance``, ``family``, ``ref_vehicles`` and one column per budget named like ``"30 s"``;
        sorted by family and instance name.
    """
    wide = df.pivot(index="instance", columns="budget_s", values="vehicles")
    wide.columns = [f"{float(b):g} s" for b in wide.columns]
    meta = df.drop_duplicates("instance").set_index("instance")[["family", "ref_vehicles"]]
    out = meta.join(wide).reset_index()
    return out.sort_values(["family", "instance"]).reset_index(drop=True)


# --------------------------------------------------------------------------- notebook helpers
def dev_instance_table(
    instances: Mapping[str, Instance], references: Mapping[str, Reference]
) -> pd.DataFrame:
    """Describe the loaded instances and their SINTEF references (columns are label keys).

    Parameters
    ----------
    instances : mapping
        Instances keyed by name.
    references : mapping
        SINTEF references keyed by name.

    Returns
    -------
    pandas.DataFrame
        One row per instance, ordered by family and name.
    """
    rows = [{
        "instance": name, "family": family_of(name), "n_customers": inst.n_customers,
        "capacity": inst.capacity, "horizon": float(inst.horizon), "ref_vehicles": references[name].vehicles,
        "ref_distance": references[name].distance,
    } for name, inst in instances.items()]
    out = pd.DataFrame(rows)
    rank = {f: i for i, f in enumerate(FAMILIES)}
    out = out.sort_values(["family", "instance"], key=lambda s: s.map(rank) if s.name == "family" else s)
    return out.reset_index(drop=True)


def selected_config_table(selected: Mapping, lang: str = "en") -> pd.DataFrame:
    """Tabulate the frozen configuration and its selection statistics.

    Parameters
    ----------
    selected : mapping
        Output of :func:`delivery_routing_vrptw.results.load_selected_config`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Two columns, ``metric`` and ``value``.
    """
    cols = get_labels(lang)["columns"]
    stats = selected["selection_stats"]
    items = {k: selected["config"][k] for k in CONFIG_KEYS}
    items["mean_rank"] = round(stats["mean_rank"], 3)
    items["mean_delta_vehicles"] = round(stats["mean_delta_vehicles"], 3)
    items["n_fleet_equal_ref"] = int(stats["n_fleet_equal_ref"])
    items["median_gap_equal_fleet"] = round(stats["median_gap_equal_fleet"], 3)
    items["mean_runtime"] = round(stats["mean_runtime"], 1)
    items["n_instances"] = int(stats["n_instances"])
    return pd.DataFrame({"metric": [cols[k] for k in items], "value": list(items.values())})


def baseline_routes(live: pd.DataFrame, instance: str, method: str) -> list[list[int]]:
    """Return the routes of one baseline on one instance from the output of :func:`run_baselines`."""
    rows = live[(live["instance"] == instance) & (live["method"] == method)]
    if len(rows) != 1:
        raise ValueError(f"no unique row for {instance}/{method}")
    return rows.iloc[0]["routes"]


def baseline_checks(live: pd.DataFrame, grid: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Report how many baseline solutions are feasible and how many reproduce the frozen grid rows.

    Parameters
    ----------
    live : pandas.DataFrame
        Output of :func:`run_baselines`.
    grid : pandas.DataFrame
        Frozen development grid.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Two columns, ``metric`` and ``value`` (``"k / n"`` strings).
    """
    names = get_labels(lang)["checks"]
    matched, total = baselines_match_grid(live, grid)
    return pd.DataFrame({
        "metric": [names["feasible"], names["match_grid"]],
        "value": [f"{int(live['feasible'].sum())} / {len(live)}", f"{matched} / {total}"],
    })


def baseline_panels(
    name: str, instances: Mapping[str, Instance], live: pd.DataFrame, grid: pd.DataFrame, lang: str = "en"
) -> list[tuple[Instance, str, list[list[int]]]]:
    """Build route-map panels: greedy, nearest neighbour and the OR-Tools Savings construction.

    Parameters
    ----------
    name : str
        Instance name.
    instances : mapping
        Instances keyed by name.
    live : pandas.DataFrame
        Output of :func:`run_baselines` (provides the two baseline routes).
    grid : pandas.DataFrame
        Frozen development grid (provides the Savings construction routes).
    lang : str
        Label language.

    Returns
    -------
    list of tuple
        ``(instance, label, routes)`` ready for :func:`delivery_routing_vrptw.viz_results.plot_route_panels`.
    """
    names = get_labels(lang)["constructions"]
    savings = get_routes(grid, name, method="ortools", first_solution="SAVINGS", metaheuristic="NONE")
    return [
        (instances[name], names["greedy_deadline"], baseline_routes(live, name, "greedy_deadline")),
        (instances[name], names["nearest_neighbor_tw"], baseline_routes(live, name, "nearest_neighbor_tw")),
        (instances[name], names["SAVINGS"], savings),
    ]


def frozen_panels(
    names: list[str], instances: Mapping[str, Instance], grid: pd.DataFrame, config: Mapping[str, str],
    lang: str = "en",
) -> list[tuple[Instance, str, list[list[int]]]]:
    """Build route-map panels of the frozen configuration for several instances.

    Parameters
    ----------
    names : list of str
        Instance names.
    instances : mapping
        Instances keyed by name.
    grid : pandas.DataFrame
        Frozen development grid.
    config : mapping
        Frozen configuration.
    lang : str
        Label language.

    Returns
    -------
    list of tuple
        ``(instance, label, routes)`` ready for :func:`delivery_routing_vrptw.viz_results.plot_route_panels`.
    """
    label = get_labels(lang)["panels"]["frozen"].format(config=" / ".join(config[k] for k in CONFIG_KEYS))
    selection = {k: config[k] for k in CONFIG_KEYS}
    return [(instances[n], label, get_routes(grid, n, tournament=True, **selection)) for n in names]
