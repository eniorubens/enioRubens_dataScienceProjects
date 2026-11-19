"""Tables and panels for the business-scenarios notebook (layer B, no SINTEF reference anywhere).

Scenarios are compared with the plan's own baseline (window factor 1.0, early-delivery tolerance 0), solved
by the same solver. Nothing here reads a SINTEF reference, a reference column or the evaluation set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .io import FAMILIES, Instance, family_of
from .labels import get_labels
from .results import assert_dev_only
from .route_compare import early_delivery_summary, reallocation_summary
from .scenarios import allow_early, tolerance_label, window_scenario_summary

BASE_TOLERANCE = "0"


def tolerance_text(label: str, lang: str = "en") -> str:
    """Return a display text for a tolerance label: ``"25%"``, ``"100%"``, ``"full"`` / ``"total"``, ``"noise"``.

    Parameters
    ----------
    label : str
        Tolerance label (``"0"``, ``"0.25"``, ``"1"``, ``"full"`` or ``"noise"``).
    lang : str
        Label language.

    Returns
    -------
    str
        Display text.
    """
    values = get_labels(lang)["values"]
    if label in ("full", "noise"):
        return values[label]
    return f"{100 * float(label):g}%"


def _tol(label: str) -> float:
    return math.inf if label == "full" else float(label)


# --------------------------------------------------------------------------- configuration and checks
def config_table(cfg: ExperimentConfig, lang: str = "en") -> pd.DataFrame:
    """Tabulate the effective layer-B configuration read from ``config/experiment.toml``.

    Parameters
    ----------
    cfg : ExperimentConfig
        Output of :func:`delivery_routing_vrptw.config.load_config`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric`` and ``value``: the solver strategy and the parameters of the window-widening
        and early-delivery scenarios.
    """
    names = get_labels(lang)["config_rows"]
    w, e, s = cfg.window_widening, cfg.early_delivery, cfg.solver
    items = {
        "fleet_strategy": s.fleet_strategy,
        "first_solution": s.first_solution,
        "metaheuristic": s.metaheuristic,
        "window_factors": ", ".join(f"{f:g}" for f in w.factors),
        "window_mode": w.mode,
        "window_budget": f"{w.time_limit_s:g}",
        "window_set": w.instance_set,
        "early_tolerances": ", ".join(tolerance_label(t) for t in e.levels()),
        "early_replicates": f"{e.replicates}",
        "early_budget": f"{e.time_limit_s:g}",
        "early_set": e.instance_set,
    }
    return pd.DataFrame({"metric": [names[k] for k in items], "value": list(items.values())})


def run_checks(
    window: pd.DataFrame, early: pd.DataFrame, split: Mapping[str, str], lang: str = "en"
) -> pd.DataFrame:
    """Audit the two frozen scenario runs: cells, feasibility and instance set.

    Parameters
    ----------
    window : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_window_scenarios`.
    early : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_early_delivery`.
    split : mapping
        Instance name -> ``"dev"``/``"eval"``.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric`` and ``value`` (formatted text).

    Raises
    ------
    ValueError
        If any instance is not a development instance.
    """
    names = get_labels(lang)["run_checks"]
    both = pd.concat([window["instance"], early["instance"]]).unique()
    assert_dev_only(both, split)
    items = {
        "window_cells": f"{len(window)}",
        "window_feasible": f"{int(window['feasible'].astype(bool).sum())} / {len(window)}",
        "early_cells": f"{len(early)}",
        "early_feasible": f"{int(early['feasible'].astype(bool).sum())} / {len(early)}",
        "early_replicates": f"{int(early['replicate'].nunique())}",
        "n_instances": f"{len(both)}",
        "not_dev": f"{sum(1 for n in both if split.get(n) != 'dev')}",
    }
    return pd.DataFrame({"metric": [names[k] for k in items], "value": list(items.values())})


# --------------------------------------------------------------------------- window widening
def window_results_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise the window-widening run against its factor-1.0 baseline, with the instance family.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_window_scenarios`.

    Returns
    -------
    pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.scenarios.window_scenario_summary` plus ``family``.
    """
    out = window_scenario_summary(rows)
    out.insert(1, "family", out["instance"].map(family_of))
    return out


FLEET_CLASSES = ("equal", "reduced", "increased")


def fleet_class(vehicles_saved: float) -> str:
    """Classify a scenario plan against its baseline plan: ``"equal"``, ``"reduced"`` or ``"increased"``.

    Parameters
    ----------
    vehicles_saved : float
        Baseline vehicles minus scenario vehicles (``NaN`` is treated as missing and returns ``"missing"``).

    Returns
    -------
    str
        ``"reduced"`` if positive, ``"increased"`` if negative, ``"equal"`` if zero.
    """
    if vehicles_saved != vehicles_saved:
        return "missing"
    return "reduced" if vehicles_saved > 0 else "increased" if vehicles_saved < 0 else "equal"


def _with_fleet(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.assign(fleet=summary["vehicles_saved"].map(fleet_class))


def _stat(values: pd.Series, how: str, digits: int = 1) -> float:
    """Mean or median rounded, ``NaN`` for an empty selection."""
    return round(float(getattr(values, how)()), digits) if len(values) else float("nan")


def window_by_factor(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the vehicles saved by window factor (baseline factor 1.0 excluded).

    The distance change is reported separately by fleet class in :func:`window_distance_by_fleet`, because
    a plan with fewer vehicles and a plan with the same fleet are not comparable in distance.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`window_results_summary`.

    Returns
    -------
    pandas.DataFrame
        One row per factor; label-key columns ``factor``, ``n_instances``, ``vehicles_saved_mean``,
        ``n_with_saving`` (instances with fewer vehicles), ``n_with_extra`` (instances with more) and
        ``n_equal`` (instances with the same fleet).
    """
    rows = []
    for factor, sub in summary[summary["factor"] > 1.0].groupby("factor"):
        rows.append({
            "factor": f"{factor:g}×", "n_instances": len(sub),
            "vehicles_saved_mean": round(float(sub["vehicles_saved"].mean()), 2),
            "n_with_saving": int((sub["vehicles_saved"] > 0).sum()),
            "n_with_extra": int((sub["vehicles_saved"] < 0).sum()),
            "n_equal": int((sub["vehicles_saved"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def window_distance_by_fleet(summary: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Report the distance change by factor and by fleet class (equal, reduced, increased), never pooled.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`window_results_summary`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per (factor, fleet class present): label-key columns ``factor``, ``fleet_class`` (text),
        ``n_instances``, ``vehicles_saved_mean``, ``distance_change_mean`` and ``distance_change_median``
        (percent against the factor-1.0 plan of the same instance, computed only within the class).
    """
    values = get_labels(lang)["values"]
    sub = _with_fleet(summary[summary["factor"] > 1.0])
    rows = []
    for factor, per_factor in sub.groupby("factor"):
        for cls in FLEET_CLASSES:
            part = per_factor[per_factor["fleet"] == cls]
            if part.empty:
                continue
            rows.append({
                "factor": f"{factor:g}×", "fleet_class": values[f"fleet_{cls}"], "n_instances": len(part),
                "vehicles_saved_mean": round(float(part["vehicles_saved"].mean()), 2),
                "distance_change_mean": _stat(part["distance_change_pct"], "mean"),
                "distance_change_median": _stat(part["distance_change_pct"], "median"),
            })
    return pd.DataFrame(rows)


def window_by_family(summary: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Aggregate the window-widening effect by family and factor, one block of columns per factor.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`window_results_summary`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per family (``family`` is a label key) and, for every factor above 1.0, the mean vehicles
        saved, the number of instances whose fleet stayed equal and the mean distance change (%) **among
        those instances only** (``NaN`` when none); column names are translated.
    """
    text = get_labels(lang)["business"]
    sub = _with_fleet(summary[summary["factor"] > 1.0])
    factors = sorted(sub["factor"].unique())
    rows = []
    for fam in FAMILIES:
        part = sub[sub["family"] == fam]
        if part.empty:
            continue
        row: dict[str, object] = {"family": fam, "n_instances": part["instance"].nunique()}
        for f in factors:
            one = part[part["factor"] == f]
            equal = one[one["fleet"] == "equal"]
            row[text["factor_vehicles"].format(factor=f)] = round(float(one["vehicles_saved"].mean()), 2)
            row[text["factor_n_equal"].format(factor=f)] = len(equal)
            row[text["factor_distance"].format(factor=f)] = _stat(equal["distance_change_pct"], "mean")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- early delivery
def early_results_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise the early-delivery run against its tolerance-0 baseline, with the instance family.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_early_delivery`.

    Returns
    -------
    pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.route_compare.early_delivery_summary` plus ``family``.
    """
    out = early_delivery_summary(rows)
    out.insert(1, "family", out["instance"].map(family_of))
    return out


def early_by_tolerance(summary: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Aggregate the early-delivery effect by tolerance and fleet class, with both departure conventions.

    The route duration is reported with every vehicle leaving the depot at time 0 (``duration_pct``) and with
    the optimal departure (``duration_min_pct``); their difference measures how much the first convention
    inflates the gain by counting waiting that a later departure would have avoided anyway. Distance and
    duration are never pooled across fleet classes: a plan with fewer vehicles is not comparable in distance
    with a plan that has the same fleet.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`early_results_summary` (representative replicate of each cell).
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per (tolerance, fleet class present), baseline excluded. Label-key columns: ``tolerance``,
        ``fleet_class``, ``n_instances``, ``vehicles_saved_mean``, ``distance_pct_mean``,
        ``duration_pct_mean`` (depart at 0), ``duration_min_pct_mean`` (optimal departure), ``inflation_pp``
        (optimal minus depart-at-0, in percentage points) and ``makespan_pct_mean``; means over instances of
        the representative plans, within the class.
    """
    values = get_labels(lang)["values"]
    sub = _with_fleet(summary[summary["tolerance_label"] != BASE_TOLERANCE])
    rows = []
    for _level, per_tol in sub.groupby("tolerance", sort=True):
        label = str(per_tol["tolerance_label"].iloc[0])
        for cls in FLEET_CLASSES:
            part = per_tol[per_tol["fleet"] == cls]
            if part.empty:
                continue
            rows.append({
                "tolerance": tolerance_text(label, lang),
                "fleet_class": values[f"fleet_{cls}"],
                "n_instances": len(part),
                "vehicles_saved_mean": round(float(part["vehicles_saved"].mean()), 2),
                "distance_pct_mean": _stat(part["distance_pct"], "mean"),
                "duration_pct_mean": _stat(part["duration_pct"], "mean"),
                "duration_min_pct_mean": _stat(part["duration_min_pct"], "mean"),
                "inflation_pp": round(float((part["duration_min_pct"] - part["duration_pct"]).mean()), 1),
                "makespan_pct_mean": _stat(part["makespan_pct"], "mean"),
            })
    return pd.DataFrame(rows)


def early_by_family(
    summary: pd.DataFrame, metric: str, lang: str = "en", fleet: str | None = None
) -> pd.DataFrame:
    """Return the mean of one early-delivery metric per family (rows) and tolerance (columns).

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`early_results_summary`.
    metric : str
        A column of the summary, for example ``"duration_min_pct"`` or ``"vehicles_saved"``.
    lang : str
        Label language.
    fleet : {"equal", "reduced", "increased"}, optional
        If given, only the instances of that fleet class (at each tolerance) enter the mean; cells without
        instance are ``NaN``. Required for every metric other than ``vehicles_saved`` so that distance and
        duration are never pooled across fleet classes.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric`` (translated metric name; with the fleet class when filtered), ``family`` (label
        key), ``n_instances`` (instances of the family, before filtering) and one column per tolerance
        (baseline excluded).

    Raises
    ------
    ValueError
        If ``fleet`` is missing for a metric that mixes fleets, or is not a fleet class.
    """
    if fleet is None and metric != "vehicles_saved":
        raise ValueError("a fleet class is required for metrics that depend on the fleet")
    if fleet is not None and fleet not in FLEET_CLASSES:
        raise ValueError(f"fleet must be one of {FLEET_CLASSES}")
    text = get_labels(lang)
    metric_name = text["business"][f"metric_{metric}"]
    if fleet is not None:
        metric_name = f"{metric_name} · {text['values'][f'fleet_{fleet}']}"
    sub = _with_fleet(summary[summary["tolerance_label"] != BASE_TOLERANCE])
    order = sub.drop_duplicates("tolerance_label").sort_values("tolerance")["tolerance_label"].tolist()
    counts = sub.groupby("family")["instance"].nunique()
    if fleet is not None:
        sub = sub[sub["fleet"] == fleet]
    wide = sub.pivot_table(index="family", columns="tolerance_label", values=metric, aggfunc="mean")
    wide = wide.reindex(index=[f for f in FAMILIES if f in counts.index], columns=order).round(2)
    wide.columns = [tolerance_text(c, lang) for c in wide.columns]
    wide.insert(0, "n_instances", counts.reindex(wide.index).astype(int))
    wide.insert(0, "metric", metric_name)
    return wide.rename_axis("family").reset_index()


def early_fleet_counts(summary: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Count, per family and tolerance, the instances whose fleet is equal, reduced or increased.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`early_results_summary`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``family``, ``tolerance``, ``n_equal``, ``n_reduced`` and ``n_increased`` (baseline
        tolerance excluded), one row per family and tolerance.
    """
    sub = _with_fleet(summary[summary["tolerance_label"] != BASE_TOLERANCE])
    rows = []
    for fam in FAMILIES:
        for _level, part in sub[sub["family"] == fam].groupby("tolerance", sort=True):
            rows.append({
                "family": fam, "tolerance": tolerance_text(str(part["tolerance_label"].iloc[0]), lang),
                **{f"n_{cls}": int((part["fleet"] == cls).sum()) for cls in FLEET_CLASSES},
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- reallocation
def reallocation_display(table: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Summarise plan-to-plan reallocation by tolerance, with the run-to-run noise floor as the last row.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.route_compare.reallocation_table`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``tolerance``, ``n_comparisons``, ``reassigned_pct`` (customers outside the matched
        route), ``pairs_kept_pct``, ``pair_agreement_pct`` and ``arcs_kept_pct`` (medians, in percent).
    """
    summ = reallocation_summary(table)
    return pd.DataFrame({
        "tolerance": [tolerance_text(t, lang) for t in summ["tolerance_label"]],
        "n_comparisons": summ["n"].to_numpy(),
        "reassigned_pct": (100 * summ["customers_reassigned_share"]).round(1).to_numpy(),
        "pairs_kept_pct": (100 * summ["pairs_kept"]).round(1).to_numpy(),
        "pair_agreement_pct": (100 * summ["pair_agreement"]).round(1).to_numpy(),
        "arcs_kept_pct": (100 * summ["arcs_kept"]).round(1).to_numpy(),
    })


def reallocation_by_family(table: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Median share of customers reassigned per family (rows) and tolerance (columns, noise last).

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.route_compare.reallocation_table`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Columns ``family`` (label key) and one column per tolerance, in percent.
    """
    t = table.assign(family=table["instance"].map(family_of))
    wide = t.pivot_table(index="family", columns="tolerance_label", values="customers_reassigned_share",
                         aggfunc="median")
    labels = list(wide.columns)
    labels.sort(key=lambda s: (s == "noise", _tol(s) if s != "noise" else 0.0))
    wide = (100 * wide.reindex(index=[f for f in FAMILIES if f in wide.index], columns=labels)).round(1)
    wide.columns = [tolerance_text(c, lang) for c in wide.columns]
    return wide.rename_axis("family").reset_index()


# --------------------------------------------------------------------------- before/after example
def pick_early_example(summary: pd.DataFrame, label: str = "full") -> str:
    """Choose the instance to draw: among those where early delivery saves a vehicle, the largest duration gain.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`early_results_summary`.
    label : str
        Tolerance label of the scenario.

    Returns
    -------
    str
        Instance name. If no instance saves a vehicle, the one with the largest duration gain overall.
    """
    sub = summary[summary["tolerance_label"] == label]
    saving = sub[sub["vehicles_saved"] >= 1]
    pool = saving if len(saving) else sub
    return str(pool.sort_values(["duration_min_pct", "instance"]).iloc[0]["instance"])


def before_after_panels(
    name: str, instances: Mapping[str, Instance], rows: pd.DataFrame, lang: str = "en",
    label: str = "full", replicate: int = 1,
) -> list[tuple[Instance, str, list[list[int]]]]:
    """Build route-map panels of the baseline plan and the early-delivery plan of one replicate.

    The scenario panel is verified against the transformed instance (same customers and coordinates, earlier
    window openings), so its title reports the verifier's vehicles and distance.

    Parameters
    ----------
    name : str
        Development instance name.
    instances : mapping
        Instances keyed by name.
    rows : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_early_delivery`.
    lang : str
        Label language.
    label : str
        Scenario tolerance label.
    replicate : int
        Replicate id used for both plans.

    Returns
    -------
    list of tuple
        ``(instance, label, routes)`` for the baseline and the scenario, ready for
        :func:`delivery_routing_vrptw.viz_results.plot_route_panels`; the labels carry the route duration
        with optimal departures.

    Raises
    ------
    ValueError
        If a plan is not found exactly once.
    """
    text = get_labels(lang)["panels"]

    def one(tol_label: str) -> pd.Series:
        sel = rows[(rows["instance"] == name) & (rows["tolerance_label"] == tol_label)
                   & (rows["replicate"] == replicate)]
        if len(sel) != 1:
            raise ValueError(f"expected one plan for {name}/{tol_label}/{replicate}, found {len(sel)}")
        return sel.iloc[0]

    base, scen = one(BASE_TOLERANCE), one(label)
    transformed = allow_early(instances[name], _tol(label))
    return [
        (instances[name], text["early_before"].format(rep=replicate, dur=float(base["total_duration_min"])),
         base["routes"]),
        (transformed, text["early_after"].format(tol=tolerance_text(label, lang), rep=replicate,
                                                 dur=float(scen["total_duration_min"])), scen["routes"]),
    ]


# --------------------------------------------------------------------------- limitations
def noise_floor_table(summary: pd.DataFrame, realloc: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Quantify the run-to-run noise of the baseline plan in the early-delivery run.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`early_results_summary`.
    realloc : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.route_compare.reallocation_table`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric`` and ``value``: replicates, baseline instances whose fleet or distance differs
        between replicates, and the median reassigned share between baseline replicates.
    """
    names = get_labels(lang)["noise_rows"]
    base = summary[summary["tolerance_label"] == BASE_TOLERANCE]
    noise = realloc[realloc["tolerance_label"] == "noise"]
    items = {
        "replicates": f"{int(base['n_replicates'].max())}",
        "fleet_differs": f"{int((base['vehicles_min'] != base['vehicles_max']).sum())} / {len(base)}",
        "distance_differs": (
            f"{int((~np.isclose(base['distance_min'], base['distance_max'], rtol=0, atol=1e-6)).sum())}"
            f" / {len(base)}"
        ),
        "noise_reassigned": f"{100 * float(noise['customers_reassigned_share'].median()):.1f}",
    }
    return pd.DataFrame({"metric": [names[k] for k in items], "value": list(items.values())})


def run_comparison_table(official: pd.DataFrame, contended: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Compare two runs of the same window experiment (clean run against the run that shared the machine).

    Parameters
    ----------
    official : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_window_scenarios` for the official run.
    contended : pandas.DataFrame
        The same for the first run, which coincided with other work on the machine.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric`` and ``value``: cells compared, cells whose fleet differs (and by how much), cells
        whose distance differs, and instances whose fleet differs at the baseline factor 1.0.
    """
    names = get_labels(lang)["run_rows"]
    keys = ["instance", "factor"]
    merged = official.merge(contended, on=keys, suffixes=("_a", "_b"))
    diff_v = (merged["vehicles_a"] - merged["vehicles_b"]).abs()
    diff_d = (merged["distance_a"] - merged["distance_b"]).abs() > 1e-6
    base = merged[merged["factor"] == 1.0]
    base_diff = (base["vehicles_a"] != base["vehicles_b"]).sum()
    items = {
        "cells": f"{len(merged)}",
        "fleet_differs": f"{int((diff_v > 0).sum())} / {len(merged)}",
        "max_fleet_diff": f"{int(diff_v.max())}",
        "distance_differs": f"{int(diff_d.sum())} / {len(merged)}",
        "baseline_fleet_differs": f"{int(base_diff)} / {len(base)}",
    }
    return pd.DataFrame({"metric": [names[k] for k in items], "value": list(items.values())})


def _lexicographic_worsening(
    frame: pd.DataFrame, level_col: str, vehicles_col: str, distance_col: str, label
) -> list[tuple]:
    """Cells of each instance whose (vehicles, distance) is lexicographically worse than an earlier level."""
    out = []
    for inst, g in frame.sort_values(["instance", level_col]).groupby("instance"):
        best: tuple[float, float] | None = None
        for r in g.itertuples():
            veh, dist = getattr(r, vehicles_col), getattr(r, distance_col)
            if np.isnan(veh) or np.isnan(dist):
                continue
            if best is not None:
                if veh > best[0]:
                    out.append((inst, label(r), int(veh), float(dist), int(best[0]), float(best[1]), "fleet"))
                elif veh == best[0] and dist > best[1] + 1e-6:
                    out.append((inst, label(r), int(veh), float(dist), int(best[0]), float(best[1]), "distance"))
            best = (veh, dist) if best is None else min(best, (veh, dist))
    return out


def nonmonotonic_cases(window: pd.DataFrame, early: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """List scenario cells that are lexicographically worse than the best plan of a less permissive level.

    Relaxing windows can never worsen the optimum of (vehicles, distance) taken lexicographically, so a cell
    whose plan has more vehicles, or the same vehicles and a longer distance, than the best plan found at any
    less permissive level of the same instance reflects the time-limited heuristic, not the problem. The
    baseline level (factor 1.0, tolerance 0) counts as a less permissive level.

    Parameters
    ----------
    window : pandas.DataFrame
        Output of :func:`window_results_summary`.
    early : pandas.DataFrame
        Output of :func:`early_results_summary` (the representative replicate of every cell is used).
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``scenario``, ``instance``, ``level``, ``vehicles``, ``distance``, ``best_lower`` and
        ``best_lower_distance`` (vehicles and distance of the best plan at a less permissive level) and
        ``worsening`` (``fleet`` or ``distance`` at equal fleet), one row per case.
    """
    text = get_labels(lang)
    scenarios, values = text["scenarios"], text["values"]
    rows = []
    for inst, level, veh, dist, bveh, bdist, kind in _lexicographic_worsening(
        window, "factor", "vehicles", "distance", lambda r: f"{r.factor:g}×"
    ):
        rows.append((scenarios["window"], inst, level, veh, round(dist, 1), bveh, round(bdist, 1),
                     values[f"worse_{kind}"]))
    for inst, level, veh, dist, bveh, bdist, kind in _lexicographic_worsening(
        early, "tolerance", "vehicles_rep", "distance_rep", lambda r: tolerance_text(r.tolerance_label, lang)
    ):
        rows.append((scenarios["early"], inst, level, veh, round(dist, 1), bveh, round(bdist, 1),
                     values[f"worse_{kind}"]))
    return pd.DataFrame(
        rows,
        columns=["scenario", "instance", "level", "vehicles", "distance", "best_lower", "best_lower_distance",
                 "worsening"],
    )
