"""Comparison against SINTEF references and the development/evaluation split."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .io import FAMILIES, Reference, family_of
from .verify import Report

DEFAULT_SEED = 20261117
DEFAULT_DEV_COUNTS = {"C1": 2, "C2": 2, "R1": 3, "R2": 3, "RC1": 2, "RC2": 2}


def compare(report: Report, reference: Reference) -> dict:
    """Compare a verified solution with a SINTEF reference.

    The objective is lexicographic (vehicles, then distance), so the distance gap is only
    meaningful when ``distance_comparable`` is true (same fleet size as the reference).

    Parameters
    ----------
    report : Report
        Output of :func:`delivery_routing_vrptw.verify.verify`.
    reference : Reference
        Best known solution of the same instance.

    Returns
    -------
    dict
        Keys: instance, feasible, vehicles, distance, ref_vehicles, ref_distance, delta_vehicles,
        gap_distance_pct, distance_comparable, beats_reference. If infeasible, numeric fields are NaN
        and the flags are false.
    """
    base = {
        "instance": reference.instance,
        "ref_vehicles": reference.vehicles,
        "ref_distance": reference.distance,
    }
    if not report.feasible:
        nan = float("nan")
        return {
            **base, "feasible": False, "vehicles": nan, "distance": nan, "delta_vehicles": nan,
            "gap_distance_pct": nan, "distance_comparable": False, "beats_reference": False,
        }
    delta = report.vehicles - reference.vehicles
    gap = 100.0 * (report.distance - reference.distance) / reference.distance
    beats = delta < 0 or (delta == 0 and report.distance < reference.distance - 0.005)
    return {
        **base, "feasible": True, "vehicles": report.vehicles, "distance": round(report.distance, 2),
        "delta_vehicles": delta, "gap_distance_pct": gap, "distance_comparable": delta == 0,
        "beats_reference": bool(beats),
    }


def make_split(
    names: list[str], seed: int = DEFAULT_SEED, dev_counts: dict[str, int] | None = None
) -> dict[str, str]:
    """Draw a reproducible development/evaluation split stratified by family.

    Parameters
    ----------
    names : list of str
        Instance names.
    seed : int
        Seed of ``numpy.random.default_rng``.
    dev_counts : dict, optional
        Number of development instances per family; defaults to ``DEFAULT_DEV_COUNTS``.

    Returns
    -------
    dict
        Instance name -> ``"dev"`` or ``"eval"``.
    """
    dev_counts = DEFAULT_DEV_COUNTS if dev_counts is None else dev_counts
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for name in names:
        groups[family_of(name)].append(name.upper())
    split: dict[str, str] = {}
    for family in FAMILIES:
        members = sorted(groups[family])
        k = dev_counts.get(family, 0)
        if k > len(members):
            raise ValueError(f"family {family} has only {len(members)} instances, asked for {k}")
        chosen = set(rng.choice(members, size=k, replace=False).tolist()) if members else set()
        for name in members:
            split[name] = "dev" if name in chosen else "eval"
    return split


def write_split(split: dict[str, str], path: str | Path) -> None:
    """Write the split as CSV with columns ``instance,family,set``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "family", "set"])
        for name, which in split.items():
            writer.writerow([name, family_of(name), which])


def read_split(path: str | Path) -> dict[str, str]:
    """Read a split CSV written by :func:`write_split`."""
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["instance"]: row["set"] for row in csv.DictReader(handle)}


def rank_configurations(df: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Rank configurations by the pre-registered selection rule.

    Per instance the configurations are ordered lexicographically by (infeasible, vehicles,
    distance), with average ranks for ties. The winner has the lowest mean rank; ties are broken by
    the lower mean ``delta_vehicles`` and then by the lower median ``gap_distance_pct`` among rows
    with ``distance_comparable``.

    Parameters
    ----------
    df : pandas.DataFrame
        Tournament rows with columns ``instance``, ``feasible``, ``vehicles``, ``distance``,
        ``delta_vehicles``, ``gap_distance_pct``, ``distance_comparable``, ``runtime_s`` and the
        configuration columns ``keys``.
    keys : list of str, optional
        Configuration columns; defaults to fleet_strategy, first_solution, metaheuristic.

    Returns
    -------
    pandas.DataFrame
        One row per configuration, sorted best first, with ``mean_rank``, ``mean_delta_vehicles``,
        ``n_fleet_equal_ref``, ``median_gap_equal_fleet`` and ``mean_runtime``.
    """
    keys = keys or ["fleet_strategy", "first_solution", "metaheuristic"]
    d = df.copy()
    d["feasible"] = d["feasible"].astype(bool)
    d["distance_comparable"] = d["distance_comparable"].astype(bool)
    d["_infeasible"] = (~d["feasible"]).astype(int)
    d["_veh"] = d["vehicles"].where(d["feasible"], np.inf)
    d["_dist"] = d["distance"].where(d["feasible"], np.inf)
    # lexicographic rank: order by the three keys, then average ranks over exact ties
    parts = []
    for _, g in d.groupby("instance", sort=False):
        order = g.sort_values(["_infeasible", "_veh", "_dist"], kind="mergesort")
        tie_key = order[["_infeasible", "_veh", "_dist"]].apply(tuple, axis=1)
        pos = pd.Series(np.arange(1, len(order) + 1), index=order.index, dtype=float)
        order = order.assign(rank=pos.groupby(tie_key.values).transform("mean").values)
        parts.append(order)
    d = pd.concat(parts)
    d["_gap_eq"] = d["gap_distance_pct"].where(d["distance_comparable"])
    out = d.groupby(keys).agg(
        mean_rank=("rank", "mean"),
        mean_delta_vehicles=("delta_vehicles", "mean"),
        n_fleet_equal_ref=("distance_comparable", "sum"),
        median_gap_equal_fleet=("_gap_eq", "median"),
        mean_runtime=("runtime_s", "mean"),
        n_instances=("instance", "nunique"),
    ).reset_index()
    out["_dv"] = out["mean_delta_vehicles"].fillna(np.inf)
    out["_gap"] = out["median_gap_equal_fleet"].fillna(np.inf)
    out = out.sort_values(["mean_rank", "_dv", "_gap"], kind="mergesort").drop(columns=["_dv", "_gap"])
    return out.reset_index(drop=True)
