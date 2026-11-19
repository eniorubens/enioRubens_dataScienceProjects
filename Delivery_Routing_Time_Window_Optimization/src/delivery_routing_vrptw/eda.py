"""Exploratory analysis of the Solomon instances: inventory, data contract, windows, fleet bounds.

Everything here depends only on instance data and (where stated) on published reference values,
never on solver output. Functions return :class:`pandas.DataFrame` objects whose column names are
label keys; use :func:`localize_columns` to translate them for display.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .distance import exact, truncated
from .io import FAMILIES, Instance, Reference, family_of, sha256_file
from .labels import get_labels
from .verify import verify

UNCONSTRAINED_SHARE = 0.9
COST_TOLERANCE = 0.05
DISTANCE_TOLERANCE = 0.01


def _instances(instances: Mapping[str, Instance] | Iterable[Instance]) -> list[Instance]:
    """Return the instances as a list, accepting a mapping or an iterable."""
    if isinstance(instances, Mapping):
        return list(instances.values())
    return list(instances)


def _family_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a frame with a ``family`` column by benchmark family order, then by row order."""
    key = df["family"].map({f: i for i, f in enumerate(FAMILIES)})
    return df.assign(_k=key).sort_values("_k", kind="mergesort").drop(columns="_k").reset_index(drop=True)


def localize_columns(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Rename label-key columns (and the ``binding`` values) to the language of ``lang``.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame whose column names are keys of the ``columns`` label group.
    lang : str
        ``"pt"`` or ``"en"`` (aliases accepted).

    Returns
    -------
    pandas.DataFrame
        A renamed copy; unknown column names are kept unchanged.
    """
    labels = get_labels(lang)
    out = df.copy()
    if "binding" in out.columns:
        out["binding"] = out["binding"].map(lambda v: labels["values"].get(v, v))
    return out.rename(columns=labels["columns"])


def family_summary(instances: Mapping[str, Instance] | Iterable[Instance]) -> pd.DataFrame:
    """Summarise the structural parameters of each family.

    Parameters
    ----------
    instances : mapping or iterable of Instance
        The instances.

    Returns
    -------
    pandas.DataFrame
        One row per family present, with ``n_instances``, ``n_customers``, ``vehicle_limit``,
        ``capacity``, ``horizon``, ``service_time`` (median over customers) and ``mean_total_demand``.
    """
    rows = []
    for inst in _instances(instances):
        rows.append({
            "family": inst.family, "n_customers": inst.n_customers, "vehicle_limit": inst.vehicle_limit,
            "capacity": inst.capacity, "horizon": inst.horizon,
            "service_time": float(np.median(inst.service[1:])), "total_demand": int(inst.demand.sum()),
        })
    df = pd.DataFrame(rows)
    out = df.groupby("family").agg(
        n_instances=("family", "size"), n_customers=("n_customers", "max"),
        vehicle_limit=("vehicle_limit", "max"), capacity=("capacity", "max"), horizon=("horizon", "max"),
        service_time=("service_time", "median"), mean_total_demand=("total_demand", "mean"),
    ).reset_index()
    out["mean_total_demand"] = out["mean_total_demand"].round(1)
    return _family_order(out)


def contract_check(instances: Mapping[str, Instance] | Iterable[Instance]) -> pd.DataFrame:
    """Expose each rule of the data contract as a boolean column.

    Parameters
    ----------
    instances : mapping or iterable of Instance
        The instances.

    Returns
    -------
    pandas.DataFrame
        One row per instance with ``ids_contiguous`` (all node arrays have the same length, so ids
        ``0..n`` index them), ``depot_zero``, ``demand_le_capacity``, ``ready_le_due``,
        ``windows_in_horizon`` and ``all_ok``.
    """
    rows = []
    for inst in _instances(instances):
        n = len(inst.x)
        row = {
            "instance": inst.name, "family": inst.family,
            "ids_contiguous": bool(n >= 2 and all(len(a) == n for a in (
                inst.y, inst.demand, inst.ready, inst.due, inst.service))),
            "depot_zero": bool(inst.demand[0] == 0 and inst.service[0] == 0),
            "demand_le_capacity": bool(np.all(inst.demand <= inst.capacity)),
            "ready_le_due": bool(np.all(inst.ready <= inst.due)),
            "windows_in_horizon": bool(np.all(inst.due <= inst.due[0])),
        }
        row["all_ok"] = all(v for k, v in row.items() if k not in ("instance", "family"))
        rows.append(row)
    return pd.DataFrame(rows)


def contract_summary(check: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Aggregate :func:`contract_check` into the number of instances passing each rule.

    Returns
    -------
    pandas.DataFrame
        Columns ``rule`` (translated), ``passed`` and ``n_instances``.
    """
    labels = get_labels(lang)["columns"]
    rules = ["ids_contiguous", "depot_zero", "demand_le_capacity", "ready_le_due", "windows_in_horizon",
             "all_ok"]
    return pd.DataFrame({
        "rule": [labels[r] for r in rules],
        "passed": [int(check[r].sum()) for r in rules],
        "n_instances": len(check),
    })


def window_stats(instances: Mapping[str, Instance] | Iterable[Instance]) -> pd.DataFrame:
    """Describe the time-window width of every instance (customers only).

    Returns
    -------
    pandas.DataFrame
        Per instance: ``mean_width``, ``median_width``, ``width_share_horizon`` (mean width over the
        horizon), ``unconstrained_share`` (share of customers whose width is at least 0.9 of the
        horizon) and ``service_share_horizon`` (mean service time over the horizon).
    """
    rows = []
    for inst in _instances(instances):
        width = (inst.due[1:] - inst.ready[1:]).astype(float)
        rows.append({
            "instance": inst.name, "family": inst.family, "mean_width": float(width.mean()),
            "median_width": float(np.median(width)),
            "width_share_horizon": float(width.mean() / inst.horizon),
            "unconstrained_share": float(np.mean(width >= UNCONSTRAINED_SHARE * inst.horizon)),
            "service_share_horizon": float(inst.service[1:].mean() / inst.horizon),
        })
    return pd.DataFrame(rows)


def window_summary(stats: pd.DataFrame) -> pd.DataFrame:
    """Average :func:`window_stats` by family and add the range of the relative width (3 decimals)."""
    cols = ["mean_width", "median_width", "width_share_horizon", "unconstrained_share",
            "service_share_horizon"]
    grouped = stats.groupby("family")
    out = grouped[cols].mean()
    out["width_share_min"] = grouped["width_share_horizon"].min()
    out["width_share_max"] = grouped["width_share_horizon"].max()
    return _family_order(out.round(3).reset_index())


def capacity_lower_bound(instance: Instance) -> int:
    """Return ``ceil(total demand / capacity)``, the fewest vehicles capacity alone allows."""
    return int(math.ceil(int(instance.demand.sum()) / instance.capacity))


def fleet_bounds(
    instances: Mapping[str, Instance] | Iterable[Instance], references: Mapping[str, Reference]
) -> pd.DataFrame:
    """Compare the capacity lower bound with the SINTEF reference fleet.

    Parameters
    ----------
    instances : mapping or iterable of Instance
        The instances.
    references : mapping of str to Reference
        SINTEF references keyed by instance name; instances without reference are skipped.

    Returns
    -------
    pandas.DataFrame
        ``capacity_lb``, ``ref_vehicles``, ``ratio`` (reference over bound) and ``binding``:
        ``"capacity"`` when the reference equals the bound, otherwise ``"time windows"``.
    """
    rows = []
    for inst in _instances(instances):
        ref = references.get(inst.name)
        if ref is None:
            continue
        lb = capacity_lower_bound(inst)
        rows.append({
            "instance": inst.name, "family": inst.family, "capacity_lb": lb,
            "ref_vehicles": ref.vehicles, "ratio": ref.vehicles / lb,
            "binding": "capacity" if ref.vehicles == lb else "time windows",
        })
    return pd.DataFrame(rows)


def fleet_summary(fleet: pd.DataFrame) -> pd.DataFrame:
    """Summarise :func:`fleet_bounds` by family (means rounded to 2 decimals, plus the reference range)."""
    out = fleet.groupby("family").agg(
        n_instances=("instance", "size"), capacity_lb=("capacity_lb", "mean"),
        ref_vehicles=("ref_vehicles", "mean"), ref_vehicles_min=("ref_vehicles", "min"),
        ref_vehicles_max=("ref_vehicles", "max"), ratio=("ratio", "mean"),
        n_capacity_binding=("binding", lambda s: int((s == "capacity").sum())),
    ).round(2).reset_index()
    return _family_order(out)


def compatibility_density(instance: Instance) -> float:
    """Return the share of ordered customer pairs ``(i, j)``, ``i != j``, that can be chained.

    Customer ``j`` can be served right after ``i`` when
    ``max(ready_i, d_0i) + service_i + d_ij <= due_j`` (exact distances).

    Returns
    -------
    float
        Value in ``[0, 1]``; ``nan`` for instances with fewer than two customers.
    """
    n = instance.n_customers
    if n < 2:
        return float("nan")
    d = exact(instance)
    leave = np.maximum(instance.ready[1:], d[0, 1:]) + instance.service[1:]
    ok = leave[:, None] + d[1:, 1:] <= instance.due[1:][None, :]
    np.fill_diagonal(ok, False)
    return float(ok.sum() / (n * (n - 1)))


def compatibility_by_instance(instances: Mapping[str, Instance] | Iterable[Instance]) -> pd.DataFrame:
    """Return :func:`compatibility_density` per instance (columns ``instance``, ``family``, ``density``)."""
    return pd.DataFrame([
        {"instance": i.name, "family": i.family, "density": compatibility_density(i)}
        for i in _instances(instances)
    ])


def compatibility_summary(density: pd.DataFrame) -> pd.DataFrame:
    """Summarise compatibility density by family (mean, min, max, rounded to 3 decimals)."""
    out = density.groupby("family")["density"].agg(
        density="mean", density_min="min", density_max="max").round(3).reset_index()
    return _family_order(out)


def synthesis_table(window: pd.DataFrame, fleet: pd.DataFrame, density: pd.DataFrame) -> pd.DataFrame:
    """Combine per-family indicators of windows, fleet bound and compatibility.

    Parameters
    ----------
    window, fleet, density : pandas.DataFrame
        Outputs of :func:`window_stats`, :func:`fleet_bounds` and :func:`compatibility_by_instance`.

    Returns
    -------
    pandas.DataFrame
        One row per family: ``width_share_horizon``, ``unconstrained_share``, ``capacity_lb``,
        ``ref_vehicles``, ``ratio`` and ``density`` (means; ``ratio`` is reference over bound).
    """
    parts = [
        window.groupby("family")[["width_share_horizon", "unconstrained_share"]].mean(),
        fleet.groupby("family")[["capacity_lb", "ref_vehicles", "ratio"]].mean(),
        density.groupby("family")[["density"]].mean(),
    ]
    out = pd.concat(parts, axis=1).round(3).reset_index()
    return _family_order(out)


def convention_comparison(
    instances: Mapping[str, Instance] | Iterable[Instance],
    sintef_refs: Mapping[str, Reference],
    cvrplib_refs: Mapping[str, tuple[list[list[int]], float]],
) -> pd.DataFrame:
    """Re-verify the CVRPLIB solutions under both distance conventions and compare with SINTEF.

    Parameters
    ----------
    instances : mapping or iterable of Instance
        The instances.
    sintef_refs : mapping of str to Reference
        SINTEF hierarchical references (vehicles, then distance, double precision).
    cvrplib_refs : mapping of str to (routes, cost)
        CVRPLIB solutions and published costs (truncated distance, 1 decimal).

    Returns
    -------
    pandas.DataFrame
        Per instance: SINTEF vehicles and distance; CVRPLIB routes and cost; feasibility and
        distance of the CVRPLIB routes under truncated and exact distance; ``cost_reproduced``
        (truncated distance within 0.05 of the published cost); ``delta_vehicles`` (CVRPLIB minus
        SINTEF); ``distance_diff_pct`` (SINTEF distance over the exact CVRPLIB distance, minus 1, in
        percent); ``equal_fleet_match`` (equal fleet and exact distance equal to the SINTEF value to
        0.01).
    """
    rows = []
    for inst in _instances(instances):
        ref, sol = sintef_refs.get(inst.name), cvrplib_refs.get(inst.name)
        if ref is None or sol is None:
            continue
        routes, cost = sol
        rep_t = verify(inst, routes, matrix=truncated(inst))
        rep_e = verify(inst, routes)
        delta = rep_e.vehicles - ref.vehicles
        rows.append({
            "instance": inst.name, "family": inst.family,
            "sintef_vehicles": ref.vehicles, "sintef_distance": ref.distance,
            "cvrplib_routes": rep_e.vehicles, "cvrplib_cost": cost,
            "feasible_truncated": rep_t.feasible, "distance_truncated": rep_t.distance,
            "cost_reproduced": bool(abs(rep_t.distance - cost) <= COST_TOLERANCE),
            "feasible_exact": rep_e.feasible, "distance_exact": rep_e.distance,
            "delta_vehicles": delta,
            "distance_diff_pct": 100.0 * (ref.distance - rep_e.distance) / rep_e.distance,
            "equal_fleet_match": bool(
                delta == 0 and abs(round(rep_e.distance, 2) - ref.distance) <= DISTANCE_TOLERANCE),
        })
    return pd.DataFrame(rows)


def _range_text(values: pd.Series, lang: str) -> str:
    """Format ``min% - max%`` with one decimal and the decimal mark of ``lang`` (``-`` if empty)."""
    if values.empty:
        return "-"
    text = f"{values.min():.1f}% - {values.max():.1f}%"
    return text.replace(".", ",") if get_labels(lang) is get_labels("pt") else text


def convention_summary(conv: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Count the headline facts of :func:`convention_comparison`.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric`` (translated) and ``value``; the row of infeasible instances lists their
        names after the count, and the last two rows give the largest vehicle difference and the range of
        the extra SINTEF distance over the instances where SINTEF uses fewer vehicles.
    """
    labels = get_labels(lang)["convention"]
    infeasible = conv.loc[~conv["feasible_exact"].astype(bool), "instance"].tolist()
    equal = conv["delta_vehicles"] == 0
    values = {
        "n_instances": str(len(conv)),
        "feasible_truncated": str(int(conv["feasible_truncated"].sum())),
        "cost_reproduced": str(int(conv["cost_reproduced"].sum())),
        "feasible_exact": str(int(conv["feasible_exact"].sum())),
        "infeasible_exact": f"{len(infeasible)}: {', '.join(infeasible)}" if infeasible else "0",
        "sintef_fewer": str(int((conv["delta_vehicles"] > 0).sum())),
        "sintef_more": str(int((conv["delta_vehicles"] < 0).sum())),
        "equal_fleet": str(int(equal.sum())),
        "equal_fleet_match": str(int(conv["equal_fleet_match"].sum())),
        "max_vehicle_diff": str(int(conv["delta_vehicles"].max())),
        "extra_distance_range": _range_text(conv.loc[conv["delta_vehicles"] > 0, "distance_diff_pct"], lang),
    }
    return pd.DataFrame({"metric": [labels[k] for k in values], "value": list(values.values())})


def loaded_counts(
    instances: Mapping[str, Instance], sintef_refs: Mapping, cvrplib_refs: Mapping, lang: str = "en"
) -> pd.DataFrame:
    """Return a two-column table with the number of loaded instances and references."""
    labels = get_labels(lang)["counts"]
    return pd.DataFrame({
        "item": [labels["instances"], labels["sintef_refs"], labels["cvrplib_refs"]],
        "count": [len(instances), len(sintef_refs), len(cvrplib_refs)],
    })


def split_summary(split: Mapping[str, str]) -> pd.DataFrame:
    """Count development and evaluation instances per family.

    Parameters
    ----------
    split : mapping
        Instance name -> ``"dev"`` or ``"eval"`` (see :func:`delivery_routing_vrptw.read_split`).

    Returns
    -------
    pandas.DataFrame
        One row per family with ``dev``, ``eval`` and ``total``, plus a final ``Total`` row.
    """
    df = pd.DataFrame({"family": [family_of(n) for n in split], "set": list(split.values())})
    table = pd.crosstab(df["family"], df["set"]).reindex(columns=["dev", "eval"], fill_value=0)
    table["total"] = table.sum(axis=1)
    table = table.reset_index()
    table = _family_order(table)
    total = {"family": "Total", "dev": int(table["dev"].sum()), "eval": int(table["eval"].sum()),
             "total": int(table["total"].sum())}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


_HASH_LINE = re.compile(r"^\s*-\s*(\S+\.(?:zip|7z))\s.*?sha256\s+([0-9a-f]{64})", re.IGNORECASE)


def manifest_hashes(path: str | Path) -> dict[str, str]:
    """Read ``file -> sha256`` pairs from the archive lines of a ``MANIFEST.md``."""
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = _HASH_LINE.match(line)
        if match:
            out[match.group(1)] = match.group(2).lower()
    return out


def provenance_table(paths: Mapping[str, Path]) -> pd.DataFrame:
    """Compare the SHA-256 of the versioned archives with the values recorded in the MANIFEST files.

    Parameters
    ----------
    paths : mapping
        Output of :func:`delivery_routing_vrptw.paths.data_paths` (keys ``sintef_zip``,
        ``sintef_manifest``, ``cvrplib_instances``, ``cvrplib_solutions``, ``cvrplib_manifest``).

    Returns
    -------
    pandas.DataFrame
        Columns ``file``, ``sha256_computed``, ``sha256_manifest`` (shortened to 16 characters for
        display) and ``match`` (full digests equal).
    """
    entries = [
        ("sintef_zip", "sintef_manifest"),
        ("cvrplib_instances", "cvrplib_manifest"),
        ("cvrplib_solutions", "cvrplib_manifest"),
    ]
    rows = []
    for key, manifest_key in entries:
        path = Path(paths[key])
        expected = manifest_hashes(paths[manifest_key]).get(path.name, "")
        computed = sha256_file(path)
        rows.append({
            "file": path.name, "sha256_computed": computed[:16] + "…",
            "sha256_manifest": (expected[:16] + "…") if expected else "—",
            "match": bool(expected) and computed == expected,
        })
    return pd.DataFrame(rows)
