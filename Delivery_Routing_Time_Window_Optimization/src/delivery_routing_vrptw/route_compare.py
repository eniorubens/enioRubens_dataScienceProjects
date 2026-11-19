"""Schedule metrics, plan-to-plan reallocation and early-delivery tables (layer B, no SINTEF)."""

from __future__ import annotations

import json
import math
from itertools import combinations

import numpy as np
import pandas as pd

from .distance import exact
from .evaluation import median_position, sort_replicates
from .io import Instance
from .verify import verify

SUMMARY_METRICS = (
    "vehicles",
    "distance",
    "total_duration",
    "total_waiting",
    "total_duration_min",
    "total_waiting_min",
    "makespan",
)
REALLOC_METRICS = ("customers_reassigned_share", "pairs_kept", "pair_agreement", "arcs_kept")


def schedule_metrics(instance: Instance, routes: list[list[int]], matrix: np.ndarray | None = None) -> dict:
    """Compute time metrics of a plan with earliest-start scheduling (the rule used by ``verify``).

    ``total_duration`` and ``total_waiting`` follow the Solomon convention: every route departs the depot
    at time 0 and a vehicle arriving before ``ready`` waits. The ``*_min`` metrics use, per route, the
    latest useful departure ``delta* = min(FTS, total route waiting)``, where the forward time slack from
    the depot is ``FTS = min_j (sum_{p<=j} W_p + due_j - t_j)`` (Savelsbergh, 1992). Departing ``delta*``
    late keeps the return time and feasibility, so the route lasts ``E - delta*`` and waits
    ``waiting - delta*``.

    Parameters
    ----------
    instance : Instance
        The instance (original or transformed).
    routes : list of list of int
        Customer ids per route, without the depot. Empty routes are ignored.
    matrix : numpy.ndarray, optional
        Travel-time matrix; defaults to :func:`delivery_routing_vrptw.distance.exact`.

    Returns
    -------
    dict
        ``vehicles``, ``distance``, ``total_duration`` (sum of return times), ``total_waiting``,
        ``total_service``, ``total_travel`` (``total_duration - waiting - service``), ``makespan``
        (latest return to the depot) and ``mean_route_duration``. Duration, waiting, distance and
        vehicles are taken from :func:`delivery_routing_vrptw.verify.verify`; these use departure at 0.
        Also ``total_duration_min``, ``total_waiting_min``, ``mean_route_duration_min`` and ``total_delay``
        (sum of the optimal departure times) with optimal departures. ``makespan`` does not depend on them.
    """
    if matrix is None:
        matrix = exact(instance)
    report = verify(instance, routes, matrix)
    returns = []
    delays = []
    service = 0.0
    for route in (r for r in routes if r):
        t, prev, cum_wait, fts = 0.0, 0, 0.0, math.inf
        for c in route:
            arrival = t + float(matrix[prev, c])
            start = max(arrival, float(instance.ready[c]))
            cum_wait += start - arrival
            fts = min(fts, cum_wait + float(instance.due[c]) - start)
            t = start + float(instance.service[c])
            service += float(instance.service[c])
            prev = c
        returns.append(t + float(matrix[prev, 0]))
        delays.append(max(0.0, min(fts, cum_wait)))
    return {
        "vehicles": report.vehicles,
        "distance": report.distance,
        "total_duration": report.duration,
        "total_waiting": report.waiting,
        "total_service": service,
        "total_travel": report.duration - report.waiting - service,
        "makespan": max(returns, default=0.0),
        "mean_route_duration": report.duration / len(returns) if returns else 0.0,
        "total_duration_min": report.duration - sum(delays),
        "total_waiting_min": report.waiting - sum(delays),
        "mean_route_duration_min": (report.duration - sum(delays)) / len(returns) if returns else 0.0,
        "total_delay": sum(delays),
    }


def optimal_departures(
    instance: Instance, routes: list[list[int]], matrix: np.ndarray | None = None
) -> list[float]:
    """Return the nominal optimal (latest useful) depot departure of every non-empty route.

    Same rule as the ``*_min`` metrics of :func:`schedule_metrics`: ``delta* = min(FTS, total waiting)`` of
    the earliest-start schedule departing at 0 (Savelsbergh, 1992), so departing ``delta*`` late keeps every
    service feasible and the return time unchanged. Infeasible routes (negative slack) give 0.

    Parameters
    ----------
    instance : Instance
        The instance.
    routes : list of list of int
        Customer ids per route, without the depot. Empty routes are skipped (and get no entry).
    matrix : numpy.ndarray, optional
        Travel-time matrix; defaults to :func:`delivery_routing_vrptw.distance.exact`.

    Returns
    -------
    list of float
        One departure time per non-empty route, in order.
    """
    if matrix is None:
        matrix = exact(instance)
    out = []
    for route in (r for r in routes if r):
        t, prev, cum_wait, fts = 0.0, 0, 0.0, math.inf
        for c in route:
            arrival = t + float(matrix[prev, c])
            start = max(arrival, float(instance.ready[c]))
            cum_wait += start - arrival
            fts = min(fts, cum_wait + float(instance.due[c]) - start)
            t = start + float(instance.service[c])
            prev = c
        out.append(max(0.0, min(fts, cum_wait)))
    return out


def _clean(routes: list[list[int]]) -> list[list[int]]:
    return [list(r) for r in routes if r]


def _arcs(routes: list[list[int]]) -> set[tuple[int, int]]:
    arcs: set[tuple[int, int]] = set()
    for route in routes:
        path = [0, *route, 0]
        arcs.update(zip(path[:-1], path[1:], strict=True))
    return arcs


def reallocation(routes_a: list[list[int]], routes_b: list[list[int]]) -> dict:
    """Compare two plans for the same customers: how much of plan A was reshuffled in plan B.

    Routes of A are matched to routes of B greedily by shared-customer count (largest first, ties by
    lowest indices); routes left without a partner match an empty route.

    Parameters
    ----------
    routes_a, routes_b : list of list of int
        Plans (customer ids per route, without the depot). Empty routes are ignored.

    Returns
    -------
    dict
        ``customers`` (count), ``vehicles_a``, ``vehicles_b``; ``customers_reassigned`` (customers outside
        their matched route) and ``customers_reassigned_share``; ``pairs_kept`` (share of customer pairs
        co-routed in A that are still co-routed in B, NaN when A has none); ``pair_agreement`` (Rand index
        over all pairs); ``arcs_kept`` (share of the directed arcs of A, depot arcs included, present in B).

    Raises
    ------
    ValueError
        If the two plans do not serve exactly the same customers, or visit one twice.
    """
    a, b = _clean(routes_a), _clean(routes_b)
    flat_a, flat_b = [c for r in a for c in r], [c for r in b for c in r]
    if len(set(flat_a)) != len(flat_a) or len(set(flat_b)) != len(flat_b) or set(flat_a) != set(flat_b):
        raise ValueError("plans must serve the same customers exactly once")
    n = len(flat_a)
    sets_a, sets_b = [set(r) for r in a], [set(r) for r in b]

    overlaps = sorted(
        ((len(sa & sb), i, j) for i, sa in enumerate(sets_a) for j, sb in enumerate(sets_b) if sa & sb),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    match: dict[int, int] = {}
    used_b: set[int] = set()
    for _, i, j in overlaps:
        if i not in match and j not in used_b:
            match[i] = j
            used_b.add(j)
    kept = sum(len(sets_a[i] & sets_b[j]) for i, j in match.items())
    reassigned = n - kept

    label_a = {c: i for i, r in enumerate(a) for c in r}
    label_b = {c: j for j, r in enumerate(b) for c in r}
    together_a = together_both = agree = total = 0
    for u, v in combinations(sorted(label_a), 2):
        ta, tb = label_a[u] == label_a[v], label_b[u] == label_b[v]
        total += 1
        agree += ta == tb
        if ta:
            together_a += 1
            together_both += tb
    arcs_a, arcs_b = _arcs(a), _arcs(b)
    return {
        "customers": n,
        "vehicles_a": len(a),
        "vehicles_b": len(b),
        "customers_reassigned": reassigned,
        "customers_reassigned_share": reassigned / n if n else 0.0,
        "pairs_kept": together_both / together_a if together_a else math.nan,
        "pair_agreement": agree / total if total else 1.0,
        "arcs_kept": len(arcs_a & arcs_b) / len(arcs_a) if arcs_a else 1.0,
    }


def _tol_value(label: str) -> float:
    return math.inf if label == "full" else float(label)


def _prepare(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df["feasible"] = df["feasible"].astype(str).str.lower().eq("true")
    df["tolerance_label"] = df["tolerance_label"].astype(str)
    df["tolerance"] = df["tolerance_label"].map(_tol_value)
    return df


def early_delivery_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise the early-delivery run per (instance, tolerance) against its own tolerance-0 baseline.

    Every (instance, tolerance) is represented by **one real plan**, the lexicographic representative
    replicate: replicates are ordered by (infeasible, vehicles, distance), ties by replicate id, and the
    one at position ``ceil(R / 2)`` is taken (the rule of :func:`delivery_routing_vrptw.evaluation.median_position`).
    Metrics are therefore never a column-wise median that no single plan attains. Minimum and maximum over
    the feasible replicates are kept to show the spread.

    Parameters
    ----------
    rows : pandas.DataFrame
        Columns ``instance``, ``tolerance_label``, ``replicate``, ``feasible`` and the metrics
        ``vehicles``, ``distance``, ``total_duration``, ``total_waiting`` (depart at 0), ``total_duration_min``,
        ``total_waiting_min`` (optimal departures), ``makespan``.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, tolerance) with ``n_replicates``, ``n_feasible``, ``rep_replicate`` (id of the
        representative replicate) and, for each metric, the value of that replicate (``<metric>_rep``) and the
        minimum and maximum over feasible replicates (``<metric>_min/_max``), plus the deltas of the
        representative plans versus tolerance 0: ``vehicles_saved`` and ``distance_pct``, ``duration_pct``,
        ``waiting_pct``, ``duration_min_pct``, ``waiting_min_pct`` (optimal departures), ``makespan_pct``
        (NaN when the baseline is missing or 0).
    """
    df = _prepare(rows)
    for m in SUMMARY_METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    keys = ["instance", "tolerance_label", "tolerance"]
    records = []
    for key, group in df.groupby(keys, sort=False):
        rep = sort_replicates(group).iloc[median_position(len(group))]
        feasible = group[group["feasible"]]
        record = dict(zip(keys, key, strict=True))
        record.update(n_replicates=len(group), n_feasible=len(feasible), rep_replicate=rep["replicate"])
        for m in SUMMARY_METRICS:
            record[f"{m}_rep"] = rep[m] if rep["feasible"] else np.nan
            record[f"{m}_min"] = feasible[m].min() if len(feasible) else np.nan
            record[f"{m}_max"] = feasible[m].max() if len(feasible) else np.nan
        records.append(record)
    out = pd.DataFrame(records)
    base = out[out["tolerance"] == 0].set_index("instance")
    for new, m in (
        ("distance_pct", "distance"),
        ("duration_pct", "total_duration"),
        ("waiting_pct", "total_waiting"),
        ("duration_min_pct", "total_duration_min"),
        ("waiting_min_pct", "total_waiting_min"),
        ("makespan_pct", "makespan"),
    ):
        ref = out["instance"].map(base[f"{m}_rep"]).replace(0, np.nan)
        out[new] = 100.0 * (out[f"{m}_rep"] - ref) / ref
    out["vehicles_saved"] = out["instance"].map(base["vehicles_rep"]) - out["vehicles_rep"]
    out["n_feasible"] = out["n_feasible"].astype(int)
    out["rep_replicate"] = out["rep_replicate"].astype(int)
    return out.sort_values(["instance", "tolerance"]).reset_index(drop=True)


def reallocation_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare plans per replicate against the baseline, plus a run-to-run noise floor.

    For every instance and tolerance ``t != 0``, replicate ``r`` is compared with the baseline
    (tolerance 0) replicate ``r``. Noise rows (``tolerance_label == "noise"``) compare baseline replicate
    ``r`` with baseline replicate ``r + 1`` (cyclic), so scenario effects can be read against the
    variability of the solver itself. Only feasible runs with routes are used.

    Parameters
    ----------
    rows : pandas.DataFrame
        Columns ``instance``, ``tolerance_label``, ``replicate``, ``feasible``, ``routes`` (JSON).

    Returns
    -------
    pandas.DataFrame
        ``instance``, ``tolerance_label``, ``replicate`` and the :func:`reallocation` fields.
    """
    df = _prepare(rows)
    df = df[df["feasible"]]
    plans = {
        (r.instance, r.tolerance_label, int(r.replicate)): json.loads(r.routes) if isinstance(r.routes, str) else r.routes
        for r in df.itertuples()
    }
    out = []
    for inst in sorted({k[0] for k in plans}):
        reps = sorted({k[2] for k in plans if k[0] == inst and k[1] == "0"})
        for lab in sorted({k[1] for k in plans if k[0] == inst and k[1] != "0"}, key=_tol_value):
            for r in reps:
                if (inst, lab, r) in plans:
                    out.append(
                        {
                            "instance": inst,
                            "tolerance_label": lab,
                            "replicate": r,
                            **reallocation(plans[(inst, "0", r)], plans[(inst, lab, r)]),
                        }
                    )
        if len(reps) > 1:
            for k, r in enumerate(reps):
                nxt = reps[(k + 1) % len(reps)]
                out.append(
                    {
                        "instance": inst,
                        "tolerance_label": "noise",
                        "replicate": r,
                        **reallocation(plans[(inst, "0", r)], plans[(inst, "0", nxt)]),
                    }
                )
    return pd.DataFrame(out)


def reallocation_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate :func:`reallocation_table` by tolerance (median over instances and replicates).

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`reallocation_table`.

    Returns
    -------
    pandas.DataFrame
        One row per tolerance label (ordered, ``noise`` last) with ``n`` comparisons and the median of
        ``customers_reassigned_share``, ``pairs_kept``, ``pair_agreement`` and ``arcs_kept``.
    """
    if table.empty:
        return pd.DataFrame(columns=["tolerance_label", "n", *REALLOC_METRICS])
    agg = table.groupby("tolerance_label").agg(
        n=("replicate", "size"), **{m: (m, "median") for m in REALLOC_METRICS}
    )
    agg = agg.reset_index()
    agg["_o"] = agg["tolerance_label"].map(lambda s: math.inf if s == "noise" else _tol_value(s))
    # "full" and "noise" both map to inf: break the tie so noise comes last
    agg["_n"] = (agg["tolerance_label"] == "noise").astype(int)
    return agg.sort_values(["_n", "_o"]).drop(columns=["_o", "_n"]).reset_index(drop=True)
