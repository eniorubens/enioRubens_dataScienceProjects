"""Pre-registered final evaluation: median-replicate selection and result tables.

The rules implemented here follow ``reports/protocolo-avaliacao.md``: per instance the replicates are
ordered lexicographically by (infeasible, vehicles, distance); the main result is the median replicate
(position ``ceil(R/2)``) and the best replicate is reported separately as "best of R".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .io import FAMILIES

ORTOOLS = "ortools"
BEST = "best_of_R_"


def _clean(rows: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with boolean flags as ``bool``, numbers as float and the gap NaN unless comparable."""
    d = rows.copy()
    for col in ("feasible", "distance_comparable", "beats_reference"):
        d[col] = d[col].astype(bool)
    for col in ("vehicles", "distance", "delta_vehicles", "gap_distance_pct"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["gap_distance_pct"] = d["gap_distance_pct"].where(d["distance_comparable"])
    return d


def sort_replicates(group: pd.DataFrame) -> pd.DataFrame:
    """Order replicates lexicographically by (infeasible, vehicles, distance), ties by replicate id.

    Parameters
    ----------
    group : pandas.DataFrame
        Rows of one instance with columns ``feasible``, ``vehicles``, ``distance``, ``replicate``.

    Returns
    -------
    pandas.DataFrame
        The same rows, best first. Infeasible replicates come last.
    """
    feasible = group["feasible"].astype(bool)
    g = group.assign(
        _inf=(~feasible).astype(int),
        _veh=group["vehicles"].where(feasible, np.inf),
        _dist=group["distance"].where(feasible, np.inf),
    )
    g = g.sort_values(["_inf", "_veh", "_dist", "replicate"], kind="mergesort")
    return g.drop(columns=["_inf", "_veh", "_dist"])


def median_position(n: int) -> int:
    """Return the 0-based index of the median replicate, ``ceil(n / 2) - 1`` (3rd of 5, 2nd of 4)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return math.ceil(n / 2) - 1


def _pick(row: pd.Series, prefix: str) -> dict:
    return {
        f"{prefix}vehicles": row["vehicles"],
        f"{prefix}distance": row["distance"],
        f"{prefix}delta_vehicles": row["delta_vehicles"],
        f"{prefix}gap_distance_pct": row["gap_distance_pct"],
        f"{prefix}distance_comparable": bool(row["distance_comparable"]),
        f"{prefix}beats_reference": bool(row["beats_reference"]),
    }


def instance_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Build the per-instance table of the OR-Tools replicates.

    Parameters
    ----------
    rows : pandas.DataFrame
        Result rows (all methods allowed; only ``method == "ortools"`` is used) with columns instance,
        family, method, replicate, feasible, vehicles, distance, runtime_s, ref_vehicles, ref_distance,
        delta_vehicles, gap_distance_pct, distance_comparable, beats_reference.

    Returns
    -------
    pandas.DataFrame
        One row per instance. Unprefixed metrics describe the median replicate (main result);
        ``best_of_R_*`` columns describe the best replicate; ``vehicles_min/max/range``,
        ``distance_min/max`` and ``distance_range_pct`` (over feasible replicates) describe variability.
        The distance gap is NaN unless the fleet equals the reference.
    """
    d = _clean(rows[rows["method"] == ORTOOLS])
    out = []
    for instance, g in d.groupby("instance", sort=True):
        s = sort_replicates(g)
        med = s.iloc[median_position(len(s))]
        best = s.iloc[0]
        feas = g[g["feasible"]]
        has = len(feas) > 0
        vmin, vmax = (feas["vehicles"].min(), feas["vehicles"].max()) if has else (np.nan, np.nan)
        dmin, dmax = (feas["distance"].min(), feas["distance"].max()) if has else (np.nan, np.nan)
        out.append({
            "instance": instance,
            "family": med["family"],
            "ref_vehicles": med["ref_vehicles"],
            "ref_distance": med["ref_distance"],
            "n_replicates": len(g),
            "n_feasible": len(feas),
            "median_replicate": int(med["replicate"]),
            "feasible": bool(med["feasible"]),
            **_pick(med, ""),
            "runtime_s": med["runtime_s"],
            f"{BEST}replicate": int(best["replicate"]),
            **_pick(best, BEST),
            "vehicles_min": vmin,
            "vehicles_max": vmax,
            "vehicles_range": vmax - vmin,
            "distance_min": dmin,
            "distance_max": dmax,
            "distance_range_pct": 100.0 * (dmax - dmin) / dmin if has else np.nan,
        })
    return pd.DataFrame(out).reset_index(drop=True)


def _summary(t: pd.DataFrame, prefix: str = "") -> dict:
    """Aggregate a one-row-per-instance table whose metric columns carry ``prefix``."""
    comp = t[f"{prefix}distance_comparable"].astype(bool)
    gap = t[f"{prefix}gap_distance_pct"].where(comp)
    return {
        f"{prefix}mean_delta_vehicles": t[f"{prefix}delta_vehicles"].mean(),
        f"{prefix}n_fleet_equal": int(comp.sum()),
        f"{prefix}median_gap_fleet_equal": gap.median() if comp.any() else np.nan,
        f"{prefix}n_beats_reference": int(t[f"{prefix}beats_reference"].astype(bool).sum()),
    }


def aggregate_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results overall and by family, for OR-Tools and every baseline.

    Parameters
    ----------
    rows : pandas.DataFrame
        Result rows as in :func:`instance_table`; any method other than ``"ortools"`` is a baseline
        with one row per instance.

    Returns
    -------
    pandas.DataFrame
        One row per (scope, method), scope being ``"overall"`` or a family. Columns: n_instances,
        n_feasible (instances whose reported result is feasible), mean_delta_vehicles (over feasible
        instances), n_fleet_equal, median_gap_fleet_equal (among fleet-equal instances),
        n_beats_reference. For OR-Tools these describe the median replicate and the same four metrics
        are repeated with the ``best_of_R_`` prefix; those are NaN for baselines.
    """
    inst = instance_table(rows)
    base = _clean(rows[rows["method"] != ORTOOLS])
    base = base.sort_values(["method", "instance", "replicate"]).groupby(["method", "instance"]).head(1)
    out = []
    for scope in ["overall", *FAMILIES]:
        pick = (lambda t: t) if scope == "overall" else (lambda t, s=scope: t[t["family"] == s])
        t = pick(inst)
        if len(t):
            out.append({
                "scope": scope, "method": ORTOOLS, "n_instances": len(t),
                "n_feasible": int(t["feasible"].sum()), **_summary(t), **_summary(t, BEST),
            })
        for method in sorted(base["method"].unique()):
            b = pick(base[base["method"] == method])
            if len(b):
                out.append({
                    "scope": scope, "method": method, "n_instances": len(b),
                    "n_feasible": int(b["feasible"].sum()), **_summary(b),
                })
    return pd.DataFrame(out).reset_index(drop=True)


# --------------------------------------------------------------------------- v1 x v2 (phase D, post hoc)
NOISE_PCT = 0.5
"""Distance differences below this percentage between versions are treated as within observed noise
(``reports/protocolo-fase-d.md``)."""


def v2_eval_guard(instance_set: str, confirm_eval: bool, post_hoc: bool, protocol: str) -> None:
    """Refuse the v2 re-evaluation of non-dev instances unless both explicit flags were given.

    Parameters
    ----------
    instance_set : str
        ``"dev"``, ``"eval"`` or ``"all"``.
    confirm_eval, post_hoc : bool
        ``--confirm-eval`` and ``--post-hoc``.
    protocol : str
        Path of the pre-registered protocol, cited in the message.

    Raises
    ------
    PermissionError
        If ``instance_set`` is not ``"dev"`` and either flag is missing.
    """
    if instance_set == "dev":
        return
    missing = [f for f, ok in (("--confirm-eval", confirm_eval), ("--post-hoc", post_hoc)) if not ok]
    if missing:
        raise PermissionError(
            f"refused: the v2 re-evaluation of instance_set={instance_set!r} is post hoc (the instances were "
            f"already seen in v1) and needs {' and '.join(missing)}; see {protocol}."
        )


def compare_versions(v1_rows: pd.DataFrame, v2_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the v1 and v2 OR-Tools results (median replicate), per instance and aggregated.

    Parameters
    ----------
    v1_rows, v2_rows : pandas.DataFrame
        Result rows as in :func:`instance_table` (baselines, if any, are ignored). Only instances present
        in both are compared.

    Returns
    -------
    by_instance : pandas.DataFrame
        One row per instance: ``v{1,2}_{feasible,vehicles,distance,delta_vehicles,gap_distance_pct,
        distance_comparable}``, ``delta_vehicles_v2_minus_v1``, ``fleet_equal_versions``,
        ``distance_diff_pct`` (v2 relative to v1, only when both are feasible with equal fleets, else NaN)
        and ``within_noise`` (``|distance_diff_pct| < 0.5``, False when NaN).
    aggregate : pandas.DataFrame
        One row per scope (overall, then families), v1 and v2 side by side: ``n_instances``,
        ``v{1,2}_mean_delta_vehicles`` (vs the reference), ``v{1,2}_n_fleet_equal``,
        ``v{1,2}_median_gap_fleet_equal``, plus ``mean_delta_vehicles_v2_minus_v1``,
        ``n_fleet_equal_versions`` and ``n_within_noise``.
    """
    t1, t2 = instance_table(v1_rows), instance_table(v2_rows)
    keep = ["instance", "family", "ref_vehicles", "feasible", "vehicles", "distance", "delta_vehicles",
            "gap_distance_pct", "distance_comparable"]
    a = t1[keep].add_prefix("v1_").rename(
        columns={"v1_instance": "instance", "v1_family": "family", "v1_ref_vehicles": "ref_vehicles"}
    )
    b = t2[keep].drop(columns=["family", "ref_vehicles"]).add_prefix("v2_").rename(
        columns={"v2_instance": "instance"}
    )
    d = a.merge(b, on="instance", how="inner").sort_values("instance").reset_index(drop=True)
    d["delta_vehicles_v2_minus_v1"] = d["v2_vehicles"] - d["v1_vehicles"]
    both = d["v1_feasible"].astype(bool) & d["v2_feasible"].astype(bool)
    d["fleet_equal_versions"] = both & (d["v2_vehicles"] == d["v1_vehicles"])
    pct = 100.0 * (d["v2_distance"] - d["v1_distance"]) / d["v1_distance"]
    d["distance_diff_pct"] = pct.where(d["fleet_equal_versions"])
    d["within_noise"] = d["distance_diff_pct"].abs() < NOISE_PCT

    out = []
    for scope in ["overall", *FAMILIES]:
        t = d if scope == "overall" else d[d["family"] == scope]
        if not len(t):
            continue
        row = {"scope": scope, "n_instances": len(t)}
        for v in ("v1", "v2"):
            comp = t[f"{v}_distance_comparable"].astype(bool)
            row[f"{v}_mean_delta_vehicles"] = t[f"{v}_delta_vehicles"].mean()
            row[f"{v}_n_fleet_equal"] = int(comp.sum())
            row[f"{v}_median_gap_fleet_equal"] = t[f"{v}_gap_distance_pct"].where(comp).median()
        row["mean_delta_vehicles_v2_minus_v1"] = t["delta_vehicles_v2_minus_v1"].mean()
        row["n_fleet_equal_versions"] = int(t["fleet_equal_versions"].sum())
        row["n_within_noise"] = int(t["within_noise"].sum())
        out.append(row)
    return d, pd.DataFrame(out).reset_index(drop=True)
