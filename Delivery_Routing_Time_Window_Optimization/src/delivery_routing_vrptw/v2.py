"""Protocol v2 helpers: grid cells, development-only gating and the pre-registered comparison rules.

Everything here follows ``reports/protocolo-v2.md``. The scripts ``run_v2_solver_grid.py``,
``select_v2.py`` and ``run_v2_layerb_warm.py`` are thin command-line wrappers around these functions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .benchmark import rank_configurations
from .business import (
    early_by_tolerance,
    early_results_summary,
    nonmonotonic_cases,
    window_by_factor,
    window_distance_by_fleet,
    window_results_summary,
)
from .config import V2Section
from .evaluation import median_position, sort_replicates

BLOCKS = ("B1", "B2", "B3", "B4")
FLEET_STRATEGIES_V2 = ("fixed_cost", "reduce_fleet_v2")
CELL_KEY = ["instance", "scale", "k", "fleet_strategy", "replicate"]
TOL = 1e-9


# --------------------------------------------------------------------------- gating and cells
def select_dev_instances(
    split: Mapping[str, str], names: list[str] | None = None, instance_set: str = "dev"
) -> list[str]:
    """Return the development instances to run; the evaluation set can never be selected in this phase.

    Parameters
    ----------
    split : mapping
        Instance name -> ``"dev"`` or ``"eval"``.
    names : list of str, optional
        Restrict to these instances; every one must be a development instance.
    instance_set : str
        Must be ``"dev"``; there is deliberately no override flag for ``"eval"`` or ``"all"``.

    Returns
    -------
    list of str
        Sorted development instance names.

    Raises
    ------
    PermissionError
        If ``instance_set`` is not ``"dev"`` or a requested instance is not a development instance.
    ValueError
        If nothing is selected.
    """
    if instance_set != "dev":
        raise PermissionError(
            f"refused: instance_set={instance_set!r}. Protocol v2 uses only the development instances and "
            "has no override for the evaluation set."
        )
    dev = sorted(n for n, s in split.items() if s == "dev")
    if names:
        names = [n.strip().upper() for n in names if n.strip()]
        bad = [n for n in names if split.get(n) != "dev"]
        if bad:
            raise PermissionError(f"refused: not development instances: {bad}")
        dev = sorted(set(names))
    if not dev:
        raise ValueError("no instances selected")
    return dev


def k_label(k: float) -> str:
    """Return the canonical text of a fixed-cost multiplier (``10``, ``0.5``)."""
    return f"{float(k):g}"


def block_workers(block: str, v2: V2Section) -> int:
    """Return the number of worker processes of a block (B4 uses ``b4_workers``)."""
    return v2.b4_workers if block == "B4" else v2.workers


def block_levels(
    block: str, v2: V2Section, scale: int | None, k: float | None, fleet_strategy: str | None
) -> list[tuple[int, float, str]]:
    """Return the ``(scale, k, fleet_strategy)`` levels run by a block.

    Parameters
    ----------
    block : {"B1", "B2", "B3", "B4"}
        Protocol block.
    v2 : V2Section
        The ``[v2]`` configuration.
    scale, k, fleet_strategy
        Choices carried over from earlier blocks (``scale`` for B2-B4, ``k`` for B3-B4, strategy for B4).

    Returns
    -------
    list of tuple
        B1: every scale at ``k_default`` with ``fixed_cost``. B2: every ``k`` except ``k_default`` at the
        given scale with ``fixed_cost``. B3: ``reduce_fleet_v2`` at the given scale and k. B4: the given
        configuration.

    Raises
    ------
    ValueError
        If a required choice is missing or the block is unknown.
    """
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")
    if block == "B1":
        return [(int(s), float(v2.k_default), "fixed_cost") for s in v2.scales]
    if scale is None:
        raise ValueError(f"{block} requires --scale")
    if scale not in v2.scales:
        raise ValueError(f"scale must be one of {v2.scales}")
    if block == "B2":
        return [(int(scale), float(x), "fixed_cost") for x in v2.k_levels if x != v2.k_default]
    if k is None:
        raise ValueError(f"{block} requires --k")
    if k not in v2.k_levels:
        raise ValueError(f"k must be one of {v2.k_levels}")
    if block == "B3":
        return [(int(scale), float(k), "reduce_fleet_v2")]
    if fleet_strategy not in FLEET_STRATEGIES_V2:
        raise ValueError(f"B4 requires --fleet-strategy in {FLEET_STRATEGIES_V2}")
    return [(int(scale), float(k), fleet_strategy)]


def block_cells(
    block: str,
    names: list[str],
    v2: V2Section,
    scale: int | None = None,
    k: float | None = None,
    fleet_strategy: str | None = None,
    replicates: int | None = None,
) -> list[dict]:
    """Return the cells (one solver run each) of a block, instances x levels x replicates.

    Returns
    -------
    list of dict
        Keys ``instance``, ``scale``, ``k``, ``fleet_strategy``, ``replicate`` (``0..R-1``) and ``block``.
    """
    reps = v2.replicates if replicates is None else replicates
    levels = block_levels(block, v2, scale, k, fleet_strategy)
    return [
        {"block": block, "instance": n, "scale": s, "k": kk, "fleet_strategy": f, "replicate": r}
        for n in names
        for s, kk, f in levels
        for r in range(reps)
    ]


def cell_key(cell: Mapping) -> tuple:
    """Return the resume key ``(instance, scale, k, fleet_strategy, replicate)`` of a cell or result row."""
    return (
        str(cell["instance"]), int(cell["scale"]), k_label(cell["k"]), str(cell["fleet_strategy"]),
        int(cell["replicate"]),
    )


def run_metadata(cfg_dict: dict, config_file: str, protocol: str, names: list[str], **extra) -> dict:
    """Return the metadata written next to the results: effective config, protocol and versions.

    Parameters
    ----------
    cfg_dict : dict
        ``ExperimentConfig.to_dict()`` after the command-line overrides.
    config_file : str
        TOML file in use.
    protocol : str
        Path of the pre-registered protocol.
    names : list of str
        Instances of the run.
    **extra
        Further keys (block, arguments, ...).

    Returns
    -------
    dict
        JSON-serialisable metadata including Python, numpy and OR-Tools versions and ``sys.argv``.
    """
    import os
    import platform
    import sys
    from datetime import datetime

    import ortools

    return {
        "timestamp": datetime.now().astimezone().isoformat(), "protocol": protocol, "config": cfg_dict,
        "config_file": config_file, "instances": names, "ortools": ortools.__version__,
        "python": platform.python_version(), "numpy": np.__version__, "cpu_count": os.cpu_count(),
        "argv": sys.argv, **extra,
    }


# --------------------------------------------------------------------------- selection rule
def _clean(rows: pd.DataFrame) -> pd.DataFrame:
    d = rows.copy()
    d["feasible"] = d["feasible"].astype(str).str.lower().eq("true")
    d["distance_comparable"] = d["distance_comparable"].astype(str).str.lower().eq("true")
    for col in ("vehicles", "distance", "delta_vehicles", "gap_distance_pct", "runtime_s"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["replicate"] = pd.to_numeric(d["replicate"]).astype(int)
    return d


def representative_table(rows: pd.DataFrame, level_col: str) -> pd.DataFrame:
    """Pick the representative replicate of every (instance, level): position 2 of 3, lexicographic.

    Replicates are ordered by (infeasible, vehicles, distance), ties by replicate id, and the one at
    ``median_position`` (``ceil(R / 2)``) is taken, exactly as in the v1 evaluation.

    Parameters
    ----------
    rows : pandas.DataFrame
        Result rows with ``instance``, ``replicate``, ``feasible``, ``vehicles``, ``distance``,
        ``delta_vehicles``, ``gap_distance_pct``, ``distance_comparable``, ``runtime_s`` and ``level_col``.
    level_col : str
        Column that defines the compared levels (``scale``, ``k`` or ``fleet_strategy``).

    Returns
    -------
    pandas.DataFrame
        One real plan per (instance, level), with the same columns as ``rows``.

    Raises
    ------
    ValueError
        If the cells do not all have the same number of replicates.
    """
    d = _clean(rows)
    counts = d.groupby(["instance", level_col]).size()
    if counts.nunique() != 1:
        raise ValueError(f"unequal number of replicates per cell: {counts.to_dict()}")
    picks = [
        sort_replicates(g).iloc[median_position(len(g))]
        for _, g in d.groupby(["instance", level_col], sort=True)
    ]
    out = pd.DataFrame(picks).reset_index(drop=True)
    out[level_col] = out[level_col].astype(d[level_col].dtype)
    return out


def _close(a: float, b: float) -> bool:
    if (a is None or (isinstance(a, float) and math.isnan(a))) and (b is None or (isinstance(b, float) and math.isnan(b))):
        return True
    return bool(np.isclose(a, b, rtol=0, atol=TOL))


def rank_levels(rows: pd.DataFrame, level_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank the levels with the tournament rule of v1 applied to the representative replicates.

    Per instance the levels are ordered lexicographically by (infeasible, vehicles, distance) with average
    ranks for exact ties; the best level has the lowest mean rank over instances; ties are broken by the
    lower mean difference in vehicles to the SINTEF reference and then by the lower median distance gap
    among the fleet-equal instances (:func:`delivery_routing_vrptw.benchmark.rank_configurations`).

    Parameters
    ----------
    rows : pandas.DataFrame
        All replicates of all levels, see :func:`representative_table`.
    level_col : str
        Column that defines the levels.

    Returns
    -------
    tuple of pandas.DataFrame
        The ranking table (best first) and the representative table it was computed from.

    Raises
    ------
    ValueError
        If the levels do not cover the same instances.
    """
    reps = representative_table(rows, level_col)
    cover = reps.groupby(level_col)["instance"].apply(lambda s: tuple(sorted(s)))
    if cover.nunique() != 1:
        raise ValueError(f"levels do not cover the same instances: {cover.to_dict()}")
    ranking = rank_configurations(reps, keys=[level_col])
    return ranking, reps


def _tie_on_all(a: pd.Series, b: pd.Series) -> bool:
    cols = ("mean_rank", "mean_delta_vehicles", "median_gap_equal_fleet")
    return all(_close(float(a[c]), float(b[c])) for c in cols)


def decide(
    ranking: pd.DataFrame, level_col: str, preference, strict: bool = False
) -> tuple[object, str]:
    """Apply a block's decision rule to a ranking table.

    Parameters
    ----------
    ranking : pandas.DataFrame
        Output of :func:`rank_levels` (best first).
    level_col : str
        Column that holds the level value.
    preference
        Level the protocol prefers on a tie (B1: scale 10000; B2: k = 10).
    strict : bool
        When true (B3) the first-ranked level wins only if its mean rank is strictly lower than the
        preferred level's; otherwise the preferred level is kept without looking at tie-breaks.

    Returns
    -------
    tuple
        ``(winner, decided_by)`` with ``decided_by`` in ``"mean_rank"``, ``"tiebreak_delta_vehicles"``,
        ``"tiebreak_median_gap"``, ``"protocol_preference"`` or ``"strictly_better_mean_rank"`` /
        ``"not_strictly_better"``.
    """
    values = ranking[level_col].tolist()
    pref_row = ranking[ranking[level_col] == preference]
    if strict:
        top = ranking.iloc[0]
        if pref_row.empty:
            raise ValueError(f"preferred level {preference!r} not in ranking")
        if values[0] != preference and top["mean_rank"] < float(pref_row.iloc[0]["mean_rank"]) - TOL:
            return values[0], "strictly_better_mean_rank"
        return preference, "not_strictly_better"
    if len(values) == 1:
        return values[0], "mean_rank"
    first, second = ranking.iloc[0], ranking.iloc[1]
    if not _close(float(first["mean_rank"]), float(second["mean_rank"])):
        return values[0], "mean_rank"
    if not _close(float(first["mean_delta_vehicles"]), float(second["mean_delta_vehicles"])):
        return values[0], "tiebreak_delta_vehicles"
    if not _close(float(first["median_gap_equal_fleet"]), float(second["median_gap_equal_fleet"])):
        return values[0], "tiebreak_median_gap"
    tied = [values[i] for i in range(len(ranking)) if _tie_on_all(first, ranking.iloc[i])]
    return (preference if preference in tied else values[0]), "protocol_preference"


def block_filter(
    rows: pd.DataFrame, block: str, scale: int | None, k: float | None
) -> tuple[pd.DataFrame, str, object, bool]:
    """Select the rows a block compares and return ``(rows, level_col, preference, strict)``.

    B1 compares the scales (preference 10000 on a tie); B2 compares the k values at ``scale`` among
    ``fixed_cost`` rows, B1's ``k = k_default`` rows included (preference k = 10); B3 compares ``fixed_cost``
    with ``reduce_fleet_v2`` at ``scale`` and ``k`` and the latter must be strictly better.
    """
    d = rows.copy()
    d["scale"] = d["scale"].astype(int)
    d["k"] = d["k"].astype(float)
    if block == "B1":
        d = d[(d["fleet_strategy"] == "fixed_cost") & (d["k"] == 10)]
        return d, "scale", 10000, False
    if scale is None:
        raise ValueError(f"{block} requires scale")
    d = d[d["scale"] == int(scale)]
    if block == "B2":
        return d[d["fleet_strategy"] == "fixed_cost"], "k", 10.0, False
    if block == "B3":
        if k is None:
            raise ValueError("B3 requires k")
        d = d[d["k"] == float(k)]
        return d, "fleet_strategy", "fixed_cost", True
    raise ValueError(f"no selection rule for block {block!r}")


def decision_record(
    block: str, rows: pd.DataFrame, scale: int | None = None, k: float | None = None
) -> tuple[dict, pd.DataFrame]:
    """Compute the decision of a block from its result rows (see :func:`block_filter`).

    Returns
    -------
    tuple
        ``(decision, ranking)``: a JSON-serialisable dictionary with the winner, how it was decided, the
        ranking and the representative plan per instance; and the ranking table.
    """
    sel, level_col, preference, strict = block_filter(rows, block, scale, k)
    if sel.empty:
        raise ValueError("no rows left after filtering")
    ranking, reps = rank_levels(sel, level_col)
    winner, how = decide(ranking, level_col, preference, strict)
    extra: dict = {}
    if block == "B1":
        extra["r110"] = instance_check(sel, "R110", level_col)
    cols = ["instance", level_col, "vehicles", "distance", "feasible", "replicate", "ref_vehicles"]
    cols = [c for c in cols if c in reps.columns]
    decision = {
        "block": block, "level_column": level_col, "scale": scale, "k": k,
        "rule": "representative replicate (position ceil(R/2), lexicographic); mean lexicographic rank; "
                "ties: mean delta vehicles, median gap; then protocol preference",
        "preference": preference, "strict": strict,
        "winner": winner.item() if hasattr(winner, "item") else winner, "decided_by": how,
        "n_instances": int(reps["instance"].nunique()),
        "ranking": json_safe(ranking.to_dict(orient="records")),
        "representatives": json_safe(reps[cols].to_dict(orient="records")),
        **extra,
    }
    return decision, ranking


def instance_check(rows: pd.DataFrame, instance: str, level_col: str) -> dict | None:
    """Report the vehicles of one instance per level (all replicates and the representative one).

    Returns ``None`` when the instance is not in ``rows``.
    """
    d = _clean(rows)
    d = d[d["instance"] == instance]
    if d.empty:
        return None
    out: dict = {}
    for level, g in d.groupby(level_col):
        rep = sort_replicates(g).iloc[median_position(len(g))]
        out[str(level)] = {
            "vehicles_by_replicate": [
                None if not f else int(v) for f, v in zip(g["feasible"], g["vehicles"], strict=True)
            ],
            "representative_vehicles": int(rep["vehicles"]) if rep["feasible"] else None,
            "ref_vehicles": int(g["ref_vehicles"].iloc[0]) if "ref_vehicles" in g else None,
        }
    return out


def json_safe(obj):
    """Convert numpy scalars and NaN into JSON-friendly values (NaN -> None), recursively."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# --------------------------------------------------------------------------- B5 (layer B)
def _empty_window() -> pd.DataFrame:
    return pd.DataFrame(columns=["instance", "factor", "vehicles", "distance"], dtype=float)


def _empty_early() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["instance", "tolerance", "tolerance_label", "vehicles_rep", "distance_rep"], dtype=float
    )


def layerb_summary(scenario: str, rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarise a layer-B run with the lexicographic tools of the frozen analysis.

    Parameters
    ----------
    scenario : {"windows", "early"}
        Which scenario the rows come from.
    rows : pandas.DataFrame
        Result rows (``tolerance_label`` as text for ``early``).

    Returns
    -------
    dict of pandas.DataFrame
        ``summary`` (per instance and level, vs the baseline level), ``fleet_split`` (distance/duration
        change by fleet class: equal, reduced, increased) and ``nonmonotonic`` (cells lexicographically worse
        than a less permissive level of the same instance). ``by_level`` counts vehicles saved per level.
    """
    if scenario == "windows":
        summary = window_results_summary(rows)
        return {
            "summary": summary,
            "by_level": window_by_factor(summary),
            "fleet_split": window_distance_by_fleet(summary),
            "nonmonotonic": nonmonotonic_cases(summary, _empty_early()),
        }
    if scenario == "early":
        summary = early_results_summary(rows)
        return {
            "summary": summary,
            "by_level": _early_level_counts(summary),
            "fleet_split": early_by_tolerance(summary),
            "nonmonotonic": nonmonotonic_cases(_empty_window(), summary),
        }
    raise ValueError("scenario must be 'windows' or 'early'")


def _early_level_counts(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[summary["tolerance"] > 0]
    rows = []
    for _, g in sub.groupby("tolerance", sort=True):
        rows.append({
            "level": str(g["tolerance_label"].iloc[0]), "n_instances": len(g),
            "vehicles_saved_mean": round(float(g["vehicles_saved"].mean()), 2),
            "n_with_saving": int((g["vehicles_saved"] > 0).sum()),
            "n_with_extra": int((g["vehicles_saved"] < 0).sum()),
            "n_equal": int((g["vehicles_saved"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def compare_layerb(scenario: str, v1_rows: pd.DataFrame, v2_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare the frozen v1 run with the warm-started v2 run on the same instances.

    Parameters
    ----------
    scenario : {"windows", "early"}
        Scenario of both runs.
    v1_rows, v2_rows : pandas.DataFrame
        Result rows; ``v1_rows`` is restricted to the instances (and levels) present in ``v2_rows``.

    Returns
    -------
    dict of pandas.DataFrame
        ``levels``: per level, mean vehicles and vehicles saved vs the baseline level, v1 and v2.
        ``nonmonotonic``: number of lexicographically non-monotonic cells (fleet / distance type), v1 and v2.
    """
    level_col = "factor" if scenario == "windows" else "tolerance_label"
    names = sorted(set(v2_rows["instance"]))
    v1 = v1_rows[v1_rows["instance"].isin(names)]
    v1 = v1[v1[level_col].astype(str).isin(set(v2_rows[level_col].astype(str)))]
    s1, s2 = layerb_summary(scenario, v1), layerb_summary(scenario, v2_rows)
    veh = "vehicles" if scenario == "windows" else "vehicles_rep"
    key = "factor" if scenario == "windows" else "tolerance_label"
    levels = []
    for lab, s in (("v1", s1), ("v2", s2)):
        g = s["summary"].groupby(key, sort=False).agg(
            n_instances=("instance", "nunique"),
            vehicles_mean=(veh, "mean"),
            vehicles_saved_mean=("vehicles_saved", "mean"),
            vehicles_saved_total=("vehicles_saved", "sum"),
        )
        levels.append(g.add_prefix(f"{lab}_"))
    merged = levels[0].join(levels[1], how="outer").reset_index().rename(columns={key: "level"})
    counts = []
    for lab, s in (("v1", s1), ("v2", s2)):
        nm = s["nonmonotonic"]
        counts.append({
            "version": lab, "n_nonmonotonic": len(nm),
            "n_fleet": _count_kind(nm, "fleet"),
            "n_distance": _count_kind(nm, "distance"),
        })
    return {"levels": merged, "nonmonotonic": pd.DataFrame(counts)}


def _count_kind(nm: pd.DataFrame, kind: str) -> int:
    from .labels import get_labels

    text = get_labels("en")["values"][f"worse_{kind}"]
    return int((nm["worsening"] == text).sum())
