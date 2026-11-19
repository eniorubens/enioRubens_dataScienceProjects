"""Display tables of the protocol-v2 diagnostics (phase B; development instances only).

The experiments themselves are run by ``scripts/run_v2_solver_grid.py`` and ``scripts/run_v2_layerb_warm.py``;
this module only turns their frozen result files into tables for the notebooks. The comparison rule is the
pre-registered one of ``reports/protocolo-v2.md`` (:mod:`delivery_routing_vrptw.v2`).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .labels import get_labels
from .v2 import block_filter, compare_layerb, instance_check, representative_table

LEVEL_COLUMN = {"B1": "scale", "B2": "k", "B3": "fleet_strategy"}
DISTANCE_TOL = 1e-6


def _level_text(column: str, level, lang: str) -> str:
    """Return the display text of one level (``scale 10000``, ``k = 10``, ``reduce_fleet_v2``)."""
    text = get_labels(lang)["v2"]
    if column == "scale":
        return text["level_scale"].format(level=int(level))
    if column == "k":
        return text["level_k"].format(level=f"{float(level):g}")
    return str(level)


def compared_rows(block: str, blocks: Mapping[str, pd.DataFrame], scale: int = 10000, k: float = 10.0):
    """Return the rows that a block of protocol v2 compared, and the column that defines its levels.

    B1 compares the two scales (``fixed_cost``, k = 10); B2 the k levels at the scale chosen in B1 (B1's
    k = 10 rows included); B3 ``fixed_cost`` with ``reduce_fleet_v2`` at the scale and k chosen before.

    Parameters
    ----------
    block : {"B1", "B2", "B3"}
        Block whose comparison is rebuilt.
    blocks : mapping
        Block name (``"B1"`` ... ``"B4"``) -> rows from :func:`delivery_routing_vrptw.results.load_v2_block`.
    scale : int
        Scale chosen in B1 (used by B2 and B3).
    k : float
        Relative fixed cost chosen in B2 (used by B3).

    Returns
    -------
    tuple
        ``(rows, level_column)``.
    """
    order = {"B1": ["B1"], "B2": ["B1", "B2"], "B3": ["B1", "B2", "B3"]}[block]
    rows = pd.concat([blocks[name] for name in order], ignore_index=True)
    sel, level_col, _preference, _strict = block_filter(rows, block, scale, k)
    return sel, level_col


def ranking_display(ranking: pd.DataFrame, level_col: str, lang: str = "en") -> pd.DataFrame:
    """Format the ranking of one block (best first) with a position column and rounded statistics.

    Parameters
    ----------
    ranking : pandas.DataFrame
        ``ranking.csv`` of the block (one row per level).
    level_col : str
        ``"scale"``, ``"k"`` or ``"fleet_strategy"``.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Label-key columns ``position``, the level column, ``mean_rank``, ``mean_delta_vehicles``,
        ``n_fleet_equal_ref``, ``median_gap_equal_fleet`` and ``mean_runtime``.
    """
    out = ranking.copy()
    out[level_col] = [_level_text(level_col, v, lang) for v in out[level_col]]
    out.insert(0, "position", np.arange(1, len(out) + 1))
    for col, digits in (("mean_rank", 3), ("mean_delta_vehicles", 3), ("median_gap_equal_fleet", 3),
                        ("mean_runtime", 1)):
        out[col] = out[col].round(digits)
    return out[["position", level_col, "mean_rank", "mean_delta_vehicles", "n_fleet_equal_ref",
                "median_gap_equal_fleet", "mean_runtime"]]


def block_coverage(blocks: Mapping[str, pd.DataFrame], lang: str = "en") -> pd.DataFrame:
    """Audit the frozen solver blocks: cells, instances, feasibility, processes and mean runtime.

    Parameters
    ----------
    blocks : mapping
        Block name (``"B1"`` ... ``"B4"``) -> rows from :func:`delivery_routing_vrptw.results.load_v2_block`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per block: ``block`` (text), ``n_cells``, ``n_instances``, ``n_feasible`` (cells feasible in
        the double-precision verifier), ``workers`` and ``mean_runtime`` (seconds).
    """
    text = get_labels(lang)["v2"]
    rows = []
    for name, df in blocks.items():
        rows.append({
            "block": text[f"block_{name}"], "n_cells": len(df), "n_instances": int(df["instance"].nunique()),
            "n_feasible": int(df["feasible"].astype(str).str.lower().eq("true").sum()),
            "workers": int(df["workers"].iloc[0]), "mean_runtime": round(float(df["runtime_s"].mean()), 1),
        })
    return pd.DataFrame(rows)


def decision_table(decisions: Mapping[str, dict], lang: str = "en") -> pd.DataFrame:
    """Summarise the pre-registered decision of each block (winner and the rule step that decided it).

    Parameters
    ----------
    decisions : mapping
        Block (``"B1"``, ``"B2"``, ``"B3"``) -> record from :func:`delivery_routing_vrptw.results.load_v2_decision`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per block: ``block`` (text), ``levels_ranked`` (levels with their mean ranks, best first),
        ``winner`` and ``decided_by`` (text of the rule step that decided).
    """
    text = get_labels(lang)["v2"]
    rows = []
    for block, rec in decisions.items():
        col = rec["level_column"]
        ranked = " · ".join(
            f"{_level_text(col, r[col], lang)}: {r['mean_rank']:.2f}" for r in rec["ranking"]
        )
        rows.append({
            "block": text[f"block_{block}"], "levels_ranked": ranked,
            "winner": _level_text(col, rec["winner"], lang), "decided_by": text[f"decided_{rec['decided_by']}"],
        })
    return pd.DataFrame(rows)


def level_by_instance(rows: pd.DataFrame, level_col: str, lang: str = "en") -> pd.DataFrame:
    """Tabulate, per instance, the representative plan of every level next to the SINTEF fleet.

    The representative plan is the replicate at position ``ceil(R / 2)`` of the lexicographic order
    (infeasible, vehicles, distance), as in the evaluation. Distances of plans with different fleets are not
    comparable and are shown only as values, never as a difference.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`compared_rows`.
    level_col : str
        Column that defines the levels.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        ``instance``, ``ref_vehicles`` (SINTEF), then ``vehicles`` and ``distance`` for each level (column
        names already localised).
    """
    names = get_labels(lang)["columns"]
    reps = representative_table(rows, level_col)
    out = reps.groupby("instance", sort=True)["ref_vehicles"].first().to_frame().reset_index()
    for level in sorted(reps[level_col].unique(), key=lambda v: (str(type(v)), v)):
        part = reps[reps[level_col] == level].set_index("instance")
        tag = _level_text(level_col, level, lang)
        out[f"{names['vehicles']} ({tag})"] = out["instance"].map(part["vehicles"]).astype(int)
        out[f"{names['distance']} ({tag})"] = out["instance"].map(part["distance"]).round(1)
    return out


def level_comparison(rows: pd.DataFrame, level_col: str, base, lang: str = "en") -> pd.DataFrame:
    """Compare every level with a base level, instance by instance, on the representative plans.

    Fleet differences are counted separately from distance differences; the distance change is computed only
    on the instances where the fleet is equal (otherwise the two distances are not comparable).

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`compared_rows`.
    level_col : str
        Column that defines the levels.
    base
        Level taken as the base (for example scale 100, k = 10 or ``fixed_cost``).
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per non-base level: ``level``, ``n_instances``, ``n_fewer`` / ``n_more`` (instances with fewer /
        more vehicles than the base), ``n_equal`` (same fleet), ``distance_change_mean`` and
        ``distance_change_median`` (percent, on the equal-fleet instances, ``NaN`` if none).
    """
    reps = representative_table(rows, level_col)
    ref = reps[reps[level_col] == base].set_index("instance")
    out = []
    for level in sorted(reps[level_col].unique(), key=lambda v: (str(type(v)), v)):
        if level == base:
            continue
        cur = reps[reps[level_col] == level].set_index("instance").loc[ref.index]
        diff = cur["vehicles"] - ref["vehicles"]
        same = diff == 0
        change = (100.0 * (cur["distance"] - ref["distance"]) / ref["distance"]).where(same)
        out.append({
            "level": f"{_level_text(level_col, level, lang)} / {_level_text(level_col, base, lang)}",
            "n_instances": len(ref), "n_fewer": int((diff < 0).sum()), "n_more": int((diff > 0).sum()),
            "n_equal": int(same.sum()),
            "distance_change_mean": round(float(change.mean()), 2) if same.any() else float("nan"),
            "distance_change_median": round(float(change.median()), 2) if same.any() else float("nan"),
        })
    return pd.DataFrame(out)


def pairwise_instances(rows: pd.DataFrame, level_col: str, base, other, lang: str = "en") -> pd.DataFrame:
    """List the instances where two levels differ in fleet or, with equal fleets, in distance by 1% or more.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`compared_rows`.
    level_col : str
        Column that defines the levels.
    base, other
        The two levels (the change is ``other`` relative to ``base``).
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        ``instance``, ``ref_vehicles``, ``vehicles_base``, ``vehicles_other``, ``distance_base``,
        ``distance_other`` and ``distance_change_pct`` (equal fleets only, else ``NaN``), sorted by instance.
    """
    reps = representative_table(rows, level_col)
    a = reps[reps[level_col] == base].set_index("instance")
    b = reps[reps[level_col] == other].set_index("instance").loc[a.index]
    change = (100.0 * (b["distance"] - a["distance"]) / a["distance"]).where(b["vehicles"] == a["vehicles"])
    out = pd.DataFrame({
        "instance": a.index, "ref_vehicles": a["ref_vehicles"].to_numpy(), "vehicles_base": a["vehicles"].astype(int).to_numpy(),
        "vehicles_other": b["vehicles"].astype(int).to_numpy(), "distance_base": a["distance"].round(1).to_numpy(),
        "distance_other": b["distance"].round(1).to_numpy(), "distance_change_pct": change.round(1).to_numpy(),
    })
    keep = (out["vehicles_base"] != out["vehicles_other"]) | (out["distance_change_pct"].abs() >= 1.0)
    return out[keep].reset_index(drop=True)


def replicates_table(rows: pd.DataFrame, instance: str, level_col: str, lang: str = "en") -> pd.DataFrame:
    """Show the vehicles of every replicate of one instance, level by level (the R110 check of B1).

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of :func:`compared_rows`.
    instance : str
        Instance name.
    level_col : str
        Column that defines the levels.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per level: ``level``, ``vehicles_replicates`` (text, one number per replicate),
        ``vehicles`` (representative) and ``ref_vehicles``.

    Raises
    ------
    ValueError
        If the instance is not in ``rows``.
    """
    check = instance_check(rows, instance, level_col)
    if check is None:
        raise ValueError(f"instance {instance!r} not found")
    out = []
    for level, rec in check.items():
        shown = level if level_col == "fleet_strategy" else float(level)
        out.append({
            "level": _level_text(level_col, shown, lang),
            "vehicles_replicates": " · ".join("-" if v is None else str(v) for v in rec["vehicles_by_replicate"]),
            "vehicles": rec["representative_vehicles"], "ref_vehicles": rec["ref_vehicles"],
        })
    return pd.DataFrame(out)


def workers_by_instance(rows_4: pd.DataFrame, rows_8: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Compare the same configuration run with 4 and with 8 processes, replicate by replicate (B4).

    Parameters
    ----------
    rows_4, rows_8 : pandas.DataFrame
        Rows with 4 and 8 processes (same instances and replicate ids); ``rows_8`` may hold other levels,
        which are ignored through the key columns ``scale``, ``k`` and ``fleet_strategy`` of ``rows_4``.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        Per instance: ``n_cells`` compared, ``n_fleet_differs``, ``n_distance_differs`` (absolute difference
        above 1e-6) and the representative distances with 8 and with 4 processes and their change (percent;
        ``NaN`` if the fleets differ).
    """
    first = rows_4.iloc[0]
    same = rows_8[(rows_8["scale"] == first["scale"]) & (rows_8["k"] == first["k"])
                  & (rows_8["fleet_strategy"] == first["fleet_strategy"])]
    merged = rows_4.merge(same, on=["instance", "replicate"], suffixes=("_4", "_8"))
    rep4 = representative_table(rows_4, "fleet_strategy").set_index("instance")
    rep8 = representative_table(same, "fleet_strategy").set_index("instance")
    out = []
    for name, g in merged.groupby("instance", sort=True):
        d4, d8 = float(rep4.loc[name, "distance"]), float(rep8.loc[name, "distance"])
        eq = rep4.loc[name, "vehicles"] == rep8.loc[name, "vehicles"]
        out.append({
            "instance": name, "n_cells": len(g), "n_fleet_differs": int((g["vehicles_4"] != g["vehicles_8"]).sum()),
            "n_distance_differs": int(((g["distance_4"] - g["distance_8"]).abs() > DISTANCE_TOL).sum()),
            "distance_8": round(d8, 1), "distance_4": round(d4, 1),
            "distance_change_pct": round(100.0 * (d4 - d8) / d8, 2) if eq else float("nan"),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- B5: warm start between levels
def layerb_levels(
    window_v1: pd.DataFrame, window_v2: pd.DataFrame, early_v1: pd.DataFrame, early_v2: pd.DataFrame,
    lang: str = "en", include_base: bool = False,
) -> pd.DataFrame:
    """Tabulate the vehicles saved per level, cold start (v1) against warm start (v2), both scenarios.

    Parameters
    ----------
    window_v1, window_v2 : pandas.DataFrame
        Window-widening rows of the frozen v1 run and of the warm-started B5 run.
    early_v1, early_v2 : pandas.DataFrame
        Early-delivery rows of the frozen v1 run and of the warm-started B5 run.
    lang : str
        Label language.
    include_base : bool
        Also return the baseline level of each scenario (window factor 1.0, tolerance 0), whose vehicles saved
        are zero by definition; its mean vehicles show whether the two versions start from the same baseline.

    Returns
    -------
    pandas.DataFrame
        One row per (scenario, level above the baseline level; the baseline too if ``include_base``):
        ``scenario``, ``level``, and for each version the mean vehicles and the mean and total vehicles saved
        against the baseline level of the same version.
    """
    from .business import tolerance_text

    text = get_labels(lang)["scenarios"]
    parts = []
    for key, scenario, v1, v2 in (("window", "windows", window_v1, window_v2), ("early", "early", early_v1, early_v2)):
        levels = compare_layerb(scenario, v1, v2)["levels"]
        base = "1.0" if scenario == "windows" else "0"
        if not include_base:
            levels = levels[levels["level"].astype(str) != base]
        level_text = [
            f"{float(v):g}×" if scenario == "windows" else tolerance_text(str(v), lang) for v in levels["level"]
        ]
        parts.append(pd.DataFrame({
            "scenario": text[key], "level": level_text,
            "v1_vehicles_mean": levels["v1_vehicles_mean"].round(2).to_numpy(),
            "v1_vehicles_saved_mean": levels["v1_vehicles_saved_mean"].round(2).to_numpy(),
            "v1_vehicles_saved_total": levels["v1_vehicles_saved_total"].astype(int).to_numpy(),
            "v2_vehicles_mean": levels["v2_vehicles_mean"].round(2).to_numpy(),
            "v2_vehicles_saved_mean": levels["v2_vehicles_saved_mean"].round(2).to_numpy(),
            "v2_vehicles_saved_total": levels["v2_vehicles_saved_total"].astype(int).to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def layerb_nonmonotonic(
    window_v1: pd.DataFrame, window_v2: pd.DataFrame, early_v1: pd.DataFrame, early_v2: pd.DataFrame,
    lang: str = "en",
) -> pd.DataFrame:
    """Count the lexicographically non-monotonic cells of each scenario and version.

    A cell is non-monotonic when its plan is worse (more vehicles, or the same vehicles and a longer distance)
    than the best plan of a less permissive level of the same instance; relaxing the windows cannot worsen the
    optimum, so such cells reflect the time-limited search (see :func:`delivery_routing_vrptw.business.nonmonotonic_cases`).

    Parameters
    ----------
    window_v1, window_v2, early_v1, early_v2 : pandas.DataFrame
        See :func:`layerb_levels`.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per (scenario, version): ``scenario``, ``version`` (text), ``n_nonmonotonic``, ``n_fleet`` and
        ``n_distance``.
    """
    text = get_labels(lang)
    rows = []
    for key, scenario, v1, v2 in (("window", "windows", window_v1, window_v2), ("early", "early", early_v1, early_v2)):
        counts = compare_layerb(scenario, v1, v2)["nonmonotonic"].set_index("version")
        for version in ("v1", "v2"):
            rows.append({
                "scenario": text["scenarios"][key], "version": text["v2"][f"version_{version}"],
                **{c: int(counts.loc[version, c]) for c in ("n_nonmonotonic", "n_fleet", "n_distance")},
            })
    return pd.DataFrame(rows)


def layerb_warm_counts(window_v2: pd.DataFrame, early_v2: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Audit the warm-started run: cells, cells that started from the previous level and feasibility.

    Parameters
    ----------
    window_v2, early_v2 : pandas.DataFrame
        Rows of the warm-started B5 runs.
    lang : str
        Label language.

    Returns
    -------
    pandas.DataFrame
        One row per scenario: ``scenario``, ``n_cells``, ``n_instances``, ``n_warm`` (cells whose
        ``warm_start`` is not ``none``), ``n_cold`` (first level of each ladder) and ``n_feasible``.
    """
    text = get_labels(lang)["scenarios"]
    rows = []
    for key, df in (("window", window_v2), ("early", early_v2)):
        warm = df["warm_start"].astype(str) != "none"
        rows.append({
            "scenario": text[key], "n_cells": len(df), "n_instances": int(df["instance"].nunique()),
            "n_warm": int(warm.sum()), "n_cold": int((~warm).sum()),
            "n_feasible": int(df["feasible"].astype(str).str.lower().eq("true").sum()),
        })
    return pd.DataFrame(rows)
