"""Phase C helpers: seeded draws, penalties, solver cells (C2, C3) and summaries.

Everything here follows ``reports/protocolo-fase-c.md``; ``scripts/run_phase_c_solver.py`` (C2) and
``scripts/run_reoptimisation.py`` (C3) are thin command-line wrappers. Layer B: no comparison with SINTEF,
every scenario is compared with the reference plan of the same instance (the v2 solver on the unmodified
instance).
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import ExperimentConfig, PhaseCSection
from .distance import exact
from .evaluation import median_position
from .io import Instance
from .route_compare import reallocation
from .scenarios import (
    insert_customers,
    limit_fleet,
    remove_customers,
    restrict_routes,
    scale_capacity,
)
from .solver import SolverConfig, distance_upper_bound, fixed_cost_for, solve
from .verify import measure_lateness, verify

CAPACITY, FLEET_PRIORITY, SOFT_WINDOWS = "capacity", "fleet_priority", "soft_windows"
SOLVER_BLOCKS = (CAPACITY, FLEET_PRIORITY, SOFT_WINDOWS)
REOPT_BLOCKS = ("removal", "insertion")
CELL_KEY = ["instance", "level", "variant", "replicate"]
REALLOC_FIELDS = ("customers_reassigned", "customers_reassigned_share", "pairs_kept", "pair_agreement", "arcs_kept")
FLEET_CLASSES = ("reduced", "equal", "increased")
_COMMON = ["block", "instance", "family", "level", "variant", "replicate", "time_limit_s", "scale", "fixed_cost_k",
           "vehicle_fixed_cost", "status", "warm_start", "runtime_s", "feasible", "vehicles", "distance"]
_EXTRA = {
    CAPACITY: ["capacity"],
    FLEET_PRIORITY: ["K", "base_penalty", "priority_ids", "dropped_ids", "n_customers", "n_dropped", "n_priority",
                     "n_dropped_priority", "dropped_demand"],
    SOFT_WINDOWS: ["K", "multiplier", "n_late", "total_lateness", "max_lateness", "late_share"],
    "removal": ["sample", "n_sample", "n_common", "warm_start_input", "orig_vehicles", "orig_distance",
                "delta_vehicles", "delta_distance_pct", *[f"realloc_{f}" for f in REALLOC_FIELDS]],
}
_EXTRA["insertion"] = _EXTRA["removal"]
for _b in (CAPACITY, FLEET_PRIORITY, SOFT_WINDOWS):
    _EXTRA[_b] = [*_EXTRA[_b], "capacity_infeasible"]

#: Ascending sort keys of the representative replicate, after ``not feasible`` (infeasible last); ties by
#: replicate id. Blocks not listed use ``DEFAULT_ORDER`` (the lexicographic rule of the evaluation).
DEFAULT_ORDER = ["vehicles", "distance"]
ORDERINGS = {
    FLEET_PRIORITY: ["n_dropped_priority", "n_dropped", "dropped_demand", "vehicles", "distance"],
    SOFT_WINDOWS: ["n_late", "total_lateness", "vehicles", "distance"],
}


def ordering(block: str) -> list[str]:
    """Return the ascending sort columns of the representative replicate of a block (after infeasibility)."""
    return list(ORDERINGS.get(block, DEFAULT_ORDER))


def capacity_lower_bound(instance: Instance) -> int:
    """Return ``ceil(sum(demand) / capacity)``, the fewest vehicles any plan serving every customer needs."""
    return int(math.ceil(int(instance.demand.sum()) / instance.capacity - 1e-12))


def result_columns(block: str) -> list[str]:
    """Return the fixed column order of a block's ``results.csv`` (``routes`` last).

    Rows are reindexed to it before being appended, so rows of different shapes (a skipped cell has fewer
    fields) never misalign the file.
    """
    if block not in _EXTRA:
        raise ValueError(f"unknown block {block!r}")
    return [*_COMMON, *_EXTRA[block], "routes"]


def as_frame(row: dict) -> pd.DataFrame:
    """Return a one-row frame of a cell result in the fixed column order of its block."""
    cols = result_columns(row["block"])
    extra = set(row) - set(cols)
    if extra:
        raise ValueError(f"fields missing from the schema of {row['block']}: {sorted(extra)}")
    return pd.DataFrame([row]).reindex(columns=cols)


# --------------------------------------------------------------------------- draws and penalties
def draw_customers(n_customers: int, share: float, seed: int, name: str, salt: str) -> list[int]:
    """Draw a reproducible share of the customers of an instance.

    The generator is seeded with ``(seed, crc32(name), crc32(salt))`` so every instance gets its own
    sample and different purposes (priorities, reoptimisation) get different samples.

    Parameters
    ----------
    n_customers : int
        Number of customers (ids ``1..n``).
    share : float
        Fraction to draw; the count is ``max(1, round(share * n))``.
    seed : int
        Phase seed (``20261118`` in the protocol).
    name : str
        Instance name (the unmodified one).
    salt : str
        Purpose of the draw.

    Returns
    -------
    list of int
        Sorted customer ids.
    """
    count = max(1, int(round(share * n_customers)))
    rng = np.random.default_rng([int(seed), zlib.crc32(name.encode()), zlib.crc32(salt.encode())])
    return sorted(int(c) + 1 for c in rng.choice(n_customers, size=min(count, n_customers), replace=False))


def base_drop_penalty(instance: Instance, config: SolverConfig) -> int:
    """Return the common dropping penalty: larger than any cost of serving one more customer.

    Serving a customer costs at most opening a vehicle for it plus a round trip from the depot, so
    ``P = F + scale * D_ub + 1`` with ``F = fixed_cost_for(instance, config)`` and ``D_ub = sum_i 2 d(0, i)``
    (``2 d(0, i) <= D_ub``). Dropping a customer therefore never pays unless the fleet limit prevents
    serving it, and dropping to save a vehicle (``F``) never pays either.

    Parameters
    ----------
    instance : Instance
        The instance.
    config : SolverConfig
        Solver configuration (scale and fixed cost).

    Returns
    -------
    int
        The base penalty in solver cost units.
    """
    return int(fixed_cost_for(instance, config) + config.scale * distance_upper_bound(instance) + 1)


def phase_c_solver_config(
    cfg: ExperimentConfig, time_limit_s: float | None = None, num_vehicles: int | None = None
) -> SolverConfig:
    """Return the v2 solver configuration of phase C (fixed cost + first solution + GLS, scale, k)."""
    pc = cfg.phase_c
    return SolverConfig(
        first_solution=cfg.solver.first_solution, metaheuristic=cfg.solver.metaheuristic,
        time_limit_s=pc.time_limit_s if time_limit_s is None else time_limit_s, scale=pc.scale,
        fixed_cost_k=pc.fixed_cost_k, num_vehicles=num_vehicles,
    )


def fleet_class(vehicles: float, ref_vehicles: float) -> str:
    """Return ``"reduced"``, ``"equal"`` or ``"increased"`` for a fleet against the reference fleet."""
    if vehicles < ref_vehicles:
        return "reduced"
    return "equal" if vehicles == ref_vehicles else "increased"


def level_label(value: float) -> str:
    """Return the canonical text of a numeric level (``1``, ``0.9``)."""
    return f"{float(value):g}"


# --------------------------------------------------------------------------- C2 cells
def _row(
    inst: Instance, block: str, level: str, variant: str, replicate: int, config: SolverConfig, sol
) -> dict:
    return {
        "block": block, "instance": inst.name.split("@")[0], "family": inst.family, "level": level,
        "variant": variant, "replicate": replicate, "time_limit_s": config.time_limit_s,
        "scale": config.scale, "fixed_cost_k": config.fixed_cost_k,
        "vehicle_fixed_cost": sol.meta.get("vehicle_fixed_cost_effective"), "status": sol.meta.get("status", ""),
        "warm_start": sol.meta.get("warm_start", ""), "runtime_s": sol.runtime_s,
    }


def _skipped(inst: Instance, block: str, level: str, variant: str, replicate: int, why: str) -> dict:
    return {
        "block": block, "instance": inst.name.split("@")[0], "family": inst.family, "level": level,
        "variant": variant, "replicate": replicate, "status": f"skipped: {why}", "feasible": False,
        "vehicles": np.nan, "distance": np.nan, "routes": "[]",
    }


def _found(sol) -> bool:
    return sol.meta.get("objective") is not None


def solve_capacity_cell(inst: Instance, factor: float, replicate: int, config: SolverConfig) -> dict:
    """Solve one C2.1 cell: the rigid model with the capacity multiplied by ``factor``.

    The plan is verified against the scenario instance (reduced capacity). When the reduced capacity is
    below the largest demand the cell is not solved and a ``skipped`` row is returned.

    Returns
    -------
    dict
        One result row (``level`` = factor, ``variant`` empty) with ``capacity``, ``feasible``,
        ``vehicles``, ``distance``, ``routes`` (JSON) and solver metadata.
    """
    level = level_label(factor)
    try:
        scen = scale_capacity(inst, factor)
    except ValueError as exc:
        return _skipped(inst, CAPACITY, level, "", replicate, str(exc))
    sol = solve(scen, config)
    rep = verify(scen, sol.routes, exact(scen))
    row = _row(inst, CAPACITY, level, "", replicate, config, sol)
    row.update(capacity=scen.capacity, feasible=bool(_found(sol) and rep.feasible), vehicles=rep.vehicles,
               distance=rep.distance, routes=json.dumps(sol.routes),
               capacity_infeasible=scen.vehicle_limit < capacity_lower_bound(scen))
    return row


def _verify_served(instance: Instance, routes: list[list[int]], dropped: list[int]):
    """Verify the plan against the customers it serves (the reduced instance when some are dropped)."""
    if dropped:
        sub, cmap = remove_customers(instance, dropped)
        return verify(sub, cmap.to_reduced(routes), exact(sub))
    return verify(instance, routes, exact(instance))


def solve_fleet_priority_cell(
    inst: Instance, ref_vehicles: int, reduction: int, variant: str, replicate: int, config: SolverConfig,
    pc: PhaseCSection,
) -> dict:
    """Solve one C2.2 cell: ``K = ref_vehicles - reduction`` vehicles, every customer droppable at a penalty.

    ``variant`` is ``"uniform"`` (every customer at the base penalty :func:`base_drop_penalty`) or
    ``"priority"`` (the drawn priority customers at ``priority_penalty_multiplier`` times that). The priority
    draw is the same for both variants and is recorded in both. The served plan is verified against the
    instance without the dropped customers and the fleet limit ``K``.

    Returns
    -------
    dict
        One row with ``K``, ``base_penalty``, ``priority_ids`` and ``dropped_ids`` (JSON), ``n_dropped``,
        ``n_dropped_priority``, ``dropped_demand``, ``feasible``, ``vehicles``, ``distance``, ``routes``.
    """
    if variant not in ("uniform", "priority"):
        raise ValueError("variant must be 'uniform' or 'priority'")
    level = f"K-{int(reduction)}"
    k = int(ref_vehicles) - int(reduction)
    if k < 1:
        return _skipped(inst, FLEET_PRIORITY, level, variant, replicate, f"K = {k} < 1")
    scen = limit_fleet(inst, k)
    cfg = replace(config, num_vehicles=k)
    base = base_drop_penalty(inst, cfg)
    priority = draw_customers(inst.n_customers, pc.priority_share, pc.seed, inst.name, "priority")
    penalties = {c: base * (pc.priority_penalty_multiplier if variant == "priority" else 1) for c in priority}
    sol = solve(scen, cfg, drop_penalty=base, drop_penalties=penalties)
    dropped = list(sol.meta["dropped"]) if _found(sol) else []
    rep = _verify_served(scen, sol.routes, dropped)
    row = _row(inst, FLEET_PRIORITY, level, variant, replicate, cfg, sol)
    row.update(
        K=k, base_penalty=base, priority_ids=json.dumps(priority), dropped_ids=json.dumps(dropped),
        n_customers=inst.n_customers,
        n_dropped=len(dropped), n_priority=len(priority), n_dropped_priority=len(set(dropped) & set(priority)),
        dropped_demand=int(inst.demand[dropped].sum()) if dropped else 0,
        feasible=bool(_found(sol) and rep.feasible), vehicles=rep.vehicles, distance=rep.distance,
        routes=json.dumps(sol.routes), capacity_infeasible=k < capacity_lower_bound(inst),
    )
    return row


def solve_soft_cell(
    inst: Instance, ref_vehicles: int, multiplier: int, replicate: int, config: SolverConfig, reduction: int = 1
) -> dict:
    """Solve one C2.3 cell: ``K = ref_vehicles - reduction``, soft due dates at ``multiplier`` x distance cost.

    The cost per scaled time unit of lateness is ``multiplier`` cost units, and one scaled distance unit
    costs 1, so ``multiplier`` is the lateness cost as a multiple of the cost per unit of distance (1x low,
    10x high). The depot return stays hard. Lateness is measured with
    :func:`delivery_routing_vrptw.verify.measure_lateness` against the original due dates.

    Returns
    -------
    dict
        One row with ``K``, ``multiplier``, ``n_late``, ``total_lateness``, ``max_lateness``, ``late_share``,
        ``feasible`` (hard constraints only), ``vehicles``, ``distance``, ``routes``.
    """
    level = f"c{int(multiplier)}"
    k = int(ref_vehicles) - int(reduction)
    if k < 1:
        return _skipped(inst, SOFT_WINDOWS, level, "", replicate, f"K = {k} < 1")
    scen = limit_fleet(inst, k)
    cfg = replace(config, num_vehicles=k)
    sol = solve(scen, cfg, soft_due_cost_per_unit=int(multiplier))
    rep = measure_lateness(scen, sol.routes, exact(scen))
    row = _row(inst, SOFT_WINDOWS, level, "", replicate, cfg, sol)
    row.update(
        K=k, multiplier=int(multiplier), n_late=rep.n_late, total_lateness=rep.total_lateness,
        max_lateness=rep.max_lateness, late_share=rep.n_late / inst.n_customers,
        feasible=bool(_found(sol) and rep.hard_feasible), vehicles=rep.vehicles, distance=rep.distance,
        routes=json.dumps(sol.routes), capacity_infeasible=k < capacity_lower_bound(inst),
    )
    return row


def solver_cells(
    block: str, names: list[str], pc: PhaseCSection, replicates: int, ref_vehicles: dict[str, int] | None = None
) -> list[dict]:
    """Return the cells (one solver run each) of a C2 block, instances x levels x variants x replicates.

    Parameters
    ----------
    block : {"capacity", "fleet_priority", "soft_windows"}
        C2 block.
    names : list of str
        Instances.
    pc : PhaseCSection
        Phase C configuration.
    replicates : int
        Replicates per cell.
    ref_vehicles : dict, optional
        Instance -> vehicles of the reference plan; required by ``fleet_priority`` and ``soft_windows``.

    Returns
    -------
    list of dict
        Keys ``block``, ``instance``, ``level``, ``variant``, ``replicate`` and the level parameters
        (``factor``, ``reduction`` or ``multiplier``, ``ref_vehicles``).

    Raises
    ------
    ValueError
        If the block is unknown or the reference misses some instance.
    """
    if block not in SOLVER_BLOCKS:
        raise ValueError(f"block must be one of {SOLVER_BLOCKS}, got {block!r}")
    if block != CAPACITY:
        missing = [n for n in names if not ref_vehicles or n not in ref_vehicles]
        if missing:
            raise ValueError(f"no reference plan (run the capacity block first) for: {missing}")
    cells = []
    for n in names:
        for r in range(replicates):
            if block == CAPACITY:
                cells += [{"block": block, "instance": n, "level": level_label(f), "variant": "", "replicate": r,
                           "factor": f} for f in pc.capacity_factors]
            elif block == FLEET_PRIORITY:
                cells += [{"block": block, "instance": n, "level": f"K-{int(k)}", "variant": v, "replicate": r,
                           "reduction": int(k), "ref_vehicles": int(ref_vehicles[n])}
                          for k in pc.fleet_reductions for v in ("uniform", "priority")]
            else:
                cells += [{"block": block, "instance": n, "level": f"c{int(m)}", "variant": "", "replicate": r,
                           "multiplier": int(m), "ref_vehicles": int(ref_vehicles[n])}
                          for m in pc.soft_cost_multipliers]
    return cells


def run_solver_cell(inst: Instance, cell: dict, config: SolverConfig, pc: PhaseCSection) -> dict:
    """Solve one C2 cell built by :func:`solver_cells` and return its verified result row."""
    block, rep = cell["block"], int(cell["replicate"])
    if block == CAPACITY:
        return solve_capacity_cell(inst, float(cell["factor"]), rep, config)
    if block == FLEET_PRIORITY:
        return solve_fleet_priority_cell(inst, cell["ref_vehicles"], cell["reduction"], cell["variant"], rep, config, pc)
    if block == SOFT_WINDOWS:
        return solve_soft_cell(inst, cell["ref_vehicles"], cell["multiplier"], rep, config, pc.soft_fleet_reduction)
    raise ValueError(f"block must be one of {SOLVER_BLOCKS}, got {block!r}")


# --------------------------------------------------------------------------- C3 cells
def solve_reopt_cell(
    inst: Instance, block: str, variant: str, replicate: int, config: SolverConfig, pc: PhaseCSection,
    original_routes: list[list[int]],
) -> dict:
    """Solve one C3 cell: reoptimise after removing or inserting a sample of customers.

    The sample is ``reopt_share`` of the customers drawn with the phase seed (the same for both blocks).

    * ``removal``: the instance without the sample is solved ``warm`` (start = ``original_routes`` without
      the sample, translated to the reduced ids) or ``cold``. ``original_routes`` is the reference plan of
      the full instance.
    * ``insertion``: ``plan90`` solves the instance without the sample (cold) and is the original plan; the
      full instance is then solved ``warm`` (start = ``original_routes`` = the plan90 routes in original ids,
      completed by cheapest insertion of the sample because OR-Tools rejects a start that leaves mandatory
      customers unrouted; no dropping is ever allowed) or ``cold``.

    Customers are never dropped. ``reallocation`` is measured on the common customers (all but the sample),
    in original ids, between the original plan and the reoptimised plan.

    Parameters
    ----------
    inst : Instance
        The unmodified instance.
    block : {"removal", "insertion"}
        C3 block.
    variant : {"warm", "cold", "plan90"}
        ``plan90`` exists only in ``insertion``.
    replicate : int
        Replicate id.
    config : SolverConfig
        Solver configuration.
    pc : PhaseCSection
        Phase C configuration (seed and share).
    original_routes : list of list of int
        Original plan in original ids (reference plan for ``removal``; plan90 for ``insertion`` warm).
        Ignored by ``cold`` insertion and ``plan90``.

    Returns
    -------
    dict
        One row with ``sample`` and ``routes`` (original ids, JSON), the verified metrics against the
        scenario instance, ``orig_vehicles``/``orig_distance`` of the original plan on the same customers,
        ``delta_vehicles``, ``delta_distance_pct`` and the ``realloc_*`` fields (NaN for ``plan90``).
    """
    if block not in REOPT_BLOCKS:
        raise ValueError(f"block must be one of {REOPT_BLOCKS}")
    if variant not in (("warm", "cold") if block == "removal" else ("warm", "cold", "plan90")):
        raise ValueError(f"variant {variant!r} is not valid for block {block!r}")
    sample = draw_customers(inst.n_customers, pc.reopt_share, pc.seed, inst.name, "reopt")
    common = [c for c in range(1, inst.n_customers + 1) if c not in set(sample)]
    reduced, cmap = remove_customers(inst, sample)
    out_nan = {f"realloc_{f}": np.nan for f in REALLOC_FIELDS}
    init = None
    start_note = "none"
    if block == "removal":
        scen = reduced
        orig_on_scen = cmap.to_reduced(original_routes)
        if variant == "warm":
            init = orig_on_scen
        original_plan = restrict_routes(original_routes, common)
    elif variant == "plan90":
        scen, orig_on_scen, original_plan = reduced, None, None
    else:
        scen = inst
        orig_on_scen = cmap.to_reduced(original_routes)  # plan90 in reduced ids, to measure the original
        original_plan = [list(r) for r in original_routes if r]
        if variant == "warm":
            init = insert_customers(inst, original_routes, sample, config.scale)
            start_note = "built" if init is not None else "not_built"
    sol = solve(scen, config, initial_routes=init)
    routes_scen = sol.routes
    rep = verify(scen, routes_scen, exact(scen))
    routes_orig = cmap.to_original(routes_scen) if scen is reduced else [list(r) for r in routes_scen if r]
    row = _row(inst, block, block, variant, replicate, config, sol)
    row.update(
        sample=json.dumps(sample), n_sample=len(sample), n_common=len(common),
        feasible=bool(_found(sol) and rep.feasible), vehicles=rep.vehicles, distance=rep.distance,
        routes=json.dumps(routes_orig), warm_start_input=start_note,
    )
    row.update(out_nan)
    row.update(orig_vehicles=np.nan, orig_distance=np.nan, delta_vehicles=np.nan, delta_distance_pct=np.nan)
    if variant != "plan90" and row["feasible"]:
        base_rep = verify(reduced, orig_on_scen, exact(reduced))
        row.update(orig_vehicles=base_rep.vehicles, orig_distance=base_rep.distance,
                   delta_vehicles=rep.vehicles - base_rep.vehicles,
                   delta_distance_pct=100.0 * (rep.distance - base_rep.distance) / base_rep.distance
                   if base_rep.distance else np.nan)
        realloc = reallocation(original_plan, restrict_routes(routes_orig, common))
        row.update({f"realloc_{f}": realloc[f] for f in REALLOC_FIELDS})
    return row


def reopt_cells(block: str, names: list[str], replicates: int, phase: str = "all") -> list[dict]:
    """Return the cells of a C3 block: ``warm`` and ``cold`` replicates, plus ``plan90`` for ``insertion``.

    Parameters
    ----------
    block : {"removal", "insertion"}
        C3 block.
    names : list of str
        Instances.
    replicates : int
        Replicates of every warm and cold cell (``plan90`` has one).
    phase : {"all", "plan", "reopt"}
        ``"plan"`` returns only the ``plan90`` cells (they must be solved first because the insertion warm
        start needs them), ``"reopt"`` only the warm and cold cells, ``"all"`` both.

    Returns
    -------
    list of dict
        Keys ``block``, ``instance``, ``level`` (= block), ``variant`` and ``replicate``.
    """
    if block not in REOPT_BLOCKS:
        raise ValueError(f"block must be one of {REOPT_BLOCKS}, got {block!r}")
    if phase not in ("all", "plan", "reopt"):
        raise ValueError("phase must be 'all', 'plan' or 'reopt'")
    cells = []
    for n in names:
        if block == "insertion" and phase in ("all", "plan"):
            cells.append({"block": block, "instance": n, "level": block, "variant": "plan90", "replicate": 0})
        if phase in ("all", "reopt"):
            cells += [{"block": block, "instance": n, "level": block, "variant": v, "replicate": r}
                      for v in ("warm", "cold") for r in range(replicates)]
    return cells


def reopt_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise C3 per (instance, variant): medians over the feasible replicates of the warm and cold runs."""
    part = rows[rows["variant"].isin(["warm", "cold"]) & rows["feasible"]].copy()
    cols = ["vehicles", "distance", "orig_vehicles", "orig_distance", "delta_vehicles", "delta_distance_pct",
            "runtime_s", *[f"realloc_{f}" for f in REALLOC_FIELDS]]
    for c in cols:
        part[c] = pd.to_numeric(part[c], errors="coerce")
    agg = part.groupby(["instance", "variant"]).agg(n_feasible=("replicate", "size"), **{c: (c, "median") for c in cols})
    return agg.reset_index()


# --------------------------------------------------------------------------- loading and tables
def read_results(path) -> pd.DataFrame:
    """Read a phase C result CSV with typed ``feasible``, ``level`` and ``variant`` columns."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"level": str, "variant": str})
    df["variant"] = df["variant"].fillna("")
    df["feasible"] = df["feasible"].astype(str).str.lower().eq("true")
    return df


def cell_key(row) -> tuple:
    """Return the resume key ``(instance, level, variant, replicate)`` of a cell or result row."""
    variant = row["variant"]
    variant = "" if variant is None or (isinstance(variant, float) and math.isnan(variant)) else str(variant)
    return str(row["instance"]), str(row["level"]), variant, int(row["replicate"])


def representative(rows: pd.DataFrame, keys: list[str], order: list[str] | None = None) -> pd.DataFrame:
    """Pick one real replicate per group: position ``ceil(R / 2)`` of an explicit ordering.

    Replicates are sorted ascending (best first) by (infeasible, ``*order``), ties by replicate id, and the
    one at :func:`delivery_routing_vrptw.evaluation.median_position` is taken. Infeasible and skipped cells
    sort last. The default order (vehicles, distance) is the lexicographic rule of the evaluation.

    Parameters
    ----------
    rows : pandas.DataFrame
        Result rows with ``feasible``, ``replicate``, the ``keys`` and the ``order`` columns.
    keys : list of str
        Grouping columns.
    order : list of str, optional
        Ascending sort columns after infeasibility; defaults to ``DEFAULT_ORDER``. Use :func:`ordering`
        for the block-specific ones.

    Returns
    -------
    pandas.DataFrame
        One row per group with the extra columns ``n_replicates`` and ``n_feasible``.
    """
    order = list(DEFAULT_ORDER if order is None else order)
    df = rows.copy()
    for col in {*order, "vehicles", "distance"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    picks = []
    for _, g in df.groupby(keys, sort=True, dropna=False):
        feasible = g["feasible"].astype(bool)
        srt = g.assign(_inf=(~feasible).astype(int))
        for col in order:
            srt[f"_k_{col}"] = g[col].where(feasible, np.inf).fillna(np.inf)
        srt = srt.sort_values(["_inf", *[f"_k_{c}" for c in order], "replicate"], kind="mergesort")
        pick = srt.iloc[median_position(len(g))].copy()
        pick["n_replicates"], pick["n_feasible"] = len(g), int(feasible.sum())
        picks.append(pick.drop(labels=[c for c in pick.index if c.startswith("_")]))
    return pd.DataFrame(picks).reset_index(drop=True)


def reference_table(capacity_rows: pd.DataFrame) -> pd.DataFrame:
    """Return the reference plan (capacity factor 1) of every instance: the representative replicate.

    Parameters
    ----------
    capacity_rows : pandas.DataFrame
        Rows of the C2.1 block (``level`` ``"1"`` is the reference).

    Returns
    -------
    pandas.DataFrame
        ``instance``, ``family``, ``ref_vehicles``, ``ref_distance``, ``ref_replicate`` and ``ref_routes``
        (a list of lists), one row per instance with a feasible reference.
    """
    ref = capacity_rows[capacity_rows["level"] == level_label(1.0)]
    reps = representative(ref, ["instance"])
    reps = reps[reps["feasible"]]
    return pd.DataFrame({
        "instance": reps["instance"], "family": reps["family"], "ref_vehicles": reps["vehicles"].astype(int),
        "ref_distance": reps["distance"], "ref_replicate": reps["replicate"].astype(int),
        "ref_routes": reps["routes"].map(json.loads),
    }).reset_index(drop=True)


def _with_reference(rep: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    out = rep.merge(ref[["instance", "ref_vehicles", "ref_distance"]], on="instance", how="left")
    ok = out["feasible"] & out["ref_vehicles"].notna()
    out["fleet_class"] = [
        fleet_class(v, r) if o else np.nan for v, r, o in zip(out["vehicles"], out["ref_vehicles"], ok, strict=True)
    ]
    out["distance_change_pct"] = (100.0 * (out["distance"] - out["ref_distance"]) / out["ref_distance"]).where(ok)
    return out


def capacity_summary(rows: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Summarise C2.1 per (instance, factor): representative plan against the reference plan.

    Returns
    -------
    pandas.DataFrame
        ``instance``, ``family``, ``level`` (factor), ``capacity``, ``feasible``, ``vehicles``, ``distance``,
        ``ref_vehicles``, ``ref_distance``, ``extra_vehicles`` (vehicles minus reference), ``fleet_class`` and
        ``distance_change_pct`` (NaN unless the cell is feasible).
    """
    out = _with_reference(representative(rows, ["instance", "level"], ordering(CAPACITY)), ref)
    out["representative_order"] = ", ".join(ordering(CAPACITY))
    out["extra_vehicles"] = (out["vehicles"] - out["ref_vehicles"]).where(out["feasible"])
    cols = ["instance", "family", "level", "capacity", "n_feasible", "feasible", "vehicles", "distance",
            "ref_vehicles", "ref_distance", "extra_vehicles", "fleet_class", "distance_change_pct",
            "capacity_infeasible", "representative_order"]
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def capacity_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate C2.1 by factor, with the distance change split by fleet class (never pooled).

    Returns
    -------
    pandas.DataFrame
        One row per (factor, fleet class) with ``n_instances``, ``extra_vehicles_mean`` and the mean and
        median distance change within the class; factor-level rows (``fleet_class = "all"``) carry the
        number of infeasible cells and the mean of extra vehicles but no distance change.
    """
    rows = []
    for level, g in summary.groupby("level", sort=False):
        rows.append({
            "level": level, "fleet_class": "all", "n_instances": len(g), "n_infeasible": int((~g["feasible"]).sum()),
            "extra_vehicles_mean": g["extra_vehicles"].mean(), "distance_change_mean": np.nan,
            "distance_change_median": np.nan,
        })
        for cls in FLEET_CLASSES:
            part = g[g["fleet_class"] == cls]
            if len(part):
                rows.append({
                    "level": level, "fleet_class": cls, "n_instances": len(part), "n_infeasible": 0,
                    "extra_vehicles_mean": part["extra_vehicles"].mean(),
                    "distance_change_mean": part["distance_change_pct"].mean(),
                    "distance_change_median": part["distance_change_pct"].median(),
                })
    return pd.DataFrame(rows)


def _cell_means(rows: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    df = rows.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    means = df[df["feasible"]].groupby(keys)[cols].mean().add_suffix("_mean").reset_index()
    return means


def fleet_priority_summary(rows: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Summarise C2.2 per (instance, K level, variant): representative plan and replicate means.

    Returns
    -------
    pandas.DataFrame
        Representative columns (``vehicles``, ``distance``, ``K``, ``n_dropped``, ``n_dropped_priority``,
        ``n_priority``, ``dropped_demand``), the reference comparison (``fleet_class``,
        ``distance_change_pct``; the fleet is smaller than the reference, so the distance covers fewer
        customers and is not comparable one to one) and ``*_mean`` columns over the feasible replicates.
    """
    keys = ["instance", "level", "variant"]
    rep = _with_reference(representative(rows, keys, ordering(FLEET_PRIORITY)), ref)
    rep["representative_order"] = ", ".join(ordering(FLEET_PRIORITY))
    cols = ["n_dropped", "n_dropped_priority", "dropped_demand", "distance"]
    out = rep.merge(_cell_means(rows, keys, cols), on=keys, how="left")
    keep = ["instance", "family", "level", "variant", "K", "n_replicates", "n_feasible", "feasible", "vehicles",
            "distance", "n_priority", "n_dropped", "n_dropped_priority", "dropped_demand", "n_customers",
            "ref_vehicles", "ref_distance", "fleet_class", "distance_change_pct", *[f"{c}_mean" for c in cols],
            "capacity_infeasible", "representative_order"]
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def fleet_priority_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate C2.2 per (K level, variant) and add the paired priority-minus-uniform differences.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`fleet_priority_summary`.

    Returns
    -------
    pandas.DataFrame
        One row per (level, variant) with the totals of dropped customers (all and priority), the shares of
        priority and of common customers dropped, the mean unserved demand and the mean distance change; and
        rows with ``variant = "priority - uniform"`` holding the mean paired differences per instance.
    """
    ok = summary[summary["feasible"]]
    rows = []
    for (level, variant), g in ok.groupby(["level", "variant"], sort=False):
        n_cust = g["n_customers"].to_numpy(dtype=float)
        n_pri = g["n_priority"].to_numpy(dtype=float)
        rows.append({
            "level": level, "variant": variant, "n_instances": len(g),
            "dropped_total": int(g["n_dropped"].sum()), "dropped_priority_total": int(g["n_dropped_priority"].sum()),
            "priority_dropped_share": float(g["n_dropped_priority"].sum() / n_pri.sum()),
            "common_dropped_share": float((g["n_dropped"] - g["n_dropped_priority"]).sum() / (n_cust - n_pri).sum()),
            "dropped_demand_mean": float(g["dropped_demand"].mean()),
            "distance_change_mean": float(g["distance_change_pct"].mean()),
        })
    for level, g in ok.groupby("level", sort=False):
        wide = g.pivot(index="instance", columns="variant", values=["n_dropped", "n_dropped_priority", "dropped_demand"])
        if {"priority", "uniform"} <= set(wide["n_dropped"].columns):
            both = wide.dropna()
            rows.append({
                "level": level, "variant": "priority - uniform", "n_instances": len(both),
                "dropped_total": float((both["n_dropped"]["priority"] - both["n_dropped"]["uniform"]).mean()),
                "dropped_priority_total": float(
                    (both["n_dropped_priority"]["priority"] - both["n_dropped_priority"]["uniform"]).mean()),
                "dropped_demand_mean": float((both["dropped_demand"]["priority"] - both["dropped_demand"]["uniform"]).mean()),
            })
    return pd.DataFrame(rows)


def soft_summary(rows: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Summarise C2.3 per (instance, cost level): representative plan, lateness and the reference comparison.

    Returns
    -------
    pandas.DataFrame
        ``vehicles``, ``distance``, ``K``, ``n_late``, ``late_share``, ``total_lateness``, ``max_lateness``,
        ``fleet_class``, ``distance_change_pct`` and ``*_mean`` columns over the feasible replicates.
    """
    keys = ["instance", "level"]
    rep = _with_reference(representative(rows, keys, ordering(SOFT_WINDOWS)), ref)
    rep["representative_order"] = ", ".join(ordering(SOFT_WINDOWS))
    cols = ["n_late", "total_lateness", "max_lateness", "distance"]
    out = rep.merge(_cell_means(rows, keys, cols), on=keys, how="left")
    keep = ["instance", "family", "level", "multiplier", "K", "n_replicates", "n_feasible", "feasible", "vehicles",
            "distance", "n_late", "late_share", "total_lateness", "max_lateness", "ref_vehicles", "ref_distance",
            "fleet_class", "distance_change_pct", *[f"{c}_mean" for c in cols], "capacity_infeasible",
            "representative_order"]
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def infeasible_breakdown(summary: pd.DataFrame) -> pd.DataFrame:
    """Count infeasible representative cells per level, separating structural from search failures.

    A cell is *capacity infeasible* when the fleet limit K is below ``ceil(sum(demand) / capacity)`` (no
    plan serving everyone can exist, so a solver failure is expected); otherwise an infeasible cell is a
    search failure.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`capacity_summary` or :func:`soft_summary` (``feasible``, ``capacity_infeasible``).

    Returns
    -------
    pandas.DataFrame
        Per level: ``n_cells``, ``n_feasible``, ``n_capacity_infeasible`` and ``n_search_failed``.
    """
    d = summary.copy()
    d["capacity_infeasible"] = d["capacity_infeasible"].astype(str).str.lower().eq("true")
    rows = []
    for level, g in d.groupby("level", sort=False):
        bad = ~g["feasible"]
        rows.append({
            "level": level, "n_cells": len(g), "n_feasible": int(g["feasible"].sum()),
            "n_capacity_infeasible": int((bad & g["capacity_infeasible"]).sum()),
            "n_search_failed": int((bad & ~g["capacity_infeasible"]).sum()),
        })
    return pd.DataFrame(rows)


def soft_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate C2.3 per cost level and fleet class: late customers, lateness and distance change."""
    ok = summary[summary["feasible"]]
    rows = []
    for (level, cls), g in ok.groupby(["level", "fleet_class"], sort=False):
        rows.append({
            "level": level, "fleet_class": cls, "n_instances": len(g), "n_late_mean": float(g["n_late"].mean()),
            "late_share_mean": float(g["late_share"].mean()), "total_lateness_mean": float(g["total_lateness"].mean()),
            "max_lateness_mean": float(g["max_lateness"].mean()), "max_lateness_max": float(g["max_lateness"].max()),
            "distance_change_mean": float(g["distance_change_pct"].mean()),
            "distance_change_median": float(g["distance_change_pct"].median()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- C3 tables
def noise_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the run-to-run noise floor: reallocation between pairs of cold replicates of the same scenario.

    For every instance the cold replicates ``r`` and ``r + 1`` (cyclic) are compared on the common customers
    (all but the sampled ones), in original ids.

    Parameters
    ----------
    rows : pandas.DataFrame
        C3 result rows (``variant`` ``"cold"``, ``routes`` JSON in original ids, ``sample`` JSON).

    Returns
    -------
    pandas.DataFrame
        ``instance``, ``replicate`` (first of the pair), ``variant = "noise"`` and the ``realloc_*`` fields.
    """
    cold = rows[(rows["variant"] == "cold") & rows["feasible"]]
    out = []
    for inst, g in cold.groupby("instance", sort=True):
        g = g.sort_values("replicate")
        if len(g) < 2:
            continue
        sample = set(json.loads(g.iloc[0]["sample"]))
        plans = [(int(r.replicate), [[c for c in rt if c not in sample] for rt in json.loads(r.routes)])
                 for r in g.itertuples()]
        plans = [(r, [rt for rt in p if rt]) for r, p in plans]
        for k, (rep, plan) in enumerate(plans):
            nxt = plans[(k + 1) % len(plans)][1]
            realloc = reallocation(plan, nxt)
            out.append({"instance": inst, "replicate": rep, "variant": "noise",
                        **{f"realloc_{f}": realloc[f] for f in REALLOC_FIELDS}})
    return pd.DataFrame(out)


def reopt_aggregate(rows: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    """Aggregate C3 by variant (warm, cold) and the noise floor: medians over instances and replicates.

    Returns
    -------
    pandas.DataFrame
        One row per variant (``warm``, ``cold``, ``noise``) with ``n``, the medians of the reallocation
        metrics and, for warm and cold, the mean change of vehicles and the median distance change against the
        original plan on the same customers.
    """
    metrics = [f"realloc_{f}" for f in REALLOC_FIELDS]
    part = rows[rows["variant"].isin(["warm", "cold"]) & rows["feasible"]]
    out = []
    for variant in ("warm", "cold"):
        g = part[part["variant"] == variant]
        if g.empty:
            continue
        out.append({
            "variant": variant, "n": len(g), **{m: g[m].median() for m in metrics},
            "delta_vehicles_mean": g["delta_vehicles"].mean(), "delta_distance_pct_median": g["delta_distance_pct"].median(),
            "runtime_s_mean": g["runtime_s"].mean(),
        })
    if len(noise):
        out.append({"variant": "noise", "n": len(noise), **{m: noise[m].median() for m in metrics}})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- C3: representative replicate
def reopt_representatives(rows: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """Pick the representative replicate of every (block, instance, variant) of C3, as the protocol prescribes.

    The replicates of the warm and cold variants of one instance and event are ordered by (infeasible,
    ``*order``), ties by replicate id, and the one at position ``ceil(R / 2)`` is taken (the 2nd of 3), so each
    instance weighs once per variant. Cells whose representative is infeasible are dropped. ``plan90`` and
    any other variant are ignored.

    Parameters
    ----------
    rows : pandas.DataFrame
        C3 result rows (:func:`read_results`).
    order : list of str, optional
        Ascending sort columns after infeasibility; defaults to ``DEFAULT_ORDER`` (vehicles, distance).

    Returns
    -------
    pandas.DataFrame
        One row per (block, instance, variant) with ``n_replicates`` and ``n_feasible``.
    """
    part = rows[rows["variant"].isin(["warm", "cold"])]
    keys = [c for c in ("block", "instance", "variant") if c in part.columns]
    rep = representative(part, keys, order)
    return rep[rep["feasible"].astype(bool)].reset_index(drop=True)


def noise_distribution(noise: pd.DataFrame) -> pd.DataFrame:
    """Describe the noise floor as a distribution over the cold-versus-cold comparisons.

    Parameters
    ----------
    noise : pandas.DataFrame
        Output of :func:`noise_table`.

    Returns
    -------
    pandas.DataFrame
        One row with ``n_pairs``, ``median_share``, ``n_positive`` (comparisons that reassigned any customer),
        ``max_share`` and ``instances``: the instances with a positive floor, each as
        ``"R111 23.3% x2"`` (share of reassigned customers and number of comparisons), joined by ``"; "``.
    """
    share = noise["realloc_customers_reassigned_share"].astype(float)
    pos = noise[share > 0]
    parts = []
    for inst, g in pos.groupby("instance", sort=True):
        counts = g["realloc_customers_reassigned_share"].round(4).value_counts().sort_index(ascending=False)
        parts.append(", ".join(f"{inst} {100 * s:.1f}% x{c}" for s, c in counts.items()))
    return pd.DataFrame([{
        "n_pairs": len(noise), "median_share": float(share.median()) if len(share) else float("nan"),
        "n_positive": int((share > 0).sum()), "max_share": float(share.max()) if len(share) else float("nan"),
        "instances": "; ".join(parts),
    }])


def reopt_aggregate_representative(rows: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    """Aggregate C3 over the representative replicate of each instance (warm, cold) plus the noise floor.

    Parameters
    ----------
    rows : pandas.DataFrame
        C3 result rows.
    noise : pandas.DataFrame
        Output of :func:`noise_table`.

    Returns
    -------
    pandas.DataFrame
        One row per variant: ``warm`` and ``cold`` (``n`` = instances; medians of the reallocation metrics,
        mean vehicle change and median distance change against the original plan) and ``noise`` (``n`` =
        comparisons, the medians of the same metrics and, in ``noise_n_positive`` and ``noise_max_share``, the
        distribution of the floor).
    """
    metrics = [f"realloc_{f}" for f in REALLOC_FIELDS]
    rep = reopt_representatives(rows)
    for col in (*metrics, "delta_vehicles", "delta_distance_pct", "runtime_s"):
        rep[col] = pd.to_numeric(rep[col], errors="coerce")
    out = []
    for variant in ("warm", "cold"):
        g = rep[rep["variant"] == variant]
        if g.empty:
            continue
        out.append({
            "variant": variant, "n": len(g), **{m: g[m].median() for m in metrics},
            "delta_vehicles_mean": g["delta_vehicles"].mean(), "delta_distance_pct_median": g["delta_distance_pct"].median(),
            "runtime_s_mean": g["runtime_s"].mean(),
        })
    if len(noise):
        dist = noise_distribution(noise).iloc[0]
        out.append({"variant": "noise", "n": len(noise), **{m: noise[m].median() for m in metrics},
                    "noise_n_positive": int(dist["n_positive"]), "noise_max_share": float(dist["max_share"]),
                    "noise_instances": dist["instances"]})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- C2.2: spread across replicates
def dropped_spread(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure, per (instance, level, variant) of C2.2, how much the dropped customers vary across replicates.

    The drop penalty encourages service but the heuristic does not guarantee a minimal set of dropped
    customers: replicates of the same cell can drop a different number of customers, or different ones.

    Parameters
    ----------
    rows : pandas.DataFrame
        C2.2 (``fleet_priority``) cells with ``n_dropped``, ``n_dropped_priority`` and ``dropped_ids`` (JSON).

    Returns
    -------
    pandas.DataFrame
        One row per cell group: ``n_replicates``, ``dropped_values`` (the numbers dropped, in replicate order,
        for example ``"9, 7, 7"``), ``dropped_min``, ``dropped_median``, ``dropped_max``, ``dropped_range``,
        ``priority_dropped_values``, ``n_distinct_sets`` (distinct sets of dropped customers).
    """
    part = rows[rows["feasible"].astype(bool)].copy()
    part["_set"] = part["dropped_ids"].map(lambda v: tuple(sorted(json.loads(v))) if isinstance(v, str) else ())
    out = []
    for (inst, level, variant), g in part.groupby(["instance", "level", "variant"], sort=True):
        g = g.sort_values("replicate")
        n = g["n_dropped"].astype(int)
        out.append({
            "instance": inst, "level": level, "variant": variant, "n_replicates": len(g),
            "dropped_values": ", ".join(str(v) for v in n), "dropped_min": int(n.min()),
            "dropped_median": float(n.median()), "dropped_max": int(n.max()), "dropped_range": int(n.max() - n.min()),
            "priority_dropped_values": ", ".join(str(int(v)) for v in g["n_dropped_priority"]),
            "n_distinct_sets": g["_set"].nunique(),
        })
    return pd.DataFrame(out)
