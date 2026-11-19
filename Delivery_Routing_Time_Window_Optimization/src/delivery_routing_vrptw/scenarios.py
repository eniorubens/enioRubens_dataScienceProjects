"""Business-layer scenarios (layer B): customer time-window widening and early delivery.

These scenarios change the problem, so their results are never compared with SINTEF references;
the only reference is the unmodified instance (factor 1.0) solved with the same solver.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .distance import exact, scaled_ceil, scaled_round
from .io import Instance, validate_instance
from .solution import Solution
from .solver import SolverConfig, solve
from .verify import verify

MODES = ("symmetric", "later_due")


def widen_windows(instance: Instance, factor: float, mode: str = "symmetric") -> Instance:
    """Return a copy of the instance with every customer time window widened.

    With width ``w = due - ready`` and ``extra = (factor - 1) * w``, ``"symmetric"`` opens the window
    ``extra / 2`` earlier and closes it ``extra / 2`` later; ``"later_due"`` only closes it ``extra``
    later. The opening is clipped at 0 and the closing at the latest service start that still allows
    the return to the depot (``horizon - service - d(i, 0)``), never below the original due date.
    ``ready`` is rounded down and ``due`` up, so the result is an integer instance whose windows
    always contain the original ones. Depot, coordinates, demand, capacity and service are unchanged.

    Parameters
    ----------
    instance : Instance
        Original instance.
    factor : float
        Width multiplier, ``>= 1``. ``1.0`` returns an equal instance with the same name.
    mode : {"symmetric", "later_due"}
        How the extra width is distributed.

    Returns
    -------
    Instance
        New instance named ``"<name>@w<factor>"`` (for example ``R101@w1.25``).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if factor < 1.0:
        raise ValueError("factor must be >= 1.0")
    if factor == 1.0:
        return replace(instance, ready=instance.ready.copy(), due=instance.due.copy())
    ready = instance.ready.astype(np.int64).copy()
    due = instance.due.astype(np.int64).copy()
    dist_back = exact(instance)[:, 0]
    latest = np.floor(instance.horizon - instance.service - dist_back)
    for i in range(1, len(ready)):
        extra = (factor - 1.0) * (int(instance.due[i]) - int(instance.ready[i]))
        lo = extra / 2 if mode == "symmetric" else 0.0
        hi = extra / 2 if mode == "symmetric" else extra
        new_ready = max(0, math.floor(instance.ready[i] - lo + 1e-9))
        new_due = math.ceil(instance.due[i] + hi - 1e-9)
        new_due = max(int(instance.due[i]), min(new_due, int(latest[i])))
        ready[i], due[i] = min(new_ready, int(instance.ready[i])), new_due
    assert np.all(ready <= instance.ready) and np.all(due >= instance.due) and np.all(ready <= due)
    widened = replace(instance, name=f"{instance.name}@w{factor:g}", ready=ready, due=due)
    validate_instance(widened)
    return widened


def tolerance_label(tolerance: float) -> str:
    """Return the label of an early-delivery tolerance: ``"full"`` for infinity, else ``"0.25"`` etc."""
    return "full" if math.isinf(tolerance) else f"{tolerance:g}"


def allow_early(instance: Instance, tolerance: float) -> Instance:
    """Return a copy of the instance in which customers accept deliveries before their window opens.

    With width ``w = due - ready``, the opening becomes ``max(0, floor(ready - tolerance * w))``;
    ``due`` is unchanged. The solver and the verifier keep the "wait until ready" rule, so the
    scenario is expressed purely as an input transformation. Depot, coordinates, demand, capacity and
    service are unchanged, and the new window always contains the original one.

    Parameters
    ----------
    instance : Instance
        Original instance.
    tolerance : float
        Fraction of the window width by which ``ready`` may be anticipated, ``>= 0``. ``0`` returns an
        equal instance with the same name; ``math.inf`` sets ``ready = 0`` (deliver any time before due).

    Returns
    -------
    Instance
        New instance named ``"<name>@e<tolerance>"`` (for example ``R101@e0.5`` or ``R101@efull``).
    """
    if math.isnan(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    if tolerance == 0:
        return replace(instance, ready=instance.ready.copy(), due=instance.due.copy())
    ready = instance.ready.astype(np.int64).copy()
    for i in range(1, len(ready)):
        if math.isinf(tolerance):
            ready[i] = 0
        else:
            width = int(instance.due[i]) - int(instance.ready[i])
            lowered = math.floor(int(instance.ready[i]) - tolerance * width + 1e-9)
            ready[i] = min(int(instance.ready[i]), max(0, lowered))
    early = replace(
        instance, name=f"{instance.name}@e{tolerance_label(tolerance)}", ready=ready, due=instance.due.copy()
    )
    validate_instance(early)
    return early


def window_scenario_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise a window-widening run against its own factor-1.0 baseline (no SINTEF involved).

    Parameters
    ----------
    rows : pandas.DataFrame
        Columns ``instance``, ``factor``, ``mode``, ``feasible``, ``vehicles``, ``distance``.

    Returns
    -------
    pandas.DataFrame
        One row per (instance, mode, factor) with ``vehicles``, ``distance``, ``vehicles_saved``
        (baseline minus scenario) and ``distance_change_pct`` (scenario vs baseline), both NaN when the
        scenario or the baseline run is infeasible.
    """
    cols = ["instance", "mode", "factor", "vehicles", "distance", "vehicles_saved", "distance_change_pct"]
    df = rows.copy()
    df["feasible"] = df["feasible"].astype(str).str.lower().eq("true")
    for col in ("vehicles", "distance"):
        df[col] = pd.to_numeric(df[col], errors="coerce").where(df["feasible"])
    base = df[df["factor"] == 1.0].set_index(["instance", "mode"])[["vehicles", "distance"]]
    base.columns = ["base_vehicles", "base_distance"]
    out = df.join(base, on=["instance", "mode"])
    out["vehicles_saved"] = out["base_vehicles"] - out["vehicles"]
    out["distance_change_pct"] = 100.0 * (out["distance"] - out["base_distance"]) / out["base_distance"]
    return out.sort_values(["instance", "mode", "factor"])[cols].reset_index(drop=True)


def solve_ladder(
    levels: Sequence[Instance], config: SolverConfig | None = None, time_limit_s: float | None = None
) -> list[Solution]:
    """Solve a ladder of increasingly permissive instances, each warm-started from the previous level.

    Level 0 is solved cold. Each next level starts from the routes of the previous level's solution, which
    are feasible for the more permissive instance by construction (its windows contain the previous ones);
    this is asserted with :func:`delivery_routing_vrptw.verify.verify` before the routes are used. Because
    the search starts from that plan and never accepts a worse one, a level cannot be lexicographically worse
    (vehicles, then distance) than the previous plan evaluated on it, up to the integer rounding of the
    solver. If a level returns no feasible plan, the next level is solved cold.

    Parameters
    ----------
    levels : sequence of Instance
        The same instance under nested constraints, from the least to the most permissive.
    config : SolverConfig, optional
        Solver configuration shared by all levels.
    time_limit_s : float, optional
        Overrides ``config.time_limit_s`` for every level.

    Returns
    -------
    list of Solution
        One solution per level; ``meta["warm_start"]`` tells whether the level started from the previous
        plan (``"used"``), was rejected by OR-Tools (``"rejected"``) or was cold (``"none"``).

    Raises
    ------
    AssertionError
        If a previous feasible plan is infeasible for the next level (the levels are not nested).
    """
    config = config or SolverConfig()
    if time_limit_s is not None:
        config = replace(config, time_limit_s=time_limit_s)
    out: list[Solution] = []
    prev_inst: Instance | None = None
    prev_routes: list[list[int]] | None = None
    for inst in levels:
        start = None
        if prev_routes and prev_inst is not None and verify(prev_inst, prev_routes).feasible:
            assert verify(inst, prev_routes).feasible, "ladder levels must be nested (more permissive)"
            start = prev_routes
        sol = solve(inst, config, initial_routes=start)
        out.append(sol)
        prev_inst, prev_routes = inst, sol.routes
    return out


# --------------------------------------------------------------------------- phase C transforms
def scale_capacity(instance: Instance, factor: float) -> Instance:
    """Return a copy of the instance with the vehicle capacity multiplied by ``factor`` (floored).

    Parameters
    ----------
    instance : Instance
        Original instance.
    factor : float
        Capacity multiplier, ``> 0``. ``1.0`` returns an equal instance with the same name.

    Returns
    -------
    Instance
        New instance named ``"<name>@q<factor>"`` with ``capacity = floor(Q * factor)``.

    Raises
    ------
    ValueError
        If the factor is not positive, or the reduced capacity is below the largest customer demand
        (the instance would be infeasible by construction; the caller should report it).
    """
    if not factor > 0:
        raise ValueError("factor must be > 0")
    if factor == 1.0:
        return replace(instance, x=instance.x.copy())
    capacity = math.floor(instance.capacity * factor + 1e-9)
    biggest = int(instance.demand.max())
    if capacity < biggest:
        raise ValueError(
            f"capacity {capacity} (= floor({instance.capacity} * {factor:g})) is below the largest demand {biggest}"
        )
    return replace(instance, name=f"{instance.name}@q{factor:g}", capacity=int(capacity))


def limit_fleet(instance: Instance, vehicles: int) -> Instance:
    """Return a copy of the instance whose fleet limit is ``vehicles`` (so :func:`verify` enforces it).

    Parameters
    ----------
    instance : Instance
        Original instance.
    vehicles : int
        New vehicle limit, ``>= 1``.

    Returns
    -------
    Instance
        New instance named ``"<name>@k<vehicles>"``.
    """
    if vehicles < 1:
        raise ValueError("vehicles must be >= 1")
    return replace(instance, name=f"{instance.name}@k{int(vehicles)}", vehicle_limit=int(vehicles))


@dataclass(frozen=True)
class CustomerMap:
    """Correspondence between the customer ids of an instance and those of a reduced copy.

    Attributes
    ----------
    new_to_old : tuple of int
        Original id of every reduced node; ``new_to_old[0] == 0`` (the depot).
    """

    new_to_old: tuple[int, ...]

    @property
    def old_to_new(self) -> dict[int, int]:
        """Return the reduced id of every kept original customer."""
        return {old: new for new, old in enumerate(self.new_to_old) if new > 0}

    def to_original(self, routes: list[list[int]]) -> list[list[int]]:
        """Translate routes of the reduced instance into original ids."""
        return [[self.new_to_old[c] for c in r] for r in routes if r]

    def to_reduced(self, routes: list[list[int]]) -> list[list[int]]:
        """Translate original-id routes into reduced ids, dropping customers that are not kept.

        Routes that become empty are removed; the visiting order of the remaining customers is unchanged.
        """
        mapping = self.old_to_new
        out = [[mapping[c] for c in r if c in mapping] for r in routes]
        return [r for r in out if r]


def remove_customers(instance: Instance, ids: Sequence[int]) -> tuple[Instance, CustomerMap]:
    """Return the instance without the given customers, re-indexed ``0..n``, and the id mapping.

    Parameters
    ----------
    instance : Instance
        Original instance.
    ids : sequence of int
        Customers to remove (ids ``1..n``; duplicates are ignored).

    Returns
    -------
    tuple
        ``(reduced, mapping)``: the reduced instance named ``"<name>@rm<count>"`` (depot, capacity, fleet
        limit and the order of the kept customers unchanged) and a :class:`CustomerMap` to translate plans
        both ways.

    Raises
    ------
    ValueError
        If an id is not a customer of the instance.
    """
    gone = {int(i) for i in ids}
    bad = sorted(i for i in gone if not 1 <= i <= instance.n_customers)
    if bad:
        raise ValueError(f"not customers of the instance: {bad}")
    keep = np.array([0] + [i for i in range(1, instance.n_customers + 1) if i not in gone], dtype=int)
    reduced = replace(
        instance, name=f"{instance.name}@rm{len(gone)}", x=instance.x[keep], y=instance.y[keep],
        demand=instance.demand[keep], ready=instance.ready[keep], due=instance.due[keep],
        service=instance.service[keep],
    )
    validate_instance(reduced)
    return reduced, CustomerMap(tuple(int(i) for i in keep))


def restrict_routes(routes: list[list[int]], keep: Sequence[int]) -> list[list[int]]:
    """Return the routes restricted to the customers in ``keep`` (order kept, empty routes dropped)."""
    wanted = set(keep)
    out = [[c for c in r if c in wanted] for r in routes]
    return [r for r in out if r]


def insert_customers(
    instance: Instance, routes: list[list[int]], customers: Sequence[int], scale: int = 10000
) -> list[list[int]] | None:
    """Complete a plan by cheapest feasible insertion of the missing customers (a warm-start builder).

    OR-Tools rejects a warm start that leaves mandatory customers unrouted, so the insertion warm start
    of phase C3 is completed here. Customers are inserted by increasing ``due`` (ties by id) at the
    cheapest position of any route that keeps capacity and every time window, in the integer arithmetic of
    the solver (``ceil(d * scale)`` times, ``round(d * scale)`` costs); when no position fits, a new route
    is opened.

    Parameters
    ----------
    instance : Instance
        The instance (all customers, fleet limit ``vehicle_limit``).
    routes : list of list of int
        Existing plan (customer ids, no depot); it is not modified.
    customers : sequence of int
        Customers to insert.
    scale : int
        Integer scale of the solver the routes will warm-start.

    Returns
    -------
    list of list of int or None
        The completed plan, or ``None`` when it would need more than ``vehicle_limit`` routes or a
        customer cannot be served even alone.
    """
    travel = scaled_ceil(instance, scale)
    cost = scaled_round(instance, scale)
    service = instance.service.astype(np.int64) * scale
    ready, due = instance.ready.astype(np.int64) * scale, instance.due.astype(np.int64) * scale
    horizon, cap = int(instance.horizon) * scale, int(instance.capacity)
    demand = instance.demand

    def fits(route: list[int]) -> bool:
        t, prev, load = 0, 0, 0
        for c in route:
            t = max(t + int(service[prev]) + int(travel[prev, c]), int(ready[c]))
            load += int(demand[c])
            if t > int(due[c]) or load > cap:
                return False
            prev = c
        return t + int(service[prev]) + int(travel[prev, 0]) <= horizon

    def length(route: list[int]) -> int:
        path = [0, *route, 0]
        return sum(int(cost[a, b]) for a, b in zip(path[:-1], path[1:], strict=True))

    plan = [list(r) for r in routes if r]
    for c in sorted(customers, key=lambda i: (int(instance.due[i]), i)):
        best: tuple[int, int, int] | None = None
        for ri, route in enumerate(plan):
            base = length(route)
            for pos in range(len(route) + 1):
                trial = [*route[:pos], c, *route[pos:]]
                if fits(trial):
                    delta = length(trial) - base
                    if best is None or delta < best[0]:
                        best = (delta, ri, pos)
        if best is not None:
            _, ri, pos = best
            plan[ri].insert(pos, c)
        elif len(plan) < instance.vehicle_limit and fits([c]):
            plan.append([c])
        else:
            return None
    return plan
