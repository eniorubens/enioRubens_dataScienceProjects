"""OR-Tools routing model for the rigid VRPTW (benchmark layer: no dropped customers)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .distance import exact, scaled_ceil, scaled_round
from .io import Instance
from .solution import Solution
from .verify import verify

_FIRST = routing_enums_pb2.FirstSolutionStrategy
_META = routing_enums_pb2.LocalSearchMetaheuristic
_STATUS = {
    0: "ROUTING_NOT_SOLVED",
    1: "ROUTING_SUCCESS",
    2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
    3: "ROUTING_FAIL",
    4: "ROUTING_FAIL_TIMEOUT",
    5: "ROUTING_INVALID",
    6: "ROUTING_INFEASIBLE",
    7: "ROUTING_OPTIMAL",
}


@dataclass(frozen=True)
class SolverConfig:
    """Configuration of one OR-Tools run.

    OR-Tools routing has no random seed in this setup: with the same limits and a single worker the
    search is deterministic, except that a wall-clock ``time_limit_s`` makes the stopping point depend
    on machine load.

    Attributes
    ----------
    first_solution : str
        Name of a ``FirstSolutionStrategy`` value.
    metaheuristic : str
        Name of a ``LocalSearchMetaheuristic`` value, or ``"NONE"`` to stop at the first solution
        (``solution_limit=1``, no local search).
    time_limit_s : float
        Wall-clock search limit in seconds.
    scale : int
        Integer scaling of distances and times.
    vehicle_fixed_cost : int
        Fixed cost per used vehicle; must dominate any distance so fleet size is minimised first.
    num_vehicles : int or None
        Vehicles available; ``None`` uses ``instance.vehicle_limit``.
    log_search : bool
        Print OR-Tools search logs.
    fixed_cost_k : float or None
        v2 option. When set, the fixed cost per vehicle of an instance is
        ``round(k * scale * D_ub)`` with ``D_ub = sum_i 2 * d(0, i)`` (see :func:`fixed_cost_for`) and
        ``vehicle_fixed_cost`` is ignored. ``None`` keeps the explicit ``vehicle_fixed_cost`` (v1).
    """

    first_solution: str = "PARALLEL_CHEAPEST_INSERTION"
    metaheuristic: str = "GUIDED_LOCAL_SEARCH"
    time_limit_s: float = 30.0
    scale: int = 100
    vehicle_fixed_cost: int = 10**7
    num_vehicles: int | None = None
    log_search: bool = False
    fixed_cost_k: float | None = None

    def __post_init__(self) -> None:
        """Validate strategy names and numeric ranges."""
        if self.first_solution not in _FIRST.Value.keys():
            raise ValueError(f"unknown first_solution strategy: {self.first_solution!r}")
        if self.metaheuristic != "NONE" and self.metaheuristic not in _META.Value.keys():
            raise ValueError(f"unknown metaheuristic: {self.metaheuristic!r}")
        if self.time_limit_s <= 0 or self.scale < 1:
            raise ValueError("time_limit_s must be positive and scale >= 1")
        if self.num_vehicles is not None and self.num_vehicles < 1:
            raise ValueError("num_vehicles must be >= 1")
        if self.fixed_cost_k is not None and self.fixed_cost_k <= 0:
            raise ValueError("fixed_cost_k must be positive")


def fixed_cost_for(instance: Instance, config: SolverConfig) -> int:
    """Return the fixed cost per used vehicle for an instance.

    Parameters
    ----------
    instance : Instance
        The instance.
    config : SolverConfig
        Solver configuration.

    Returns
    -------
    int
        ``config.vehicle_fixed_cost`` when ``config.fixed_cost_k`` is ``None`` (v1 behaviour); otherwise
        ``round(k * scale * D_ub)`` with ``D_ub = sum_i 2 * d(0, i)`` computed from the exact distances,
        a simple upper bound of the total distance, so the fixed cost scales with the integer scale.
    """
    if config.fixed_cost_k is None:
        return config.vehicle_fixed_cost
    return int(round(config.fixed_cost_k * config.scale * distance_upper_bound(instance)))


def distance_upper_bound(instance: Instance) -> float:
    """Return ``D_ub = sum_i 2 * d(0, i)``, a simple upper bound of the total distance of any plan.

    Parameters
    ----------
    instance : Instance
        The instance.

    Returns
    -------
    float
        Sum over customers of the round-trip distance depot -> customer -> depot.
    """
    return float(2.0 * exact(instance)[0, 1:].sum())


def solve(
    instance: Instance,
    config: SolverConfig | None = None,
    initial_routes: list[list[int]] | None = None,
    drop_penalty: int | None = None,
    drop_penalties: dict[int, int] | None = None,
    soft_due_cost_per_unit: int | None = None,
) -> Solution:
    """Solve the VRPTW with OR-Tools routing (rigid by default; optional dropping and soft due dates).

    Arc cost is ``round(d*scale)``; the time dimension uses ``service*scale + ceil(d*scale)`` so that a
    solver-feasible schedule is feasible in double precision. By default customers cannot be dropped and
    every time window is hard.

    Parameters
    ----------
    instance : Instance
        The instance.
    config : SolverConfig, optional
        Solver configuration; defaults to ``SolverConfig()``.
    initial_routes : list of list of int, optional
        v2 warm start: routes (customer ids, no depot) the search starts from, read with
        ``ReadAssignmentFromRoutes`` and improved with ``SolveFromAssignmentWithParameters``. They must be
        feasible for the model (in its integer arithmetic); if OR-Tools rejects them, or they need more
        vehicles than available, the solve falls back to a cold start and ``meta["warm_start"]`` is
        ``"rejected"``.
    drop_penalty : int, optional
        v2 penalised dropping: when set, every customer gets ``AddDisjunction([index], drop_penalty)`` and
        may be left unserved at that cost; dropped customers are listed in ``meta["dropped"]`` and are not in
        ``routes``. ``None`` (default) keeps the v1 behaviour (all customers mandatory).
    drop_penalties : dict of int to int, optional
        Phase C: per-customer dropping penalties (customer id -> penalty), for priorities. A customer in
        this mapping gets ``AddDisjunction`` with its own penalty, overriding ``drop_penalty``; customers
        in neither stay mandatory.
    soft_due_cost_per_unit : int, optional
        Phase C: soft window ends. When set, the hard upper bound of every customer's service start is
        relaxed to the horizon and ``SetCumulVarSoftUpperBound(index, due * scale, cost)`` charges
        ``soft_due_cost_per_unit`` cost units per *scaled* time unit of lateness. Arc costs are
        ``round(d * scale)``, so one scaled distance unit costs 1; ``soft_due_cost_per_unit = 1`` therefore
        prices one unit of lateness like one unit of distance and ``10`` ten times that. The depot return
        stays hard. Lateness is measured with :func:`delivery_routing_vrptw.verify.measure_lateness`.

    Returns
    -------
    Solution
        Routes of the non-empty vehicles. If no solution is found, ``routes`` is empty and
        ``meta["status"]`` explains why; this function never raises for search failures. ``meta`` also holds
        ``warm_start`` (``"none"``, ``"used"`` or ``"rejected"``), ``dropped`` (sorted customer ids) and
        ``vehicle_fixed_cost_effective``.
    """
    config = config or SolverConfig()
    if soft_due_cost_per_unit is not None and int(soft_due_cost_per_unit) < 1:
        raise ValueError("soft_due_cost_per_unit must be >= 1")
    t0 = time.perf_counter()
    scale = config.scale
    n_nodes = instance.n_customers + 1
    n_veh = config.num_vehicles if config.num_vehicles is not None else instance.vehicle_limit
    horizon = instance.horizon * scale

    cost = scaled_round(instance, scale).tolist()
    travel = scaled_ceil(instance, scale).tolist()
    service = [int(s) * scale for s in instance.service]
    demand = [int(q) for q in instance.demand]

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_veh, 0)
    routing = pywrapcp.RoutingModel(manager)

    def arc_cost(i: int, j: int) -> int:
        return cost[manager.IndexToNode(i)][manager.IndexToNode(j)]

    def transit_time(i: int, j: int) -> int:
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return service[a] + travel[a][b]

    def load(i: int) -> int:
        return demand[manager.IndexToNode(i)]

    cost_idx = routing.RegisterTransitCallback(arc_cost)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)
    fixed_cost = fixed_cost_for(instance, config)
    routing.SetFixedCostOfAllVehicles(fixed_cost)

    time_idx = routing.RegisterTransitCallback(transit_time)
    routing.AddDimension(time_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    soft = soft_due_cost_per_unit is not None
    for node in range(1, n_nodes):
        # a soft due date relaxes the hard upper bound to the horizon (SetRange only narrows a domain)
        upper = horizon if soft else int(instance.due[node]) * scale
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(int(instance.ready[node]) * scale, upper)
    for v in range(n_veh):
        time_dim.CumulVar(routing.Start(v)).SetRange(0, horizon)
        time_dim.CumulVar(routing.End(v)).SetRange(0, horizon)

    load_idx = routing.RegisterUnaryTransitCallback(load)
    routing.AddDimensionWithVehicleCapacity(load_idx, 0, [instance.capacity] * n_veh, True, "Load")

    if soft:
        for node in range(1, n_nodes):
            time_dim.SetCumulVarSoftUpperBound(
                manager.NodeToIndex(node), int(instance.due[node]) * scale, int(soft_due_cost_per_unit)
            )

    penalties: dict[int, int] = {}
    if drop_penalty is not None:
        penalties = {node: int(drop_penalty) for node in range(1, n_nodes)}
    if drop_penalties:
        bad = [c for c in drop_penalties if not 1 <= c < n_nodes]
        if bad or any(int(p) < 1 for p in drop_penalties.values()):
            raise ValueError("drop_penalties needs customer ids in 1..n and penalties >= 1")
        penalties.update({int(c): int(p) for c, p in drop_penalties.items()})
    for node, pen in penalties.items():
        routing.AddDisjunction([manager.NodeToIndex(node)], pen)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(_FIRST, config.first_solution)
    if config.metaheuristic == "NONE":
        params.solution_limit = 1
    else:
        params.local_search_metaheuristic = getattr(_META, config.metaheuristic)
    params.time_limit.FromMilliseconds(int(config.time_limit_s * 1000))
    params.log_search = config.log_search

    warm_start = "none"
    assignment = None
    if initial_routes is not None:
        warm_start = "rejected"
        initial = None
        routes_in = [list(r) for r in initial_routes if r]
        if 0 < len(routes_in) <= n_veh:
            routing.CloseModelWithParameters(params)
            initial = routing.ReadAssignmentFromRoutes(
                [[manager.NodeToIndex(int(c)) for c in r] for r in routes_in], True
            )
        if initial is not None:
            assignment = routing.SolveFromAssignmentWithParameters(initial, params)
            if assignment is not None:
                warm_start = "used"
    if assignment is None:
        assignment = routing.SolveWithParameters(params)
    status = _STATUS.get(routing.status(), str(routing.status()))
    meta = {
        **asdict(config),
        "status": status,
        "objective": None,
        "solver_vehicles": n_veh,
        "warm_start": warm_start,
        "dropped": [],
        "vehicle_fixed_cost_effective": fixed_cost,
        "soft_due_cost_per_unit": soft_due_cost_per_unit,
    }
    routes: list[list[int]] = []
    if assignment is not None:
        meta["objective"] = int(assignment.ObjectiveValue())
        for v in range(n_veh):
            index = routing.Start(v)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    route.append(node)
                index = assignment.Value(routing.NextVar(index))
            if route:
                routes.append(route)
        if penalties:
            served = {c for r in routes for c in r}
            meta["dropped"] = [c for c in range(1, n_nodes) if c not in served]
    return Solution(routes, "ortools", time.perf_counter() - t0, meta)


def minimize_fleet(
    instance: Instance,
    config: SolverConfig | None = None,
    start_vehicles: int | None = None,
    step_time_s: float | None = None,
    total_time_s: float | None = None,
    first_step_time_s: float | None = None,
) -> Solution:
    """Fleet-reduction loop: re-solve with one vehicle fewer until no solution is found.

    Parameters
    ----------
    instance : Instance
        The instance.
    config : SolverConfig, optional
        Base configuration (its ``num_vehicles`` is overridden).
    start_vehicles : int, optional
        Fleet size of the first solve; defaults to ``config.num_vehicles`` or the instance limit.
    step_time_s : float, optional
        Time limit of every solve; defaults to ``config.time_limit_s``.
    total_time_s : float, optional
        Total wall-clock budget of the loop. When set, the first solve uses ``first_step_time_s`` (or the
        step time) and every later solve uses ``min(step_time, remaining)``; the loop stops when less
        than 1 s remains. ``None`` keeps the unbudgeted behaviour.
    first_step_time_s : float, optional
        Time limit of the first solve when ``total_time_s`` is set.

    Returns
    -------
    Solution
        Best verified-feasible solution (fewest vehicles, then shortest distance). ``meta["attempts"]``
        lists ``(fleet_size, found, vehicles, distance)`` per solve; ``runtime_s`` is the total.

    Notes
    -----
    Every solve starts cold and cannot drop customers, so the attempt with one vehicle fewer usually fails
    in the construction phase, almost at once; the loop then stops and the remaining budget is discarded. In
    the frozen development grid the total runtime was therefore about the first step (10 s of a 30 s budget)
    for two of the three first-solution strategies (see ``analysis.reduce_fleet_timing``); an earlier
    explanation that the step was too short was wrong. A fair test needs a warm start and penalised dropping.
    """
    config = config or SolverConfig()
    t0 = time.perf_counter()
    matrix = exact(instance)
    step = replace(config, time_limit_s=step_time_s if step_time_s is not None else config.time_limit_s)
    fleet = start_vehicles or config.num_vehicles or instance.vehicle_limit
    best: Solution | None = None
    best_key: tuple[int, float] | None = None
    attempts: list[tuple[int, bool, int, float]] = []
    while fleet >= 1:
        solve_cfg = step
        if total_time_s is not None:
            remaining = total_time_s - (time.perf_counter() - t0)
            if remaining < 1.0:
                break
            limit = first_step_time_s if (first_step_time_s is not None and not attempts) else step.time_limit_s
            solve_cfg = replace(step, time_limit_s=min(limit, remaining))
        sol = solve(instance, replace(solve_cfg, num_vehicles=fleet))
        rep = verify(instance, sol.routes, matrix) if sol.routes else None
        ok = rep is not None and rep.feasible
        attempts.append((fleet, ok, rep.vehicles if rep else 0, rep.distance if rep else float("nan")))
        if not ok:
            break
        key = (rep.vehicles, rep.distance)
        if best_key is None or key < best_key:
            best, best_key = sol, key
        fleet = rep.vehicles - 1
    if best is None:
        best = Solution([], "ortools_minimize_fleet", 0.0, {"status": "NO_SOLUTION"})
    meta = {**best.meta, "attempts": attempts}
    return Solution(best.routes, "ortools_minimize_fleet", time.perf_counter() - t0, meta)


def minimize_fleet_v2(
    instance: Instance,
    config: SolverConfig | None = None,
    total_time_s: float = 120.0,
    first_fraction: float = 1 / 3,
    step_time_s: float = 20.0,
    drop_penalty_factor: int = 100,
) -> Solution:
    """Fleet reduction with warm start and penalised dropping (protocol v2, block B3).

    The first solve is cold and uses ``first_fraction * total_time_s``. While at least 1 s of the total
    budget remains, the loop takes the best plan and tries to empty one of its routes, fewest customers
    first: the remaining routes are the warm start of a solve with ``best.vehicles - 1`` vehicles in which
    every customer may be dropped at ``drop_penalty_factor * vehicle_fixed_cost`` (larger than anything
    else, so dropping is a last resort). The attempt is accepted only if no customer is dropped **and** the
    plan is feasible against the original instance (double precision); the loop then continues from the new
    plan. On a failure the next candidate route (next fewest customers) is tried, so a fast failure never
    discards the remaining budget. The loop ends when no candidate of the current plan succeeds or the
    budget is exhausted.

    The time limit of an attempt is ``min(step_time_s + carry_over, remaining)``, where ``carry_over`` is the
    time the previous attempt left unused (its limit minus its duration, never negative): unused time of an
    accepted attempt passes to the next one, and a fast failure does not waste its share either.

    Parameters
    ----------
    instance : Instance
        The instance.
    config : SolverConfig, optional
        Base configuration (``num_vehicles`` of the first solve is kept; ``time_limit_s`` is overridden).
    total_time_s : float
        Total wall-clock budget.
    first_fraction : float
        Fraction of the total budget given to the first (cold) solve.
    step_time_s : float
        Nominal time limit of one reduction attempt.
    drop_penalty_factor : int
        The drop penalty is ``drop_penalty_factor * fixed_cost_for(instance, config)``.

    Returns
    -------
    Solution
        Best verified-feasible plan (``NO_SOLUTION`` status with empty routes if the first solve fails).
        ``meta["attempts"]`` lists one dict per solve with ``fleet`` (vehicles available), ``removed_route_size``
        (``None`` for the first solve), ``dropped_count``, ``accepted``, ``seconds`` and ``warm_start``.
    """
    config = config or SolverConfig()
    t0 = time.perf_counter()
    matrix = exact(instance)
    penalty = drop_penalty_factor * fixed_cost_for(instance, config)
    attempts: list[dict] = []

    def elapsed() -> float:
        return time.perf_counter() - t0

    first_limit = min(first_fraction * total_time_s, total_time_s)
    first = solve(instance, replace(config, time_limit_s=max(first_limit, 0.001)))
    rep = verify(instance, first.routes, matrix) if first.routes else None
    ok = rep is not None and rep.feasible
    attempts.append({
        "fleet": first.meta["solver_vehicles"], "removed_route_size": None, "dropped_count": 0,
        "accepted": bool(ok), "seconds": round(first.runtime_s, 3), "warm_start": "none",
    })
    if not ok:
        meta = {**first.meta, "status": "NO_SOLUTION", "attempts": attempts}
        return Solution([], "ortools_minimize_fleet_v2", elapsed(), meta)
    best, best_rep = first, rep
    carry = max(0.0, first_limit - first.runtime_s)
    while True:
        order = sorted(range(len(best.routes)), key=lambda i: (len(best.routes[i]), i))
        improved = False
        for idx in order:
            remaining = total_time_s - elapsed()
            if remaining < 1.0 or best_rep.vehicles <= 1:
                break
            limit = min(step_time_s + carry, remaining)
            rest = [r for j, r in enumerate(best.routes) if j != idx]
            fleet = best_rep.vehicles - 1
            sol = solve(
                instance,
                replace(config, time_limit_s=limit, num_vehicles=fleet),
                initial_routes=rest,
                drop_penalty=penalty,
            )
            dropped = sol.meta["dropped"]
            new_rep = verify(instance, sol.routes, matrix) if (sol.routes and not dropped) else None
            accepted = new_rep is not None and new_rep.feasible and new_rep.vehicles < best_rep.vehicles
            attempts.append({
                "fleet": fleet, "removed_route_size": len(best.routes[idx]), "dropped_count": len(dropped),
                "accepted": bool(accepted), "seconds": round(sol.runtime_s, 3),
                "warm_start": sol.meta["warm_start"],
            })
            carry = max(0.0, limit - sol.runtime_s)
            if accepted:
                best, best_rep, improved = sol, new_rep, True
                break
        if not improved:
            break
    meta = {**best.meta, "attempts": attempts}
    return Solution(best.routes, "ortools_minimize_fleet_v2", elapsed(), meta)
