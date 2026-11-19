import time

import pytest
from conftest import make_text

from delivery_routing_vrptw import (
    Solution,
    SolverConfig,
    allow_early,
    fixed_cost_for,
    minimize_fleet_v2,
    parse_instance,
    solve,
    verify,
    widen_windows,
)
from delivery_routing_vrptw import solver as solver_module
from delivery_routing_vrptw.scenarios import solve_ladder


def lex_ok(new, old, tol=0.05):
    """New (vehicles, distance) is lexicographically no worse than old, up to integer rounding."""
    return new.vehicles < old.vehicles or (new.vehicles == old.vehicles and new.distance <= old.distance + tol)


def test_fixed_cost_v1_default_unchanged(inst):
    assert fixed_cost_for(inst, SolverConfig()) == 10**7
    assert fixed_cost_for(inst, SolverConfig(vehicle_fixed_cost=123, scale=10000)) == 123
    assert solve(inst, SolverConfig(time_limit_s=0.5)).meta["vehicle_fixed_cost_effective"] == 10**7


def test_fixed_cost_k_scales_with_scale_and_k(inst):
    a = fixed_cost_for(inst, SolverConfig(fixed_cost_k=10, scale=100))
    b = fixed_cost_for(inst, SolverConfig(fixed_cost_k=10, scale=10000))
    c = fixed_cost_for(inst, SolverConfig(fixed_cost_k=100, scale=100))
    assert b == 100 * a and c == 10 * a
    d_ub = 2 * sum(((x - 0) ** 2 + (y - 0) ** 2) ** 0.5 for x, y in [(3, 4), (6, 8), (-3, 4), (-6, 8), (0, -5)])
    assert a == round(10 * 100 * d_ub)
    assert solve(inst, SolverConfig(time_limit_s=0.5, fixed_cost_k=10)).meta["vehicle_fixed_cost_effective"] == a


def test_fixed_cost_k_validation():
    with pytest.raises(ValueError):
        SolverConfig(fixed_cost_k=0)


def test_warm_start_never_worse(inst):
    cold = solve(inst, SolverConfig(time_limit_s=0.5))
    base = verify(inst, cold.routes)
    warm = solve(inst, SolverConfig(time_limit_s=0.5), initial_routes=cold.routes)
    assert warm.meta["warm_start"] == "used"
    assert cold.meta["warm_start"] == "none"
    rep = verify(inst, warm.routes)
    assert rep.feasible and lex_ok(rep, base)


def test_warm_start_from_weak_plan_improves(inst):
    singles = [[c] for c in range(1, 6)]
    wide = parse_instance(make_text(vehicles=6))
    base = verify(wide, singles)
    assert base.feasible and base.vehicles == 5
    warm = solve(wide, SolverConfig(time_limit_s=1), initial_routes=singles)
    assert warm.meta["warm_start"] == "used"
    rep = verify(wide, warm.routes)
    assert rep.feasible and lex_ok(rep, base) and rep.vehicles <= 3


def test_warm_start_rejected_falls_back_to_cold(inst):
    overloaded = solve(inst, SolverConfig(time_limit_s=0.5), initial_routes=[[1, 2, 3, 4, 5]])
    assert overloaded.meta["warm_start"] == "rejected"
    assert verify(inst, overloaded.routes).feasible
    too_many = solve(inst, SolverConfig(time_limit_s=0.5), initial_routes=[[1], [2], [3], [4]])
    assert too_many.meta["warm_start"] == "rejected"
    assert verify(inst, too_many.routes).feasible
    empty = solve(inst, SolverConfig(time_limit_s=0.5), initial_routes=[])
    assert empty.meta["warm_start"] == "rejected"


def test_drop_penalty_default_is_v1(inst):
    sol = solve(inst, SolverConfig(time_limit_s=0.5))
    assert sol.meta["dropped"] == []
    short = solve(inst, SolverConfig(time_limit_s=0.5, num_vehicles=2))
    assert short.routes == []  # v1: mandatory customers, infeasible with too few vehicles


def test_drop_penalty_drops_instead_of_failing(inst):
    sol = solve(inst, SolverConfig(time_limit_s=1, num_vehicles=2), drop_penalty=10**9)
    assert sol.routes and sol.meta["dropped"]
    served = {c for r in sol.routes for c in r}
    assert served.isdisjoint(sol.meta["dropped"])
    assert served | set(sol.meta["dropped"]) == {1, 2, 3, 4, 5}
    full = solve(inst, SolverConfig(time_limit_s=1), drop_penalty=10**9)
    assert full.meta["dropped"] == [] and verify(inst, full.routes).feasible


def test_minimize_fleet_v2_never_increases_and_is_feasible(inst):
    base = solve(inst, SolverConfig(time_limit_s=0.5))
    t0 = time.perf_counter()
    best = minimize_fleet_v2(inst, SolverConfig(), total_time_s=3, step_time_s=0.5)
    elapsed = time.perf_counter() - t0
    rep = verify(inst, best.routes)
    assert rep.feasible and rep.vehicles <= verify(inst, base.routes).vehicles
    assert elapsed < 3 + 1.5
    assert best.method == "ortools_minimize_fleet_v2"
    first = best.meta["attempts"][0]
    assert first["removed_route_size"] is None and first["accepted"]
    for a in best.meta["attempts"][1:]:
        assert set(a) == {"fleet", "removed_route_size", "dropped_count", "accepted", "seconds", "warm_start"}


def test_minimize_fleet_v2_accepts_reductions_from_weak_start(monkeypatch):
    wide = parse_instance(make_text(vehicles=6))
    singles = [[c] for c in range(1, 6)]
    real = solver_module.solve
    calls = {"n": 0}

    def fake(instance, config=None, initial_routes=None, drop_penalty=None):
        calls["n"] += 1
        if calls["n"] == 1:
            meta = {"solver_vehicles": 6, "warm_start": "none", "dropped": []}
            return Solution(singles, "ortools", 0.01, meta)
        return real(instance, config, initial_routes, drop_penalty)

    monkeypatch.setattr(solver_module, "solve", fake)
    best = minimize_fleet_v2(wide, SolverConfig(), total_time_s=8, first_fraction=0.1, step_time_s=0.5)
    rep = verify(wide, best.routes)
    assert rep.feasible and rep.vehicles == 3
    attempts = best.meta["attempts"]
    assert sum(a["accepted"] for a in attempts[1:]) >= 1
    assert all(a["warm_start"] == "used" for a in attempts[1:])
    assert attempts[-1]["accepted"] is False  # the 3 -> 2 reduction is impossible: tried, then stopped
    fleets = [a["fleet"] for a in attempts[1:]]
    assert fleets[0] == 4 and fleets[-1] == 2


def test_minimize_fleet_v2_no_first_solution(inst):
    best = minimize_fleet_v2(inst, SolverConfig(num_vehicles=2), total_time_s=2, step_time_s=0.5)
    assert best.routes == [] and best.meta["status"] == "NO_SOLUTION"


@pytest.mark.parametrize("kind", ["windows", "early"])
def test_ladder_never_lexicographically_worse(inst, kind):
    if kind == "windows":
        levels = [widen_windows(inst, f) for f in (1.0, 1.5, 2.0)]
    else:
        levels = [allow_early(inst, t) for t in (0.0, 0.5, float("inf"))]
    sols = solve_ladder(levels, SolverConfig(), time_limit_s=0.5)
    assert [s.meta["warm_start"] for s in sols] == ["none", "used", "used"]
    for i in range(1, len(levels)):
        prev_on_new = verify(levels[i], sols[i - 1].routes)
        now = verify(levels[i], sols[i].routes)
        assert prev_on_new.feasible and now.feasible
        assert lex_ok(now, prev_on_new)
