import pytest

from delivery_routing_vrptw import SolverConfig, minimize_fleet, solve, verify


def test_solve_feasible(inst):
    sol = solve(inst, SolverConfig(time_limit_s=1))
    rep = verify(inst, sol.routes)
    assert rep.feasible
    assert rep.vehicles == 3
    assert sol.method == "ortools"
    assert sol.meta["status"] == "ROUTING_SUCCESS"
    assert sol.meta["first_solution"] == "PARALLEL_CHEAPEST_INSERTION"
    assert sol.runtime_s > 0


def test_default_config(inst):
    sol = solve(inst, SolverConfig(time_limit_s=0.5, log_search=False))
    assert sol.routes


def test_no_metaheuristic(inst):
    cfg = SolverConfig(metaheuristic="NONE", first_solution="PATH_CHEAPEST_ARC", time_limit_s=1)
    assert verify(inst, solve(inst, cfg).routes).feasible


def test_infeasible_fleet_returns_empty(inst):
    sol = solve(inst, SolverConfig(num_vehicles=2, time_limit_s=1))
    assert sol.routes == []
    assert sol.meta["status"] != "ROUTING_SUCCESS"


def test_invalid_config():
    with pytest.raises(ValueError):
        SolverConfig(first_solution="NOPE")
    with pytest.raises(ValueError):
        SolverConfig(metaheuristic="NOPE")
    with pytest.raises(ValueError):
        SolverConfig(time_limit_s=0)
    with pytest.raises(ValueError):
        SolverConfig(num_vehicles=0)


def test_minimize_fleet(inst):
    base = solve(inst, SolverConfig(time_limit_s=1))
    best = minimize_fleet(inst, SolverConfig(time_limit_s=1))
    assert verify(inst, best.routes).feasible
    assert best.num_vehicles <= base.num_vehicles
    assert best.meta["attempts"][-1][1] is False
    assert best.method == "ortools_minimize_fleet"


def test_minimize_fleet_no_solution(inst):
    best = minimize_fleet(inst, SolverConfig(time_limit_s=1), start_vehicles=2, step_time_s=0.5)
    assert best.routes == [] and best.meta["status"] == "NO_SOLUTION"


def test_minimize_fleet_total_budget(inst):
    import time

    t0 = time.perf_counter()
    best = minimize_fleet(
        inst, SolverConfig(time_limit_s=5), step_time_s=1, total_time_s=3, first_step_time_s=1
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 3 + 1.5
    assert verify(inst, best.routes).feasible
    vehicles = [a[2] for a in best.meta["attempts"] if a[1]]
    assert vehicles == sorted(vehicles, reverse=True)
    assert vehicles[0] <= inst.vehicle_limit


def test_minimize_fleet_budget_too_small(inst):
    best = minimize_fleet(inst, SolverConfig(time_limit_s=1), total_time_s=0.5)
    assert best.routes == [] and best.meta["attempts"] == []
