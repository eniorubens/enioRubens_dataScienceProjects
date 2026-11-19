from dataclasses import replace

import numpy as np
import pytest

from delivery_routing_vrptw import exact, optimal_departures, schedule_metrics, verify
from delivery_routing_vrptw.robustness import (
    ShockScenario,
    monte_carlo,
    nominal_slack,
    protocol_scenarios,
    simulate_execution,
)

ROUTES = [[1, 2], [3, 5], [4]]


def tight(inst):
    """Same instance with due = nominal start rounded up (slack below one time unit) for every visit."""
    ex = simulate_execution(inst, ROUTES, 1.0, 1.0)
    due = inst.due.copy()
    due[ex.customers] = np.ceil(ex.start[0] - 1e-9).astype(due.dtype)  # integer due dates keep the plan feasible
    return replace(inst, due=due)


def test_plan_is_feasible(inst):
    assert verify(inst, ROUTES).feasible


def test_departures_match_schedule_metrics(inst):
    deps = optimal_departures(inst, ROUTES)
    assert len(deps) == 3 and all(d >= 0 for d in deps)
    assert sum(deps) == pytest.approx(schedule_metrics(inst, ROUTES, exact(inst))["total_delay"])
    assert optimal_departures(inst, [[], [1]]) == optimal_departures(inst, [[1]])


def test_execution_hand_calculation(inst):
    # route [1]: d(0,1) = 5, ready 0, service 2; travel x2 -> arrival 10, leaves at 12, back at 22
    ex = simulate_execution(inst, [[1]], 2.0, 1.0, departures=[0.0])
    assert ex.start[0, 0] == pytest.approx(10.0) and ex.returns[0, 0] == pytest.approx(22.0)
    assert ex.lateness[0, 0] == 0.0
    ex = simulate_execution(inst, [[1]], 1.0, 3.0, departures=[0.0])  # service x3 = 6
    assert ex.returns[0, 0] == pytest.approx(5.0 + 6.0 + 5.0)


def test_late_customer_is_served_anyway_and_delay_propagates(inst):
    tight_inst = replace(inst, due=np.array([100, 5, 60, 70, 80, 90]))  # customer 1 due at 5 (nominal start 5)
    ex = simulate_execution(tight_inst, [[1, 2]], 2.0, 1.0, departures=[0.0])
    assert ex.lateness[0, 0] == pytest.approx(5.0)  # arrival 10 > due 5, served anyway
    assert ex.start[0, 1] == pytest.approx(10.0 + 2.0 + 10.0)  # next visit starts after the late service


def test_waits_until_ready(inst):
    ex = simulate_execution(inst, [[3]], 1.0, 1.0, departures=[0.0])
    assert ex.start[0, 0] == pytest.approx(20.0)  # arrival 5 < ready 20


def test_zero_shock_gives_zero_lateness(inst):
    for sc in (ShockScenario("none"), ShockScenario("zero", travel_a=0.0, service_b=0.0)):
        res = monte_carlo(inst, ROUTES, sc, 20, 7)
        assert res.metrics["late_customer_share"] == 0.0
        assert res.metrics["routes_with_late_share"] == 0.0 and res.metrics["late_return_share"] == 0.0
        assert res.metrics["nominal_late_customers"] == 0
        assert np.isnan(res.metrics["mean_lateness_late"]) and np.isnan(res.metrics["p95_lateness_late"])
        assert (res.late_rate == 0).all()


def test_tight_plan_is_late_under_a_large_shock(inst):
    tinst = tight(inst)
    assert verify(tinst, ROUTES).feasible
    res = monte_carlo(tinst, ROUTES, ShockScenario("t", travel_a=0.2), 200, 3)
    assert res.metrics["late_customer_share"] > 0 and res.metrics["nominal_late_customers"] == 0
    assert 0.0 <= res.metrics["nominal_slack_min"] < 1.0
    assert res.metrics["mean_lateness_late"] > 0 and res.metrics["p95_lateness_late"] >= res.metrics["mean_lateness_late"]


def test_monotone_in_travel_shock(inst):
    tinst = tight(inst)
    results = [monte_carlo(tinst, ROUTES, ShockScenario("t", travel_a=a), 500, 11).metrics for a in (0.05, 0.10, 0.20)]
    for key in ("late_customer_share", "routes_with_late_share", "mean_lateness_all", "late_return_share"):
        values = [m[key] for m in results]
        assert values == sorted(values), key
    assert results[-1]["mean_lateness_all"] > results[0]["mean_lateness_all"]


def test_service_shock_adds_lateness(inst):
    tinst = tight(inst)
    base = monte_carlo(tinst, ROUTES, ShockScenario("t", travel_a=0.1), 300, 5).metrics
    both = monte_carlo(tinst, ROUTES, ShockScenario("ts", travel_a=0.1, service_b=0.2), 300, 5).metrics
    assert both["mean_lateness_all"] >= base["mean_lateness_all"]
    assert monte_carlo(tinst, ROUTES, ShockScenario("s", service_b=0.2), 300, 5).metrics["late_customer_share"] > 0


def test_reproducible_with_seed(inst):
    tinst = tight(inst)
    sc = ShockScenario("ts", travel_a=0.1, service_b=0.2)
    a = monte_carlo(tinst, ROUTES, sc, 100, [1, 2, 3])
    b = monte_carlo(tinst, ROUTES, sc, 100, [1, 2, 3])
    c = monte_carlo(tinst, ROUTES, sc, 100, [1, 2, 4])
    assert a.metrics == b.metrics and np.array_equal(a.late_rate, b.late_rate)
    assert a.metrics["mean_lateness_all"] != c.metrics["mean_lateness_all"]


def test_nominal_slack_and_validation(inst):
    customers, slack = nominal_slack(inst, ROUTES)
    assert sorted(customers.tolist()) == [1, 2, 3, 4, 5] and (slack >= 0).all()
    with pytest.raises(ValueError):
        monte_carlo(inst, ROUTES, ShockScenario("x"), 0, 1)
    with pytest.raises(ValueError):
        ShockScenario("bad", travel_a=-0.1)
    with pytest.raises(ValueError):
        simulate_execution(inst, ROUTES, np.ones((4, 3)), 1.0)
    with pytest.raises(ValueError):
        simulate_execution(inst, ROUTES, 1.0, 1.0, departures=[0.0])


def test_protocol_scenarios():
    sc = protocol_scenarios()
    assert [(s.travel_a, s.service_b) for s in sc] == [
        (0.05, 0.0), (0.10, 0.0), (0.20, 0.0), (0.0, 0.2), (0.10, 0.2)
    ]
