"""Robustness of fixed plans under random travel and service delays (phase C1, no solver).

A plan is executed as planned (same visiting sequence, no re-planning) while travel times and service times
are inflated by random factors. A vehicle that arrives after ``due`` is recorded as late and still serves
(see ``reports/protocolo-fase-c.md``). Draws are vectorised over the Monte Carlo dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distance import exact
from .io import Instance
from .route_compare import optimal_departures

TOL = 1e-9
SERVICE_B = 0.20
TRAVEL_A = (0.05, 0.10, 0.20)
COMBINED_A = 0.10


@dataclass(frozen=True)
class ShockScenario:
    """A pair of independent multiplicative shocks.

    Attributes
    ----------
    name : str
        Scenario label.
    travel_a : float
        Per-leg travel time is multiplied by ``1 + eps`` with ``eps ~ U(0, travel_a)``; 0 disables it.
    service_b : float
        Per-customer service time is multiplied by ``1 + eta`` with ``eta ~ U(0, service_b)``; 0 disables it.
    """

    name: str
    travel_a: float = 0.0
    service_b: float = 0.0

    def __post_init__(self) -> None:
        """Validate the ranges."""
        if self.travel_a < 0 or self.service_b < 0:
            raise ValueError("travel_a and service_b must be >= 0")


def protocol_scenarios() -> tuple[ShockScenario, ...]:
    """Return the five scenarios of the phase C1 protocol (travel 5/10/20%, service, travel 10% + service)."""
    travel = tuple(ShockScenario(f"travel_{a:.2f}", travel_a=a) for a in TRAVEL_A)
    return (
        *travel,
        ShockScenario("service", service_b=SERVICE_B),
        ShockScenario(f"travel_{COMBINED_A:.2f}+service", travel_a=COMBINED_A, service_b=SERVICE_B),
    )


@dataclass(frozen=True)
class Execution:
    """Outcome of executing a fixed plan for several draws.

    Attributes
    ----------
    customers : numpy.ndarray
        Customer visited at each position (plan order, routes concatenated), shape ``(V,)``.
    route_of : numpy.ndarray
        Index of the route of each position, shape ``(V,)``.
    start : numpy.ndarray
        Service start times, shape ``(D, V)``.
    lateness : numpy.ndarray
        ``max(0, start - due)``, shape ``(D, V)``.
    returns : numpy.ndarray
        Depot return time of each route, shape ``(D, R)``.
    horizon : float
        End of the planning horizon.
    """

    customers: np.ndarray
    route_of: np.ndarray
    start: np.ndarray
    lateness: np.ndarray
    returns: np.ndarray
    horizon: float


def _factor(value, n_cols: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full((1, n_cols), float(arr))
    elif arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != n_cols:
        raise ValueError(f"{name} must be a scalar or have {n_cols} columns, got shape {arr.shape}")
    return arr


def simulate_execution(
    instance: Instance,
    routes: list[list[int]],
    travel_factor,
    service_factor,
    departures: list[float] | None = None,
    matrix: np.ndarray | None = None,
) -> Execution:
    """Execute a fixed plan under multiplicative delays, without re-planning.

    Every vehicle leaves the depot at its nominal optimal departure and follows the planned sequence: the
    service starts at ``max(arrival, ready)``; an arrival after ``due`` is recorded as lateness and the
    customer is served anyway, so the delay propagates down the route.

    Parameters
    ----------
    instance : Instance
        The instance.
    routes : list of list of int
        Customer ids per route, without the depot. Empty routes are ignored.
    travel_factor : float or numpy.ndarray
        Multiplier of every leg time (``depot -> c1``, ..., ``cm -> depot`` of each route, routes
        concatenated, ``sum(len(r) + 1)`` legs): a scalar, a vector (one draw) or an array ``(D, legs)``.
    service_factor : float or numpy.ndarray
        Multiplier of every service time, in plan order: a scalar, a vector or an array ``(D, visits)``.
    departures : list of float, optional
        Depot departure of each non-empty route; defaults to the nominal Savelsbergh optimal departures
        (:func:`delivery_routing_vrptw.route_compare.optimal_departures`).
    matrix : numpy.ndarray, optional
        Travel-time matrix; defaults to the exact Euclidean one.

    Returns
    -------
    Execution
        Start times, lateness and return times for every draw.
    """
    if matrix is None:
        matrix = exact(instance)
    plan = [list(r) for r in routes if r]
    if departures is None:
        departures = optimal_departures(instance, plan, matrix)
    if len(departures) != len(plan):
        raise ValueError("departures needs one value per non-empty route")
    n_visits = sum(len(r) for r in plan)
    n_legs = n_visits + len(plan)
    tf = _factor(travel_factor, n_legs, "travel_factor")
    sf = _factor(service_factor, n_visits, "service_factor")
    draws = max(tf.shape[0], sf.shape[0])
    if tf.shape[0] not in (1, draws) or sf.shape[0] not in (1, draws):
        raise ValueError("travel_factor and service_factor must have the same number of draws")
    start = np.zeros((draws, n_visits))
    late = np.zeros((draws, n_visits))
    returns = np.zeros((draws, len(plan)))
    customers = np.zeros(n_visits, dtype=int)
    route_of = np.zeros(n_visits, dtype=int)
    leg = visit = 0
    for r, route in enumerate(plan):
        t = np.full(draws, float(departures[r]))
        prev = 0
        for c in route:
            arrival = t + matrix[prev, c] * tf[:, leg]
            begin = np.maximum(arrival, float(instance.ready[c]))
            start[:, visit] = begin
            late[:, visit] = np.maximum(begin - float(instance.due[c]), 0.0)
            customers[visit], route_of[visit] = c, r
            t = begin + float(instance.service[c]) * sf[:, visit]
            prev, leg, visit = c, leg + 1, visit + 1
        returns[:, r] = t + matrix[prev, 0] * tf[:, leg]
        leg += 1
    return Execution(customers, route_of, start, late, returns, float(instance.due[0]))


def nominal_slack(
    instance: Instance, routes: list[list[int]], matrix: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nominal slack ``due - start`` of every visit when the plan is executed without shocks.

    The nominal execution already uses the optimal departures, so the slack is the one the executed plan
    really has (the critical customer of a route without waiting has slack 0).

    Parameters
    ----------
    instance : Instance
        The instance.
    routes : list of list of int
        Customer ids per route, without the depot.
    matrix : numpy.ndarray, optional
        Travel-time matrix.

    Returns
    -------
    tuple of numpy.ndarray
        ``(customers, slack)`` in plan order.
    """
    ex = simulate_execution(instance, routes, 1.0, 1.0, matrix=matrix)
    slack = instance.due[ex.customers].astype(float) - ex.start[0]
    return ex.customers, np.where(np.abs(slack) < TOL, 0.0, slack)


@dataclass(frozen=True)
class RobustnessResult:
    """Monte Carlo outcome of one plan under one scenario.

    Attributes
    ----------
    metrics : dict
        ``late_customer_share`` (share of customer-draws served late), ``routes_with_late_share`` (share of
        route-draws with at least one late customer), ``draws_with_late_share`` (share of draws with any late
        customer), ``mean_lateness_late`` and ``p95_lateness_late`` (over the pooled late events; NaN when
        there are none), ``mean_lateness_all`` (over all customer-draws), ``late_return_share`` (share of
        route-draws returning after the horizon), the nominal check ``nominal_late_customers`` (should be 0
        for a feasible plan), ``nominal_slack_mean``/``nominal_slack_min`` and the sizes.
    customers : numpy.ndarray
        Customer of each position (plan order).
    slack : numpy.ndarray
        Nominal slack per position.
    late_rate : numpy.ndarray
        Share of draws in which each position was late.
    """

    metrics: dict
    customers: np.ndarray
    slack: np.ndarray
    late_rate: np.ndarray


def monte_carlo(
    instance: Instance,
    routes: list[list[int]],
    scenario: ShockScenario,
    n_draws: int,
    seed: int | list[int],
    matrix: np.ndarray | None = None,
) -> RobustnessResult:
    """Run the Monte Carlo robustness experiment for one plan and one scenario.

    The uniform variates are drawn in a fixed order (all legs, then all services) from
    ``numpy.random.default_rng(seed)`` whatever the scenario, and scaled by ``travel_a`` / ``service_b``.
    Two scenarios run with the same seed therefore share random numbers, so the lateness is monotone in
    ``travel_a`` draw by draw, and the result is deterministic given the seed.

    Parameters
    ----------
    instance : Instance
        The instance.
    routes : list of list of int
        Plan to execute (customer ids per route, without the depot).
    scenario : ShockScenario
        Shock sizes.
    n_draws : int
        Number of Monte Carlo draws (``>= 1``).
    seed : int or list of int
        Seed of the random generator.
    matrix : numpy.ndarray, optional
        Travel-time matrix.

    Returns
    -------
    RobustnessResult
        Metrics of the protocol plus the per-customer late rate and nominal slack.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    if matrix is None:
        matrix = exact(instance)
    plan = [list(r) for r in routes if r]
    n_visits = sum(len(r) for r in plan)
    rng = np.random.default_rng(seed)
    u_travel = rng.random((n_draws, n_visits + len(plan)))
    u_service = rng.random((n_draws, n_visits))
    departures = optimal_departures(instance, plan, matrix)
    ex = simulate_execution(
        instance, plan, 1.0 + scenario.travel_a * u_travel, 1.0 + scenario.service_b * u_service,
        departures=departures, matrix=matrix,
    )
    nominal = simulate_execution(instance, plan, 1.0, 1.0, departures=departures, matrix=matrix)
    slack = instance.due[ex.customers].astype(float) - nominal.start[0]
    slack = np.where(np.abs(slack) < TOL, 0.0, slack)  # a critical customer has slack 0, not -1e-14
    is_late = ex.lateness > TOL
    pooled = ex.lateness[is_late]
    route_late = np.zeros((n_draws, len(plan)), dtype=bool)
    for r in range(len(plan)):
        route_late[:, r] = is_late[:, ex.route_of == r].any(axis=1)
    metrics = {
        "n_customers": n_visits,
        "n_routes": len(plan),
        "n_draws": n_draws,
        "late_customer_share": float(is_late.mean()) if n_visits else 0.0,
        "routes_with_late_share": float(route_late.mean()) if plan else 0.0,
        "draws_with_late_share": float(is_late.any(axis=1).mean()),
        "mean_lateness_late": float(pooled.mean()) if pooled.size else float("nan"),
        "p95_lateness_late": float(np.percentile(pooled, 95)) if pooled.size else float("nan"),
        "mean_lateness_all": float(ex.lateness.mean()) if n_visits else 0.0,
        "late_return_share": float((ex.returns > ex.horizon + TOL).mean()) if plan else 0.0,
        "nominal_late_customers": int((nominal.lateness[0] > TOL).sum()),
        "nominal_slack_mean": float(slack.mean()) if n_visits else float("nan"),
        "nominal_slack_min": float(slack.min()) if n_visits else float("nan"),
    }
    return RobustnessResult(metrics, ex.customers, slack, is_late.mean(axis=0))
