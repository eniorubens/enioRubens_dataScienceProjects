"""Independent double-precision feasibility checker and metric calculator."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .distance import exact
from .io import Instance

TOL = 1e-9


@dataclass
class Report:
    """Result of verifying a set of routes against an instance.

    Attributes
    ----------
    feasible : bool
        ``True`` when no violation of any kind was found.
    vehicles : int
        Number of non-empty routes.
    distance : float
        Total travelled distance (``math.fsum``, unrounded).
    duration : float
        Sum over routes of the return time at the depot minus departure time 0
        (travel + waiting + service).
    waiting : float
        Total waiting time before windows open.
    unserved : list of int
        Sorted customers that are not visited.
    duplicates : list of int
        Customers visited more than once.
    unknown : list of int
        Ids outside ``1..n``.
    capacity_violations : list of tuple
        ``(route_idx, load)`` for overloaded routes.
    window_violations : list of tuple
        ``(customer, start_time, due)`` for late service starts.
    depot_violations : list of int
        Indices of routes returning after the depot horizon.
    too_many_vehicles : bool
        ``True`` when ``vehicles`` exceeds the instance vehicle limit.
    """

    feasible: bool
    vehicles: int
    distance: float
    duration: float
    waiting: float
    unserved: list[int] = field(default_factory=list)
    duplicates: list[int] = field(default_factory=list)
    unknown: list[int] = field(default_factory=list)
    capacity_violations: list[tuple[int, int]] = field(default_factory=list)
    window_violations: list[tuple[int, float, float]] = field(default_factory=list)
    depot_violations: list[int] = field(default_factory=list)
    too_many_vehicles: bool = False


def verify(instance: Instance, routes: list[list[int]], matrix: np.ndarray | None = None) -> Report:
    """Verify routes against all constraints of the rigid VRPTW.

    Parameters
    ----------
    instance : Instance
        The instance.
    routes : list of list of int
        Customer ids per route, without the depot. Empty routes are ignored.
    matrix : numpy.ndarray, optional
        Travel-time/distance matrix; defaults to :func:`delivery_routing_vrptw.distance.exact`.

    Returns
    -------
    Report
        Feasibility flags and metrics. Unknown ids are reported and skipped in the schedule.
    """
    if matrix is None:
        matrix = exact(instance)
    n = instance.n_customers
    horizon = float(instance.due[0])
    routes = [list(r) for r in routes if r]

    counts: dict[int, int] = {}
    for route in routes:
        for c in route:
            counts[c] = counts.get(c, 0) + 1
    unknown = sorted(c for c in counts if not 1 <= c <= n)
    duplicates = sorted(c for c, k in counts.items() if k > 1 and 1 <= c <= n)
    unserved = sorted(set(range(1, n + 1)) - set(counts))

    capacity_violations: list[tuple[int, int]] = []
    window_violations: list[tuple[int, float, float]] = []
    depot_violations: list[int] = []
    legs: list[float] = []
    duration = 0.0
    waiting = 0.0
    for idx, route in enumerate(routes):
        valid = [c for c in route if 1 <= c <= n]
        load = int(sum(instance.demand[c] for c in valid))
        if load > instance.capacity:
            capacity_violations.append((idx, load))
        t, prev = 0.0, 0
        for c in valid:
            legs.append(float(matrix[prev, c]))
            arrival = t + float(matrix[prev, c])
            start = max(arrival, float(instance.ready[c]))
            waiting += start - arrival
            if start > instance.due[c] + TOL:
                window_violations.append((c, float(start), float(instance.due[c])))
            t = start + float(instance.service[c])
            prev = c
        legs.append(float(matrix[prev, 0]))
        back = t + float(matrix[prev, 0])
        duration += back
        if back > horizon + TOL:
            depot_violations.append(idx)

    too_many = len(routes) > instance.vehicle_limit
    feasible = not (
        unserved or duplicates or unknown or capacity_violations or window_violations
        or depot_violations or too_many
    )
    return Report(
        feasible=feasible, vehicles=len(routes), distance=math.fsum(legs), duration=duration,
        waiting=waiting, unserved=unserved, duplicates=duplicates, unknown=unknown,
        capacity_violations=capacity_violations, window_violations=window_violations,
        depot_violations=depot_violations, too_many_vehicles=too_many,
    )


def report_to_dict(report: Report) -> dict:
    """Return the report as a plain dictionary."""
    return asdict(report)


@dataclass
class LatenessReport:
    """Result of measuring (not enforcing) the customer due dates of a set of routes.

    Attributes
    ----------
    hard_feasible : bool
        ``True`` when every customer is served exactly once, no route is overloaded, every route returns
        by the horizon and the fleet limit holds; due dates of customers are *not* part of this flag.
    vehicles : int
        Number of non-empty routes.
    distance : float
        Total travelled distance.
    late : list of tuple
        ``(customer, lateness)`` for service starts after ``due`` (earliest-start schedule, late service is
        performed anyway).
    n_late : int
        Number of late customers.
    total_lateness, max_lateness : float
        Sum and maximum of the lateness over late customers (0 when none).
    unserved, duplicates, unknown : list of int
        As in :class:`Report`.
    capacity_violations : list of tuple
        ``(route_idx, load)`` for overloaded routes.
    depot_violations : list of int
        Indices of routes returning after the horizon.
    too_many_vehicles : bool
        ``True`` when ``vehicles`` exceeds the instance vehicle limit.
    """

    hard_feasible: bool
    vehicles: int
    distance: float
    late: list[tuple[int, float]] = field(default_factory=list)
    n_late: int = 0
    total_lateness: float = 0.0
    max_lateness: float = 0.0
    unserved: list[int] = field(default_factory=list)
    duplicates: list[int] = field(default_factory=list)
    unknown: list[int] = field(default_factory=list)
    capacity_violations: list[tuple[int, int]] = field(default_factory=list)
    depot_violations: list[int] = field(default_factory=list)
    too_many_vehicles: bool = False


def measure_lateness(instance: Instance, routes: list[list[int]], matrix: np.ndarray | None = None) -> LatenessReport:
    """Measure the lateness of a plan instead of failing on it (soft time windows).

    The schedule is the earliest-start one of :func:`verify`: depart the depot at 0, wait until ``ready``,
    serve even when the start is after ``due`` and continue. All other constraints (coverage, capacity,
    return by the horizon, fleet limit) are checked as in :func:`verify`; :func:`verify` itself is unchanged.

    Parameters
    ----------
    instance : Instance
        The instance (original due dates).
    routes : list of list of int
        Customer ids per route, without the depot. Empty routes are ignored.
    matrix : numpy.ndarray, optional
        Travel-time/distance matrix; defaults to :func:`delivery_routing_vrptw.distance.exact`.

    Returns
    -------
    LatenessReport
        Hard-constraint flag, metrics and the late customers with their lateness (start minus due).
    """
    if matrix is None:
        matrix = exact(instance)
    base = verify(instance, routes, matrix)
    n = instance.n_customers
    late: list[tuple[int, float]] = []
    for route in (list(r) for r in routes if r):
        t, prev = 0.0, 0
        for c in (c for c in route if 1 <= c <= n):
            start = max(t + float(matrix[prev, c]), float(instance.ready[c]))
            if start > instance.due[c] + TOL:
                late.append((c, float(start - instance.due[c])))
            t = start + float(instance.service[c])
            prev = c
    hard = not (
        base.unserved or base.duplicates or base.unknown or base.capacity_violations or base.depot_violations
        or base.too_many_vehicles
    )
    lateness = [x for _, x in late]
    return LatenessReport(
        hard_feasible=hard, vehicles=base.vehicles, distance=base.distance, late=late, n_late=len(late),
        total_lateness=math.fsum(lateness), max_lateness=max(lateness, default=0.0), unserved=base.unserved,
        duplicates=base.duplicates, unknown=base.unknown, capacity_violations=base.capacity_violations,
        depot_violations=base.depot_violations, too_many_vehicles=base.too_many_vehicles,
    )
