"""Time-window-aware constructive baselines (deterministic)."""

from __future__ import annotations

import time

import numpy as np

from .distance import exact
from .io import Instance
from .solution import Solution

_TOL = 1e-9


def _check_servable(instance: Instance, d: np.ndarray) -> None:
    for c in range(1, instance.n_customers + 1):
        start = max(d[0, c], float(instance.ready[c]))
        if start > instance.due[c] + _TOL or start + instance.service[c] + d[c, 0] > instance.due[0] + _TOL:
            raise ValueError(f"customer {c} cannot be served even alone")


def greedy_deadline(instance: Instance) -> Solution:
    """Earliest-deadline-first sequential greedy.

    Customers are sorted by ``(due, ready, id)``; each is appended to the end of the first existing
    route that stays feasible (capacity, window, return to depot), otherwise a new route is opened.

    Returns
    -------
    Solution
        Method name ``"greedy_deadline"``.
    """
    t0 = time.perf_counter()
    d = exact(instance)
    _check_servable(instance, d)
    order = sorted(
        range(1, instance.n_customers + 1), key=lambda c: (instance.due[c], instance.ready[c], c)
    )
    routes: list[list[int]] = []
    loads: list[int] = []
    last: list[int] = []
    free: list[float] = []  # time at which the vehicle can leave its last node
    for c in order:
        placed = False
        for r in range(len(routes)):
            if loads[r] + instance.demand[c] > instance.capacity:
                continue
            start = max(free[r] + d[last[r], c], float(instance.ready[c]))
            if start > instance.due[c] + _TOL:
                continue
            if start + instance.service[c] + d[c, 0] > instance.due[0] + _TOL:
                continue
            routes[r].append(c)
            loads[r] += int(instance.demand[c])
            last[r] = c
            free[r] = start + float(instance.service[c])
            placed = True
            break
        if not placed:
            start = max(d[0, c], float(instance.ready[c]))
            routes.append([c])
            loads.append(int(instance.demand[c]))
            last.append(c)
            free.append(start + float(instance.service[c]))
    return Solution(routes, "greedy_deadline", time.perf_counter() - t0)


def nearest_neighbor_tw(
    instance: Instance, weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
) -> Solution:
    """Solomon (1987) time-oriented nearest-neighbour heuristic.

    Routes are built one at a time. From the last node ``i`` the next customer ``j`` minimises
    ``w1*d_ij + w2*T_ij + w3*v_ij`` with ``T_ij`` the time difference between the start of service at
    ``j`` and the end of service at ``i``, and ``v_ij`` the urgency ``due_j - (end_i + d_ij)``, among
    customers that keep capacity, window and depot return feasible. Ties go to the lowest id.

    Parameters
    ----------
    instance : Instance
        The instance.
    weights : tuple of float
        ``(w1, w2, w3)`` for distance, waiting-time difference and urgency.

    Returns
    -------
    Solution
        Method name ``"nearest_neighbor_tw"``.
    """
    t0 = time.perf_counter()
    w1, w2, w3 = weights
    d = exact(instance)
    _check_servable(instance, d)
    unrouted = set(range(1, instance.n_customers + 1))
    routes: list[list[int]] = []
    while unrouted:
        route: list[int] = []
        node, leave, load = 0, 0.0, 0
        while True:
            best, best_cost, best_start = None, np.inf, 0.0
            for j in sorted(unrouted):
                if load + instance.demand[j] > instance.capacity:
                    continue
                start = max(leave + d[node, j], float(instance.ready[j]))
                if start > instance.due[j] + _TOL:
                    continue
                if start + instance.service[j] + d[j, 0] > instance.due[0] + _TOL:
                    continue
                cost = (
                    w1 * d[node, j] + w2 * (start - leave) + w3 * (instance.due[j] - (leave + d[node, j]))
                )
                if cost < best_cost - 1e-12:
                    best, best_cost, best_start = j, cost, start
            if best is None:
                break
            route.append(best)
            unrouted.discard(best)
            load += int(instance.demand[best])
            node, leave = best, best_start + float(instance.service[best])
        if not route:  # pragma: no cover - guarded by _check_servable
            raise ValueError("no customer can start a new route")
        routes.append(route)
    return Solution(routes, "nearest_neighbor_tw", time.perf_counter() - t0)
