"""Deliberately broken plans, used to show what the independent verifier flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .distance import exact
from .io import Instance
from .labels import get_labels
from .verify import Report, verify

VARIANTS = ("original", "drop_customer", "duplicate_customer", "break_window", "break_capacity")


def _copy(routes: list[list[int]]) -> list[list[int]]:
    return [list(r) for r in routes if r]


def _break_window(instance: Instance, routes: list[list[int]], matrix: np.ndarray) -> list[list[int]]:
    """Move the customer with the earliest deadline to the end of another route, breaking only its window.

    The first pass also requires that no route returns late to the depot; if no such move exists the
    second pass accepts it.
    """
    order = sorted(range(1, instance.n_customers + 1), key=lambda c: (instance.due[c], c))
    for strict in (True, False):
        for c in order:
            home = next(i for i, r in enumerate(routes) if c in r)
            for target in range(len(routes)):
                if target == home:
                    continue
                trial = _copy(routes)
                trial[home].remove(c)
                trial[target].append(c)
                rep = verify(instance, trial, matrix)
                hit = any(v[0] == c for v in rep.window_violations)
                if hit and not rep.capacity_violations and not (strict and rep.depot_violations):
                    return trial
    raise ValueError("no customer can be moved so that only a time window is broken")


def _break_capacity(instance: Instance, routes: list[list[int]]) -> list[list[int]]:
    """Merge the two routes with the largest loads when their sum exceeds the capacity."""
    loads = [int(sum(instance.demand[c] for c in r)) for r in routes]
    first, second = sorted(range(len(routes)), key=lambda i: (-loads[i], i))[:2]
    if loads[first] + loads[second] <= instance.capacity:
        raise ValueError("the two largest routes fit together; no capacity violation can be created")
    merged = routes[first] + routes[second]
    return [r for i, r in enumerate(routes) if i not in (first, second)] + [merged]


def demo_violations(instance: Instance, routes: list[list[int]]) -> dict[str, list[list[int]]]:
    """Build a plan and four deliberately broken variants of it.

    Variants (keys of the result, in this order): ``original``; ``drop_customer`` (the first customer
    of the first route is removed); ``duplicate_customer`` (that same customer is also appended to the
    last route); ``break_window`` (the customer with the earliest deadline is moved to the end of another
    route, choosing the first move that violates its window without violating capacity);
    ``break_capacity`` (the two routes with the largest loads are merged).

    Parameters
    ----------
    instance : Instance
        The instance.
    routes : list of list of int
        A plan, typically a feasible one, with at least two routes.

    Returns
    -------
    dict
        Variant name -> routes. The input is not modified.

    Raises
    ------
    ValueError
        If the plan has fewer than two routes or a violation of the requested kind cannot be built.
    """
    base = _copy(routes)
    if len(base) < 2:
        raise ValueError("at least two non-empty routes are needed")
    matrix = exact(instance)
    victim = base[0][0]

    dropped = _copy(base)
    dropped[0].remove(victim)
    duplicated = _copy(base)
    duplicated[-1].append(victim)
    return {
        "original": base,
        "drop_customer": [r for r in dropped if r],
        "duplicate_customer": duplicated,
        "break_window": _break_window(instance, base, matrix),
        "break_capacity": _break_capacity(instance, base),
    }


def violation_row(report: Report) -> dict[str, object]:
    """Summarise a :class:`~delivery_routing_vrptw.verify.Report` as counts of each violation kind."""
    return {
        "feasible": report.feasible,
        "vehicles": report.vehicles,
        "n_unserved": len(report.unserved),
        "n_duplicates": len(report.duplicates),
        "n_window_violations": len(report.window_violations),
        "n_capacity_violations": len(report.capacity_violations),
        "n_depot_violations": len(report.depot_violations),
    }


def violation_table(
    instance: Instance, variants: dict[str, list[list[int]]], lang: str = "en"
) -> pd.DataFrame:
    """Verify each variant and tabulate which violations are flagged.

    Parameters
    ----------
    instance : Instance
        The instance.
    variants : dict
        Variant name -> routes, as returned by :func:`demo_violations`.
    lang : str
        Language of the variant names.

    Returns
    -------
    pandas.DataFrame
        One row per variant. Column names are keys of the ``columns`` label group
        (use :func:`delivery_routing_vrptw.eda.localize_columns` to translate them).
    """
    names = get_labels(lang)["variants"]
    matrix = exact(instance)
    rows = []
    for key, routes in variants.items():
        rows.append({"variant": names.get(key, key), **violation_row(verify(instance, routes, matrix))})
    return pd.DataFrame(rows)
