"""Container for a routing solution produced by any method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Solution:
    """A set of routes plus provenance.

    Attributes
    ----------
    routes : list of list of int
        Customer ids visited by each vehicle, in order, without the depot.
    method : str
        Name of the method that produced the solution.
    runtime_s : float
        Wall-clock time spent by the method, in seconds.
    meta : dict
        Free-form method metadata (configuration, solver status, ...).
    """

    routes: list[list[int]]
    method: str
    runtime_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_vehicles(self) -> int:
        """Return the number of non-empty routes."""
        return sum(1 for route in self.routes if route)
