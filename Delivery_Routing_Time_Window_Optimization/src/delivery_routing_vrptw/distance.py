"""Distance matrices under the conventions found in the VRPTW literature.

``exact`` is the SINTEF convention (double precision). ``truncated`` is the CVRPLIB/exact-method
convention (truncated to one decimal). ``scaled_ceil`` is the integer matrix fed to OR-Tools: rounding
travel times *up* guarantees that a schedule feasible for the solver is feasible in double precision.
"""

from __future__ import annotations

import math

import numpy as np

from .io import Instance


def exact(instance: Instance) -> np.ndarray:
    """Return the Euclidean distance matrix in double precision."""
    dx = instance.x[:, None] - instance.x[None, :]
    dy = instance.y[:, None] - instance.y[None, :]
    return np.hypot(dx, dy)


def truncated(instance: Instance, decimals: int = 1) -> np.ndarray:
    """Return distances truncated to ``decimals`` places (CVRPLIB convention)."""
    factor = 10**decimals
    return np.floor(exact(instance) * factor) / factor


def scaled_ceil(instance: Instance, scale: int) -> np.ndarray:
    """Return integer distances ``ceil(d * scale)`` for solver time dimensions.

    A tiny tolerance avoids turning an exact integer product into the next integer due to
    floating-point noise.
    """
    return np.ceil(exact(instance) * scale - 1e-9).astype(np.int64)


def scaled_round(instance: Instance, scale: int) -> np.ndarray:
    """Return integer distances ``round(d * scale)`` for solver arc costs."""
    return np.rint(exact(instance) * scale).astype(np.int64)


def route_length(route: list[int], matrix: np.ndarray) -> float:
    """Return the length of ``depot -> route -> depot`` under ``matrix``."""
    if not route:
        return 0.0
    path = [0, *route, 0]
    return float(math.fsum(matrix[a, b] for a, b in zip(path, path[1:], strict=False)))
