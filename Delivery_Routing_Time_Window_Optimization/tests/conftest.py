"""Shared synthetic fixtures (no real data, no network)."""

from __future__ import annotations

import os

# Work around a dgemm crash (0xc06d007f) of conda MKL 2026.1 + llvm-openmp 23 on Windows.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pytest  # noqa: E402

from delivery_routing_vrptw import Instance, parse_instance  # noqa: E402

# id x y demand ready due service. Capacity 10, 3 vehicles, depot horizon 100.
ROWS = [
    (0, 0, 0, 0, 0, 100, 0),
    (1, 3, 4, 5, 0, 50, 2),
    (2, 6, 8, 5, 0, 60, 2),
    (3, -3, 4, 6, 20, 70, 2),
    (4, -6, 8, 5, 0, 80, 2),
    (5, 0, -5, 4, 30, 90, 0),
]


def make_text(name: str = "R101", rows=ROWS, vehicles: int = 3, capacity: int = 10) -> str:
    """Build Solomon-format text."""
    head = [
        name, "", "VEHICLE", "NUMBER     CAPACITY", f"  {vehicles}    {capacity}", "", "CUSTOMER",
        "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME", "",
    ]
    body = ["  ".join(str(v) for v in row) for row in rows]
    return "\n".join(head + body) + "\n"


@pytest.fixture
def text() -> str:
    """Return the tiny instance text."""
    return make_text()


@pytest.fixture
def inst(text: str) -> Instance:
    """Return the tiny instance (5 customers, needs 3 routes)."""
    return parse_instance(text)
