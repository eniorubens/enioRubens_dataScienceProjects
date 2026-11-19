"""Reading and validating Solomon VRPTW instances and SINTEF reference values.

The data contract follows Solomon (1987): one depot (node 0), integer planar coordinates,
integer demands, hard time windows for the *start* of service, constant service times and a
homogeneous fleet bounded by ``vehicle_limit`` with capacity ``capacity``. Travel time equals
Euclidean distance.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
_HEADER_LINES = 9


@dataclass(frozen=True)
class Instance:
    """A Solomon VRPTW instance.

    Arrays are indexed by node id; index 0 is the depot.

    Attributes
    ----------
    name : str
        Instance name in upper case (for example ``"C101"``).
    vehicle_limit : int
        Maximum number of vehicles available (``K``).
    capacity : int
        Capacity of every vehicle (``Q``).
    x, y : numpy.ndarray
        Planar coordinates.
    demand : numpy.ndarray
        Demand of each node (0 for the depot).
    ready, due : numpy.ndarray
        Earliest and latest start of service. ``due[0]`` is the end of the planning horizon.
    service : numpy.ndarray
        Service duration at each node.
    """

    name: str
    vehicle_limit: int
    capacity: int
    x: np.ndarray
    y: np.ndarray
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray

    @property
    def n_customers(self) -> int:
        """Return the number of customers (excluding the depot)."""
        return len(self.x) - 1

    @property
    def family(self) -> str:
        """Return the Solomon family (``C1``, ``C2``, ``R1``, ``R2``, ``RC1`` or ``RC2``)."""
        return family_of(self.name)

    @property
    def horizon(self) -> int:
        """Return the end of the planning horizon (depot due date)."""
        return int(self.due[0])

    def subset(self, n_customers: int) -> Instance:
        """Return the instance restricted to its first ``n_customers`` customers.

        This is the usual convention for the 25- and 50-customer Solomon variants.
        """
        if not 1 <= n_customers <= self.n_customers:
            raise ValueError(f"n_customers must be in [1, {self.n_customers}]")
        k = n_customers + 1
        return Instance(
            name=f"{self.name}.{n_customers}",
            vehicle_limit=self.vehicle_limit,
            capacity=self.capacity,
            x=self.x[:k], y=self.y[:k], demand=self.demand[:k],
            ready=self.ready[:k], due=self.due[:k], service=self.service[:k],
        )


def family_of(name: str) -> str:
    """Return the Solomon family of an instance name such as ``"rc104"`` -> ``"RC1"``.

    An ``@...`` scenario suffix (for example ``"R101@w1.25"``) is accepted and ignored.
    """
    match = re.fullmatch(r"(C|R|RC)([12])\d{2}[A-Z]?(\.\d+)?(@.+)?", name.upper())
    if not match:
        raise ValueError(f"not a Solomon instance name: {name!r}")
    return match.group(1) + match.group(2)


def parse_instance(text: str) -> Instance:
    """Parse the text of a Solomon instance file.

    Parameters
    ----------
    text : str
        Full file content (SINTEF or CVRPLIB layout; both are whitespace separated).

    Returns
    -------
    Instance
        The parsed instance, validated with :func:`validate_instance`.
    """
    lines = [line.strip() for line in text.splitlines()]
    name = lines[0].upper()
    try:
        vehicle_limit, capacity = (int(v) for v in lines[4].split())
    except (IndexError, ValueError) as exc:
        raise ValueError("could not read VEHICLE NUMBER and CAPACITY on line 5") from exc
    rows = []
    for line in lines[_HEADER_LINES:]:
        if not line:
            continue
        values = line.split()
        if len(values) != 7:
            raise ValueError(f"expected 7 columns per customer line, got {len(values)}: {line!r}")
        rows.append([int(v) for v in values])
    table = np.array(rows, dtype=np.int64)
    if not np.array_equal(table[:, 0], np.arange(len(table))):
        raise ValueError("node ids must be contiguous and start at 0 (depot)")
    instance = Instance(
        name=name, vehicle_limit=vehicle_limit, capacity=capacity,
        x=table[:, 1], y=table[:, 2], demand=table[:, 3],
        ready=table[:, 4], due=table[:, 5], service=table[:, 6],
    )
    validate_instance(instance)
    return instance


def validate_instance(instance: Instance) -> None:
    """Check the data contract and raise ``ValueError`` on the first violation."""
    if instance.vehicle_limit < 1 or instance.capacity < 1:
        raise ValueError("vehicle limit and capacity must be positive")
    if instance.demand[0] != 0 or instance.service[0] != 0:
        raise ValueError("the depot must have zero demand and zero service time")
    if np.any(instance.demand < 0) or np.any(instance.service < 0):
        raise ValueError("demand and service time must be non-negative")
    if np.any(instance.demand > instance.capacity):
        raise ValueError("a single customer demand exceeds vehicle capacity")
    if np.any(instance.ready > instance.due):
        raise ValueError("ready time after due date")
    if np.any(instance.due > instance.due[0]):
        raise ValueError("a customer window ends after the depot horizon")


def read_instance(path: str | Path) -> Instance:
    """Read one instance file from disk."""
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def read_archive(path: str | Path, names: Iterable[str] | None = None) -> dict[str, Instance]:
    """Read the instances inside the SINTEF ``solomon-100.zip`` archive.

    Parameters
    ----------
    path : str or pathlib.Path
        The archive.
    names : iterable of str, optional
        When given, only the members with these instance names (case-insensitive) are read and parsed;
        other files are never opened. ``None`` reads every instance.

    Returns
    -------
    dict
        Instances keyed by upper-case name, in benchmark order.
    """
    wanted = None if names is None else {n.upper() for n in names}
    instances = {}
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if not member.lower().endswith(".txt"):
                continue
            if wanted is not None and Path(member).stem.upper() not in wanted:
                continue
            text = io.TextIOWrapper(archive.open(member), encoding="utf-8").read()
            inst = parse_instance(text)
            instances[inst.name] = inst
    return dict(sorted(instances.items(), key=lambda kv: _sort_key(kv[0])))


def _sort_key(name: str) -> tuple[int, str]:
    return FAMILIES.index(family_of(name)), name


@dataclass(frozen=True)
class Reference:
    """Best known solution value under the SINTEF hierarchical convention."""

    instance: str
    vehicles: int
    distance: float
    source: str
    note: str = ""


def read_references(path: str | Path) -> dict[str, Reference]:
    """Read the transcribed SINTEF best-known-solution table (CSV)."""
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["instance"].upper(): Reference(
            instance=row["instance"].upper(), vehicles=int(row["vehicles"]),
            distance=float(row["distance"]), source=row["reference"], note=row.get("note", ""),
        )
        for row in rows
    }


def read_cvrplib_references(path: str | Path) -> dict[str, tuple[list[list[int]], float]]:
    """Read the versioned CVRPLIB best-known-solution table (CSV).

    Parameters
    ----------
    path : str or pathlib.Path
        CSV with columns ``instance``, ``routes`` (JSON list of lists), ``cost`` and ``source``.

    Returns
    -------
    dict
        Upper-case instance name -> ``(routes, cost)`` with the cost as published (1 decimal,
        truncated-distance convention).
    """
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["instance"].upper(): ([list(map(int, r)) for r in json.loads(row["routes"])], float(row["cost"]))
        for row in rows
    }


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hexadecimal SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
