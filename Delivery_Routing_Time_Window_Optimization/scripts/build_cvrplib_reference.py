"""Build the versioned CSV of CVRPLIB best known solutions from the extracted ``.sol`` files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "cvrplib_solomon_100" / "extracted" / "bks"
TARGET = ROOT / "data" / "reference" / "cvrplib_bks_100.csv"


def parse_sol(path: Path) -> tuple[list[list[int]], float]:
    """Return ``(routes, cost)`` from a CVRPLIB ``.sol`` file."""
    routes: list[list[int]] = []
    cost = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Route"):
            routes.append([int(v) for v in line.split(":")[1].split()])
        elif line.startswith("Cost"):
            cost = float(line.split()[1])
    if cost is None:
        raise ValueError(f"no cost line in {path}")
    return routes, cost


def main() -> None:
    """Write ``data/reference/cvrplib_bks_100.csv``."""
    files = sorted(SOURCE.glob("*.sol"))
    with open(TARGET, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "routes", "cost", "source"])
        for path in files:
            routes, cost = parse_sol(path)
            writer.writerow([path.stem.upper(), json.dumps(routes), f"{cost:.1f}", "CVRPLIB"])
    print(f"wrote {TARGET} ({len(files)} rows)")


if __name__ == "__main__":
    main()
