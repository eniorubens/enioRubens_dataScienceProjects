"""Draw and persist the stratified development/evaluation split of the 56 Solomon instances."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from delivery_routing_vrptw import family_of, make_split, read_archive, write_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Read the SINTEF archive, draw the split and write ``data/reference/split.csv``."""
    instances = read_archive(ROOT / "data" / "raw" / "sintef" / "solomon-100.zip")
    split = make_split(list(instances))
    out = ROOT / "data" / "reference" / "split.csv"
    write_split(split, out)
    counts = Counter((family_of(name), which) for name, which in split.items())
    print(f"wrote {out} ({len(split)} rows)")
    for family in sorted({f for f, _ in counts}):
        print(f"{family:4s} dev={counts[(family, 'dev')]} eval={counts[(family, 'eval')]}")
    print("dev total:", sum(1 for v in split.values() if v == "dev"))


if __name__ == "__main__":
    main()
