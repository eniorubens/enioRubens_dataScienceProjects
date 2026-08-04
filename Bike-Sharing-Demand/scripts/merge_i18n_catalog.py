"""Merge reviewed translations into a JSON catalog and remove duplicate keys."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("corrections", type=Path)
    args = parser.parse_args()

    pairs = json.loads(
        args.catalog.read_text(encoding="utf-8"),
        object_pairs_hook=lambda values: values,
    )
    counts = Counter(key for key, _ in pairs)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    catalog = dict(pairs)
    corrections = json.loads(args.corrections.read_text(encoding="utf-8"))
    catalog.update(corrections)
    args.catalog.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Merged {len(corrections)} reviewed translation(s); "
        f"removed {duplicate_count} duplicate key(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
