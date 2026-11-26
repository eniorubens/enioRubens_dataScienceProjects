"""Local validation on the real (licensed) panel. Not part of CI.

Usage:
    python scripts/validate_data.py
Exit code 1 if any check fails.
"""

import sys

import pandas as pd

from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.validation import EXPECTED_REAL, validate_panel


def main() -> int:
    path = PROCESSED_DIR / "panel.parquet"
    if not path.exists():
        print(f"{path} not found. Run scripts/ingest.py and scripts/build_panel.py first.")
        return 1
    results = validate_panel(pd.read_parquet(path), EXPECTED_REAL)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name:36s} {r.detail}")
    n_fail = sum(not r.passed for r in results)
    print(f"{len(results) - n_fail}/{len(results)} checks passed")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
