"""Build the reproducible ADR-010 Kaggle bundle for the V2-D1 benchmark.

The bundle carries code, frozen configs, and the S7 fallback package only.
It never includes raw data, local results, or sealed-partition access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consumer_complaint_intelligence.kaggle_execution import write_bundle


def main() -> None:
    """Write the bundle zip and manifest into the Kaggle staging folder."""

    summary = write_bundle(ROOT, ROOT / "kaggle" / "dataset-bundle")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
