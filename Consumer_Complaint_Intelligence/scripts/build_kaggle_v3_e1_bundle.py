"""Build the reviewable V3 E1 Kaggle code bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Write the V3 E1 bundle zip and manifest into Kaggle staging."""

    sys.path.insert(0, str(ROOT / "src"))
    from consumer_complaint_intelligence.v3_kaggle import (
        write_v3_e1_bundle,
    )

    summary = write_v3_e1_bundle(
        ROOT,
        ROOT / "kaggle" / "dataset-v3-e1",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
