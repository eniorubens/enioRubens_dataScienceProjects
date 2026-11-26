"""Read the dunnhumby Excel workbook once and cache each sheet as Parquet in data/interim/.

Usage:
    python scripts/ingest.py [--overwrite]
"""

import argparse
import time

import pandas as pd

from retail_multilevel.io import excel_to_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="re-read the Excel even if Parquet exists")
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = excel_to_parquet(overwrite=args.overwrite)
    for name, path in paths.items():
        df = pd.read_parquet(path)
        print(f"{name:13s} {df.shape[0]:>8,d} rows x {df.shape[1]:>2d} cols -> {path}")
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
