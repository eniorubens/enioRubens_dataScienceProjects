"""Run protocol P1 on the two internal folds (weeks 0-103 only) and store the run as ``p1_folds``.

Usage: ``python scripts/run_p1_folds.py [--mode auto|train|load]``. No row with WEEK >= 104 is fitted or scored.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from retail_multilevel import cleaning, features, fixedeffects, multilevel, p1, split
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, module_files
from retail_multilevel.split import model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    args = parser.parse_args()

    config = p1.protocol_config()

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        panel = pd.read_parquet(PANEL_PATH)
        train = model_sample(panel, "train")  # weeks 0-103 only
        del panel
        if int(train["WEEK"].max()) >= split.FIRST_TEST_WEEK:
            raise RuntimeError("rows of the final window must not enter P1")
        out = p1.run_p1(train, log=lambda s: print(f"[{time.perf_counter() - t0:7.0f}s] {s}", flush=True))
        for name in ("selection_log", "convergence", "metrics", "coverage", "weekly_diff"):
            run.save_table(name, out[name])
        run.save_json("selected", out["selected"])
        return {"selected": out["selected"]["model"], "seconds_total": time.perf_counter() - t0}

    code = module_files(p1, features, fixedeffects, multilevel, split, cleaning)
    run = get_or_fit("p1_folds", config, fit, inputs=[PANEL_PATH], mode=args.mode, code_files=code)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)
    print(run.load_table("selection_log"))
    print(run.load_table("convergence"))
    print(run.load_json("selected"))
    print("run:", run.path)


if __name__ == "__main__":
    main()
