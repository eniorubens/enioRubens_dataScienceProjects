"""Per-pair OLS discount slopes on the training weeks (notebook 01).

Usage: ``python scripts/fit_pair_ols_slopes.py [--mode auto|train|load]``. Training weeks (0-103) only. The
specification is the P1 per-pair OLS (``LOG_UNITS ~ LOG_DISC + FEATURE + DISPLAY + TPR_ONLY``, pairs with at
least 8 weeks).
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from retail_multilevel import features, p1, pair_slopes, split
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, module_files
from retail_multilevel.split import FIRST_TEST_WEEK, model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    args = parser.parse_args()

    config = {
        "step": "pair_ols_slopes",
        "sample": "train",
        "first_test_week": FIRST_TEST_WEEK,
        "spec": "LOG_UNITS ~ LOG_DISC + FEATURE + DISPLAY + TPR_ONLY, one OLS per store x product pair",
        "terms": p1.OLS_TERMS,
        "min_weeks": p1.MIN_WEEKS_OLS,
        "strata": p1.STRATA,
        "coverage": "p1.pair_ols_fit: terms without variation or collinear are not estimated",
    }

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        panel = pd.read_parquet(PANEL_PATH)
        train = model_sample(panel, "train")
        del panel
        max_week = int(train["WEEK"].max())
        assert max_week <= FIRST_TEST_WEEK - 1, f"training sample goes up to week {max_week}"
        slopes = pair_slopes.pair_slopes(train, p1.MIN_WEEKS_OLS)
        summary = pair_slopes.summarize_slopes(slopes)
        run.save_table("slopes", slopes)
        run.save_table("summary", summary)
        by = summary.set_index("stratum")
        key = {
            "n_pairs_train": int(len(slopes)),
            "n_pairs_fitted": int(slopes["has_ols"].sum()),
            "n_pairs_with_slope": int(slopes["slope"].notna().sum()),
            "n_pairs_fitted_without_slope": int((slopes["has_ols"] & slopes["slope"].isna()).sum()),
            "n_pairs_below_min_weeks": int((~slopes["has_ols"]).sum()),
            "max_week": max_week,
            "nobs": int(len(train)),
            "share_positive_pct": {k: float(v) for k, v in by["share_positive_pct"].items()},
            "n_pairs": {k: int(v) for k, v in by["n_pairs"].items()},
            "n_positive": {k: int(v) for k, v in by["n_positive"].items()},
            "median_slope": float(by.loc["all", "median"]),
        }
        run.save_json("key_numbers", key)
        return {**{k: v for k, v in key.items() if not isinstance(v, dict)}, "seconds_total": time.perf_counter() - t0}

    code = module_files(pair_slopes, p1, split, features)
    run = get_or_fit("pair_ols_slopes", config, fit, inputs=[PANEL_PATH], mode=args.mode, code_files=code)
    pd.set_option("display.width", 250)
    print(run.load_table("summary"))
    print(run.load_json("key_numbers"))
    print("run:", run.path)


if __name__ == "__main__":
    main()
