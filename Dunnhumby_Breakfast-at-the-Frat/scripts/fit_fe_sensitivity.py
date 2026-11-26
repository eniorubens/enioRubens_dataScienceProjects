"""Two-way fixed-effects (pair + week) sensitivity on the training sample; stores the run ``fe_sensitivity``.

Usage: ``python scripts/fit_fe_sensitivity.py [--mode auto|train|load]``.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from retail_multilevel import sensitivity, split
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, load_run, module_files
from retail_multilevel.sensitivity import FE_REGRESSORS, two_way_fe_ols
from retail_multilevel.split import FIRST_TEST_WEEK, model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
TOL = 1e-10


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    args = parser.parse_args()

    config = {
        "step": "fe_sensitivity",
        "sample": "train",
        "first_test_week": FIRST_TEST_WEEK,
        "y": "LOG_UNITS",
        "regressors": FE_REGRESSORS,
        "effects": ["PAIR_ID", "WEEK"],
        "cluster": "PAIR_ID",
        "se": "CR1",
        "se_twoway": "CGM pair + week - pair x week, CR1-type factor per component, eigenvalue fix, t(min G - 1) CI",
        "tol": TOL,
    }

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        train = model_sample(pd.read_parquet(PANEL_PATH), "train").copy()
        train["LOG_DISC"] = train["LOG_PRICE"] - train["LOG_BASE_PRICE"]
        for col in ("FEATURE", "DISPLAY", "TPR_ONLY"):
            train[col] = train[col].astype(int)
        table, info = two_way_fe_ols(train, "LOG_UNITS", FE_REGRESSORS, tol=TOL)
        run.save_table("fe_two_way", table)
        run.save_json("sample", {"nobs": len(train), **info})
        return {
            "coef": dict(zip(table["term"], table["estimate"].astype(float), strict=True)),
            "n_sweeps": info["n_sweeps"],
            "twoway_psd_fixed": info["twoway_psd_fixed"],
            "n_clusters": {
                "pair": info["n_clusters_pair"],
                "week": info["n_clusters_week"],
                "cell": info["n_clusters_cell"],
            },
            "seconds_total": time.perf_counter() - t0,
        }

    run = get_or_fit(
        "fe_sensitivity", config, fit, inputs=[PANEL_PATH], mode=args.mode,
        code_files=module_files(sensitivity, split),
    )  # fmt: skip
    pd.set_option("display.width", 200)
    table = run.load_table("fe_two_way").set_index("term")
    f3 = load_run("fixed_effects").load_table("fixef").set_index("term")
    print(run.metrics)
    out = table[["estimate", "se_cluster", "se_week", "se_twoway", "ci_low_twoway", "ci_high_twoway"]].copy()
    f3_name = {"LOG_BASE_PRICE": "LOG_BASE_DEV"}  # same slope, F3 parametrises the base price as a deviation
    out["se_f3"] = [f3.loc[f3_name.get(t, t), "std_error"] for t in out.index]
    for col in ("se_cluster", "se_week", "se_twoway"):
        out[f"{col}/f3"] = out[col] / out["se_f3"]
    print(out.round(4).to_string())
    print("run:", run.path)


if __name__ == "__main__":
    main()
