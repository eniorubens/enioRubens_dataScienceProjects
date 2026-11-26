"""Fit the null models (variance decomposition) on the training sample and store the run.

Usage: ``python scripts/fit_null_models.py [--mode auto|train|load]``.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from retail_multilevel import multilevel, nullmodels, split
from retail_multilevel.nullmodels import (
    DEFAULT_CONTROL,
    NULL_MODELS,
    crosscheck_table,
    fit_null_models,
    statsmodels_crosscheck,
)
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, module_files
from retail_multilevel.split import FIRST_TEST_WEEK, model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    args = parser.parse_args()

    panel = pd.read_parquet(PANEL_PATH)
    train = model_sample(panel, "train")
    config = {
        "step": "null",
        "sample": "train",
        "first_test_week": FIRST_TEST_WEEK,
        "models": NULL_MODELS,
        "control": DEFAULT_CONTROL,
    }

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        tables = fit_null_models(train, control=DEFAULT_CONTROL)
        for name, df in tables.items():
            run.save_table(name, df)
        run.save_table("crosscheck", crosscheck_table(statsmodels_crosscheck(train), tables["varcomp"]))
        run.save_json(
            "sample",
            {
                "nobs": len(train),
                "n_stores": int(train["STORE_NUM"].nunique()),
                "n_products": int(train["UPC"].nunique()),
                "n_pairs": int(train.groupby(["STORE_NUM", "UPC"]).ngroups),
                "first_week": str(train["WEEK_END_DATE"].min()),
                "last_week": str(train["WEEK_END_DATE"].max()),
            },
        )
        icc4 = tables["icc"].query("model == 'M4_crossed_pair'")
        return {
            "icc_M4": dict(zip(icc4["group"], icc4["share"].astype(float), strict=True)),
            "lr_min_stat": float(tables["lr"]["stat"].min()),
            "seconds_total": time.perf_counter() - t0,
        }

    # Only the modules the fit depends on enter the fingerprint (cleaning reaches it via the panel file).
    code = module_files(multilevel, nullmodels, split)
    run = get_or_fit("null_models", config, fit, inputs=[PANEL_PATH], mode=args.mode, code_files=code)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(run.load_table("fits")[["name", "nobs", "npar", "loglik", "aic", "n_messages", "seconds"]])
    print(run.load_table("icc").query("model == 'M4_crossed_pair'"))
    print(run.load_table("lr"))
    print(run.load_table("crosscheck"))
    print("run:", run.path)


if __name__ == "__main__":
    main()
