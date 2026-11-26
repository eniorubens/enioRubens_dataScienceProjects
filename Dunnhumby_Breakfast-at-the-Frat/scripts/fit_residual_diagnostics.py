"""Residual diagnostics of the final F3 (REML): dependence by lag and common week shocks, training weeks only.

Usage: ``python scripts/fit_residual_diagnostics.py [--mode auto|train|load]``. Only aggregates are stored.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from retail_multilevel import diagnostics, features, fixedeffects, multilevel, split
from retail_multilevel.features import add_model_features, pair_means, upc_means
from retail_multilevel.fixedeffects import DEFAULT_FE_CONTROL, FE_MODELS, RANDOM
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, module_files
from retail_multilevel.split import FIRST_TEST_WEEK, model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
N_FOURIER = 3
MAX_LAG = 4
KINDS = {"cond": "resid_cond", "marg": "resid_marg"}


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    args = parser.parse_args()

    config = {
        "step": "residual_diagnostics",
        "sample": "train",
        "first_test_week": FIRST_TEST_WEEK,
        "model": FE_MODELS[-1],
        "random": RANDOM,
        "n_fourier": N_FOURIER,
        "control": DEFAULT_FE_CONTROL,
        "max_lag": MAX_LAG,
        "features": "pair mean on the training window; UPC mean fallback for unseen pairs",
    }

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        panel = pd.read_parquet(PANEL_PATH)
        train = model_sample(panel, "train")
        del panel
        df = add_model_features(train, pair_means(train), upc_means(train), n_fourier=N_FOURIER)
        if (df["MEAN_SOURCE"] != "pair").any():
            raise RuntimeError("training rows must all have a pair mean")
        if int(df["WEEK"].max()) >= FIRST_TEST_WEEK:
            raise RuntimeError("only training weeks are allowed")
        res, info = diagnostics.fit_f3_residuals(df, control=DEFAULT_FE_CONTROL)
        out: dict = {"nobs": len(res), "n_pairs": int(res["PAIR_ID"].nunique()), **info}
        for kind, col in KINDS.items():
            d = diagnostics.residual_diagnostics(res, col, MAX_LAG)
            prefix = "" if kind == "cond" else "marg_"
            run.save_table(f"{prefix}lag_corr", d["lag_corr"])
            run.save_table(f"{prefix}lag1_by_category", d["lag1_by_category"])
            run.save_table(f"{prefix}week_shocks", d["week_shocks"])
            run.save_table(f"{prefix}consecutive", pd.DataFrame([d["consecutive"]]))
            run.save_table(f"{prefix}week_summary", pd.DataFrame([d["week_summary"]]))
            out[kind] = {"consecutive": d["consecutive"], "week_summary": d["week_summary"]}
        run.save_json("key_numbers", out)
        out["seconds_total"] = time.perf_counter() - t0
        return out

    code = module_files(multilevel, fixedeffects, features, split, diagnostics)
    run = get_or_fit("residual_diagnostics", config, fit, inputs=[PANEL_PATH], mode=args.mode, code_files=code)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for prefix, label in (("", "conditional"), ("marg_", "marginal")):
        print(f"--- {label} residuals ---")
        print(run.load_table(f"{prefix}lag_corr"))
        print("consecutive:", run.load_table(f"{prefix}consecutive").to_dict("records")[0])
        print(run.load_table(f"{prefix}lag1_by_category"))
        print(run.load_table(f"{prefix}week_summary").T)
    print("run:", run.path)


if __name__ == "__main__":
    main()
