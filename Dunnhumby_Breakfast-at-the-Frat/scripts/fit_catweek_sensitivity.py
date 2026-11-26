"""Sensitivity of F3 to category x week shocks: F3 versus F3 + (1 | CATEGORY:WEEK), training weeks only.

Usage: ``python scripts/fit_catweek_sensitivity.py [--mode auto|train|load]``. Only aggregates are stored.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from retail_multilevel import catweek, diagnostics, features, fixedeffects, multilevel, split
from retail_multilevel.features import add_model_features, pair_means, upc_means
from retail_multilevel.fixedeffects import DEFAULT_FE_CONTROL, RANDOM
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, module_files
from retail_multilevel.split import FIRST_TEST_WEEK, model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
N_FOURIER = 3
MAX_LAG = 4


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    args = parser.parse_args()

    config = {
        "step": "catweek_sensitivity",
        "sample": "train",
        "first_test_week": FIRST_TEST_WEEK,
        "reference": catweek.reference_formula(),
        "sensitivity": catweek.sensitivity_formula(),
        "random": RANDOM,
        "catweek_factor": "CATEGORY|WEEK string column converted to an R factor",
        "n_fourier": N_FOURIER,
        "control": DEFAULT_FE_CONTROL,
        "fallback_control": catweek.FALLBACK_CONTROL,
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
        catweek.check_training(df)
        df = catweek.add_catweek(df)
        cnt = catweek.counts(df)
        if cnt["n_missing_category"] or cnt["max_week"] > 103:
            raise RuntimeError(f"unexpected sample: {cnt}")
        fits, attempts, n_rows = catweek.fit_both(df, control=DEFAULT_FE_CONTROL)
        ref, sens = fits[catweek.REFERENCE], fits[catweek.SENSITIVITY]
        if ref.nobs != sens.nobs or ref.nobs != n_rows:
            raise RuntimeError("the two fits must use identical rows")

        fixef = catweek.fixef_compare(ref, sens)
        varcomp = catweek.varcomp_compare(fits)
        conv = catweek.convergence_table(attempts)
        lr = catweek.descriptive_lr(ref, sens)
        res = {name: catweek.conditional_residuals(f, df) for name, f in fits.items()}
        summ = catweek.residual_summaries(res, MAX_LAG)

        run.save_table("fixef_compare", fixef)
        run.save_table("varcomp", varcomp)
        run.save_table("convergence", conv)
        run.save_table("counts", pd.DataFrame([cnt]))
        for name, table in summ.items():
            run.save_table(name, table)
        cw = varcomp[(varcomp["model"] == catweek.SENSITIVITY) & (varcomp["group"] == catweek.CATWEEK_COL)]
        out = {
            "counts": cnt,
            "catweek_variance": float(cw["variance"].iloc[0]),
            "catweek_sd": float(cw["sd"].iloc[0]),
            "catweek_share": float(cw["share"].iloc[0]),
            "lr_descriptive": lr,
            "singular": {n: catweek.is_singular(f) for n, f in fits.items()},
            "convergence_messages": {n: catweek.convergence_messages(f) for n, f in fits.items()},
            "limitations": [f"{n}: {' | '.join(f.messages)}" for n, f in fits.items() if f.messages],
        }
        run.save_json("key_numbers", out)
        out["seconds_total"] = time.perf_counter() - t0
        return out

    code = module_files(catweek, fixedeffects, features, multilevel, diagnostics, split)
    run = get_or_fit("p2_catweek", config, fit, inputs=[PANEL_PATH], mode=args.mode, code_files=code)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for name in ("counts", "convergence", "fixef_compare", "varcomp", "lag_corr", "week_summary"):
        print(f"--- {name} ---")
        print(run.load_table(name))
    print(run.load_json("key_numbers"))
    print("run:", run.path)


if __name__ == "__main__":
    main()
