"""Fit the fixed effects of price and promotion (step-up F0..F3) on the training sample and store the run.

Usage: ``python scripts/fit_fixed_effects.py [--mode auto|train|load] [--exclude-fallback]``.

``--exclude-fallback`` drops the rows whose BASE_PRICE fell back to PRICE (before the features are learned) and
fits only the final F3 by REML, stored as experiment ``fixed_effects_no_fallback``.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from retail_multilevel import features, fixedeffects, multilevel, split
from retail_multilevel.features import add_model_features, pair_means, upc_means
from retail_multilevel.fixedeffects import DEFAULT_FE_CONTROL, FE_MODELS, RANDOM, fit_f3_reml, fit_fixed_effects
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, module_files
from retail_multilevel.split import FIRST_TEST_WEEK, model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
N_FOURIER = 3
KEY_TERMS = ["LOG_DISC", "LOG_BASE_DEV", "FEATURE", "DISPLAY", "TPR_ONLY"]


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    parser.add_argument("--exclude-fallback", action="store_true", help="sensitivity: drop BASE_FALLBACK rows")
    args = parser.parse_args()
    experiment = "fixed_effects_no_fallback" if args.exclude_fallback else "fixed_effects"

    config = {
        "step": "fixed_effects",
        "sample": "train",
        "first_test_week": FIRST_TEST_WEEK,
        "models": FE_MODELS,
        "random": RANDOM,
        "n_fourier": N_FOURIER,
        "control": DEFAULT_FE_CONTROL,
        "features": "pair mean on the training window; UPC mean fallback for unseen pairs",
    }
    if args.exclude_fallback:
        config |= {"step": "fixed_effects_no_fallback", "exclude_fallback": True, "models": [FE_MODELS[-1]]}

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        panel = pd.read_parquet(PANEL_PATH)
        train = model_sample(panel, "train")
        del panel
        n_fallback = int(train["BASE_FALLBACK"].sum())
        if args.exclude_fallback:
            train = train.loc[~train["BASE_FALLBACK"]]
        df = add_model_features(train, pair_means(train), upc_means(train), n_fourier=N_FOURIER)
        if (df["MEAN_SOURCE"] != "pair").any():
            raise RuntimeError("training rows must all have a pair mean")
        if args.exclude_fallback:
            tables = fit_f3_reml(df, control=DEFAULT_FE_CONTROL)
        else:
            tables = fit_fixed_effects(df, control=DEFAULT_FE_CONTROL)
        for name, table in tables.items():
            run.save_table(name, table)
        run.save_json(
            "sample",
            {
                "nobs": len(df),
                "n_fallback_rows_in_train": n_fallback,
                "excluded_fallback": bool(args.exclude_fallback),
                "n_pairs": int(df["PAIR_ID"].nunique()),
                "log_disc_mean": float(df["LOG_DISC"].mean()),
                "log_disc_sd": float(df["LOG_DISC"].std()),
                "share_disc_below_5pct": float((df["LOG_DISC"] < -0.05).mean()),
            },
        )
        fx = tables["fixef"].set_index("term")["estimate"]
        out = {"coef_F3": {t: float(fx[t]) for t in KEY_TERMS}}
        if "lr" in tables:
            lr = tables["lr"]
            out["lr_stats"] = {
                f"{r}->{f}": float(s) for r, f, s in zip(lr["restricted"], lr["full"], lr["stat"], strict=True)
            }
        out["seconds_total"] = time.perf_counter() - t0
        return out

    code = module_files(multilevel, fixedeffects, features, split)
    run = get_or_fit(experiment, config, fit, inputs=[PANEL_PATH], mode=args.mode, code_files=code)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(run.load_table("fits")[["name", "nobs", "npar", "loglik", "aic", "bic", "seconds", "n_messages"]])
    print(run.load_table("fixef"))
    if not args.exclude_fallback:
        print(run.load_table("lr"))
        print(run.load_table("pseudo_r2"))
        print(run.load_table("ols_pooled"))
    print("run:", run.path)


if __name__ == "__main__":
    main()
