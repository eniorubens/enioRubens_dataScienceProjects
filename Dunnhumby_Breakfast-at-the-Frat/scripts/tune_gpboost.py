"""Tune the GPBoost challenger of protocol P1 (Optuna, 50 trials, internal folds only) and store the run.

Usage: ``python scripts/tune_gpboost.py [--mode auto|train|load] [--trials N] [--fit-frac F] [--max-rounds R]``.
Rows with WEEK >= 104 are dropped before any fit. Tuning fits may use a fixed pair subsample (--fit-frac) and a cap on
the boosting rounds (--max-rounds) for speed; the per-fold metrics of the best parameters use the full fit window.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from retail_multilevel import features, gpb, split
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, module_files

PANEL_PATH = PROCESSED_DIR / "panel.parquet"


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    parser.add_argument("--trials", type=int, default=gpb.N_TRIALS)
    parser.add_argument("--fit-frac", type=float, default=1.0, help="fraction of fit-window pairs used in tuning")
    parser.add_argument("--max-rounds", type=int, default=None, help="cap on num_boost_round during tuning")
    args = parser.parse_args()

    config = {
        "step": "gpboost_tuning",
        "protocol": "P1 secondary challenger",
        "folds": gpb.FOLDS,
        "first_test_week": split.FIRST_TEST_WEEK,
        "features": gpb.FEATURES,
        "groups": gpb.GROUP_COLS,
        "n_trials": args.trials,
        "sampler": "TPE",
        "seed": gpb.SEED,
        "search_space": gpb.SEARCH_SPACE,
        "objective": "mean over folds of per-pair-mean MAE of LOG_UNITS on seen pairs",
        "fit_frac": args.fit_frac,
        "max_rounds": args.max_rounds,
        "num_threads": gpb.NUM_THREADS,
    }

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        panel = pd.read_parquet(PANEL_PATH)
        panel = panel.loc[panel["IN_MODEL"] & (panel["WEEK"] < split.FIRST_TEST_WEEK)]
        folds = gpb.prepare_folds(panel)
        del panel
        study = gpb.tune(folds, args.trials, gpb.SEED, args.fit_frac, args.max_rounds)
        trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        run.save_table("trials", trials)
        best = dict(study.best_params)
        rounds = int(best.pop("num_boost_round"))
        if args.max_rounds is not None:
            rounds = min(rounds, args.max_rounds)
        run.save_json("best_params", {"params": best, "num_boost_round": rounds, "objective": study.best_value})
        rows = []
        for f in folds:
            rows += gpb.fold_metrics(f, best, rounds, 1.0)  # final metrics: full fit window
        run.save_table("fold_metrics", pd.DataFrame(rows))
        return {
            "best_objective": float(study.best_value),
            "best_trial": study.best_trial.number,
            "seconds_total": time.perf_counter() - t0,
        }

    code = module_files(gpb, features, split)
    run = get_or_fit("gpboost_tuning", config, fit, inputs=[PANEL_PATH], mode=args.mode, code_files=code)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(run.load_json("best_params"))
    print(run.load_table("fold_metrics"))
    print("metrics:", run.metrics)
    print("run:", run.path)


if __name__ == "__main__":
    main()
