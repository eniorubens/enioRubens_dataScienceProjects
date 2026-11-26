"""Scenario layer (notebook 06): refit F3 by REML with its joint covariance and count the observed support.

Usage: ``python scripts/fit_scenarios.py [--mode auto|train|load]``. Training weeks (0-103) only. The refit must
reproduce the coefficients of the stored ``fixed_effects`` run to 1e-6, otherwise the script stops.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from retail_multilevel import features, fixedeffects, interpret, multilevel, scenarios, split
from retail_multilevel.features import add_model_features, pair_means, upc_means
from retail_multilevel.fixedeffects import DEFAULT_FE_CONTROL, FE_MODELS, RANDOM
from retail_multilevel.interpret import DEFAULT_SCENARIOS
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, load_run, module_files
from retail_multilevel.split import FIRST_TEST_WEEK, model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
N_FOURIER = 3
COEF_TOL = 1e-6
EXEC_A, EXEC_B = "scenario_tpr20", "scenario_display20"


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    args = parser.parse_args()

    config = {
        "step": "p2_scenarios",
        "sample": "train",
        "first_test_week": FIRST_TEST_WEEK,
        "model": FE_MODELS[-1],
        "random": RANDOM,
        "estimator": "F3 REML, joint vcov(m) of lme4",
        "n_fourier": N_FOURIER,
        "control": DEFAULT_FE_CONTROL,
        "scenarios": DEFAULT_SCENARIOS,
        "support_tol": scenarios.SUPPORT_TOL,
        "coef_tol": COEF_TOL,
        "executive": [EXEC_A, EXEC_B],
        "features": "pair mean on the training window; UPC mean fallback for unseen pairs",
    }

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        panel = pd.read_parquet(PANEL_PATH)
        train = model_sample(panel, "train")
        del panel
        scenarios.assert_training_only(train)
        df = add_model_features(train, pair_means(train), upc_means(train), n_fourier=N_FOURIER)
        if (df["MEAN_SOURCE"] != "pair").any():
            raise RuntimeError("training rows must all have a pair mean")

        model, vcov = scenarios.fit_f3_with_vcov(df, control=DEFAULT_FE_CONTROL)
        fixef = model.fixef
        stored = load_run("fixed_effects").load_table("fixef").set_index("term")["estimate"]
        recon = fixef.merge(stored.rename("stored_estimate").reset_index(), on="term", how="outer")
        recon = recon.rename(columns={"estimate": "refit_estimate"})[["term", "stored_estimate", "refit_estimate"]]
        recon["abs_diff"] = (recon["refit_estimate"] - recon["stored_estimate"]).abs()
        max_diff = float(recon["abs_diff"].max())
        if recon["abs_diff"].isna().any() or max_diff > COEF_TOL:
            raise RuntimeError(f"refit does not reproduce the stored F3 coefficients (max abs diff {max_diff:.3e})")

        terms = list(fixef["term"])
        vcov = vcov.loc[terms, terms]
        contrast = interpret.scenario_contrast(fixef, vcov, DEFAULT_SCENARIOS)
        # point estimates against notebook 03's table (scenario_effects on the stored fixef)
        nb03 = interpret.scenario_effects(stored.rename("estimate").reset_index(), DEFAULT_SCENARIOS)
        eff_diff = float(np.abs(contrast["effect_pct"].to_numpy() - nb03["effect_pct"].to_numpy()).max())
        if eff_diff > 1e-4:
            raise RuntimeError(f"scenario effects differ from notebook 03 (max abs diff {eff_diff:.3e} pp)")
        if (
            contrast.loc[contrast["scenario"] == "scenario_regular", ["dlog", "se", "effect_pct"]]
            .abs()
            .to_numpy()
            .max()
        ):
            raise RuntimeError("regular week must give exactly 0")

        support = scenarios.support_table(df, DEFAULT_SCENARIOS)
        table = contrast.merge(support, on="scenario", how="left")
        run.save_table("fixef", fixef[["term", "estimate", "std_error"]].rename(columns={"std_error": "se"}))
        run.save_table("vcov", vcov.rename_axis("term").reset_index())
        run.save_table("scenarios", table)
        run.save_table("reconciliation", recon)

        by_name = {s[0]: s for s in DEFAULT_SCENARIOS}
        a, b = by_name[EXEC_A], by_name[EXEC_B]
        row = contrast.set_index("scenario")
        diff = interpret.contrast_between(fixef, vcov, a, b)
        run.save_json(
            "executive",
            {
                "a": EXEC_A,
                "b": EXEC_B,
                "effect_a": {
                    k: float(row.loc[EXEC_A, k]) for k in ("dlog", "se", "effect_pct", "ci_low_pct", "ci_high_pct")
                },
                "effect_b": {
                    k: float(row.loc[EXEC_B, k]) for k in ("dlog", "se", "effect_pct", "ci_low_pct", "ci_high_pct")
                },
                "difference_b_minus_a": diff | {"dlog_label": "display20 - tpr20, log scale, joint SE"},
                "interval_kind": "pointwise 95% (normal, joint REML covariance); ratio = modelled sales ratio",
            },
        )
        run.save_json(
            "reconciliation",
            {
                "coef_max_abs_diff": max_diff,
                "coef_tol": COEF_TOL,
                "coef_match": True,
                "scenario_effect_max_abs_diff_pp": eff_diff,
                "nobs": len(df),
                "max_week": int(df["WEEK"].max()),
                "n_messages": len(model.messages),
                "messages": model.messages,
            },
        )
        return {
            "nobs": len(df),
            "coef_max_abs_diff": max_diff,
            "scenario_effect_max_abs_diff_pp": eff_diff,
            "n_outside_support": int(table["outside_support"].sum()),
            "seconds_total": time.perf_counter() - t0,
        }

    code = module_files(scenarios, interpret, fixedeffects, features, multilevel, split)
    run = get_or_fit("p2_scenarios", config, fit, inputs=[PANEL_PATH], mode=args.mode, code_files=code)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(run.load_table("scenarios"))
    print(run.load_table("reconciliation"))
    print(run.load_json("executive"))
    print("run:", run.path)


if __name__ == "__main__":
    main()
