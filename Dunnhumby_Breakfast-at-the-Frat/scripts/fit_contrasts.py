"""Planned interaction tests (notebook 05, part 1); stores the run ``p2_contrasts``.

Usage: ``python scripts/fit_contrasts.py [--mode auto|train|load]``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import pandas as pd

from retail_multilevel import cleaning, contrasts, sensitivity, split
from retail_multilevel.contrasts import SPECS, assert_training_only, build_population, fit_contrast, holm, prepare_frame
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, module_files
from retail_multilevel.sensitivity import FE_REGRESSORS
from retail_multilevel.split import FIRST_TEST_WEEK, model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
TOL = 1e-10
ALPHA = 0.05


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    args = parser.parse_args()

    config = {
        "step": "p2_contrasts",
        "decision": "REGISTRO-DE-DECISOES 2026-11-24 P2, notebook 05 planned contrasts",
        "sample": "train",
        "first_test_week": FIRST_TEST_WEEK,
        "y": "LOG_UNITS",
        "base_regressors": FE_REGRESSORS,
        "interaction": "LOG_DISC x 1[level != reference]",
        "effects": ["PAIR_ID", "WEEK"],
        "tests": [{k: str(v) for k, v in asdict(s).items()} for s in SPECS],
        "wald": "delta' V^-1 delta, V = full two-way CGM covariance; F = W/q; F(q, min(G_pair, G_week) - 1)",
        "unavailable_rule": f"rank-deficient design, q x q block not PD, or cond > {contrasts.COND_MAX:.0e}",
        "variation_tol": contrasts.VARIATION_TOL,
        "holm": {"family": [s.name for s in SPECS], "alpha": ALPHA, "unavailable_p": 1.0},
        "ci": "pointwise, t(min(G_pair, G_week) - 1)",
        "tol": TOL,
    }

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        train = model_sample(pd.read_parquet(PANEL_PATH), "train")
        assert_training_only(train)
        assert int(train["WEEK"].max()) <= 103, "training data must stop at week 103"
        train = prepare_frame(train)
        pops, supports, coefs, slopes, walds = [], [], [], [], []
        for spec in SPECS:
            pop, info = build_population(train, spec)
            res = fit_contrast(pop, spec, tol=TOL)
            pops.append(info)
            supports.append(res["support"])
            coefs.append(res["coefficients"])
            slopes.append(res["slopes"])
            walds.append(res["wald"])
        wald = pd.DataFrame(walds)
        p_holm, reject = holm(wald["p_raw"].to_numpy(), wald["available"].to_numpy(), ALPHA)
        holm_df = pd.DataFrame(
            {
                "test": wald["test"],
                "p_raw": wald["p_raw"],
                "p_holm": p_holm,
                "reject_5pct": reject,
                "available": wald["available"],
            }
        )
        wald_cols = [
            "test",
            "moderator",
            "reference",
            "q",
            "W",
            "F",
            "df1",
            "df2",
            "G_pair",
            "G_week",
            "p_raw",
            "available",
            "reason",
            "cond_number",
            "min_eig",
            "design_rank",
            "n_terms",
            "rank_deficient",
            "twoway_psd_fixed",
            "design_cond",
            "nobs",
            "nobs_restricted",
        ]
        run.save_table("populations", pd.DataFrame(pops))
        run.save_table("support", pd.concat(supports, ignore_index=True).astype({"level": str}))
        run.save_table("coefficients", pd.concat(coefs, ignore_index=True))
        run.save_table("slopes", pd.concat(slopes, ignore_index=True).astype({"level": str}))
        run.save_table("wald", wald[wald_cols])
        run.save_table("holm", holm_df)
        return {
            "p_raw": dict(zip(wald["test"], wald["p_raw"].astype(float), strict=True)),
            "p_holm": dict(zip(wald["test"], [None if pd.isna(v) else float(v) for v in p_holm], strict=True)),
            "available": dict(zip(wald["test"], wald["available"].astype(bool), strict=True)),
            "n_excluded": {p["test"]: p["n_excluded"] for p in pops},
            "max_week": int(train["WEEK"].max()),
            "seconds_total": time.perf_counter() - t0,
        }

    run = get_or_fit(
        "p2_contrasts", config, fit, inputs=[PANEL_PATH], mode=args.mode,
        code_files=module_files(contrasts, sensitivity, split, cleaning),
    )  # fmt: skip
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    for name in ("populations", "support", "wald", "holm", "slopes", "coefficients"):
        print(f"--- {name}")
        print(run.load_table(name).round(5).to_string())
    print(run.metrics)
    print("run:", run.path)


if __name__ == "__main__":
    main()
