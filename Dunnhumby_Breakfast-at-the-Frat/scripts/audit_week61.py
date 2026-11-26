"""Descriptive audit of week 61 (week ending 2010-03-17), training weeks only; stores the run ``week61_audit``.

The week is not dropped or altered. Aggregate tables only. Usage: ``python scripts/audit_week61.py [--mode ...]``.
"""

from __future__ import annotations

import argparse

import pandas as pd

from retail_multilevel import split, week_audit
from retail_multilevel.paths import INTERIM_DIR, PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, load_run, module_files
from retail_multilevel.split import FIRST_TEST_WEEK, model_sample

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
TX_PATH = INTERIM_DIR / "transactions.parquet"
STORES_PATH = INTERIM_DIR / "stores.parquet"
PRODUCTS_PATH = INTERIM_DIR / "products.parquet"
WEEK = 61
K = 4
YEAR_WEEKS = 52  # same calendar week one year earlier (still training data)


def _raw_rows() -> pd.DataFrame:
    tx = week_audit.with_week_index(pd.read_parquet(TX_PATH))
    stores = pd.read_parquet(STORES_PATH).drop_duplicates("STORE_ID")
    stores = stores.rename(columns={"STORE_ID": "STORE_NUM", "ADDRESS_STATE_PROV_CODE": "STATE"})[
        ["STORE_NUM", "STATE"]
    ]
    products = pd.read_parquet(PRODUCTS_PATH)[["UPC", "CATEGORY"]]
    tx = tx.merge(stores, on="STORE_NUM", how="left").merge(products, on="UPC", how="left")
    return tx[tx["WEEK"] < FIRST_TEST_WEEK].reset_index(drop=True)


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "train", "load"], default="auto")
    args = parser.parse_args()
    config = {
        "step": "week61_audit",
        "week": WEEK,
        "k": K,
        "first_test_week": FIRST_TEST_WEEK,
        "year_weeks": YEAR_WEEKS,
        "spend_tol": week_audit.SPEND_TOL,
    }

    def fit(run: Run) -> dict:
        raw = _raw_rows()
        panel = model_sample(pd.read_parquet(PANEL_PATH), "train")
        window = list(range(WEEK - 8, WEEK + 9))
        prof = week_audit.week_profile(raw)
        run.save_table("profile_window", prof[prof["WEEK"].isin(window)])
        cmp_raw = week_audit.compare_to_neighbours(prof, WEEK, K)
        run.save_table("vs_neighbours", cmp_raw)
        run.save_table("vs_neighbours_prior_year", week_audit.compare_to_neighbours(prof, WEEK - YEAR_WEEKS, K))
        pprof = week_audit.week_profile(panel)
        run.save_table("vs_neighbours_model_sample", week_audit.compare_to_neighbours(pprof, WEEK, K))
        for col, name in (("CATEGORY", "by_category"), ("STATE", "by_state"), ("STORE_NUM", "by_store")):
            run.save_table(name, week_audit.group_ratio(raw, WEEK, col, K).astype({"group": str}))
        mp = week_audit.missing_pairs(raw, WEEK)
        run.save_table("missing_by_category", mp["CATEGORY"].value_counts().rename_axis("CATEGORY").reset_index())
        run.save_table("missing_by_state", mp["STATE"].value_counts().rename_axis("STATE").reset_index())
        presence = week_audit.pair_presence(raw, WEEK)
        run.save_json("presence", presence)
        run.save_table("pair_ratio", week_audit.pair_ratio_summary(raw, window, K))
        integ = week_audit.integrity_by_week(raw, window)
        run.save_table("integrity_window", integ)
        verdict = week_audit.classify_week(cmp_raw, integ, WEEK, K)
        run.save_json("verdict", verdict)
        shock = load_run("residual_diagnostics").load_table("week_shocks")
        row = shock[shock["WEEK"] == WEEK].iloc[0]
        stores = week_audit.group_ratio(raw, WEEK, "STORE_NUM", K)
        c = cmp_raw.set_index("metric")
        return {
            "week": WEEK,
            "week_end_date": str(raw.loc[raw["WEEK"] == WEEK, "WEEK_END_DATE"].iloc[0].date()),
            "verdict": verdict,
            "presence": presence,
            "units_ratio": float(c.loc["units", "ratio"]),
            "visits_ratio": float(c.loc["visits", "ratio"]),
            "units_per_visit_ratio": float(c.loc["units_per_visit", "ratio"]),
            "stores_below_0.6": int((stores["ratio"] < 0.6).sum()),
            "n_stores": int(len(stores)),
            "mean_resid_f3": float(row["mean_resid"]),
        }

    run = get_or_fit("week61_audit", config, fit, inputs=[PANEL_PATH, TX_PATH, STORES_PATH, PRODUCTS_PATH],
                     mode=args.mode, code_files=module_files(week_audit, split))  # fmt: skip
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 100)
    print("run:", run.path)
    print(run.metrics)
    for name in (
        "vs_neighbours",
        "vs_neighbours_prior_year",
        "by_category",
        "by_state",
        "missing_by_category",
        "pair_ratio",
        "integrity_window",
    ):
        print(f"\n== {name}")
        print(run.load_table(name).round(3).to_string(index=False))
    print("\n== by_store (top 10 deficits)")
    print(run.load_table("by_store").head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
