"""Single final evaluation of protocol P1: fit on weeks 0-103, score weeks 104-155 once (stored as ``p1_final``).

Frozen in docs/REGISTRO-DE-DECISOES.md ("Congelamento para a avaliação final"): model C0 (REML, bobyqa,
nloptwrap retry, no substitute on failure), pooled OLS, per-pair OLS (E2 coverage) and GPBoost with the frozen
parameters of ``gpboost_tuning``. C1 and any other candidate are NOT fitted or scored. Only aggregates are stored.

Usage: ``python scripts/run_p1_final.py``.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from retail_multilevel import cleaning, features, fixedeffects, gpb, multilevel, p1, split
from retail_multilevel.fixedeffects import RANDOM
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import Run, get_or_fit, load_run, module_files

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
FIT_WEEKS = (0, 103)
VAL_WEEKS = (104, 155)
GPB_NAME = "GPBoost"
FROZEN_HASHES = {
    "p1.py": "ea3a8451b6ad", "gpb.py": "1f66d53e668a", "features.py": "cc5f2d2e5216",
    "fixedeffects.py": "7b8725e5e6d5", "multilevel.py": "bfde1e86abb6", "split.py": "36ab08e67f0e",
    "cleaning.py": "047e1279b041", "panel.parquet": "f3066b4c7fce",
}  # fmt: skip
MODULES = (p1, gpb, features, fixedeffects, multilevel, split, cleaning)


def _sha12(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def check_frozen_hashes() -> dict[str, str]:
    """Return the current hashes and raise if any differs from the frozen ones."""
    now = {m.__file__.replace("\\", "/").rsplit("/", 1)[1]: _sha12(m.__file__) for m in MODULES}
    now["panel.parquet"] = _sha12(PANEL_PATH)
    bad = {k: (now[k], v) for k, v in FROZEN_HASHES.items() if now.get(k) != v}
    if bad:
        raise RuntimeError(f"frozen hashes differ: {bad}")
    return now


def evaluate(rows: pd.DataFrame, fit_weeks: tuple[int, int], val_weeks: tuple[int, int], gpb_best: dict,
             max_week: int, log=lambda s: None) -> dict:  # fmt: skip
    """Fit C0, pooled OLS, per-pair OLS and GPBoost on ``fit_weeks`` and score ``val_weeks`` once.

    Returns
    -------
    dict
        ``metrics, coverage, convergence, weekly_diff, weekly_summary`` DataFrames and ``summary`` dict.
    """
    fit_df, val_df = p1.prepare_fold(rows, fit_weeks, val_weeks, max_week=max_week)
    log(f"fit {len(fit_df)} rows / {fit_df['PAIR_ID'].nunique()} pairs; validation {len(val_df)} rows")
    pp, cov, info = p1.pair_ols_predict(fit_df, val_df)
    ev = p1.eval_frame(val_df, fit_df, cov)
    preds = {p1.OLS_POOLED: p1.pooled_ols_predict(fit_df, val_df), p1.OLS_PAIR: pp}

    ok = (val_df["MEAN_SOURCE"] != "none").to_numpy()
    booster = gpb.fit_model(fit_df, gpb_best["params"], int(gpb_best["num_boost_round"]))
    g = np.full(len(val_df), np.nan)
    g[ok] = gpb.predict(booster, val_df.loc[ok])
    preds[GPB_NAME] = g
    log("GPBoost done")

    pred, conv = p1.fit_predict_lmer(fit_df, val_df, RANDOM, p1.BASE_MODEL, "final")
    log(f"C0 ok={conv['converged_ok']} {conv['seconds']:.0f}s")
    if pred is not None:
        preds[p1.BASE_MODEL] = pred

    metrics = p1.metric_table(ev, preds, "final")
    cov_row = p1.coverage_row(ev, info, "final")
    weekly, summ = [], []
    if p1.BASE_MODEL in preds:
        for ref in (p1.OLS_POOLED, p1.OLS_PAIR, GPB_NAME):
            e = ev
            if ref == p1.OLS_PAIR:  # per-pair OLS only exists on covered rows: compare on those rows
                e = ev.copy()
                e["POP"] = np.where(ev["COVERED"], "seen", "other")
            w = p1.weekly_diff(e, preds[p1.BASE_MODEL], preds[ref], "final", (p1.BASE_MODEL, ref))
            weekly.append(w)
            d = w["mae_diff"]
            summ.append({"reference": ref, "n_weeks": len(w), "n_weeks_c0_better": int((d < 0).sum()),
                         "mean_diff": float(d.mean()), "min_diff": float(d.min()), "max_diff": float(d.max()),
                         "worst_week_for_c0": int(w.loc[d.idxmax(), "WEEK"]),
                         "best_week_for_c0": int(w.loc[d.idxmin(), "WEEK"])})  # fmt: skip
    return {
        "metrics": metrics,
        "coverage": pd.DataFrame([cov_row]),
        "convergence": pd.DataFrame([{k: conv.get(k) for k in p1.CONV_COLS}]),
        "weekly_diff": pd.concat(weekly, ignore_index=True) if weekly else pd.DataFrame(),
        "weekly_summary": pd.DataFrame(summ),
        "summary": {"c0_converged_ok": bool(conv["converged_ok"]), "c0_scored": p1.BASE_MODEL in preds,
                    "n_fit_rows": int(len(fit_df)), "n_val_rows": int(len(val_df))},
    }  # fmt: skip


def main() -> None:
    """Run the single final evaluation."""
    hashes = check_frozen_hashes()
    gpb_run = load_run("gpboost_tuning")
    best = gpb_run.load_json("best_params")
    config = {
        "step": "p1_final",
        "fit_weeks": list(FIT_WEEKS),
        "val_weeks": list(VAL_WEEKS),
        "model": {"name": p1.BASE_MODEL, "fixed": p1.FIXED_FORMULA, "random": RANDOM, "reml": True,
                  "control": fixedeffects.DEFAULT_FE_CONTROL, "retry_control": p1.RETRY_CONTROL},
        "failure_rule": "convergence messages -> retry nloptwrap; still failing or singular -> recorded as failure, "
        "no substitute model",
        "prediction": "lme4 predict(allow.new.levels=TRUE); unseen pairs get pair effect 0",
        "references": {"pooled_ols": "F3 fixed part", "pair_ols": {"terms": p1.OLS_TERMS, "min_weeks": p1.MIN_WEEKS_OLS},
                       "gpboost": {"run_id": gpb_run.run_id, "params": best["params"],
                                   "num_boost_round": best["num_boost_round"], "num_threads": gpb.NUM_THREADS,
                                   "seed": gpb.SEED, "groups": gpb.GROUP_COLS, "features": gpb.FEATURES}},
        "populations": "a seen pairs; b rows covered by per-pair OLS (E2); c unseen pairs; none rows counted only",
        "strata": p1.STRATA,
        "weekly_diff": "per week, mean over seen rows of |err C0| - |err ref|; negative = C0 better",
        "not_scored": ["C1", "C2", "C3", "C4"],
        "frozen_hashes": FROZEN_HASHES,
        "current_hashes": hashes,
    }  # fmt: skip

    def fit(run: Run) -> dict:
        t0 = time.perf_counter()
        panel = pd.read_parquet(PANEL_PATH)
        rows = panel.loc[panel["IN_MODEL"]]
        del panel
        out = evaluate(rows, FIT_WEEKS, VAL_WEEKS, best, max_week=VAL_WEEKS[1],
                       log=lambda s: print(f"[{time.perf_counter() - t0:7.0f}s] {s}", flush=True))  # fmt: skip
        for name in ("metrics", "coverage", "convergence", "weekly_diff", "weekly_summary"):
            run.save_table(name, out[name])
        run.save_json("summary", out["summary"])
        return {**out["summary"], "seconds_total": time.perf_counter() - t0}

    run = get_or_fit("p1_final", config, fit, inputs=[PANEL_PATH], mode="auto", code_files=module_files(*MODULES))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    print(run.load_table("convergence"))
    print(run.load_table("coverage").T)
    print(run.load_table("weekly_summary"))
    print("run:", run.path, "duration_s:", run.metadata.get("duration_s"))


if __name__ == "__main__":
    main()
