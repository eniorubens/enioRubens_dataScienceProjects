"""
train_model.py — Train and serialise BetaGeoModel + GammaGammaModel.

Usage
-----
    python train_model.py
    python train_model.py --data dataset/online_retail.csv --output models/
    python train_model.py --fit-method mcmc   # full Bayesian (~5-10 min)
    python train_model.py --t 365             # 1-year CLTV horizon

Output
------
    models/bgm_model.nc
    models/gg_model.nc
    models/metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BG/NBD + Gamma-Gamma CLTV models on Online Retail data."
    )
    parser.add_argument(
        "--data",
        default="dataset/online_retail.csv",
        help="Path to the raw CSV file (default: dataset/online_retail.csv)",
    )
    parser.add_argument(
        "--output",
        default="models/",
        help="Directory to save trained models (default: models/)",
    )
    parser.add_argument(
        "--fit-method",
        default="map",
        choices=["map", "mcmc"],
        help="Fitting method: 'map' (fast) or 'mcmc' (full Bayesian, slow). Default: map",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=180,
        help="CLTV prediction horizon in days (default: 180)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    data_path = Path(args.data)
    output_path = Path(args.output)

    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # ── Imports ──────────────────────────────────────────────────────────────
    from src.cltv_model import CLTVModel
    from src.evaluation import CLTVEvaluator
    from src.preprocessing import OnlineRetailPreprocessor
    from src.utils import get_logger

    logger = get_logger("train_model")

    # ── Load & clean data ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Customer Lifetime Value — Model Training")
    logger.info("=" * 60)
    logger.info("Data: %s | Output: %s | Method: %s | t: %d days", data_path, output_path, args.fit_method, args.t)

    preprocessor = OnlineRetailPreprocessor()
    rfm, modeling_base, cleaning_summary = preprocessor.prepare(data_path)

    print("\n── Cleaning Summary ─────────────────────────────────────────")
    print(cleaning_summary.to_string(index=False))
    print(f"\nRFM: {len(rfm):,} customers | Modeling base: {len(modeling_base):,}")

    # ── Gamma-Gamma independence check ────────────────────────────────────────
    evaluator = CLTVEvaluator()
    pearson = evaluator.pearson_independence_check(rfm)
    if not pearson["assumption_holds"]:
        logger.warning(
            "Gamma-Gamma assumption may be violated (Pearson r=%.3f). "
            "Proceeding, but inspect the data.",
            pearson["pearson_r"],
        )
    else:
        logger.info("Pearson independence check passed (r=%.3f)", pearson["pearson_r"])

    # ── Holdout validation ────────────────────────────────────────────────────
    logger.info("Running calibration/holdout validation …")
    try:
        df_clean = preprocessor.clean(preprocessor.load(data_path))[0]
        cal_rfm, holdout_rfm = preprocessor.calibration_holdout_split(df_clean)

        # Join to get full calibration_holdout format for evaluator
        import pandas as pd
        ch_data = cal_rfm.merge(holdout_rfm, on="customer_id")
        ch_modeling = ch_data.loc[
            (ch_data["frequency_cal"] > 0) & (ch_data["monetary_value_cal"] > 0)
        ].copy()

        if len(ch_modeling) > 10:
            cal_model = CLTVModel(fit_method=args.fit_method)
            cal_model.fit(
                pd.DataFrame(
                    {
                        "customer_id": ch_modeling["customer_id"],
                        "frequency": ch_modeling["frequency_cal"],
                        "recency": ch_modeling["recency_cal"],
                        "T": ch_modeling["T_cal"],
                        "monetary_value": ch_modeling["monetary_value_cal"],
                    }
                )
            )
            validation = evaluator.evaluate(cal_model, ch_modeling)
            print("\n── Holdout Validation ───────────────────────────────────────")
            print(f"  RMSE (frequency): {validation['rmse_frequency']:.4f}")
            print(f"  MAE  (frequency): {validation['mae_frequency']:.4f}")
            print(f"  Pearson r:        {validation['pearson_r']:.4f}  (assumption holds: {validation['assumption_holds']})")
            print(f"  Holdout customers:{validation['n_customers_holdout']:,}")
        else:
            logger.warning("Too few calibration customers for holdout validation — skipping.")
            validation = {}
    except Exception as exc:
        logger.warning("Holdout validation failed: %s — continuing with full fit.", exc)
        validation = {}

    # ── Train on full dataset ─────────────────────────────────────────────────
    logger.info("Training final model on full dataset …")
    model = CLTVModel(fit_method=args.fit_method)
    model.fit(modeling_base)

    # Add validation metrics to metadata
    if validation:
        model.metadata["validation"] = validation

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    params = model.get_model_params()
    print("\n── Model Parameters (MAP) ───────────────────────────────────")
    print("  BG/NBD:")
    for k, v in params["bgm"].items():
        print(f"    {k}: {v:.4f}")
    print("  Gamma-Gamma:")
    for k, v in params["gg"].items():
        print(f"    {k}: {v:.4f}")

    print(f"\n── Models saved to {output_path} ─────────────────────────────")
    print("  bgm_model.nc | gg_model.nc | metadata.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
