"""
evaluation.py — Temporal validation of CLTV models via calibration/holdout split.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .utils import get_logger

if TYPE_CHECKING:
    from .cltv_model import CLTVModel

logger = get_logger(__name__)


class CLTVEvaluator:
    """
    Validate a CLTV model using a temporal calibration / holdout split.

    Valida um modelo CLTV usando split temporal calibration/holdout.

    The correct validation strategy for BTYD models is temporal:
    - Calibration: learn patterns in period t0 → t_cal
    - Holdout: validate predictions in period t_cal → t_end

    k-fold cross-validation violates the temporal structure of the data
    and MUST NOT be used.

    Estratégia correta de validação para modelos BTYD é temporal:
    - Calibração: aprende padrões no período t0 → t_cal
    - Holdout: valida predições no período t_cal → t_end

    Não usar k-fold cross-validation — viola a estrutura temporal dos dados.
    """

    # ------------------------------------------------------------------
    # Scalar metrics
    # ------------------------------------------------------------------

    def compute_rmse(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        label: str = "",
    ) -> float:
        """
        Compute RMSE between actual and predicted purchase frequency in the holdout.

        Calcula RMSE entre frequência real e predita no período holdout.
        """
        rmse = float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))
        if label:
            logger.info("RMSE [%s]: %.4f", label, rmse)
        return rmse

    def compute_mae(
        self,
        actual: pd.Series,
        predicted: pd.Series,
    ) -> float:
        """
        Compute MAE between actual and predicted purchase frequency in the holdout.

        Calcula MAE entre frequência real e predita no período holdout.
        """
        return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))

    # ------------------------------------------------------------------
    # Gamma-Gamma assumption check
    # ------------------------------------------------------------------

    def pearson_independence_check(self, rfm: pd.DataFrame) -> dict:
        """
        Test the Gamma-Gamma independence assumption: frequency ⊥ monetary_value.

        Testa a premissa do Gamma-Gamma: independência entre frequency e monetary_value.

        Returns
        -------
        dict with keys: pearson_r, p_value, assumption_holds
            assumption_holds = True if |r| < 0.3 (conservative threshold).
        """
        base = rfm.loc[(rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)]
        if len(base) < 10:
            logger.warning("Too few observations (%d) for Pearson check.", len(base))
            return {"pearson_r": np.nan, "p_value": np.nan, "assumption_holds": True}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = stats.pearsonr(base["frequency"], base["monetary_value"])

        assumption_holds = abs(r) < 0.3
        if not assumption_holds:
            logger.warning(
                "Gamma-Gamma independence assumption may be violated: "
                "Pearson r=%.3f (|r| >= 0.3). Inspect the data before proceeding.",
                r,
            )
        else:
            logger.info("Pearson independence check passed: r=%.3f, p=%.4f", r, p)

        return {
            "pearson_r": float(r),
            "p_value": float(p),
            "assumption_holds": bool(assumption_holds),
        }

    # ------------------------------------------------------------------
    # Calibration plot
    # ------------------------------------------------------------------

    def calibration_plot(
        self,
        model: "CLTVModel",
        calibration_holdout_df: pd.DataFrame,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot predicted vs actual purchases in the holdout period by decile.

        Plota predicted vs actual purchases no período holdout por decil.

        X-axis: predicted purchases (deciles)
        Y-axis: actual purchases (mean per decile)
        Reference line: y=x (perfect prediction)

        Parameters
        ----------
        model                   : fitted CLTVModel
        calibration_holdout_df  : DataFrame with columns frequency_cal, recency_cal,
                                  T_cal, monetary_value_cal, frequency_holdout, T_holdout
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        ch = calibration_holdout_df.copy()

        # Build a minimal RFM for calibration-period prediction
        cal_rfm = pd.DataFrame(
            {
                "customer_id": ch["customer_id"].astype(str),
                "frequency": ch["frequency_cal"],
                "recency": ch["recency_cal"],
                "T": ch["T_cal"],
                "monetary_value": ch["monetary_value_cal"],
            }
        )

        t_holdout = float(ch["T_holdout"].median())
        predicted = model.predict_purchases(cal_rfm, t=int(round(t_holdout)))
        actual = ch["frequency_holdout"].values

        ch_eval = pd.DataFrame({"predicted": predicted.values, "actual": actual})
        ch_eval["decile"] = pd.qcut(ch_eval["predicted"].rank(method="first"), q=10, labels=False)

        grouped = ch_eval.groupby("decile").agg(
            pred_mean=("predicted", "mean"),
            act_mean=("actual", "mean"),
        )

        ax.scatter(grouped["pred_mean"], grouped["act_mean"], color="#4C78A8", s=70, zorder=3)
        ax.plot(
            [0, grouped["pred_mean"].max()],
            [0, grouped["pred_mean"].max()],
            color="#999",
            linestyle="--",
            label="Perfect prediction (y=x)",
        )
        ax.set_xlabel("Predicted purchases (decile mean)")
        ax.set_ylabel("Actual purchases (decile mean)")
        ax.set_title("Calibration Plot — Holdout Predicted vs Actual\nBy purchase frequency decile", loc="left")
        ax.legend(frameon=False)
        return ax

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: "CLTVModel",
        calibration_holdout_df: pd.DataFrame,
    ) -> dict:
        """
        Run a complete holdout evaluation of the fitted model.

        Realiza avaliação completa do modelo com o período holdout.

        Returns
        -------
        dict with keys:
            rmse_frequency, mae_frequency,
            pearson_r, pearson_p, assumption_holds,
            n_customers_holdout
        """
        ch = calibration_holdout_df.copy()

        cal_rfm = pd.DataFrame(
            {
                "customer_id": ch["customer_id"].astype(str),
                "frequency": ch["frequency_cal"],
                "recency": ch["recency_cal"],
                "T": ch["T_cal"],
                "monetary_value": ch["monetary_value_cal"],
            }
        )

        t_holdout = int(round(float(ch["T_holdout"].median())))
        predicted = model.predict_purchases(cal_rfm, t=t_holdout)
        actual = ch["frequency_holdout"]

        rmse = self.compute_rmse(actual, predicted, label="holdout_frequency")
        mae = self.compute_mae(actual, predicted)

        pearson = self.pearson_independence_check(cal_rfm)

        results = {
            "rmse_frequency": round(rmse, 4),
            "mae_frequency": round(mae, 4),
            "pearson_r": pearson["pearson_r"],
            "pearson_p": pearson["p_value"],
            "assumption_holds": pearson["assumption_holds"],
            "n_customers_holdout": int(len(ch)),
        }
        logger.info("Holdout evaluation: %s", results)
        return results
