"""
segmentation.py — Customer segmentation by CLTV quartile and P(alive).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

SEGMENT_ORDER = ["Low Value", "Mid Value", "High Value", "Top Value"]
ACTION_ORDER = ["Low Priority", "Low-Cost Nurture", "Reactivation Priority", "Retain and Grow"]

VALUE_PALETTE = {
    "Low Value": "#7A7A7A",
    "Mid Value": "#4C78A8",
    "High Value": "#59A14F",
    "Top Value": "#B07AA1",
}

ACTION_PALETTE = {
    "Low Priority": "#7A7A7A",
    "Low-Cost Nurture": "#59A14F",
    "Reactivation Priority": "#C44E52",
    "Retain and Grow": "#4C78A8",
}


class CustomerSegmenter:
    """
    Segment customers into CLTV quartiles and assign marketing actions
    based on a CLTV × P(alive) decision matrix.

    Segmenta clientes em quartis de CLTV e define ação de marketing
    com base na matriz de decisão CLTV × P(alive).

    Decision matrix (2×2):

                        | Low P(alive)          | High P(alive)    |
    Low/Mid Value       | Low Priority          | Low-Cost Nurture |
    High/Top Value      | Reactivation Priority | Retain and Grow  |

    Parameters
    ----------
    alive_threshold : float or None
        Median of ``probability_alive`` is used when None.
        Fixing the threshold ensures consistency between train and inference.
    """

    def __init__(self, alive_threshold: float | None = None) -> None:
        self.alive_threshold = alive_threshold
        self._fitted_threshold: float | None = None

    def segment(
        self,
        cltv_df: pd.DataFrame,
        cltv_col: str = "cltv_180d",
    ) -> tuple[pd.DataFrame, float]:
        """
        Add segmentation columns to the CLTV DataFrame.

        Adiciona colunas de segmentação ao DataFrame de CLTV.

        Columns added
        -------------
        cltv_segment             : Low / Mid / High / Top Value (quartile)
        probability_alive_group  : High / Low Probability Alive
        value_group              : High/Top Value vs Low/Mid Value
        marketing_action         : one of ACTION_ORDER

        Parameters
        ----------
        cltv_df  : DataFrame with columns ``cltv_col`` and ``probability_alive``
        cltv_col : column name for the CLTV metric

        Returns
        -------
        (segmented_df, alive_threshold_used)
        """
        required = {cltv_col, "probability_alive"}
        missing = required - set(cltv_df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        df = cltv_df.copy()

        n = len(df)
        if n >= 4:
            df["cltv_segment"] = pd.qcut(
                df[cltv_col].rank(method="first"),
                q=4,
                labels=SEGMENT_ORDER,
            )
        else:
            # Too few customers for quartile binning — assign by relative rank
            ranks = df[cltv_col].rank(method="first").astype(int) - 1  # 0-indexed
            seg_idx = (ranks * 4 // max(n, 1)).clip(0, 3).astype(int)
            df["cltv_segment"] = pd.Categorical(
                [SEGMENT_ORDER[i] for i in seg_idx],
                categories=SEGMENT_ORDER,
                ordered=True,
            )

        threshold = self.alive_threshold
        if threshold is None:
            threshold = float(df["probability_alive"].median())
        self._fitted_threshold = threshold

        df["probability_alive_group"] = np.where(
            df["probability_alive"] >= threshold,
            "High Probability Alive",
            "Low Probability Alive",
        )
        df["value_group"] = np.where(
            df["cltv_segment"].isin(["High Value", "Top Value"]),
            "High/Top Value",
            "Low/Mid Value",
        )

        conditions = [
            (df["value_group"] == "High/Top Value") & (df["probability_alive_group"] == "High Probability Alive"),
            (df["value_group"] == "High/Top Value") & (df["probability_alive_group"] == "Low Probability Alive"),
            (df["value_group"] == "Low/Mid Value") & (df["probability_alive_group"] == "High Probability Alive"),
            (df["value_group"] == "Low/Mid Value") & (df["probability_alive_group"] == "Low Probability Alive"),
        ]
        choices = ["Retain and Grow", "Reactivation Priority", "Low-Cost Nurture", "Low Priority"]
        df["marketing_action"] = np.select(conditions, choices, default="Review")

        logger.info(
            "Segmentation complete — threshold=%.4f | segments: %s",
            threshold,
            df["cltv_segment"].value_counts().to_dict(),
        )
        return df, threshold

    def build_decision_matrix(self) -> pd.DataFrame:
        """
        Return the 2×2 marketing-action decision matrix as a DataFrame.

        Retorna a matriz 2×2 de ações de marketing como DataFrame.
        """
        return pd.DataFrame(
            {
                "Low Probability Alive": ["Low Priority", "Reactivation Priority"],
                "High Probability Alive": ["Low-Cost Nurture", "Retain and Grow"],
            },
            index=["Low/Mid Value", "High/Top Value"],
        )

    def segment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate segment statistics by marketing action.

        Agrega estatísticas por ação de marketing.

        Returns
        -------
        DataFrame indexed by ACTION_ORDER with columns:
            customers, expected_revenue, avg_cltv, median_p_alive
        """
        cltv_col = next(
            (c for c in df.columns if c.startswith("cltv_") and c not in ("cltv_segment",)),
            None,
        )
        if cltv_col is None:
            raise ValueError("No CLTV column found (expected 'cltv_NNNd').")

        summary = (
            df.groupby("marketing_action", observed=True)
            .agg(
                customers=("marketing_action", "count"),
                expected_revenue=(cltv_col, "sum"),
                avg_cltv=(cltv_col, "mean"),
                median_p_alive=("probability_alive", "median"),
            )
            .reindex(ACTION_ORDER)
        )
        return summary
