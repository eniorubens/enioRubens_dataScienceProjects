"""
preprocessing.py — Load, clean, and transform Online Retail transactions into RFM matrices.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import get_logger, log_dataframe_summary

logger = get_logger(__name__)


class OnlineRetailPreprocessor:
    """
    Prepare Online Retail transactional data for CLTV modelling.

    Prepara dados transacionais Online Retail para modelagem CLTV.

    Designed for reuse: override ``load()`` to adapt to other transactional
    datasets that share the same structure (customer_id, date, value).
    """

    REQUIRED_COLUMNS = {"InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    OUTLIER_QUANTILE = 0.995
    CALIBRATION_SPLIT = 0.75

    # Column aliases: normalised → canonical
    _COLUMN_ALIASES = {
        "invoiceno": "InvoiceNo",
        "stockcode": "StockCode",
        "description": "Description",
        "quantity": "Quantity",
        "invoicedate": "InvoiceDate",
        "unitprice": "UnitPrice",
        "customerid": "CustomerID",
        "country": "Country",
    }

    def load(self, path: str | Path) -> pd.DataFrame:
        """
        Load a CSV file and normalise column names.

        Carrega um CSV e normaliza os nomes das colunas.

        Raises
        ------
        ValueError
            If any required column is missing after normalisation.
        """
        path = Path(path)
        logger.info("Loading data from %s", path)
        df = pd.read_csv(path)

        df = df.rename(
            columns={
                col: self._COLUMN_ALIASES.get(col.strip().lower(), col.strip())
                for col in df.columns
            }
        )

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        log_dataframe_summary(df, "raw", logger)
        return df

    def clean(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply the standard cleaning pipeline.

        Aplica o pipeline de limpeza padrão.

        Steps
        -----
        1. Parse InvoiceDate (format %m/%d/%y %H:%M)
        2. Remove rows with null CustomerID
        3. Remove cancellations (InvoiceNo starts with 'C')
        4. Remove Quantity <= 0 and UnitPrice <= 0
        5. Create TotalPrice = Quantity * UnitPrice
        6. Cap TotalPrice outliers at OUTLIER_QUANTILE

        Returns
        -------
        (df_clean, cleaning_summary)
            cleaning_summary has columns: step, rows, customers, rows_removed_from_previous_step
        """
        cleaned = df.copy()
        steps = []

        def snapshot(step: str) -> None:
            steps.append(
                {
                    "step": step,
                    "rows": cleaned.shape[0],
                    "customers": cleaned["CustomerID"].nunique(dropna=True),
                }
            )

        snapshot("raw")

        cleaned["InvoiceDate"] = pd.to_datetime(
            cleaned["InvoiceDate"],
            format="%m/%d/%y %H:%M",
            errors="coerce",
        )
        cleaned = cleaned.dropna(subset=["InvoiceDate"])
        snapshot("valid invoice dates")

        cleaned = cleaned.dropna(subset=["CustomerID"])
        snapshot("known customers")

        cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str)
        cleaned = cleaned.loc[~cleaned["InvoiceNo"].str.upper().str.startswith("C", na=False)]
        snapshot("cancellations removed")

        cleaned = cleaned.loc[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]
        snapshot("positive quantity and unit price")

        cleaned["CustomerID"] = cleaned["CustomerID"].astype(float).astype(int).astype(str)
        cleaned["TotalPrice"] = cleaned["Quantity"] * cleaned["UnitPrice"]

        upper_limit = cleaned["TotalPrice"].quantile(self.OUTLIER_QUANTILE)
        cleaned = cleaned.loc[cleaned["TotalPrice"].between(0, upper_limit, inclusive="right")].copy()
        snapshot(f"TotalPrice <= {self.OUTLIER_QUANTILE:.1%} percentile")

        summary = pd.DataFrame(steps)
        summary["rows_removed_from_previous_step"] = (
            summary["rows"].shift().sub(summary["rows"]).fillna(0).astype(int)
        )

        logger.info("Cleaning complete — %d rows, %d customers", cleaned.shape[0], cleaned["CustomerID"].nunique())
        return cleaned, summary

    def build_rfm(
        self,
        df: pd.DataFrame,
        observation_period_end: pd.Timestamp | None = None,
        time_unit: str = "D",
    ) -> pd.DataFrame:
        """
        Build an RFM matrix using pymc_marketing.clv.utils.rfm_summary().

        Constrói a matriz RFM usando pymc_marketing.clv.utils.rfm_summary().

        Output columns: customer_id, frequency, recency, T, monetary_value
        Customers with T == 0 are removed.

        Parameters
        ----------
        df                    : cleaned transaction DataFrame
        observation_period_end: if None, uses max(InvoiceDate)
        time_unit             : time unit for recency / T ('D' = days)
        """
        from pymc_marketing.clv.utils import rfm_summary  # type: ignore

        if observation_period_end is None:
            observation_period_end = df["InvoiceDate"].max()

        rfm = rfm_summary(
            transactions=df,
            customer_id_col="CustomerID",
            datetime_col="InvoiceDate",
            monetary_value_col="TotalPrice",
            observation_period_end=observation_period_end,
            time_unit=time_unit,
        )

        # rfm_summary returns customer_id as the index — normalise to a column
        if rfm.index.name == "CustomerID" or rfm.index.name == "customer_id":
            rfm = rfm.reset_index()
            rfm = rfm.rename(columns={rfm.columns[0]: "customer_id"})
        elif "customer_id" not in rfm.columns:
            rfm = rfm.reset_index()
            rfm.columns = ["customer_id"] + list(rfm.columns[1:])

        rfm = rfm.loc[rfm["T"] > 0].copy()
        rfm["customer_id"] = rfm["customer_id"].astype(str)

        logger.info(
            "RFM built — %d customers | repeat: %d | single: %d",
            len(rfm),
            (rfm["frequency"] > 0).sum(),
            (rfm["frequency"] == 0).sum(),
        )
        return rfm

    def calibration_holdout_split(
        self,
        df: pd.DataFrame,
        calibration_period_end: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a temporal calibration / holdout split for model validation.

        Realiza o split temporal calibration/holdout para validação do modelo.

        When ``calibration_period_end`` is None the split point is computed as
        ``CALIBRATION_SPLIT`` (75 %) of the observation period.

        Returns
        -------
        (calibration_rfm, holdout_rfm)
            Both DataFrames have columns:
                customer_id, frequency_cal, recency_cal, T_cal,
                monetary_value_cal, frequency_holdout, T_holdout
        """
        from pymc_marketing.clv.utils import rfm_train_test_split  # type: ignore

        obs_start = df["InvoiceDate"].min()
        obs_end = df["InvoiceDate"].max()

        if calibration_period_end is None:
            period_days = (obs_end - obs_start).days
            calibration_period_end = obs_start + pd.Timedelta(days=int(period_days * self.CALIBRATION_SPLIT))

        logger.info(
            "Calibration period: %s → %s | Holdout: %s → %s",
            obs_start.date(),
            calibration_period_end.date(),
            calibration_period_end.date(),
            obs_end.date(),
        )

        ch_data = rfm_train_test_split(
            transactions=df,
            customer_id_col="CustomerID",
            datetime_col="InvoiceDate",
            monetary_value_col="TotalPrice",
            train_period_end=calibration_period_end,
            test_period_end=obs_end,
        )

        # rfm_train_test_split returns customer_id as a column already.
        # Rename to the _cal/_holdout convention used by CLTVEvaluator.
        ch_data = ch_data.rename(
            columns={
                "frequency": "frequency_cal",
                "recency": "recency_cal",
                "T": "T_cal",
                "monetary_value": "monetary_value_cal",
                "test_frequency": "frequency_holdout",
                "test_T": "T_holdout",
            }
        )
        ch_data["customer_id"] = ch_data["customer_id"].astype(str)

        cal_cols = ["customer_id", "frequency_cal", "recency_cal", "T_cal", "monetary_value_cal"]
        hold_cols = ["customer_id", "frequency_holdout", "T_holdout"]

        calibration = ch_data[cal_cols].copy()
        holdout = ch_data[hold_cols].copy()

        return calibration, holdout

    def get_modeling_base(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to customers eligible for Gamma-Gamma modelling.

        Filtra para clientes elegíveis para o modelo Gamma-Gamma.

        Gamma-Gamma requires frequency > 0 and monetary_value > 0.
        """
        base = rfm.loc[(rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)].copy()
        excluded = len(rfm) - len(base)
        logger.info(
            "Modeling base: %d customers (excluded %d with frequency=0 or monetary_value<=0)",
            len(base),
            excluded,
        )
        return base

    def prepare(
        self,
        path: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: load → clean → build_rfm → get_modeling_base.

        Pipeline completo: load → clean → build_rfm → get_modeling_base.

        Returns
        -------
        (rfm_full, modeling_base, cleaning_summary)
        """
        df = self.load(path)
        df_clean, cleaning_summary = self.clean(df)
        rfm = self.build_rfm(df_clean)
        modeling_base = self.get_modeling_base(rfm)
        return rfm, modeling_base, cleaning_summary
