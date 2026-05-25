"""
cltv_model.py — BetaGeoModel + GammaGammaModel wrapper for end-to-end CLTV prediction.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


class CLTVModel:
    """
    Wrapper for BG/NBD (BetaGeoModel) + Gamma-Gamma (GammaGammaModel) from pymc-marketing.

    Wrapper para BG/NBD (BetaGeoModel) + Gamma-Gamma (GammaGammaModel) do pymc-marketing.

    NOT a sklearn Pipeline — the two models are sequentially dependent and
    the correct validation strategy is temporal, not cross-validation.

    NÃO é um sklearn Pipeline — os dois modelos são sequencialmente dependentes
    e a validação correta é temporal, não cross-validation.

    Parameters
    ----------
    fit_method      : 'map' (default, fast) or 'mcmc' (full Bayesian, slow)
    bgm_model_config: prior overrides for BetaGeoModel (empty = library defaults)
    gg_model_config : prior overrides for GammaGammaModel
    sampler_config  : kwargs forwarded to pm.sample() when fit_method='mcmc'
    """

    DEFAULT_BGM_CONFIG: dict = {}
    DEFAULT_GG_CONFIG: dict = {}
    DEFAULT_FIT_METHOD = "map"

    RFM_COLUMNS = {"customer_id", "frequency", "recency", "T", "monetary_value"}
    BGM_COLUMNS = {"customer_id", "frequency", "recency", "T"}
    GG_COLUMNS = {"customer_id", "frequency", "monetary_value"}

    def __init__(
        self,
        fit_method: str = "map",
        bgm_model_config: dict | None = None,
        gg_model_config: dict | None = None,
        sampler_config: dict | None = None,
    ) -> None:
        self.fit_method = fit_method
        self.bgm_model_config = bgm_model_config or dict(self.DEFAULT_BGM_CONFIG)
        self.gg_model_config = gg_model_config or dict(self.DEFAULT_GG_CONFIG)
        self.sampler_config = sampler_config or {}

        self.bgm: Any = None  # BetaGeoModel instance
        self.gg: Any = None   # GammaGammaModel instance
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, rfm: pd.DataFrame) -> "CLTVModel":
        """
        Train BetaGeoModel and GammaGammaModel sequentially.

        Treina BetaGeoModel e GammaGammaModel sequencialmente.

        Parameters
        ----------
        rfm : DataFrame with columns customer_id, frequency, recency, T, monetary_value

        Returns
        -------
        self (for method chaining)
        """
        from pymc_marketing.clv import BetaGeoModel, GammaGammaModel  # type: ignore

        self.validate_rfm_input(rfm)

        modeling_base = rfm.loc[(rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)].copy()
        logger.info(
            "Fitting on %d total customers | %d in modeling base",
            len(rfm),
            len(modeling_base),
        )

        # ── BG/NBD ──────────────────────────────────────────────────────────
        bgm_data = rfm[["customer_id", "frequency", "recency", "T"]].copy()
        logger.info("Fitting BetaGeoModel (fit_method=%s) …", self.fit_method)
        self.bgm = BetaGeoModel(
            model_config=self.bgm_model_config if self.bgm_model_config else None,
            sampler_config=self.sampler_config if self.sampler_config else None,
        )
        self.bgm.build_model(data=bgm_data)
        self.bgm.fit(method=self.fit_method, **self.sampler_config)
        bgm_params = self._extract_params(self.bgm)
        logger.info("BetaGeoModel params: %s", bgm_params)

        # ── Gamma-Gamma ──────────────────────────────────────────────────────
        gg_data = modeling_base[["customer_id", "frequency", "monetary_value"]].copy()
        logger.info("Fitting GammaGammaModel (fit_method=%s) …", self.fit_method)
        self.gg = GammaGammaModel(
            model_config=self.gg_model_config if self.gg_model_config else None,
            sampler_config=self.sampler_config if self.sampler_config else None,
        )
        self.gg.build_model(data=gg_data)
        self.gg.fit(method=self.fit_method, **self.sampler_config)
        gg_params = self._extract_params(self.gg)
        logger.info("GammaGammaModel params: %s", gg_params)

        self.metadata = {
            "fit_method": self.fit_method,
            "fit_date": datetime.now().isoformat(),
            "n_customers": int(len(rfm)),
            "n_modeling_base": int(len(modeling_base)),
            "bgm_params": bgm_params,
            "gg_params": gg_params,
        }
        return self

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict_purchases(
        self,
        rfm: pd.DataFrame,
        t: int = 180,
    ) -> pd.Series:
        """
        Predict expected number of purchases in ``t`` days per customer.

        Prediz o número esperado de compras em ``t`` dias para cada cliente.
        Uses BetaGeoModel.expected_num_purchases().
        """
        self._assert_fitted("predict_purchases")
        self.validate_rfm_input(rfm)

        bgm_data = rfm[["customer_id", "frequency", "recency", "T"]].copy()
        result = self.bgm.expected_purchases(data=bgm_data, future_t=t)
        return self._to_series(result, rfm["customer_id"])

    def predict_probability_alive(self, rfm: pd.DataFrame) -> pd.Series:
        """
        Predict the probability that each customer is still active.

        Prediz a probabilidade de cada cliente ainda estar ativo.
        Uses BetaGeoModel.expected_probability_alive().
        """
        self._assert_fitted("predict_probability_alive")
        self.validate_rfm_input(rfm)

        bgm_data = rfm[["customer_id", "frequency", "recency", "T"]].copy()
        result = self.bgm.expected_probability_alive(data=bgm_data)
        return self._to_series(result, rfm["customer_id"])

    def predict_expected_spend(self, rfm: pd.DataFrame) -> pd.Series:
        """
        Predict expected monetary value per future transaction.

        Prediz o valor monetário médio esperado por transação futura.
        Uses GammaGammaModel.expected_customer_spend().
        Requires frequency > 0 — raises ValueError for zero-frequency customers.
        """
        self._assert_fitted("predict_expected_spend")
        self.validate_rfm_input(rfm)

        if (rfm["frequency"] == 0).any():
            raise ValueError(
                "predict_expected_spend requires frequency > 0 for all customers. "
                "Filter to modeling_base first or use predict_cltv() which handles this."
            )
        gg_data = rfm[["customer_id", "frequency", "monetary_value"]].copy()
        result = self.gg.expected_customer_spend(data=gg_data)
        return self._to_series(result, rfm["customer_id"])

    def predict_cltv(
        self,
        rfm: pd.DataFrame,
        t: int = 180,
        discount_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Predict CLTV for all customers by combining BG/NBD + Gamma-Gamma.

        Prediz CLTV para todos os clientes combinando BG/NBD + Gamma-Gamma.

        Customers with frequency == 0 receive cltv == 0 and expected_spend == NaN
        (they have no repeat-purchase signal for the Gamma-Gamma model).

        Returns
        -------
        DataFrame with columns:
            customer_id, frequency, recency, T, monetary_value,
            predicted_purchases, probability_alive, expected_spend, cltv_{t}d
        """
        self._assert_fitted("predict_cltv")
        self.validate_rfm_input(rfm)

        cltv_col = f"cltv_{t}d"
        result = rfm.copy()

        # BG/NBD predictions for all customers
        result["predicted_purchases"] = self.predict_purchases(rfm, t=t).values
        result["probability_alive"] = self.predict_probability_alive(rfm).values

        # Gamma-Gamma only for modeling base (frequency > 0)
        modeling_mask = (rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)
        modeling_base = rfm.loc[modeling_mask].copy()

        result["expected_spend"] = np.nan
        result[cltv_col] = 0.0

        if len(modeling_base) > 0:
            gg_data = modeling_base[["customer_id", "frequency", "monetary_value"]].copy()

            raw_cltv = self.gg.expected_customer_lifetime_value(
                transaction_model=self.bgm,
                data=modeling_base[["customer_id", "frequency", "recency", "T", "monetary_value"]].copy(),
                future_t=t,
                time_unit="D",
                discount_rate=discount_rate,
            )
            cltv_series = self._to_series(raw_cltv, modeling_base["customer_id"])

            raw_spend = self.gg.expected_customer_spend(data=gg_data)
            spend_series = self._to_series(raw_spend, modeling_base["customer_id"])

            # Align by customer_id
            result = result.set_index("customer_id")
            result.loc[modeling_base["customer_id"].values, "expected_spend"] = spend_series.values
            result.loc[modeling_base["customer_id"].values, cltv_col] = cltv_series.values
            result = result.reset_index()

        logger.info(
            "CLTV predicted for %d customers | t=%d days | non-zero: %d",
            len(result),
            t,
            (result[cltv_col] > 0).sum(),
        )
        return result

    # ------------------------------------------------------------------
    # Parameters / metadata
    # ------------------------------------------------------------------

    def get_model_params(self) -> dict:
        """
        Return MAP parameters of both models as a serialisable dict.

        Retorna parâmetros MAP de ambos os modelos como dict serializável.
        Format: {"bgm": {...}, "gg": {...}}
        """
        self._assert_fitted("get_model_params")
        return {
            "bgm": self._extract_params(self.bgm),
            "gg": self._extract_params(self.gg),
        }

    def is_fitted(self) -> bool:
        """Return True if both bgm and gg have been trained."""
        return self.bgm is not None and self.gg is not None and getattr(self.bgm, "idata", None) is not None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dirpath: str | Path) -> None:
        """
        Save models and metadata to ``dirpath/``.

        Salva modelos e metadados em ``dirpath/``.

        Files created
        -------------
        bgm_model.nc   — BetaGeoModel idata (NetCDF via ArviZ)
        gg_model.nc    — GammaGammaModel idata (NetCDF via ArviZ)
        metadata.json  — fit_method, fit_date, params, n_customers

        NOTE: joblib/pickle are NOT used — PyMC objects are not safe to pickle.
        NOTA: joblib/pickle NÃO são usados — objetos PyMC não são seguros para pkl.
        """
        self._assert_fitted("save")
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        self.bgm.idata.to_netcdf(str(dirpath / "bgm_model.nc"))
        logger.info("Saved bgm_model.nc → %s", dirpath)

        self.gg.idata.to_netcdf(str(dirpath / "gg_model.nc"))
        logger.info("Saved gg_model.nc → %s", dirpath)

        with open(dirpath / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info("Saved metadata.json → %s", dirpath)

    @classmethod
    def load(cls, dirpath: str | Path) -> "CLTVModel":
        """
        Load previously saved models from ``dirpath/``.

        Carrega modelos salvos previamente de ``dirpath/``.

        Raises
        ------
        FileNotFoundError
            If bgm_model.nc, gg_model.nc, or metadata.json are missing.
        """
        from pymc_marketing.clv import BetaGeoModel, GammaGammaModel  # type: ignore

        dirpath = Path(dirpath)
        for fname in ("bgm_model.nc", "gg_model.nc", "metadata.json"):
            fpath = dirpath / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Expected file not found: {fpath}\n"
                    "Run train_model.py to generate model artefacts before loading."
                )

        with open(dirpath / "metadata.json") as f:
            metadata = json.load(f)

        instance = cls(fit_method=metadata.get("fit_method", "map"))
        instance.metadata = metadata

        # Reconstruct model instances with dummy data, then inject loaded idata.
        # Prediction methods use self.idata for parameters and the ``data`` argument
        # for customer-specific computation — the initialisation data is not reused.
        dummy_bgm = pd.DataFrame(
            {"customer_id": ["_"], "frequency": [1], "recency": [1.0], "T": [2.0]}
        )
        instance.bgm = BetaGeoModel()
        instance.bgm.build_model(data=dummy_bgm)
        instance.bgm.idata = az.from_netcdf(str(dirpath / "bgm_model.nc"))
        logger.info("Loaded bgm_model.nc from %s", dirpath)

        dummy_gg = pd.DataFrame(
            {"customer_id": ["_"], "frequency": [1], "monetary_value": [10.0]}
        )
        instance.gg = GammaGammaModel()
        instance.gg.build_model(data=dummy_gg)
        instance.gg.idata = az.from_netcdf(str(dirpath / "gg_model.nc"))
        logger.info("Loaded gg_model.nc from %s", dirpath)

        return instance

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_rfm_input(self, rfm: pd.DataFrame) -> None:
        """
        Validate that the DataFrame has the required columns and types.

        Valida que o DataFrame tem as colunas necessárias e tipos corretos.

        Raises
        ------
        ValueError
            With a descriptive message if validation fails.
        """
        required = {"customer_id", "frequency", "recency", "T", "monetary_value"}
        missing = required - set(rfm.columns)
        if missing:
            raise ValueError(
                f"RFM DataFrame is missing required columns: {sorted(missing)}. "
                f"Present columns: {sorted(rfm.columns)}"
            )
        if rfm.empty:
            raise ValueError("RFM DataFrame is empty.")

        numeric_cols = ["frequency", "recency", "T", "monetary_value"]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(rfm[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric, got {rfm[col].dtype}."
                )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self, method: str) -> None:
        if not self.is_fitted():
            raise RuntimeError(
                f"CLTVModel.{method}() called before fit(). "
                "Call fit() first or load a saved model with CLTVModel.load()."
            )

    @staticmethod
    def _extract_params(model: Any) -> dict:
        """Extract MAP parameter point estimates from a pymc-marketing model."""
        if model is None or not hasattr(model, "idata") or model.idata is None:
            return {}
        try:
            params = {}
            for var in model.idata.posterior.data_vars:
                val = float(np.array(model.idata.posterior[var].values).mean())
                params[str(var)] = round(val, 6)
            return params
        except Exception as exc:
            logger.warning("Could not extract params: %s", exc)
            return {}

    @staticmethod
    def _to_series(result: Any, customer_ids: pd.Series) -> pd.Series:
        """Convert an xarray DataArray or similar to a pandas Series."""
        try:
            # xr.DataArray path
            arr = np.array(result.values).flatten()
        except AttributeError:
            arr = np.array(result).flatten()

        ids = customer_ids.values
        if len(arr) == len(ids):
            return pd.Series(arr, index=ids, name="prediction")

        # Some pymc-marketing versions return mean across chains/draws
        if arr.ndim > 1:
            arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
        return pd.Series(arr.flatten(), index=ids, name="prediction")
