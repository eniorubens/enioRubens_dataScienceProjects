"""Feature selection utilities.

Faithful port of notebook cells:
  [26/28] build_phik_significance_df + phik helpers
  [30]    VIF analysis block
  [55/57] MultivariateFeatureAnalysis class (impurity / permutation / SHAP / ablation)
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phik  # noqa: F401  # registers .phik_matrix()/.significance_matrix() accessors
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from src.i18n import localize_table, resolve_lang as _resolve_lang


# ---------------------------------------------------------------------------
# Phik helpers (cell [26/28])
# ---------------------------------------------------------------------------

PHIK_INTERVAL_COLS: List[str] = [
    "Rented Bike Count",
    "Hour",
    "Temperature(C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    # v4 (2015-2024 dataset) extra continuous weather columns — without these,
    # phik would treat them as categorical and mis-bin them.
    "Sunshine (hr)",
    "Cloud Cover (oktas)",
    "Ground Temp(C)",
    "DayNumberOnWeek",
]

# Date columns retained by the pipeline for time-series analysis but useless
# (and catastrophically slow) for phik — dropped by default in both phik functions.
_PHIK_DATE_COLS: List[str] = ["Date", "DateTime"]


def _existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def _phik_significance_matrix(df: pd.DataFrame, interval_cols: List[str]):
    """Compute significance matrix with FutureWarning suppressed."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The default of observed=False is deprecated.*",
            category=FutureWarning,
            module=r"phik\.significance",
        )
        return df.significance_matrix(interval_cols=interval_cols)


def _compute_phik_matrices(
    df: pd.DataFrame,
    interval_cols: List[str],
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop date columns, resolve interval_cols, return (phik_matrix, significance_matrix)."""
    data = df.drop(columns=drop_cols, errors="ignore")
    resolved = _existing_columns(data, interval_cols)
    return data.phik_matrix(interval_cols=resolved), _phik_significance_matrix(data, resolved)


def build_phik_significance_df(
    df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
    interval_cols: Optional[List[str]] = None,
    *,
    return_matrices: bool = False,
) -> pd.DataFrame:
    """Build a DataFrame with variable pairs, Phik correlation, and significance.

    Parameters
    ----------
    df:
        Input DataFrame (already feature-engineered).
    drop_cols:
        Columns to exclude before computing Phik. Defaults to ``["Date", "DateTime"]``
        because those high-cardinality datetime columns make phik intractable.
    interval_cols:
        Columns treated as continuous/ordinal. Defaults to ``PHIK_INTERVAL_COLS``.
    return_matrices:
        When True, return ``(pairs_df, phik_matrix_df, significance_matrix_df)`` so
        the caller can pass the pre-computed matrices to ``plot_phik_matrix`` and
        avoid running the expensive computation twice.

    Returns
    -------
    pd.DataFrame or tuple
        Sorted by phik descending; columns: var1, var2, phik, significance.
        If *return_matrices* is True, returns ``(pairs_df, phik_matrix, sig_matrix)``.
    """
    if drop_cols is None:
        drop_cols = _PHIK_DATE_COLS

    phik_values, significance_values = _compute_phik_matrices(
        df, interval_cols or PHIK_INTERVAL_COLS, drop_cols
    )

    records = []
    cols = phik_values.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            records.append(
                {
                    "var1": cols[i],
                    "var2": cols[j],
                    "phik": phik_values.iloc[i, j],
                    "significance": significance_values.iloc[i, j],
                }
            )

    result_df = pd.DataFrame(records).sort_values(by="phik", ascending=False)

    if return_matrices:
        return result_df, phik_values, significance_values
    return result_df


def plot_phik_matrix(
    df: Optional[pd.DataFrame] = None,
    interval_cols: Optional[List[str]] = None,
    lang=None,
    drop_cols: Optional[List[str]] = None,
    *,
    phik_matrix_df: Optional[pd.DataFrame] = None,
    significance_matrix_df: Optional[pd.DataFrame] = None,
):
    """Plot the Phik correlation and significance matrices.

    Pass *phik_matrix_df* and *significance_matrix_df* (returned by
    ``build_phik_significance_df(..., return_matrices=True)``) to skip the
    expensive matrix computation that would otherwise run a second time.
    """
    from phik.report import plot_correlation_matrix
    from src.plotting import set_graph_parameters

    lang = _resolve_lang(lang)
    labels = lang(
        {
            "phik_title": "Correlação Phik",
            "significance_title": "Significância dos coeficientes",
        }
    )

    if phik_matrix_df is not None and significance_matrix_df is not None:
        phik_view = phik_matrix_df
        significance_overview = significance_matrix_df
    else:
        if df is None:
            raise ValueError(
                "Provide either 'df' or both 'phik_matrix_df' and 'significance_matrix_df'."
            )
        if drop_cols is None:
            drop_cols = _PHIK_DATE_COLS
        phik_view, significance_overview = _compute_phik_matrices(
            df, interval_cols or PHIK_INTERVAL_COLS, drop_cols
        )

    with plt.style.context(["default"]):
        set_graph_parameters()

        plot_correlation_matrix(
            phik_view.values,
            x_labels=phik_view.columns,
            y_labels=phik_view.index,
            vmin=0,
            vmax=1,
            color_map="Greens",
            title=labels["phik_title"],
            fontsize_factor=1.5,
            figsize=(18, 9),
        )

        plot_correlation_matrix(
            significance_overview.fillna(0).values,
            x_labels=significance_overview.columns,
            y_labels=significance_overview.index,
            vmin=-5,
            vmax=5,
            title=labels["significance_title"],
            usetex=False,
            fontsize_factor=1.5,
            figsize=(18, 9),
        )
        plt.tight_layout()


# ---------------------------------------------------------------------------
# VIF analysis (cell [30])
# ---------------------------------------------------------------------------


def compute_vif(dataframe: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor for numerical features.

    A constant is added to each auxiliary regression so that the VIF follows
    the usual OLS specification with an intercept. The constant itself is not
    included in the returned table.

    Parameters
    ----------
    dataframe:
        DataFrame containing the features.
    features:
        Column names to include in the VIF computation.

    Returns
    -------
    pd.DataFrame
        Columns: feature, VIF.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    missing = [feature for feature in features if feature not in dataframe.columns]
    if missing:
        raise ValueError(f"Features not found in DataFrame: {missing}")

    X = dataframe[features].dropna().copy()
    if X.empty:
        raise ValueError("VIF cannot be computed because no complete rows remain")

    X.insert(0, "_intercept", 1.0)
    vif_data = pd.DataFrame(
        {
            "feature": features,
            "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(features))],
        }
    )
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# MultivariateFeatureAnalysis (cell [55/57])
# ---------------------------------------------------------------------------


class MultivariateFeatureAnalysis:
    """Perform multivariate explainability analysis with RandomForestRegressor.

    The workflow includes impurity-based feature importance, permutation
    importance, one-feature-drop ablation, and SHAP values.

    Parameters
    ----------
    target_col : str, default='Rented Bike Count'
        Target variable to be predicted.
    drop_cols : list[str] or None, default=None
        Columns removed before fitting the model.
    test_size : float, default=0.3
        Fraction of the latest observations reserved for holdout.
    random_state : int, default=42
        Random state used in model reproducibility and sampling.
    n_estimators : int, default=300
        Number of trees in the random forest.
    n_repeats : int, default=15
        Number of shuffles in permutation importance.
    scoring : str, default='neg_root_mean_squared_error'
        Scoring metric used in permutation importance.
    max_shap_samples : int, default=1000
        Maximum number of observations used for SHAP computation.
    """

    def __init__(
        self,
        target_col: str = "Rented Bike Count",
        drop_cols: Optional[List[str]] = None,
        test_size: float = 0.3,
        random_state: int = 42,
        n_estimators: int = 300,
        n_repeats: int = 15,
        scoring: str = "neg_root_mean_squared_error",
        max_shap_samples: int = 1000,
    ) -> None:
        self.target_col = target_col
        self.drop_cols = list(drop_cols) if drop_cols is not None else ["Date", "DateTime"]
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.n_repeats = int(n_repeats)
        self.scoring = scoring
        self.max_shap_samples = int(max_shap_samples)

        self.model_ = None
        self.label_encoders_: dict = {}
        self.feature_names_: List[str] = []

        self.X_train_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_test_ = None

        self.baseline_metrics_ = None
        self.impurity_importance_df_ = pd.DataFrame()
        self.permutation_importance_df_ = pd.DataFrame()
        self.ablation_importance_df_ = pd.DataFrame()
        self.shap_importance_df_ = pd.DataFrame()

        self.shap_values_ = None
        self.X_shap_ = None
        self.shap_available_: bool = False
        self.shap_note_: str = ""

    @staticmethod
    def _validate_columns(dataframe: pd.DataFrame, columns: List[str]) -> List[str]:
        """Return only columns that exist in the dataframe."""
        return [col for col in columns if col in dataframe.columns]

    def _encode_categorical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Label-encode object/category features in a copy of X."""
        X_encoded = X.copy(deep=True)
        categorical_cols = X_encoded.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in categorical_cols:
            encoder = LabelEncoder()
            X_encoded[col] = encoder.fit_transform(X_encoded[col].astype(str))
            self.label_encoders_[col] = encoder

        return X_encoded

    def _temporal_split(self, X: pd.DataFrame, y: pd.Series):
        """Split data into train and holdout preserving temporal order."""
        split_index = int(len(X) * (1 - self.test_size))
        split_index = min(max(split_index, 1), len(X) - 1)

        X_train = X.iloc[:split_index].copy()
        X_test = X.iloc[split_index:].copy()
        y_train = y.iloc[:split_index].copy()
        y_test = y.iloc[split_index:].copy()

        return X_train, X_test, y_train, y_test

    @staticmethod
    def _regression_metrics(y_true, y_pred) -> dict:
        """Compute regression metrics from predictions."""
        mse = mean_squared_error(y_true, y_pred)
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def _build_rf(self) -> RandomForestRegressor:
        """Instantiate RandomForestRegressor with configured parameters."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, dataframe: pd.DataFrame) -> "MultivariateFeatureAnalysis":
        """Fit baseline model and compute all explainability diagnostics."""
        df = dataframe.copy(deep=True).dropna(axis=0).reset_index(drop=True)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe.")

        drop_cols = self._validate_columns(df, self.drop_cols)
        df_model = df.drop(columns=drop_cols, errors="ignore")

        y = df_model[self.target_col].astype(float)
        X = df_model.drop(columns=[self.target_col]).copy()

        X_encoded = self._encode_categorical_columns(X)
        self.feature_names_ = X_encoded.columns.tolist()

        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = self._temporal_split(
            X_encoded, y
        )

        self.model_ = self._build_rf()
        self.model_.fit(self.X_train_, self.y_train_)

        y_pred = self.model_.predict(self.X_test_)
        self.baseline_metrics_ = pd.Series(
            self._regression_metrics(self.y_test_, y_pred),
            name="baseline_holdout",
        )

        self.impurity_importance_df_ = (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "impurity_importance": self.model_.feature_importances_,
                }
            )
            .sort_values("impurity_importance", ascending=False)
            .reset_index(drop=True)
        )

        permutation_result = permutation_importance(
            self.model_,
            self.X_test_,
            self.y_test_,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=self.scoring,
            n_jobs=-1,
        )

        self.permutation_importance_df_ = (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "permutation_importance_mean": permutation_result.importances_mean,
                    "permutation_importance_std": permutation_result.importances_std,
                }
            )
            .sort_values("permutation_importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        self.ablation_importance_df_ = self._run_ablation_study()
        self.shap_importance_df_ = self._compute_shap_importance()

        return self

    def _run_ablation_study(self) -> pd.DataFrame:
        """Run one-feature-drop ablation on holdout RMSE/MAE/R2."""
        baseline_rmse = self.baseline_metrics_["rmse"]
        baseline_mae = self.baseline_metrics_["mae"]
        baseline_r2 = self.baseline_metrics_["r2"]

        rows = []
        for feature in self.feature_names_:
            X_train_drop = self.X_train_.drop(columns=[feature])
            X_test_drop = self.X_test_.drop(columns=[feature])

            model_drop = self._build_rf()
            model_drop.fit(X_train_drop, self.y_train_)
            pred_drop = model_drop.predict(X_test_drop)

            metrics_drop = self._regression_metrics(self.y_test_, pred_drop)
            rows.append(
                {
                    "removed_feature": feature,
                    "rmse_without_feature": metrics_drop["rmse"],
                    "mae_without_feature": metrics_drop["mae"],
                    "r2_without_feature": metrics_drop["r2"],
                    "delta_rmse": metrics_drop["rmse"] - baseline_rmse,
                    "delta_mae": metrics_drop["mae"] - baseline_mae,
                    "delta_r2": metrics_drop["r2"] - baseline_r2,
                }
            )

        return pd.DataFrame(rows).sort_values("delta_rmse", ascending=False).reset_index(drop=True)

    def _compute_shap_importance(self) -> pd.DataFrame:
        """Compute SHAP importance table when shap is available."""
        try:
            import shap
        except Exception:
            self.shap_available_ = False
            lang = _resolve_lang(None)
            self.shap_note_ = lang(
                {
                    "shap_note": (
                        "SHAP não computado: o pacote 'shap' não está instalado neste ambiente."
                    )
                }
            )["shap_note"]
            warnings.warn(self.shap_note_)
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])

        self.shap_available_ = True

        X_shap = self.X_test_.copy()
        if len(X_shap) > self.max_shap_samples:
            X_shap = X_shap.sample(self.max_shap_samples, random_state=self.random_state)

        explainer = shap.TreeExplainer(self.model_)
        shap_values = explainer.shap_values(X_shap)

        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[0]
        if shap_array.shape[0] != X_shap.shape[0] and shap_array.shape[1] == X_shap.shape[0]:
            shap_array = shap_array.T

        mean_abs_shap = np.abs(shap_array).mean(axis=0)

        self.shap_values_ = shap_values
        self.X_shap_ = X_shap

        return (
            pd.DataFrame(
                {
                    "feature": X_shap.columns,
                    "mean_abs_shap": mean_abs_shap,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def get_feature_importance_table(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Combine impurity, permutation, and SHAP importances in one table."""
        table = self.impurity_importance_df_.merge(
            self.permutation_importance_df_,
            on="feature",
            how="outer",
        )

        if not self.shap_importance_df_.empty:
            table = table.merge(self.shap_importance_df_, on="feature", how="left")
        else:
            table["mean_abs_shap"] = np.nan

        table = table.sort_values("permutation_importance_mean", ascending=False).reset_index(
            drop=True
        )
        if top_n is not None:
            table = table.head(int(top_n)).copy()
        return table

    def split_report(self, years: pd.Series, lang=None) -> pd.DataFrame:
        """Localized train/holdout split summary (years, N, target moments, flag).

        ``years`` is the calendar year per row of the sample the model was fit on,
        in the same order as the fitted ``X`` (train rows followed by holdout).
        """
        lang = _resolve_lang(lang)
        labels = lang(
            {
                "split": "Divisão",
                "train": "Treino",
                "holdout": "Validação temporal",
                "years": "Anos",
                "n": "N",
                "target_mean": "Média do alvo",
                "target_median": "Mediana do alvo",
            }
        )
        years = pd.Series(np.asarray(years)).reset_index(drop=True)
        n_train = len(self.X_train_)
        train_years = years.iloc[:n_train]
        holdout_years = years.iloc[n_train:]
        return pd.DataFrame(
            {
                labels["split"]: [labels["train"], labels["holdout"]],
                labels["years"]: [
                    f"{train_years.min()}–{train_years.max()}",
                    f"{holdout_years.min()}–{holdout_years.max()}",
                ],
                labels["n"]: [len(self.X_train_), len(self.X_test_)],
                labels["target_mean"]: [self.y_train_.mean(), self.y_test_.mean()],
                labels["target_median"]: [self.y_train_.median(), self.y_test_.median()],
                "is_anomalous_2020 = 1": [
                    int(self.X_train_["is_anomalous_2020"].sum()),
                    int(self.X_test_["is_anomalous_2020"].sum()),
                ],
            }
        )

    def plot_feature_importance(self, kind: str = "permutation", top_n: int = 20, lang=None):
        """Plot selected importance ranking as a horizontal bar chart.

        Returns the plotted ``chart_df``; the caller decides when to show it.
        """
        lang = _resolve_lang(lang)
        label_map = lang(
            {
                "impurity": "Importância por impureza",
                "permutation": "Importância por permutação",
                "shap": "|SHAP| médio",
                "no_data": "Sem dados para plotar (kind={kind}).",
                "title": "Random Forest — {measure} (Top {n})",
            }
        )
        mapping = {
            "impurity": ("impurity_importance_df_", "impurity_importance"),
            "permutation": ("permutation_importance_df_", "permutation_importance_mean"),
            "shap": ("shap_importance_df_", "mean_abs_shap"),
        }
        if kind not in mapping:
            raise ValueError("kind must be one of: 'impurity', 'permutation', 'shap'.")

        df_name, value_col = mapping[kind]
        measure = label_map[kind]
        data = getattr(self, df_name)

        if data.empty:
            print(label_map["no_data"].format(kind=kind))
            if kind == "shap" and self.shap_note_:
                print(self.shap_note_)
            return None

        chart_df = data.head(int(top_n)).iloc[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(chart_df["feature"], chart_df[value_col], color="royalblue")
        plt.xlabel(measure)
        plt.title(label_map["title"].format(measure=measure, n=min(top_n, len(data))))
        plt.tight_layout()
        return chart_df

    def plot_ablation(self, top_n: int = 20, lang=None):
        """Plot one-feature-drop ablation impact using delta RMSE."""
        lang = _resolve_lang(lang)
        labels = lang(
            {
                "no_data": "Sem resultados de ablação. Execute fit() primeiro.",
                "xlabel": "Delta RMSE após remover a feature",
                "title": "Ablação por remoção de uma feature (Top {n})",
            }
        )
        if self.ablation_importance_df_.empty:
            print(labels["no_data"])
            return None

        chart_df = self.ablation_importance_df_.head(int(top_n)).iloc[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(chart_df["removed_feature"], chart_df["delta_rmse"], color="darkorange")
        plt.axvline(0.0, linestyle="--", color="black", linewidth=1)
        plt.xlabel(labels["xlabel"])
        plt.title(labels["title"].format(n=min(top_n, len(self.ablation_importance_df_))))
        plt.tight_layout()
        return chart_df

    def plot_shap_summary(self, max_display: int = 20, lang=None):
        """Plot SHAP beeswarm summary when SHAP values are available."""
        lang = _resolve_lang(lang)
        labels = lang(
            {
                "not_available": "Resumo SHAP indisponível.",
                "title": "Random Forest — resumo SHAP",
            }
        )
        if not self.shap_available_ or self.shap_values_ is None or self.X_shap_ is None:
            print(labels["not_available"])
            if self.shap_note_:
                print(self.shap_note_)
            return None

        import shap

        shap.summary_plot(self.shap_values_, self.X_shap_, max_display=max_display, show=False)
        plt.title(labels["title"])
        plt.tight_layout()
        return True


# ---------------------------------------------------------------------------
# Sampling and report builders (notebook 02)
# ---------------------------------------------------------------------------


def stratified_sample_by(
    df: pd.DataFrame, by: pd.Series, n_per_group: int, seed: int = 42
) -> pd.DataFrame:
    """Return a reproducible stratified sample of ``df``.

    Draws up to ``n_per_group`` rows from each level of ``by`` (index-aligned to
    ``df``), giving every group equal weight. Replaces the ``stratified_sample``
    helper that used to be defined inline in the notebook.
    """
    idx = (
        by.groupby(by)
        .apply(lambda s: s.sample(min(len(s), n_per_group), random_state=seed))
        .index.get_level_values(-1)
    )
    return df.loc[idx]


def prepare_rf_sample(
    dataframe: pd.DataFrame,
    year_meta: pd.Series,
    n_per_group: int = 1000,
    seed: int = 42,
    missing_label: str = "Missing",
) -> Tuple[pd.DataFrame, pd.Series, int]:
    """Stratified RF sample with categorical NaNs turned into an explicit category.

    ``Cloud Cover Cat`` is created before the numeric-median imputation step, so
    its remaining missing values are kept as an explicit ``missing_label``
    category instead of being silently dropped by
    ``MultivariateFeatureAnalysis.fit()`` (which calls ``dropna()``), which
    would otherwise shift the temporal train/holdout split.

    Returns ``(rf_sample, rf_sample_years, categorical_missing_count)``.
    """
    rf_sample = stratified_sample_by(dataframe, year_meta, n_per_group, seed=seed).copy()
    rf_sample_years = year_meta.loc[rf_sample.index].reset_index(drop=True)
    categorical_cols = rf_sample.select_dtypes(include=["object", "category"]).columns
    categorical_missing = int(rf_sample[categorical_cols].isna().sum().sum())
    rf_sample[categorical_cols] = rf_sample[categorical_cols].astype("object").fillna(missing_label)
    assert not rf_sample.isna().any().any()
    return rf_sample, rf_sample_years, categorical_missing


def filter_significant_phik_pairs(
    phik_pairs: pd.DataFrame,
    phik_threshold: float = 0.50,
    significance_threshold: float = 3.0,
) -> pd.DataFrame:
    """Keep pairs with strong and statistically significant association.

    Each pair appears once (``var1``/``var2``); sorted by phik then significance.
    """
    return (
        phik_pairs.loc[
            phik_pairs["phik"].ge(phik_threshold)
            & phik_pairs["significance"].ge(significance_threshold),
            ["var1", "var2", "phik", "significance"],
        ]
        .sort_values(["phik", "significance"], ascending=False)
        .reset_index(drop=True)
    )


def phik_pairs_with_target(phik_pairs: pd.DataFrame, target: str, lang=None) -> pd.DataFrame:
    """Pairs involving ``target``, with the other variable named in one column."""
    lang = _resolve_lang(lang)
    labels = lang({"other": "Outra variável", "significance": "significância"})
    pairs = phik_pairs[(phik_pairs.var1 == target) | (phik_pairs.var2 == target)].copy()
    pairs[labels["other"]] = np.where(pairs.var1 == target, pairs.var2, pairs.var1)
    return (
        pairs[[labels["other"], "phik", "significance"]]
        .rename(columns={"significance": labels["significance"]})
        .reset_index(drop=True)
    )


def compute_vif_report(dataframe: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """VIF table plus the auxiliary R2 (share of a feature's variance the others
    linearly reconstruct)."""
    vif_df = compute_vif(dataframe, features)
    vif_df["R2_auxiliar"] = 1 - 1 / vif_df["VIF"]
    return vif_df


# ---------------------------------------------------------------------------
# Presentation layer: PT-BR display copies. The internal English schema of the
# tables above (consumed by callers/tests) is never renamed in place; only the
# copy returned by the ``localize_*`` functions below is.
# ---------------------------------------------------------------------------

_PHIK_PAIRS_COLUMN_LABELS = {
    "var1": "Variável 1",
    "var2": "Variável 2",
    "significance": "significância",
}

_VIF_COLUMN_LABELS = {"feature": "Variável"}

_IMPORTANCE_COLUMN_LABELS = {
    "feature": "Variável",
    "impurity_importance": "Importância por impureza",
    "permutation_importance_mean": "Importância por permutação (média)",
    "permutation_importance_std": "Importância por permutação (desvio-padrão)",
    "mean_abs_shap": "|SHAP| médio",
}

_ABLATION_COLUMN_LABELS = {
    "removed_feature": "Variável removida",
    "rmse_without_feature": "RMSE sem a variável",
    "mae_without_feature": "MAE sem a variável",
    "r2_without_feature": "R² sem a variável",
    "delta_rmse": "Δ RMSE",
    "delta_mae": "Δ MAE",
    "delta_r2": "Δ R²",
}


def localize_phik_report(pairs: pd.DataFrame, lang=None) -> pd.DataFrame:
    """Cópia da tabela de pares Phik (``var1``/``var2``/``significance``) em PT-BR."""
    lang = _resolve_lang(lang)
    return localize_table(pairs, lang, columns=_PHIK_PAIRS_COLUMN_LABELS)


def localize_vif_report(vif_df: pd.DataFrame, lang=None) -> pd.DataFrame:
    """Cópia do relatório de VIF com o cabeçalho ``feature`` -> ``Variável``."""
    lang = _resolve_lang(lang)
    return localize_table(vif_df, lang, columns=_VIF_COLUMN_LABELS)


def localize_importance_table(table: pd.DataFrame, lang=None) -> pd.DataFrame:
    """Cópia da tabela de importância (impureza/permutação/SHAP) com cabeçalhos em PT."""
    lang = _resolve_lang(lang)
    return localize_table(table, lang, columns=_IMPORTANCE_COLUMN_LABELS)


def localize_ablation_table(table: pd.DataFrame, lang=None) -> pd.DataFrame:
    """Cópia da tabela de ablação (remoção de uma variável por vez) com cabeçalhos em PT."""
    lang = _resolve_lang(lang)
    return localize_table(table, lang, columns=_ABLATION_COLUMN_LABELS)
