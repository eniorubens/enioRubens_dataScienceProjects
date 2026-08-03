"""Evaluation, residual diagnostics, temporal memory, and LOSO validation.

Faithful port of notebook cells:
  [81]  display_best_results
  [91]  regression_metrics, display_ranked
  [93]  feature_ablation_study (build_rf_pipeline, make_onehot_encoder, fit_predict_rf)
  [94]  leakage_audit
  [95]  build_winner_estimator, _normalized_target_to_raw, evaluate_loso_generalization
  [96]  plot_loso_winner_generalization
  [97]  render_loso_winner_insight
  [98]  TimeSeriesResidualAnalyzer
  [99]  build_temporal_memory_features, _find_column_transformer,
        extend_numeric_branch, evaluate_winner_variant
  [101] _fold_temporal_memory_frames
  [102] plot_loso_temporal_comparison
  [103] render_loso_temporal_memory_insight
  [104] plot_final_holdout

REGRA DE OURO: port fiel — nenhuma lógica alterada.
globals() (model_winner, transformer_winner, max_label, …) foram convertidos
em parâmetros explícitos para viabilizar o uso como módulo; o comportamento
é idêntico.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson

from src.cv import LeaveOneSeasonOut
from src.i18n import resolve_lang as _resolve_lang

logger = logging.getLogger(__name__)

# NOTE (i18n): módulo de modelagem, ainda não exercitado pelos três notebooks
# de EDA. Os textos-fonte enviados a lang(...) já são canônicos em PT-BR,
# como o restante do projeto (BASE_LANG="pt"); resolve_lang garante o
# fallback PT passthrough (sem chamadas de rede) quando lang não é fornecido.
# Nomes internos de coluna (Model, MAE, RMSE, R2, N, Feature, …) permanecem
# em inglês — são o contrato estável consumido por tests/código, não texto
# de apresentação; ver localize_report em src/stats_tests.py para o padrão
# de cópia localizada que este módulo deverá adotar quando ganhar notebook.


# ---------------------------------------------------------------------------
# Basic metrics helpers (cell [91])
# ---------------------------------------------------------------------------


def regression_metrics(y_true, y_pred, label: str) -> Dict[str, Any]:
    """Return a dict of MAE, MSE, RMSE, R2 for one prediction set."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)

    if not mask.any():
        return {"Model": label, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan, "N": 0}

    y_t = y_true_arr[mask]
    y_p = y_pred_arr[mask]
    mse = mean_squared_error(y_t, y_p)

    return {
        "Model": label,
        "MAE": mean_absolute_error(y_t, y_p),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_t, y_p),
        "N": int(mask.sum()),
    }


def display_ranked(
    df: pd.DataFrame,
    metric: str = "RMSE",
    ascending: bool = True,
    title: Optional[str] = None,
) -> pd.DataFrame:
    """Sort ``df`` by ``metric``, print an optional title, and display it."""
    out = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    if title:
        print(title)
    display(out)
    return out


# ---------------------------------------------------------------------------
# Display best results (cell [81])
# ---------------------------------------------------------------------------


def display_best_results(metric_dataframe: pd.DataFrame, top_n: int = 3, lang=None) -> pd.DataFrame:
    """Display the top-n entries from metric_dataframe ranked by Test R2.

    Returns the sorted copy so the notebook can use it downstream.
    """
    lang = _resolve_lang(lang)

    labels = lang(
        {
            "header": "### Os Melhores Resultados Tabulares (Pré-Memória)",
            "intro": (
                "Neste ponto do notebook, estes são os estimadores tabulares mais fortes e "
                "livres de leakage antes que features explícitas de memória temporal sejam "
                "introduzidas. Eles são suficientemente flexíveis para lidar com dados "
                "tabulares heterogêneos, com uma mistura de features categóricas e numéricas, "
                "desde que o número de amostras seja grande o bastante."
            ),
            "conclusion": (
                "Esses resultados são fortes sob holdout temporal, mas ainda são modelos "
                "tabulares sem memória explícita de demanda. As seções de resíduos e de "
                "memória temporal abaixo testam se este vencedor permanece o melhor candidato "
                "operacional quando informação de demanda defasada é permitida com segurança."
            ),
        }
    )

    metrics_copy = metric_dataframe.copy()
    metrics_copy = metrics_copy.sort_values(
        by="Test R2",
        ascending=False,
        key=lambda col: pd.to_numeric(col, errors="coerce"),
    )

    best = metrics_copy.head(top_n)

    lines = []
    for idx, row in best.iterrows():
        estimator_name = idx[0]
        optimization = idx[1]
        pre_process_pipeline = idx[2]
        lines.append(
            f"- {estimator_name}, {optimization}, {pre_process_pipeline} \n"
            f"  - `MAE` {row['Test MAE'] * 100:.4f}% and `R2` {row['Test R2'] * 100:.4f}%"
        )

    text = f"""
<span style="color:steelblue; text-align: left">

{labels["header"]}

</span>

{labels["intro"]}

{chr(10).join(lines)}

{labels["conclusion"]}"""

    display(Markdown(text))
    return best


# ---------------------------------------------------------------------------
# Learning curves — variance vs. capacity diagnosis for the GBM winners
#
# Not a port of an existing notebook cell: added to distinguish, for each of
# the Best-Tabular-Results winners, whether their train/test R2 gap is more
# fixable-with-more-data (variance) or more excess-model-capacity (needs
# regularization/depth reduction), by tracking train/val R2 as a function of
# training-set size.
# ---------------------------------------------------------------------------


def compute_learning_curves(
    estimator_pipelines: Dict[str, object],
    X_train_opt: pd.DataFrame,
    y_train_opt: pd.Series,
    ts_cv,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = "r2",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute sklearn learning curves per estimator pipeline using the temporal ``ts_cv``.

    ``TimeSeriesSplit.split()`` never shuffles — each fold's train indices are
    always a contiguous, temporally-ordered prefix of that fold's window — and
    ``learning_curve``'s default ``shuffle=False`` truncates each fold's train
    indices from the start, preserving that order. Safe to use unmodified for
    this time series (no adaptation needed for temporal ordering).

    ``estimator_pipelines`` should map an estimator label (e.g. "XGBRegressor")
    to its already-fitted (or unfitted) full pipeline — typically loaded from
    ``models/{estimator}_full_pipeline.pkl.gz`` and passed through ``clone()``
    internally, so the exact winning hyperparameters are reused with no risk
    of manual transcription error.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 8)

    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for name, pipeline in estimator_pipelines.items():
        sizes, train_scores, test_scores = learning_curve(
            clone(pipeline),
            X_train_opt,
            y_train_opt,
            cv=ts_cv,
            train_sizes=train_sizes,
            scoring=scoring,
            shuffle=False,
        )
        curves[name] = {
            "train_sizes": sizes,
            "train_scores": train_scores,
            "test_scores": test_scores,
        }
    return curves


def plot_learning_curves(
    curves: Dict[str, Dict[str, np.ndarray]],
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """One subplot per estimator: train/val R2 vs. training-set size, shaded std band."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "suptitle": "Curvas de Aprendizado — Diagnóstico de Variância vs. Capacidade",
            "train_r2": "R2 de treino",
            "val_r2": "R2 de validação",
            "xlabel": "Exemplos de treino",
            "ylabel": "R2",
        }
    )

    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    axes = axes[0]
    fig.suptitle(labels["suptitle"], fontsize=14, fontweight="bold")

    for ax, (name, curve) in zip(axes, curves.items()):
        sizes = curve["train_sizes"]
        train_mean = curve["train_scores"].mean(axis=1)
        train_std = curve["train_scores"].std(axis=1)
        test_mean = curve["test_scores"].mean(axis=1)
        test_std = curve["test_scores"].std(axis=1)

        ax.plot(sizes, train_mean, "o-", color="#4e79a7", label=labels["train_r2"])
        ax.fill_between(
            sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#4e79a7"
        )
        ax.plot(sizes, test_mean, "o-", color="#e15759", label=labels["val_r2"])
        ax.fill_between(
            sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#e15759"
        )

        ax.set_title(name, fontsize=12)
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig, axes


def render_learning_curve_insight(
    curves: Dict[str, Dict[str, np.ndarray]],
    gap_threshold: float = 0.03,
    lang=None,
) -> str:
    """Return a Markdown insight distinguishing variance from excess-capacity overfitting.

    Heuristic per estimator: if the validation score is still rising and/or
    the train/val gap is still shrinking at the largest training size tested,
    that is consistent with **variance** — more training data would plausibly
    help. If instead the train score stays high and roughly flat while the
    validation score plateaus early and the gap stays persistent regardless of
    training-set size, that is consistent with **excess model capacity** —
    more data is unlikely to close the gap on its own, and regularization or
    shallower trees is the more promising lever.
    """
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "variance": (
                "o escore de validação ainda está melhorando e/ou o gap treino/validação ainda "
                "está encolhendo no maior tamanho de treino testado — consistente com "
                "**variância**: mais dados de treino plausivelmente reduziriam o gap remanescente."
            ),
            "capacity": (
                "o escore de validação estabiliza cedo enquanto o escore de treino permanece "
                "alto, e o gap treino/validação permanece aproximadamente constante "
                "independentemente do tamanho do conjunto de treino — consistente com "
                "**capacidade excessiva do modelo**: mais dados dificilmente fecham esse gap "
                "sozinhos, e regularização/redução de profundidade é a alavanca mais promissora."
            ),
            "line": (
                "**{name}**: R2 de treino foi de {train_start:.3f} a {train_end:.3f}, e R2 de "
                "validação foi de {test_start:.3f} a {test_end:.3f} conforme a janela de treino "
                "cresceu de {size_start} a {size_end} linhas (gap {gap_start:.3f} → "
                "{gap_end:.3f}); {diagnosis}"
            ),
        }
    )

    lines = []
    for name, curve in curves.items():
        sizes = curve["train_sizes"]
        train_mean = curve["train_scores"].mean(axis=1)
        test_mean = curve["test_scores"].mean(axis=1)
        gap = train_mean - test_mean

        gap_shrinking = gap[-1] < gap[0] - gap_threshold
        midpoint = len(test_mean) // 2
        val_still_rising = (test_mean[-1] - test_mean[midpoint]) > gap_threshold

        diagnosis = (
            labels["variance"] if (gap_shrinking or val_still_rising) else labels["capacity"]
        )

        lines.append(
            labels["line"].format(
                name=name,
                train_start=train_mean[0],
                train_end=train_mean[-1],
                test_start=test_mean[0],
                test_end=test_mean[-1],
                size_start=int(sizes[0]),
                size_end=int(sizes[-1]),
                gap_start=gap[0],
                gap_end=gap[-1],
                diagnosis=diagnosis,
            )
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Feature ablation (cell [93])
# ---------------------------------------------------------------------------


def make_onehot_encoder() -> OneHotEncoder:
    """Return a OneHotEncoder with ``sparse_output`` keyword guarded for sklearn compat."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_rf_pipeline(df_train: pd.DataFrame) -> Pipeline:
    """Build a RandomForest pipeline that handles mixed num/cat columns."""
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_train.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2)
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", reg)])


def fit_predict_rf(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
) -> np.ndarray:
    """Fit a fresh RF pipeline and return predictions on X_te."""
    pipe = build_rf_pipeline(X_tr)
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def feature_ablation_study(
    X_train_opt: pd.DataFrame,
    y_train_raw: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout_raw: pd.Series,
    max_label: float,
    lang=None,
) -> Dict[str, pd.DataFrame]:
    """Run feature-ablation and target-scale sensitivity diagnostics.

    Port of notebook cell [93].

    Returns a dict with keys:
        'ablation': ablation_df
        'target_scale': target_scale_df
        'cause2_consolidated': cause2_consolidated
    """
    lang = _resolve_lang(lang)

    titles = lang(
        {
            "title_ablation": "Ablação no Holdout (Causa 2)",
            "title_target": "Sensibilidade à Escala do Alvo (Causa 3)",
            "title_consolidated": "Ranking Consolidado da Causa 2 (Holdout)",
        }
    )

    temporal_candidates = [
        "Hour",
        "Month",
        "Year",
        "DayNumberOnWeek",
        "Weekday",
        "WeekStatus",
        "Time_Period",
        "Rush_Hour",
        "Holiday",
        "Functioning Day",
        "Seasons",
    ]
    temporal_cols = [c for c in temporal_candidates if c in X_train_opt.columns]
    non_temporal_cols = [c for c in X_train_opt.columns if c not in temporal_cols]

    feature_sets = {
        "ML_Full_Features": X_train_opt.columns.tolist(),
        "ML_Temporal_Only": temporal_cols,
        "ML_No_Temporal": non_temporal_cols,
    }

    ablation_rows = []
    for name, cols in feature_sets.items():
        if not cols:
            continue
        pred = fit_predict_rf(X_train_opt[cols], y_train_raw, X_holdout[cols])
        ablation_rows.append(regression_metrics(y_holdout_raw, pred, name))

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df = display_ranked(
        ablation_df, metric="RMSE", ascending=True, title=titles["title_ablation"]
    )

    y_train_norm = y_train_raw / max_label
    pred_raw_target = fit_predict_rf(X_train_opt, y_train_raw, X_holdout)
    pred_norm_target = fit_predict_rf(X_train_opt, y_train_norm, X_holdout) * max_label

    target_scale_df = pd.DataFrame(
        [
            regression_metrics(y_holdout_raw, pred_raw_target, "Target_RAW_Counts"),
            regression_metrics(y_holdout_raw, pred_norm_target, "Target_Normalized_x_MaxLabel"),
        ]
    )
    target_scale_df = display_ranked(
        target_scale_df, metric="RMSE", ascending=True, title=titles["title_target"]
    )

    cause2_consolidated = pd.concat(
        [
            ablation_df[["Model", "MAE", "RMSE", "R2"]].copy(),
            target_scale_df[["Model", "MAE", "RMSE", "R2"]].copy(),
        ],
        axis=0,
        ignore_index=True,
    )
    display_ranked(
        cause2_consolidated, metric="RMSE", ascending=True, title=titles["title_consolidated"]
    )

    return {
        "ablation": ablation_df,
        "target_scale": target_scale_df,
        "cause2_consolidated": cause2_consolidated,
    }


# ---------------------------------------------------------------------------
# Leakage audit (cell [94])
# ---------------------------------------------------------------------------


def leakage_audit(
    transformed_df1: pd.DataFrame, target_col: str = "Rented Bike Count"
) -> pd.DataFrame:
    """Compute Spearman association of each feature vs current and lagged target.

    Port of notebook cell [94]. Returns leakage_audit_df sorted by
    descending association with current target.
    """
    from scipy.stats import spearmanr

    def _numeric_view(series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.astype("int64")
        if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == "object":
            return series.astype("category").cat.codes.replace(-1, np.nan)
        return pd.to_numeric(series, errors="coerce")

    current_target = transformed_df1[target_col]
    lagged_target = current_target.shift(1)

    rows = []
    for feature in transformed_df1.columns:
        if feature == target_col:
            continue

        x = _numeric_view(transformed_df1[feature])

        current_pair = pd.concat([x, current_target], axis=1).dropna()
        lagged_pair = pd.concat([x, lagged_target], axis=1).dropna()

        current_assoc = (
            abs(spearmanr(current_pair.iloc[:, 0], current_pair.iloc[:, 1]).correlation)
            if len(current_pair) > 2
            else np.nan
        )
        lagged_assoc = (
            abs(spearmanr(lagged_pair.iloc[:, 0], lagged_pair.iloc[:, 1]).correlation)
            if len(lagged_pair) > 2
            else np.nan
        )

        rows.append(
            {
                "Feature": feature,
                "Association with current target": current_assoc,
                "Association with lagged target": lagged_assoc,
                "Current-minus-lagged gap": current_assoc - lagged_assoc,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["Association with current target", "Current-minus-lagged gap"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Winner estimator factory (cells [95], [99] — identical definition)
# ---------------------------------------------------------------------------


def build_winner_estimator(
    preprocessing_pipeline,
    model_winner,
    transformer_winner,
) -> Pipeline:
    """Assemble a full sklearn Pipeline from winning preprocessing + model.

    Port of notebook cells [95] and [99] (same body in both).
    """
    return Pipeline(
        steps=[
            ("features", clone(preprocessing_pipeline)),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=clone(model_winner),
                    transformer=clone(transformer_winner),
                ),
            ),
        ]
    )


def _normalized_target_to_raw(y_normalized: pd.Series, target_scale: float) -> pd.Series:
    return pd.Series(
        y_normalized.to_numpy(dtype=float) * target_scale,
        index=y_normalized.index,
        name="target_raw",
    )


# ---------------------------------------------------------------------------
# evaluate_loso_generalization (cell [95])
# ---------------------------------------------------------------------------


def evaluate_loso_generalization(
    label: str,
    X_base: pd.DataFrame,
    y_base: pd.Series,
    preprocessing_pipeline,
    cv,
    target_scale: float,
    model_winner,
    transformer_winner,
    temporal_memory: bool = False,
    align_to_memory_safe_rows: bool = False,
    verbose: bool = True,
    lang=None,
) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate one model variant with Leave-One-Season-Out validation.

    Port of notebook cell [95].

    When ``temporal_memory=False`` only the current winner is used.
    When ``temporal_memory=True`` the same function is reused after temporal
    memory features are formally validated (cell [101] calls).
    """
    lang = _resolve_lang(lang)

    verbose_labels = lang({"train_label": "treino", "test_label": "teste"})

    target_raw = _normalized_target_to_raw(y_base, target_scale)
    season_names = list(getattr(cv, "SEASONS", {}).keys())
    results = []
    fold_predictions = {}

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_base)):
        season = season_names[fold_id] if fold_id < len(season_names) else f"Fold_{fold_id + 1}"
        fold_train_idx = train_idx.copy()
        fold_preprocessing = preprocessing_pipeline
        X_train_source = X_base
        X_test_source = X_base
        temporal_cols: List[str] = []

        if temporal_memory or align_to_memory_safe_rows:
            (
                X_memory_train,
                X_memory_test,
                temporal_cols,
                complete_train_mask,
            ) = _fold_temporal_memory_frames(
                X_base=X_base,
                target_raw=target_raw,
                test_idx=test_idx,
            )
            fold_train_idx = train_idx[complete_train_mask[train_idx]]

            if temporal_memory:
                X_train_source = X_memory_train
                X_test_source = X_memory_test
                fold_preprocessing = extend_numeric_branch(preprocessing_pipeline, temporal_cols)

        if len(fold_train_idx) == 0:
            raise ValueError(
                f"No training rows remained for {label} / {season}. "
                "Check the LOSO filtering logic."
            )

        X_tr = X_train_source.iloc[fold_train_idx].copy()
        X_ts = X_test_source.iloc[test_idx].copy()
        y_tr = y_base.iloc[fold_train_idx].copy()
        y_ts_raw = target_raw.iloc[test_idx].copy()

        estimator = build_winner_estimator(fold_preprocessing, model_winner, transformer_winner)
        estimator.fit(X_tr, y_tr)
        y_pred_raw = estimator.predict(X_ts) * target_scale
        mse = mean_squared_error(y_ts_raw, y_pred_raw)

        results.append(
            {
                "Model": label,
                "Season": season,
                "n_train": len(fold_train_idx),
                "n_test": len(test_idx),
                "MAE": mean_absolute_error(y_ts_raw, y_pred_raw),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2_score(y_ts_raw, y_pred_raw),
                "Temporal Memory": temporal_memory,
                "Memory Features": len(temporal_cols),
            }
        )
        fold_predictions[(label, season)] = pd.Series(
            y_pred_raw, index=X_base.iloc[test_idx].index, name="prediction"
        )

        if verbose:
            r = results[-1]
            print(
                f"  {label:<24} {season:<8} "
                f"MAE={r['MAE']:.1f}  RMSE={r['RMSE']:.1f}  R2={r['R2']:.3f}  "
                f"{verbose_labels['train_label']}={r['n_train']}  "
                f"{verbose_labels['test_label']}={r['n_test']}"
            )

    return pd.DataFrame(results), fold_predictions


# ---------------------------------------------------------------------------
# plot_loso_winner_generalization (cell [96])
# ---------------------------------------------------------------------------


def plot_loso_winner_generalization(
    seasonal_generalization_df: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Bar-chart of MAE / RMSE / R2 per season for the LOSO current-winner run."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "suptitle": "Leave-One-Season-Out - Vencedor Atual",
            "mean": "Média={mean_val:.2f}",
        }
    )
    season_order = list(LeaveOneSeasonOut.SEASONS.keys())
    plot_df = seasonal_generalization_df.copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(labels["suptitle"], fontsize=14, fontweight="bold")

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        metric_series = plot_df.set_index("Season").reindex(season_order)[metric]
        metric_series.plot(kind="bar", ax=ax, color=colors, edgecolor="white", rot=30)
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("")
        mean_val = metric_series.mean()
        ax.axhline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=labels["mean"].format(mean_val=mean_val),
        )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# render_loso_winner_insight (cell [97])
# ---------------------------------------------------------------------------


def render_loso_winner_insight(
    results_df: pd.DataFrame,
    model_label: str = "Current_XGB_winner",
    lang=None,
) -> str:
    """Return a Markdown insight string for the LOSO current-winner evaluation."""
    lang = _resolve_lang(lang)

    labels = lang(
        {
            "para1": (
                "A validação leave-one-season-out é um teste de robustez mais rigoroso do que o "
                "holdout temporal regular, pois o modelo é obrigado a prever um regime sazonal "
                "completo que não foi observado durante o ajuste. Neste ponto do notebook, apenas "
                "o vencedor atual foi introduzido, então esta seção é intencionalmente limitada ao "
                "pipeline vencedor `XGBRegressor` e ainda não avalia features de memória temporal."
            ),
            "business_para": (
                "A interpretação de negócio deve ser lida por estação, e não apenas pela média. "
                "Um resultado forte no holdout temporal pode coexistir com transferência sazonal "
                "fraca, se a estação retida se comportar de forma diferente das estações "
                "disponíveis durante o ajuste. Portanto, esta validação atua como uma checagem de "
                "governança para a robustez sazonal: o vencedor atual só pode ser considerado "
                "confiável para regimes sazonais adequadamente representados nos dados históricos "
                "de treino, enquanto a estação mais fraca deve receber atenção diagnóstica "
                "adicional antes do uso em produção."
            ),
            "season_sentence": (
                "Para {season}, o modelo produziu MAE = {mae:.2f}, RMSE = {rmse:.2f} e R2 = "
                "{r2:.3f}."
            ),
            "summary_para": (
                "Entre as estações, o vencedor atual alcançou MAE médio de **{mae:.2f}**, RMSE "
                "médio de **{rmse:.2f}** e R2 médio de **{r2:.3f}**. {season_sentences} A "
                "transferência sazonal mais forte foi observada em **{best_r2_season}** pelo R2, "
                "enquanto a mais fraca foi observada em **{worst_r2_season}**. Em termos de "
                "escala de erro, **{best_rmse_season}** teve o menor RMSE e "
                "**{worst_rmse_season}** teve o maior RMSE."
            ),
        }
    )

    model_df = results_df[results_df["Model"] == model_label].copy()
    season_order = list(LeaveOneSeasonOut.SEASONS.keys())
    by_season = model_df.set_index("Season").reindex(season_order)
    summary = model_df[["MAE", "RMSE", "R2"]].mean()

    best_r2_season = by_season["R2"].idxmax()
    worst_r2_season = by_season["R2"].idxmin()
    best_rmse_season = by_season["RMSE"].idxmin()
    worst_rmse_season = by_season["RMSE"].idxmax()

    season_sentences = [
        labels["season_sentence"].format(
            season=season, mae=row["MAE"], rmse=row["RMSE"], r2=row["R2"]
        )
        for season, row in by_season.iterrows()
    ]

    summary_para = labels["summary_para"].format(
        mae=summary["MAE"],
        rmse=summary["RMSE"],
        r2=summary["R2"],
        season_sentences=" ".join(season_sentences),
        best_r2_season=best_r2_season,
        worst_r2_season=worst_r2_season,
        best_rmse_season=best_rmse_season,
        worst_rmse_season=worst_rmse_season,
    )

    return f"{labels['para1']}\n\n{summary_para}\n\n{labels['business_para']}"


# ---------------------------------------------------------------------------
# TimeSeriesResidualAnalyzer (cell [98])
# ---------------------------------------------------------------------------


class TimeSeriesResidualAnalyzer:
    """Run residual diagnostics for time-series regression predictions.

    Parameters
    ----------
    y_true : array-like
        Observed target values in the original scale.
    y_pred : array-like
        Predicted target values in the original scale.
    model_name : str
        Label used in titles and summary outputs.
    max_lags : int
        Maximum lag displayed in ACF and PACF charts.
    """

    def __init__(self, y_true, y_pred, model_name: str = "Model", max_lags: int = 72) -> None:
        self.model_name = model_name
        self.max_lags = int(max_lags)
        self.data = self._build_residual_frame(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _to_1d_float_array(values, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float).ravel()
        if array.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        return array

    def _build_residual_frame(self, y_true, y_pred) -> pd.DataFrame:
        observed = self._to_1d_float_array(y_true, "y_true")
        fitted = self._to_1d_float_array(y_pred, "y_pred")

        if observed.size != fitted.size:
            raise ValueError("y_true and y_pred must have the same number of observations.")

        finite_mask = np.isfinite(observed) & np.isfinite(fitted)
        if finite_mask.sum() < 10:
            raise ValueError("At least 10 finite observations are required for diagnostics.")

        frame = pd.DataFrame(
            {"observed": observed[finite_mask], "fitted": fitted[finite_mask]}
        ).reset_index(drop=True)
        frame["residual"] = frame["observed"] - frame["fitted"]
        return frame

    def _resolved_lags(self) -> int:
        n_obs = len(self.data)
        pacf_limit = max(1, (n_obs // 2) - 1)
        return min(self.max_lags, n_obs - 1, pacf_limit)

    def summary(self) -> pd.Series:
        """Return residual summary statistics useful for model checks."""
        residuals = self.data["residual"].to_numpy()
        return pd.Series(
            {
                "n_obs": residuals.size,
                "mean_residual": residuals.mean(),
                "std_residual": residuals.std(ddof=1),
                "mae_residual": np.mean(np.abs(residuals)),
                "rmse_residual": np.sqrt(np.mean(np.square(residuals))),
                "durbin_watson": durbin_watson(residuals),
            },
            name=self.model_name,
        )

    def plot(self, figsize: Tuple[int, int] = (16, 10), lang=None) -> Tuple[plt.Figure, np.ndarray]:
        """Plot Residuals vs Fitted, Q-Q, ACF, and PACF diagnostics."""
        lang = _resolve_lang(lang)
        labels = lang(
            {
                "residuals_vs_fitted": "{model} - Resíduos vs. Ajustados",
                "fitted_values": "Valores ajustados",
                "residuals": "Resíduos",
                "qq_plot": "{model} - Gráfico Q-Q dos resíduos",
                "acf": "{model} - ACF dos resíduos (até a defasagem {lags})",
                "pacf": "{model} - PACF dos resíduos (até a defasagem {lags})",
            }
        )
        residuals = self.data["residual"].to_numpy()
        fitted = self.data["fitted"].to_numpy()
        lags = self._resolved_lags()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        axes[0, 0].scatter(fitted, residuals, alpha=0.45, s=20)
        axes[0, 0].axhline(0.0, color="red", linestyle="--", linewidth=1)
        axes[0, 0].set_title(labels["residuals_vs_fitted"].format(model=self.model_name))
        axes[0, 0].set_xlabel(labels["fitted_values"])
        axes[0, 0].set_ylabel(labels["residuals"])

        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title(labels["qq_plot"].format(model=self.model_name))

        plot_acf(residuals, lags=lags, ax=axes[1, 0], alpha=0.05, zero=False)
        axes[1, 0].set_title(labels["acf"].format(model=self.model_name, lags=lags))

        plot_pacf(residuals, lags=lags, ax=axes[1, 1], alpha=0.05, method="ywm", zero=False)
        axes[1, 1].set_title(labels["pacf"].format(model=self.model_name, lags=lags))

        fig.tight_layout()
        plt.show()
        return fig, axes


# ---------------------------------------------------------------------------
# Temporal memory features (cell [99])
# ---------------------------------------------------------------------------


def build_temporal_memory_features(
    X_base: pd.DataFrame,
    target_raw: pd.Series,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create leakage-safe lag and rolling demand features.

    Port of notebook cell [99].
    """
    X_memory = X_base.copy()
    target_history = pd.Series(
        target_raw.to_numpy(dtype=float), index=X_base.index, name="target_history"
    )

    X_memory["lag_1h"] = target_history.shift(1)
    X_memory["lag_24h"] = target_history.shift(24)
    X_memory["lag_168h"] = target_history.shift(168)

    shifted_target = target_history.shift(1)
    X_memory["rolling_mean_3h"] = shifted_target.rolling(window=3, min_periods=3).mean()
    X_memory["rolling_mean_6h"] = shifted_target.rolling(window=6, min_periods=6).mean()
    X_memory["rolling_mean_24h"] = shifted_target.rolling(window=24, min_periods=24).mean()
    X_memory["rolling_mean_168h"] = shifted_target.rolling(window=168, min_periods=168).mean()
    X_memory["rolling_std_24h"] = shifted_target.rolling(window=24, min_periods=24).std()
    X_memory["rolling_std_168h"] = shifted_target.rolling(window=168, min_periods=168).std()

    if {"DayNumberOnWeek", "Hour"}.issubset(X_memory.columns):
        day_number = pd.to_numeric(X_memory["DayNumberOnWeek"], errors="coerce")
        hour = pd.to_numeric(X_memory["Hour"], errors="coerce")
        hour_of_week = day_number * 24 + hour
    else:
        hour_of_week = pd.Series(np.arange(len(X_memory)) % 168, index=X_memory.index, dtype=float)

    X_memory["hour_of_week"] = hour_of_week
    X_memory["hour_of_week_sin"] = np.sin(2 * np.pi * hour_of_week / 168)
    X_memory["hour_of_week_cos"] = np.cos(2 * np.pi * hour_of_week / 168)

    temporal_memory_cols = [
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_3h",
        "rolling_mean_6h",
        "rolling_mean_24h",
        "rolling_mean_168h",
        "rolling_std_24h",
        "rolling_std_168h",
        "hour_of_week",
        "hour_of_week_sin",
        "hour_of_week_cos",
    ]
    return X_memory, temporal_memory_cols


def _find_column_transformer(estimator) -> Optional[ColumnTransformer]:
    """Recursively locate the first ColumnTransformer inside an estimator."""
    if isinstance(estimator, ColumnTransformer):
        return estimator
    if hasattr(estimator, "steps"):
        for _, step in estimator.steps:
            found = _find_column_transformer(step)
            if found is not None:
                return found
    return None


def extend_numeric_branch(preprocessing_pipeline, extra_numeric_features: List[str]):
    """Add temporal memory columns to the winner's numeric preprocessing branch.

    Port of notebook cell [99].
    """
    extended_pipeline = clone(preprocessing_pipeline)
    column_transformer = _find_column_transformer(extended_pipeline)

    if column_transformer is None:
        raise TypeError("No ColumnTransformer was found inside the winning preprocessing pipeline.")

    updated_transformers = []
    numeric_branch_updated = False

    for name, transformer, columns in column_transformer.transformers:
        if name == "num":
            current_columns = list(columns)
            current_columns.extend(
                [col for col in extra_numeric_features if col not in current_columns]
            )
            updated_transformers.append((name, transformer, current_columns))
            numeric_branch_updated = True
        else:
            updated_transformers.append((name, transformer, columns))

    if not numeric_branch_updated:
        raise ValueError("The winning pipeline does not expose a transformer named 'num'.")

    column_transformer.transformers = updated_transformers
    return extended_pipeline


def evaluate_winner_variant(
    label: str,
    preprocessing_pipeline,
    X_train_eval: pd.DataFrame,
    y_train_eval: pd.Series,
    X_test_eval: pd.DataFrame,
    y_test_raw: pd.Series,
    model_winner,
    transformer_winner,
    max_label: float,
) -> Tuple[Pipeline, np.ndarray, Dict[str, Any]]:
    """Fit a winner variant and return (estimator, predictions, metrics_dict).

    Port of notebook cell [99].
    """
    estimator = build_winner_estimator(preprocessing_pipeline, model_winner, transformer_winner)
    estimator.fit(X_train_eval, y_train_eval)
    pred_raw = estimator.predict(X_test_eval) * max_label
    residuals = y_test_raw.to_numpy(dtype=float) - pred_raw
    mse = mean_squared_error(y_test_raw, pred_raw)

    return (
        estimator,
        pred_raw,
        {
            "Model": label,
            "MAE": mean_absolute_error(y_test_raw, pred_raw),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test_raw, pred_raw),
            "Mean Residual": residuals.mean(),
            "Residual RMSE": np.sqrt(np.mean(np.square(residuals))),
            "Durbin-Watson": durbin_watson(residuals),
            "N": len(y_test_raw),
        },
    )


# ---------------------------------------------------------------------------
# _fold_temporal_memory_frames (cell [101])
# ---------------------------------------------------------------------------


def _fold_temporal_memory_frames(
    X_base: pd.DataFrame,
    target_raw: pd.Series,
    test_idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray]:
    """Build memory-safe training and test frames for one LOSO fold.

    Training memory is built with the withheld season masked to avoid
    target leakage into lag windows. Port of notebook cell [101].
    """
    train_target_history = target_raw.copy()
    train_target_history.iloc[test_idx] = np.nan

    X_memory_train, temporal_memory_cols = build_temporal_memory_features(
        X_base, train_target_history
    )
    X_memory_test, _ = build_temporal_memory_features(X_base, target_raw)
    complete_train_mask = X_memory_train[temporal_memory_cols].notna().all(axis=1).to_numpy()

    return X_memory_train, X_memory_test, temporal_memory_cols, complete_train_mask


# ---------------------------------------------------------------------------
# plot_loso_temporal_comparison (cell [102])
# ---------------------------------------------------------------------------


def plot_loso_temporal_comparison(
    seasonal_temporal_memory_comparison_df: pd.DataFrame,
    seasonal_temporal_memory_summary_df: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Grouped bar charts comparing winner vs temporal-memory across seasons."""
    lang = _resolve_lang(lang)
    labels = lang({"suptitle": "Leave-One-Season-Out - Vencedor Atual vs. Memória Temporal"})
    season_order = list(LeaveOneSeasonOut.SEASONS.keys())
    model_order = seasonal_temporal_memory_summary_df["Model"].tolist()
    plot_df = seasonal_temporal_memory_comparison_df.copy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.suptitle(
        labels["suptitle"],
        fontsize=14,
        fontweight="bold",
    )

    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        pivot = plot_df.pivot(index="Season", columns="Model", values=metric).reindex(
            index=season_order, columns=model_order
        )
        pivot.plot(kind="bar", ax=ax, edgecolor="white", rot=30)
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# render_loso_temporal_memory_insight (cell [103])
# ---------------------------------------------------------------------------


def render_loso_temporal_memory_insight(
    results_df: pd.DataFrame,
    baseline_label: str = "Current_XGB_same_safe_rows",
    challenger_label: str = "XGB_with_temporal_memory",
    lang=None,
) -> str:
    """Return a Markdown insight string for the LOSO temporal-memory comparison."""
    lang = _resolve_lang(lang)

    labels = lang(
        {
            "context_para": (
                "Este segundo teste leave-one-season-out é posicionado após a validação de memória "
                "temporal, pois a candidata já foi formalmente introduzida. Para manter a "
                "comparação controlada, ambas as variantes são treinadas sobre as mesmas linhas "
                "seguras para memória em cada fold, enquanto a candidata de memória temporal "
                "recebe features de demanda defasada e móvel. A estação retida continua excluída "
                "do ajuste, mas sua demanda passada já observada é permitida durante a previsão; "
                "portanto, a versão temporal deve ser interpretada como um modelo operacional "
                "rolling one-step-ahead, e não como uma previsão de estação completa em cold-start."
            ),
            "business_para": (
                "A leitura de negócio deve permanecer sazonal, não apenas baseada na média. "
                "Se o inverno melhorar materialmente, a falha anterior foi causada "
                "majoritariamente pela ausência de contexto temporal de curto prazo, e não "
                "apenas pela classe do "
                "estimador. Se o inverno permanecer fraco, então a memória temporal ajuda a "
                "dinâmica local da demanda, mas não resolve totalmente a transferência de regime "
                "sazonal, e o inverno ainda deve ser tratado com calibração sazonal, interações "
                "clima-estação mais fortes, ponderação de amostras ou um modelo dedicado ao "
                "inverno. Em qualquer caso, a função LOSO reutilizável passa a funcionar como uma "
                "checagem de governança aplicável a futuros pipelines candidatos antes de serem "
                "promovidos."
            ),
            "season_change": (
                "Para {season}, o RMSE mudou de {base_rmse:.2f} para {mem_rmse:.2f} e o R2 mudou "
                "de {base_r2:.3f} para {mem_r2:.3f}."
            ),
            "decision_improved": (
                "A candidata de memória temporal melhorou o perfil médio de generalização sazonal. "
                "O mesmo estimador vencedor se tornou mais forte ao receber contexto histórico de "
                "demanda livre de leakage."
            ),
            "decision_partial": (
                "A candidata de memória temporal produziu um ganho sazonal parcial. A melhoria "
                "deve ser lida por estação, pois a média esconde onde o sinal de memória ajuda ou "
                "prejudica."
            ),
            "decision_none": (
                "A candidata de memória temporal não melhorou o perfil de generalização sazonal. "
                "O vencedor atual permanece preferível sob este protocolo leave-one-season-out."
            ),
            "no_season": "nenhuma estação",
            "summary_para": (
                "{decision} Em média, o MAE mudou de **{base_mae:.2f}** para **{mem_mae:.2f}**, o "
                "RMSE mudou de **{base_rmse:.2f}** para **{mem_rmse:.2f}**, e o R2 mudou de "
                "**{base_r2:.3f}** para **{mem_r2:.3f}**. A melhora de RMSE apareceu em "
                "**{improved}**, enquanto **{degraded}** não melhorou por RMSE."
            ),
        }
    )

    comparison = results_df.set_index(["Model", "Season"])
    summary = results_df.groupby("Model", observed=False)[["MAE", "RMSE", "R2"]].mean()

    baseline_summary = summary.loc[baseline_label]
    challenger_summary = summary.loc[challenger_label]
    delta_mae = challenger_summary["MAE"] - baseline_summary["MAE"]
    delta_rmse = challenger_summary["RMSE"] - baseline_summary["RMSE"]
    delta_r2 = challenger_summary["R2"] - baseline_summary["R2"]

    season_order = list(LeaveOneSeasonOut.SEASONS.keys())
    seasonal_lines = []
    improved_rmse_seasons = []
    degraded_rmse_seasons = []

    for season in season_order:
        base = comparison.loc[(baseline_label, season)]
        challenger = comparison.loc[(challenger_label, season)]
        rmse_delta = challenger["RMSE"] - base["RMSE"]
        if rmse_delta < 0:
            improved_rmse_seasons.append(season)
        else:
            degraded_rmse_seasons.append(season)
        seasonal_lines.append(
            labels["season_change"].format(
                season=season,
                base_rmse=base["RMSE"],
                mem_rmse=challenger["RMSE"],
                base_r2=base["R2"],
                mem_r2=challenger["R2"],
            )
        )

    if delta_rmse < 0 and delta_mae < 0 and delta_r2 > 0:
        decision = labels["decision_improved"]
    elif delta_rmse < 0 or delta_mae < 0:
        decision = labels["decision_partial"]
    else:
        decision = labels["decision_none"]

    improved_text = (
        ", ".join(improved_rmse_seasons) if improved_rmse_seasons else labels["no_season"]
    )
    degraded_text = (
        ", ".join(degraded_rmse_seasons) if degraded_rmse_seasons else labels["no_season"]
    )

    summary_para = labels["summary_para"].format(
        decision=decision,
        base_mae=baseline_summary["MAE"],
        mem_mae=challenger_summary["MAE"],
        base_rmse=baseline_summary["RMSE"],
        mem_rmse=challenger_summary["RMSE"],
        base_r2=baseline_summary["R2"],
        mem_r2=challenger_summary["R2"],
        improved=improved_text,
        degraded=degraded_text,
    )

    return (
        f"{labels['context_para']}\n\n"
        f"{summary_para}\n\n"
        f"{' '.join(seasonal_lines)}\n\n"
        f"{labels['business_para']}"
    )


# ---------------------------------------------------------------------------
# render_temporal_memory_insight
#
# Dynamic replacement for what used to be a hand-written markdown cell in the
# notebook ("Insight Temporal Memory Validation"). The static prose went stale
# the first time the winner was retrained (its quoted MAE/RMSE/R2/Durbin-Watson
# values no longer matched the recomputed tables two cells above it), so the
# numbers are now pulled from the same metrics dicts the tables are built from.
# ---------------------------------------------------------------------------


def render_temporal_memory_insight(
    baseline_metrics: Dict[str, Any],
    memory_metrics: Dict[str, Any],
    lang=None,
) -> str:
    """Return a Markdown insight for the holdout temporal-memory comparison.

    Parameters are the metrics dicts returned by ``evaluate_winner_variant``
    for the plain winner and the temporal-memory challenger (same aligned
    window and holdout), including the ``Durbin-Watson`` statistic of the
    holdout residuals.
    """
    lang = _resolve_lang(lang)

    labels = lang(
        {
            "header": "**`Insight Validação de Memória Temporal:`**",
            "business_para": (
                "A interpretação de negócio é direta: a demanda recente é um sinal operacional "
                "altamente informativo. Clima, calendário, sazonalidade e hora do dia explicam "
                "grande parte do nível de demanda, mas não capturam totalmente o momentum, "
                "choques de curto prazo e a persistência das horas anteriores ou de períodos "
                "históricos comparáveis. Ao adicionar `lag_1h`, defasagens diárias e semanais, "
                "médias móveis, volatilidade móvel e codificações de hora-da-semana, o modelo "
                "ganha acesso a esse contexto de curto prazo."
            ),
            "caveat_para": (
                "O resultado deve ser adotado com uma condição importante de modelagem. Esta "
                "validação representa um caso de uso **rolling one-step-ahead**, no qual aluguéis "
                "recentemente observados estão disponíveis quando a próxima previsão é feita. "
                "Isso é realista para monitoramento operacional, alocação de frota de curto prazo "
                "e atualizações horárias de demanda. Se o objetivo mudar para prever um horizonte "
                "futuro completo sem observar aluguéis reais intermediários, as mesmas features "
                "precisariam ser produzidas recursivamente ou substituídas por valores de "
                "defasagem previstos."
            ),
            "verdict_improved": (
                "O experimento de memória temporal fornece evidência forte de que a estrutura de "
                "erro remanescente não era ruído aleatório, mas comportamento sequencial de "
                "demanda não modelado."
            ),
            "adoption_improved": (
                "Sob a suposição operacional atual, a versão de memória temporal é claramente "
                "superior e deve se tornar a nova candidata vencedora para avaliação subsequente."
            ),
            "verdict_not_improved": (
                "O experimento de memória temporal não melhorou o perfil de holdout nesta "
                "execução, o que enfraquece a hipótese de estrutura sequencial para o vencedor "
                "atual."
            ),
            "adoption_not_improved": (
                "O vencedor simples deve ser mantido como candidato operacional até que a "
                "variante de memória temporal seja revisitada."
            ),
            "dw_sentence": (
                "Os resíduos da linha de base tinham Durbin-Watson igual a **{base_dw:.3f}** (a "
                "referência de ausência de autocorrelação é 2). Após a memória temporal ser "
                "adicionada, o Durbin-Watson moveu-se para **{mem_dw:.3f}**, {dw_direction}"
            ),
            "dw_closer": (
                "mais próximo da região desejada — a demanda defasada e os resumos móveis "
                "trataram diretamente a dependência serial apontada pela análise de resíduos."
            ),
            "dw_not_closer": (
                "o que não está mais próximo da região desejada — a dependência serial "
                "remanescente não foi resolvida apenas pelas features de memória."
            ),
            "summary_para": (
                "{verdict} Quando o vencedor atual foi reavaliado na mesma janela alinhada, "
                "alcançou um MAE de **{base_mae:.2f} bicicletas**, um RMSE de **{base_rmse:.2f} "
                "bicicletas** e um R2 de **{base_r2:.4f}**. Com features de demanda defasada e "
                "móvel livres de leakage, o mesmo estimador obteve MAE de **{mem_mae:.2f} "
                "bicicletas**, RMSE de **{mem_rmse:.2f} bicicletas** e R2 de **{mem_r2:.4f}** — "
                "uma mudança de **{delta_mae:+.2f} bicicletas no MAE** e **{delta_rmse:+.2f} "
                "bicicletas no RMSE** em relação à linha de base."
            ),
        }
    )

    base_mae, mem_mae = baseline_metrics["MAE"], memory_metrics["MAE"]
    base_rmse, mem_rmse = baseline_metrics["RMSE"], memory_metrics["RMSE"]
    base_r2, mem_r2 = baseline_metrics["R2"], memory_metrics["R2"]
    base_dw, mem_dw = baseline_metrics["Durbin-Watson"], memory_metrics["Durbin-Watson"]

    improved = mem_rmse < base_rmse and mem_r2 > base_r2
    if improved:
        verdict = labels["verdict_improved"]
        adoption = labels["adoption_improved"]
    else:
        verdict = labels["verdict_not_improved"]
        adoption = labels["adoption_not_improved"]

    # Durbin-Watson: 2.0 is the no-autocorrelation reference point.
    dw_moved_closer = abs(mem_dw - 2.0) < abs(base_dw - 2.0)
    dw_sentence = labels["dw_sentence"].format(
        base_dw=base_dw,
        mem_dw=mem_dw,
        dw_direction=labels["dw_closer"] if dw_moved_closer else labels["dw_not_closer"],
    )

    summary_para = labels["summary_para"].format(
        verdict=verdict,
        base_mae=base_mae,
        mem_mae=mem_mae,
        base_rmse=base_rmse,
        mem_rmse=mem_rmse,
        base_r2=base_r2,
        mem_r2=mem_r2,
        delta_mae=base_mae - mem_mae,
        delta_rmse=base_rmse - mem_rmse,
    )

    return (
        f"{labels['header']}\n\n"
        f"{summary_para}\n\n"
        f"{dw_sentence}\n\n"
        f"{labels['business_para']}\n\n"
        f"{labels['caveat_para']} {adoption}"
    )


# ---------------------------------------------------------------------------
# render_final_winner_decision
#
# Dynamic replacement for the hand-written "Final Winner Decision" markdown
# cell: the tabular winner's name/pipeline and the seasonal claims are derived
# from the current metric_dataframe and LOSO comparison instead of being
# hard-coded prose that silently goes stale after a re-run (the old cell kept
# claiming a Winter improvement that the recomputed LOSO table no longer showed).
# ---------------------------------------------------------------------------


def render_final_winner_decision(
    metric_dataframe: pd.DataFrame,
    baseline_metrics: Dict[str, Any],
    memory_metrics: Dict[str, Any],
    loso_results_df: pd.DataFrame,
    baseline_label: str = "Current_XGB_same_safe_rows",
    challenger_label: str = "XGB_with_temporal_memory",
    lang=None,
) -> str:
    """Return the Markdown final-winner narrative with live numbers.

    ``metric_dataframe`` supplies the best pre-memory tabular winner
    (highest Test R2 row); ``baseline_metrics``/``memory_metrics`` come from
    ``evaluate_winner_variant`` on the aligned holdout; ``loso_results_df``
    is the LOSO comparison frame with `Model`/`Season`/`R2` columns.
    """
    lang = _resolve_lang(lang)

    labels = lang(
        {
            "header": "### Decisão do Vencedor Final",
            "tabular_para_tail": (
                "Esse resultado é útil porque prova que o modelo permanece forte após a remoção "
                "de leakage e sem depender de faixas derivadas do alvo."
            ),
            "closing_para": (
                "Sob a suposição de deployment rolling one-step-ahead, a versão de memória "
                "temporal deve ser tratada como o vencedor operacional; sob uma previsão de "
                "longo horizonte em que defasagens futuras observadas não estão disponíveis, um "
                "design de previsão recursivo ou multi-step separado ainda seria necessário."
            ),
            "intro_para": (
                "O vencedor final depende do horizonte de previsão. Para uma comparação tabular "
                "pura de holdout temporal, antes de qualquer informação de demanda defasada ser "
                "introduzida, `{best_estimator}` com o pipeline `{best_pipeline}` é o melhor "
                "estimador (Test R2 **{best_test_r2:.4f}**). {tail}"
            ),
            "operational_improved": (
                "Para o caso de uso operacional para o qual a análise de resíduos naturalmente "
                "aponta, a candidata final deve ser o mesmo vencedor com features de memória "
                "temporal livres de leakage. Essa versão preserva a espinha dorsal de modelagem, "
                "mas adiciona contexto histórico de demanda por meio de defasagens e resumos "
                "móveis calculados apenas a partir de observações passadas. {holdout_sentence} "
                "{winter_sentence}"
            ),
            "operational_not_improved": (
                "A candidata de memória temporal não dominou claramente nesta execução, então a "
                "decisão deve permanecer com o vencedor tabular simples até que a variante de "
                "memória seja revalidada. {holdout_sentence} {winter_sentence}"
            ),
            "holdout_improved": (
                "Ela melhora o holdout temporal alinhado (R2 {base_r2:.4f} → {mem_r2:.4f}, RMSE "
                "{base_rmse:.2f} → {mem_rmse:.2f} bicicletas)."
            ),
            "holdout_not_improved": (
                "Ela não melhorou o holdout temporal alinhado nesta execução (R2 {base_r2:.4f} → "
                "{mem_r2:.4f})."
            ),
            "winter_improved": (
                "Ela também fortalece a generalização leave-one-season-out, incluindo o inverno, "
                "em que o vencedor tabular simples era frágil (R2 do inverno {winter_base:.3f} → "
                "{winter_mem:.3f})."
            ),
            "winter_not_improved": (
                "Seus ganhos leave-one-season-out se concentram fora do inverno (R2 do inverno "
                "permanece em {winter_mem:.3f}): sob o protocolo LOSO mascarado, o inverno retido "
                "não tem demanda previamente observada para alimentar as features de defasagem, "
                "então a memória não pode ajudar um inverno em cold-start — com um único inverno "
                "observado nos dados, essa estação ainda requer calibração sazonal, tratamento "
                "dedicado ou mais invernos de dados reais de treino."
            ),
        }
    )

    ranked = metric_dataframe.sort_values(
        by="Test R2",
        ascending=False,
        key=lambda col: pd.to_numeric(col, errors="coerce"),
    )
    best_idx = ranked.index[0]
    best_estimator, _, best_pipeline = best_idx[0], best_idx[1], best_idx[2]
    best_test_r2 = float(ranked.iloc[0]["Test R2"])

    holdout_improved = (
        memory_metrics["R2"] > baseline_metrics["R2"]
        and memory_metrics["RMSE"] < baseline_metrics["RMSE"]
    )

    comparison = loso_results_df.set_index(["Model", "Season"])
    loso_avg = loso_results_df.groupby("Model", observed=False)["R2"].mean()
    loso_improved = loso_avg[challenger_label] > loso_avg[baseline_label]
    winter_base = float(comparison.loc[(baseline_label, "Winter"), "R2"])
    winter_mem = float(comparison.loc[(challenger_label, "Winter"), "R2"])
    winter_improved = winter_mem > winter_base + 1e-9

    if winter_improved:
        winter_sentence = labels["winter_improved"].format(
            winter_base=winter_base, winter_mem=winter_mem
        )
    else:
        winter_sentence = labels["winter_not_improved"].format(winter_mem=winter_mem)

    if holdout_improved:
        holdout_sentence = labels["holdout_improved"].format(
            base_r2=baseline_metrics["R2"],
            mem_r2=memory_metrics["R2"],
            base_rmse=baseline_metrics["RMSE"],
            mem_rmse=memory_metrics["RMSE"],
        )
    else:
        holdout_sentence = labels["holdout_not_improved"].format(
            base_r2=baseline_metrics["R2"], mem_r2=memory_metrics["R2"]
        )

    if holdout_improved and loso_improved:
        operational_para = labels["operational_improved"].format(
            holdout_sentence=holdout_sentence, winter_sentence=winter_sentence
        )
    else:
        operational_para = labels["operational_not_improved"].format(
            holdout_sentence=holdout_sentence, winter_sentence=winter_sentence
        )

    intro_para = labels["intro_para"].format(
        best_estimator=best_estimator,
        best_pipeline=best_pipeline,
        best_test_r2=best_test_r2,
        tail=labels["tabular_para_tail"],
    )

    return (
        f"{labels['header']}\n\n"
        f"{intro_para}\n\n"
        f"{operational_para}\n\n"
        f"{labels['closing_para']}"
    )


# ---------------------------------------------------------------------------
# plot_final_holdout (cell [104])
# ---------------------------------------------------------------------------


def plot_final_holdout(
    y_holdout_raw: pd.Series,
    preds_winner: np.ndarray,
    preds_lgbm: np.ndarray,
    pred_temporal_memory: Optional[np.ndarray] = None,
    last_hours: int = 168,
    lang=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Line plot of last ``last_hours`` hours on the holdout for all models."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "suptitle": "Previsões por modelos de Regressão não lineares",
            "actual_demand": "Demanda real",
            "temporal_memory_label": "XGBRegressor + Memória Temporal (rolling one-step-ahead)",
            "ylabel": "Rented Bike Count",
            "xlabel": "Últimas {n} horas do holdout",
        }
    )
    sl = slice(-last_hours, None)

    actual = np.asarray(y_holdout_raw, dtype=float)[sl]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle(labels["suptitle"])

    ax.plot(actual, "x-", alpha=1.0, label=labels["actual_demand"], color="black")
    ax.plot(preds_winner[sl], "x-", alpha=0.75, label="XGBRegressor")
    ax.plot(preds_lgbm[sl], "x-", alpha=0.75, label="LGBMRegressor")

    if pred_temporal_memory is not None:
        ax.plot(
            pred_temporal_memory[sl],
            "o-",
            linewidth=0.5,
            markersize=4,
            alpha=0.9,
            color="tab:green",
            label=labels["temporal_memory_label"],
        )

    ax.set_ylabel(labels["ylabel"])
    ax.set_xlabel(labels["xlabel"].format(n=last_hours))
    ax.legend()
    plt.show()
    return fig, ax
