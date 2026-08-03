"""EDA of the engineered features (notebook 03).

Pure report builders return display-ready DataFrames; plot builders return
``(fig, axes)`` and never call ``plt.show()``. Visible text is localized to
PT via ``LangMap`` (passthrough for PT); internal column/category names are the
analytic contract and are left unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.i18n import resolve_lang as _resolve_lang
from src.seasonal import anomaly_diagnosis, anomaly_mask, demand_index_moving_average

# Ordered category vocabulary for the engineered features (shared by the nb03
# distribution, boxplot, effect and encoding cells).
CAT_ORDERS: Dict[str, List[str]] = {
    "Time_Period": ["Dawn", "Morning", "Afternoon", "Evening"],
    "WeekStatus": ["Weekday", "Weekend"],
    "Rush_Hour": ["No Rush", "Rush"],
    "Rush_Period": ["Non-Rush", "Morning Rush", "Evening Rush"],
    "Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "Rainfall Cat": ["No Rain", "Light Rain", "Moderate Rain", "Heavy Rain"],
    "Snowfall Cat": ["No Snow", "Light Snow", "Moderate Snow", "Heavy Snow"],
    "Sunshine Cat": ["No Sun", "Low Sun", "Moderate Sun", "Full Sun"],
    "Cloud Cover Cat": ["Clear", "Partly Cloudy", "Overcast"],
}

CREATED_FEATURES: List[str] = [
    "Date",
    "Time_Period",
    "Weekday",
    "DayNumberOnWeek",
    "WeekStatus",
    "Rush_Hour",
    "Rush_Period",
    "Rainfall Cat",
    "Snowfall Cat",
    "Sunshine Cat",
    "Cloud Cover Cat",
    "Month",
    "Year",
]

SEASON_ORDER: List[str] = ["Winter", "Spring", "Summer", "Autumn"]


# ---------------------------------------------------------------------------
# 1. Inventory, missingness and the legacy rush-hour audit
# ---------------------------------------------------------------------------


def created_features_inventory(
    df: pd.DataFrame, created: Sequence[str] = CREATED_FEATURES, lang=None
) -> pd.DataFrame:
    """Presence / dtype / cardinality / missing count for the engineered features."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "variable": "Variável",
            "present": "Presente",
            "dtype": "Tipo",
            "distinct": "Valores distintos",
            "missing": "Ausentes",
            "absent": "ausente",
        }
    )
    return pd.DataFrame(
        {
            labels["variable"]: list(created),
            labels["present"]: [col in df.columns for col in created],
            labels["dtype"]: [
                str(df[col].dtype) if col in df.columns else labels["absent"] for col in created
            ],
            labels["distinct"]: [
                df[col].nunique(dropna=True) if col in df.columns else np.nan for col in created
            ],
            labels["missing"]: [
                df[col].isna().sum() if col in df.columns else np.nan for col in created
            ],
        }
    )


def missing_features_summary(df: pd.DataFrame, lang=None) -> pd.DataFrame:
    """Columns with missing values, count and percentage (descending)."""
    lang = _resolve_lang(lang)
    labels = lang({"missing": "Ausentes", "pct": "Percentual"})
    return (
        df.isna()
        .sum()
        .loc[lambda s: s.gt(0)]
        .rename(labels["missing"])
        .to_frame()
        .assign(**{labels["pct"]: lambda x: 100 * x[labels["missing"]] / len(df)})
        .sort_values(labels["missing"], ascending=False)
    )


def rush_hour_audit(df: pd.DataFrame) -> Dict[str, int]:
    """Weekend rows the legacy vs revised rush rule would flag as ``Rush``.

    The legacy rule mistook ``Functioning Day`` for a weekday indicator, so
    weekends could be flagged. The revised rule requires a weekday.
    """
    legacy_rush = (
        df["Functioning Day"].eq("Yes")
        & df["Holiday"].eq("No Holiday")
        & (df["Hour"].between(7, 9) | df["Hour"].between(16, 19))
    )
    legacy_weekend = legacy_rush & df["WeekStatus"].eq("Weekend")
    revised_weekend = df["Rush_Hour"].eq("Rush") & df["WeekStatus"].eq("Weekend")
    return {
        "legacy_weekend": int(legacy_weekend.sum()),
        "revised_weekend": int(revised_weekend.sum()),
    }


def plot_category_distribution(
    df: pd.DataFrame, cat_orders: Dict[str, List[str]] = CAT_ORDERS, lang=None
) -> Tuple[plt.Figure, np.ndarray]:
    """3x3 grid of relative-frequency bars for the created categories."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "ylabel": "Participação (%)",
            "suptitle": "Distribuição relativa das categorias criadas",
            "missing": "Ausente",
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(20, 13))
    for ax, (col, order) in zip(axes.ravel(), cat_orders.items()):
        values = df[col].astype("string").fillna(labels["missing"])
        plot_order = [level for level in order if level in set(values)]
        if values.eq(labels["missing"]).any():
            plot_order.append(labels["missing"])
        shares = values.value_counts(normalize=True).reindex(plot_order).mul(100)
        shares.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(labels["ylabel"])
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(labels["suptitle"], fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 2. Detrended descriptive index
# ---------------------------------------------------------------------------


def detrended_index_frame(
    df: pd.DataFrame,
    target: str = "Rented Bike Count",
    index_col: str = "Demand_Index",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Attach the detrended index and drop rows where the baseline is undefined.

    Returns ``(mv, demand_index, anomaly_mask)`` where ``mv`` keeps every column
    plus ``index_col`` for the retained rows. The 2020 anomaly window is excluded
    only from the baseline estimate (shared gate from ``src.seasonal``).
    """
    diagnosis = anomaly_diagnosis(df, target)
    mask = anomaly_mask(df, diagnosis)
    demand_index = demand_index_moving_average(df, anomaly_mask=mask)
    keep = demand_index.notna()
    mv = df.loc[keep].copy()
    mv[index_col] = demand_index.loc[keep]
    return mv, demand_index, mask


def detrend_diagnostics(
    df: pd.DataFrame,
    mv: pd.DataFrame,
    demand_index: pd.Series,
    mask: pd.Series,
    target: str = "Rented Bike Count",
    index_col: str = "Demand_Index",
    lang=None,
) -> pd.DataFrame:
    """One-column diagnostics table for the detrended index."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "anom_months": "Meses anômalos de 2020",
            "kept": "Linhas preservadas",
            "growth": "Crescimento médio 2015→2024",
            "mean": "Média do índice",
            "median": "Mediana do índice",
            "amplitude": "Amplitude das médias mensais",
            "diag": "Diagnóstico",
        }
    )
    yearly_mean = df.groupby("Year")[target].mean()
    calendar_growth = yearly_mean.iloc[-1] / yearly_mean.iloc[0]
    monthly = mv.groupby("Month")[index_col].mean()
    monthly_amplitude = monthly.max() - monthly.min()
    anomalous_months = sorted(df.loc[mask, "Month"].unique().tolist())
    keep = demand_index.notna()
    series = pd.Series(
        {
            labels["anom_months"]: ", ".join(map(str, anomalous_months)),
            labels["kept"]: f"{len(mv):,} ({100 * keep.mean():.1f}%)",
            labels["growth"]: f"{calendar_growth:.1f}×",
            labels["mean"]: f"{mv[index_col].mean():.3f}",
            labels["median"]: f"{mv[index_col].median():.3f}",
            labels["amplitude"]: f"{monthly_amplitude:.3f}",
        },
        name=labels["diag"],
    )
    return series.to_frame()


def plot_detrend_baseline(
    df: pd.DataFrame,
    mv: pd.DataFrame,
    mask: pd.Series,
    target: str = "Rented Bike Count",
    index_col: str = "Demand_Index",
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Daily demand + 365-day baseline (top) and monthly median index (bottom)."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "daily": "Demanda média diária",
            "baseline": "Linha de base centrada (365 dias)",
            "rented": "Bicicletas alugadas",
            "top_title": "Demanda diária e linha de base usada no des-tendenciamento",
            "monthly": "Mediana mensal do índice",
            "date": "Data",
            "bottom_title": "Demanda relativa após a remoção do nível móvel",
        }
    )
    daily = (
        df.assign(Day=df["Date"].dt.normalize(), Anomaly=mask)
        .groupby("Day")
        .agg(Demand=(target, "mean"), Anomaly=("Anomaly", "any"))
    )
    daily["Baseline_365d"] = (
        daily["Demand"].where(~daily["Anomaly"]).rolling(365, center=True, min_periods=180).mean()
    )
    monthly_index = mv.set_index("Date")[index_col].resample("MS").median()

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    axes[0].plot(
        daily.index, daily["Demand"], color="lightsteelblue", lw=0.7, label=labels["daily"]
    )
    axes[0].plot(
        daily.index, daily["Baseline_365d"], color="navy", lw=2.0, label=labels["baseline"]
    )
    axes[0].set_ylabel(labels["rented"])
    axes[0].set_title(labels["top_title"])
    axes[0].legend()

    axes[1].plot(monthly_index.index, monthly_index, color="darkorange", lw=1.5)
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_ylabel(labels["monthly"])
    axes[1].set_xlabel(labels["date"])
    axes[1].set_title(labels["bottom_title"])
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 3. Relative demand by created variable
# ---------------------------------------------------------------------------


def plot_category_boxplots(
    mv: pd.DataFrame,
    cat_orders: Dict[str, List[str]] = CAT_ORDERS,
    index_col: str = "Demand_Index",
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """3x3 grid of index boxplots per category level (fliers hidden)."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "ylabel": "Índice de demanda",
            "suptitle": "Índice des-tendenciado por variável criada",
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharey=True)
    for ax, (col, order) in zip(axes.ravel(), cat_orders.items()):
        observed_levels = set(mv[col].astype("string").dropna())
        plot_order = [level for level in order if level in observed_levels]
        sns.boxplot(
            data=mv,
            x=col,
            y=index_col,
            order=plot_order,
            ax=ax,
            showfliers=False,
            color="cornflowerblue",
        )
        ax.axhline(1.0, ls="--", color="gray", lw=1)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(labels["ylabel"])
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(labels["suptitle"], fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def category_effect_summary(
    mv: pd.DataFrame,
    cat_orders: Dict[str, List[str]] = CAT_ORDERS,
    index_col: str = "Demand_Index",
    lang=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-level index stats and a per-feature median-amplitude summary.

    Returns ``(effect_summary, category_effects)`` — the compact summary sorted
    by amplitude and the full per-(feature, level) statistics.
    """
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "mean": "Média",
            "median": "Mediana",
            "feature": "Feature",
            "lowest_level": "Menor mediana",
            "lowest_val": "Valor mínimo",
            "highest_level": "Maior mediana",
            "highest_val": "Valor máximo",
            "amplitude": "Amplitude",
            "min_n": "Menor n por nível",
            "level": "Nível",
        }
    )
    tables = []
    rows = []
    for col, order in cat_orders.items():
        levels = [level for level in order if level in set(mv[col].astype("string").dropna())]
        stats = (
            mv.groupby(col, observed=True)[index_col]
            .agg(
                n="size",
                **{
                    labels["mean"]: "mean",
                    labels["median"]: "median",
                    "Q1": lambda s: s.quantile(0.25),
                    "Q3": lambda s: s.quantile(0.75),
                },
            )
            .reindex(levels)
        )
        stats["IQR"] = stats["Q3"] - stats["Q1"]
        stats.index = pd.MultiIndex.from_product(
            [[col], stats.index], names=[labels["feature"], labels["level"]]
        )
        tables.append(stats)

        medians = stats[labels["median"]]
        rows.append(
            {
                labels["feature"]: col,
                labels["lowest_level"]: medians.idxmin()[1],
                labels["lowest_val"]: medians.min(),
                labels["highest_level"]: medians.idxmax()[1],
                labels["highest_val"]: medians.max(),
                labels["amplitude"]: medians.max() - medians.min(),
                labels["min_n"]: int(stats["n"].min()),
            }
        )

    category_effects = pd.concat(tables)
    effect_summary = pd.DataFrame(rows).sort_values(labels["amplitude"], ascending=False)
    return effect_summary, category_effects


# ---------------------------------------------------------------------------
# 3.1 MeanEncoder inspection
# ---------------------------------------------------------------------------


def mean_encoding_by_season(
    mv: pd.DataFrame,
    cat_orders: Dict[str, List[str]] = CAT_ORDERS,
    index_col: str = "Demand_Index",
    season_order: Sequence[str] = SEASON_ORDER,
    lang=None,
) -> pd.DataFrame:
    """In-sample per-season target means per level (a MeanEncoder 'peek')."""
    lang = _resolve_lang(lang)
    labels = lang({"feature": "Feature", "level": "Nível", "across": "Amplitude entre estações"})
    season_order = list(season_order)
    tables = []
    for col, order in cat_orders.items():
        levels = [level for level in order if level in set(mv[col].astype("string").dropna())]
        per_season = (
            mv.groupby([col, "Seasons"], observed=True)[index_col]
            .mean()
            .unstack("Seasons")
            .reindex(index=levels, columns=season_order)
        )
        per_season["Overall"] = mv.groupby(col, observed=True)[index_col].mean().reindex(levels)
        per_season["n"] = mv.groupby(col, observed=True)[index_col].size().reindex(levels)
        per_season[labels["across"]] = per_season[season_order].max(axis=1) - per_season[
            season_order
        ].min(axis=1)
        per_season.index = pd.MultiIndex.from_product(
            [[col], per_season.index], names=[labels["feature"], labels["level"]]
        )
        tables.append(per_season)
    return pd.concat(tables)


def within_season_spread(
    mv: pd.DataFrame,
    cat_orders: Dict[str, List[str]] = CAT_ORDERS,
    index_col: str = "Demand_Index",
    season_order: Sequence[str] = SEASON_ORDER,
    lang=None,
) -> pd.DataFrame:
    """Per (feature x season) spread between levels (max - min of level means)."""
    lang = _resolve_lang(lang)
    feature_label = lang({"feature": "Feature"})["feature"]
    season_order = list(season_order)
    spread = pd.DataFrame(
        {
            col: (
                mv.groupby([col, "Seasons"], observed=True)[index_col]
                .mean()
                .unstack("Seasons")
                .reindex(columns=season_order)
                .agg(lambda s: s.max() - s.min())
            )
            for col in cat_orders
        }
    ).T
    spread.index.name = feature_label
    return spread


def meanencoder_verification(
    mv: pd.DataFrame,
    encoding_by_season: pd.DataFrame,
    index_col: str = "Demand_Index",
    demo_col: str = "Rush_Period",
    lang=None,
) -> pd.DataFrame:
    """Confirm the 'Overall' column reproduces a real MeanEncoder in-sample."""
    from feature_engine.encoding import MeanEncoder

    lang = _resolve_lang(lang)
    labels = lang(
        {
            "level": "Nível",
            "encoded": "MeanEncoder demonstrativo",
            "overall": "Geral da tabela",
            "abs_diff": "Diferença absoluta",
        }
    )
    x_demo = mv[[demo_col]].astype(object)
    encoded_demo = MeanEncoder(variables=[demo_col]).fit_transform(x_demo, mv[index_col])[demo_col]
    mapping_demo = (
        pd.DataFrame({labels["level"]: mv[demo_col].astype(str), "enc": encoded_demo})
        .groupby(labels["level"])["enc"]
        .first()
    )
    diagnostic_overall = encoding_by_season.loc[demo_col, "Overall"]
    verification = pd.concat(
        [mapping_demo.rename(labels["encoded"]), diagnostic_overall.rename(labels["overall"])],
        axis=1,
    )
    verification[labels["abs_diff"]] = (
        verification[labels["encoded"]] - verification[labels["overall"]]
    ).abs()
    return verification


# ---------------------------------------------------------------------------
# 4. New weather columns
# ---------------------------------------------------------------------------


def new_weather_summary(
    mv: pd.DataFrame,
    new_weather: Sequence[str],
    index_col: str = "Demand_Index",
    lang=None,
) -> pd.DataFrame:
    """Validity, missingness, skew and Spearman-with-index for the new weather cols."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "variable": "Variável",
            "valid_n": "n válido",
            "missing_pct": "Ausentes (%)",
            "skew": "Assimetria",
            "spearman": "Spearman com o índice",
        }
    )
    rows = []
    for col in new_weather:
        pair = mv[[col, index_col]].dropna()
        rows.append(
            {
                labels["variable"]: col,
                labels["valid_n"]: len(pair),
                labels["missing_pct"]: 100 * mv[col].isna().mean(),
                labels["skew"]: pair[col].skew(),
                labels["spearman"]: pair[col].corr(pair[index_col], method="spearman"),
            }
        )
    return pd.DataFrame(rows).set_index(labels["variable"])


def _relationship_summary(data: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    """Bin a feature (raw levels if <=20 unique, else 12 quantiles) and summarise."""
    pair = data[[feature, target]].dropna().copy()
    if pair[feature].nunique() <= 20:
        pair["Faixa"] = pair[feature]
    else:
        pair["Faixa"] = pd.qcut(pair[feature], q=12, duplicates="drop")

    summary = (
        pair.groupby("Faixa", observed=True)[target]
        .agg(
            n="size",
            Media="mean",
            Mediana="median",
            Q1=lambda s: s.quantile(0.25),
            Q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    if isinstance(summary["Faixa"].dtype, pd.CategoricalDtype):
        summary["x"] = summary["Faixa"].map(lambda interval: interval.mid).astype(float)
    else:
        summary["x"] = pd.to_numeric(summary["Faixa"])
    return summary.sort_values("x")


def plot_weather_relationships(
    mv: pd.DataFrame,
    new_weather: Sequence[str],
    index_col: str = "Demand_Index",
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Histograms (top) and index-by-range curves (bottom) for the new weather cols."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "freq": "Frequência",
            "skew": "assimetria",
            "index_by": "Índice por",
            "ylabel": "Índice de demanda",
            "mean": "Média",
            "median": "Mediana",
            "band": "Q1–Q3",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    for ax, col in zip(axes[0], new_weather):
        sns.histplot(mv[col], bins=30, ax=ax, color="seagreen")
        ax.set_title(f"{col} | {labels['skew']} = {mv[col].skew():.2f}")
        ax.set_xlabel(col)
        ax.set_ylabel(labels["freq"])

    for ax, col in zip(axes[1], new_weather):
        relation = _relationship_summary(mv, col, index_col)
        x = relation["x"].to_numpy(dtype=float)
        ax.plot(x, relation["Media"], marker="o", color="darkorange", label=labels["mean"])
        ax.plot(x, relation["Mediana"], marker="s", color="navy", label=labels["median"])
        ax.fill_between(
            x,
            relation["Q1"],
            relation["Q3"],
            color="lightsteelblue",
            alpha=0.35,
            label=labels["band"],
        )
        ax.axhline(1.0, ls="--", color="gray", lw=1)
        ax.set_title(f"{labels['index_by']} {col}")
        ax.set_xlabel(col)
        ax.set_ylabel(labels["ylabel"])
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 5. Engineered-feature vs modelling-space audit
# ---------------------------------------------------------------------------


def feature_model_audit(
    df: pd.DataFrame,
    created: Sequence[str] = CREATED_FEATURES,
    lang=None,
) -> pd.DataFrame:
    """Map each created feature to its origin, model availability and role."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="IProgress not found.*")
        from src.optimizer import RegressionOptimizer

    lang = _resolve_lang(lang)
    labels = lang(
        {
            "feature": "Variável criada",
            "derived": "Derivada de",
            "in_model": "Disponível no modelo base",
            "role": "Papel",
            "yes": "Sim",
            "no": "Não",
        }
    )
    model_features = set(RegressionOptimizer._NUMERICAL_FEATURES) | set(
        RegressionOptimizer._CATEGORICAL_FEATURES
    )
    # Derivation expressions are internal column-name references — never translated.
    origins = {
        "Date": "DateTime + Hour",
        "Time_Period": "Hour",
        "Weekday": "Date",
        "DayNumberOnWeek": "Date",
        "WeekStatus": "Date",
        "Rush_Hour": "Hour + WeekStatus + Holiday + Functioning Day",
        "Rush_Period": "Hour + WeekStatus + Holiday + Functioning Day",
        "Rainfall Cat": "Rainfall(mm)",
        "Snowfall Cat": "Snowfall (cm)",
        "Sunshine Cat": "Sunshine (hr)",
        "Cloud Cover Cat": "Cloud Cover (oktas)",
        "Month": "Date",
        "Year": "Date",
    }
    roles = lang(
        {
            "Date": "Ordenação e validação temporal",
            "Year": "Controle/EDA; não usado como preditor",
            "Sunshine Cat": "EDA; candidata a comparação futura",
            "Cloud Cover Cat": "EDA; candidata a comparação futura",
            "default": "Candidata aos seletores do otimizador",
        }
    )
    return pd.DataFrame(
        {
            labels["feature"]: list(created),
            labels["derived"]: [origins[col] for col in created],
            labels["in_model"]: [
                labels["yes"] if col in model_features else labels["no"] for col in created
            ],
            labels["role"]: [roles.get(col, roles["default"]) for col in created],
        }
    )


# ---------------------------------------------------------------------------
# 6. Presentation layer: PT-BR display copies for `Feature`/season-named tables
# ---------------------------------------------------------------------------

_FEATURE_TABLE_DISPLAY_LABELS = {
    "Feature": "Variável",
    "Winter": "Inverno",
    "Spring": "Primavera",
    "Summer": "Verão",
    "Autumn": "Outono",
    "Overall": "Geral",
}


def localize_feature_table(table: pd.DataFrame, lang=None) -> pd.DataFrame:
    """Cópia com rótulos em PT-BR para exibição.

    Aplica-se às tabelas de ``category_effect_summary`` (coluna/nível
    ``Feature``), ``mean_encoding_by_season`` e ``within_season_spread``
    (colunas de estação + ``Overall``): o schema interno dessas funções
    (usado por ``meanencoder_verification`` e pelos testes) nunca é
    alterado — apenas esta cópia, destinada a ``display()``.
    """
    lang = _resolve_lang(lang)
    labels = lang(_FEATURE_TABLE_DISPLAY_LABELS)
    display_df = table.rename(columns=labels)
    if isinstance(table.index, pd.MultiIndex):
        display_df.index = display_df.index.set_names(
            [labels.get(name, name) for name in table.index.names]
        )
    elif table.index.name in labels:
        display_df.index = display_df.index.rename(labels[table.index.name])
    return display_df
