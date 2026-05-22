from __future__ import annotations

from json import dumps, loads

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] == threshold
    return ['background-color: lime' if is_max.any() else '' for v in is_max]


def highlight_row(row):
    if row.Threshold >= 0.42 and row.Threshold <= 0.421:
        return ['background-color: royalblue'] * len(row)

    elif row.Threshold >= 0.22 and row.Threshold <= 0.221:
        return ['background-color: lime'] * len(row)

    else:
        return [''] * len(row)


def find_intersection_point(df_metric):
    """
    Encontra o ponto onde as métricas se cruzam (melhor ajuste)
    Usa minimização da variância entre as métricas

    Returns:
        tuple: (threshold_optimal, metrics_at_optimal)
    """
    metrics_cols = ['Roc_auc', 'Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro']

    metrics_df = df_metric[metrics_cols].copy()

    df_metric['std_metrics'] = metrics_df.std(axis=1)

    optimal_idx = df_metric['std_metrics'].idxmin()
    optimal_threshold = df_metric.loc[optimal_idx, 'Threshold']

    optimal_metrics = {
        'Roc_auc': df_metric.loc[optimal_idx, 'Roc_auc'],
        'Accuracy': df_metric.loc[optimal_idx, 'Accuracy'],
        'Precision': df_metric.loc[optimal_idx, 'Precision_macro'],
        'Recall': df_metric.loc[optimal_idx, 'Recall_macro'],
        'F1': df_metric.loc[optimal_idx, 'F1_macro'],
        'std': df_metric.loc[optimal_idx, 'std_metrics']
    }

    return optimal_threshold, optimal_metrics


def plot_metrics(df_metric, show_intersection=True):
    """
    Plota métricas com destaque no ponto de intersecção

    Args:
        df_metric: DataFrame com colunas Threshold, Roc_auc, Accuracy,
                Precision_macro, Recall_macro, F1_macro
        show_intersection: Se True, mostra ponto e comentário de intersecção
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(df_metric.Threshold, df_metric.Roc_auc, color='brown', alpha=0.7,
            label='Roc AUC', linewidth=1)
    ax.plot(df_metric.Threshold, df_metric.Accuracy, color='lime', alpha=0.7,
            label='Accuracy', linewidth=1)
    ax.plot(df_metric.Threshold, df_metric.Precision_macro, color='blue', alpha=0.7,
            label='Precision', linewidth=1)
    ax.plot(df_metric.Threshold, df_metric.Recall_macro, color='green', alpha=0.7,
            label='Recall', linewidth=1)
    ax.plot(df_metric.Threshold, df_metric.F1_macro, color='orange', alpha=0.7,
            label='F1', linewidth=1)

    if show_intersection:
        optimal_threshold, optimal_metrics = find_intersection_point(df_metric)

        ax.axvline(x=optimal_threshold, color='red', linestyle='--',
                   linewidth=1, alpha=0.6, label='Optimal Threshold')

        avg_metric = np.mean([
            optimal_metrics['Roc_auc'],
            optimal_metrics['Accuracy'],
            optimal_metrics['Precision'],
            optimal_metrics['Recall'],
            optimal_metrics['F1']
        ])
        ax.scatter(optimal_threshold, avg_metric, color='red', s=30,
                   zorder=5, edgecolors='red', linewidth=1, label='Optimal Point')

        info_text = (
            f"Optimal Balance Point\n"
            f"Threshold: {optimal_threshold:.3f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Roc AUC:     {optimal_metrics['Roc_auc']:.3f}\n"
            f"Accuracy:    {optimal_metrics['Accuracy']:.3f}\n"
            f"Precision:   {optimal_metrics['Precision']:.3f}\n"
            f"Recall:      {optimal_metrics['Recall']:.3f}\n"
            f"F1:          {optimal_metrics['F1']:.3f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Std Dev:     {optimal_metrics['std']:.4f}"
        )

        if optimal_threshold < 0.5:
            bbox_x = optimal_threshold + 0.12
            ha_align = 'left'
        else:
            bbox_x = optimal_threshold - 0.12
            ha_align = 'right'

        ax.annotate(info_text,
                    xy=(optimal_threshold, avg_metric),
                    xytext=(bbox_x, 0.75),
                    fontsize=9,
                    family='monospace',
                    bbox=dict(boxstyle='round,pad=0.8',
                              facecolor='lightyellow',
                              alpha=0.7,
                              edgecolor='red',
                              linewidth=0.5),
                    arrowprops=dict(arrowstyle='->',
                                    connectionstyle='arc3,rad=0.3',
                                    color='red',
                                    lw=1),
                    ha=ha_align)

    plt.xlabel('Threshold', fontsize=12, weight='bold')
    plt.ylabel('Metric value', fontsize=12, weight='bold')

    ax.text(x=0.0, y=.93,
            s="Metrics Curve for customized Decision Function",
            transform=fig.transFigure,
            ha='left',
            fontsize=16,
            weight='bold',
            alpha=.8)

    ax.text(x=0.0, y=.90,
            s="The metrics trade-off",
            transform=fig.transFigure,
            ha='left',
            fontsize=12,
            alpha=.8)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.1, 0.95])

    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()


def save_threshold_metrics(
    df_metric: pd.DataFrame,
    thresholds: list[float],
    metric_df: pd.DataFrame,
    baseline_estimator_name: str = "LogisticRegression",
    baseline_encoder_name: str = "OrdinalEncoder",
    baseline_data_model: str = "Baseline",
    threshold_data_model: str = "Baseline + Threshold tuning",
    w_recall: float = 7.0,
    w_precision: float = 3.0,
    w_time: float = 0.0,
    time_reference: float = 1.0,
    metric_path: str = "./dataset/metric_dataframe.csv"
) -> pd.DataFrame:
    """
    Save threshold tuning results into metric_dataframe, inheriting
    training and timing information from the baseline row.
    """
    if isinstance(metric_df.index, pd.MultiIndex):
        metric_df = metric_df.reset_index()

    baseline_mask = (
        (metric_df["Estimator"] == baseline_estimator_name)
        & (metric_df["Optimization/Data model"] == baseline_data_model)
        & (metric_df["Pre-Process Pipeline"] == baseline_encoder_name)
    )

    if not baseline_mask.any():
        raise KeyError(
            "Baseline row not found in metric_df: "
            f"{(baseline_estimator_name, baseline_data_model, baseline_encoder_name)}"
        )

    baseline_row = metric_df.loc[baseline_mask].iloc[0]

    try:
        baseline_params = loads(baseline_row.get("Parameters", "{}") or "{}")
    except Exception:
        baseline_params = {}

    if not isinstance(baseline_params, dict):
        baseline_params = {}

    business_weights = dumps({
        "w_recall": w_recall,
        "w_precision": w_precision,
        "w_time": w_time,
        "time_reference": time_reference
    })

    fit_time = baseline_row.get("Fit Time", np.nan)
    score_time = baseline_row.get("Score Time", np.nan)
    total_time = baseline_row.get("Total Time", np.nan)

    for threshold in thresholds:
        row = df_metric.loc[np.isclose(df_metric["Threshold"], threshold)]

        if row.empty:
            print(f"[WARN] Threshold {threshold:.4f} não encontrado em df_metric.")
            continue

        row = row.iloc[0]
        pipeline_name = f"{baseline_encoder_name} | thr={threshold:.2f}"
        row_identity = {
            "Estimator": baseline_estimator_name,
            "Optimization/Data model": threshold_data_model,
            "Pre-Process Pipeline": pipeline_name,
        }

        if pd.notna(total_time) and time_reference > 0:
            time_penalty = total_time / time_reference
        else:
            time_penalty = 0.0

        weight_sum = w_recall + w_precision + w_time
        if weight_sum <= 0:
            raise ValueError("A soma dos pesos deve ser maior que zero.")

        wr = w_recall / weight_sum
        wp = w_precision / weight_sum
        wt = w_time / weight_sum

        business_score = (
            (wr * row["Recall_macro"]) +
            (wp * row["Precision_macro"]) -
            (wt * time_penalty)
        )

        threshold_params = {
            **baseline_params,
            "threshold": float(threshold),
        }

        values_to_save = {
            **row_identity,
            "Train Roc auc": baseline_row.get("Train Roc auc", np.nan),
            "Test Roc auc": row["Roc_auc"],
            "Train Balanced Accuracy": baseline_row.get("Train Balanced Accuracy", np.nan),
            "Test Balanced Accuracy": row["Accuracy"],
            "Train Recall": baseline_row.get("Train Recall", np.nan),
            "Test Recall": row["Recall_macro"],
            "Train Precision": baseline_row.get("Train Precision", np.nan),
            "Test Precision": row["Precision_macro"],
            "Train F1": baseline_row.get("Train F1", np.nan),
            "Test F1": row["F1_macro"],
            "Parameters": dumps(threshold_params, default=str),
            "Fit Time": fit_time,
            "Score Time": score_time,
            "Total Time": total_time,
            "Business Score": business_score,
            "Business Weights": business_weights
        }

        row_mask = (
            (metric_df["Estimator"] == row_identity["Estimator"])
            & (metric_df["Optimization/Data model"] == row_identity["Optimization/Data model"])
            & (metric_df["Pre-Process Pipeline"] == row_identity["Pre-Process Pipeline"])
        )

        if row_mask.any():
            metric_df = metric_df.loc[~row_mask].copy()

        new_row = pd.DataFrame([values_to_save])

        if metric_df.empty:
            metric_df = new_row
        else:
            metric_df = pd.concat(
                [metric_df, new_row],
                ignore_index=True,
            )

    metric_df.to_csv(metric_path, index=False)
    return metric_df
