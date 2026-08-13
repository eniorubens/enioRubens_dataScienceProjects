"""Orchestration, presentation and persistence for Notebook 07 (S8).

The causal/economic primitives remain in :mod:`src.policy`.  This module owns
the reproducible S8 report assembled from train/validation data, its tables,
plots, localized display and artifact persistence.  It never loads the sealed
test or S6 artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PROJECT_ROOT, SEED
from .i18n import resolve_lang
from .policy import (
    all_arm_actions,
    build_roi_sensitivity,
    evaluate_binary_policies,
    evaluate_three_way_policies,
    fit_binary_policy_scores,
    fit_three_way_models,
    fit_three_way_surrogate,
    make_binary_policy_masks,
    predict_three_way_models,
    random_three_way_actions,
    roi_metrics,
    three_way_actions,
    three_way_net_gains,
)
from .viz import apply_plot_style

DEFAULT_BUDGETS = tuple(round(value, 2) for value in np.arange(0.05, 1.001, 0.05))
DEFAULT_REPRESENTATIVE_BUDGETS = (0.10, 0.30, 0.50)
DEFAULT_MARGINS = (0.30, 0.50, 0.70)
DEFAULT_EMAIL_COSTS = (0.02, 0.05, 0.10)
DEFAULT_ILLUSTRATIVE_MARGIN = 0.50
DEFAULT_ILLUSTRATIVE_EMAIL_COST = 0.05
DEFAULT_BINARY_N_BOOT = 200
DEFAULT_THREE_WAY_N_BOOT = 100

S8_LABELS_PT = {
    "train": "treino",
    "validation": "validação",
    "fit_boundary": "Modelos ajustados em train_df; políticas avaliadas em val_df.",
    "binary_table": "Valor IPW de visit e delta vs. propensão nos budgets representativos",
    "three_way_table": "Política 3-way learned: alocação e valor incremental de spend nos budgets representativos",
    "surrogate_header": "Surrogate descritivo da política 3-way (não causal)",
    "fidelity": "fidelity",
    "balanced_accuracy": "balanced_accuracy",
    "action_distribution": "Distribuição das ações",
    "rules": "Regras descritivas",
    "artifacts_saved": "Artefatos S8 salvos em",
    "budget": "Orçamento de contato",
    "ipw_visit": "Valor incremental IPW de visit por cliente",
    "policy_value": "Valor da política na validação: outcome primário",
    "net_profit": "Lucro líquido ilustrativo por 1.000 clientes",
    "economic_sensitivity": "Sensibilidade econômica",
    "email_cost": "Custo do e-mail",
    "gross_margin": "Margem bruta",
    "heatmap_title": "Sensibilidade do lucro líquido ilustrativo: UpliftTree com budget de 30%",
    "allocation_share": "Participação na alocação",
    "allocation_title": "Alocação exploratória da política 3-way aprendida",
}
S8_POLICY_LABELS_PT = {
    "random": "Aleatória",
    "propensity": "Propensão",
    "no_contact": "Não tratar",
    "treat_all": "Tratar todos",
    "uplift_tree": "UpliftTree",
    "x_tree": "X+Tree",
}
S8_ARM_LABELS_PT = {
    "no_email": "No E-Mail",
    "mens_email": "Mens E-Mail",
    "womens_email": "Womens E-Mail",
}


def _labels(lang=None):
    return resolve_lang(lang)(S8_LABELS_PT)


def _policy_labels(lang=None):
    return resolve_lang(lang)(S8_POLICY_LABELS_PT)


def _arm_labels(lang=None):
    return resolve_lang(lang)(S8_ARM_LABELS_PT)


def _as_float_tuple(values, name):
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def _representative_table(frame, budgets, columns, sort_columns):
    return (
        frame[frame["budget"].isin(budgets)][columns]
        .sort_values(sort_columns)
        .round(4)
    )


def build_s8_policy_report(
    train_df,
    val_df,
    *,
    budgets=DEFAULT_BUDGETS,
    representative_budgets=DEFAULT_REPRESENTATIVE_BUDGETS,
    margins=DEFAULT_MARGINS,
    email_costs=DEFAULT_EMAIL_COSTS,
    illustrative_margin=DEFAULT_ILLUSTRATIVE_MARGIN,
    illustrative_email_cost=DEFAULT_ILLUSTRATIVE_EMAIL_COST,
    binary_n_boot=DEFAULT_BINARY_N_BOOT,
    three_way_n_boot=DEFAULT_THREE_WAY_N_BOOT,
    seed=SEED,
):
    """Build the complete exploratory S8 report from train and validation data."""
    budgets = tuple(round(value, 2) for value in _as_float_tuple(budgets, "budgets"))
    representative_budgets = _as_float_tuple(representative_budgets, "representative_budgets")
    margins = _as_float_tuple(margins, "margins")
    email_costs = _as_float_tuple(email_costs, "email_costs")

    binary_scores = fit_binary_policy_scores(train_df, val_df, outcome="visit", seed=seed)
    binary_masks = make_binary_policy_masks(binary_scores, budgets, seed=seed)
    policy_curve = evaluate_binary_policies(
        val_df,
        binary_masks,
        outcomes=("visit", "conversion", "spend"),
        n_boot=binary_n_boot,
        seed=seed,
    )
    binary_table = _representative_table(
        policy_curve[policy_curve["outcome"] == "visit"],
        representative_budgets,
        ["policy", "budget", "incremental_value", "delta_vs_propensity", "ci_low", "ci_high", "budget_feasible"],
        ["budget", "policy"],
    )

    roi_sensitivity = build_roi_sensitivity(
        policy_curve,
        margins,
        email_costs,
        outcome="spend",
        n_customers=len(val_df),
    )
    economic_rows = []
    for row in policy_curve[policy_curve["outcome"] == "spend"].itertuples(index=False):
        economic_rows.append({
            "policy": row.policy,
            "budget": row.budget,
            "budget_feasible": row.budget_feasible,
            "contact_rate": row.contact_rate,
            "incremental_spend_per_customer": row.incremental_value,
            **roi_metrics(
                row.incremental_value,
                row.contact_rate,
                illustrative_margin,
                illustrative_email_cost,
                n_customers=len(val_df),
            ),
        })
    economic_curve = pd.DataFrame(economic_rows)

    three_way_models = fit_three_way_models(train_df, seed=seed)
    three_way_predictions = predict_three_way_models(three_way_models, val_df)
    three_way_gains = three_way_net_gains(
        three_way_predictions,
        gross_margin=illustrative_margin,
        email_cost=illustrative_email_cost,
        outcome="spend",
    )
    three_way_actions_by_policy = {
        "all_no_email": {budget: all_arm_actions(len(val_df), 0) for budget in budgets},
        "all_mens": {budget: all_arm_actions(len(val_df), 1) for budget in budgets},
        "all_womens": {budget: all_arm_actions(len(val_df), 2) for budget in budgets},
        "random": {
            budget: random_three_way_actions(
                len(val_df), budget, seed=seed + int(budget * 1000)
            )
            for budget in budgets
        },
        "learned": {budget: three_way_actions(three_way_gains, budget) for budget in budgets},
    }
    three_way_curve = evaluate_three_way_policies(
        val_df,
        three_way_actions_by_policy,
        n_boot=three_way_n_boot,
        seed=seed,
    )
    surrogate_actions = three_way_actions(three_way_gains, 0.30)
    surrogate = fit_three_way_surrogate(val_df, surrogate_actions, seed=seed)
    three_way_table = _representative_table(
        three_way_curve[
            (three_way_curve["outcome"] == "spend")
            & (three_way_curve["policy"] == "learned")
        ],
        representative_budgets,
        ["policy", "budget", "no_email_rate", "mens_rate", "womens_rate", "incremental_value", "ci_low", "ci_high", "budget_feasible"],
        ["budget"],
    )

    policy_comparisons = policy_curve[
        policy_curve["outcome"].isin(["visit", "spend"])
        & policy_curve["policy"].isin(
            ["random", "propensity", "uplift_tree", "x_tree", "no_contact", "treat_all"]
        )
    ].copy()
    assumptions = {
        "status": "post-confirmatory exploratory",
        "fit_partition": "train_df only",
        "evaluation_partition": "val_df only",
        "pooled_propensity": 2 / 3,
        "three_arm_propensity": 1 / 3,
        "gross_margin_grid": list(margins),
        "email_cost_grid": list(email_costs),
        "illustrative_margin": float(illustrative_margin),
        "illustrative_email_cost": float(illustrative_email_cost),
        "economic_note": "Observed spend is a revenue proxy; gross margin and email cost are illustrative scenarios, not company facts.",
        "treat_all_note": "treat_all and all-arm policies are unconditional benchmarks and are infeasible below 100% contact.",
        "s6_note": "S8 does not change the confirmatory S6 result or select a retrospective winner.",
    }
    return {
        "train_n": len(train_df),
        "validation_n": len(val_df),
        "budgets": budgets,
        "representative_budgets": representative_budgets,
        "margins": margins,
        "email_costs": email_costs,
        "seed": int(seed),
        "binary_n_boot": int(binary_n_boot),
        "three_way_n_boot": int(three_way_n_boot),
        "binary_scores": binary_scores,
        "binary_masks": binary_masks,
        "policy_curve": policy_curve,
        "binary_table": binary_table,
        "roi_sensitivity": roi_sensitivity,
        "economic_curve": economic_curve,
        "three_way_models": three_way_models,
        "three_way_predictions": three_way_predictions,
        "three_way_gains": three_way_gains,
        "three_way_actions_by_policy": three_way_actions_by_policy,
        "three_way_curve": three_way_curve,
        "three_way_table": three_way_table,
        "surrogate": surrogate,
        "policy_comparisons": policy_comparisons,
        "assumptions": assumptions,
    }


def save_s8_policy_artifacts(report, output_dir=None):
    """Persist the four S8 tables and assumptions using the established names."""
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "artifacts" / "s8"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "policy_curve": output_dir / "s8_policy_curve.csv",
        "policy_comparisons": output_dir / "s8_policy_comparisons.csv",
        "roi_sensitivity": output_dir / "s8_roi_sensitivity.csv",
        "three_way_policy": output_dir / "s8_three_way_policy.csv",
        "assumptions": output_dir / "s8_assumptions.json",
    }
    report["policy_curve"].to_csv(files["policy_curve"], index=False)
    report["policy_comparisons"].to_csv(files["policy_comparisons"], index=False)
    report["roi_sensitivity"].to_csv(files["roi_sensitivity"], index=False)
    report["three_way_curve"].to_csv(files["three_way_policy"], index=False)
    files["assumptions"].write_text(
        json.dumps(report["assumptions"], indent=2), encoding="utf-8"
    )
    return files


def display_s8_policy_report(report, lang=None, artifacts_dir=None):
    """Print the representative S8 tables and surrogate in the selected language."""
    labels = _labels(lang)
    print(f"{labels['train']}={report['train_n']:,} | {labels['validation']}={report['validation_n']:,}")
    print(labels["fit_boundary"])
    print(labels["binary_table"])
    print(report["binary_table"].to_string(index=False))
    print(labels["three_way_table"])
    print(report["three_way_table"].to_string(index=False))
    surrogate = report["surrogate"]
    print(labels["surrogate_header"])
    print(f"{labels['fidelity']}={surrogate['fidelity']:.3f}")
    print(f"{labels['balanced_accuracy']}={surrogate['balanced_accuracy']:.3f}")
    print(f"{labels['action_distribution']}: {surrogate['action_distribution']}")
    print(labels["rules"])
    print(surrogate["rules"])
    if artifacts_dir is not None:
        artifacts_path = Path(artifacts_dir)
        try:
            display_path = artifacts_path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = artifacts_path
        print(f"{labels['artifacts_saved']} {display_path}")


def plot_s8_policy_report(report, lang=None, *, show=True):
    """Render the four S8 figures and return their figure objects."""
    labels = _labels(lang)
    policy_labels = _policy_labels(lang)
    arm_labels = _arm_labels(lang)
    apply_plot_style()
    policy_curve = report["policy_curve"]
    economic_curve = report["economic_curve"]
    roi_sensitivity = report["roi_sensitivity"]
    three_way_curve = report["three_way_curve"]
    figures = []

    fig, ax = plt.subplots(figsize=(14, 6))
    for policy in ["random", "propensity", "uplift_tree", "x_tree", "no_contact", "treat_all"]:
        sub = policy_curve[
            (policy_curve["policy"] == policy) & (policy_curve["outcome"] == "visit")
        ].sort_values("budget")
        style = "--" if policy == "treat_all" else "-"
        ax.plot(sub["budget"], sub["incremental_value"], label=policy_labels[policy], linestyle=style)
        if policy != "treat_all":
            ax.fill_between(sub["budget"], sub["ci_low"], sub["ci_high"], alpha=0.10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel=labels["budget"], ylabel=labels["ipw_visit"], title=labels["policy_value"])
    ax.legend(ncol=3)
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    for policy in ["random", "propensity", "uplift_tree", "x_tree", "no_contact"]:
        sub = economic_curve[
            (economic_curve["policy"] == policy) & economic_curve["budget_feasible"]
        ].sort_values("budget")
        ax.plot(sub["budget"], sub["value_per_1000"], label=policy_labels[policy])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel=labels["budget"],
        ylabel=labels["net_profit"],
        title=f"{labels['economic_sensitivity']} ({labels['gross_margin']}={report['assumptions']['illustrative_margin']:.0%}, {labels['email_cost']}={report['assumptions']['illustrative_email_cost']:.2f})",
    )
    ax.legend(ncol=3)
    figures.append(fig)

    heat = roi_sensitivity[
        (roi_sensitivity["policy"] == "uplift_tree") & (roi_sensitivity["budget"] == 0.30)
    ].pivot(index="gross_margin", columns="email_cost", values="net_profit")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title(labels["heatmap_title"])
    ax.set_xlabel(labels["email_cost"])
    ax.set_ylabel(labels["gross_margin"])
    figures.append(fig)

    plot_three = three_way_curve[
        (three_way_curve["outcome"] == "spend") & (three_way_curve["policy"] == "learned")
    ].sort_values("budget")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(
        plot_three["budget"],
        plot_three["no_email_rate"],
        plot_three["mens_rate"],
        plot_three["womens_rate"],
        labels=[arm_labels["no_email"], arm_labels["mens_email"], arm_labels["womens_email"]],
    )
    ax.set(xlabel=labels["budget"], ylabel=labels["allocation_share"], title=labels["allocation_title"])
    ax.legend(loc="upper left")
    figures.append(fig)

    if show:
        plt.show()
    return figures
