"""Reporting-only robustness and limitations report for S9.

This module consumes only persisted summaries from S6, S7 and S8.  It does
not access row-level data, fitted models or the confirmatory test partition.
All aggregations are descriptive and are deliberately labelled exploratory
when they originate after the confirmatory analysis.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ARTIFACTS_DIR, PROJECT_ROOT
from .i18n import resolve_lang
from .viz import apply_plot_style

DEFAULT_ARTIFACTS_DIR = ARTIFACTS_DIR
EXPECTED_S6_DELTA = -0.008844719981198396
EXPECTED_S6_CI = (-0.04915708997452239, 0.03024856926795489)

S9_LABELS_PT = {
    "title": "S9 — Robustez e limitações",
    "confirmatory": "Confirmatória",
    "exploratory": "Exploratória",
    "evidence": "Registro de evidências",
    "robustness": "Resumo de robustez",
    "limitations": "Registro de limitações",
    "decision": "Fronteira de decisão",
    "status": "não pronto para implantação direta",
    "inputs": "Entradas permitidas validadas",
    "s6_claim": "S6: UpliftTree não demonstrou vantagem confirmatória sobre o baseline de propensão",
    "s7_claim": "S7: a heterogeneidade exploratória não se alinha de forma consistente entre outcomes",
    "s8_claim": "S8: políticas e cenários econômicos permanecem descritivos e dependentes da validação",
    "s6_interpretation": "O IC95% inclui zero; não há evidência confirmatória de vantagem.",
    "visit_spend": "A correlação visit–spend é baixa; rankings não devem ser tratados como equivalentes.",
    "budget_warning": "Budgets são pontos descritivos da mesma exploração, não testes confirmatórios independentes.",
    "positive_scenarios": "Proporção de cenários com lucro líquido positivo entre linhas factíveis",
    "binary_stability_map": "Estabilidade descritiva por política, outcome e budget",
    "economic_map": "Sensibilidade econômica agregada",
    "limitation_map": "Mapa de limitações e prontidão",
    "budget": "Budget",
    "policy": "Política",
    "outcome": "Outcome",
    "share_positive": "Proporção com delta positivo",
    "share_ci_above": "Proporção com IC acima de zero",
    "proportion": "Proporção",
    "scenario_count": "Cenários",
    "count": "Contagem",
    "severity": "Severidade",
    "estimate": "Estimativa",
    "interval": "Intervalo",
    "not_ready": "Não pronto para implantação direta",
    "saved": "Artefatos S9 salvos em",
}


def _labels(lang=None):
    return resolve_lang(lang)(S9_LABELS_PT)


def _require_columns(frame, required, name):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} schema missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")


def _read_json(path, name):
    if not path.exists():
        raise FileNotFoundError(f"Missing allowed input: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {name} JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _read_csv(path, required, name):
    if not path.exists():
        raise FileNotFoundError(f"Missing allowed input: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid {name} CSV") from exc
    _require_columns(frame, required, name)
    return frame


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_s9_inputs(artifacts_dir=None):
    """Read and validate the persisted summaries allowed by the S9 contract."""
    root = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
    s6 = root / "s6"
    s7 = root / "s7"
    s8 = root / "s8"
    s6_results_path = s6 / "s6_results.json"
    preregistration_path = s6 / "preregistration.json"
    s6_results = _read_json(s6_results_path, "s6_results")
    preregistration = _read_json(preregistration_path, "preregistration")
    if s6_results.get("preregistration_sha256") != _sha256(preregistration_path):
        raise ValueError("S6 registration hash does not match the persisted registration")
    bootstrap = s6_results.get("bootstrap", {})
    deltas = bootstrap.get("deltas", {})
    primary = deltas.get("UpliftTree - Baseline (propensão)")
    if not isinstance(primary, dict):
        raise TypeError("S6 primary comparison is missing")
    observed = (float(primary["point_estimate"]), float(primary["ci_low"]), float(primary["ci_high"]))
    if not np.allclose(observed, (EXPECTED_S6_DELTA, *EXPECTED_S6_CI), rtol=0, atol=1e-12):
        raise ValueError("S6 primary conclusion does not match the frozen methodological contract")
    if preregistration.get("primary_outcome") != "visit":
        raise ValueError("S6 registration primary outcome is inconsistent")

    s7_files = {
        "funnel_metrics": s7 / "s7_funnel_metrics.csv",
        "funnel_spearman": s7 / "s7_funnel_spearman.csv",
        "quantile_outcomes": s7 / "s7_quantile_outcomes.csv",
        "top_bottom_profile": s7 / "s7_top_bottom_profile.csv",
    }
    s7_frames = {
        "funnel_metrics": _read_csv(
            s7_files["funnel_metrics"],
            ["outcome", "qini_auc", "uplift_auc", "uplift_at_30pct", "incremental_mean_top_30pct"],
            "s7_funnel_metrics",
        ),
        "funnel_spearman": _read_csv(
            s7_files["funnel_spearman"], ["score_a", "score_b", "spearman_corr", "p_value"], "s7_funnel_spearman"
        ),
        "quantile_outcomes": _read_csv(
            s7_files["quantile_outcomes"], ["quantile", "n", "visit_mean", "conversion_mean", "spend_mean"], "s7_quantile_outcomes"
        ),
        "top_bottom_profile": _read_csv(
            s7_files["top_bottom_profile"], ["variable", "level", "bottom_quantile", "top_quantile", "delta_top_minus_bottom"], "s7_top_bottom_profile"
        ),
    }

    s8_files = {
        "policy_comparisons": s8 / "s8_policy_comparisons.csv",
        "roi_sensitivity": s8 / "s8_roi_sensitivity.csv",
        "three_way_policy": s8 / "s8_three_way_policy.csv",
        "assumptions": s8 / "s8_assumptions.json",
    }
    s8_frames = {
        "policy_comparisons": _read_csv(
            s8_files["policy_comparisons"],
            ["policy", "budget", "outcome", "budget_feasible", "delta_vs_propensity", "delta_ci_low", "delta_ci_high"],
            "s8_policy_comparisons",
        ),
        "roi_sensitivity": _read_csv(
            s8_files["roi_sensitivity"], ["policy", "budget", "budget_feasible", "net_profit"], "s8_roi_sensitivity"
        ),
        "three_way_policy": _read_csv(
            s8_files["three_way_policy"],
            ["policy", "budget", "outcome", "budget_feasible", "incremental_value", "ci_low", "ci_high"],
            "s8_three_way_policy",
        ),
    }
    assumptions = _read_json(s8_files["assumptions"], "s8_assumptions")
    if assumptions.get("status") != "post-confirmatory exploratory":
        raise ValueError("S8 assumptions do not identify a post-confirmatory report")
    paths = {"s6_results": s6_results_path, "preregistration": preregistration_path, **s7_files, **s8_files}
    return {"s6_results": s6_results, "preregistration": preregistration, "s7": s7_frames, "s8": s8_frames, "s8_assumptions": assumptions, "paths": paths}


def _empty_robustness_row(section, subject, outcome, interpretation):
    return {
        "section": section, "subject": subject, "outcome": outcome, "n_units": 0,
        "positive_count": 0, "negative_count": 0, "zero_count": 0,
        "ci_above_zero_count": 0, "ci_below_zero_count": 0, "ci_includes_zero_count": 0,
        "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "positive_proportion": np.nan, "ci_above_zero_proportion": np.nan,
        "interpretation": interpretation,
    }


def _sign_counts(frame, value_col, low_col, high_col):
    values = pd.to_numeric(frame[value_col], errors="raise")
    lows = pd.to_numeric(frame[low_col], errors="raise")
    highs = pd.to_numeric(frame[high_col], errors="raise")
    return {
        "n_units": len(frame),
        "positive_count": int((values > 0).sum()),
        "negative_count": int((values < 0).sum()),
        "zero_count": int((values == 0).sum()),
        "ci_above_zero_count": int((lows > 0).sum()),
        "ci_below_zero_count": int((highs < 0).sum()),
        "ci_includes_zero_count": int(((lows <= 0) & (highs >= 0)).sum()),
        "positive_proportion": float((values > 0).mean()),
        "ci_above_zero_proportion": float((lows > 0).mean()),
    }


def _build_robustness_summary(inputs):
    s7 = inputs["s7"]
    s8 = inputs["s8"]
    rows = []
    comparisons = s8["policy_comparisons"]
    for policy in ("uplift_tree", "x_tree"):
        for outcome in ("visit", "spend"):
            frame = comparisons[(comparisons["policy"] == policy) & (comparisons["outcome"] == outcome)]
            if frame.empty:
                raise ValueError(f"S8 comparison missing {policy}/{outcome}")
            row = _empty_robustness_row("binary_policy_stability", policy, outcome, "Mesma curva exploratória por budget; nenhum budget é teste confirmatório.")
            row.update(_sign_counts(frame, "delta_vs_propensity", "delta_ci_low", "delta_ci_high"))
            rows.append(row)

    economics = s8["roi_sensitivity"]
    for policy, frame in economics[economics["budget_feasible"]].groupby("policy", sort=True):
        positive = frame["net_profit"] > 0
        row = _empty_robustness_row(
            "economic_sensitivity",
            policy,
            "spend",
            "Apenas cenários econômicos factíveis são contados a partir das estimativas pontuais; margens e custos são ilustrativos, e a proporção não propaga a incerteza dos efeitos.",
        )
        row.update({
            "n_units": len(frame),
            "positive_count": int(positive.sum()),
            "negative_count": int((frame["net_profit"] < 0).sum()),
            "zero_count": int((frame["net_profit"] == 0).sum()),
            "positive_proportion": float(positive.mean()),
        })
        rows.append(row)

    learned = s8["three_way_policy"]
    learned = learned[learned["policy"] == "learned"]
    for outcome, frame in learned.groupby("outcome", sort=True):
        row = _empty_robustness_row("three_way_learned", "learned", outcome, "Intervalos por budget resumem uma curva exploratória de validação; não são testes independentes.")
        row.update(_sign_counts(frame, "incremental_value", "ci_low", "ci_high"))
        rows.append(row)

    spearman = s7["funnel_spearman"]
    pairs = spearman[(spearman["score_a"] < spearman["score_b"]) & (spearman["score_a"] != spearman["score_b"])].copy()
    for item in pairs.itertuples(index=False):
        row = _empty_robustness_row("cross_outcome_alignment", f"{item.score_a}_vs_{item.score_b}", "rankings", "Alinhamento entre outcomes é descritivo; a baixa correlação visit–spend enfraquece a transferência de um ranking de visit para a economia.")
        row.update({"n_units": 1, "estimate": float(item.spearman_corr), "positive_count": int(item.spearman_corr > 0), "positive_proportion": float(item.spearman_corr > 0)})
        rows.append(row)
    return pd.DataFrame(rows)


def _build_evidence_register(inputs):
    s6 = inputs["s6_results"]["bootstrap"]["deltas"]["UpliftTree - Baseline (propensão)"]
    metrics = inputs["s7"]["funnel_metrics"].set_index("outcome")
    spearman = inputs["s7"]["funnel_spearman"].set_index(["score_a", "score_b"])
    rows = [
        {"evidence_id": "S6_primary", "stage": "S6", "evidence_class": "confirmatory", "claim": "UpliftTree - baseline de propensão", "estimate": s6["point_estimate"], "ci_low": s6["ci_low"], "ci_high": s6["ci_high"], "status": "no_confirmatory_advantage", "interpretation": "O IC95% inclui zero; a hipótese primária não foi confirmada."},
        {"evidence_id": "S7_funnel", "stage": "S7", "evidence_class": "exploratory", "claim": "Heterogeneidade e ranking no outcome visit", "estimate": float(metrics.loc["visit", "qini_auc"]), "ci_low": np.nan, "ci_high": np.nan, "status": "descriptive_only", "interpretation": "A curva é exploratória e não substitui a avaliação confirmatória."},
        {"evidence_id": "S7_visit_spend_alignment", "stage": "S7", "evidence_class": "exploratory", "claim": "Correlação entre rankings visit e spend", "estimate": float(spearman.loc[("visit", "spend"), "spearman_corr"]), "ci_low": np.nan, "ci_high": np.nan, "status": "low_alignment", "interpretation": "O baixo alinhamento limita a transferência de targeting de visit para receita."},
        {"evidence_id": "S8_policy_validation", "stage": "S8", "evidence_class": "exploratory", "claim": "Políticas de contato e ROI na validação", "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "status": "descriptive_only", "interpretation": "Consultar o resumo de robustez; budgets e cenários econômicos não autorizam deployment nem vencedor retrospectivo."},
    ]
    return pd.DataFrame(rows)


def _build_limitation_register():
    rows = [
        ("hypothesis", "Hipótese primária não confirmada", "IC95% de S6 inclui zero", "Não sustenta claim de superioridade", "Preservar S6 como resultado nulo e desenhar piloto novo", "high"),
        ("inference", "Pós-seleção e exploração", "S7/S8 foram produzidos após S6", "Risco de otimismo e multiplicidade", "Tratar como geração de hipótese; pré-especificar próximo teste", "high"),
        ("data", "Conversion raro", "Taxa baixa e ICs amplos nos resumos", "Baixa potência para decisões de conversão", "Definir outcome e amostra com power analysis", "high"),
        ("economics", "Spend é proxy e margem/custo são hipotéticos", "S8 usa cenários ilustrativos", "ROI não é business case", "Usar receita incremental, margem e custo reais", "high"),
        ("alignment", "Baixa correlação visit–spend", "Correlação Spearman visit–spend persistida em S7", "Ranking de visit pode não maximizar valor", "Avaliar outcome econômico pré-especificado", "high"),
        ("uncertainty", "Incerteza dos ICs", "Muitos intervalos incluem zero", "Direção e magnitude são instáveis", "Aumentar amostra e congelar análise", "high"),
        ("interpretability", "Surrogate descritivo da ação", "S8 mede aproximação da ação, não efeito causal", "Regras não são explicação causal individual", "Usar perfis por quantil e surrogate apenas para comunicação", "medium"),
        ("external_validity", "Validade externa limitada", "Hillstrom é campanha histórica", "Resultados podem não transportar", "Revalidar em campanha e população atuais", "high"),
        ("operations", "Política usa validação, não piloto prospectivo", "Nenhuma execução randomizada futura foi observada", "Não há evidência operacional", "Executar piloto randomizado com política congelada", "high"),
        ("operations", "Ausência de efeitos adversos, fadiga, capacidade e entregabilidade", "Artefatos não medem esses guardrails", "Risco operacional e de experiência", "Monitorar supressão, reclamações, saturação e capacidade", "high"),
        ("multiplicity", "Comparações secundárias e multiplicidade", "Vários modelos, outcomes e budgets", "Falsos positivos exploratórios", "Não selecionar máximos; pré-especificar uma análise", "high"),
    ]
    return pd.DataFrame(rows, columns=["category", "risk", "evidence", "impact", "mitigation_next_step", "severity"])


def _decision_boundary():
    return {
        "status": "not_ready_for_direct_deployment",
        "reasons": ["S6 não confirmou vantagem do UpliftTree sobre o baseline", "S7/S8 são pós-confirmatórios e exploratórios", "visit e spend têm baixo alinhamento", "custos, margens e guardrails operacionais não estão validados"],
        "allowed_uses": ["aprendizado e geração de hipóteses", "desenho de piloto prospectivo randomizado", "simulação ilustrativa com premissas explicitamente substituíveis"],
        "prohibited_uses": ["claim de superioridade confirmatória", "deployment automático", "escolha retrospectiva de vencedor", "tratar máximo entre budgets como teste independente"],
        "prospective_randomized_pilot_requirements": {
            "policy_frozen_before_launch": True,
            "real_cost_and_margin": True,
            "primary_outcome_pre_specified": True,
            "operational_guardrails": ["fadiga", "entregabilidade", "descadastro", "reclamações", "capacidade"],
            "analysis": ["ITT", "IPW se houver aderência/ponderação definida antes"],
            "sample_size_and_power": True,
            "monitoring_plan": True,
        },
    }


def build_s9_report(inputs):
    """Build S9 tables from an already validated, persisted-input bundle."""
    if not isinstance(inputs, dict) or not {"s6_results", "s7", "s8", "paths"}.issubset(inputs):
        raise ValueError("S9 inputs must come from load_s9_inputs or an equivalent validated bundle")
    return {
        "evidence_register": _build_evidence_register(inputs),
        "robustness_summary": _build_robustness_summary(inputs),
        "limitation_register": _build_limitation_register(),
        "decision_boundary": _decision_boundary(),
        "source_manifest": {
            "stage": "S9",
            "status": "reporting-only post-confirmatory",
            "allowed_inputs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in inputs["paths"].items()},
            "sealed_data_copied": False,
            "models_loaded": False,
            "confirmatory_metrics_recalculated": False,
        },
    }


def save_s9_artifacts(report, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else ARTIFACTS_DIR / "s9"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "evidence_register": output_dir / "s9_evidence_register.csv",
        "robustness_summary": output_dir / "s9_robustness_summary.csv",
        "limitation_register": output_dir / "s9_limitation_register.csv",
        "decision_boundary": output_dir / "s9_decision_boundary.json",
        "source_manifest": output_dir / "s9_source_manifest.json",
    }
    report["evidence_register"].to_csv(files["evidence_register"], index=False)
    report["robustness_summary"].to_csv(files["robustness_summary"], index=False)
    report["limitation_register"].to_csv(files["limitation_register"], index=False)
    files["decision_boundary"].write_text(json.dumps(report["decision_boundary"], indent=2, ensure_ascii=False), encoding="utf-8")
    files["source_manifest"].write_text(json.dumps(report["source_manifest"], indent=2, ensure_ascii=False), encoding="utf-8")
    return files


_SECTION_PT = {
    "binary_policy_stability": "Estabilidade da política binária",
    "economic_sensitivity": "Sensibilidade econômica",
    "three_way_learned": "Política aprendida de três braços",
    "cross_outcome_alignment": "Alinhamento entre outcomes",
}
_CLASS_PT = {"confirmatory": "Confirmatória", "exploratory": "Exploratória"}
_CATEGORY_PT = {
    "hypothesis": "Hipótese", "inference": "Inferência", "data": "Dados", "economics": "Economia",
    "alignment": "Alinhamento", "uncertainty": "Incerteza", "interpretability": "Interpretabilidade",
    "external_validity": "Validade externa", "operations": "Operações", "multiplicity": "Multiplicidade",
}
_SEVERITY_PT = {"high": "Alta", "medium": "Média", "low": "Baixa"}
_POLICY_PT = {
    "no_contact": "Não tratar",
    "propensity": "Propensão",
    "random": "Aleatória",
    "treat_all": "Tratar todos",
    "uplift_tree": "UpliftTree",
    "x_tree": "X+Tree",
    "learned": "Aprendida",
}
_OUTCOME_PT = {"visit": "visit", "spend": "spend", "rankings": "rankings"}
_STATUS_PT = {
    "no_confirmatory_advantage": "Sem vantagem confirmatória",
    "descriptive_only": "Somente descritivo",
    "low_alignment": "Baixo alinhamento",
}
_DECISION_KEY_PT = {
    "status": "status", "reasons": "razões", "allowed_uses": "usos permitidos",
    "prohibited_uses": "usos proibidos", "prospective_randomized_pilot_requirements": "requisitos do piloto prospectivo randomizado",
    "policy_frozen_before_launch": "política congelada antes do lançamento", "real_cost_and_margin": "custo e margem reais",
    "primary_outcome_pre_specified": "outcome primário pré-especificado", "operational_guardrails": "guardrails operacionais",
    "analysis": "análise", "sample_size_and_power": "tamanho da amostra e poder", "monitoring_plan": "plano de monitoramento",
}
_DECISION_VALUE_PT = {
    "not_ready_for_direct_deployment": "Não pronto para implantação direta",
    "reporting-only post-confirmatory": "Reporting-only pós-confirmatório",
    "S6 não confirmou vantagem do UpliftTree sobre o baseline": "S6 não confirmou vantagem do UpliftTree sobre o baseline",
    "S7/S8 são pós-confirmatórios e exploratórios": "S7/S8 são pós-confirmatórios e exploratórios",
    "visit e spend têm baixo alinhamento": "visit e spend têm baixo alinhamento",
    "custos, margens e guardrails operacionais não estão validados": "custos, margens e guardrails operacionais não estão validados",
    "aprendizado e geração de hipóteses": "aprendizado e geração de hipóteses",
    "desenho de piloto prospectivo randomizado": "desenho de piloto prospectivo randomizado",
    "simulação ilustrativa com premissas explicitamente substituíveis": "simulação ilustrativa com premissas explicitamente substituíveis",
    "claim de superioridade confirmatória": "claim de superioridade confirmatória",
    "deployment automático": "deployment automático",
    "escolha retrospectiva de vencedor": "escolha retrospectiva de vencedor",
    "tratar máximo entre budgets como teste independente": "tratar máximo entre budgets como teste independente",
    "fadiga": "fadiga", "entregabilidade": "entregabilidade", "descadastro": "descadastro", "reclamações": "reclamações", "capacidade": "capacidade",
    "ITT": "ITT", "IPW se houver aderência/ponderação definida antes": "IPW se houver aderência/ponderação definida antes",
}


def _translate_text(value, lang):
    if not isinstance(value, str):
        return value
    return resolve_lang(lang)({"_value": value})["_value"]


def _localize_frame(frame, lang, mappings):
    localized = frame.copy()
    for column, mapping in mappings.items():
        if column in localized:
            localized[column] = localized[column].map(
                lambda value, mapping=mapping: _translate_text(mapping.get(value, value), lang)
            )
    return localized


def _localize_decision(value, lang):
    if isinstance(value, dict):
        return {_translate_text(_DECISION_KEY_PT.get(key, key), lang): _localize_decision(item, lang) for key, item in value.items()}
    if isinstance(value, list):
        return [_localize_decision(item, lang) for item in value]
    if isinstance(value, str):
        return _translate_text(_DECISION_VALUE_PT.get(value, value), lang)
    return value


def _localized_report_frames(report, lang):
    evidence = _localize_frame(report["evidence_register"], lang, {
        "evidence_class": _CLASS_PT,
        "status": _STATUS_PT,
    })
    for column in ("claim", "interpretation"):
        evidence[column] = evidence[column].map(lambda value: _translate_text(value, lang))
    robustness = _localize_frame(report["robustness_summary"], lang, {"section": _SECTION_PT})
    robustness["interpretation"] = robustness["interpretation"].map(lambda value: _translate_text(value, lang))
    limitations = report["limitation_register"].copy()
    for column in ("risk", "evidence", "impact", "mitigation_next_step"):
        limitations[column] = limitations[column].map(lambda value: _translate_text(value, lang))
    limitations["severity"] = limitations["severity"].map(lambda value: _translate_text(_SEVERITY_PT.get(value, value), lang))
    limitations["category"] = limitations["category"].map(lambda value: _translate_text(_CATEGORY_PT.get(value, value), lang))
    return evidence, robustness, limitations


def display_s9_report(report, lang=None, artifacts_dir=None):
    labels = _labels(lang)
    evidence, robustness, limitations = _localized_report_frames(report, lang)
    print(f"{labels['title']} | {labels['status']}")
    print(labels["evidence"])
    print(evidence.to_string(index=False))
    print(labels["robustness"])
    print(robustness.to_string(index=False))
    print(labels["limitations"])
    print(limitations.to_string(index=False))
    print(labels["decision"])
    print(json.dumps(_localize_decision(report["decision_boundary"], lang), indent=2, ensure_ascii=False))
    if artifacts_dir is not None:
        try:
            path = Path(artifacts_dir).relative_to(PROJECT_ROOT)
        except ValueError:
            path = Path(artifacts_dir)
        print(f"{labels['saved']} {path}")


def plot_s9_report(report, lang=None, *, show=True):
    """Render three audit-oriented figures and return their figure objects."""
    labels = _labels(lang)
    apply_plot_style()
    figures = []
    stability = report["robustness_summary"]
    stability = stability[stability["section"] == "binary_policy_stability"].copy()
    stability["label"] = stability["subject"].map(lambda value: _translate_text(_POLICY_PT.get(value, value), lang)) + " / " + stability["outcome"].map(lambda value: _translate_text(_OUTCOME_PT.get(value, value), lang))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(stability))
    width = 0.38
    ax.bar(x - width / 2, stability["positive_proportion"], width, label=labels["share_positive"], color="#2a9d8f")
    ax.bar(x + width / 2, stability["ci_above_zero_proportion"], width, label=labels["share_ci_above"], color="#1f4e79")
    ax.set_xticks(x, stability["label"], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(labels["binary_stability_map"])
    ax.set_xlabel(f"{labels['policy']} / {labels['outcome']}")
    ax.set_ylabel(labels["proportion"])
    ax.legend()
    figures.append(fig)

    economic = report["robustness_summary"]
    economic = economic[economic["section"] == "economic_sensitivity"].sort_values("positive_proportion").copy()
    economic["subject"] = economic["subject"].map(lambda value: _translate_text(_POLICY_PT.get(value, value), lang))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=economic, x="positive_proportion", y="subject", color="#2a9d8f", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel(labels["proportion"])
    ax.set_ylabel("")
    ax.set_title(labels["economic_map"])
    figures.append(fig)

    limitations = report["limitation_register"].copy()
    severity_order = ["low", "medium", "high"]
    counts = limitations["severity"].value_counts().reindex(severity_order, fill_value=0)
    counts.index = [_translate_text(_SEVERITY_PT.get(value, value), lang) for value in counts.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, color="#8d99ae", ax=ax)
    ax.set_xlabel(labels["count"])
    ax.set_ylabel(labels["severity"])
    ax.set_title(f"{labels['limitation_map']} — {labels['not_ready']}")
    figures.append(fig)
    if show:
        plt.show()
    return figures
