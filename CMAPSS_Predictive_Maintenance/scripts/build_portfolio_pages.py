"""Gera o relatório bilíngue estático para GitHub Pages a partir de evidências salvas."""
from pathlib import Path
import hashlib
import html
import json
import shutil

import pandas as pd
import plotly.express as px
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
REPORTS = ROOT / "outputs" / "reports"
ASSETS = SITE / "assets"
DATASET_REPOSITORY = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
DATASET_CATALOG = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
TEXT = {
    "pt": {
        "lang": "pt-BR", "other": "English", "other_file": "en.html",
        "title": "Quanto tempo resta ao motor?",
        "dataset_heading": "De onde vêm os dados?",
        "dataset_origin": "O Turbofan Engine Degradation Simulation Data Set foi disponibilizado pelo Prognostics Center of Excellence (PCoE) do NASA Ames Research Center (Saxena e Goebel, 2008). NASA significa National Aeronautics and Space Administration, a agência espacial e aeronáutica dos Estados Unidos. No título, NASA identifica a origem dos dados; a análise e o modelo são trabalhos autorais de portfólio.",
        "dataset_context": "C-MAPSS significa Commercial Modular Aero-Propulsion System Simulation e simula motores aeronáuticos turbofan comerciais. No FD001, há 100 motores de treino e 100 de teste, uma condição operacional e degradação do compressor de alta pressão (HPC). O treino acompanha os motores até a falha; os históricos de teste terminam antes dela, com RUL final fornecido separadamente. São dados simulados, e não evidência de desempenho em uma frota real.",
        "repository_label": "Repositório oficial da NASA",
        "catalog_label": "Catálogo oficial do dataset",
        "tag": "NASA C-MAPSS · FD001 · Portfólio de ciência de dados",
        "intro": "Previsão de vida útil restante com histórico dos sensores, validação por motor e análise das consequências de manutenção.",
        "metric": "MAE oficial", "selected": "Atributos selecionados", "engines": "Motores no teste",
        "scope": "Dados simulados · previsão em ciclos · Extra Trees",
        "contents": "Nesta análise", "names": ["Benchmark", "Histórico dos sensores", "Alertas", "Custos", "Incerteza", "Reprodução"],
        "benchmark": "O Extra Trees apresentou o menor MAE e RMSE entre os três finalistas. O erro assinado de +10,08 ciclos revela superestimação da vida restante. Cada motor do teste contribuiu com uma previsão no último ciclo disponível.",
        "benchmark_note": "Hiperparâmetros foram congelados antes do benchmark. A escolha do vencedor ocorreu após a comparação no teste oficial; estes números não são uma avaliação independente adicional após a seleção.",
        "features": "Dos 205 atributos oferecidos, 93 foram selecionados: 13 de base, 60 médias móveis e 20 desvios-padrão móveis. As janelas de 5, 10, 20, 40 e 80 ciclos incluem somente o presente e o passado de cada motor.",
        "features_note": "Depois do ciclo observado, desvios-padrão em janelas longas aparecem entre as maiores importâncias. A importância por permutação mede a piora do erro ao embaralhar uma feature. Atributos correlacionados podem dividir importância; a análise não estabelece causalidade.",
        "alerts": "O primeiro alerta é confirmado após três ciclos consecutivos abaixo do limiar. Uma antecedência de 5 a 30 ciclos foi considerada útil neste cenário ilustrativo.",
        "alerts_note": "Com limiar 20, os 100 motores OOF tiveram alertas oportunos, com mediana de 16 ciclos de antecedência. Isso é um resultado de desenvolvimento: os folds também participaram da otimização. Não constitui validação industrial.",
        "costs": "Foram atribuídos custos relativos a alertas oportunos, prematuros, tardios e não detectados. Alertas tardios receberam a mesma penalidade de falhas não detectadas. Os valores são hipotéticos.",
        "costs_note": "Os cenários empatam no limiar 20. O verde identifica o cenário Intermediário como referência visual; ele não é sempre o cenário de menor custo.",
        "uncertainty": "Metade dos motores de cada fold OOF foi destinada à calibração da largura e metade à avaliação. A cobertura média dos ciclos por motor foi 85,9%, com largura média de 91,3 ciclos; 46% das trajetórias foram inteiramente cobertas.",
        "uncertainty_note": "Para RUL previsto acima de 100 ciclos, a cobertura caiu para 76,4%. A faixa global ficou ampla e não atingiu a referência em todos os regimes. O percentil 90 de calibração não oferece garantia de cobertura. Os intervalos não foram validados para o modelo refitado.",
        "reproduction": "O artefato final fornece previsão pontual a partir de um CSV com histórico contínuo desde o ciclo 1. A exportação verificou serialização, consistência da engenharia e invariância de previsões passadas quando são acrescentados ciclos futuros.",
        "limits": "A entrega cobre somente FD001 simulado. Custos, alertas e intervalos são exploratórios. Generalização para FD002–FD004 e validação industrial permanecem extensões.",
        "download": "Baixar síntese técnica (.ipynb)", "readme": "Baixar instruções (.md)",
        "data": "Baixar dados dos gráficos", "back": "Voltar ao início", "evidence": "Fontes e rastreabilidade",
        "source": "Os gráficos foram gerados a partir dos relatórios CSV locais, sem novo treino ou avaliação no teste. O manifesto registra os hashes das fontes. Os valores oficiais vêm do registro congelado da decisão final.",
        "cycles": "Ciclos", "model": "Modelo", "feature": "Atributo", "importance": "Aumento médio do MAE (ciclos)",
        "threshold": "Limiar de RUL previsto (ciclos)", "percent": "Motores (%)", "outcome": "Resultado",
        "coverage": "Cobertura média por motor (%)", "band": "Faixa de RUL previsto",
        "cost": "Custo relativo médio por motor", "scenario": "Cenário", "reference": "Referência de calibração: 90%",
        "statuses": ["Prematuro", "Oportuno", "Tardio", "Não detectada"],
        "scenarios": ["Intermediário", "Falha muito cara", "Intervenção prematura cara"],
        "bands": ["Até 30 ciclos", "31 a 60 ciclos", "61 a 100 ciclos", "Acima de 100 ciclos"],
    },
    "en": {
        "lang": "en-US", "other": "Português", "other_file": "index.html",
        "title": "How much life does the engine have left?",
        "dataset_heading": "Where do the data come from?",
        "dataset_origin": "The Turbofan Engine Degradation Simulation Data Set was provided by the Prognostics Center of Excellence (PCoE) at NASA Ames Research Center (Saxena and Goebel, 2008). NASA stands for National Aeronautics and Space Administration, the United States agency for space and aeronautics. In the title, NASA identifies the data source; the analysis and model are independent portfolio work.",
        "dataset_context": "C-MAPSS stands for Commercial Modular Aero-Propulsion System Simulation and simulates commercial aircraft turbofan engines. FD001 contains 100 training and 100 test engines, one operating condition and high-pressure compressor (HPC) degradation. Training trajectories continue until failure; test histories stop earlier, with the final RUL supplied separately. These are simulated data, not evidence of performance on a real fleet.",
        "repository_label": "Official NASA repository",
        "catalog_label": "Official dataset catalog",
        "tag": "NASA C-MAPSS · FD001 · Data science portfolio",
        "intro": "Remaining useful life prediction using sensor history, engine-level validation and analysis of maintenance consequences.",
        "metric": "Official MAE", "selected": "Selected features", "engines": "Test engines",
        "scope": "Simulated data · predictions in cycles · Extra Trees",
        "contents": "In this report", "names": ["Benchmark", "Sensor history", "Alerts", "Costs", "Uncertainty", "Reproduction"],
        "benchmark": "Extra Trees achieved the lowest MAE and RMSE among the three finalists. Its mean signed error of +10.08 cycles reveals overestimation of remaining life. Each test engine contributed one prediction at the last available cycle.",
        "benchmark_note": "Hyperparameters were frozen before the benchmark. The winner was selected after comparing official test results; these numbers are not another independent evaluation after selection.",
        "features": "Of 205 offered features, 93 were selected: 13 base features, 60 rolling means and 20 rolling standard deviations. Windows of 5, 10, 20, 40 and 80 cycles use only each engine's current and past measurements.",
        "features_note": "After observed cycle, standard deviations over longer windows rank among the most influential features. Permutation importance measures error deterioration when a feature is shuffled. Correlated features may share importance; this analysis does not establish causality.",
        "alerts": "The first alert is confirmed after three consecutive cycles below the threshold. Lead times of 5 to 30 cycles were considered useful in this illustrative scenario.",
        "alerts_note": "At threshold 20, all 100 OOF engines received timely alerts, with median lead time of 16 cycles. This is a development result: these folds also participated in optimization. It is not industrial validation.",
        "costs": "Relative costs were assigned to timely, premature, late and undetected alerts. Late alerts received the same penalty as undetected failures. Values are hypothetical.",
        "costs_note": "All scenarios tie at threshold 20. Green identifies the Intermediate scenario as a visual reference; it is not always the least expensive scenario.",
        "uncertainty": "Half of the engines within each OOF fold were used to calibrate width and half to evaluate it. Mean within-engine cycle coverage was 85.9%, with mean width of 91.3 cycles; 46% of trajectories were fully covered.",
        "uncertainty_note": "For predicted RUL above 100 cycles, coverage fell to 76.4%. The global interval was wide and did not reach the reference in every regime. The calibration 90th percentile is not a coverage guarantee. Intervals were not validated for the refitted model.",
        "reproduction": "The final artifact supplies point predictions from a CSV containing continuous history starting at cycle 1. Export checks verified serialization, feature construction and invariance of earlier predictions when future cycles are appended.",
        "limits": "The delivery covers simulated FD001 only. Costs, alerts and intervals are exploratory. Generalization to FD002–FD004 and industrial validation remain extensions.",
        "download": "Download technical summary (.ipynb)", "readme": "Download instructions (.md)",
        "data": "Download chart data", "back": "Back to top", "evidence": "Sources and traceability",
        "source": "Charts were generated from saved local CSV reports, without retraining or another test evaluation. The manifest records source hashes. Official values come from the frozen final-selection record.",
        "cycles": "Cycles", "model": "Model", "feature": "Feature", "importance": "Mean increase in MAE (cycles)",
        "threshold": "Predicted RUL threshold (cycles)", "percent": "Engines (%)", "outcome": "Outcome",
        "coverage": "Mean within-engine coverage (%)", "band": "Predicted RUL band",
        "cost": "Mean relative cost per engine", "scenario": "Scenario", "reference": "Calibration reference: 90%",
        "statuses": ["Premature", "Timely", "Late", "Undetected"],
        "scenarios": ["Intermediate", "Expensive failure", "Expensive early intervention"],
        "bands": ["Up to 30 cycles", "31–60 cycles", "61–100 cycles", "Above 100 cycles"],
    },
}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    inputs = [
        "extra-trees-importancia-permutacao-fd001.csv",
        "extra-trees-alertas-resumo-fd001.csv",
        "extra-trees-alertas-custos-fd001.csv",
        "extra-trees-incerteza-faixas-fd001.csv",
        "extra-trees-incerteza-resumo-fd001.csv",
        "decisao-final-extra-trees-fd001.md",
    ]
    for name in inputs:
        shutil.copyfile(REPORTS / name, ASSETS / name)
    benchmark = pd.DataFrame({
        "model": ["Extra Trees", "XGBoost", "HistGradientBoosting"],
        "MAE": [16.23, 17.29, 17.77], "RMSE": [23.80, 25.15, 27.01],
        "signed": [10.08, 9.31, 8.83]})
    benchmark.to_csv(ASSETS / "benchmark.csv", index=False)
    importance = pd.read_csv(REPORTS / inputs[0]).head(10).sort_values("importance_mean")
    alerts = pd.read_csv(REPORTS / inputs[1])
    costs = pd.read_csv(REPORTS / inputs[2])
    uncertainty = pd.read_csv(REPORTS / inputs[3])
    for key, t in TEXT.items():
        notebook = ROOT / "notebooks" / ("pt-BR/27_conclusao_entrega_fd001.ipynb" if key == "pt" else "en-US/27_conclusion_delivery_fd001.ipynb")
        summary_name = f"summary-{key}.ipynb"
        shutil.copyfile(notebook, ASSETS / summary_name)
        shutil.copyfile(ROOT / ("README.md" if key == "pt" else "README.en.md"), ASSETS / f"README-{key}.md")
        charts = []
        fig = px.bar(benchmark.melt(id_vars="model", value_vars=["MAE", "RMSE"], var_name="metric", value_name="value"),
                     x="model", y="value", color="metric", barmode="group",
                     color_discrete_map={"MAE": "#2563eb", "RMSE": "#64748b"},
                     labels={"model": t["model"], "value": t["cycles"], "metric": "Metric" if key == "en" else "Métrica"})
        charts.append(fig)
        charts.append(px.bar(importance, x="importance_mean", y="feature", orientation="h",
                             error_x="importance_std_between_folds",
                             labels={"importance_mean": t["importance"], "feature": t["feature"],
                                     "importance_std_between_folds": "Fold SD" if key == "en" else "Desvio entre folds"}))
        long_alerts = alerts.melt(id_vars=["limiar", "motores"], value_vars=TEXT["pt"]["statuses"], var_name="status", value_name="count")
        long_alerts["percentage"] = 100 * long_alerts["count"] / long_alerts.motores
        long_alerts["status"] = long_alerts["status"].replace(dict(zip(TEXT["pt"]["statuses"], t["statuses"])))
        charts.append(px.bar(long_alerts, x="limiar", y="percentage", color="status", barmode="stack",
                             category_orders={"status": t["statuses"]},
                             color_discrete_map=dict(zip(t["statuses"], ["#FF7F0E", "#2CA02C", "#D62728", "#7F7F7F"])),
                             labels={"limiar": t["threshold"], "percentage": t["percent"], "status": t["outcome"]}))
        translated_costs = costs.copy()
        translated_costs["cenario"] = translated_costs.cenario.replace(dict(zip(TEXT["pt"]["scenarios"], t["scenarios"])))
        charts.append(px.line(translated_costs, x="limiar", y="custo_medio_motor", color="cenario", markers=True,
                              color_discrete_map=dict(zip(t["scenarios"], ["#2CA02C", "#D62728", "#FF7F0E"])),
                              labels={"limiar": t["threshold"], "custo_medio_motor": t["cost"], "cenario": t["scenario"]}))
        bands = uncertainty.copy()
        bands["faixa"] = t["bands"]
        bands["percentage"] = 100 * bands.cobertura
        fig = px.bar(bands, x="faixa", y="percentage",
                     category_orders={"faixa": t["bands"]},
                     labels={"faixa": t["band"], "percentage": t["coverage"]})
        fig.add_hline(y=90, line_dash="dash", annotation_text=t["reference"])
        fig.update_yaxes(range=[0, 108])
        charts.append(fig)
        fragments = []
        for i, fig in enumerate(charts):
            fig.update_layout(template="plotly_white", height=430 if i != 1 else 500,
                              font=dict(family="Arial, sans-serif", size=13, color="#24364b"),
                              margin=dict(l=35, r=25, t=35, b=70), autosize=True,
                              legend=dict(orientation="h", y=1.14, x=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            fragments.append(fig.to_html(full_html=False, include_plotlyjs=False,
                                        div_id=f"chart-{key}-{i}",
                                        config={"responsive": True, "displaylogo": False}))
        fields = ["benchmark", "features", "alerts", "costs", "uncertainty"]
        sections = []
        for i, field in enumerate(fields):
            fallback = benchmark if i == 0 else importance if i == 1 else alerts if i == 2 else costs if i == 3 else uncertainty
            table = fallback.copy()
            if key == "en":
                table = table.rename(columns=dict(zip(TEXT["pt"]["statuses"], t["statuses"])))
                for col, originals, translated in [("cenario", TEXT["pt"]["scenarios"], t["scenarios"]), ("faixa", TEXT["pt"]["bands"], t["bands"])]:
                    if col in table:
                        table[col] = table[col].replace(dict(zip(originals, translated)))
            table = table.rename(columns={"model": t["model"], "signed": "Mean signed error" if key == "en" else "Erro médio assinado", "feature": t["feature"], "importance_mean": t["importance"], "importance_std_between_folds": "Fold SD" if key == "en" else "Desvio entre folds", "repeats_std_mean": "Repeat SD" if key == "en" else "Desvio entre repetições", "familia": "Family" if key == "en" else "Família", "tipo": "Type" if key == "en" else "Tipo", "janela": "Window" if key == "en" else "Janela", "selecionada_optuna": "Selected" if key == "en" else "Selecionado", "limiar": t["threshold"], "motores": "Engines" if key == "en" else "Motores", "cenario": t["scenario"], "custo_medio_motor": t["cost"], "faixa": t["band"], "cobertura": "Coverage (fraction)" if key == "en" else "Cobertura (fração)", "largura_media": "Mean width (cycles)" if key == "en" else "Largura média (ciclos)"})
            if i == 1:
                table = table[[t["feature"], t["importance"], "Fold SD" if key == "en" else "Desvio entre folds"]].iloc[::-1]
            table_label = "View numerical results" if key == "en" else "Ver resultados numéricos"
            data_table = '<details><summary>' + table_label + '</summary><div class="table-scroll">' + table.to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0) + '</div></details>'
            sections.append(f'<section id="{field}"><span class="section-no">0{i+1}</span><h2>{html.escape(t["names"][i])}</h2><p>{html.escape(t[field])}</p>{fragments[i]}{data_table}<p class="insight">{html.escape(t[field+"_note"])}</p></section>')
        navigation = '<a href="#dataset">' + html.escape(t["dataset_heading"]) + '</a> ' + " ".join(f'<a href="#{field}">{html.escape(t["names"][i])}</a>' for i, field in enumerate(fields+["reproduction"]))
        metric = "16,23" if key == "pt" else "16.23"
        source_links = " · ".join(f'<a href="assets/{name}">{html.escape(name)}</a>' for name in inputs[:5])
        document = f"""<!doctype html>
<html lang="{t['lang']}">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>C-MAPSS FD001 — {html.escape(t['title'])}</title>
<meta name="description" content="{html.escape(t['intro'], quote=True)}">
<link rel="alternate" hreflang="pt-BR" href="index.html"><link rel="alternate" hreflang="en" href="en.html">
<link rel="stylesheet" href="assets/report.css"><script src="assets/plotly.min.js"></script></head>
<body id="top"><a class="skip" href="#benchmark">{html.escape(t['contents'])}</a>
<header><div class="brand">C-MAPSS <span>FD001</span></div><a class="language" href="{t['other_file']}" lang="{'en' if key=='pt' else 'pt-BR'}">{t['other']} ↗</a></header>
<main><div class="hero"><p class="eyebrow">{html.escape(t['tag'])}</p><h1>{html.escape(t['title'])}</h1><p class="lead">{html.escape(t['intro'])}</p><p class="scope">{html.escape(t['scope'])}</p>
<div class="stats"><div><strong>{metric}</strong><span>{t['metric']} · {t['cycles'].lower()}</span></div><div><strong>93 / 205</strong><span>{t['selected']}</span></div><div><strong>100</strong><span>{t['engines']}</span></div></div></div>
<nav aria-label="{t['contents']}">{navigation}</nav>
<section id="dataset"><h2>{t['dataset_heading']}</h2><p>{html.escape(t['dataset_origin'])}</p><p>{html.escape(t['dataset_context'])}</p><p><a href="{DATASET_REPOSITORY}">{t['repository_label']}</a> · <a href="{DATASET_CATALOG}">{t['catalog_label']}</a></p><p class="insight">Saxena, A.; Goebel, K. (2008). <cite>Turbofan Engine Degradation Simulation Data Set</cite>. NASA Ames, Prognostics Data Repository.</p></section>
<noscript><p>JavaScript {'é necessário para os gráficos interativos. Os dados estão disponíveis abaixo.' if key=='pt' else 'is required for interactive charts. Data are available below.'}</p></noscript>
{''.join(sections)}
<section id="reproduction"><span class="section-no">06</span><h2>{t['names'][5]}</h2><p>{t['reproduction']}</p>
<pre><code>python scripts/predict_final_extra_trees_fd001.py --input history.csv --output predictions.csv</code></pre>
<div class="downloads"><a href="assets/{summary_name}" download>{t['download']}</a><a href="assets/README-{key}.md" download>{t['readme']}</a></div><p class="insight">{t['limits']}</p></section>
<section class="sources"><h2>{t['evidence']}</h2><p>{t['source']}</p><details><summary>{t['data']}</summary><p class="file-links">{source_links}</p><p><a href="assets/benchmark.csv">benchmark.csv</a> · <a href="assets/build-manifest.json">build-manifest.json</a> · <a href="assets/decisao-final-extra-trees-fd001.md">FD001 benchmark record (.md, PT-BR)</a></p></details></section>
</main><footer>NASA C-MAPSS · FD001 <a href="#top">{t['back']} ↑</a></footer></body></html>"""
        (SITE / ("index.html" if key == "pt" else "en.html")).write_text(document, encoding="utf-8")
    manifest = {"sources": {str((REPORTS/name).relative_to(ROOT)): hashlib.sha256((REPORTS/name).read_bytes()).hexdigest() for name in inputs},
                "benchmark": "Frozen values from decisao-final-extra-trees-fd001.md; no new evaluation.",
                "languages": ["pt-BR", "en-US"], "training_performed": False,
                "dataset_provenance": {"repository": DATASET_REPOSITORY, "catalog": DATASET_CATALOG,
                                       "citation": "Saxena and Goebel (2008), Turbofan Engine Degradation Simulation Data Set",
                                       "local_readme_sha256": hashlib.sha256((ROOT / "data/raw/dataset/readme.txt").read_bytes()).hexdigest()}}
    (ASSETS / "build-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Relatório bilíngue gerado: site/index.html e site/en.html")


if __name__ == "__main__":
    main()
