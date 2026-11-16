# Arquitetura modular do projeto

## Objetivo

Manter a lógica analítica em `src/` e usar os notebooks apenas como interfaces narrativas e executáveis. Os notebooks PT-BR e EN-US devem importar as mesmas funções e variar somente a narrativa e o idioma dos rótulos.

## Estrutura aprovada

```text
src/ecom_delivery_sla/
  io.py          # leitura controlada das tabelas
  targets.py     # datas, alvos e população supervisionada
  labels.py      # rótulos PT-BR e EN-US
  eda.py         # tabelas e gráficos reutilizáveis
  geo.py         # geolocalização, Haversine e composição
  maps.py        # mapas demonstrativos sem rota histórica
  features.py    # tabela candidata, catálogo e features de rede no scoring
  temporal.py    # cortes, folds, resumo de folds e sensibilidade de censura
  modeling.py    # pré-processador compartilhado, baselines, métricas, torneios e resumo
  optimization.py# estudos Optuna persistidos (usa o pré-processador de modeling.py)
  audit.py       # auditoria reproduzível de anomalias
  network.py     # histórico vendedor–destino as-of, priors globais e agregação por pedido
  drift.py       # diagnóstico de drift populacional
  inferential.py # OLS/logit inferencial e diagnósticos
  run_cache.py   # cache de execução (fingerprint + pickle/CSV) para treinos pesados dos notebooks; load_run_unchecked para execuções congeladas
  final_evaluation.py # avaliação final única (notebook 09): janelas fixas, bootstrap por blocos semanais, leitura pré-registrada, uso único e ledger
  posthoc.py     # avaliação F2 pós-hoc (notebook 10): rótulos conhecidos, variantes A–E, gate de desenvolvimento, blocos diários e congelamento compartilhado com o F1

src/ecom_delivery_sla_viz/   # pacote de apresentação; não importa ecom_delivery_sla e fica fora do fingerprint de run_cache
  charts.py      # gráficos dos notebooks 05, 07, 09 e 10 (folds, PSI, métricas por fold, ganho, intervalos, regime da promessa, rótulos)
  routes.py      # grafo de rotas origem–destino do notebook 03 (pares pedido–vendedor, IC de Wilson, mapa, matriz e rotas entre cidades)
  report.py      # tabelas e gráficos do relatório final (notebook 11)
  labels.py      # rótulos PT-BR e EN-US dos gráficos

scripts/
  verify_data.py   # confere o SHA-256 do ZIP da Olist e extrai os CSVs
  export_report.py # exporta o notebook 11 para HTML em reports/
  build_site.py    # gera o relatório executivo estático em site/ a partir dos resultados gravados
  audit_unknown_categories.py # conta categorias ausentes do treino por fold e no F1 (T-11)
  vendor/          # contorno do mapa (topojson) embutido no site

site/            # relatório executivo bilíngue (index.html, en.html, assets/)
reports/         # HTML exportado do notebook 11

notebooks/
  pt_br/
    01_apresentacao_e_pre_analise_olist.ipynb
    02_eda_atraso_sla_olist.ipynb
    03_eda_geografia_e_composicao_pedido.ipynb
    04_tabela_analitica_candidata.ipynb
    05_contrato_temporal_e_split.ipynb
    06_modeling_baseline_ols_sklearn.ipynb
    07_tournament_estimators.ipynb
    08_otimizacao_optuna.ipynb
    09_avaliacao_final.ipynb
    10_pos_hoc_f2.ipynb
    11_relatorio_final.ipynb
  en_us/
    01_apresentacao_e_pre_analise_olist.ipynb
    02_eda_atraso_sla_olist.ipynb
    03_eda_geography_and_order_composition.ipynb
    04_candidate_analytical_table.ipynb
    05_temporal_contract_and_split.ipynb
    06_modeling_baseline_ols_sklearn.ipynb
    07_estimator_tournament.ipynb
    08_optuna_optimization.ipynb
    09_final_evaluation.ipynb
    10_post_hoc_f2.ipynb
    11_final_report.ipynb

tests/
  test_targets.py
  test_eda.py
  test_features.py
  test_audit.py
  test_network.py
  test_temporal.py
  test_modeling.py
  test_inferential.py
  test_drift.py
  test_run_cache.py
  test_final_evaluation.py
  test_posthoc.py
  test_notebook_contract.py
  test_verify_data.py
  test_export_report.py
  test_build_site.py
  test_viz.py
  test_viz_routes.py
  test_viz_report.py
  test_audit_unknown_categories.py

pyproject.toml
environment.yml
.github/workflows/ci.yml
```

## Regras de implementação

1. A lógica de negócio não deve ser duplicada nas células dos notebooks.
2. Funções públicas em `src` devem ter docstrings compatíveis com PEP 257 e estilo PEP 8.
3. Os testes unitários devem usar dados sintéticos pequenos e não depender dos CSVs locais.
4. O CI deve validar lint, testes e o contrato estrutural dos notebooks.
5. A execução completa dos notebooks com dados reais é uma validação local/reprodutível separada do CI.
6. O código não deve corrigir anomalias temporais sem registrar a regra no contrato de dados.
7. A linguagem deve ser controlada por parâmetro, não por cópia de funções entre notebooks.

## Padrão narrativo Markdown-Python-Insights

Cada bloco analítico do notebook deve seguir a sequência:

1. **Markdown de abertura:** explica a pergunta, a métrica, a tabela ou o gráfico que será apresentado e registra as cautelas necessárias.
2. **Python:** chama as funções reutilizáveis de `src/` e exibe a tabela ou visualização, sem duplicar a lógica de negócio.
3. **Markdown de insight:** narra os achados observados, preferencialmente com valores concretos, e separa descrição, limitação e decisão para a próxima etapa.

O insight não deve inventar explicações causais. Quando uma conclusão depender de uma heurística, limiar mínimo, anomalia ou variável pós-evento, essa condição deve aparecer no texto. Os notebooks bilíngues mantêm a mesma sequência e as mesmas células Python; somente a narrativa e o idioma dos rótulos variam.

## Cache de execução dos notebooks pesados

`run_cache.py` (`run_fingerprint`, `cached_run`, `describe_run`) evita treinar duas vezes o mesmo resultado nos notebooks PT-BR e EN-US. O notebook 07 usa `RUN_VERSION`, `RUN_MODE` (`"auto"`, `"train"` ou `"load"`) e `CACHE_ROOT = outputs/runs`; os resultados ficam em `outputs/runs/<nome>/<versão>/` (pickle, CSV e `metadata.json`). Convenção de uso: executar o notebook 07 em um idioma (treina e salva) e depois no outro (carrega o mesmo resultado, com a procedência impressa por `describe_run`). Mudança de versão, configuração, dados ou código do pacote invalida o cache e provoca novo treino; o idioma não entra no fingerprint. Os modelos treinados não são persistidos, apenas as tabelas de resultados. O notebook 08 já reaproveita estudos Optuna pelo `STUDY_VERSION` no SQLite (`outputs/optuna/`) e não usa este cache.

## Avaliação final única (notebook 09)

`final_evaluation.py` implementa o pré-registro P-05 (`docs/PLANO-NOTEBOOK-09.md`). `split_final_windows` fixa o treino (`< 2018-06-01`) e o teste (`[2018-07-01, 2018-09-01)`); `weekly_block_bootstrap`, `paired_metric_difference`, `classify_preregistered_outcome` e `reliability_table` produzem a incerteza e a leitura; `run_final_evaluation(modeling_frame, allow_test=True, version="F1", ...)` é a única chamada do projeto que abre o teste. Os notebooks `09_avaliacao_final` (PT) e `09_final_evaluation` (EN) apenas carregam os dados como o notebook 07, chamam essa função e exibem as tabelas.

Os resultados ficam em `outputs/runs/final_evaluation/<versão>/` com semântica congelada (T-09): se o diretório da versão existe, o resultado é sempre carregado (aviso em caso de divergência de fingerprint) e nunca recalculado, de modo que o segundo idioma apenas carrega o resultado do primeiro. Cada cálculo acrescenta uma linha ao ledger append-only `outputs/runs/final_evaluation/LEDGER.jsonl` (versão, timestamp, fingerprint, tamanhos, `post_hoc`). Versões diferentes de `F1` são pós-hoc. O modelo final é salvo com joblib em `outputs/models/catboost_final/<versão>/model.joblib` com `metadata.json`. O notebook 09 foi executado com os CSVs reais em 2026-11-14 (versão F1, teste aberto uma vez); os testes usam apenas dados sintéticos.

## Avaliação F2 pós-hoc (notebook 10)

`posthoc.py` implementa o P-06 (`docs/PLANO-F2-POS-HOC.md`; decisão técnica T-10). A avaliação é **pós-hoc e exploratória**, não substitui o F1 e mantém `production_eligible = false`. `known_label_mask` define os rótulos conhecidos em uma data (entrega estritamente anterior); `recency_weights`, `recalibration_split` e `promise_vs_recent_median` implementam os mecanismos das variantes C, D e E; `run_development_gate` aplica o gate de não inferioridade nos folds; `run_posthoc_evaluation(modeling_frame, supervised_orders=..., allow_test=True, version="F2", f1_model_path=...)` roda primeiro o gate e depois pontua agosto/2018 só com as variantes aprovadas, com bootstrap por blocos diários. O congelamento, o ledger e o fingerprint são compartilhados com o F1 por `final_evaluation._run_frozen`. Os notebooks `10_pos_hoc_f2` (PT) e `10_post_hoc_f2` (EN) carregam os dados como o notebook 09, fazem uma única chamada e exibem as tabelas; devem ser executados primeiro em PT (o EN apenas carrega o resultado congelado). O notebook 10 foi executado com os CSVs reais em 2026-11-14 (PT calculou, EN carregou); os testes (`tests/test_posthoc.py`) usam apenas dados sintéticos.

## Estado desta migração

Estado em 2026-11-16. A versão de 2026-11-11 desta seção descrevia notebooks ainda não executados e foi substituída.

A lógica analítica está concentrada em `ecom_delivery_sla`, e os notebooks PT-BR e EN-US 01 a 11 contêm apenas narrativa, configuração de caminho e chamadas aos pacotes. Todos foram executados pelo responsável com os CSVs da Olist, e o teste de saídas (`REQUIRE_NOTEBOOK_OUTPUTS=1`) passa. O notebook 07 foi reexecutado na Fase 9 e gerou o cache `outputs/runs/tournament/V1/` com o código atual. A rede as-of, os folds com embargo, as baselines obrigatórias, a política única de desbalanceamento e o Optuna por média de folds foram validados nos dados reais. O teste foi aberto uma única vez (F1, notebook 09), e a análise pós-hoc F2 (notebook 10) o reutilizou de forma declarada. Os resultados estão em `REGISTRO-DE-DECISOES.md`.

A Fase 9 (`docs/PLANO-FASE-9.md`, P-07, P-08 e P-09) acrescentou gráficos aos notebooks 03, 05, 07, 09 e 10, o grafo de rotas do notebook 03, citações e referências em todos os notebooks com definições, o notebook 11 de relatório, o relatório executivo em `site/`, a arrumação do repositório e o `README.md`. As funções de gráfico ficam no pacote de apresentação separado, `ecom_delivery_sla_viz`, porque qualquer alteração em `src/ecom_delivery_sla/` muda o fingerprint de `run_cache` e invalidaria o cache do torneio. Pelo mesmo motivo, o `.gitattributes` fixa os finais de linha em LF. O teste de saídas de `tests/test_notebook_contract.py` é pulado por padrão e ativado com `REQUIRE_NOTEBOOK_OUTPUTS=1`. `production_eligible = false` permanece.
