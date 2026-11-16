# Sequência de trabalho do projeto

Esta sequência organiza o projeto por gates de evidência. A etapa seguinte só será iniciada quando a anterior produzir evidência suficiente ou registrar explicitamente uma limitação.

> **Revisão de 2026-11-10** (ver `docs/REVISAO-TECNICA-ESCOPO-E-GAPS.md`): as Fases 7 e 8 foram executadas antes do fechamento das Fases 2, 5 e 6, e o período de teste foi consultado nos notebooks 06 e 07. Os resultados dos notebooks 07 e 08 ficam reclassificados como **exploratórios e não comparáveis** até que os gates abaixo sejam cumpridos. A ordem das fases foi ajustada à execução real (EDA antes da tabela analítica) e a Fase 5 passou a ser pré-requisito explícito da competição preditiva.

## Pergunta de pesquisa e reenquadramento dos alvos

É possível identificar, **no momento da compra**, pedidos com maior risco de violar o prazo prometido, usando características do pedido, a relação vendedor–destino e o histórico observado da rede?

- **Alvo principal:** `sla_breach` (classificação de risco). A violação é avaliada por **dia de calendário**, porque `order_estimated_delivery_date` não tem componente de hora: há violação quando a data de entrega ao cliente é posterior à data estimada. O momento da previsão é **a compra**.
- **Alvo secundário:** prazo realizado (`actual_lead_days`) por regressão quantílica (P50/P90), comparado à promessa.
- **Análise descritiva/inferencial:** `delivery_delay_days` contínuo, usado no OLS e na EDA.

## Status por fase

| Fase | Estado | Pendência que bloqueia o gate |
|---|---|---|
| 1 — Fonte e pergunta | Concluída | — |
| 2 — Auditoria e contratos | Concluída (2026-11-16) | — (auditoria reproduzível em `audit.py`; perguntas do anexo com decisão registrada, as últimas fechadas pelo P-07) |
| 3 — EDA do atraso e do SLA | Concluída (descritiva) | ICs para taxas por grupo; números dos insights gerados por células |
| 4 — Tabela analítica | Concluída (2026-11-14) | — (catálogo temporal, features logísticas e de calendário, tabela de scoring com a rede) |
| 5 — Rede e histórico | Concluída (2026-11-14) | — (features as-of com prior global e cold start, `network.py`) |
| 6 — OLS inferencial | Concluída (2026-11-14) | — (HC3, cluster, VIF, resíduos e influência em `inferential.py`, notebook 06) |
| 7 — Competição preditiva | Concluída (2026-11-14) | — (CatBoost candidato pelo P-01/P-04; regressão sem candidato) |
| 8 — Optuna e avaliação final | Concluída (2026-11-14) | — (F1 inconclusivo no notebook 09; F2 pós-hoc exploratório no notebook 10; ver `REGISTRO-DE-DECISOES.md`) |
| 9 — Relatórios e publicação | Concluída (2026-11-16) | — (notebook 11 e relatório executivo em `site/`; publicação em repositório remoto após a revisão do responsável) |

## Fase 1 — Apresentação da fonte e pergunta de pesquisa

Objetivo: apresentar o Olist, suas tabelas, relações, população observável, alvo e limites de interpretação.

Entrega: notebook `01_apresentacao_e_pre_analise_olist.ipynb` (PT-BR e EN-US).

Gate: a fonte deve permitir uma tabela de pedidos com alvo válido e atributos disponíveis antes da entrega.

## Fase 2 — Auditoria e contratos

Documentos de trabalho: `docs/CONTRATO-DADOS-FASE-2.md` e `docs/REGISTRO-DE-DECISOES.md` (uma linha por pergunta do anexo).

Serão definidos:

- população supervisionada e população de scoring;
- regra do alvo binário por dia de calendário e regra do alvo contínuo;
- tratamento de cancelados, pedidos sem entrega e censura à direita no fim da série;
- ponto de decisão aprovado: compra;
- granularidade por pedido e política para múltiplos vendedores;
- campos permitidos e proibidos no ponto de decisão;
- tipo, contagem e tratamento de cada anomalia temporal, com função reproduzível em `src/`;
- regras de integridade e reprodutibilidade.

Gate: contratos testáveis, anomalias classificadas sem correções silenciosas e perguntas 1–17 do anexo com decisão registrada.

## Fase 3 — EDA do atraso e do SLA

Notebooks `02_eda_atraso_sla_olist.ipynb` e `03_eda_geografia_e_composicao_pedido.ipynb`.

Serão examinados distribuição do atraso, sazonalidade, estados, distância, composição do pedido, valores ausentes, outliers e diferenças entre pedidos entregues, cancelados e não concluídos. As taxas por grupo devem trazer intervalo de confiança ou volume mínimo explícito. Todo número citado em insight deve ser produzido por uma célula.

Gate: o alvo, sua escala e suas limitações serão considerados adequados para classificação de risco, regressão do prazo realizado e análise inferencial do atraso.

## Fase 4 — Tabela analítica de pedidos

Notebook `04_tabela_analitica_candidata.ipynb`, implementação em `src/ecom_delivery_sla/features.py`.

A tabela tem uma linha por pedido, com agregação explícita de itens, pagamentos, produtos, vendedores e localização. Nenhuma variável derivada da entrega real é incorporada como preditor. Devem ser incluídas features logísticas (peso, volume, prazo de despacho permitido, UF do vendedor, mesma UF) e de calendário, e removidas as redundantes (`payment_value_sum` é colinear com preço + frete). Haverá uma função para a população de scoring, separada da população supervisionada.

O split temporal (`src/ecom_delivery_sla/temporal.py`, notebook `05_contrato_temporal_e_split.ipynb`) passa a ser usado como **validação multi-janela** (expanding window com folds mensais e embargo para features históricas). A última janela fica reservada como teste final.

Gate: chaves, cardinalidades, cobertura e isolamento temporal verificados; nenhuma feature com status pendente na matriz de modelagem.

## Fase 5 — Rede e variáveis históricas

Granularidade de partida: grafo bipartido dirigido vendedor → UF de destino (vendedor → CEP de 3 dígitos como extensão). Métricas de rede e de confiabilidade são calculadas **as-of**: para cada pedido, só entram pedidos cujo desfecho já era conhecido antes da compra atual (entrega ao cliente, ou despacho ao transportador, anterior ao `order_purchase_timestamp`). As taxas usam suavização bayesiana empírica e há indicadores explícitos de cold start.

Gate: teste automatizado que falha se qualquer métrica usar o próprio pedido, informação futura ou desfecho ainda não conhecido no instante da compra.

## Fase 6 — OLS inferencial

Notebook `06_modeling_baseline_ols_sklearn.ipynb`. O OLS é usado para associação e interpretação de coeficientes, com categoria de referência explícita, erros-padrão HC3 e agrupados, VIF, diagnóstico de resíduos e de influência e análise de sensibilidade às caudas. A logística complementa com odds ratios e intervalos. Não serão feitas afirmações causais sem desenho causal apropriado. Métricas de teste não são exibidas nesta fase.

Gate: especificação, erros-padrão, diagnósticos e limitações documentados antes de qualquer escolha de modelo.

## Fase 7 — Competição preditiva

Notebook `07_tournament_estimators.ipynb`. Participam:

- **Baselines obrigatórias:** regra ingênua de prazo e score pela promessa.
- **Modelos:** OLS/Ridge/ElasticNet, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM e CatBoost.

Todos usam o mesmo contrato de features, os mesmos folds temporais, a mesma política de desbalanceamento e as mesmas métricas:

- **Classificação:** AP, ROC AUC, Brier, ECE e precisão/recall@k.
- **Regressão quantílica:** MAE e pinball loss.

O teste não é avaliado nesta fase.

Gate: o vencedor é escolhido por ganho sobre a melhor baseline, estabilidade entre folds, calibração, interpretabilidade, complexidade, latência e conformidade.

## Fase 8 — Optuna e avaliação final

O notebook `08_otimizacao_optuna.ipynb` executa estudos persistidos em SQLite, com objetivo igual à média dos folds temporais e orçamento comparável por estimator. O espaço de busca inclui o pré-processamento relevante para cada família, o seletor de atributos e os hiperparâmetros.

A configuração escolhida é refitada em treino + validação e avaliada **uma única vez** no teste final, num notebook separado (`09_avaliacao_final`), com intervalo de confiança por bootstrap em blocos temporais e diagnóstico de drift. Como o período jul–ago/2018 já foi consultado nas versões anteriores, o relatório final deve declarar esse fato ou usar uma janela final nova.

Gate: resultados reproduzíveis e artefatos aprovados nos gates de governança antes da avaliação única no teste.

## Fase 9 — Relatórios e publicação

Serão preparados notebooks PT-BR e EN-US executados de ponta a ponta no mesmo ambiente, código modular, testes, contratos, relatório HTML, README, limitações e conclusão sobre o que é ou não apoiado pelos dados.

Até a conclusão dos gates, `production_eligible = false`.
