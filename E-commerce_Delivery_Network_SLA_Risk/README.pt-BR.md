# Rede de entrega e risco de SLA no e-commerce (Olist)

Este projeto investiga o atraso de entrega e o risco de violação do prazo prometido (SLA) em pedidos do marketplace brasileiro Olist. O código reutilizável fica no pacote ecom_delivery_sla, os gráficos dos notebooks 03 (rede de rotas), 05, 09 e 10 ficam no pacote de apresentação ecom_delivery_sla_viz, e a análise é conduzida em notebooks bilíngues (português e inglês).

O estado do projeto é **exploratório**: production_eligible = false. Nenhum modelo foi considerado elegível para uso operacional.

## Objetivo e pergunta de pesquisa

A pergunta central é se, no momento da compra, o risco de um pedido ser entregue depois da data prometida pode ser previsto com ganho material sobre baselines simples, usando apenas informações disponíveis naquele momento (características do pedido, distância, calendário e histórico da rede vendedor → UF de destino). O alvo binário é definido por dia de calendário, e o atraso contínuo, em dias fracionários, é tratado de forma descritiva e inferencial. A associação estimada pelo OLS não é interpretada como causalidade.

## Dados

A fonte é o conjunto público *Brazilian E-Commerce Public Dataset by Olist*, publicado no Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Os dados brutos não são versionados, e o download é manual, porque o Kaggle exige credenciais. O arquivo brazilian-ecommerce.zip deve ser salvo em data/raw/olist/, e a proveniência e o SHA-256 esperado estão registrados em data/raw/olist/README.md.

A integridade do ZIP é conferida, e os nove CSVs são extraídos para data/raw/olist/csv/ com o script abaixo:

~~~
python scripts/verify_data.py
python scripts/verify_data.py --extract
~~~

## Como reproduzir

O ambiente é criado com conda a partir de environment.yml, que também instala o projeto em modo editável:

~~~
conda env create -f environment.yml
conda activate ecom-delivery-sla
python scripts/verify_data.py --extract
~~~

Os notebooks são executados na ordem 01 → 11. A versão em português calcula e grava os resultados caros, e a versão em inglês carrega esses caches nos notebooks correspondentes. A avaliação final (F1) abre o período de teste uma única vez, e o notebook 08 reaproveita os estudos Optuna gravados em SQLite.

O CI instala .[dev], executa lint, testes de contrato, paridade PT/EN e exportação do relatório final. Ele não executa os notebooks nem gera o site, porque os CSVs brutos não são versionados.

~~~
pytest
ruff check src tests scripts
~~~

## Mapa dos notebooks

| Etapa | PT-BR | EN-US | O que apresenta |
|---|---|---|---|
| 01 — Apresentação do projeto e da fonte | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/01_apresentacao_e_pre_analise_olist.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/01_apresentacao_e_pre_analise_olist.ipynb) | Pergunta de pesquisa, dados e escopo |
| 02 — EDA de atraso e SLA | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/02_eda_atraso_sla_olist.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/02_eda_atraso_sla_olist.ipynb) | Distribuição dos atrasos e violações do prazo |
| 03 — Geografia, rotas e composição do pedido | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/03_eda_geografia_e_composicao_pedido.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/03_eda_geography_and_order_composition.ipynb) | Distância, rede de rotas e composição dos pedidos |
| 04 — Tabela analítica candidata | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/04_tabela_analitica_candidata.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/04_candidate_analytical_table.ipynb) | Features por pedido e cobertura dos dados |
| 05 — Contrato temporal e divisão cronológica | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/05_contrato_temporal_e_split.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/05_temporal_contract_and_split.ipynb) | Disponibilidade na compra e validação temporal |
| 06 — OLS e modelos de referência | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/06_modeling_baseline_ols_sklearn.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/06_modeling_baseline_ols_sklearn.ipynb) | Diagnósticos estatísticos e comparações iniciais |
| 07 — Torneio de estimadores | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/07_tournament_estimators.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/07_estimator_tournament.ipynb) | Comparação dos modelos nas janelas de desenvolvimento |
| 08 — Otimização com Optuna | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/08_otimizacao_optuna.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/08_optuna_optimization.ipynb) | Busca de hiperparâmetros nos dados de desenvolvimento |
| 09 — Avaliação final oficial | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/09_avaliacao_final.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/09_final_evaluation.ipynb) | Teste pré-registrado e decisão |
| 10 — Análise pós-hoc F2 | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/10_pos_hoc_f2.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/10_post_hoc_f2.ipynb) | Cenários exploratórios de atualização após F1 |
| 11 — Relatório final | [PT-BR](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/pt_br/11_relatorio_final.ipynb) | [EN-US](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/E-commerce_Delivery_Network_SLA_Risk/notebooks/en_us/11_final_report.ipynb) | Síntese das evidências e limitações |

O notebook 11 é o relatório final. Ele lê apenas resultados já gravados, sem treinar modelos nem reabrir o período de teste, e é exportado para HTML em reports/.

## Relatório executivo

O relatório executivo bilíngue, em páginas estáticas com gráficos interativos, está em site/index.html (português) e site/en.html (inglês). Ele pode ser reconstruído com:

~~~
python scripts/build_site.py
~~~

## Resultados

O torneio de desenvolvimento usa nove folds temporais com embargo de 30 dias. Na classificação, CatBoost foi o único candidato segundo o gate pré-registrado, com AP média de 0,175 contra 0,154 da melhor baseline. Na regressão, nenhum modelo foi candidato.

A avaliação final oficial foi **inconclusiva**: CatBoost obteve AP 0,082 contra 0,084 da baseline logística, com diferença pareada de −0,002 e IC 95% de −0,032 a +0,064. A análise complementar F2 é exploratória e nenhuma variante superou as referências.

## Limitações

O conjunto Olist cobre um único marketplace e um período curto, terminando em 2018. A censura à direita foi tratada pela exclusão dos pedidos sem desfecho maduro. O período de teste está esgotado para novas avaliações, e a regressão do atraso permanece descritiva e inferencial, sem candidato. Por isso, production_eligible = false permanece.

## Documentação

Os documentos internos de trabalho permanecem no projeto-fonte e não fazem parte desta cópia pública do portfólio. A reconstrução do site a partir da fonte exige esses documentos locais.
