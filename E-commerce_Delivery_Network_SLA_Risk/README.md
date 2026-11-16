# Rede de entrega e risco de SLA no e-commerce (Olist)

Este projeto investiga o atraso de entrega e o risco de violação do prazo prometido (SLA) em pedidos do marketplace brasileiro Olist. O código reutilizável fica no pacote `ecom_delivery_sla`, os gráficos dos notebooks 03 (rede de rotas), 05, 09 e 10 ficam no pacote de apresentação `ecom_delivery_sla_viz`, e a análise é conduzida em notebooks bilíngues (português e inglês). A versão em inglês deste documento está na seção [English summary](#english-summary), ao final.

O estado do projeto é **exploratório**: `production_eligible = false`. Nenhum modelo foi considerado elegível para uso operacional.

## Objetivo e pergunta de pesquisa

A pergunta central é se, no momento da compra, o risco de um pedido ser entregue depois da data prometida pode ser previsto com ganho material sobre baselines simples, usando apenas informações disponíveis naquele momento (características do pedido, distância, calendário e histórico da rede vendedor → UF de destino). O alvo binário é definido por dia de calendário, e o atraso contínuo, em dias fracionários, é tratado de forma descritiva e inferencial. A associação estimada pelo OLS não é interpretada como causalidade.

## Dados

A fonte é o conjunto público *Brazilian E-Commerce Public Dataset by Olist*, publicado no Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Os dados brutos não são versionados, e o download é manual, porque o Kaggle exige credenciais. O arquivo `brazilian-ecommerce.zip` deve ser salvo em `data/raw/olist/`, e a proveniência e o SHA-256 esperado estão registrados em `data/raw/olist/README.md`.

A integridade do ZIP é conferida, e os nove CSVs são extraídos para `data/raw/olist/csv/` (a pasta lida pelos notebooks), com o script abaixo. O script usa apenas a biblioteca padrão e termina com código diferente de zero se o ZIP estiver ausente ou se o hash divergir.

```bash
python scripts/verify_data.py             # confere o SHA-256 do ZIP
python scripts/verify_data.py --extract   # confere e extrai os 9 CSVs para data/raw/olist/csv/
```

## Como reproduzir

O ambiente é criado com conda a partir de `environment.yml`, que também instala o projeto em modo editável (`pip install -e .[dev]`), de modo que `import ecom_delivery_sla` funciona nos notebooks e nos testes. No portfólio, a suíte do projeto é executada pelo workflow CI da raiz.

```bash
conda env create -f environment.yml
conda activate ecom-delivery-sla
python scripts/verify_data.py --extract
```

Os notebooks são executados na ordem 01 → 11. A versão em português é executada primeiro: ela calcula e grava os resultados caros (torneio do notebook 07, avaliação final do 09 e análise pós-hoc do 10), e a versão em inglês carrega esses caches nos notebooks 07, 09 e 10. A avaliação final (F1) abre o período de teste uma única vez, e o registro dessa abertura fica em `outputs/runs/final_evaluation/LEDGER.jsonl`. O notebook 08 reaproveita os estudos Optuna já gravados em SQLite, em vez de repetir a busca. Os resultados gravados em `outputs/` não são versionados, com exceção dos CSVs, dos `metadata.json` e do `LEDGER.jsonl` de `outputs/runs/final_evaluation/`.

O `environment.yml` define `PYTHONNOUSERSITE=1` para que o ambiente use apenas os próprios pacotes. Sem isso, pacotes instalados na pasta do usuário (no Windows, `AppData\Roaming\Python`) têm precedência e podem quebrar o `pytest`, por exemplo com `ModuleNotFoundError: No module named 'platformdirs.pytest_plugin'`, e o `nbconvert`. Num ambiente criado antes dessa regra, a variável é aplicada com `conda env config vars set PYTHONNOUSERSITE=1 -n ecom-delivery-sla`, seguida de `conda deactivate` e `conda activate ecom-delivery-sla`. Fora do conda, `pip install -e ".[notebooks]"` instala o necessário para executar e exportar os notebooks (Jupyter, `ipykernel`, `nbclient` e `nbconvert`), e `pip install -e ".[dev]"` acrescenta testes e lint. O CI (`.github/workflows/ci.yml`) instala `.[dev]` e roda o lint, os testes com `REQUIRE_NOTEBOOK_OUTPUTS=1` (estrutura, paridade PT/EN, citações e saídas dos notebooks versionados) e a exportação do relatório final, publicada como artefato. Ele não executa os notebooks nem gera o site, porque os CSVs brutos não são versionados. Os testes e a verificação de estilo são executados com os comandos abaixo. O teste de saídas dos notebooks 01 a 11 é pulado por padrão e é ativado com `REQUIRE_NOTEBOOK_OUTPUTS=1`.

```bash
pytest
ruff check src tests scripts
```

## Mapa dos notebooks

| Nº | Notebook em português (`notebooks/pt_br/`) | Notebook em inglês (`notebooks/en_us/`) |
|---:|---|---|
| 01 | `01_apresentacao_e_pre_analise_olist.ipynb` | `01_apresentacao_e_pre_analise_olist.ipynb` |
| 02 | `02_eda_atraso_sla_olist.ipynb` | `02_eda_atraso_sla_olist.ipynb` |
| 03 | `03_eda_geografia_e_composicao_pedido.ipynb` | `03_eda_geography_and_order_composition.ipynb` |
| 04 | `04_tabela_analitica_candidata.ipynb` | `04_candidate_analytical_table.ipynb` |
| 05 | `05_contrato_temporal_e_split.ipynb` | `05_temporal_contract_and_split.ipynb` |
| 06 | `06_modeling_baseline_ols_sklearn.ipynb` | `06_modeling_baseline_ols_sklearn.ipynb` |
| 07 | `07_tournament_estimators.ipynb` | `07_estimator_tournament.ipynb` |
| 08 | `08_otimizacao_optuna.ipynb` | `08_optuna_optimization.ipynb` |
| 09 | `09_avaliacao_final.ipynb` | `09_final_evaluation.ipynb` |
| 10 | `10_pos_hoc_f2.ipynb` | `10_post_hoc_f2.ipynb` |
| 11 | `11_relatorio_final.ipynb` | `11_final_report.ipynb` |

O notebook 11 (português e inglês) é o relatório final. Ele lê apenas resultados já gravados, sem treinar modelos nem reabrir o período de teste, e é exportado para HTML em `reports/` com `python scripts/export_report.py`, dentro do ambiente do projeto.

## Relatório executivo

O relatório executivo bilíngue, em páginas estáticas com gráficos interativos, está em [`site/index.html`](site/index.html) (português) e [`site/en.html`](site/en.html) (inglês). As páginas leem apenas resultados já gravados em `outputs/` e os CSVs brutos do Olist, sem treinar modelos nem reabrir o período de teste, e são geradas com o comando abaixo, dentro do ambiente do projeto. O manifesto `site/assets/build-manifest.json` registra o SHA-256 de cada arquivo de origem.

```bash
python scripts/build_site.py              # grava site/index.html, site/en.html e site/assets/
python scripts/build_site.py --output outro_diretorio
```

## Resultados

Todos os valores desta seção provêm de `docs/REGISTRO-DE-DECISOES.md` (seções "Aprovadas pelo responsável" e "Abertura do teste"), que é a fonte de referência.

### Torneio de desenvolvimento (P-01 e P-04)

O torneio multi-janela usa 9 folds de 2017-10 a 2018-06, com embargo de 30 dias. Um modelo só é considerado candidato se o ganho relativo médio sobre a melhor baseline for de pelo menos 5% em average precision (AP) na classificação, ou de 3% em MAE na regressão, e se a melhor baseline for superada em pelo menos 7 dos 9 folds (P-01). Na classificação, o CatBoost foi o único candidato; na regressão, nenhum modelo foi candidato (P-04).

| Item | Resultado (P-04) |
|---|---|
| CatBoost, AP média | 0,175, contra 0,154 da melhor baseline (`baseline_logistic_promise_plus_state`) |
| Ganho do CatBoost sobre a baseline | +13,9%, com vitória em 8 de 9 folds |
| HistGradientBoosting e XGBoost | +1,7% (7/9) e +0,6% (6/9), abaixo do ganho mínimo |
| Regressão | Nenhum candidato; todos os modelos ficaram abaixo da melhor baseline em MAE (`baseline_state_median_actual_minus_promise`, MAE 5,94) |
| Optuna (`V4_folds`) | Melhor AP de 0,174 (LightGBM), sem superar de forma relevante o CatBoost com parâmetros padrão |

### Avaliação final oficial, F1 (P-05)

A avaliação final foi pré-registrada (P-05) e o teste foi aberto uma única vez em 2026-11-14. O treino usou 77.867 pedidos (compras anteriores a 2018-06-01, prevalência de 7,53%), e o teste, 12.507 pedidos em 10 semanas (601 violações, prevalência de 4,81%). O resultado pelo critério pré-registrado foi **inconclusivo**.

| Quantidade | Valor |
|---|---|
| AP do CatBoost (IC 95%) | 0,082 (0,056 a 0,134) |
| AP da `baseline_logistic_promise_plus_state` (IC 95%) | 0,084 (0,056 a 0,116) |
| Diferença pareada de AP (IC 95%) | −0,002 (−0,032 a +0,064) |

A vantagem de +13,9% observada nos folds de desenvolvimento não se reproduziu no teste. Entre as observações descritivas, a baseline `negative_promised_lead_days` obteve a maior AP no teste (0,142), coerente com o drift da política de promessa (PSI de `promised_lead_days` de 0,77), e o CatBoost superestimou o risco entre 0,1 e 0,5 (ECE global de 0,039). Essas observações não alteram o resultado e não autorizam seleção de modelo.

### Análise complementar pós-hoc, F2 (P-06)

O F2 simula uma implantação com atualização em 2018-08-01, usando apenas rótulos já conhecidos nessa data, e avalia agosto de 2018 (6.351 compras, 393 violações, prevalência de 6,19%). Todos os resultados são exploratórios e marcados como pós-hoc, e nenhuma variante superou as referências.

| Modelo ou referência | AP (IC 95%) |
|---|---|
| A (F1 congelado) | 0,069 (0,054 a 0,084) |
| B (retreino mensal) | 0,068 (0,054 a 0,083) |
| C (pesos por recência) | 0,067 (0,052 a 0,082) |
| E (feature `promise_vs_recent_median`) | 0,069 (0,055 a 0,083) |
| `baseline_logistic_promise_plus_state` | 0,094 (0,071 a 0,121) |
| Regra `negative_promised_lead_days` | 0,179 (0,105 a 0,284) |

No gate de não inferioridade, as variantes C e E foram aprovadas e a D foi reprovada. A diferença pareada de AP contra a logística ficou entre −0,026 e −0,025 em todas as variantes, com IC abaixo de zero. Julho e agosto de 2018 foram consultados no F1 e no F2, e nenhuma nova versão deve ser avaliada nesse período.

## Limitações

O conjunto Olist cobre um único marketplace e um período curto, e termina em 2018, de modo que o resultado não pode ser validado em dados posteriores. A censura à direita no fim da série (pedidos ainda não entregues em julho e agosto de 2018) foi tratada pela exclusão desses pedidos (P-02), e o cenário `include_as_breach` é apenas uma sensibilidade descritiva. O período de teste está esgotado para novas avaliações, e a regressão do atraso permanece como análise descritiva e inferencial, sem candidato. Itens do R-19 (remoção de `sys.path` nos notebooks e `OneHotEncoder(drop=None)`) não foram aplicados, para não alterar o pré-processamento depois da abertura do teste. Por tudo isso, `production_eligible = false` permanece.

O pré-processamento codifica as variáveis categóricas com `OneHotEncoder(drop="first", handle_unknown="ignore")`. Uma categoria ausente do treino vira vetor zero, igual à categoria de referência omitida, e o scikit-learn emite o aviso "Found unknown categories". A auditoria `scripts/audit_unknown_categories.py` mostra que isso só ocorreu com a UF do vendedor, em 26 pedidos de dois folds de desenvolvimento e em nenhum pedido do teste F1. O encoder não foi alterado depois da abertura do teste, e a separação entre categoria desconhecida e categoria de referência fica para um próximo ciclo (decisão T-11 no registro).

## Documentação

A pasta `docs/` reúne a documentação do projeto. Os documentos principais são `REGISTRO-DE-DECISOES.md` (decisões aprovadas e resultados), `PLANO-NOTEBOOK-09.md` (pré-registro da avaliação final), `PLANO-F2-POS-HOC.md` (análise complementar), `PLANO-FASE-9.md` (relatório e repositório), `CONTRATO-DADOS-FASE-2.md` e `DICIONARIO-DADOS-OLIST.md` (dados), `ARQUITETURA-MODULAR.md` (código), `SEQUENCIA-DE-TRABALHO.md` e `REVISAO-TECNICA-ESCOPO-E-GAPS.md` (andamento e revisão).

## English summary

This project studies delivery delay and the risk of breaching the promised delivery date (SLA) for orders of the Brazilian marketplace Olist. The project status is exploratory (`production_eligible = false`).

**Data.** The source is the public Kaggle dataset *Brazilian E-Commerce Public Dataset by Olist*. The download is manual (Kaggle requires a login) and the ZIP must be saved as `data/raw/olist/brazilian-ecommerce.zip`. Its SHA-256, recorded in `data/raw/olist/README.md`, is verified, and the nine CSVs are extracted, with `python scripts/verify_data.py --extract`.

**Reproduction.** The environment is created with `conda env create -f environment.yml`, which also installs the package in editable mode. Notebooks 01 to 11 are run in order, Portuguese first: it computes and stores the expensive results (notebooks 07, 09 and 10), which the English notebooks load from cache. Notebook 08 reuses stored Optuna studies. Tests are run with `pytest` and `ruff check src tests`. Without conda, `pip install -e ".[notebooks]"` installs what is needed to run and export the notebooks, and `.[dev]` adds tests and lint; CI runs lint, the tests with `REQUIRE_NOTEBOOK_OUTPUTS=1` and the report export, but cannot execute the notebooks because the raw CSVs are not versioned. The environment sets `PYTHONNOUSERSITE=1` so that packages in the user folder cannot shadow its own (for an existing environment: `conda env config vars set PYTHONNOUSERSITE=1 -n ecom-delivery-sla`, then reactivate). The notebook map above lists the file names of both languages. The final report (notebook 11) reads stored results only and is exported to HTML in `reports/` with `python scripts/export_report.py`.

**Results** (source: `docs/REGISTRO-DE-DECISOES.md`). In the development tournament (9 folds, 2017-10 to 2018-06, 30-day embargo), CatBoost was the only classification candidate (mean AP 0.175 against 0.154 for the best baseline, +13.9%, winning 8 of 9 folds), and no regression model was a candidate. The official final evaluation (F1), opened once on 2026-11-14, was **inconclusive**: CatBoost AP was 0.082 against 0.084 for the logistic baseline, with a paired difference of −0.002 (95% CI −0.032 to +0.064). The complementary post-hoc analysis (F2, August 2018) is exploratory, and no variant beat the references.

**Executive report.** A bilingual static report is in `site/index.html` (Portuguese) and `site/en.html` (English); it is built from stored results only, with `python scripts/build_site.py`.

**Limitations.** A single marketplace and a short period ending in 2018; right-censored orders excluded; July and August 2018 exhausted for further evaluation; regression without a candidate. Key documents are in `docs/`.
