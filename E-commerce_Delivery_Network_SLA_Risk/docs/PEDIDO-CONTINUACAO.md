# Pedido de continuação — correções pendentes

**Para:** agente executor
**Data:** 2026-11-10
**Referência obrigatória:** `docs/REVISAO-TECNICA-ESCOPO-E-GAPS.md` (IDs `R-xx`) e `docs/REGISTRO-DE-DECISOES.md`

## Contexto

A validação das correções anteriores mostrou o seguinte:

- **Concluídos e aceitos:** R-02 (alvo `sla_breach` por dia de calendário, 6.534 violações = 6,77%) e R-03 (registro de decisões). Não refaça.
- **Parcial:** R-01.
- **Não iniciados:** todos os demais itens.

Execute as etapas abaixo **na ordem**. Ao fim de cada etapa:

1. rode `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` e `ruff check src tests`;
2. registre no `REGISTRO-DE-DECISOES.md` toda decisão nova;
3. não avance com testes ou lint falhando.

## Regras gerais

- Toda lógica fica em `src/ecom_delivery_sla/`. Os notebooks só chamam funções do pacote, e PT-BR e EN-US têm código idêntico (o teste de paridade deve passar).
- Os testes usam dados sintéticos pequenos e não leem os CSVs.
- **Nenhuma métrica do período de teste (jul–ago/2018) pode aparecer em notebooks 01–08.** O teste só será usado no futuro `09_avaliacao_final`, fora deste pedido.
- Não rode buscas Optuna completas. O responsável pelo projeto roda o notebook 08.
- Todo número citado em markdown de insight deve aparecer no output de uma célula.
- Não altere a fonte em `data/raw/olist/csv/`.

---

## Etapa 1 — Fechar P0

### 1.1 CI verde (R-04)
- Adicione `catboost>=1.2,<2` às dependências do `pyproject.toml`.
- Alinhe `environment.yml` e `pyproject.toml`: mesmas bibliotecas de runtime, e o `environment.yml` passa a instalar o pacote com `pip: ["-e .[dev]"]`.
- Em `[tool.ruff.lint.pydocstyle]`, defina `convention = "numpy"` e rode `ruff check --fix src tests`.

*Aceite:* `ruff check src tests` sem erros; `pip install -e ".[dev]"` num ambiente limpo seguido de `python -c "import ecom_delivery_sla"` funciona.

### 1.2 Teste fora do desenvolvimento (R-11)
- `run_regression_tournament` e `run_classification_tournament` recebem `evaluate_splits: tuple[str, ...] = ("validation",)` e levantam `ValueError` se `"test"` for pedido sem `allow_test=True`.
- Notebook 06: remova as métricas de teste e o texto "o teste permanece intocado". Substitua por: o período jul–ago/2018 foi observado em versões anteriores e está reservado ao notebook 09.
- Notebook 07: mesma correção.

*Aceite:* teste unitário garantindo que o teste não é avaliado por padrão; `grep -n '"test"' notebooks/*/0[6-7]*.ipynb` não retorna métricas.

### 1.3 Documentos coerentes (R-01, restante)
- `EDA-FASE-1-OLIST.md`:
  - status deixa de dizer "Nenhum modelo foi treinado";
  - tabela "Retrato do alvo" com 6.534 / 6,77%, citando a regra por calendário e o valor anterior (7.826 / 8,11%) como histórico.
- `PRE-ANALISE-FASE-1-OLIST.md` (linhas 63–64): marque 8,11% como valor pela regra fracionária, substituído por 6,77%.
- `CONTRATO-DADOS-FASE-2.md`, seção "Estado do gate": liste o que já existe (tabela candidata, split, alvo por calendário) e o que falta (anomalias, features, rede, multi-janela).
- `SEQUENCIA-DE-TRABALHO.md`:
  - linha 39: remova "(a criar)";
  - linhas 46–47: remova a duplicidade e deixe só "ponto de decisão aprovado: compra".
- `ARQUITETURA-MODULAR.md`: liste todos os módulos (`io`, `targets`, `labels`, `eda`, `geo`, `maps`, `features`, `temporal`, `modeling`, `optimization` e os novos) e os notebooks 01–09.

*Aceite:* nenhuma afirmação de status contradiz o código ou o registro de decisões.

### 1.4 Paridade do notebook 08
- Copie para `notebooks/en_us/08_optuna_optimization.ipynb` o valor atual de `STUDY_VERSION` do notebook PT-BR (`"V3"`).
- Não execute o notebook.

---

## Etapa 2 — Contrato de dados e features (P1)

### 2.1 Auditoria de anomalias reproduzível (R-06)
- Crie `audit_temporal_anomalies(orders)` em `src/ecom_delivery_sla/audit.py`. A função retorna uma linha por (regra, `order_id`) para as 6 regras da pré-análise:
  1. despacho antes da compra (166);
  2. entrega antes do despacho (23);
  3. não entregue com data real (6);
  4. status fora de `shipped`/`delivered` com despacho (75);
  5. estimada antes da compra (0);
  6. entrega antes da compra (0).
- Crie também `anomaly_policy_summary()`, com tipo, contagem, decisão (`include` / `exclude` / `sensitivity`) e justificativa.
- Registre a decisão por tipo no contrato.

*Aceite:* um teste sintético por regra; as contagens reais conferem com a pré-análise.

### 2.2 Disponibilidade e redundância de features (R-05)
- No catálogo: `promised_lead_days`, `primary_payment_type` e `payment_installments_max` passam a `available_at_purchase` (registre a fonte: dicionário oficial do dataset no Kaggle, que deve ser conferido).
- Remova da matriz de modelagem `payment_value_sum` (correlação 1,0 com preço + frete), `distance_min_km` e `distance_max_km` (iguais à média em 98% dos pedidos).
- `NUMERIC_FEATURES` e `CATEGORICAL_FEATURES` passam a ser derivadas do catálogo (status `available_at_purchase`), não de listas manuais.

*Aceite:* teste que falha se uma feature `pending*` ou `quality_control` entrar na matriz.

### 2.3 Novas features no momento da compra (R-07)
Em `features.py`, uma linha por pedido:

- **Logística:**
  - `total_weight_g`, `max_weight_g`, `total_volume_cm3`;
  - `handling_days_min` e `handling_days_max` (`shipping_limit_date` − compra);
  - `freight_per_kg`.
- **Geografia:**
  - `seller_state` (UF do vendedor de maior valor no pedido);
  - `same_state_share`;
  - `customer_is_capital`.
- **Calendário:**
  - `purchase_month`, `purchase_dayofweek`, `purchase_hour`;
  - `days_to_black_friday` e `days_to_christmas`, com datas fixas por ano em constante documentada.

*Aceite:* testes de granularidade (1 linha por `order_id`) e de agregação em pedido com 2 vendedores. Antes, confirme no contrato que `shipping_limit_date` está disponível no momento da compra. Se não estiver, marque como `available_at_approval` e deixe fora da matriz.

### 2.4 Tabela de scoring (R-08)
- `build_order_features(orders_all, ...)` gera features para todos os pedidos, sem alvo.
- `build_modeling_frame` faz o join apenas com a população supervisionada.

---

## Etapa 3 — Rede as-of, validação e baselines (P2)

### 3.1 Rede e histórico (R-09) — novo módulo `src/ecom_delivery_sla/network.py`
- **Regra central:** para o pedido comprado em `t`, use só pedidos com `order_delivered_customer_date < t` (desfechos de entrega) ou `order_delivered_carrier_date < t` (desfechos de despacho). O próprio pedido nunca entra.
- **Vendedor:**
  - `seller_hist_orders`;
  - `seller_hist_breach_rate`, suavizada por bayes empírico com prior global e força `m` documentada;
  - `seller_hist_dispatch_late_rate`, só em pedidos de vendedor único (despacho > `shipping_limit_date`);
  - `seller_days_since_last`.
- **Aresta vendedor → UF destino:**
  - `edge_hist_orders`;
  - `edge_hist_breach_rate` (suavizada);
  - `edge_hist_median_lead_days`.
- **Destino (UF):**
  - `dest_hist_breach_rate`;
  - `dest_hist_median_lead_days`.
- **Grafo:** `seller_out_degree` (UFs atendidas até `t`) e `dest_in_degree`.
- **Derivada:** `promise_slack = promised_lead_days − edge_hist_median_lead_days` (fallback para destino e depois global).
- **Cold start:** indicadores `seller_cold_start` e `edge_cold_start`.
- **Pedidos com vários vendedores:** agregue por máximo e por média ponderada por valor.
- **Implementação:** ordenação por tempo com `merge_asof` ou somas cumulativas por evento de desfecho. Não use loop por pedido.

*Aceite (obrigatório):*
- teste sintético em que um pedido A, comprado antes de B mas entregue depois da compra de B, **não** influencia as features de B;
- teste de que o próprio pedido não entra;
- teste do cold start.

### 3.2 Validação temporal multi-janela (R-10) — em `temporal.py`
- `expanding_window_folds(data, timestamp_col, first_val_month, last_val_month, embargo_days=30)` gera folds mensais de validação de 2017-10 a 2018-06. O treino termina `embargo_days` antes do início de cada fold.
- O período ≥ 2018-07-01 nunca aparece.
- Sensibilidade de censura: função que reporta a prevalência de jul–ago/2018 incluindo os pedidos `shipped` não entregues como violação e como exclusão. Só a contagem, sem métricas de modelo.

*Aceite:* testes de não sobreposição, embargo e ausência do período de teste.

### 3.3 Baselines (R-12) — em `modeling.py`
- **Regressão:**
  - média do treino;
  - `median(actual_lead_days) − promised_lead_days`;
  - a mesma regra por UF de destino.
- **Classificação:**
  - prevalência;
  - score = `−promised_lead_days`;
  - logística com `promised_lead_days` + UF.
- Proponha o ganho mínimo no `REGISTRO-DE-DECISOES.md` (sugestão: ≥ 5% em AP e ≥ 3% em MAE na média dos folds, vencendo em ≥ 3/4 dos folds), com status "pendente de aprovação".

### 3.4 Métricas (R-13)
- `classification_metrics` passa a incluir:
  - `brier`;
  - `ece_10bins`;
  - `precision_at_5pct`, `recall_at_5pct`, `precision_at_10pct`, `recall_at_10pct`.
- Remova `balanced_accuracy` a 0,5 do ranking (pode ficar como diagnóstico).
- Regressão quantílica do `actual_lead_days` (P50/P90) com pinball loss. Use `HistGradientBoostingRegressor(loss="quantile")` ou LightGBM `objective="quantile"`.

### 3.5 OLS inferencial (R-14) — notebook 06
- Use fórmula `statsmodels` com `C(customer_state, Treatment("SP"))`.
- Erros-padrão HC3 e cluster pelo vendedor principal.
- VIF, resíduos vs ajustado, QQ, Breusch–Pagan, distância de Cook.
- Sensibilidade sem |atraso| > 30 dias.
- Coeficientes padronizados.
- Logit com odds ratios e IC 95%.
- Somente o período de treino; sem métricas de teste.

*Aceite:* tabela de coeficientes com IC e texto de limitações.

---

## Etapa 4 — Torneio e Optuna (P3; implementar, não rodar a busca completa)

### 4.1 Torneio (R-15)
- Notebook 07 refeito sobre os folds da etapa 3.2, com as baselines.
- Modelos lineares com `StandardScaler`.
- Uma única política de desbalanceamento para todos: sem pesos de classe, com calibração isotônica ajustada no último fold de validação.
- Tempo de treino e de inferência por modelo.
- Resultado: média ± desvio por fold e ganho sobre a melhor baseline.

### 4.2 Optuna (R-16)
- Objetivo = média das métricas dos folds.
- `trial.report(score, step=fold_idx)` por fold, para que o pruner funcione.
- Scaler apenas para modelos lineares.
- Mesma política de desbalanceamento e mesmo orçamento (trials + `timeout`) por estimator.
- Nova `STUDY_VERSION` para não misturar com estudos anteriores.
- Unifique o pré-processamento com `modeling.py` (R-19): uma função só.
- Remova `importlib.reload` do notebook.

*Aceite:* testes unitários com 2 trials em dados sintéticos. O notebook 08 fica pronto, mas **não é executado** por você.

### 4.3 Drift (R-17)
- Função `population_stability_index(expected, actual, bins=10)`.
- Seção no notebook 05 com PSI por feature e prevalência mensal, só treino e validação.
- Destaque `promised_lead_days`.

---

## Etapa 5 — Reexecução e entrega

1. Reexecute de ponta a ponta, com `nbclient` e o mesmo kernel, os notebooks **01–07 PT-BR e EN-US**. Não reexecute o 08.
2. Atualize os textos de insight com os números do novo alvo (6.534 / 6,77%) e das novas análises. Isso inclui os insights dos notebooks 01 e 02, que ainda citam 7.826 / 8,11%.
3. Adicione a `tests/test_notebook_contract.py` um teste que exige `execution_count` sequencial e outputs não vazios em todas as células de código dos notebooks 01–07.
4. Atualize `docs/REVISAO-TECNICA-ESCOPO-E-GAPS.md` com uma coluna ou seção de status por `R-xx` (feito / parcial / pendente + evidência).

## Relatório final esperado
Liste:

- cada `R-xx` com status, arquivos alterados e testes adicionados;
- a saída final de `pytest` e `ruff`;
- as decisões novas registradas e pendentes de aprovação;
- os números-chave atualizados: prevalência por fold, melhor baseline vs melhor modelo por fold.

Não declare nenhum item como concluído sem a evidência correspondente.
