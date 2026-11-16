# Revisão técnica — escopo, notebooks, gaps e recomendações

**Data da revisão:** 2026-11-10
**Objeto:** `docs/*`, `notebooks/pt_br/*`, `notebooks/en_us/*`, `src/ecom_delivery_sla/*`, `tests/*`, `pyproject.toml`, `environment.yml`, `.github/workflows/ci.yml`
**Natureza:** revisão sem alteração de código. Este documento é o insumo de trabalho para a IA que executará as correções. Cada recomendação tem um identificador (`R-xx`) e um critério de aceite.
**Status documental:** registro histórico da situação encontrada em 2026-11-10. As pendências descritas abaixo não representam, isoladamente, o estado atual; o fechamento deve ser consultado em `REGISTRO-DE-DECISOES.md`, nos planos de fase e nos gates executados.

---

## 1. Veredito executivo

1. **O escopo é sólido no conceito e honesto nos limites** (SLA como proxy, unidade = pedido, proibição de pós-evento, nada de causalidade, sem APIs de rota). A documentação de riscos é acima da média.
2. **A execução atropelou os próprios gates.** Os documentos dizem que competição de estimators e Optuna só começariam depois de fechar contrato de features, grafo (Fase 5) e diagnósticos OLS (Fase 6). Na prática, as Fases 7 e 8 foram executadas com a Fase 2 "não aprovada", a Fase 5 inexistente e a Fase 6 incompleta.
3. **O conjunto de teste já foi consultado** (notebooks 06 e 07 reportam e ordenam por métricas de teste). Com isso, a promessa de usar o teste uma única vez (Fase 8) foi quebrada.
4. **Há um defeito na definição do alvo binário.** `order_estimated_delivery_date` não tem hora (100% dos valores são meia-noite). Por isso, entregas feitas **no próprio dia prometido** contam como violação. São 1.292 dos 7.826 "atrasos" (16,5%), e a taxa cai de 8,11% para 6,77% com a regra por dia de calendário.
5. **Os modelos quase não superam uma regra ingênua, e na classificação ficam abaixo dela no teste.**
   - Regressão: a regra "mediana do prazo realizado no treino − prazo prometido" tem RMSE de teste 6,306. O OLS fica em 6,272 e o melhor modelo (ElasticNet) em 6,238, um ganho de 0,5–1%.
   - Classificação: só a feature `−promised_lead_days` dá AUC 0,774 / AP 0,274 no teste. O melhor modelo do torneio dá AUC 0,658 / AP 0,122, e os modelos de árvore caem para AUC ≈ 0,52.
   - A causa provável é o **drift da política de promessa**: a mediana do prazo prometido cai de 23,7 dias (treino) para 16,4 dias (teste).
6. **O "Network" do título não existe no código.** Não há grafo, nem features históricas de vendedor/rota, nem política de cold start, e 30,1% dos pedidos do teste têm pelo menos um vendedor nunca visto no treino.
7. **O CI está quebrado.** `catboost` não está em `pyproject.toml`, mas é importado por `modeling.py`/`optimization.py` via `__init__`. Além disso, o `ruff check` acusa 59 erros com a configuração do próprio projeto.

**Conclusão:** o projeto deve voltar formalmente para o fim da Fase 2 / Fase 5. Os resultados dos notebooks 07 e 08 devem ser rebaixados a "exploratórios, não comparáveis" até que alvo, features, grafo e protocolo de validação sejam corrigidos.

---

## 2. Validação do escopo (`docs/`)

### 2.1 O que está correto e deve ser mantido

| Item | Onde | Avaliação |
|---|---|---|
| SLA operacionalizado como proxy (`order_estimated_delivery_date`) | CONTRATO, EDA, DICIONÁRIO | Correto e bem justificado |
| Unidade = pedido; não explodir em pedido–vendedor como observação de SLA | CONTRATO §múltiplos vendedores | Correto; argumento sobre `shipping_limit_date` vs data única de despacho é preciso |
| Lista de variáveis pós-evento proibidas | PRE-ANALISE, CONTRATO | Correta |
| Taxa histórica de vendedor estimada preferencialmente em pedidos de vendedor único | CONTRATO | Correto e sofisticado |
| Haversine sem API de rota/tráfego | CONTRATO, EDA | Correto (determinístico, sem dependência externa) |
| Não remover extremos sem auditoria | PRE-ANALISE | Correto |
| `production_eligible = false` até os gates | todos | Correto |

### 2.2 Inconsistências e erros no escopo

| ID | Problema | Evidência | Correção |
|---|---|---|---|
| S-01 | Ordem das fases não corresponde à execução. A EDA (Fase 4) foi feita antes da tabela analítica (Fase 3). | Notebooks 02/03 precedem o 04 | Reordenar: EDA → tabela analítica. Feito em `SEQUENCIA-DE-TRABALHO.md`. |
| S-02 | Gates declarados e não respeitados: as Fases 7 e 8 foram executadas sem as Fases 2, 5 e 6 fechadas | PRE-ANALISE: "aprovada apenas para continuar a pré-análise, não para iniciar competição de estimators, Optuna" | Registrar desvio; reclassificar 07/08 como exploratórios |
| S-03 | Status desatualizados | PRE-ANALISE: "Modelagem iniciada: não"; EDA: "Nenhum modelo foi treinado"; ARQUITETURA lista só `io/targets/labels/eda` e notebooks 01–03; CONTRATO "Estado do gate" ainda cita a tabela analítica e a divisão temporal como faltantes | Atualizar status (R-01) |
| S-04 | Definição de `sla_breach` ignora que a data estimada não tem hora | 100% de `order_estimated_delivery_date` à meia-noite; 1.292 entregas no próprio dia contadas como violação | Alvo binário por dia de calendário (R-02) |
| S-05 | Regra de histórico sem lag de rótulo: "excluindo o próprio pedido e qualquer observação posterior à sua compra" | O atraso de um pedido anterior só é conhecido **após a entrega**. Um pedido comprado antes mas ainda não entregue no instante da compra atual vaza informação futura | Regra "as-of": usar só pedidos cujo desfecho era conhecido antes da compra atual (R-09) |
| S-06 | `promised_lead_days` está "pendente de confirmação" sem caminho para fechar | O dicionário oficial do dataset no Kaggle descreve `order_estimated_delivery_date` como a data estimada informada ao cliente no momento da compra | Fechar como `available_at_purchase` citando a fonte (verificar o texto na página do Kaggle) e registrar a decisão no contrato (R-05) |
| S-07 | O escopo não define uma baseline ingênua nem um critério de "o modelo agrega valor" | Ausente em todos os docs; a pergunta 18 do anexo fica sem resposta | Incluir baselines obrigatórias e critério de ganho mínimo (R-12) |
| S-08 | Perguntas bloqueadoras do anexo (2, 6, 8, 9) e de modelagem (11–17) não têm respostas registradas, embora as fases dependentes já tenham sido executadas | ANEXO | Criar um registro de decisões (R-03) |
| S-09 | Fase 8 promete "teste usado uma única vez", mas as Fases 6 e 7 já reportam o teste | notebooks 06/07 | Declarar o teste "consultado" e redefinir o protocolo (R-11) |
| S-10 | O escopo trata a regressão de `delivery_delay_days` sobre toda a população como objetivo principal, mas ~92% dos valores são antecipações. RMSE/R² medem sobretudo "quão cedo" chega, o que tem pouco valor de negócio | Mediana −11,9 dias; o R² vem quase todo de `promised_lead_days` (alvo = prazo realizado − prometido) | Reenquadrar: classificação de risco como principal; regressão do prazo realizado (ETA) com quantis como secundária (R-13) |
| S-11 | Censura à direita no fim da série não é tratada | Jul–ago/2018: 107 pedidos `shipped` não entregues até o corte (17/10/2018) ficam fora da população; tendem a ser atrasos, o que reduz artificialmente a taxa do teste | Documentar e fazer análise de sensibilidade (R-10) |
| S-12 | Ruído de dados: CSVs duplicados em `data/raw/olist/` e `data/raw/olist/csv/` (284 MB); `data/` fora do `.gitignore`; sem README na raiz | Árvore de arquivos | R-20 |

---

## 3. Aderência dos notebooks ao escopo

| Notebook | Fase | Aderência | Problemas principais |
|---|---|---|---|
| 01 apresentação | 1 | Boa | EN-US **não executado** (sem outputs) |
| 02 EDA atraso | 4 | Boa | Correlação −0,50 citada no insight **não é calculada em nenhuma célula**; anomalias mostram só 4 regras (as 166/6/75 da pré-análise não são reproduzíveis por `src`); taxas mensais sem intervalo de confiança apesar de o doc pedir |
| 03 EDA geografia | 4 | Boa | Cobertura de 99,51% e mediana de 433,83 km citadas no texto **não aparecem em output**; taxas por composição sem IC (grupos com n = 5, 8, 14); `same_state_share` é calculado e descartado |
| 04 tabela candidata | 3 | Parcial | **Outputs desatualizados**: o notebook mostra 22 colunas e catálogo com 19 linhas; o código atual gera 24 colunas e 21 entradas (`distance_missing`, `payment_record_missing`). Tabela construída só sobre a população supervisionada (não há tabela de scoring) |
| 05 split temporal | 3 | Parcial | Cortes "provisórios" mas usados como definitivos em 06–08; sem embargo; sem análise do drift entre partições (prevalência 9,2% / 5,1% / 7,5%; promessa mediana 23,7 / 24,1 / 16,4 dias) |
| 06 baseline OLS | 6 | **Fraca** | Sem `summary()`, erros-padrão, HC3/cluster, VIF, resíduos; coeficientes ranqueados por magnitude bruta (dependente de escala); categoria de referência = AC (64 pedidos no treino); `payment_value_sum` tem correlação 1,0 com `total_price + total_freight_value`; distâncias min/média/máx idênticas em 98,35% dos pedidos; **teste reportado** |
| 07 torneio | 7 | **Não conforme** | Executado antes dos gates; **avalia e ordena pelo teste**; sem baseline ingênua; `balanced_accuracy` a 0,5 com pesos de classe distintos não é comparável; sem calibração, latência, estabilidade multi-fold; EN-US não executado |
| 08 Optuna | 8 | **Não conforme** | 9 estudos × 100 trials numa única janela de validação (overfitting à validação); pruner inútil; scaler buscado para árvores; sem refit treino+validação; sem avaliação final; EN-US não executado; `importlib.reload` no notebook |

---

## 4. Gaps e falhas

Severidade: **C** = crítico (invalida conclusão), **A** = alto, **M** = médio, **B** = baixo.

### 4.1 Conceito

| ID | Sev. | Falha | Evidência |
|---|---|---|---|
| C-01 | C | Alvo binário contaminado pelo horário: data estimada sem hora vs entrega com hora | 1.292 violações (16,5%) são entregas no dia prometido |
| C-02 | C | Teste consultado antes da seleção final | 06/07 exibem e ordenam pelo teste |
| C-03 | C | Ausência de baseline ingênua esconde que o modelo quase não agrega valor e, no teste, perde para uma regra de uma variável | Ver §1, item 5 |
| C-04 | A | Drift da política de promessa não foi investigado | Mediana da promessa 23,7 → 16,4 dias; árvores não extrapolam para valores fora do treino (AUC teste ≈ 0,52) |
| C-05 | A | Fase 5 (rede/grafo e histórico de vendedor/rota) inexistente; é o núcleo do título do projeto | Nenhum módulo de grafo; `networkx` só no `environment.yml` |
| C-06 | A | Regra de histórico sem lag de rótulo (vazamento sutil quando for implementada) | CONTRATO §indicador histórico |
| C-07 | A | Validação em janela única: a escolha do vencedor depende do regime abr–jun/2018 (junho com 1,36% de violação) | 05/08 |
| C-08 | A | OLS declarado "inferencial" sem nenhum aparato inferencial | 06 |
| C-09 | M | Regressão de atraso sobre toda a população tem baixo valor de negócio; RMSE dominado pela antecipação | §2.2 S-10 |
| C-10 | M | Pesos de classe (`balanced`, `scale_pos_weight`) distorcem as probabilidades; sem calibração; limiar 0,5 arbitrário | `modeling.py`, `optimization.py` |
| C-11 | M | Censura à direita no teste | 107 `shipped` em jul–ago/2018 |
| C-12 | M | Cold start não definido; relevante (30,1% dos pedidos de teste com vendedor novo) | — |
| C-13 | M | Features logísticas óbvias ausentes: peso/volume do produto, `shipping_limit_date − compra`, UF do vendedor, mesma UF, sazonalidade/calendário (mês, dia da semana, proximidade da Black Friday/feriados), capital vs interior | `NUMERIC_FEATURES` |
| C-14 | M | Redundância/colinearidade perfeita entre features | correlação 1,0; distâncias idênticas em 98% |
| C-15 | B | Pedidos com múltiplos vendedores/produtos com taxa menor: pode haver seleção (promessa maior: 24,4 vs 23,2 dias) | 03 |
| C-16 | B | Box-Cox (pergunta 12) inviável com valores negativos; não decidido | ANEXO |

### 4.2 Execução

| ID | Sev. | Falha | Evidência |
|---|---|---|---|
| E-01 | C | CI quebra na importação: `catboost` ausente em `pyproject.toml` | `modeling.py:9`, `optimization.py:10` |
| E-02 | A | `ruff check src tests` falha: 59 erros (51 D202, 4 I001, 3 D413, 1 UP035) | execução local com a config do projeto |
| E-03 | A | `run_*_tournament` avalia sempre validação **e teste**; não há modo "somente desenvolvimento" | `modeling.py` |
| E-04 | A | Notebooks EN-US 01, 07 e 08 sem execução; 04 PT/EN com outputs desatualizados | §3 |
| E-05 | A | Auditoria de anomalias da pré-análise (166/23/6/75) não é reproduzível pelo pacote; `anomaly_summary` cobre outras regras | `eda.py` |
| E-06 | M | Optuna: `MedianPruner` com um único `report(step=0)` só descarta trials já calculados (não economiza nada e marca como "pruned" trials válidos, o que enviesa o TPE) | `optimization.py` |
| E-07 | M | Optuna: espaço de busca com scaler para árvores (dimensão inútil); `scale_pos_weight` buscado em XGBoost enquanto LGBM/CatBoost usam `balanced` fixo (orçamento não comparável) | `optimization.py` |
| E-08 | M | Sem refit em treino + validação e sem avaliação final única (prometidos na Fase 8) | — |
| E-09 | M | Ridge/Logística no torneio e baseline sem padronização (L2 penaliza de forma desigual); só ElasticNet tem scaler | `modeling.py` |
| E-10 | M | `OneHotEncoder(drop="first")` → referência arbitrária (AC); coeficientes OLS ranqueados sem padronização | `modeling.py`, 06 |
| E-11 | M | Números nos insights sem célula que os produza (−0,50; 99,51%; 433,83 km) | 02, 03 |
| E-12 | M | Pré-processamento duplicado entre `modeling._make_preprocessor` e `optimization._preprocessor` | — |
| E-13 | B | `inventory_summary` com colunas fixas em PT (`tabela`, `linhas`, `colunas`) também no EN | `eda.py` |
| E-14 | B | Notebooks manipulam `sys.path` em vez de instalar o pacote (`pip install -e .`) | todos |
| E-15 | B | `environment.yml` e `pyproject.toml` divergem (catboost, networkx, matplotlib, seaborn) | — |
| E-16 | B | `__init__` com imports fora de ordem; `build_route_demo_pairs` fora de ordem; `__all__` não ordenado | — |
| E-17 | B | O ambiente local `ecom-delivery-sla` tem plugin pytest quebrado (`platformdirs.pytest_plugin`); os testes só rodam com `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` | execução local |

---

## 5. Recomendações (backlog para a IA executora)

Ordem obrigatória: **P0 → P1 → P2 → P3**. Não reexecutar 07/08 antes de concluir P0 e P1.

### P0 — Integridade (bloqueia tudo)

**R-01 Atualizar status dos documentos.** Corrigir os estados em PRE-ANALISE, EDA-FASE-1, ARQUITETURA-MODULAR e CONTRATO (gate). A ARQUITETURA deve listar todos os módulos e notebooks.
*Aceite:* nenhum documento afirma algo contrário ao estado do código.

**R-02 Corrigir o alvo binário.** Definir `sla_breach = order_delivered_customer_date.normalize() > order_estimated_delivery_date` (equivalente: prazo até 23:59:59 do dia estimado). Manter `delivery_delay_days` contínuo e acrescentar `delivery_delay_calendar_days` (inteiro). Registrar no CONTRATO a regra, a justificativa (data estimada sem hora) e o impacto (8,11% → 6,77%). Atualizar testes de `targets.py` com o caso "entregue às 15h do dia prometido = não violação" e todos os números citados nos docs.
*Aceite:* teste unitário cobrindo o caso; docs e notebooks reexecutados com a nova taxa.

**R-03 Criar `docs/REGISTRO-DE-DECISOES.md`.** Uma linha por pergunta do ANEXO (1–26): decisão, data, evidência, status. Fechar no mínimo 1–10 e 11–17 antes de P2.

**R-04 Corrigir CI.** Adicionar `catboost` às dependências do `pyproject.toml`, alinhar com o `environment.yml`, aplicar `ruff --fix` e resolver o conflito D203/D211 e D212/D213 com `convention = "numpy"` em `[tool.ruff.lint.pydocstyle]`.
*Aceite:* CI verde no GitHub Actions.

**R-11 Retirar o teste do desenvolvimento.** `run_regression_tournament` / `run_classification_tournament` passam a ter `evaluate_test: bool = False` e lançam erro se o teste for pedido fora da etapa final. Remover as métricas de teste dos notebooks 06 e 07. No CONTRATO, registrar que o teste jul–ago/2018 foi consultado nas versões anteriores. A avaliação final deve ser explicitamente rotulada como "holdout já observado uma vez sem seleção baseada nele", ou deve-se reservar a última janela (ex.: 2018-08) como novo teste congelado, com a perda de volume reportada.

### P1 — Contrato de dados e features (Fases 2–3)

**R-05 Fechar disponibilidade de features.** `promised_lead_days` passa a `available_at_purchase` (citar o dicionário oficial do Kaggle). `primary_payment_type` e parcelas também (escolhidos no checkout). `shipping_limit_date` passa a `available_at_approval` (decidir se o ponto de decisão é compra ou aprovação). Remover `payment_value_sum` (colinear) e manter só uma distância (`distance_mean_km`) mais `seller_count`.
*Aceite:* catálogo sem nenhum status "pending"; teste que impede features "pending" ou "quality_control" na matriz.

**R-06 Auditoria de anomalias reproduzível.** Uma função em `src` que gera as 6 regras da pré-análise (166/23/6/75/0/0) com IDs dos pedidos e uma decisão por tipo (incluir / excluir / sensibilidade).
*Aceite:* tabela reproduzida por notebook; teste sintético por regra.

**R-07 Enriquecer features disponíveis na compra.** Peso total, volume total (cm³) e peso máximo dos itens; `handling_days = shipping_limit_date − purchase` (mín/máx); UF do vendedor (ou UF do vendedor principal por valor); `same_state_share`; mês, dia da semana, dias até a Black Friday / Natal; flag de capital (cidade do cliente = capital da UF); `freight_value / peso`.
*Aceite:* catálogo atualizado e testes de granularidade.

**R-08 Tabela de scoring.** Separar `build_order_features(orders_all)` (todas as ordens, sem alvo) de `build_modeling_frame` (só supervisionados).

### P2 — Rede, histórico e protocolo de validação (Fases 5–6)

**R-09 Implementar a Fase 5 com regra as-of.**
- Granularidade sugerida: vendedor → UF de destino (e, opcionalmente, vendedor → prefixo CEP de 3 dígitos), em grafo bipartido dirigido.
- Para cada pedido no instante `t = purchase_timestamp`, usar só pedidos com `order_delivered_customer_date < t` (desfecho conhecido) e, para despacho, `order_delivered_carrier_date < t`.
- Features:
  - Vendedor: volume histórico, taxa de violação histórica com shrinkage bayesiano empírico (prior global), taxa de atraso de despacho vs `shipping_limit_date` (só pedidos de vendedor único), recência.
  - Aresta vendedor→UF: volume, taxa de violação suavizada, mediana do prazo realizado.
  - Nó destino (UF/CEP3): taxa suavizada e prazo realizado mediano.
  - Rede: grau do vendedor (número de UFs atendidas), grau do destino.
  - Indicadores explícitos de cold start (`seller_history_n == 0`) com fallback para o prior.
- Feature derivada de alto valor: `promise_slack = promised_lead_days − mediana histórica do prazo realizado na aresta`.
*Aceite:* teste sintético que falha se um pedido usar o desfecho de outro pedido entregue depois da sua compra; teste de que o próprio pedido nunca entra.

**R-10 Validação temporal multi-janela.** Expanding window com 4–6 folds mensais (ex.: validar em 2017-10…2018-06), com embargo de 30 dias entre o fim do treino e o início da validação para as features históricas. Reportar média ± desvio por fold. Fazer análise de sensibilidade da censura à direita: rotular os 107 `shipped` de jul–ago como violação e como exclusão, e reportar o intervalo.

**R-12 Baselines obrigatórias e critério de valor.**
- Regressão: média do treino; `mediana(prazo realizado) − promised_lead_days`; mesma regra por UF de destino.
- Classificação: prevalência; score = `−promised_lead_days`; logística só com `promised_lead_days` + UF.
- Um modelo só "agrega valor" se superar a melhor baseline em ≥ X% (a decidir no R-03) na média dos folds **e** em pelo menos 3/4 deles.

**R-13 Reenquadrar alvos.**
- **Principal:** classificação `sla_breach`, com métricas AP, Brier, ECE (10 bins), curva de confiabilidade e precisão/recall@k. O k deve ser definido por capacidade operacional (ex.: top 5% e top 10% do volume diário; responder às perguntas 13–14).
- **Secundário:** regressão do **prazo realizado** (`actual_lead_days`) com regressão quantílica (P50/P90). A probabilidade de violação implícita, P(prazo realizado > promessa), serve de comparação com o classificador.
- Manter `delivery_delay_days` só como análise descritiva/OLS.

**R-14 OLS inferencial de verdade (Fase 6).** Fórmula `statsmodels` com SP como referência de UF; erros-padrão HC3 e cluster por vendedor principal; VIF; resíduos vs ajustado, QQ, Breusch–Pagan; influência (Cook). Sensibilidade sem as caudas > 30 dias. Coeficientes padronizados para comparar magnitudes. Logit com odds ratios e IC. Se houver transformação, usar Yeo-Johnson (não Box-Cox).
*Aceite:* tabela de coeficientes com IC 95% e texto de limitação.

### P3 — Competição, otimização e entrega (Fases 7–9)

**R-15 Torneio refeito** sobre o protocolo R-10/R-12, com padronização para modelos lineares, sem pesos de classe (ou com pesos e calibração isotônica/Platt ajustada em fold separado), limiar escolhido na validação por custo ou capacidade, e tempo de treino e de inferência medidos. Árvores com `promised_lead_days` também na forma relativa (`promise_slack`) para mitigar o drift.

**R-16 Optuna corrigido.** Objetivo = média dos folds do R-10. Remover o pruner ou reportar por fold (`trial.report(score_fold, step=fold)`). Remover o scaler do espaço das árvores. Mesma política de desbalanceamento para todos. Orçamento igual por estimator em trials **e** tempo (pergunta 17). Refit do melhor em treino + validação e avaliação única no teste num notebook separado (`09_avaliacao_final`), com IC por bootstrap por bloco temporal.

**R-17 Diagnóstico de drift.** Notebook ou seção com PSI por feature entre treino, validação e teste; destaque para `promised_lead_days` e prevalência mensal. Explicar a queda de AUC das árvores no teste.

**R-18 Paridade bilíngue.** Reexecutar todos os notebooks PT e EN do zero (`nbclient`) com o mesmo ambiente e adicionar um teste que exige `execution_count` sequencial e outputs não vazios em todas as células de código. Gerar os números dos insights a partir de células (ou de um JSON de métricas versionado), não digitados.

**R-19 Higiene de código.** Unificar o pré-processamento (`modeling` e `optimization`); `OneHotEncoder(drop=None)` para modelos regularizados e de árvore, e referência explícita só no OLS; `inventory_summary` localizado; instalar o pacote via `pip install -e .` no `environment.yml` e remover a manipulação de `sys.path`; mover `importlib.reload` para fora dos notebooks.

**R-20 Repositório.** Remover os CSVs duplicados da raiz de `data/raw/olist/` (manter `csv/` + ZIP + hash); incluir `data/raw/**/*.csv` e `*.zip` no `.gitignore` com um script de download/verificação do SHA-256; criar `README.md` (objetivo, como reproduzir, estado dos gates, limitações).

---

## 6. Evidência numérica desta revisão

Reproduzida com `src/` atual e os CSVs locais (ambiente `ecom-delivery-sla`).

| Verificação | Resultado |
|---|---|
| `order_estimated_delivery_date` com componente de hora | 0% (todas à meia-noite) |
| Violações entregues no próprio dia estimado | 1.292 de 7.826 (16,5%) |
| Taxa de violação fracionária vs por dia de calendário | 8,11% vs 6,77% |
| Prevalência treino / validação / teste | 9,16% / 5,09% / 7,48% |
| Mediana de `promised_lead_days` treino / validação / teste | 23,70 / 24,06 / 16,40 dias |
| RMSE teste: média do treino / regra ingênua / OLS / melhor (ElasticNet) | 8,175 / 6,306 / 6,272 / 6,238 |
| RMSE validação: regra ingênua / OLS | 7,911 / 7,295 |
| AUC/AP teste só com `−promised_lead_days` | 0,774 / 0,274 |
| AUC/AP teste melhor modelo do torneio (Logística) | 0,658 / 0,122 |
| AUC/AP validação só com `−promised_lead_days` | 0,600 / 0,065 |
| Correlação `payment_value_sum` × (preço + frete) | 1,000 |
| Pedidos com distância mín = máx | 98,35% |
| Pedidos de teste com vendedor ausente no treino | 30,1% |
| Clientes recorrentes (`customer_unique_id` repetido) | 6,1% |
| `shipped` não entregues com compra em jul/ago 2018 | 60 / 47 |
| Testes unitários | 25 passam (com `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`) |
| `ruff check src tests` | 59 erros |

Observação sobre a inversão de ranking na classificação: na validação, os modelos (AUC ≈ 0,75–0,78) superam a regra de uma variável (0,60). No teste ocorre o contrário (0,52–0,66 vs 0,77). Esse é o sintoma típico de mudança de regime: a promessa encurtou e o conjunto de treino termina no pico de violação de fev–mar/2018 (16–21%). Por isso a validação em janela única (R-10) e o diagnóstico de drift (R-17) são pré-requisitos para qualquer escolha de vencedor.

---

## 7. Decisões que cabem ao responsável pelo projeto
1. Aceitar a redefinição de `sla_breach` por dia de calendário (R-02). É recomendada; altera todos os números publicados.
  - Aprovado a redefinição de sla_breach

2. Ponto de decisão: **compra** ou **aprovação do pagamento** (define se `shipping_limit_date` e o tempo de aprovação entram).
  - Me questione oportunamente e vejamos os prós e contra desta dicotomia

3. Tratamento do teste já consultado: rotular como "observado" ou reservar uma nova janela final menor.
  - rotule como "observado"

4. Granularidade do grafo: vendedor→UF (recomendado para o MVP) ou vendedor→CEP3.
  - Aprovado
5. Ganho mínimo sobre a baseline para declarar valor preditivo (sugestão: ≥ 5% em AP e ≥ 3% em RMSE/MAE na média dos folds).
  - Aprovado

6. Capacidade operacional que define o top-k da classificação.
  - Aprovado

## 8. Status de execução desta revisão

Atualizado em 2026-11-16, ao fim da Fase 8 e no início da Fase 9. A versão de 2026-11-11 desta tabela registrava o estado antes da execução dos notebooks com dados reais e foi substituída. Legenda: **feito** = código, teste sintético e execução dos notebooks PT e EN com os CSVs da Olist; **Fase 9** = item incluído no plano aprovado em `docs/PLANO-FASE-9.md` (P-07).

| ID | Status em 2026-11-16 | Evidência | Observação |
|---|---|---|---|
| R-01 | feito | Documentos alinhados ao alvo por calendário e ao estado dos gates; notebooks 01–08 reexecutados pelo responsável | Atualizações de documentos posteriores seguem a Fase 9 |
| R-02 | feito | `targets.py`; teste do mesmo dia prometido | — |
| R-03 | feito | `REGISTRO-DE-DECISOES.md`, uma linha por pergunta do anexo | Perguntas 12, 13, 19, 22, 24, 25 e 26 fechadas pelo P-07 |
| R-04 | feito localmente | `pyproject.toml`, `environment.yml`, Ruff verde, `.github/workflows/ci.yml` | O CI remoto nunca foi executado; a publicação está fora da Fase 9 |
| R-05 | feito | Catálogo em `features.py`, com 14 features de rede `available_at_purchase` | — |
| R-06 | feito | `audit.py` e notebooks de auditoria | — |
| R-07 | feito | Peso, volume, handling, seller principal, calendário e geografia em `features.py` | — |
| R-08 | feito | Tabela de scoring com a rede, uma linha por `order_id` | Tempo e memória não foram medidos formalmente |
| R-09 | feito | `network.py` (T-01 a T-03); prior global as-of verificado nos dados reais | — |
| R-10 | feito | Folds expanding-window com embargo e sensibilidade de censura no notebook 05; P-02 aprovado | Gráfico dos folds na Fase 9 (frente F) |
| R-11 | feito | Teste bloqueado no desenvolvimento; aberto uma única vez no notebook 09 (F1) e reutilizado de forma pós-hoc no notebook 10 (F2) | Julho e agosto de 2018 estão esgotados para novas avaliações |
| R-12 | feito | Baselines obrigatórias por fold, `tournament_summary` e critério P-01 aplicados no notebook 07 (P-04) | — |
| R-13 | feito | Brier, ECE, top-k e curva de confiabilidade nos notebooks 09 e 10 | — |
| R-14 | feito | `inferential.py` e notebook 06 | — |
| R-15 | feito | Política única sem pesos de classe e com calibração isotônica (T-05), tempos de ajuste (T-06) | Limiar por custo ou capacidade fora do escopo; top 5% e 10% como cenários (pergunta 13) |
| R-16 | feito | Optuna com objetivo na média dos folds (`STUDY_VERSION = "V4_folds"`), sem ganho relevante sobre a configuração padrão; avaliação final no notebook 09 (P-05) | — |
| R-17 | feito | PSI treino × validação no notebook 05 e treino × teste no notebook 09; drift discutido nas observações do F1 | Gráficos de PSI na Fase 9 (frente F) |
| R-18 | feito | Paridade de código PT/EN verde; notebooks 01–11 executados em PT e EN; teste de saídas (`REQUIRE_NOTEBOOK_OUTPUTS=1`) passa; notebook 07 reexecutado na Fase 9 | Números digitados nos insights dos notebooks 01–06 não foram auditados um a um; os dos notebooks 03 (rotas), 09, 10 e 11 foram conferidos contra as saídas |
| R-19 | encerrado parcialmente | Pré-processamento unificado; `pip install -e .` no `environment.yml` (Fase 9) | `OneHotEncoder(drop=None)` e remoção de `sys.path` não aplicados após a abertura do teste (P-07) |
| R-20 | feito | CSVs duplicados removidos, `.gitignore` e `.gitattributes`, verificação do SHA-256 dos dados (`scripts/verify_data.py`), `README.md` e repositório git local | A publicação remota fica para depois da revisão do responsável |

`production_eligible = false` permanece.
