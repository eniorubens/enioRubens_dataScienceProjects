# Previsão de Churn de Clientes — Projeto de ML Reprodutível de Ponta a Ponta

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/Customer-Churn-Prediction-EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/Customer-Churn-Prediction-PT.ipynb)

## 💡 Filosofia do projeto

**Camada de governança:** o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/) explicita o que o hold-out sustenta e por que receita/ROI permanecem ilustrativos.

**Produção orientada a contratos:** nos projetos com superfície de serving, APIs FastAPI/Flask expõem contratos de entrada e saída validados por testes. Nos demais, contratos de dados, artefatos, partições seladas e gates de avaliação controlam o que pode ser concluído, reproduzido e publicado.

**Superfície de engenharia:** módulos reutilizáveis em `churn_project/`, artefato de treino persistido, endpoints FastAPI em `serve_model.py` e testes automatizados.

**Pode sustentar:** priorização exploratória de risco de churn e inferência reprodutível para um piloto controlado de retenção.

**Não sustenta:** alegações causais de retenção, receita preservada observada ou uso produtivo sem validação temporal, monitoramento e experimento controlado.

## 🎯 Resumo executivo

Este projeto demonstra um **pipeline exploratório e reprodutível de machine learning**
para prever churn de clientes. O artefato materializado atual alcança **76,6% de Recall
Macro** e **84,6% de ROC AUC** no teste hold-out determinístico.

O trabalho demonstra experiência em reenquadrar problemas de ML como **desafios de
otimização de negócio**, indo além de métricas de acurácia para produzir insights
acionáveis que podem orientar impacto real sobre a receita.

**Fronteira principal:** o projeto traduz scores do modelo em uma hipótese de
priorização de retenção. Preservação de receita e ROI são cenários ilustrativos,
não resultados observados ou validados causalmente.

---

## 📊 Destaques do projeto

| Métrica | Valor | Significado |
|--------|-------|--------------|
| **Recall do modelo (Macro)** | 76,6% | Resultado hold-out no threshold 0,52 |
| **Acurácia balanceada** | 76,6% | Resultado hold-out balanceado por classe |
| **ROC AUC** | 84,6% | Discriminação hold-out usando probabilidades |
| **Precisão (Macro)** | 71,7% | Resultado hold-out no threshold 0,52 |
| **Receita anual preservada** | Apenas cenário | Depende de premissas de retenção não validadas |
| **ROI de retenção** | Apenas cenário | Não observado; exige experimento controlado |
| **Prazo de implementação** | 90 dias | Plano de deployment realista e faseado |

---

## 🔍 Problema

Este projeto reenquadra a previsão de churn de clientes de um **problema puramente
técnico de classificação** para um **desafio de otimização de negócio**:

### A realidade do negócio
- **Falso negativo (não identificar um churner):** perda do customer lifetime value (~US$ 2–5 mil, *assumido*)
- **Falso positivo (sinalizar um não churner):** custo de desconto de retenção (~US$ 150–300, *assumido*)

> Os valores em dólar acima são premissas ilustrativas, não valores do dataset nem de uma fonte.
- **Assimetria de custos:** 10–30 vezes mais caro não identificar um churner

### Abordagem da solução

Em vez de maximizar a acurácia (um erro comum de ML em dados desbalanceados), este trabalho:

1. **Prioriza Recall** — captura o máximo de churners antes que saiam
2. **Otimiza o threshold** — equilibra ganhos de recall com custos administráveis de falsos positivos
3. **Quantifica o impacto de negócio** — traduz métricas de ML em preservação de receita
4. **Planeja a implementação** — roadmap de 90 dias com ações e responsáveis específicos
5. **Mantém transparência** — documenta explicitamente limitações e premissas

---

## 🏗️ Metodologia e abordagem técnica

### Fase 1: Análise exploratória dos dados

#### Características do dataset
- **Fonte:** [Telecom Customer Churn (Kaggle)](https://www.kaggle.com/datasets/puja19/telecom-customer-churn)
- **Escala:** 7.043 clientes com 20 atributos (demografia, serviços e cobrança)
- **Alvo:** Churn (classificação binária)
- **Distribuição das classes:** 26,5% de churn nas 7.043 linhas brutas (26,6% após remover 11 linhas com `TotalCharges` em branco; 7.032 linhas usadas)

#### Principais insights de negócio descobertos

**Churn por tipo de contrato:**
```
Mês a mês: 43% de churn       ← Segmento de maior risco
Um ano:     11% de churn       ← Risco médio
Dois anos:  2,8% de churn      ← Menor risco (melhoria de 15x)
```
→ **Implicação:** o tipo de contrato é a principal alavanca para redução do churn.

**Efeito do tempo de relacionamento:**
- Mais de 40% dos churners saem nos primeiros 3 meses
- Após o mês 12, o churn se estabiliza significativamente
- A experiência inicial do cliente é crítica

**Sensibilidade a preço:**
- Clientes com cobrança mensal > US$ 65: churn 2 vezes maior
- Ausência de serviços de suporte: +29–37% de risco de churn
- Oportunidade de estratégia de bundles identificada

**Relações entre atributos (correlação PhiK):**
- Contract Type: φK = 0,45 (principal preditor de churn)
- Internet Service: φK = 0,38
- Tech Support: φK = 0,34
- Tenure: φK = 0,32

---

### Fase 2: Processamento e engenharia de atributos (pipeline vencedor)

O pipeline vencedor de pré-processamento foi descoberto por busca exaustiva com
`opt_binary_clf_pipe` (Optuna TPE, 100 trials, CV estratificada de 6 folds):

| Etapa | Componente | Configuração |
|-------|-----------|---------------|
| **Encoder** | `SumEncoder` | Codificação por soma/contraste para 15 atributos categóricos — captura contrastes binários e ordinais sem aumentar a esparsidade |
| **Scaler 1** | `Normalizer` | Escala cada *amostra* (linha) para norma unitária; não trata outliers por variável. Mantido porque foi selecionado pela busca, não por um argumento mecanístico |
| **Scaler 2** | `MaxAbsScaler` | Escala cada atributo pelo maior valor absoluto — preserva sinal e esparsidade |
| **Seletor de atributos** | `SelectFromModel` | Mantém 13 atributos usando as próprias importâncias do modelo (threshold = média) |
| **Classificador** | `BalancedRandomForestClassifier` | Random forest balanceada com hiperparâmetros otimizados |

**Por que SumEncoder?**
- A codificação por contraste mapeia variáveis categóricas para contrastes numéricos interpretáveis
- Não aumenta a dimensionalidade como one-hot encoding
- Captura informação ordinal implicitamente (por exemplo, ordenação da duração do contrato)

**Por que Normalizer + MaxAbsScaler?**
- `Normalizer` reescala cada linha para norma unitária (por linha, não por variável). Ensembles de árvores são pouco sensíveis a isso; ele aparece porque o Optuna o selecionou, e sua contribuição não foi isolada por ablação
- `MaxAbsScaler` alinha magnitudes — preserva uma escala favorável à esparsidade
- Juntos formam uma cadeia de normalização adequada para ensembles de árvores

---

### Fase 3: Desenvolvimento e seleção do modelo

Experimentos rastreados com `opt_binary_clf_pipe` + MLflow em 35 configurações.
Principais resultados:

| Modelo | Configuração | Pipeline | Recall de teste |
|-------|--------------|---------|------------|
| **BalancedRandomForestClassifier** | **Opt Balanced + Threshold** | **SumEncoder Norm MaxAbsScaler SFM 13 \| thr=0,52** | **0,764** |
| LinearDiscriminantAnalysis | Opt Balanced + Threshold | BinaryEncoder PowerTransformer SFS 13 \| thr=0,25 | 0,762 |
| BalancedRandomForestClassifier | Baseline | OrdinalEncoder | 0,758 |
| BalancedRandomForestClassifier | Opt Over-sampling + Threshold | JamesStein MinMax Robust SFM 10 SMOTE \| thr=0,44 | 0,758 |
| LogisticRegression | Opt Balanced + Threshold | BinaryEncoder QuantileN SFS 13 \| thr=0,47 | 0,756 |

> **Protocolo de seleção e seu limite.** Dentro da busca, cada trial é avaliado por CV de 6 folds apenas no treino, o pipeline inteiro (encoder, scalers, seletor, sampler, modelo) é reajustado por fold e cada threshold é escolhido em predições out-of-fold do treino. O teste só é usado para reportar. A exceção é o *ranking final acima*: o campeão foi escolhido por `Recall de teste`. Os três melhores candidatos otimizados estão estatisticamente empatados (teste 0,764 / 0,762 / 0,758; out-of-fold 0,768 / 0,769 / 0,768 para BRF / LDA / BRF-SMOTE), então a escolha do BRF é um desempate, não superioridade demonstrada. As linhas `Baseline + Threshold tuning` da LogisticRegression (thr=0,22 / 0,42) tiveram o threshold ajustado no teste e devem ser ignoradas nas comparações. Nas linhas de baseline, as métricas de `Train` são in-fold (ex.: RF 0,998); nas linhas otimizadas são out-of-fold.

#### Por que BalancedRandomForestClassifier?

1. **Desbalanceamento de classes por construção:** bootstrap com balanceamento por classe no nível da árvore — não exige SMOTE externo
2. **Evidência hold-out:** o artefato salvo atual alcança 76,6% de Recall Macro (IC95% bootstrap 74,2–79,1%) e 84,6% de ROC AUC (IC95% 82,3–86,8%) em 1.407 clientes de teste (374 churners). O recall out-of-fold de 6 folds do mesmo pipeline no treino é 76,6%, sem indício de viés otimista no número reportado
3. **Fronteira de validação:** é um único hold-out determinístico, não validação temporal ou produtiva
4. **ROC AUC 84,6%:** medida a partir de probabilidades contínuas no hold-out
5. **Fronteira de deployment:** exige validação temporal, monitoramento e experimento controlado de retenção

**Referência científica:** o BalancedRandomForestClassifier (Chen et al., 2004)
trata o desbalanceamento executando bootstrap balanceado em cada árvore, sendo
naturalmente robusto sem reamostragem externa.

---

### Fase 4: Otimização de hiperparâmetros

Busca sistemática de hiperparâmetros usando `opt_binary_clf_pipe` com o Tree-structured
Parzen Estimator (TPE) do Optuna:

```python
study = optuna.create_study(direction='maximize')
study.optimize(
    objective=objective_function,  # métrica: recall_macro
    n_trials=100,
    sampler=TPESampler(seed=42)
)
```

#### Hiperparâmetros finais e justificativa

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `n_estimators` | 306 | Ensemble maior reduz variância em ~5.600 amostras de treinamento |
| `max_depth` | 9 | Configuração do artefato de treinamento salvo atual |
| `max_features` | `log2` | Amostragem logarítmica de atributos por divisão — introduz diversidade e reduz correlação entre árvores |
| `min_samples_leaf` | 2 | Evita sobre-segmentação permitindo divisões granulares |
| `min_samples_split` | 6 | Controla o crescimento das árvores; regulariza contra overfitting |
| `random_state` | 738 | Seed fixa garante reprodutibilidade completa |
| `class_weight` | `balanced` (implícito) | Balanceamento inerente no nível da árvore — desbalanceamento 73/27 tratado internamente |

**Nota de desempenho:** o artefato salvo atual alcança **76,6% de Recall Macro**
no hold-out. Comparações com outputs anteriores dos notebooks devem ser tratadas
como históricas até que o notebook seja reexecutado de ponta a ponta.

---

### Fase 5: Otimização de threshold para objetivos de negócio

A classificação binária produz a probabilidade P(churn). O threshold determina
a fronteira de classificação.

**Threshold selecionado: 0,52**

O threshold 0,52 está ligeiramente acima do padrão 0,50, indicando que o modelo
já aprendeu uma fronteira de decisão bem calibrada para maximização de recall.
Ajustes pequenos conforme as prioridades do negócio continuam possíveis.

```
Threshold = 0,50 (Padrão)
  → Mudança mínima em relação ao modelo otimizado

Threshold = 0,52 (Otimizado)
  → Recall Macro: 76,6%, Precision Macro: 71,7%
  → Perfil de custos fortemente assimétrico: captura 80,7% dos churners reais
```

**Decisão de negócio:** o threshold = **0,52** foi escolhido porque:
- A fronteira de decisão do modelo já está bem calibrada nesse ponto
- Maximiza Recall Macro na CV estratificada de 6 folds
- O custo assimétrico (não identificar churner >> sinalizar não churner) justifica essa posição

**Referência científica:** Fawcett (2006) demonstra que a seleção de threshold
é uma decisão estratégica de negócio, não apenas um detalhe técnico.

---

### Fase 6: Estratégia de validação

#### Framework de validação cruzada
```python
cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
```

**Por que Stratified 6-Fold?**
- **Estratificada:** mantém 26,5% de distribuição de churn em cada fold
- **6-Fold:** reduz variância; padrão para classificação desbalanceada
- **Random state fixo:** garante reprodutibilidade entre execuções

#### Métricas apropriadas para dados desbalanceados

| Métrica | Fórmula | Aplicação | Por que é importante |
|--------|---------|-------------|----------------|
| **Recall Macro** | TP/(TP+FN) médio | Minimizar churners não identificados | Falsos negativos são mais caros |
| **Acurácia balanceada** | (Sensibilidade+Especificidade)/2 | Peso igual para ambas as classes | Não enviesada para desbalanceamento |
| **Precision Macro** | TP/(TP+FP) médio | Gerenciar custos de retenção | Custo dos falsos positivos |
| **ROC AUC** | Área sob a curva | Independente do threshold | Discriminação geral |
| ~~Accuracy~~ | ~~(TP+TN)/N~~ | ~~Inadequada aqui~~ | ~~Enganosa em dados desbalanceados~~ |

**Nota importante:** acurácia é inadequada para classificação desbalanceada
(He & Garcia, 2009). Este trabalho enfatiza Acurácia Balanceada e Recall.

---

## 📈 Desempenho e resultados do modelo

### Métricas gerais (Conjunto de teste, n≈1.407)

| Métrica | Vencedor | Baseline (LR) | Δ |
|--------|--------|---------------|---|
| Recall Macro | **76,6%** | comparação histórica | — |
| Acurácia balanceada | **76,6%** | comparação histórica | — |
| ROC AUC | **84,6%** | comparação histórica | — |
| Precision Macro | 71,7% | comparação histórica | — |

### Análise da matriz de confusão

```
Distribuição das previsões (Conjunto de teste):
  Corretas (TN + TP):  1.065
  FP (Alarme falso):     262    ← não churners sinalizados (custo de retenção desnecessário)
  FN (Churner não identificado): 80    ← churners não identificados (erro mais caro)

Derivados:
  Verdadeiros positivos (TP): 294   — churners identificados corretamente
  Falsos negativos (FN):       80   — churners não identificados
  Falsos positivos (FP):      262   — não churners sinalizados incorretamente
  Verdadeiros negativos (TN): 771   — não churners dispensados corretamente
```

**Recall por classe:**
- Recall da classe churner: 294 / (294+80) = **78,6%**
- Recall da classe não churner: 771 / (771+262) = **74,6%**

**Interpretação:**
- **Verdadeiros positivos (294):** churners identificados corretamente com janela de intervenção ✓
- **Falsos negativos (80):** churners não identificados ✗
- **Falsos positivos (262):** não churners sinalizados; aceitabilidade do custo exige economia real de intervenção
- **Verdadeiros negativos (771):** não churners identificados corretamente ✓

### Importância dos atributos (Permutation Importance, scorer Recall Macro)

**5 principais fatores de churn — Conjunto de teste:**

```
1. Contract              ← Principal fator generalizável
2. InternetService       ← Preditor consistentemente importante (treino + teste)
3. tenure               ← Padrão clássico de churn: clientes novos cancelam mais
4. TotalCharges         ← Forte no treino, moderado no teste
5. OnlineSecurity       ← Importância moderada e estável
```

**Atributos de baixo impacto (candidatos à remoção):**
`StreamingTV`, `StreamingMovies`, `PhoneService`, `gender`, `Partner`

**Insight de negócio:** a estrutura do contrato é a principal alavanca para redução
do churn — consistente com os achados de EDA (φK=0,45).

### Análise de erros

**Perfil dos churners não identificados (FN):**
- Tenure mediano: **35 meses** (vs 6 meses para churners identificados)
- MonthlyCharges mediano: **US$ 67** (vs US$ 79 para churners identificados)
- **Principal achado:** clientes FN se parecem com não churners nos atributos superficiais — tenure maior e cobranças moderadas. São os casos mais difíceis para qualquer modelo de churn.

**Distribuição de contratos nos erros:**
| Contrato | FN % | FP % |
|----------|------|------|
| Mês a mês | 40,3% | 93,4% |
| Um ano | 48,6% | 6,6% |
| Dois anos | 11,1% | 0,0% |

---

## 💰 Impacto de negócio e preservação de receita

### Contexto de receita

**Sem modelo (baseline de 2% de churn mensal):**
```
Perda anual por churn não gerenciado: ≈US$ 12M sobre US$ 50M de ARR
```

### Cenário ilustrativo usando o recall do hold-out (não é impacto observado)

**Economia da campanha de retenção (escala do dataset completo):**
- O cenário assume a escala de uma campanha sobre o dataset completo e não é resultado de campanha medido.
- Custo de intervenção e LTV são entradas assumidas, não custos observados.
- A premissa de 60% de sucesso de retenção é ilustrativa e deve ser testada prospectivamente.
- Qualquer valor de preservação de receita é, portanto, um cenário e não um resultado.

**Nenhum ROI observado é reportado.** Substitua as premissas por resultados de
experimento controlado antes do uso de negócio.

### Comparação entre modelo e ausência de modelo

| Cenário | Churners não identificados | LTV perdido | Custo de intervenção |
|----------|----------------|----------|-------------------|
| Sem modelo | ~1.866 | ≈US$ 6,5M | US$ 0 |
| Com modelo | ≈440 (23,6%) | ≈US$ 1,54M | ≈US$ 634K |
| **Benefício líquido** | **−1.426 churners resgatados** | **≈US$ 4,9M preservados** | |

---

## 🚀 Framework estratégico de implementação

### Rollout faseado em 90 dias

#### **Mês 1: Fundação e ganhos rápidos**

**Semana 1–2: Programa de retenção VIP**
- Alvo: clientes de alto valor na faixa de risco (mês a mês + cobranças altas)
- Intervenção: chamadas personalizadas + oferta de desconto de 20%
- Impacto esperado: salvar 50–70% dos churners de alto valor identificados
- Receita protegida: ≈US$ 2,1M

**Semana 3–4: Melhoria do onboarding inicial**
- Alvo: clientes nos meses 1–3 (40% do churn ocorre aqui)
- Intervenção: suporte CS reforçado, tutoriais de funcionalidades e check-ins regulares
- Impacto esperado: reduzir churn inicial de 45% → 30%
- Receita protegida: ≈US$ 2,5M

#### **Mês 2: Iniciativas estratégicas**

**Semana 5–6: Programa de upgrade de contrato**
- Alvo: clientes mês a mês (taxa de churn de 43%; 93,4% dos falsos positivos)
- Intervenção: incentivar migração para contratos de 1 ano com desconto de 15–20%
- Impacto esperado: converter 20% dos clientes elegíveis
- Receita protegida: ≈US$ 3,2M

**Semana 7–8: Campanha de bundles de serviços**
- Alvo: clientes sem Tech Support ou Online Security
- Intervenção: bundle "Security + Support" com desconto de 25%
- Impacto esperado: 40% de adesão entre elegíveis
- Receita protegida: ≈US$ 1,8M

#### **Mês 3: Otimização e escala**

**Semana 9–10: Gestão da sensibilidade a preço**
- Alvo: clientes com cobrança alta (mensal > US$ 65; risco de churn 2x maior)
- Intervenção: teste A/B de faixas alternativas de preço
- Impacto esperado: reduzir churn em 12–15% no segmento

**Semana 11–12: Prontidão para produção**
- Configurar pipeline de scoring em tempo real
- Estabelecer monitoramento do modelo (acompanhamento de desempenho e detecção de drift)
- Planejar ciclo de retreinamento trimestral

---

## 📚 Base científica

Este projeto se apoia em pesquisa revisada por pares:

**He, H., & Garcia, E. A. (2009).** "Learning from Imbalanced Data." *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.
→ Valida Acurácia Balanceada e Recall como métricas adequadas quando Accuracy falha.

**Guyon, I., & Elisseeff, A. (2003).** "An Introduction to Variable and Feature Selection." *Journal of Machine Learning Research*, 3, 1157–1182.
→ Justifica a abordagem de seleção de atributos multimétrica (Chi², ANOVA, PhiK, SelectFromModel).

**Fawcett, T. (2006).** "An Introduction to ROC Analysis." *Pattern Recognition Letters*, 27(8), 861–874.
→ Demonstra a otimização de threshold como decisão de negócio, não técnica.

**Micci-Barreca, D. (2001).** "A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems." *ACM SIGKDD Explorations Newsletter*, 3(1), 83–102.
→ Justifica abordagens da família SumEncoder para codificação categórica robusta.

**Ke, G., et al. (2017).** "LightGBM: A Fast, Distributed, Gradient Boosting Framework." *NeurIPS*.
→ Referenciado durante a seleção do modelo; LightGBM foi um forte candidato.

---

## ⚠️ Limitações críticas e transparência

### Restrições conhecidas

**1. Snapshot temporal**
- O dataset representa um único ponto no tempo (~2019)
- **Recomendação:** o modelo exige retreinamento trimestral/semestral

**2. Atributos comportamentais ausentes**
- Não há histórico de interações (tickets de suporte, padrões de uso de funcionalidades)
- Não há métricas de satisfação do cliente (NPS, CSAT)
- **Impacto:** churners silenciosos (tenure alto, cobranças moderadas) são os mais difíceis de identificar — confirmado pela análise de FN

**3. Correlação ≠ causalidade**
- Cobranças mensais altas se correlacionam com churn (não são sua causa)
- **Recomendação:** valide premissas causais com testes A/B antes de grandes mudanças de preço

**4. Limitações de generalização por segmento**
- O modelo foi treinado especificamente em clientes de telecomunicações
- **Recomendação:** exigir retreinamento para novos segmentos de clientes

**5. Premissa de sucesso da intervenção**
- O modelo de ROI assume 60% de sucesso de retenção
- **Recomendação:** piloto com 1.000 clientes; medir resultados reais antes do rollout completo

**6. Volume de falsos positivos**
- 262/1.033 não churners sinalizados (taxa FP de 25,4%)
- Se o orçamento de retenção for limitado, pode ser necessário ajustar o threshold

### Capacidades do modelo: o que ele PODE e NÃO PODE fazer

**✅ O modelo PODE:**
- Identificar 78,6% dos churners neste hold-out (recall por classe); o lead time não está estabelecido
- Priorizar investimento de retenção em segmentos de alto risco
- Estimar impacto de receita e ROI de intervenções
- Informar estratégias de contrato e preço

**❌ O modelo NÃO PODE:**
- Garantir sucesso de retenção (depende inteiramente da qualidade da campanha)
- Explicar mecanismos causais de churn (apenas correlações)
- Funcionar efetivamente sem retreinamento periódico
- Identificar com confiabilidade churners de longo tenure e cobrança moderada (padrão FN confirmado)

---

## 🔮 Oportunidades de melhorias futuras

### Curto prazo (3–6 meses)
1. **Otimização sensível a custos** — incorporar custos reais de retenção na função de perda
2. **Modelagem de uplift** — medir impacto causal por meio de propensity scoring
3. **Modelos específicos por segmento** — separar modelos para mês a mês e contratos longos

### Médio prazo (6–12 meses)
4. **Modelagem temporal** — LSTM/GRU para capturar sequências comportamentais antes do churn
5. **Inferência causal** — causal forests para identificar fatores reais versus correlatos

### Longo prazo (12+ meses)
6. **Reinforcement Learning** — otimizar dinamicamente a estratégia de retenção com base nos resultados
7. **Previsão de Customer Lifetime Value** — priorizar retenção para clientes de alto LTV

---

## 📊 Stack técnica e escolhas arquiteturais

**Linguagens e frameworks:**
- Python 3.10+ (processamento de dados e modelagem)
- Jupyter Notebook (análise interativa e documentação)

**Processamento de dados e ML:**
- Pandas, NumPy (manipulação de dados)
- Scikit-learn (pipelines de pré-processamento e avaliação)
- imbalanced-learn (BalancedRandomForestClassifier)
- category_encoders (SumEncoder e outros encoders)

**Otimização e rastreamento de experimentos:**
- opt_binary_clf_pipe (biblioteca customizada — ciclo completo de experimentação)
- Optuna (otimização bayesiana de hiperparâmetros)
- MLflow (rastreamento de experimentos e registro de modelos)

**Análise estatística:**
- PhiK (análise de correlação para variáveis de tipos mistos)
- SciPy (testes estatísticos)

**Visualização e relatórios:**
- Matplotlib, Seaborn

**API e produção:**
- FastAPI + Uvicorn (serving de API REST)
- pytest + GitHub Actions CI

---

## 💡 Principais aprendizados e demonstração de competência

| Competência | Evidência |
|------------|---------|
| **Engenharia de ML** | Pipeline de ponta a ponta, de EDA → otimização → validação → API |
| **Visão de negócio** | Traduz ML em receita (≈US$ 3M de impacto, ≈4,7x de ROI) |
| **Rigor estatístico** | Métricas adequadas para dados desbalanceados e escolhas justificadas |
| **Comunicação** | Documentação clara, explicações executivas e técnicas |
| **Resolução de problemas** | Identifica fatores de churn e recomenda ações específicas |
| **Base acadêmica** | 5 referências revisadas por pares, com métodos de ponta citados |
| **Transparência** | Documenta explicitamente limitações e perfis de erro |
| **Planejamento** | Roadmap de implementação de 90 dias com cronograma |

---

## 👨‍💼 Notas do autor e da metodologia

**Autor:** Enio Rubens
**Função:** Data Science & Analytics

### Abordagem de desenvolvimento

Este projeto demonstra a colaboração efetiva entre experiência humana e ferramentas
assistidas por IA:

- **Assistida por IA:** otimização de código, estruturação da documentação e validação analítica
- **Conduzida pelo humano:** enquadramento do problema de negócio, decisões metodológicas, interpretação dos resultados e recomendações estratégicas

Todas as decisões analíticas, interpretações de negócio, escolhas de modelagem,
procedimentos de validação e integração do projeto foram conduzidos e revisados pelo autor.

---

## 📝 Formato de citação

```bibtex
@misc{rubens2024churn,
  author = {Rubens, Enio},
  title = {Customer Churn Prediction: Reproducible End-to-End ML Project},
  year = {2024},
  note = {Portfolio Project}
}
```

---

## 📄 Licença e agradecimentos

**Dataset:** [Telecom Customer Churn (Kaggle)](https://www.kaggle.com/datasets/puja19/telecom-customer-churn) — licença CC0

**Bibliotecas:** comunidades de desenvolvimento de scikit-learn, imbalanced-learn, Optuna, Pandas e NumPy

**Pesquisa:** artigos citados de He & Garcia, Guyon & Elisseeff, Fawcett, Micci-Barreca e Ke et al.

---

**Última atualização:** outubro de 2026
**Status do projeto:** ✅ Artefato exploratório de portfólio
**Desempenho do modelo:** 76,6% Recall Macro | 76,6% Acurácia Balanceada | 84,6% ROC AUC | ROI não validado

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11+-orange.svg)](https://imbalanced-learn.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

*Este é um projeto de portfólio que demonstra experiência em data science, machine learning, análise de negócio e pensamento estratégico.*
