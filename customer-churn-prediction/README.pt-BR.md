# Previsão de churn de clientes: projeto de ML reprodutível de ponta a ponta

[English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/Customer-Churn-Prediction-EN.ipynb) | [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/Customer-Churn-Prediction-PT.ipynb)

## Escopo e evidências

Pode sustentar: priorização exploratória de risco de churn e inferência reprodutível para um piloto controlado de retenção.

Não sustenta: alegações causais de retenção, receita preservada observada ou uso produtivo sem validação temporal, monitoramento e experimento controlado.

Superfície de engenharia: módulos reutilizáveis em `churn_project/`, artefato de treino persistido, endpoints FastAPI em `serve_model.py` e testes automatizados.

Camada de governança: o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/) explicita o que o hold-out sustenta e por que receita/ROI permanecem ilustrativos.

## Resumo executivo

O projeto implementa um pipeline exploratório e reprodutível de machine learning para prever churn de clientes. O artefato materializado atual alcança 76,4% de Recall Macro (equivalente à acurácia balanceada) e 83,5% de ROC AUC no teste hold-out determinístico.

O problema de classificação é reenquadrado como uma decisão de negócio: as métricas de acurácia são substituídas pelo Recall Macro, e os scores do modelo são traduzidos em uma hipótese de priorização de retenção.

Fronteira principal: preservação de receita e ROI são cenários ilustrativos, baseados em premissas, e não resultados observados ou validados causalmente.

---

## Destaques do projeto

| Métrica | Valor | Significado |
|--------|-------|--------------|
| Recall Macro (equivalente à acurácia balanceada) | 76,4% | Resultado hold-out no threshold 0,52 |
| ROC AUC | 83,5% | Discriminação hold-out usando scores contínuos |
| Precisão (Macro) | 71,2% | Resultado hold-out no threshold 0,52 |
| Receita anual preservada | Apenas cenário | Depende de premissas de retenção não validadas |
| ROI de retenção | Apenas cenário | Não observado; exige experimento controlado |

---

## Problema

O projeto reenquadra a previsão de churn de um problema puramente técnico de classificação para uma decisão de negócio com custos assimétricos.

### A realidade do negócio

Um falso negativo (não identificar um churner) significa perda do customer lifetime value, assumido em cerca de US$ 2–5 mil. Um falso positivo (sinalizar um não churner) significa o custo de um desconto de retenção, assumido em cerca de US$ 150–300. Os valores em dólar são premissas ilustrativas, não valores do dataset nem de uma fonte. Com essas premissas, não identificar um churner custa de 10 a 30 vezes mais que sinalizar um não churner.

### Abordagem da solução

Em vez de maximizar a acurácia, que engana em dados desbalanceados, o trabalho prioriza o Recall Macro para capturar o máximo de churners antes que saiam. O threshold foi escolhido para maximizar essa métrica, e não por um cálculo de custo. A tradução das métricas em preservação de receita é feita apenas como cenário ilustrativo, e as limitações e premissas são documentadas explicitamente.

---

## Metodologia e abordagem técnica

### Fase 1: análise exploratória dos dados

O dataset é o [Telecom Customer Churn (Kaggle)](https://www.kaggle.com/datasets/puja19/telecom-customer-churn), com 7.043 clientes e 20 atributos (demografia, serviços e cobrança). O alvo é Churn (classificação binária), com 26,5% de churn nas 7.043 linhas brutas (26,6% após remover 11 linhas com `TotalCharges` em branco; 7.032 linhas usadas).

#### Principais achados de negócio

O churn por tipo de contrato mostra a maior diferença de risco:

```
Mês a mês: 43% de churn       ← Segmento de maior risco
Um ano:     11% de churn       ← Risco médio
Dois anos:  2,8% de churn      ← Menor risco (melhoria de 15x)
```

O tipo de contrato é, portanto, a principal alavanca associada à redução do churn.

Quanto ao tempo de relacionamento, mais de 40% dos churners saem nos primeiros 3 meses, e após o mês 12 o churn se estabiliza de forma significativa; a experiência inicial do cliente é crítica. Quanto ao preço, clientes com cobrança mensal acima de US$ 65 têm churn 2 vezes maior, e a ausência de serviços de suporte está associada a um risco de churn 29–37% maior, o que sugere explorar uma estratégia de bundles.

Pela correlação PhiK, o tipo de contrato é o principal preditor de churn (φK = 0,45), seguido por Internet Service (0,38), Tech Support (0,34) e Tenure (0,32).

---

### Fase 2: processamento e engenharia de atributos (pipeline vencedor)

O pipeline vencedor de pré-processamento foi selecionado pela busca com `opt_binary_clf_pipe` (Optuna TPE, 100 trials, CV estratificada de 6 folds):

| Etapa | Componente | Configuração |
|-------|-----------|---------------|
| Encoder | `SumEncoder` | Codificação por soma (desvio) para 15 atributos categóricos: cada nível é contrastado com a média geral, sem codificar ordem entre os níveis |
| Scaler 1 | `Normalizer` | Escala cada amostra (linha) para norma unitária; não trata outliers por variável. Selecionado pela busca, sem justificativa clara para modelos de árvore e sem ablação |
| Scaler 2 | `MaxAbsScaler` | Escala cada atributo pelo maior valor absoluto; preserva sinal e esparsidade |
| Seletor de atributos | `SelectFromModel` | Mantém 13 atributos usando as próprias importâncias do modelo (threshold = média) |
| Classificador | `BalancedRandomForestClassifier` | Random forest balanceada com hiperparâmetros otimizados |

A codificação por soma (sum ou deviation coding) mapeia cada variável categórica em colunas de contraste, em que cada nível é comparado com a média geral. Ela não aumenta a dimensionalidade tanto quanto a codificação one-hot, mas também não representa a ordem dos níveis, como a duração do contrato.

O `Normalizer` reescala cada linha para norma unitária (por linha, não por variável). Ensembles de árvores são pouco sensíveis a essa operação, e a normalização por linha não tem justificativa clara antes de um modelo de árvore. Ela aparece no pipeline porque a busca do Optuna a selecionou, e sua contribuição não foi isolada por ablação. O `MaxAbsScaler` alinha as magnitudes dos atributos preservando uma escala favorável à esparsidade.

---

### Fase 3: desenvolvimento e seleção do modelo

Os experimentos foram rastreados com `opt_binary_clf_pipe` e MLflow em 35 configurações. Os principais resultados:

| Modelo | Configuração | Pipeline | Recall de teste |
|-------|--------------|---------|------------|
| BalancedRandomForestClassifier | Opt Balanced + Threshold | SumEncoder Norm MaxAbsScaler SFM 13 \| thr=0,52 | 0,764 |
| LinearDiscriminantAnalysis | Opt Balanced + Threshold | BinaryEncoder PowerTransformer SFS 13 \| thr=0,25 | 0,762 |
| BalancedRandomForestClassifier | Baseline | OrdinalEncoder | 0,758 |
| BalancedRandomForestClassifier | Opt Over-sampling + Threshold | JamesStein MinMax Robust SFM 10 SMOTE \| thr=0,44 | 0,758 |
| LogisticRegression | Opt Balanced + Threshold | BinaryEncoder QuantileN SFS 13 \| thr=0,47 | 0,756 |

> Protocolo de seleção e seu limite. Dentro da busca, cada trial é avaliado por CV de 6 folds apenas no treino, o pipeline inteiro (encoder, scalers, seletor, sampler, modelo) é reajustado por fold e cada threshold é escolhido em predições out-of-fold do treino. O teste só é usado para reportar. A exceção é o ranking final acima: o campeão foi escolhido por `Recall de teste`. Os três melhores candidatos otimizados estão estatisticamente empatados (teste 0,764 / 0,762 / 0,758; out-of-fold 0,768 / 0,769 / 0,768 para BRF / LDA / BRF-SMOTE), então a escolha do BRF é um desempate, não superioridade demonstrada. As linhas `Baseline + Threshold tuning` da LogisticRegression (thr=0,22 / 0,42) tiveram o threshold ajustado no teste e devem ser ignoradas nas comparações. As colunas `Test`/`Train` também mudam de significado conforme o tipo de linha: nas linhas `Baseline` são médias de validação cruzada no treino (fold de validação e in-fold, ex.: RF `Train` 0,998); nas linhas `Opt …`, `Train` é out-of-fold e `Test` é o teste hold-out; nas linhas `Baseline + Threshold tuning`, `Test` vem de uma varredura de limiar no próprio teste. Só as linhas `Opt …` são comparáveis entre si: o valor `Baseline` do BalancedRandomForest (0,758) é uma média de validação cruzada, a comparar com o 0,768 out-of-fold do campeão, não com seu teste de 0,764.

#### Por que BalancedRandomForestClassifier?

O BalancedRandomForestClassifier trata o desbalanceamento de classes por construção: usa bootstrap com balanceamento por classe no nível da árvore, sem exigir SMOTE externo (Chen et al., 2004). O campeão do notebook (artefato reproduzido por `train_model.py` com a mesma divisão estratificada 80/20, semente 42) alcança 76,4% de Recall Macro (IC95% bootstrap 74,0–78,7%) e 83,5% de ROC AUC (IC95% 81,4–85,6%) em 1.407 clientes de teste (374 churners). O ROC AUC é medido a partir de scores contínuos no hold-out. O recall out-of-fold de 6 folds no treino é 76,8% (AUC 84,2%), sem indício de viés otimista no número reportado.

A validação é um único hold-out determinístico, não validação temporal ou produtiva. O uso em produção exige validação temporal, monitoramento e experimento controlado de retenção.

---

### Fase 4: otimização de hiperparâmetros

A busca de hiperparâmetros usou `opt_binary_clf_pipe` com o Tree-structured Parzen Estimator (TPE) do Optuna:

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
| `max_features` | `log2` | Amostragem logarítmica de atributos por divisão; introduz diversidade e reduz correlação entre árvores |
| `min_samples_leaf` | 2 | Evita sobre-segmentação permitindo divisões granulares |
| `min_samples_split` | 6 | Controla o crescimento das árvores; regulariza contra overfitting |
| `random_state` | 738 | Seed fixa garante reprodutibilidade completa |
| `class_weight` | `balanced` (implícito) | Balanceamento inerente no nível da árvore; desbalanceamento 73/27 tratado internamente |

Nota de desempenho: o artefato salvo atual alcança 76,4% de Recall Macro no hold-out. Comparações com outputs anteriores dos notebooks devem ser tratadas como históricas até que o notebook seja reexecutado de ponta a ponta.

---

### Fase 5: seleção do threshold

O classificador produz um score contínuo de risco, e o threshold determina a fronteira de classificação. Esse score não é uma probabilidade calibrada: as saídas da Balanced Random Forest não são calibradas, e nenhuma análise de calibração foi feita.

O threshold selecionado foi 0,52, ligeiramente acima do padrão de 0,50. Ele foi escolhido para maximizar o Recall Macro nas predições out-of-fold da CV estratificada de 6 folds no treino, e não por um cálculo de custo. Um threshold baseado em custo exigiria custos reais de intervenção e de perda de cliente, que o dataset não traz.

```
Threshold = 0,50 (padrão)
  → Mudança mínima em relação ao modelo otimizado

Threshold = 0,52 (escolhido por Recall Macro)
  → Recall Macro: 76,4%, Precision Macro: 71,2%
  → Captura 80,5% dos churners reais (301 de 374)
```

A escolha do threshold depende do contexto de uso e não é apenas um detalhe técnico (Fawcett, 2006).

---

### Fase 6: estratégia de validação

```python
cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
```

A validação cruzada é estratificada para manter os 26,5% de churn em cada fold, usa 6 folds para reduzir a variância em classificação desbalanceada e fixa o estado aleatório para garantir reprodutibilidade entre execuções.

#### Métricas apropriadas para dados desbalanceados

| Métrica | Fórmula | Aplicação | Por que é importante |
|--------|---------|-------------|----------------|
| Recall Macro (equivalente à acurácia balanceada) | TP/(TP+FN) médio por classe = (Sensibilidade+Especificidade)/2 | Minimizar churners não identificados | Falsos negativos são mais caros; não enviesada pelo desbalanceamento |
| Precision Macro | TP/(TP+FP) médio | Gerenciar custos de retenção | Custo dos falsos positivos |
| ROC AUC | Área sob a curva | Independente do threshold | Discriminação geral |
| Acurácia | (TP+TN)/N | Inadequada aqui | Enganosa em dados desbalanceados |

Com uma única classe positiva, o Recall Macro e a acurácia balanceada são a mesma quantidade. A acurácia é inadequada para classificação desbalanceada (He & Garcia, 2009), e por isso o trabalho enfatiza o Recall Macro.

---

## Desempenho e resultados do modelo

### Métricas gerais (conjunto de teste, n≈1.407)

A tabela abaixo compara o campeão com uma regressão logística de referência (one-hot, padronização das variáveis numéricas, `class_weight="balanced"`, threshold 0,50 sem ajuste), treinada na mesma divisão 80/20 com semente 42 e avaliada nos mesmos 1.407 clientes de teste (374 churners).

| Métrica | Vencedor (BRF) | Baseline (LR) | Diferença (BRF − LR) |
|--------|--------|---------------|---|
| Recall Macro (equivalente à acurácia balanceada) | 76,4% | 74,8% | +1,6 p.p. |
| Precision Macro | 71,2% | 69,8% | +1,4 p.p. |
| ROC AUC | 83,5% | 83,5% | 0,0 p.p. (IC95% bootstrap −1,0 a +1,0) |
| Gini (2 × AUC − 1) | 0,671 | 0,670 | +0,001 |
| KS | 0,535 | 0,509 | +0,026 |
| PR AUC (precisão média) | 0,635 | 0,618 | +0,017 |
| Brier score (menor é melhor) | 0,172 | 0,174 | −0,001 |
| Precisão no decil de maior score | 73,0% | 69,5% | +3,5 p.p. |
| Lift no decil de maior score | 2,75 | 2,61 | +0,14 |
| Churners capturados no decil de maior score | 27,5% | 26,2% | +1,3 p.p. |

A regressão logística empata com o campeão na discriminação: a diferença de ROC AUC é nula, com intervalo de confiança bootstrap (2.000 reamostragens do teste) entre −1,0 e +1,0 ponto percentual. As vantagens do BRF em KS, PR AUC e lift no primeiro decil são pequenas e não foram testadas formalmente, e a diferença de Recall Macro mistura o modelo com o threshold, que só foi ajustado no BRF. O score médio dos dois modelos no teste (0,42) fica bem acima da taxa base de churn (0,266), o que confirma que nenhum dos dois produz probabilidades calibradas; o balanceamento de classes desloca os scores para cima. Para um uso que dependa de interpretabilidade, a regressão logística entrega a mesma ordenação de risco com um modelo mais simples. A comparação é reproduzida pelo script `evaluate_baseline.py`, executado nesta pasta com o dataset em `dataset/telco.csv` (`python evaluate_baseline.py`).

### Análise da matriz de confusão

```
Distribuição das previsões (conjunto de teste):
  Corretas (TN + TP):  1.048
  FP (alarme falso):     286    ← não churners sinalizados (custo de retenção desnecessário)
  FN (churner não identificado): 73    ← churners não identificados (erro mais caro)

Derivados:
  Verdadeiros positivos (TP): 301   churners identificados corretamente
  Falsos negativos (FN):       73   churners não identificados
  Falsos positivos (FP):      286   não churners sinalizados incorretamente
  Verdadeiros negativos (TN): 747   não churners dispensados corretamente
```

O recall da classe churner é 301 / (301+73) = 80,5%, e o da classe não churner é 747 / (747+286) = 72,3%. Os 301 verdadeiros positivos são churners identificados corretamente, e os 73 falsos negativos são churners não identificados. Os 286 falsos positivos são não churners sinalizados, e a aceitabilidade desse custo exige a economia real da intervenção. Os 747 verdadeiros negativos são não churners identificados corretamente.

### Importância dos atributos (permutation importance, scorer Recall Macro)

Os cinco principais fatores de churn no conjunto de teste:

```
1. Contract              ← Principal fator generalizável
2. InternetService       ← Preditor consistentemente importante (treino + teste)
3. tenure               ← Padrão clássico de churn: clientes novos cancelam mais
4. TotalCharges         ← Forte no treino, moderado no teste
5. OnlineSecurity       ← Importância moderada e estável
```

Os atributos de baixo impacto, candidatos à remoção, são `StreamingTV`, `StreamingMovies`, `PhoneService`, `gender` e `Partner`. A estrutura do contrato é a principal alavanca associada à redução do churn, consistente com os achados da análise exploratória (φK=0,45).

### Análise de erros

Os churners não identificados (FN) têm tenure mediano de 35 meses (contra 6 meses para os churners identificados) e MonthlyCharges mediano de US$ 67 (contra US$ 79). Esses clientes se parecem com não churners nos atributos superficiais, com tenure maior e cobranças moderadas, e são os casos mais difíceis para qualquer modelo de churn.

Distribuição de contratos nos erros:

| Contrato | FN % | FP % |
|----------|------|------|
| Mês a mês | 40,3% | 93,4% |
| Um ano | 48,6% | 6,6% |
| Dois anos | 11,1% | 0,0% |

---

## Impacto de negócio e preservação de receita

### Contexto de receita

Sem modelo (baseline de 2% de churn mensal):

```
Perda anual por churn não gerenciado: ≈US$ 12M sobre US$ 50M de ARR
```

### Cenário ilustrativo usando o recall do hold-out (não é impacto observado)

O cenário abaixo é ilustrativo, baseado em premissas e não causal. Ele assume a escala de uma campanha sobre o dataset completo e não é resultado de campanha medido. Custo de intervenção e LTV são entradas assumidas, não custos observados, e a premissa de 60% de sucesso de retenção deve ser testada prospectivamente. Qualquer valor de preservação de receita é, portanto, um cenário e não um resultado.

Base: no threshold do hold-out (0,52) o modelo captura 80,5% dos churners (301 de 374) e sinaliza 27,7% dos não churners (286 de 1.033). Escala: ≈7.043 clientes (≈1.866 churners). Entradas assumidas: US$ 250 por intervenção, LTV de US$ 3.500 por churner, 60% de sucesso de retenção.

Nenhum ROI observado é reportado. As premissas devem ser substituídas por resultados de experimento controlado antes do uso de negócio.

### Comparação entre modelo e ausência de modelo

| Cenário | Churners perdidos | LTV perdido | Custo de intervenção |
|----------|----------------|----------|-------------------|
| Sem modelo | ~1.866 | ≈US$ 6,5M | US$ 0 |
| Com modelo | ≈965 (≈364 não identificados + ≈601 sinalizados e não retidos) | ≈US$ 3,4M | ≈US$ 734K (≈2.935 sinalizados) |
| Diferença ilustrativa | ≈901 churners retidos | ≈US$ 3,2M preservados | ≈4,3x (LTV preservado / custo de intervenção) |

---

## Referências

He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data." *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. Citada para a escolha de métricas adequadas a dados desbalanceados, quando a acurácia falha.

Guyon, I., & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." *Journal of Machine Learning Research*, 3, 1157–1182. Citada como base da seleção de atributos multimétrica (Chi², ANOVA, PhiK, SelectFromModel).

Fawcett, T. (2006). "An Introduction to ROC Analysis." *Pattern Recognition Letters*, 27(8), 861–874. Citada para a seleção do threshold como decisão dependente do contexto de uso.

Ke, G., et al. (2017). "LightGBM: A Fast, Distributed, Gradient Boosting Framework." *NeurIPS*. Referenciada durante a seleção do modelo; o LightGBM foi um candidato forte.

---

## Limitações e transparência

### Restrições conhecidas

Snapshot temporal: o dataset representa um único ponto no tempo (~2019), e o modelo exigiria retreinamento trimestral ou semestral.

Atributos comportamentais ausentes: não há histórico de interações (tickets de suporte, padrões de uso de funcionalidades) nem métricas de satisfação do cliente (NPS, CSAT). Os churners silenciosos (tenure alto, cobranças moderadas) são os mais difíceis de identificar, como confirma a análise de FN.

Correlação não é causalidade: cobranças mensais altas se correlacionam com churn, sem que isso as torne sua causa. As premissas causais devem ser validadas com testes A/B antes de grandes mudanças de preço.

Generalização por segmento: o modelo foi treinado especificamente em clientes de telecomunicações, e novos segmentos de clientes exigem retreinamento.

Premissa de sucesso da intervenção: o cenário de ROI assume 60% de sucesso de retenção. Um piloto com 1.000 clientes mediria resultados reais antes de um rollout completo.

Volume de falsos positivos: 286 dos 1.033 não churners foram sinalizados (taxa de FP de 27,7%). Se o orçamento de retenção for limitado, pode ser necessário ajustar o threshold.

### O que o modelo pode e não pode fazer

No hold-out, o modelo identifica 80,5% dos churners (recall por classe); o lead time não está estabelecido. Ele permite priorizar o investimento de retenção em segmentos de alto risco, informar estratégias de contrato e preço e subsidiar cenários ilustrativos de receita e ROI com premissas explícitas.

O modelo não garante sucesso de retenção, que depende da qualidade da campanha. Ele também não explica mecanismos causais de churn (apenas correlações), não funciona de forma efetiva sem retreinamento periódico e não identifica com confiabilidade churners de longo tenure e cobrança moderada, padrão de FN confirmado.

---

## Próximos passos

Os próximos passos realistas não exigem dados novos além dos já descritos. A calibração das probabilidades permitiria interpretar os scores como probabilidades. Um threshold baseado em custo exigiria custos reais de retenção e de perda de cliente. Modelos específicos por segmento (mês a mês e contratos longos) podem ser comparados com o modelo único. Se houver dados com carimbo de tempo, a validação out-of-time substituiria o hold-out aleatório. A modelagem de uplift depende de dados de um experimento de retenção.

---

## Stack técnica

O projeto usa Python 3.10+ e Jupyter Notebook (análise interativa e documentação). O processamento de dados e o ML usam Pandas e NumPy, Scikit-learn (pipelines de pré-processamento e avaliação), imbalanced-learn (BalancedRandomForestClassifier) e category_encoders (SumEncoder e outros encoders). A otimização e o rastreamento de experimentos usam opt_binary_clf_pipe (biblioteca customizada para o ciclo completo de experimentação), Optuna (otimização bayesiana de hiperparâmetros) e MLflow (rastreamento de experimentos e registro de modelos). A análise estatística usa PhiK (correlação para variáveis de tipos mistos) e SciPy (testes estatísticos), e a visualização usa Matplotlib e Seaborn. A API usa FastAPI e Uvicorn, e os testes usam pytest com GitHub Actions CI.

---

## Autor e notas da metodologia

Autor: Enio Rubens

Função: Data Science & Analytics

Ferramentas assistidas por IA foram usadas na otimização de código, na estruturação da documentação e na validação analítica. O enquadramento do problema de negócio, as decisões metodológicas, a interpretação dos resultados e as recomendações foram conduzidos pelo autor. Todas as decisões analíticas, interpretações de negócio, escolhas de modelagem, procedimentos de validação e integração do projeto foram conduzidos e revisados pelo autor.

---

## Formato de citação

```bibtex
@misc{rubens2026churn,
  author = {Rubens, Enio},
  title = {Customer Churn Prediction: Reproducible End-to-End ML Project},
  year = {2026},
  note = {Portfolio Project}
}
```

---

## Licença e agradecimentos

Dataset: [Telecom Customer Churn (Kaggle)](https://www.kaggle.com/datasets/puja19/telecom-customer-churn), licença CC0.

Bibliotecas: comunidades de desenvolvimento de scikit-learn, imbalanced-learn, Optuna, Pandas e NumPy.

Pesquisa: artigos citados de He & Garcia, Guyon & Elisseeff, Fawcett e Ke et al.

---

Última atualização: outubro de 2026

Status do projeto: artefato exploratório de portfólio

Desempenho do modelo: 76,4% Recall Macro (equivalente à acurácia balanceada) | 83,5% ROC AUC | ROI não validado

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11+-orange.svg)](https://imbalanced-learn.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

Projeto de portfólio de ciência de dados sobre predição de churn de clientes.
