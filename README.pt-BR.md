# Projetos de Data Science

**Autor:** [Enio Rubens](https://github.com/eniorubens)

**Idioma:** [English](README.md) | Português (Brasil)

Um portfólio de projetos completos de data science cobrindo todo o espectro analítico, de estatística descritiva e testes inferenciais a modelagem preditiva e otimização prescritiva. Os projetos incluem notebooks reprodutíveis, código-fonte modular e suítes de testes automatizados; o status de idioma de cada edição é documentado em seu próprio README.

---

## Projetos

### 1. [ds_toolkit](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/ds_toolkit) — Biblioteca reutilizável de ML
![Status](https://img.shields.io/badge/status-active-brightgreen)

Um toolkit Python modular com componentes reutilizáveis compartilhados entre os projetos.

**Módulos principais:**
- `opt_binary_clf_pipe` — Otimizador de pipelines com Optuna para classificação binária (encoding → seleção de features → modelo → threshold)
- `multilang` — Dicionário estático de tradução para saídas bilíngues (EN/PT) em notebooks

**Tecnologias:** Optuna · scikit-learn · imbalanced-learn · category_encoders · MLflow

---

### 2. [Customer Churn Prediction](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/customer-churn-prediction) — Pipeline de ML ponta a ponta
![Status](https://img.shields.io/badge/status-completed-blue)

Prevê churn de clientes, com taxa base de 26,5%, usando **76,4% de Recall Macro**, o que se traduz em uma estimativa de **US$ 3M de receita anual preservada** com **ROI de 4,7x**. Reposiciona ML como um problema de otimização de negócio.

**Métodos:** Classificação desbalanceada · BalancedRandomForestClassifier · Otimização de threshold · Correlação PhiK · EDA · Roadmap de implantação em fases de 90 dias

**Tecnologias:** scikit-learn · imbalanced-learn · Optuna · MLflow · FastAPI · pytest · category_encoders · PhiK

---

### 3. [Customer Lifetime Value](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Customer%20Lifetime%20Value) — Modelagem probabilística BTYD
![Status](https://img.shields.io/badge/status-completed-blue)

Estima a receita futura esperada por cliente em uma **janela de 180 dias** usando modelos probabilísticos Buy-Till-You-Die com validação por holdout temporal.

**Métodos:** Engenharia de features RFM · Modelo BG/NBD (frequência de compra e P(alive)) · Modelo Gamma-Gamma (valor monetário) · Ajuste MAP e MCMC · Segmentação de clientes na matriz CLTV × P(alive)

**Tecnologias:** pymc-marketing · Pandas · NumPy · scikit-learn · Flask API · pytest

---

### 4. [Customer Segmentation & Next Best Action](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Customer_Segmentation_and_Next_Best_Action) — Analytics prescritivo
![Status](https://img.shields.io/badge/status-completed-blue)

Pipeline completo de dados transacionais a decisões acionáveis de marketing: **RFM → segmentação com K-Means → probabilidade de recompra → otimização prescritiva com restrição de orçamento**, alcançando **melhoria de 2,4x no ROI de campanha** em comparação com campanhas uniformes.

**Métodos:** Clusterização K-Means (Elbow + Silhouette, k=6) · Classificador probabilístico calibrado · Simulação de incentivos · Cálculo de lucro esperado · Otimizador de orçamento

**Tecnologias:** scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · pytest · GitHub Actions CI

---

### 5. [Marketing Campaign Optimization and Retention Analytics](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Marketing_Campaign_Optimization_and_Retention_Analytics) — Pipeline analítico completo
![Status](https://img.shields.io/badge/status-completed-blue)

Um pipeline analítico em quatro camadas — **descritiva → inferencial → preditiva → prescritiva** — para análise de desempenho de campanhas e retenção de usuários, com classificação automatizada de usuários em 6 categorias de ação e exportação para Excel com múltiplas planilhas.

**Métodos:** Análise de desempenho por canal · Teste Z para A/B testing · Modelagem supervisionada de retenção · Diagnóstico de falsos positivos / viés · Simulação financeira de ROI · Sistema automatizado de recomendações

**Tecnologias:** Python · Pandas · NumPy · scikit-learn · Statsmodels · Matplotlib · OpenPyXL

---

### 6. [Cross-Sell Association Rules](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Cross_Sell_Association_Rules) — Análise de cesta de compras
![Status](https://img.shields.io/badge/status-completed-blue)

Identifica oportunidades genuínas de cross-selling em dados transacionais de varejo usando regras de associação. O insight central: **alta frequência de item ≠ associação relevante** — leite integral e vegetais, os dois itens mais populares, apresentam lift de 0,77, mostrando que popularidade isolada induz a erro. As regras são filtradas por lift > 1 e métrica de Zhang > 0,2 para revelar combinações estatisticamente relevantes.

**Métodos:** Algoritmo Apriori · Mineração de itemsets frequentes · Avaliação multimétrica (Support, Confidence, Lift, Conviction, métrica de Zhang) · One-hot encoding · Agregação de transações

**Tecnologias:** Python · Pandas · NumPy · MLxtend · Matplotlib · Seaborn

---

### 7. [Seoul Bike Sharing Demand](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Bike-Sharing-Demand) — Previsão temporal de demanda
![Status](https://img.shields.io/badge/status-completed-blue)

Prevê a demanda horária de bicicletas em condições operacionais normais usando nove anos de dados de mobilidade e meteorologia. Um holdout temporal selado confirmou CatBoost com **MAE 1.118,1** e **R² 0,839**. Nos folds de desenvolvimento, a calibração conformal adaptativa elevou a cobertura de intervalos de 90% de **81,0% para 90,1%**, reduzindo a largura média dos intervalos em **25,6%**.

**Métodos:** Validação cruzada expansiva por ano meteorológico · Pré-processamento dinâmico por estimador · Optuna · MLflow · CatBoost · Diagnósticos residuais · SHAP · Inferência conformal adaptativa · Replay operacional

**Tecnologias:** Python · Pandas · scikit-learn · CatBoost · XGBoost · LightGBM · Optuna · MLflow · SHAP · pytest · Jupyter

---

### 8. [Marketing Campaign Uplift Modeling](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Marketing_Promotion_Campaign_Uplift_Modelling) — Targeting causal e policy learning
![Status](https://img.shields.io/badge/status-completed-blue)

Estima quem deve receber um e-mail porque o tratamento muda seu resultado,
usando o experimento randomizado Hillstrom. O fluxo bilíngue compara targeting
por propensão, meta-learners, causal forests e uplift trees, depois avalia
políticas com restrição de orçamento e cenários de ROI. Seu resultado
pré-registrado em teste selado é deliberadamente conservador: o candidato
primário de uplift não demonstrou superioridade sobre o baseline, portanto o
projeto recomenda um piloto randomizado prospectivo em vez de deployment.

**Métodos:** Auditoria de randomização · S/T/X/R-learners · Causal forest · Uplift trees · Qini/AUUC · Holdout selado · Perfis de heterogeneidade por quantil · Policy learning · Avaliação IPW · Sensibilidade de ROI

**Tecnologias:** Python · Pandas · scikit-learn · EconML · CausalML · Matplotlib · pytest · Jupyter

---

## Resumo de competências

| Área | Métodos e técnicas |
|------|--------------------|
| **Aprendizado supervisionado** | Random Forest, CatBoost, Gradient Boosting, Logistic Regression, otimização de threshold |
| **Modelagem probabilística** | BG/NBD, Gamma-Gamma, MCMC, MAP |
| **Séries temporais e previsão** | Validação cruzada expansiva, holdout temporal, previsão de demanda, diagnósticos residuais |
| **Quantificação de incerteza** | Inferência conformal adaptativa, cobertura condicional, Winkler Score, bootstrap temporal |
| **Explicabilidade de modelos** | SHAP, importância por permutação, análise de ablação |
| **Aprendizado não supervisionado** | Clusterização K-Means, segmentação RFM |
| **Analytics prescritivo** | Otimização de orçamento, simulação de lucro esperado, Next Best Action |
| **Estatística inferencial** | Teste Z, A/B testing, testes de hipótese |
| **Inferência causal** | Estimação de CATE, meta-learners, causal forests, avaliação de uplift, policy learning |
| **MLOps** | Optuna, MLflow, FastAPI, pytest, GitHub Actions CI |
| **Engenharia de dados** | Pipelines de features, estratégias de encoding, tratamento de dados desbalanceados |
| **Market basket analysis** | Apriori, regras de associação, lift, conviction, métrica de Zhang |

## Tecnologias principais

Python · Pandas · NumPy · scikit-learn · CatBoost · XGBoost · LightGBM · EconML · CausalML · Optuna · MLflow · SHAP · feature-engine · pymc-marketing · FastAPI · Flask · pytest · Matplotlib · Seaborn · Statsmodels · category_encoders · MLxtend · Jupyter

---

*Português é o idioma canônico dos notebooks recém-desenvolvidos; edições em inglês são publicadas como entregáveis separados do portfólio quando concluídas.*
