# Projetos de Data Science

**Autor:** [Enio Rubens](https://github.com/eniorubens)

**Idioma:** [English](README.md) | Português (Brasil)

Um portfólio de projetos completos de data science cobrindo todo o espectro analítico, de estatística descritiva e testes inferenciais a modelagem preditiva e otimização prescritiva. Os projetos incluem notebooks reprodutíveis, código-fonte modular e suítes de testes automatizados; o status de idioma de cada edição é documentado em seu próprio README.

---

## Filosofia do portfólio

Este não é um conjunto de notebooks isolados. Os projetos analíticos combinam enquadramento de negócio, implementação modular, validação baseada em evidências e uma camada executiva de comunicação. Cada relatório explicita o que os resultados sustentam, suas limitações e o caminho de reprodução; a prontidão para deployment é declarada por projeto, nunca presumida.

## Acesso rápido

| Projeto | Domínio | Relatório executivo | Código e testes |
| :--- | :--- | :--- | :--- |
| **Customer Churn Prediction** | Retenção / classificação | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/) | [Repositório](./customer-churn-prediction) |
| **Customer Lifetime Value** | Marketing probabilístico | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/) | [Repositório](./Customer%20Lifetime%20Value) |
| **Customer Segmentation & Next Best Action** | Segmentação / prescrição | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/) | [Repositório](./Customer_Segmentation_and_Next_Best_Action) |
| **Marketing Campaign Optimization** | Pipeline analítico completo | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) | [Repositório](./Marketing_Campaign_Optimization_and_Retention_Analytics) |
| **Cross-Sell Association Rules** | Análise de cesta | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/) | [Repositório](./Cross_Sell_Association_Rules) |
| **Seoul Bike Sharing Demand** | Séries temporais / conformal | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/Bike-Sharing-Demand/) | [Repositório](./Bike-Sharing-Demand) |
| **Marketing Campaign Uplift Modeling** | Targeting causal | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/) | [Repositório](./Marketing_Promotion_Campaign_Uplift_Modelling) |
| **Consumer Complaint Intelligence** | NLP desbalanceado | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/Consumer_Complaint_Intelligence/) | [Repositório](./Consumer_Complaint_Intelligence) |
| **NASA C-MAPSS** | Manutenção preditiva | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/) | [Repositório](./CMAPSS_Predictive_Maintenance) |
| **Model Risk & Responsible ML** | Governança de risco de modelo | [📊 Ver relatório](https://eniorubens.github.io/enioRubens_dataScienceProjects/ModelRisk_ResponsibleML/) | [Repositório](./ModelRisk_ResponsibleML) |
| **ds_toolkit** | Biblioteca reutilizável de ML | [README da biblioteca](./ds_toolkit) | [Código](./ds_toolkit) |

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

Prevê churn de clientes, com taxa base de 26,6%, usando **76,4% de Recall Macro** e **83,5% de ROC AUC** no artefato hold-out atual. Receita preservada e ROI permanecem ilustrativos, sem validação causal.

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/README.pt-BR.md)

**Métodos:** Classificação desbalanceada · BalancedRandomForestClassifier · Otimização de threshold · Correlação PhiK · EDA · Roadmap de implantação em fases de 90 dias

**Tecnologias:** scikit-learn · imbalanced-learn · Optuna · MLflow · FastAPI · pytest · category_encoders · PhiK

---

### 3. [Customer Lifetime Value](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Customer%20Lifetime%20Value) — Modelagem probabilística BTYD
![Status](https://img.shields.io/badge/status-completed-blue)

Estima a receita futura esperada por cliente em uma **janela de 180 dias** usando modelos probabilísticos Buy-Till-You-Die com validação por holdout temporal.

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer%20Lifetime%20Value/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer%20Lifetime%20Value/README.pt-BR.md)

**Métodos:** Engenharia de features RFM · Modelo BG/NBD (frequência de compra e P(alive)) · Modelo Gamma-Gamma (valor monetário) · Ajuste MAP e MCMC · Segmentação de clientes na matriz CLTV × P(alive)

**Tecnologias:** pymc-marketing · Pandas · NumPy · scikit-learn · Flask API · pytest

---

### 4. [Customer Segmentation & Next Best Action](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Customer_Segmentation_and_Next_Best_Action) — Analytics prescritivo
![Status](https://img.shields.io/badge/status-completed-blue)

Pipeline completo de dados transacionais a decisões acionáveis de marketing: **RFM → segmentação com K-Means → probabilidade de recompra → otimização prescritiva com restrição de orçamento**. O projeto reporta simulações de lucro esperado sob premissas explícitas; uplift causal e superioridade sobre uma campanha uniforme não estão estabelecidos.

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer_Segmentation_and_Next_Best_Action/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer_Segmentation_and_Next_Best_Action/README.pt-BR.md)

**Métodos:** Clusterização K-Means (Elbow + Silhouette, k=6) · Classificador probabilístico calibrado · Simulação de incentivos · Cálculo de lucro esperado · Otimizador de orçamento

**Tecnologias:** scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · pytest

---

### 5. [Marketing Campaign Optimization and Retention Analytics](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Marketing_Campaign_Optimization_and_Retention_Analytics) — Pipeline analítico completo
![Status](https://img.shields.io/badge/status-completed-blue)

Um pipeline analítico em quatro camadas — **descritiva → inferencial → preditiva → prescritiva** — para análise de desempenho de campanhas e retenção de usuários, com classificação automatizada de usuários em 6 categorias de ação e exportação para Excel com múltiplas planilhas.

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/README.pt-BR.md)

**Métodos:** Análise de desempenho por canal · Teste Z para A/B testing · Modelagem supervisionada de retenção · Diagnóstico de falsos positivos / viés · Simulação financeira de ROI · Sistema automatizado de recomendações

**Tecnologias:** Python · Pandas · NumPy · scikit-learn · Statsmodels · Matplotlib · OpenPyXL

---

### 6. [Cross-Sell Association Rules](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Cross_Sell_Association_Rules) — Análise de cesta de compras
![Status](https://img.shields.io/badge/status-completed-blue)

Identifica oportunidades genuínas de cross-selling em dados transacionais de varejo usando regras de associação. O insight central: **alta frequência de item ≠ associação relevante** — leite integral e vegetais, os dois itens mais populares, apresentam lift de 0,77, mostrando que popularidade isolada induz a erro. As regras são filtradas por lift > 1 e métrica de Zhang > 0,2 para revelar combinações estatisticamente relevantes.

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Cross_Sell_Association_Rules/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Cross_Sell_Association_Rules/README.pt-BR.md)

**Métodos:** Algoritmo Apriori · Mineração de itemsets frequentes · Avaliação multimétrica (Support, Confidence, Lift, Conviction, métrica de Zhang) · One-hot encoding · Agregação de transações

**Tecnologias:** Python · Pandas · NumPy · MLxtend · Matplotlib · Seaborn

---

### 7. [Seoul Bike Sharing Demand](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Bike-Sharing-Demand) — Previsão temporal de demanda
![Status](https://img.shields.io/badge/status-completed-blue)

Prevê a demanda horária de bicicletas em condições operacionais normais usando nove anos de dados de mobilidade e meteorologia. Um holdout temporal selado confirmou CatBoost com **MAE 1.118,1** e **R² 0,839**. Nos folds de desenvolvimento, a calibração conformal adaptativa elevou a cobertura de intervalos de 90% de **81,0% para 90,1%**, reduzindo a largura média dos intervalos em **25,6%**.

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Bike-Sharing-Demand/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Bike-Sharing-Demand/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Bike-Sharing-Demand/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Bike-Sharing-Demand/README.pt-BR.md)

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

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Promotion_Campaign_Uplift_Modelling/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Promotion_Campaign_Uplift_Modelling/README.pt-BR.md)

**Métodos:** Auditoria de randomização · S/T/X/R-learners · Causal forest · Uplift trees · Qini/AUUC · Holdout selado · Perfis de heterogeneidade por quantil · Policy learning · Avaliação IPW · Sensibilidade de ROI

**Tecnologias:** Python · Pandas · scikit-learn · EconML · CausalML · Matplotlib · pytest · Jupyter

---

### 9. [Consumer Complaint Intelligence](https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Consumer_Complaint_Intelligence) — NLP desbalanceado e avaliação confirmatória
![Status](https://img.shields.io/badge/status-completed-blue)

Constrói um sistema reprodutível de roteamento de reclamações em `en-US` sobre
um snapshot do CFPB com 3,8 milhões de linhas. O projeto avança de auditorias
limitadas com Polars/DuckDB e deduplicação temporal para classificadores
TF-IDF, mineração de hard negatives, detector hierárquico, bundles congelados
de inferência e duas avaliações seladas pré-registradas. Nenhum ciclo foi
promovido retrospectivamente: o S8 passou 2 de 3 gates e o V2.1-C passou 3 de
4; ambos permanecem `NOT_CONFIRMED` e `deploy=false`.

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Consumer_Complaint_Intelligence/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Consumer_Complaint_Intelligence/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Consumer_Complaint_Intelligence/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Consumer_Complaint_Intelligence/README.pt-BR.md)

**Métodos:** NLP multiclasse desbalanceado · TF-IDF por palavras e caracteres · LinearSVC · Regressão logística SGD · Divisão temporal · Deduplicação por grupos · Mineração de hard negatives · Classificação hierárquica · Calibração de threshold · Intervalos bootstrap · Avaliação confirmatória selada

**Tecnologias:** Python · Polars · DuckDB · scikit-learn · imbalanced-learn · joblib · Kaggle · Jupyter · unittest

---

### 10. [NASA C-MAPSS — Manutenção preditiva](CMAPSS_Predictive_Maintenance/README.pt-BR.md) — Vida útil restante

Estima a vida útil restante de motores simulados do FD001 com histórico causal dos sensores. Um pipeline Extra Trees selecionou 93 de 205 atributos e registrou **MAE de 16,23 ciclos** e **RMSE de 23,80 ciclos** no benchmark oficial de 100 motores. A escolha final ocorreu após essa comparação; o resultado não é uma avaliação independente adicional após a seleção. Alertas, custos hipotéticos e faixas empíricas de incerteza permanecem exploratórios, sem validação industrial.

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/CMAPSS_Predictive_Maintenance/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/CMAPSS_Predictive_Maintenance/README.pt-BR.md)

**Métodos:** Validação por motor · Features móveis causais · Optuna · Seleção de atributos · Diagnóstico de resíduos · Importância por permutação · Alertas e custos ilustrativos de manutenção

**Tecnologias:** Python · Pandas · scikit-learn · Extra Trees · XGBoost · HistGradientBoosting · Optuna · Plotly · Jupyter · GitHub Pages

---

### 11. [Model Risk & Responsible ML](ModelRisk_ResponsibleML/README.pt-BR.md) — Auditoria responsável de risco de modelo

Audita calibração, fairness, incerteza e isolamento do teste final sobre o conjunto UCI Adult. O portfólio bilíngue documenta uma avaliação congelada com HistGradientBoosting, threshold 0,31 e 18 testes automatizados de contrato. O resultado é explicitamente exploratório e `production_eligible=false`.

📊 **Relatório executivo:** [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/ModelRisk_ResponsibleML/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/ModelRisk_ResponsibleML/en.html) · README: [English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/ModelRisk_ResponsibleML/README.md) · [Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/ModelRisk_ResponsibleML/README.pt-BR.md)

**Métodos:** Partições group-aware · Calibração Platt · Auditoria de threshold · Intervalo de Wilson para proporções de grupo · Diagnósticos de fairness · Testes de contrato

**Tecnologias:** Python · Pandas · scikit-learn · Matplotlib · Jupyter · unittest · GitHub Pages

---

## Resumo de competências

| Área | Métodos e técnicas |
|------|--------------------|
| **Aprendizado supervisionado** | Random Forest, CatBoost, Gradient Boosting, Logistic Regression, otimização de threshold |
| **Processamento de linguagem natural** | TF-IDF, modelos lineares esparsos, mineração de hard negatives, classificação hierárquica, classificação de texto desbalanceada |
| **Modelagem probabilística** | BG/NBD, Gamma-Gamma, MCMC, MAP |
| **Séries temporais e previsão** | Validação cruzada expansiva, holdout temporal, previsão de demanda, diagnósticos residuais |
| **Quantificação de incerteza** | Inferência conformal adaptativa, cobertura condicional, Winkler Score, bootstrap temporal |
| **Explicabilidade de modelos** | SHAP, importância por permutação, análise de ablação |
| **Aprendizado não supervisionado** | Clusterização K-Means, segmentação RFM |
| **Analytics prescritivo** | Otimização de orçamento, simulação de lucro esperado, Next Best Action |
| **Estatística inferencial** | Teste Z, A/B testing, testes de hipótese |
| **Inferência causal** | Estimação de CATE, meta-learners, causal forests, avaliação de uplift, policy learning |
| **MLOps** | Optuna, MLflow, FastAPI, pytest, GitHub Actions CI |
| **Engenharia de dados** | Polars, DuckDB, pipelines de features, estratégias de encoding, tratamento de dados desbalanceados |
| **Market basket analysis** | Apriori, regras de associação, lift, conviction, métrica de Zhang |

## Tecnologias principais

Python · Pandas · Polars · DuckDB · NumPy · scikit-learn · imbalanced-learn · CatBoost · XGBoost · LightGBM · EconML · CausalML · Optuna · MLflow · SHAP · feature-engine · pymc-marketing · FastAPI · Flask · pytest · Matplotlib · Seaborn · Statsmodels · category_encoders · MLxtend · Jupyter

---

*Português é o idioma canônico dos notebooks recém-desenvolvidos; edições em inglês são publicadas como entregáveis separados do portfólio quando concluídas.*

## Licença

O código é distribuído sob a [Licença MIT](LICENSE). Datasets de terceiros (CFPB, Kaggle, UCI, NASA, Hillstrom/MineThatData) permanecem sob seus termos originais; as fontes estão no README de cada projeto.
