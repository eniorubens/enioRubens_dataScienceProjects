# Segmentação de Clientes e Next Best Action

[🇺🇸 English](notebooks/Customer_Segmentation_and_Next_Best_Action-EN.ipynb) | [🇧🇷 Português](notebooks/Customer_Segmentation_and_Next_Best_Action-PT.ipynb)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/LICENSE)

---

**Leitura:** [Relatório executivo · Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/) · [Relatório em inglês](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/en.html)

## 🎯 Escopo e evidências

**Pode sustentar:** segmentação RFM/K-Means, score exploratório de recompra e simulações de ações sob premissas.

**Não sustenta:** uplift causal, superioridade sobre uma campanha uniforme ou prescrição produtiva automatizada.

**Superfície de engenharia:** módulos reutilizáveis em `src/`, notebooks bilíngues e uma suíte pytest; este projeto não expõe uma API de serving.

**Camada de governança:** o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/) separa os outputs observados em holdout das simulações sob restrição de orçamento.

## Outputs observados e status da simulação

O projeto aplica segmentação, um modelo de recompra e uma simulação prescritiva ao
dataset UCI Online Retail. Os valores abaixo são outputs materializados do notebook
em português; não são resultados causais de negócio.

| Segmento | Clientes | Participação na receita | Ação recomendada |
|---|---:|---:|---|
| **Campeões** | 449 | 56.66% | Tratamento VIP, acesso antecipado e exclusividade |
| **Clientes fiéis** | 1,089 | 19.26% | Cross-sell e combos |
| **Alto valor em risco** | 540 | 15.03% | Reativação prioritária |
| **Compradores ocasionais** | 1,142 | 5.57% | Campanhas de segunda compra de baixo custo |
| **Clientes inativos** | 670 | 2.80% | Reativação seletiva ou baixa prioridade |
| **Baixo valor** | 448 | 0.68% | Automação de marketing de baixo custo |

> **Importante:** os valores de uplift usados pela simulação são premissas
> heurísticas de portfólio. Não são estimativas causais e devem ser calibrados
> contra resultados de testes A/B antes do uso operacional.

**Simulação de orçamento materializada (holdout, limite padrão de $10.000):**
- Orçamento utilizado: **$2.895**
- Clientes exportados para CRM: **353**
- Lucro incremental esperado sob a heurística configurada: **$14.733,53**

A comparação de cenários é sensível ao objetivo: a tabela de cenários materializada
reporta um ganho de **$10.275** para a campanha uniforme contra o baseline,
enquanto o cenário de prescrição baseado em margem é negativo nessa comparação.
Objetivo, margem, custo e população precisam ser reconciliados antes de alegar
superioridade ou prontidão para produção.

---

## O que este projeto faz

Transforma dados transacionais brutos em decisões acionáveis de marketing por meio
de três camadas analíticas:

```
Transações → Atributos RFM → Segmentos (K-Means)
                                       ↓
                         Probabilidade de recompra (supervisionada)
                                       ↓
               Simulação de incentivo → Lucro esperado → Next Best Action
                                       ↓
                            Alocação sob restrição de orçamento → Exportação CRM
```

Inspirado em *From Predictive to Prescriptive Analytics* (Bertsimas & Kallus, 2019).

---

## Validação dos clusters

O número ideal de segmentos (k = 6) foi selecionado usando dois métodos complementares:

- **Método do cotovelo:** a curva de inércia apresenta uma inflexão clara entre k = 5 e k = 7
- **Silhouette Score:** atinge o pico em k = 6 (pontuação ≈ 0,38), confirmando a coesão dos segmentos

Ambos os gráficos são renderizados no notebook. A função `search_k()` em
`src/segmentation.py` torna simples repetir esta análise com qualquer intervalo de k.

---

## Estrutura do projeto

```
customer_segmentation_nba/
│
├── src/
│   ├── __init__.py
│   ├── config.py          # Todos os parâmetros operacionais em um só lugar
│   ├── multilang.py       # Internacionalização EN/PT (estática, sem chamadas de API)
│   ├── data.py            # load_raw(), clean(), build_rfm()
│   ├── segmentation.py    # KMeansClusterAdder, search_k(), segment_names()
│   ├── prediction.py      # Fábricas de pipeline, evaluate_classifier(), calibrate()
│   ├── prescriptive.py    # simulate_actions(), select_best_action(), budget_optimizer()
│   └── viz.py             # Tema corporativo + funções de gráficos reutilizáveis
│
├── tests/
│   └── test_suite.py      # Mais de 30 testes unitários e de integração
│
├── notebooks/
│   ├── Customer_Segmentation_and_Next_Best_Action-EN.ipynb
│   └── Customer_Segmentation_and_Next_Best_Action-PT.ipynb
│
├── data/
│   └── online_retail.csv
│
├── notebooks/outputs/       # exportação CRM gerada; artefato local, não payload público
│
├── .github/workflows/test.yml
├── pyproject.toml
└── README.md
```

---

## Início rápido

```bash
# 1. Clonar e instalar
git clone https://github.com/eniorubens/customer-segmentation-nba.git
cd customer-segmentation-nba
pip install -e ".[dev]"

# 2. Executar testes
pytest tests/ -v --cov=src

# 3. Abrir o notebook
jupyter notebook notebooks/Customer_Segmentation_and_Next_Best_Action-EN.ipynb
```

---

## Exemplo de uso

```python
from src.multilang import set_language
from src.data import load_raw, clean, build_rfm
from src.segmentation import search_k, segment_names, KMeansClusterAdder
from src.prediction import make_repurchase_pipeline, evaluate_classifier
from src.prescriptive import simulate_actions, select_best_action, build_recommendation
from src.viz import set_corporate_theme

# Alternar o idioma uma vez — todos os gráficos e saídas de print() acompanham
set_language("pt")   # ou "en" (padrão)
set_corporate_theme()

# Pipeline de dados
raw = load_raw("data/online_retail_II.csv")
df  = clean(raw)
rfm = build_rfm(df)

# Segmentação
from sklearn.preprocessing import StandardScaler
import numpy as np
x = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))
k_metrics = search_k(x, k_range=range(2, 10))

# Camada prescritiva
rfm["prob_repurchase_90d"]  = fitted_pipeline.predict_proba(X_holdout)[:, 1]
rfm["PredictiveSegment"]    = rfm["Cluster"].map(segment_map)

simulation   = simulate_actions(rfm)
best_actions = select_best_action(simulation)
best_actions["PrescriptiveNextBestAction"] = best_actions.apply(build_recommendation, axis=1)
```

---

## Sistema Multilang

Os notebooks são gerados a partir de uma única base de código. Todas as strings
visíveis ao usuário são escritas em inglês e passam por `t()` antes da renderização:

```python
from src.multilang import set_language, t

set_language("pt")
ax.set_ylabel(t("Customers"))          # → "Clientes"
ax.set_title(t("Revenue by Segment"))  # → "Receita por Segmento"
print(t("RFM table: {} customers").format(n))  # → "Tabela RFM: N clientes"
```

**Decisão de design:** as traduções são armazenadas como um dicionário Python
estático em `src/multilang.py`. Não há chamadas de API em tempo de execução,
dependência de rede nem latência durante a execução dos notebooks. Para adicionar
um novo idioma, acrescente uma nova chave a `_TRANSLATIONS`.

---

## Principais decisões de design

| Decisão | Justificativa |
|---|---|
| `KMeansClusterAdder` dentro de `Pipeline` | Impede que o vazamento de rótulos de cluster da validação/holdout alcance o treinamento |
| `CalibratedClassifierCV` | Garante probabilidades bem calibradas para uso em cálculos de lucro |
| Dicionário multilang estático | Sem chamadas de API em tempo de execução; traduções disponíveis offline; fácil auditoria |
| Uplifts heurísticos reconhecidos explicitamente | Honestidade sobre as premissas da simulação; estabelece estrutura para substituição futura por A/B |
| `config.py` como fonte única de parâmetros | Permite repetir qualquer cenário editando um único arquivo |

---

## Conceitos aplicados

Análise RFM · Segmentação de clientes · Clusterização K-Means · Método do cotovelo · Silhouette Score ·
Analytics preditivo · Analytics prescritivo · Otimização do valor esperado ·
Otimização sob restrição de orçamento · Priorização por ROI · Calibração de probabilidade ·
Analytics de marketing · Ciência da decisão · Validação temporal · Engenharia modular de ML

---

## Melhorias futuras

- **Modelagem de uplift** — substituir uplifts heurísticos por uma abordagem de dois modelos ou meta-learner
- **Inferência causal** — aplicar DID ou PSM quando houver dados de teste A/B
- **Framework de teste A/B** — validar experimentalmente as recomendações de Next Best Action
- **MLflow** — rastrear experimentos e registrar modelos
- **Otimização dinâmica de orçamento** — formulação de programação inteira (PuLP / OR-Tools)
- **Scoring em tempo real** — endpoint FastAPI consumindo o pipeline ajustado

---

## Referências

- Bertsimas, D., and Kallus, N. (2019). *From Predictive to Prescriptive Analytics*. Management Science. https://doi.org/10.1287/mnsc.2018.3253
- Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- Fader, P. S., Hardie, B. G. S., and Lee, K. L. (2005). *RFM and CLV: Using Iso-Value Curves for Customer Base Analysis*. Journal of Marketing Research, 42(4), 415–430.

---

**Autor:** Enio Rubens — Data Science & Analytics

> Partes deste projeto foram desenvolvidas com auxílio de ferramentas de
> codificação e escrita baseadas em IA para refinamento de código, estruturação,
> documentação e brainstorming analítico. Todas as decisões analíticas,
> interpretações de negócio, escolhas de modelagem, procedimentos de validação e
> integração final do projeto foram conduzidos e revisados pelo autor.
