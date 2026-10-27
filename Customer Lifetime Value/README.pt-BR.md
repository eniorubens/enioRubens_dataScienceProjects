# Previsão de Customer Lifetime Value

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer%20Lifetime%20Value/notebooks/Customer_Lifetime_Value_EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer%20Lifetime%20Value/notebooks/Customer_Lifetime_Value_PT.ipynb)

**Leitura:** [Relatório executivo · Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/) · [Relatório em inglês](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/en.html)

> **Estimativa prospectiva de receita no nível individual do cliente —
> não médias históricas, mas previsões probabilísticas de valor futuro.**

---

## 💡 Filosofia do projeto

**Camada de governança:** o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/) documenta o holdout temporal, as premissas e as decisões sustentadas.

**Produção orientada a contratos:** nos projetos com superfície de serving, APIs FastAPI/Flask expõem contratos de entrada e saída validados por testes. Nos demais, contratos de dados, artefatos, partições seladas e gates de avaliação controlam o que pode ser concluído, reproduzido e publicado.

**Superfície de engenharia:** módulos reutilizáveis em `src`, uma API Flask em `serve_model.py`, fluxo de modelo persistido e uma suíte pytest.

**Pode sustentar:** estimativa exploratória de customer lifetime value em 180 dias e segmentação para priorização analítica.

**Não sustenta:** garantia de receita futura, alegações causais sobre o comportamento dos clientes ou uso produtivo sem monitoramento e revalidação temporal.

## Visão geral

Este projeto constrói um pipeline reprodutível de CLTV (Customer Lifetime Value)
usando modelos probabilísticos BTYD sobre dados transacionais de e-commerce. Ele estima,
para cada cliente, com que frequência comprará, quanto gastará por
transação e qual a probabilidade de ainda estar ativo — e então combina essas
estimativas em uma previsão de receita para 180 dias e uma ação de marketing
priorizada por cliente.

A saída não é um rótulo de segmento. É um sinal auditável de alocação
no nível individual para equipes de marketing, vendas e finanças.

---

## Problema de negócio

Nem todos os clientes têm o mesmo valor futuro. A alocação uniforme de orçamento desperdiça
investimento em clientes de baixo retorno e deixa de investir o suficiente em clientes
de alto valor. O modelo responde a três perguntas que orientam decisões de alocação:

- **Marketing:** onde concentrar o investimento de campanha para maximizar o retorno esperado?
- **Vendas:** quais contas a equipe deve gerenciar ativamente em vez de deixá-las no autosserviço?
- **Finanças:** qual é a projeção defensável de receita para os próximos dois trimestres,
  desagregada por risco de retenção?

---

## Resultados (Online Retail UCI, horizonte de 180 dias)

| Métrica | Valor |
|---|---|
| Receita total esperada (180 dias) | GBP 99,751,217 |
| 10% dos clientes no topo — participação na receita | 40.1% |
| 20% dos clientes no topo — participação na receita | 54.2% |
| Clientes na base de modelagem | 2,762 |

**Distribuição por segmento:**

| Segmento | Clientes | Receita esperada | Participação |
|---|---|---|---|
| Reter e desenvolver | 1,000 | GBP 66,937,937 | 67.1% |
| Prioridade de reativação | 381 | GBP 12,853,917 | 12.9% |
| Baixa prioridade | 1,000 | GBP 13,358,174 | 13.4% |
| Nutrição de baixo custo | 381 | GBP 6,601,190 | 6.6% |

Reter e desenvolver + Prioridade de reativação = **80% da receita esperada de 56% dos clientes.**

---

## Metodologia

### 1 — Preparação dos dados

O conjunto [Online Retail UCI](https://archive.ics.uci.edu/dataset/352/online+retail)
contém 541.909 linhas brutas de itens de fatura de um varejista de e-commerce
do Reino Unido (dez. 2010 – dez. 2011). O pipeline de limpeza remove quatro categorias de
registros problemáticos:

| Etapa | Ação | Linhas removidas |
|---|---|---|
| CustomerID nulo | Sessões anônimas — sem atribuição individual | ~135.080 |
| Cancelamentos | InvoiceNo iniciado por 'C' | ~8.905 |
| Quantidade/preço inválidos | Quantity ou UnitPrice não positivos | ~40 |
| Limite de outliers | TotalPrice limitado ao percentil 99,5 | ~1.923 |

Resultado: ~4.309 clientes na matriz RFM; 2.762 na base de modelagem.

### 2 — Matriz RFM

O histórico de compras de cada cliente é comprimido em quatro números que correspondem
diretamente às premissas generativas dos modelos BTYD:

- **frequency** — compras repetidas após a primeira transação
- **recency** — dias entre a primeira e a compra mais recente
- **T** — idade do cliente desde a primeira compra até o fim da janela de observação
- **monetary_value** — receita média por transação repetida

### 3 — Validação dos modelos

A validação usa uma **divisão temporal de calibração / holdout** (75% / 25%),
e não validação cruzada k-fold. O k-fold embaralha o tempo — compras futuras
vazam para o fold de treinamento, inflando todas as métricas de validação e
produzindo um modelo que falha imediatamente em um deployment real.

### 4 — Modelo BG/NBD *(Fader, Hardie & Lee, 2005)*

Estima, para cada cliente:
- Número esperado de transações futuras em uma determinada janela de tempo
- Probabilidade de ainda estar ativo (`P(alive)`)

O modelo combina um processo de compra de Poisson (com heterogeneidade
distribuída como Gamma entre clientes) e um mecanismo de abandono
Beta-Geometric (churn permanente com probabilidade distribuída como Beta entre clientes).

**Parâmetros ajustados (MAP, Online Retail):**

| Parâmetro | Valor | Interpretação |
|---|---|---|
| `r` | 1.5829 | Heterogeneidade da taxa de compra — cauda direita pronunciada |
| `α` | 85.0371 | Escala da taxa — taxa média de compra ≈ 0.0186/dia (~1 a cada 54 dias) |
| `a` | 0.0294 | Forma Beta α — probabilidade de churn |
| `b` | 0.9707 | Forma Beta β — abandono médio ≈ 2,9% por transação |

### 5 — Modelo Gamma-Gamma *(Fader & Hardie, 2013)*

Estima o valor monetário esperado por transação futura usando shrinkage
bayesiano: as estimativas de gasto individual são aproximadas da média
populacional proporcionalmente à pouca quantidade de dados disponível para aquele cliente.
Um cliente com 2 transações é corrigido substancialmente; um com 50 é
quase não ajustado.

**Pré-requisito:** frequência de compra e valor monetário devem ser independentes
entre os clientes. Testado empiricamente por meio da correlação de Pearson —
`|r| = 0.134 < 0.3` ✓

**Parâmetros ajustados (MAP, Online Retail):**

| Parâmetro | Valor | Interpretação |
|---|---|---|
| `p` | 1.9806 | Forma da distribuição de gastos |
| `q` | 4.1655 | Forma da escala no nível populacional |
| `v` | 621.07 | Escala — gasto médio implícito ≈ p×v/q ≈ £295 |

### 6 — CLTV e segmentação

```
CLTV(t) = E[transações em t dias | BG/NBD] × E[gasto por transação | Gamma-Gamma]
```

Os clientes são segmentados em duas dimensões ortogonais:

- **Quartil de CLTV** (Baixo / Médio / Alto / Topo de valor) — quanto vale este cliente?
- **Limiar de P(alive)** (mediana da população modelada) — este cliente ainda pode ser alcançado?

| CLTV | P(alive) | Segmento | Ação recomendada |
|---|---|---|---|
| Alto | Alto | Reter e desenvolver | Incentivos de lealdade, gestão de conta |
| Alto | Baixo | Prioridade de reativação | Campanhas de recuperação — a janela está se fechando |
| Baixo | Alto | Nutrição de baixo custo | Fluxos automatizados, desenvolvimento de frequência |
| Baixo | Baixo | Baixa prioridade | Apenas retenção passiva |

---

## Arquitetura do projeto

```
Customer Lifetime Value/
├── src/
│   ├── preprocessing.py   # OnlineRetailPreprocessor
│   ├── cltv_model.py      # CLTVModel — wrapper BG/NBD + Gamma-Gamma
│   ├── segmentation.py    # CustomerSegmenter + matriz de decisão
│   ├── evaluation.py      # CLTVEvaluator — validação holdout + teste de Pearson
│   └── utils.py           # logging, helpers de formatação de gráficos
├── tests/                 # 43 testes — pytest, fixtures de dados sintéticos
├── notebooks/
│   ├── Customer_Lifetime_Value_EN.ipynb
│   └── Customer_Lifetime_Value_PT.ipynb
├── train_model.py         # CLI — --data, --output, --fit-method, --t
├── serve_model.py         # API Flask — /health /model/info /predict/*
└── models/                # bgm_model.nc + gg_model.nc + metadata.json
```

### Cobertura de testes

```
src/__init__.py       100%
src/evaluation.py      97%
src/preprocessing.py   95%
src/segmentation.py    95%
src/cltv_model.py      89%
src/utils.py           70%
─────────────────────────
TOTAL                  91%   (43 aprovados, 0 avisos)
```

### Persistência dos modelos

Os modelos são serializados por meio de `idata.to_netcdf()` / `az.from_netcdf()` (ArviZ).
**joblib e pickle não são usados** — objetos PyMC não são seguros para pickle.

---

## API Flask

| Endpoint | Método | Descrição |
|---|---|---|
| `/health` | GET | Status do modelo |
| `/model/info` | GET | Parâmetros MAP e metadados do ajuste |
| `/predict/single` | POST | Pontua um cliente |
| `/predict/batch` | POST | Pontua uma lista de clientes |

**Resposta de `/predict/single`:**
```json
{
  "customer_id": "17850",
  "predicted_purchases": 2.34,
  "probability_alive": 0.87,
  "expected_spend": 28.10,
  "cltv": 65.77,
  "cltv_segment": "High Value",
  "marketing_action": "Retain and Grow"
}
```

---

## Stack tecnológica

| Camada | Ferramentas |
|---|---|
| Modelagem | `pymc-marketing` (BetaGeoModel, GammaGammaModel) |
| Inferência | MAP (padrão, segundos) / MCMC (posterior completo, opcional) |
| Dados | `pandas`, `numpy` |
| Visualização | `matplotlib`, `seaborn` |
| Serialização | `arviz` (NetCDF) |
| API | `Flask` |
| Testes | `pytest`, `pytest-cov` |
| CI | GitHub Actions |
| Notebooks | EN + PT bilíngues via [`multilang`](../../ds_toolkit/multilang) `LangMap` |

> **Nota:** este projeto migrou da biblioteca `lifetimes` (arquivada em 2023)
> para `pymc-marketing`, sua sucessora ativamente mantida.

---

## Limitações conhecidas

1. **Receita, não margem** — os valores de CLTV são baseados em receita; a lucratividade depende
   do mix de produtos e dos custos de atendimento não disponíveis neste conjunto de dados.
2. **Premissa de churn permanente** — BG/NBD não modela reativação.
   O modelo Pareto/NBD *(Schmittlein, Morrison & Colombo, 1987)* relaxa essa premissa.
3. **Sem covariáveis** — não há dados demográficos, canal de aquisição ou categoria de produto.
4. **Premissa de independência** — Gamma-Gamma exige frequência ⊥ monetary_value.
   Verifique sempre com um teste de Pearson antes de aplicar a um novo conjunto de dados.

---

## Referências

1. Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). "Counting your customers" the easy
   way. *Marketing Science*, 24(2), 275–284.
2. Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary value.
   [brucehardie.com/notes/025](http://www.brucehardie.com/notes/025/)
3. Schmittlein, D. C., Morrison, D. G., & Colombo, R. (1987). Counting your customers:
   Who are they and what will they do next? *Management Science*, 33(1), 1–24.
4. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry.
   *Journal of Database Marketing*, 19(3), 197–208.
5. PyMC-Marketing (2024). [Módulo Customer Lifetime Value](https://www.pymc-marketing.io/en/stable/notebooks/clv/clv_quickstart.html).

---

## Notas de desenvolvimento

Partes deste projeto foram desenvolvidas com auxílio de ferramentas de
codificação e escrita baseadas em IA para refinamento de código, estruturação,
documentação e brainstorming analítico. Todas as decisões analíticas,
interpretações de negócio, escolhas de modelagem, procedimentos de validação e
integração final do projeto foram conduzidos e revisados pelo autor.

**Autor:** Enio Rubens — Data Science & Analytics
