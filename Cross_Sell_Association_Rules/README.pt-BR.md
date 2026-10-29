# Identificação de Oportunidades de Cross-Sell no Varejo com Regras de Associação

**Leitura:** [Relatório executivo · Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/) · [Relatório em inglês](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/en.html)

## 🎯 Escopo e evidências

**Pode sustentar:** análise exploratória de cesta de compras e hipóteses candidatas de cross-sell.

**Não sustenta:** incrementalidade causal, uplift garantido de conversão ou deployment automático de ofertas.

**Superfície de engenharia:** análise reprodutível em notebook e dataset, mais um pequeno pacote `src/cross_sell_rules` (construção das cestas, métricas de associação, mineração Apriori, seleção de candidatas) coberto por 36 testes pytest que reproduzem os números do relatório executivo. Os dois notebooks chamam esse pacote para leitura, cestas, métricas, mineração e seleção de regras, então a análise e os testes executam o mesmo código. Não há API de serving.

**Camada de governança:** o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/) apresenta as regras como associações comportamentais, não como evidência de intervenção.

## Visão geral

Este projeto aplica Market Basket Analysis para identificar associações entre produtos
em dados transacionais de varejo.

O objetivo principal é distinguir itens comprados com frequência de associações
significativas sustentadas por filtros de lift e Zhang, permitindo decisões mais
confiáveis em estratégias de cross-selling.

---

## Problema de negócio

Datasets de varejo frequentemente contêm produtos muito frequentes, mas frequência
isolada não implica uma associação significativa.

Este projeto se concentra em identificar combinações de produtos que ocorrem com
frequência maior que a esperada, evitando conclusões enganosas baseadas apenas na
popularidade dos itens.

---

## Metodologia

- Agregação dos dados por transação (cliente + data)
- Transformação por one-hot encoding
- Algoritmo Apriori para mineração de conjuntos frequentes
- Geração de regras de associação
- Avaliação usando:
  - Support
  - Confidence
  - Lift
  - Conviction
  - Métrica de Zhang

---

## Principais insights

- Itens de alta frequência não necessariamente indicam associação forte
- Lift é essencial para identificar relações significativas entre produtos
- Muitas combinações frequentes são explicadas pela popularidade dos itens, e não por associação real
- Regras relevantes surgem quando support, confidence e lift são combinados

---

## Aplicações de negócio

- Estratégias de cross-selling
- Combos de produtos
- Otimização da disposição de produtos
- Sistemas de recomendação

---

## Tecnologias

- Python
- Pandas
- MLxtend
- Matplotlib / Seaborn

---

## Estrutura do repositório

```text
Cross_Sell_Association_Rules/
│
├── notebooks/
│   ├── Cross_Sell_Association_Rules_PT.ipynb            # Notebook de análise em português
│   └── Cross_Sell_Association_Rules_EN.ipynb            # Notebook de análise em inglês
├── dataset/Groceries_dataset.csv                        # Dataset de entrada
├── src/cross_sell_rules/                                # Código de análise reutilizável e documentado
│   ├── data.py                                          # Leitura e construção das cestas
│   ├── metrics.py                                       # Confidence, lift, conviction, Zhang
│   ├── mining.py                                        # Apriori, regras, seleção de candidatas
│   └── report.py                                        # Recalcula os números do relatório
├── tests/                                               # Suíte pytest (36 testes)
├── pyproject.toml                                       # Configuração do pacote e dos testes
└── README.md                                            # Documentação do projeto
```

## Testes

A suíte protege os números publicados no relatório executivo (38.765 ocorrências de itens, 14.963 transações, 167 produtos, lift 0,77 e Zhang −0,2625 para whole milk → other vegetables, 216/750 itemsets, 210/648 regras e as 16 candidatas de cross-sell).

```bash
cd Cross_Sell_Association_Rules
pip install -e ".[dev]"
python -m pytest -q
```
## Conclusão

Este projeto demonstra como regras de associação podem ser usadas para extrair
insights acionáveis de dados de varejo.

Ao combinar métricas estatísticas como support, confidence e lift, é possível
diferenciar a simples popularidade de um item de associações significativas entre
produtos, apoiando decisões de negócio mais eficazes.


## Nota de reprodutibilidade

Ambos os notebooks usam o mesmo caminho relativo para o dataset a partir do
diretório `notebooks/` e os mesmos thresholds analíticos. Os outputs salvos foram
verificados quanto à consistência numérica; uma nova execução ainda exige um
ambiente funcional com NumPy, pandas, mlxtend, SciPy, Matplotlib e Seaborn. Os notebooks importam `cross_sell_rules` (instale com `pip install -e .` na pasta do projeto; sem instalar, eles usam `../src`). A ordem das regras é determinística: o mlxtend as devolve numa ordem que muda entre execuções, então o pacote as ordena por antecedente e consequente.
