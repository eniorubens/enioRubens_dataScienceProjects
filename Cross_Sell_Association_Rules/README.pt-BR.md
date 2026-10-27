# Identificação de Oportunidades de Cross-Sell com Regras de Associação

[Relatório executivo PT-BR](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/) · [Relatório em inglês](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/en.html)

[Notebook PT-BR](notebooks/Cross_Sell_Association_Rules_PT.ipynb) · [Notebook EN-US](notebooks/Cross_Sell_Association_Rules_EN.ipynb)

## Visão geral

Market Basket Analysis aplicada a transações de varejo para distinguir itens populares de associações relevantes. A frequência alta, isoladamente, não demonstra que dois produtos estejam associados.

## Métodos

- Agregação por transação, cliente e data.
- One-hot encoding e mineração de conjuntos frequentes com Apriori.
- Geração de regras e avaliação por support, confidence, lift, conviction e métrica de Zhang.
- Filtros por lift e Zhang para destacar associações interpretáveis.

## Resultado e limitações

O notebook materializa regras como sausage → beverages, napkins → pastry, curd → sausage e brown bread → canned beer, com lift acima de 1 nos cenários analisados. Esses padrões são exploratórios: não demonstram causalidade, incrementalidade de vendas nem superioridade de uma ação de cross-sell.

## Estrutura

```text
Cross_Sell_Association_Rules/
├── notebooks/ · dataset/
├── site/              # relatório executivo PT-BR/EN-US
└── README.md / README.pt-BR.md
```
