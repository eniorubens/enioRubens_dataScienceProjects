# Segmentação de Clientes e Next Best Action

[Relatório executivo PT-BR](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/) · [Relatório em inglês](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer_Segmentation_and_Next_Best_Action/en.html)

[Notebook PT-BR](notebooks/Customer_Segmentation_and_Next_Best_Action-PT.ipynb) · [Notebook EN-US](notebooks/Customer_Segmentation_and_Next_Best_Action-EN.ipynb)

## Visão geral

O projeto transforma transações do Online Retail II em segmentação RFM, probabilidade de recompra e recomendação prescritiva sob restrição de orçamento. A cadeia é modular: dados → RFM → K-Means → previsão → simulação de incentivos → ação recomendada.

## Resultados materializados

- Seis segmentos interpretáveis, com concentração desigual de receita.
- Simulação padrão com orçamento de `$10.000`, uso de `$2.895` e 353 clientes exportados para CRM.
- Lucro incremental esperado de `$14.733,53` no cenário heurístico configurado.
- A comparação de cenários registra ganho de `$10.275` para a campanha uniforme; a prescrição baseada em margem é negativa nessa comparação. A função objetivo precisa ser reconciliada antes de qualquer alegação de superioridade.

## Métodos

- RFM e K-Means, com escolha de `k = 6` por elbow e silhouette.
- Classificador probabilístico calibrado e holdout temporal.
- Simulação de incentivos, lucro esperado e otimizador de orçamento.
- Exportação de recomendações para CRM.

## Limitações

Os uplifts usados na simulação são premissas heurísticas, não estimativas causais. O projeto não demonstra que a prescrição supera uma campanha uniforme nem está pronto para automatizar ações sem experimento controlado.

## Estrutura

```text
Customer_Segmentation_and_Next_Best_Action/
├── src/ · tests/ · notebooks/ · data/
├── site/              # relatório executivo PT-BR/EN-US
└── README.md / README.pt-BR.md
```
