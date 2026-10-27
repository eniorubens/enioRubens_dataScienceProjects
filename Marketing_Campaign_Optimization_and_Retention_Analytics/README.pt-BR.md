# Otimização de Campanhas de Marketing e Analytics de Retenção

[Relatório executivo PT-BR](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) · [Relatório em inglês](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/en.html)

[Notebook PT-BR](notebooks/Marketing_Campaign_Optimization_and_Retention_Analytics_PT.ipynb) · [Notebook EN-US](notebooks/Marketing_Campaign_Optimization_and_Retention_Analytics_EN.ipynb)

## 💡 Filosofia do projeto

**Camada de governança:** o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) conecta os achados analíticos às limitações de negócio explícitas.

**Superfície de engenharia:** análise orientada por notebooks com outputs CRM/Excel; este projeto não expõe pacote reutilizável `src/`, API de serving ou suíte automatizada de testes.

**Pode sustentar:** exploração descritiva, inferencial, preditiva e prescritiva de decisões de campanha e retenção.

**Não sustenta:** serviço produtivo de scoring, impacto causal de campanha ou decisões automáticas sem engenharia e validação adicionais.

## Visão geral

Pipeline de analytics de marketing em quatro camadas: análise descritiva, teste A/B, previsão de retenção e recomendação operacional. O projeto organiza conversão, retenção, diagnóstico de erros, simulação financeira e exportação de resultados.

## Resultados resumidos

- 10.037 registros brutos, 9.757 exposições após limpeza e 6.849 usuários estáveis na base canônica.
- A personalização não mostrou uplift no cohort canônico: diferença observada de −0,06 ponto percentual, sem evidência de superioridade.
- O cenário preditivo publicado alcançou ROC-AUC de 0,956.
- ROI, lucro esperado e recomendações usam premissas simuladas e não representam contabilidade financeira observada.

## Métodos

- Funil de conversão, retenção e desempenho por canal.
- Z-test para diferença entre proporções e interpretação de uplift.
- Pipeline supervisionado para retenção com pré-processamento e codificação categórica.
- Diagnóstico de falsos positivos, simulação de ROI e recomendação por valor esperado.

## Limitações

O teste permaneceu exploratório e a base canônica ficou abaixo do tamanho planejado para o lift de interesse. As recomendações exigem custos reais, atribuição clara e experimento controlado antes de uso operacional.

## Estrutura

```text
Marketing_Campaign_Optimization_and_Retention_Analytics/
├── notebooks/ · src/ · tests/
├── site/              # relatório executivo PT-BR/EN-US
└── README.md / README.pt-BR.md
```
