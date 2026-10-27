# Customer Lifetime Value Prediction

[Relatório executivo PT-BR](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/) · [Relatório em inglês](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/en.html)

[Notebook PT-BR](notebooks/Customer_Lifetime_Value_PT.ipynb) · [Notebook EN-US](notebooks/Customer_Lifetime_Value_EN.ipynb)

## 💡 Filosofia do projeto

**Camada de governança:** o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Customer%20Lifetime%20Value/) documenta o holdout temporal, as premissas e as decisões sustentadas.

**Produção orientada a contratos:** nos projetos com superfície de serving, APIs FastAPI/Flask expõem contratos de entrada e saída validados por testes. Nos demais, contratos de dados, artefatos, partições seladas e gates de avaliação controlam o que pode ser concluído, reproduzido e publicado.

**Superfície de engenharia:** módulos reutilizáveis em `src/`, API Flask em `serve_model.py`, fluxo de modelo persistido e suíte pytest.

**Pode sustentar:** estimativa exploratória de valor de vida em 180 dias e segmentação para priorização analítica.

**Não sustenta:** garantia de receita futura, causalidade sobre comportamento ou uso produtivo sem monitoramento e revalidação temporal.

## Visão geral

Este projeto estima a receita futura esperada por cliente em um horizonte de 180 dias usando modelos probabilísticos Buy-Till-You-Die. A análise usa histórico transacional, separação temporal entre desenvolvimento e holdout e uma leitura explícita das limitações de extrapolar comportamento passado.

## Métodos

- Engenharia de atributos RFM: recência, frequência e valor monetário.
- Modelo BG/NBD para frequência de compra e probabilidade de o cliente permanecer ativo.
- Modelo Gamma-Gamma para valor monetário esperado.
- Validação temporal e comparação entre estimativas pontuais e observadas.
- Implementação modular com testes e possibilidade de servir previsões por API.

## Interpretação

O resultado é uma estimativa forward-looking, não receita realizada nem garantia individual de compra. As decisões de relacionamento devem considerar margem, acessibilidade, custos de contato e monitoramento de drift.

## Estrutura

```text
Customer Lifetime Value/
├── notebooks/       # análises PT-BR e EN-US
├── src/              # lógica reutilizável
├── tests/            # contratos e validações
├── site/             # relatório executivo estático
└── README.md / README.pt-BR.md
```

O relatório HTML resume os resultados publicados; os notebooks permanecem como fonte executável da análise.
