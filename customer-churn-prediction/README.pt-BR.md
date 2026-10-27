# Customer Churn Prediction — Projeto reprodutível de ponta a ponta

[🇺🇸 English](README.md) · [🇧🇷 Notebook PT-BR](notebooks/Customer-Churn-Prediction-PT.ipynb)

## Resumo executivo

Pipeline modular para priorização exploratória de retenção em clientes de telecomunicações. O artefato treinado atual usa `BalancedRandomForestClassifier`, codificação `SumEncoder`, normalização, escala `MaxAbsScaler`, seleção de atributos e threshold `0,52`.

No hold-out determinístico de 1.407 clientes, o artefato registra:

| Métrica | Resultado |
| --- | ---: |
| Recall macro | **76,6%** |
| Acurácia balanceada | **76,6%** |
| ROC AUC | **84,6%** |
| Precision macro | **71,7%** |

Esses números corrigem as saídas materializadas anteriormente no README e nos notebooks (`76,4%` / `83,5%`). O notebook contém resultados históricos; a página executiva usa o artefato atual e explicita essa diferença.

## O que o projeto sustenta

- comparação interna do pipeline no split hold-out;
- priorização inicial de clientes para investigação de retenção;
- reprodução da inferência por `serve_model.py` e seus endpoints FastAPI;
- separação modular entre dados, features, avaliação e visualização.

## O que não sustenta

- causalidade entre alerta e retenção;
- US$ 3 milhões de receita preservada ou ROI de 4,7x como resultado observado;
- uso produtivo sem validação temporal, monitoramento, custos reais e experimento controlado.

As estimativas financeiras são cenários ilustrativos dependentes de hipóteses de taxa de sucesso, LTV e custo de intervenção. O uso recomendado é de portfólio e investigação, com piloto prospectivo antes de qualquer decisão operacional.

## Estrutura modular

- `churn_project/data.py`: leitura, limpeza e divisão estratificada;
- `churn_project/features.py`: transformações e seleção;
- `churn_project/evaluation.py`: métricas e avaliação;
- `churn_project/viz.py`: gráficos reutilizáveis;
- `train_model.py`: treino e persistência do artefato;
- `serve_model.py`: `/health`, `/predict`, `/batch_predict` e `/model/info`;
- `tests/`: contratos de dados, pipeline e API.

## Relatório e reprodução

- [Relatório executivo PT-BR](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/)
- [Executive report EN-US](https://eniorubens.github.io/enioRubens_dataScienceProjects/customer-churn-prediction/en.html)
- [Notebook PT-BR](notebooks/Customer-Churn-Prediction-PT.ipynb)
- [Notebook EN-US](notebooks/Customer-Churn-Prediction-EN.ipynb)

Dataset: [Telecom Customer Churn — Kaggle](https://www.kaggle.com/datasets/puja19/telecom-customer-churn).
