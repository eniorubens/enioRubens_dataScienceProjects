# Decisão final e análise de erros — FD001

## Decisão metodológica

O benchmark oficial foi congelado antes desta análise. Os três candidatos receberam os mesmos 205 atributos temporais, foram treinados com o mesmo protocolo e avaliados nos mesmos 100 motores do conjunto oficial de teste. Nenhum hiperparâmetro foi ajustado usando esse conjunto.

O Extra Trees foi escolhido como modelo final do FD001 porque apresentou o menor MAE e o menor RMSE no benchmark oficial:

| modelo | mae | rmse | erro_medio |
| --- | --- | --- | --- |
| Extra Trees | 16.23 | 23.8 | 10.08 |
| XGBoost | 17.29 | 25.15 | 9.31 |
| HistGradientBoosting | 17.77 | 27.01 | 8.83 |

Os valores acima são o registro congelado do notebook `21_benchmark_oficial_finalistas_atributos_temporais_rul_fd001.ipynb`. A análise abaixo foi recalculada com o estudo Optuna congelado do Extra Trees (`fd001_temporal_205_extra_trees_v1`), usando os mesmos 205 atributos.

## Métricas oficiais do Extra Trees

- MAE: **16.23 ciclos**
- RMSE: **23.80 ciclos**
- Erro médio assinado: **10.08 ciclos**
- Mediana do erro absoluto por motor: **8.79 ciclos**
- Motores avaliados: **100**

O erro médio positivo indica tendência de **superestimação do RUL**: o modelo prevê, em média, mais vida útil do que a observada. Essa direção deve ser considerada na interpretação operacional.

## Erros por motor

O maior erro absoluto ocorreu no motor **1**, no ciclo observado **31**, com RUL real de **112.00**, RUL previsto de **188.51** e erro assinado de **76.51 ciclos**.

Os dez maiores erros absolutos estão no arquivo [extra-trees-erros-por-motor-fd001.csv](extra-trees-erros-por-motor-fd001.csv). A tabela abaixo mostra os casos mais críticos:

| group | cycle | y_true | y_predito | erro | erro_absoluto | faixa_rul | faixa_cycle | faixa_rul_predito |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 31 | 112.0 | 188.51 | 76.51 | 76.51 | Acima de 100 ciclos | Até 50 ciclos | Acima de 100 ciclos |
| 67 | 71 | 77.0 | 150.26 | 73.26 | 73.26 | 61 a 100 ciclos | 51 a 100 ciclos | Acima de 100 ciclos |
| 78 | 72 | 107.0 | 178.81 | 71.81 | 71.81 | Acima de 100 ciclos | 51 a 100 ciclos | Acima de 100 ciclos |
| 50 | 74 | 79.0 | 136.18 | 57.18 | 57.18 | 61 a 100 ciclos | 51 a 100 ciclos | Acima de 100 ciclos |
| 79 | 101 | 63.0 | 113.78 | 50.78 | 50.78 | 61 a 100 ciclos | 101 a 150 ciclos | Acima de 100 ciclos |
| 83 | 73 | 137.0 | 187.09 | 50.09 | 50.09 | Acima de 100 ciclos | 51 a 100 ciclos | Acima de 100 ciclos |
| 2 | 49 | 98.0 | 145.15 | 47.15 | 47.15 | 61 a 100 ciclos | Até 50 ciclos | Acima de 100 ciclos |
| 14 | 46 | 107.0 | 152.71 | 45.71 | 45.71 | Acima de 100 ciclos | Até 50 ciclos | Acima de 100 ciclos |
| 85 | 34 | 118.0 | 158.97 | 40.97 | 40.97 | Acima de 100 ciclos | Até 50 ciclos | Acima de 100 ciclos |
| 48 | 78 | 92.0 | 131.14 | 39.14 | 39.14 | 61 a 100 ciclos | 51 a 100 ciclos | Acima de 100 ciclos |

## Erros por faixa de RUL real

Esta divisão usa o RUL real somente para diagnóstico retrospectivo e medição do desempenho. Ele não foi usado como atributo de entrada.

| faixa_rul | erro_medio_motor | mae_medio_motor | rmse_medio_motor | superestimacao_percentual | subestimacao_percentual | motores |
| --- | --- | --- | --- | --- | --- | --- |
| Até 30 ciclos | 1.75 | 2.96 | 2.96 | 76.0 | 24.0 | 25 |
| 31 a 60 ciclos | 7.63 | 9.62 | 9.62 | 78.57 | 21.43 | 14 |
| 61 a 100 ciclos | 16.1 | 21.75 | 21.75 | 71.43 | 28.57 | 28 |
| Acima de 100 ciclos | 12.31 | 24.4 | 24.4 | 66.67 | 33.33 | 33 |

O padrão esperado aparece claramente: o erro tende a crescer nas faixas de RUL mais altas. Nelas, o modelo precisa estimar uma vida útil longa a partir de sinais ainda relativamente distantes da falha, o que torna a tarefa mais incerta.

## Erros por ciclo observado

No benchmark oficial há uma observação final por motor. Portanto, `cycle` representa o último ciclo disponível antes da previsão para aquele motor, e não todos os ciclos da trajetória.

| faixa_cycle | erro_medio_motor | mae_medio_motor | rmse_medio_motor | superestimacao_percentual | subestimacao_percentual | motores |
| --- | --- | --- | --- | --- | --- | --- |
| Até 50 ciclos | 37.9 | 37.9 | 37.9 | 100.0 | 0.0 | 8 |
| 51 a 100 ciclos | 24.84 | 28.09 | 28.09 | 86.36 | 13.64 | 22 |
| 101 a 150 ciclos | 8.16 | 12.7 | 12.7 | 76.47 | 23.53 | 34 |
| 151 a 200 ciclos | -1.23 | 6.6 | 6.6 | 67.86 | 32.14 | 28 |
| 201 a 250 ciclos | -11.33 | 11.33 | 11.33 | 0.0 | 100.0 | 7 |
| Acima de 250 ciclos | -5.72 | 5.72 | 5.72 | 0.0 | 100.0 | 1 |

Essa análise é útil para verificar se motores observados em diferentes durações de operação apresentam regimes de erro distintos. A concentração de superestimação nas primeiras faixas é uma associação observada no teste oficial, mas deve ser tratada como exploratória até ser comparada com o OOF. Ela não deve ser confundida com uma feature de faixa de ciclo: `faixa_cycle` foi criada apenas depois da previsão, para diagnóstico.

## Limites e próximo uso

- A avaliação oficial usa os 100 motores do teste FD001 uma única vez.
- As faixas de RUL e ciclo são estratificações diagnósticas; não são entradas do modelo.
- O benchmark não deve ser reutilizado para escolher hiperparâmetros ou criar novas features.
- Qualquer nova melhoria deve ser avaliada em uma nova divisão de desenvolvimento, preservando este benchmark como resultado final selado.

Arquivos detalhados:

- [Erros por motor](extra-trees-erros-por-motor-fd001.csv)
- [Erros por faixa de RUL](extra-trees-erros-por-rul-fd001.csv)
- [Erros por faixa de ciclo](extra-trees-erros-por-ciclo-fd001.csv)
