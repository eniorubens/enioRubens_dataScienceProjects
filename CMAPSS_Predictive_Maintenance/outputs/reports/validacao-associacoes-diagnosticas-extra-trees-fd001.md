# Validação das associações diagnósticas — Extra Trees — FD001

## Protocolo

Esta etapa não ajusta o modelo e não usa o teste oficial para seleção. O Extra Trees mantém os hiperparâmetros Optuna congelados. A comparação é entre:

- **Teste oficial:** uma previsão final por cada um dos 100 motores do teste;
- **OOF desenvolvimento:** previsões out-of-fold com cinco partições por motor no conjunto de treino.

Os intervalos de confiança foram calculados por bootstrap reamostrando motores, e não linhas, para respeitar a dependência temporal dentro de cada trajetória.

## Distribuição geral

| origem | motores_ou_observacoes | superestimados | subestimados | percentual_superestimacao | percentual_subestimacao | erro_medio | mediana_erro | magnitude_media_superestimacao | magnitude_media_subestimacao |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Teste oficial | 100 | 72 | 28 | 72.0 | 28.0 | 10.08 | 4.56 | 18.27 | 10.98 |
| OOF desenvolvimento (média por motor) | 100 | 64 | 36 | 64.0 | 36.0 | 3.59 | 6.93 | 18.29 | 22.54 |

## Associação com a faixa de RUL

As associações por RUL são diagnósticas. O RUL real não foi usado como entrada do modelo.

| origem | categoria | motores | erro_medio_medio | erro_medio_ic95_inferior | erro_medio_ic95_superior | mae_medio | mae_ic95_inferior | mae_ic95_superior | superestimacao_medio | superestimacao_ic95_inferior | superestimacao_ic95_superior | subestimacao_medio | subestimacao_ic95_inferior | subestimacao_ic95_superior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Teste oficial | 31 a 60 ciclos | 14 | 7.63 | 2.05 | 13.43 | 9.62 | 5.0 | 14.42 | 78.57 | 57.14 | 100.0 | 21.43 | 0.0 | 42.86 |
| Teste oficial | 61 a 100 ciclos | 28 | 16.1 | 7.52 | 24.96 | 21.75 | 14.98 | 29.49 | 71.43 | 53.57 | 85.71 | 28.57 | 14.29 | 46.43 |
| Teste oficial | Acima de 100 ciclos | 33 | 12.31 | 2.7 | 21.89 | 24.4 | 18.48 | 30.64 | 66.67 | 48.48 | 81.82 | 33.33 | 18.18 | 51.52 |
| Teste oficial | Até 30 ciclos | 25 | 1.75 | 0.51 | 2.93 | 2.96 | 2.21 | 3.8 | 76.0 | 60.0 | 92.0 | 24.0 | 8.0 | 40.0 |
| OOF desenvolvimento | Até 30 ciclos | 100 | 1.31 | 0.82 | 1.82 | 2.59 | 2.28 | 2.87 | 68.42 | 61.64 | 75.29 | 31.58 | 24.71 | 38.36 |
| OOF desenvolvimento | 31 a 60 ciclos | 100 | 5.52 | 3.79 | 7.49 | 8.36 | 7.07 | 9.79 | 68.63 | 60.97 | 76.83 | 31.37 | 23.17 | 39.03 |
| OOF desenvolvimento | 61 a 100 ciclos | 100 | 14.26 | 10.36 | 18.31 | 19.31 | 16.33 | 22.49 | 73.72 | 65.62 | 81.13 | 26.28 | 18.87 | 34.38 |
| OOF desenvolvimento | Acima de 100 ciclos | 100 | 4.44 | -2.67 | 11.67 | 31.21 | 27.3 | 35.24 | 58.15 | 49.73 | 66.34 | 41.85 | 33.66 | 50.27 |

## Associação com o ciclo observado

No teste oficial, o ciclo observado é o último ciclo disponível para cada motor. No OOF, há várias observações temporais por motor; por isso, a agregação e o bootstrap foram feitos primeiro no nível do motor.

| origem | categoria | motores | erro_medio_medio | erro_medio_ic95_inferior | erro_medio_ic95_superior | mae_medio | mae_ic95_inferior | mae_ic95_superior | superestimacao_medio | superestimacao_ic95_inferior | superestimacao_ic95_superior | subestimacao_medio | subestimacao_ic95_inferior | subestimacao_ic95_superior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Teste oficial | 101 a 150 ciclos | 34 | 8.16 | 3.26 | 13.57 | 12.7 | 8.9 | 16.92 | 76.47 | 61.76 | 91.18 | 23.53 | 8.82 | 38.24 |
| Teste oficial | 151 a 200 ciclos | 28 | -1.23 | -5.72 | 2.56 | 6.6 | 3.47 | 10.12 | 67.86 | 50.0 | 85.71 | 32.14 | 14.29 | 50.0 |
| Teste oficial | 201 a 250 ciclos | 7 | -11.33 | -20.94 | -2.97 | 11.33 | 2.97 | 20.94 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| Teste oficial | 51 a 100 ciclos | 22 | 24.84 | 15.43 | 35.12 | 28.09 | 20.42 | 36.76 | 86.36 | 72.73 | 100.0 | 13.64 | 0.0 | 27.27 |
| Teste oficial | Acima de 250 ciclos | 1 | -5.72 | -5.72 | -5.72 | 5.72 | 5.72 | 5.72 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| Teste oficial | Até 50 ciclos | 8 | 37.9 | 25.89 | 51.66 | 37.9 | 25.89 | 51.66 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| OOF desenvolvimento | Até 50 ciclos | 100 | -4.06 | -13.61 | 5.41 | 36.88 | 30.96 | 42.81 | 52.98 | 44.18 | 62.06 | 47.02 | 37.94 | 55.82 |
| OOF desenvolvimento | 51 a 100 ciclos | 100 | 0.42 | -7.69 | 8.45 | 32.22 | 27.5 | 37.33 | 60.52 | 51.58 | 69.38 | 39.48 | 30.62 | 48.42 |
| OOF desenvolvimento | 101 a 150 ciclos | 100 | 0.85 | -3.47 | 4.76 | 15.73 | 13.32 | 18.61 | 68.87 | 60.42 | 76.87 | 31.13 | 23.13 | 39.58 |
| OOF desenvolvimento | 151 a 200 ciclos | 93 | -0.63 | -2.42 | 1.03 | 5.42 | 4.1 | 6.95 | 65.21 | 58.05 | 72.44 | 34.79 | 27.56 | 41.95 |
| OOF desenvolvimento | 201 a 250 ciclos | 46 | -2.32 | -4.36 | -0.7 | 4.02 | 2.65 | 5.91 | 45.29 | 34.54 | 56.66 | 54.71 | 43.34 | 65.46 |
| OOF desenvolvimento | Acima de 250 ciclos | 17 | -2.47 | -4.56 | -0.8 | 3.47 | 2.05 | 5.34 | 33.64 | 18.76 | 50.07 | 66.36 | 49.93 | 81.24 |

## Interpretação

O padrão deve ser considerado robusto apenas quando a direção aparece tanto no teste oficial quanto no OOF e os intervalos de confiança não forem excessivamente largos. A distribuição geral do OOF foi resumida pela média do erro de cada motor, para ficar comparável ao teste oficial. Faixas com poucos motores devem permanecer como evidência fraca, mesmo quando a média parecer grande.

Os arquivos completos estão em:

- [Previsões OOF](extra-trees-oof-erros-fd001.csv)
- [Distribuição geral](extra-trees-validacao-distribuicao-fd001.csv)
- [Validação por RUL](extra-trees-validacao-associacao-rul-fd001.csv)
- [Validação por ciclo](extra-trees-validacao-associacao-ciclo-fd001.csv)
