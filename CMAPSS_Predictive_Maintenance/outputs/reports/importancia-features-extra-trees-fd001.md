# Importância dos atributos — Extra Trees final — FD001

## Protocolo

O modelo usado é o Extra Trees final, reconstruído com o estudo Optuna congelado `fd001_temporal_205_extra_trees_v1` e treinado nos 205 atributos temporais. A seleção do Optuna manteve **93 atributos**.

Foram calculadas duas medidas:

- **Importância nativa:** redução média de impureza usada pelas árvores. É descritiva e pode distribuir importância entre atributos correlacionados.
- **Importância por permutação:** aumento do MAE quando o atributo é embaralhado. Foi calculada em cinco folds por motor, com três repetições, usando somente o desenvolvimento OOF.

O conjunto oficial de teste não foi usado para calcular a importância.

## Maiores importâncias nativas

| feature | familia | tipo | janela | selecionada_optuna | importancia_nativa |
| --- | --- | --- | --- | --- | --- |
| cycle | Base | Base | Atual | True | 0.1878 |
| s4_media_10 | Média móvel | Média | 10 | True | 0.0353 |
| s11_media_10 | Média móvel | Média | 10 | True | 0.0321 |
| s11_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.0311 |
| s11_media_5 | Média móvel | Média | 5 | True | 0.0311 |
| s15_media_10 | Média móvel | Média | 10 | True | 0.0283 |
| s21_media_10 | Média móvel | Média | 10 | True | 0.0252 |
| s17_media_10 | Média móvel | Média | 10 | True | 0.0248 |
| s4_media_5 | Média móvel | Média | 5 | True | 0.0245 |
| s3_media_20 | Média móvel | Média | 20 | True | 0.0214 |
| s9_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.0201 |
| s4_media_20 | Média móvel | Média | 20 | True | 0.0193 |
| s12_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.0182 |
| s21_media_5 | Média móvel | Média | 5 | True | 0.0174 |
| s4_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.0173 |
| s15_media_5 | Média móvel | Média | 5 | True | 0.0171 |
| s17_media_20 | Média móvel | Média | 20 | True | 0.0169 |
| s14_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.0166 |
| s3_media_10 | Média móvel | Média | 10 | True | 0.0166 |
| s21_media_20 | Média móvel | Média | 20 | True | 0.0157 |

## Maiores importâncias por permutação

Uma importância positiva indica que a permutação piorou o MAE. Valores negativos sugerem redundância, instabilidade ou que a permutação acidentalmente melhorou a previsão naquele conjunto de validação.

| feature | familia | tipo | janela | selecionada_optuna | importance_mean | importance_std_between_folds |
| --- | --- | --- | --- | --- | --- | --- |
| cycle | Base | Base | Atual | True | 7.715 | 0.7868 |
| s11_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 1.3476 | 0.1816 |
| s9_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.8754 | 0.2001 |
| s14_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.6444 | 0.0952 |
| s12_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.5718 | 0.1744 |
| s15_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.3679 | 0.1221 |
| s20_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.3013 | 0.2892 |
| s4_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.2997 | 0.0923 |
| s14_desvio_40 | Desvio-padrão móvel | Desvio-padrão | 40 | True | 0.2893 | 0.1013 |
| s8_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.2445 | 0.1627 |
| s7_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.2434 | 0.1368 |
| s7_media_80 | Média móvel | Média | 80 | True | 0.2401 | 0.35 |
| s11_media_10 | Média móvel | Média | 10 | True | 0.2357 | 0.202 |
| s4_media_10 | Média móvel | Média | 10 | True | 0.2158 | 0.198 |
| s9_desvio_40 | Desvio-padrão móvel | Desvio-padrão | 40 | False | 0.1928 | 0.1309 |
| s11_media_5 | Média móvel | Média | 5 | True | 0.1731 | 0.1843 |
| s13_desvio_80 | Desvio-padrão móvel | Desvio-padrão | 80 | True | 0.1714 | 0.1522 |
| s20_media_80 | Média móvel | Média | 80 | True | 0.1629 | 0.2436 |
| s21_media_10 | Média móvel | Média | 10 | True | 0.1564 | 0.1983 |
| s4_media_20 | Média móvel | Média | 20 | True | 0.1436 | 0.1331 |

## Importância por família

### Importância nativa

| familia | tipo | atributos | selecionados | importancia_nativa_total |
| --- | --- | --- | --- | --- |
| Média móvel | Média | 85 | 60 | 0.6075 |
| Desvio-padrão móvel | Desvio-padrão | 85 | 20 | 0.2005 |
| Base | Base | 18 | 13 | 0.1921 |
| Delta | Delta | 17 | 0 | 0.0 |

### Importância por permutação

| familia | tipo | atributos | selecionados | importancia_permutacao_media | importancia_permutacao_abs_media | atributos_importancia_positiva |
| --- | --- | --- | --- | --- | --- | --- |
| Base | Base | 18 | 13 | 0.4283 | 0.4293 | 7 |
| Desvio-padrão móvel | Desvio-padrão | 85 | 20 | 0.068 | 0.0682 | 19 |
| Média móvel | Média | 85 | 60 | 0.0138 | 0.0419 | 31 |
| Delta | Delta | 17 | 0 | 0.0 | 0.0 | 0 |

## Limitações

Importância não significa causalidade. Atributos temporais da mesma variável e de janelas próximas são correlacionados; por isso, a contribuição pode ser compartilhada entre eles. A importância por permutação deve ser interpretada junto com a estabilidade entre folds e não como uma prova de que uma variável isolada determina o RUL.

Arquivos detalhados:

- [Importância nativa](extra-trees-importancia-nativa-fd001.csv)
- [Importância por permutação](extra-trees-importancia-permutacao-fd001.csv)
- [Famílias — importância nativa](extra-trees-importancia-familias-nativa-fd001.csv)
- [Famílias — importância por permutação](extra-trees-importancia-familias-permutacao-fd001.csv)
