# Faixas empíricas de incerteza — Extra Trees final — FD001

Percentil 90 dos erros absolutos, com peso igual por motor. Metade dos motores de cada fold OOF para calibração e metade para avaliação (semente 42). Análise exploratória: folds usados na otimização; sem garantia de cobertura e sem validação de intervalo do modelo refitado.

```text
 motores_avaliacao  cobertura_media_motor  cobertura_trajetoria_inteira  largura_media_motor  rul_real_abaixo_da_faixa  rul_real_acima_da_faixa
                50               0.858677                          0.46            91.336216                  0.072899                 0.068424
```

```text
 fold  meia_largura  motores_calibracao  motores_avaliacao  referencia
    1     38.813052                  10                 10         0.9
    2     44.884634                  10                 10         0.9
    3     52.670496                  10                 10         0.9
    4     46.163536                  10                 10         0.9
    5     59.845174                  10                 10         0.9
```

```text
              faixa  cobertura  largura_media  motores
      Até 30 ciclos   1.000000      63.226703       50
     31 a 60 ciclos   1.000000      90.368356       50
    61 a 100 ciclos   0.996364      96.950757       50
Acima de 100 ciclos   0.764456      96.950757       50
```
