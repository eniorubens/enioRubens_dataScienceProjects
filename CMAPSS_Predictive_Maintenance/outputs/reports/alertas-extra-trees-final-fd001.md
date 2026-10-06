# Alertas do Extra Trees final — FD001

Análise exploratória sobre previsões OOF usadas no desenvolvimento. Janela útil de 5 a 30 ciclos, confirmação em 3 ciclos. Custos relativos ilustrativos; alertas tardios recebem o mesmo custo de falhas não detectadas.

```text
 limiar  Prematuro  Oportuno  Tardio  Não detectada  motores
     10          0        87      13              0      100
     20          0       100       0              0      100
     30         15        85       0              0      100
     40         73        27       0              0      100
     50         98         2       0              0      100
```

Menor custo observado em cada cenário (sem validação independente da escolha):

```text
                   cenario  limiar  custo_medio_motor
          Falha muito cara      20                1.0
             Intermediário      20                1.0
Intervenção prematura cara      20                1.0
```