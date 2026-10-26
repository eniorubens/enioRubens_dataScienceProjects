# Referência sobre threshold e escolha do limiar

## O que significa `threshold == 0.5`?

`threshold == 0.5` significa um limiar de 50%, não de 5%. Quando a probabilidade prevista pelo modelo é maior ou igual a 0,5, o registro é classificado como `>50K`.

Os 5% aparecem apenas no passo da grade de avaliação:

```text
0,10, 0,15, 0,20, ..., 0,50, ..., 0,90
```

Isso significa que os thresholds foram comparados em incrementos de cinco pontos percentuais.

## O valor 0,5 foi escolhido pela análise?

Não. O valor 0,5 foi usado como referência convencional e provisória. Ele não foi escolhido por uma regra de negócio, por uma restrição de fairness ou porque a análise tenha demonstrado que seria o melhor ponto.

Assim, a auditoria de grupos no threshold 0,5 deve ser interpretada como uma fotografia de referência. Ela permite comparar sexo, raça e interseções sob uma mesma regra de decisão, mas não representa ainda o threshold ideal.

## O que a grade de validação mostra?

Para o HistGradientBoosting, a validação atual apresentou:

| Threshold | Precisão | Recall/TPR | F1 | Taxa de seleção |
|---:|---:|---:|---:|---:|
| 0,35 | 0,670 | 0,803 | 0,731 | 0,288 |
| 0,50 | 0,785 | 0,658 | 0,715 | 0,202 |

Na partição de validação atual, 0,35 maximiza o F1 entre os pontos avaliados. Entretanto, isso não autoriza escolhê-lo como threshold final, porque a mesma partição foi usada para comparar os thresholds. Escolher e relatar o desempenho no mesmo conjunto pode produzir uma estimativa otimista.

## Como os gráficos serão usados

Os gráficos da seção de auditoria de taxas serão comparados posteriormente durante a escolha do threshold. Para cada candidato, serão observados o desempenho global, a taxa de seleção, TPR, FPR, FNR, PPV, as diferenças entre grupos e os intervalos de Wilson.

A escolha definitiva deverá seguir uma regra explícita, como recall mínimo, custo relativo de falsos positivos e falsos negativos, limite de taxa de seleção ou compromisso entre desempenho e disparidades. O threshold não deverá ser escolhido por maximização isolada de uma métrica.

Até essa etapa, a formulação correta é:

> O threshold de 0,5 é uma referência provisória para a auditoria. A escolha definitiva será feita posteriormente, comparando os gráficos, as métricas, a incerteza e os critérios de risco, sem usar o teste oficial para seleção.

## Estado do protocolo

- O teste oficial permanece reservado.
- Os thresholds avaliados ainda não foram selecionados.
- A análise de calibração atual é diagnóstica; nenhum calibrador final foi ajustado.
- A escolha futura deverá considerar também a estabilidade e as limitações dos dados históricos do UCI Adult.
